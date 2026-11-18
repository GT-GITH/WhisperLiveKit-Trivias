"""End-to-end testrunner (laag 1): stuurt gebouwde scenario's door de echte server.

Doet per scenario precies wat de browser-frontend doet: per kanaal een eigen
WebSocket (/asr, gedeelde session_id, gate_framed=1), 16 kHz s16le PCM in
realtime-tempo, daarna een leeg bericht = Stop. Dan worden twee transcriptpaden
gescoord tegen de referentie:
  - incremental: het tijdens de sessie opgebouwde, opgeslagen transcript
    (GET /sessions/{id}/transcript?channel_id=...)
  - refresh: "Ververs Transcriptie" (POST /sessions/{id}/refresh_transcript),
    de volledige batch-herbouw vanaf de WAV.

Alleen stdlib + `websockets` (al een projectdependency). Draai op de pod,
naast de server:
    python tests/e2e/run_e2e.py                       # alle scenario's
    python tests/e2e/run_e2e.py nl_basis --speed 1    # één scenario
    python tests/e2e/run_e2e.py --save-baseline       # huidige uitkomst = nieuwe basislijn

Uitvoer: tests/e2e/results/<tijdstempel>/report.json + report.md (+ ruwe
server-berichten per kanaal). Exitcode 1 bij fouten of regressie t.o.v.
tests/e2e/baseline.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import LIVE_LAG_FAIL_S, LIVE_STALL_FAIL_S, score_batch_latency, score_channel, score_live  # noqa: E402

# Serverlog-patronen die tijdens een run als FOUT tellen. Les van 2026-11-15: de
# live-decoder faalde met exceptions/time-outs terwijl het rapport "Fouten: geen" zei,
# omdat alleen de WebSocket werd bekeken.
SERVER_LOG_ERROR_PATTERNS = {
    "Traceback": "Traceback",
    "live-time-out": "[LIVE][TIMEOUT]",
    "live-abort": "[LIVE][ABORT]",
    "lusbeveiliging": "[Loop Detection]",
    "transcriptie-exception": "Exception in transcription_processor",
    # batch-venster mislukt (bv. CUDA OOM) -- gemist op 2026-11-18 omdat dit niet telde
    "batch-fout": "[BATCH][WORKER] error",
    "batch-venster-definitief-mislukt": "[BATCH][FAILED]",
}
# Geen fouten, wel meetwaarden voor het live-buffer-onderzoek (2026-11-17).
SERVER_LOG_INFO_PATTERNS = {
    "live_rewinds": "[rewind detected]",
    "live_rewind_hard_resets": "HARD RESET LIVE SEGMENT",
    "live_nospeech_stops": "no speech, stop",
    "live_reanchors": "[LIVE][ANCHOR]",
}
LIVE_GAP_REGRESSION_S = 3.0


def server_log_offset(path: str | None) -> int:
    try:
        return Path(path).stat().st_size if path else -1
    except OSError:
        return -1


def server_log_errors(path: str | None, offset: int, patterns: dict = SERVER_LOG_ERROR_PATTERNS) -> dict:
    """Tel foutpatronen in het deel van de serverlog dat tijdens deze run is bijgeschreven."""
    if not path or offset < 0:
        return {}
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return {}
    counts = {name: text.count(pat) for name, pat in patterns.items()}
    return {k: v for k, v in counts.items() if v}

HERE = Path(__file__).resolve().parent
SCENARIOS = HERE / "scenarios"
RESULTS = HERE / "results"
BASELINE = HERE / "baseline.json"
SR = 16000

# Regressiedrempels t.o.v. de basislijn (absolute toename).
WER_REGRESSION = 0.03
COUNT_REGRESSION = 0  # missed/spurious/live_fallback: elke toename telt
TS_REGRESSION_MS = 300


def http_json(method: str, url: str, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# Nabootsing van de client-gate in web_trivias/pcm_worklet.js: per AudioWorklet-frame
# (128 samples @ 48 kHz = 2.67 ms ~ 43 samples @ 16 kHz) RMS >= drempel -> open,
# daarna ~300 ms nahouden (113 frames). Elk frame is in de browser een eigen
# WS-bericht met eigen vlag (recorder_worker.js), dus hier ook.
GATE_FRAME_SAMPLES = 43
GATE_HOLD_FRAMES = 113


def client_gate_frames(pcm: bytes, threshold: float) -> list:
    """[(vlag, pcm-bytes)] per worklet-frame, zoals de browser ze verstuurt."""
    import array
    samples = array.array("h")
    samples.frombytes(pcm)
    out, hold, n = [], 0, GATE_FRAME_SAMPLES
    for i in range(0, len(samples), n):
        frame = samples[i:i + n]
        rms = (sum((x / 32768.0) ** 2 for x in frame) / max(1, len(frame))) ** 0.5
        if rms >= threshold:
            hold = GATE_HOLD_FRAMES
            gate = True
        elif hold > 0:
            hold -= 1
            gate = True
        else:
            gate = False
        out.append((1 if gate else 0, frame.tobytes()))
    return out


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return w.readframes(w.getnframes())


async def stream_channel(ws_base: str, session_id: str, channel_id: str, cfg: dict, pcm: bytes,
                         start_evt: asyncio.Event, t0_holder: dict, speed: float, chunk_ms: int,
                         stop_timeout: float, client_gate: float | None = None) -> dict:
    params = {"session_id": session_id, "channel_id": channel_id, "lang": cfg["lang"],
              "gate_framed": "1", "gate_threshold": str(client_gate or 0)}
    if cfg.get("lang2"):
        params["lang2"] = cfg["lang2"]
    url = f"{ws_base}/asr?{urlencode(params)}"
    messages, errors = [], []
    got_ready = asyncio.Event()

    async with websockets.connect(url, max_size=None, open_timeout=30) as ws:
        async def receiver():
            try:
                async for raw in ws:
                    msg = json.loads(raw) if isinstance(raw, str) else {"_binary": len(raw)}
                    messages.append({"t": round(time.monotonic() - t0_holder.get("t0", time.monotonic()), 3),
                                     "msg": msg})
                    if msg.get("type") == "ready_to_stop":
                        got_ready.set()
                    if msg.get("error"):
                        errors.append(str(msg.get("error")))
            except websockets.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(receiver())
        await start_evt.wait()
        t0 = t0_holder["t0"]
        if client_gate:
            frames = client_gate_frames(pcm, client_gate)
            per_batch = max(1, int(SR * chunk_ms / 1000 / GATE_FRAME_SAMPLES))
            for b, i in enumerate(range(0, len(frames), per_batch)):
                target = t0 + (b * per_batch * GATE_FRAME_SAMPLES / SR) / speed
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                for flag, data in frames[i:i + per_batch]:
                    await ws.send(bytes([flag]) + data)
        else:
            step = int(SR * 2 * chunk_ms / 1000)
            for i, off in enumerate(range(0, len(pcm), step)):
                target = t0 + (i * chunk_ms / 1000) / speed
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                await ws.send(b"\x01" + pcm[off:off + step])  # vlag 1 = naar ASR (gate open)
        await ws.send(b"")  # leeg bericht = Stop (zie audio_processor.process_audio)
        try:
            await asyncio.wait_for(got_ready.wait(), timeout=stop_timeout)
        except asyncio.TimeoutError:
            errors.append(f"geen ready_to_stop binnen {stop_timeout}s na Stop")
        await ws.close()
        await asyncio.wait_for(recv_task, timeout=10)
    return {"messages": messages, "errors": errors}


async def wait_auto_refresh(server: str, session_id: str, start_timeout_s: float = 20.0,
                            done_timeout_s: float = 900.0):
    """Wacht op het automatische Ververs na Stop. None = niet gestart binnen
    start_timeout_s (bv. TRIVIAS_AUTO_REFRESH=0); anders de eindstatus."""
    t0 = time.monotonic()
    started = False
    while time.monotonic() - t0 < done_timeout_s:
        try:
            st = http_json("GET", f"{server}/sessions/{session_id}/refresh_status")
        except Exception:
            st = {}
        if st.get("trigger") == "auto_after_stop" or st.get("status") == "running":
            started = True
            if st.get("status") in ("done", "failed"):
                return st
        elif not started and time.monotonic() - t0 > start_timeout_s:
            return None
        await asyncio.sleep(1.0)
    return {"status": "timeout"}


async def run_scenario(name: str, server: str, speed: float, chunk_ms: int, do_refresh: bool,
                       out_dir: Path, client_gate: float | None = None,
                       server_log: str | None = None) -> dict:
    sdir = SCENARIOS / name
    ref = json.loads((sdir / "reference.json").read_text(encoding="utf-8"))
    session_id = str(uuid.uuid4())
    ws_base = server.replace("http://", "ws://").replace("https://", "wss://")
    print(f"[{name}] session={session_id} duur={ref['duration_ms'] / 1000:.0f}s speed={speed}")

    log_offset = server_log_offset(server_log)
    start_evt, t0_holder = asyncio.Event(), {}
    tasks = {
        ch: asyncio.create_task(stream_channel(
            ws_base, session_id, ch, cfg, read_pcm(sdir / f"{ch}.wav"), start_evt, t0_holder,
            speed, chunk_ms, stop_timeout=180 + ref["duration_ms"] / 1000, client_gate=client_gate))
        for ch, cfg in ref["channels"].items()
    }
    await asyncio.sleep(2.0)  # alle kanalen verbonden voor de gezamenlijke start
    t0_holder["t0"] = time.monotonic()
    start_evt.set()
    streamed = {}
    for ch, task in tasks.items():
        try:
            streamed[ch] = await task
        except Exception as e:
            streamed[ch] = {"messages": [], "errors": [f"{type(e).__name__}: {e}"]}
    wall = time.monotonic() - t0_holder["t0"]

    (out_dir / f"{name}.messages.json").write_text(
        json.dumps({ch: s["messages"] for ch, s in streamed.items()}, ensure_ascii=False), encoding="utf-8")

    result = {"scenario": name, "session_id": session_id, "wall_s": round(wall, 1),
              "errors": {ch: s["errors"] for ch, s in streamed.items() if s["errors"]}, "paths": {}}

    # Sinds 2026-11-18 draait Ververs automatisch na Stop (server). Wacht daarop, dan
    # het incrementele transcript (bewaard vóór de refresh) en het definitieve lezen.
    auto = await wait_auto_refresh(server, session_id)
    if auto is None and do_refresh:
        result["errors"].setdefault("_refresh", []).append(
            "geen automatisch Ververs na Stop gestart (binnen 20 s) -- handmatig uitgevoerd")
    elif auto is not None:
        result["refresh_auto"] = auto
        if auto.get("status") != "done":
            result["errors"].setdefault("_refresh", []).append(f"automatisch Ververs: {auto}")

    incremental = {}
    for ch in ref["channels"]:
        try:
            q = urlencode({'channel_id': ch, 'version': 'incremental'})
            data = http_json("GET", f"{server}/sessions/{session_id}/transcript?{q}")
            incremental[ch] = data.get("segments", [])
        except Exception as e:
            result["errors"].setdefault(ch, []).append(f"transcript ophalen: {e}")
            incremental[ch] = []
    result["paths"]["incremental"] = {
        ch: score_channel(ref["channels"][ch]["utterances"], incremental[ch]) for ch in ref["channels"]}

    if do_refresh:
        try:
            if auto is not None and auto.get("status") == "done":
                result["refresh_s"] = auto.get("duration_s")
                data = http_json("GET", f"{server}/sessions/{session_id}/transcript?channel_id=all")
            else:
                t = time.monotonic()
                data = http_json("POST", f"{server}/sessions/{session_id}/refresh_transcript")
                result["refresh_s"] = round(time.monotonic() - t, 1)
            by_ch = {ch: [s for s in data.get("segments", []) if s.get("channel_id") == ch] for ch in ref["channels"]}
            result["paths"]["refresh"] = {
                ch: score_channel(ref["channels"][ch]["utterances"], by_ch[ch]) for ch in ref["channels"]}
        except Exception as e:
            result["errors"].setdefault("_refresh", []).append(str(e))

    # Live: beweegt live mee tijdens echte spraak? Een live-stilte langer dan
    # LIVE_STALL_FAIL_S binnen spraak is een fout (live-stilstand, 2026-11-15).
    result["live"] = {ch: score_live(ref["channels"][ch]["utterances"], streamed[ch]["messages"])
                      for ch in ref["channels"]}
    # Wachttijd tot definitief: hoe snel verschijnt de batch-tekst na het einde van een uiting.
    for ch in ref["channels"]:
        result["live"][ch].update(score_batch_latency(ref["channels"][ch]["utterances"], streamed[ch]["messages"]))
    if speed != 1.0:
        result["live_note"] = "speed != 1: live-timing niet vergelijkbaar met audiotijd"
    else:
        for ch, lm in result["live"].items():
            if lm["live_max_gap_s"] > LIVE_STALL_FAIL_S:
                result["errors"].setdefault(f"{ch}/live", []).append(
                    f"live stil tijdens spraak: {lm['live_max_gap_s']}s (bij {lm['live_max_gap_at_s']})")
            if lm["live_lag_p90_s"] is not None and lm["live_lag_p90_s"] > LIVE_LAG_FAIL_S:
                result["errors"].setdefault(f"{ch}/live", []).append(
                    f"live-tijdstempels lopen achter: p90 {lm['live_lag_p90_s']}s (mediaan {lm['live_lag_median_s']}s)")
    # Definitieve tekst: ELKE gemiste uiting is een fout, ongeacht de basislijn --
    # voor de doelgroep (gehoren/zittingen) is een ontbrekende zin onacceptabel (2026-11-15).
    for pad, per_ch in result["paths"].items():
        for ch, m in per_ch.items():
            if m["missed"] > 0:
                result["errors"].setdefault(f"{ch}/{pad}", []).append(
                    f"{m['missed']}/{m['n_ref_utterances']} uiting(en) ontbreken in de definitieve tekst "
                    f"(waarvan {m.get('missed_marked', 0)} zichtbaar als '[geen betrouwbare tekst]', "
                    f"{m['missed'] - m.get('missed_marked', 0)} onzichtbaar)")
    # Serverlog: exceptions/time-outs tijdens deze run zijn fouten.
    result["server_log"] = server_log_errors(server_log, log_offset)
    result["server_log_info"] = server_log_errors(server_log, log_offset, SERVER_LOG_INFO_PATTERNS)
    if result["server_log"]:
        result["errors"].setdefault("_serverlog", []).extend(f"{k}: {v}x" for k, v in result["server_log"].items())
    return result


def compare(results: list, baseline: dict) -> list:
    """Regressies t.o.v. de basislijn, als leesbare regels."""
    regressions = []
    base = {r["scenario"]: r for r in baseline.get("results", [])}
    for r in results:
        b = base.get(r["scenario"])
        if not b:
            continue
        for path, chans in r["paths"].items():
            for ch, m in chans.items():
                bm = b.get("paths", {}).get(path, {}).get(ch)
                if not bm:
                    continue
                tag = f"{r['scenario']}/{path}/{ch}"
                if m["wer"] is not None and bm["wer"] is not None and m["wer"] - bm["wer"] > WER_REGRESSION:
                    regressions.append(f"{tag}: WER {bm['wer']:.3f} -> {m['wer']:.3f}")
                for k in ("missed", "spurious", "live_fallback"):
                    if m[k] - bm[k] > COUNT_REGRESSION:
                        regressions.append(f"{tag}: {k} {bm[k]} -> {m[k]}")
                if m["ts_max_ms"] is not None and bm["ts_max_ms"] is not None \
                        and m["ts_max_ms"] - bm["ts_max_ms"] > TS_REGRESSION_MS:
                    regressions.append(f"{tag}: ts_max {bm['ts_max_ms']} -> {m['ts_max_ms']} ms")
        for ch, lm in r.get("live", {}).items():
            blm = b.get("live", {}).get(ch)
            if not blm:
                continue
            tag = f"{r['scenario']}/live/{ch}"
            if lm["live_missed"] > blm["live_missed"]:
                regressions.append(f"{tag}: live_missed {blm['live_missed']} -> {lm['live_missed']}")
            if lm["live_max_gap_s"] - blm["live_max_gap_s"] > LIVE_GAP_REGRESSION_S:
                regressions.append(f"{tag}: live_max_gap {blm['live_max_gap_s']} -> {lm['live_max_gap_s']} s")
    return regressions


def fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v * 100:.1f}%" if pct else str(v)


def write_markdown(md_path: Path, results: list, regressions: list, health: dict) -> None:
    lines = [f"# E2E-rapport {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "",
             f"Server: model={health.get('model', '?')} ", ""]
    lines += ["| scenario | pad | kanaal | WER | CER | gemist | spurious | live-fallback | ts med/max ms |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for path, chans in r["paths"].items():
            for ch, m in chans.items():
                lines.append(f"| {r['scenario']} | {path} | {ch} | {fmt(m['wer'], True)} | {fmt(m['cer'], True)} | "
                             f"{m['missed']}/{m['n_ref_utterances']} | {m['spurious']} | {m['live_fallback']} | "
                             f"{fmt(m['ts_median_ms'])}/{fmt(m['ts_max_ms'])} |")
    lines += ["", "## Live en wachttijd tot definitief", "",
              "| scenario | kanaal | live gemist | langste live-stilte | live-achterstand med/p90 | live-tekst WER | definitief na (med/p90/max) | niet binnen sessie |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        for ch, lm in r.get("live", {}).items():
            lines.append(f"| {r['scenario']} | {ch} | {lm['live_missed']} | {lm['live_max_gap_s']} s "
                         f"(bij {lm['live_max_gap_at_s']}) | {lm['live_lag_median_s']}/{lm['live_lag_p90_s']} s | "
                         f"{lm.get('live_text_wer')} | "
                         f"{lm.get('batch_latency_median_s')}/{lm.get('batch_latency_p90_s')}/{lm.get('batch_latency_max_s')} s | "
                         f"{lm.get('batch_not_in_session')} |")
    lines += ["", "Live-decoder (serverlog, per scenario): " + "; ".join(
        f"{r['scenario']}: {r.get('server_log_info') or {}}" for r in results)]
    errs = [(r["scenario"], ch, e) for r in results for ch, es in r["errors"].items() for e in es]
    lines += ["", "## Fouten", ""] + ([f"- {s}/{c}: {e}" for s, c, e in errs] or ["Geen."])
    lines += ["", "## Regressies t.o.v. basislijn", ""] + ([f"- {x}" for x in regressions] or ["Geen."])
    for r in results:
        for path, chans in r["paths"].items():
            for ch, m in chans.items():
                if m["spurious_segments"] or m["missed_utterances"]:
                    lines += ["", f"### {r['scenario']} / {path} / {ch}"]
                    lines += [f"- spurious @{s['start_ms']}ms: {s['text']}" for s in m["spurious_segments"]]
                    lines += [f"- gemist: {t}" for t in m["missed_utterances"]]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", help="namen; leeg = alle gebouwde scenario's")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = realtime (zoals een echte microfoon)")
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--server-log", default="/workspace/trivias.log",
                    help="serverlog om foutpatronen in te tellen (leeg = overslaan)")
    ap.add_argument("--client-gate", type=float, default=None,
                    help="boots de browser-gate na met deze RMS-drempel (bv. 0.015); standaard altijd open")
    args = ap.parse_args()

    try:
        health = http_json("GET", f"{args.server}/health", timeout=10)
    except Exception as e:
        print(f"Server niet bereikbaar op {args.server}: {e}")
        return 2

    names = args.scenarios or sorted(p.name for p in SCENARIOS.iterdir() if (p / "reference.json").exists())
    if not names:
        print("Geen scenario's gevonden -- eerst: python tests/e2e/build_scenarios.py")
        return 2

    out_dir = RESULTS / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in names:
        try:
            results.append(asyncio.run(run_scenario(name, args.server, args.speed, args.chunk_ms,
                                                    not args.no_refresh, out_dir, args.client_gate,
                                                    args.server_log or None)))
        except Exception as e:
            results.append({"scenario": name, "errors": {"_run": [f"{type(e).__name__}: {e}"]}, "paths": {}})

    baseline = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    regressions = compare(results, baseline)
    report = {"health": health, "args": vars(args), "results": results, "regressions": regressions}
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    write_markdown(out_dir / "report.md", results, regressions, health)
    print((out_dir / "report.md").read_text(encoding="utf-8"))
    if args.save_baseline:
        BASELINE.write_text(json.dumps({"saved": out_dir.name, "results": results}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"Basislijn opgeslagen: {BASELINE}")

    has_errors = any(r["errors"] for r in results)
    return 1 if (has_errors or regressions) else 0


if __name__ == "__main__":
    sys.exit(main())
