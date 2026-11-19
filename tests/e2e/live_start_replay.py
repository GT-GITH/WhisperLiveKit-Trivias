"""Deterministische herhaling van het begin van een live-sessie (live-decoder alleen).

Waarom: het live-startprobleem (eerste uiting vervangen door een verzonnen
YouTube-afsluitzin, live-test 2026-11-19 sessie ebaacadc) hangt af van het exacte
moment van de eerste decodes (0.92 s buffer i.p.v. 1.12 s). De e2e-runner haalt die
timing niet; dit script voert de live-decoder (SimulStreamingOnlineProcessor, zelfde
model en config als de server) de audio in met een opgegeven reeks bufferlengtes,
uit het serverlog ('segments_len=' van [DIAG][LIVE_PERF]).

Audio voor de ASR = WAV met de client-gate nagebootst (dichte frames -> nullen),
zoals de server ze krijgt. Laadt alleen het live-model (batch wordt niet geladen).

  python tests/e2e/live_start_replay.py WAV --lang tr --gate 0.015 \
      --schedule 0.92,1.43,1.63,1.94,2.45,2.66,3.37,3.68,4.08,4.70,5.00,5.41,5.92

Draai met dezelfde env-variabelen als de server om een fix te vergelijken.
"""
import argparse
import logging
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 16000


def gated_audio(path: str, gate: float) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = w.readframes(w.getnframes())
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if not gate:
        return x
    from run_e2e import client_gate_frames  # zelfde nabootsing als de e2e-suite
    out = []
    for flag, data in client_gate_frames(pcm, gate):
        f = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        out.append(f if flag else np.zeros_like(f))
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--lang", default="tr")
    ap.add_argument("--gate", type=float, default=0.015)
    ap.add_argument("--schedule", required=True, help="bufferlengtes (s), komma-gescheiden")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--start", type=float, default=0.0, help="begin in de WAV (s)")
    ap.add_argument("--zero-until", type=float, default=0.0,
                    help="audio tot hier (s) als stilte, zoals de server die als 'Silence of' invoegt")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
    for name in ("whisperlivekit.simul_whisper.simul_whisper",):
        logging.getLogger(name).setLevel(logging.INFO)

    import whisperlivekit.core as core
    core.BatchFasterWhisperASR = lambda *a, **k: None  # alleen live nodig
    from whisperlivekit.simul_whisper.backend import SimulStreamingOnlineProcessor

    engine = core.TranscriptionEngine(model_size=args.model, lan="nl", pcm_input=True)
    proc = SimulStreamingOnlineProcessor(engine.asr, language=args.lang)

    audio = gated_audio(args.wav, args.gate)[int(args.start * SR):].copy()
    audio[:int(args.zero_until * SR)] = 0.0
    fed = 0
    words = []
    for t in [float(s) for s in args.schedule.split(",") if s.strip()]:
        n = int(round(t * SR))
        if n > fed:
            proc.insert_audio_chunk(audio[fed:n], audio_stream_end_time=args.start + n / SR)
            fed = n
        toks, _ = proc.process_iter(is_last=False)
        txt = "".join(getattr(tk, "text", "") for tk in toks)
        words.append(txt)
        print(f"[REPLAY] buffer={t:5.2f}s -> {txt!r}", flush=True)
    print("[REPLAY] LIVE: " + "".join(words).strip(), flush=True)


if __name__ == "__main__":
    main()
