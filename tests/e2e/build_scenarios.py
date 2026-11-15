"""Bouwt testscenario's (per-kanaal-WAV's + referentie) uit recepten.

Draait op de pod (internet nodig voor de datasets), in een EIGEN venv -- niet
in de server-.venv: `datasets` trekt huggingface-hub>=1.0 en numpy>=2 binnen
en breekt daarmee transformers/tokenizers van de server:
    python3 -m venv /workspace/e2e-venv
    /workspace/e2e-venv/bin/pip install datasets soundfile librosa faster-whisper

Gebruik:
    /workspace/e2e-venv/bin/python tests/e2e/build_scenarios.py            # alle recepten
    /workspace/e2e-venv/bin/python tests/e2e/build_scenarios.py nl_basis   # één recept

Recept (tests/e2e/recipes/<naam>.json):
{
  "name": "...", "description": "...",
  "channels": {"<channel_id>": {"lang": "nl", "lang2": null}},
  "timeline": [
    {"channel": "<channel_id>", "source": {"dataset": "fleurs", "config": "nl_nl", "index": 0},
     "gap_before_ms": 1500},          # optioneel, default 1500; negatief = overlap
    {"silence_ms": 45000},            # pauze waarin niemand spreekt
    {"mark": "naam"}                  # benoemd tijdstip (voor achtergrond)
  ],
  "tail_ms": 4000,                    # optioneel, stilte/achtergrond na de laatste uiting
  "background": [                     # optioneel: instrumentale muziek (FMA, CC-BY)
    {"channel": "<channel_id>", "source": {"dataset": "fma_instrumental", "index": 0},
     "from": "begin", "to": "naam", "rms_dbfs": -32}   # marks: begin/end/eigen namen
  ]
}

De tijdlijn is één beurtenreeks over alle kanalen (zoals een gehoor): elk
item start na het vorige + gap. Uitvoer in tests/e2e/scenarios/<naam>/:
<channel_id>.wav (16 kHz mono s16le, alle kanalen even lang) en reference.json
met per uiting de exacte start_ms/end_ms in de stream.

Afgeleide audio wordt NIET in git gezet (gitignored): Casablanca is
cc-by-nc-nd, en zo blijft de repo klein. Recepten zijn deterministisch
(vaste dataset-indexen), dus elke pod bouwt exact dezelfde scenario's.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # zie project-memory modelroutering-poc

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

SR = 16000
HERE = Path(__file__).resolve().parent
RECIPES = HERE / "recipes"
OUT = HERE / "scenarios"
LEAD_IN_MS = 1500
TAIL_MS = 4000
DEFAULT_GAP_MS = 1500
TARGET_RMS_DBFS = -22.0

DATASETS = {
    # naam -> (HF-repo, tekstveld, default split)
    "fleurs": ("google/fleurs", "raw_transcription", "test"),
    "casablanca": ("UBC-NLP/Casablanca", "transcription", "test"),
}

_cache: dict = {}

# Achtergrondmuziek: Free Music Archive, alleen tracks met een licentie die ook
# commercieel gebruik toestaat (CC-BY e.d.), al in 16 kHz. Instrumentaal is niet
# getagd, dus de builder kiest zelf: een track telt als instrumentaal als een frisse
# VAD (per 10 s-stuk) er < MUSIC_MAX_SPEECH_RATIO spraak in vindt. Muziek met zang
# zou Whisper als spraak transcriberen en dan meten we iets anders. Volgorde van de
# dataset is vast, dus de selectie is reproduceerbaar. Titel/artiest/licentie gaan
# mee in reference.json (naamsvermelding, CC-BY).
FMA_REPO = "benjamin-paine/free-music-archive-commercial-16khz-full"
MUSIC_MAX_SPEECH_RATIO = 0.03
MUSIC_MIN_DURATION_S = 20.0
_music: list = []  # [(audio, attributie-dict)]
_music_iter = None

try:  # zelfde Silero-VAD als faster-whisper (pip install faster-whisper in de e2e-venv)
    from faster_whisper.vad import VadOptions, get_speech_timestamps
except Exception:  # pragma: no cover
    get_speech_timestamps = None
VAD_MARGIN_S = 0.1


def trim_to_speech(audio: np.ndarray) -> np.ndarray:
    """Knip niet-spraak aan begin en eind weg, zodat start_ms/end_ms in de
    referentie echt het begin/einde van de spraak zijn.

    Met VAD, niet met een energiedrempel: FLEURS-fragmenten hebben soms seconden
    luide ruis vóór de spraak. Eerste e2e-runs (2026-11-15) gaven daardoor een
    referentiestart 5 s te vroeg (librosa top_db=35 liet -25 dB ruis staan),
    wat als tijdstempelfout van de server werd gemeld terwijl de server gelijk had."""
    if get_speech_timestamps is not None:
        ts = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300))
        if ts:
            m = int(VAD_MARGIN_S * SR)
            return audio[max(0, ts[0]["start"] - m): min(len(audio), ts[-1]["end"] + m)]
    print("  WAARSCHUWING: geen VAD beschikbaar, val terug op energie-trim (minder nauwkeurig)")
    trimmed, _ = librosa.effects.trim(audio, top_db=35)
    return trimmed


def _speech_ratio(audio: np.ndarray, chunk_s: float = 10.0) -> float:
    """Fractie spraak volgens een FRISSE VAD per stuk (geen opgebouwde toestand)."""
    if get_speech_timestamps is None:
        raise RuntimeError("faster-whisper (Silero-VAD) nodig om instrumentale muziek te kiezen")
    n = int(chunk_s * SR)
    speech = 0
    for off in range(0, len(audio), n):
        for t in get_speech_timestamps(audio[off:off + n], VadOptions(min_silence_duration_ms=300)):
            speech += t["end"] - t["start"]
    return speech / max(1, len(audio))


def load_music(index: int) -> tuple[np.ndarray, dict]:
    """De index-de instrumentale track uit FMA (zie FMA_REPO)."""
    global _music_iter
    import tempfile
    if _music_iter is None:
        print(f"  laad {FMA_REPO} (streaming)...")
        ds = load_dataset(FMA_REPO, split="train", streaming=True)
        lic = ds.features["license"]
        _music_iter = ((x, lic) for x in ds.cast_column("audio", Audio(decode=False)))
    while len(_music) <= index:
        x, lic = next(_music_iter)
        with tempfile.NamedTemporaryFile(suffix=".audio") as tmp:
            tmp.write(x["audio"]["bytes"])
            tmp.flush()
            audio, _ = librosa.load(tmp.name, sr=SR, mono=True)
        if len(audio) < MUSIC_MIN_DURATION_S * SR:
            continue
        ratio = _speech_ratio(audio)
        attr = {"title": x.get("title"), "artist": x.get("artist"), "license": lic.int2str(x["license"]),
                "url": x.get("url"), "speech_ratio": round(ratio, 3)}
        if ratio < MUSIC_MAX_SPEECH_RATIO:
            print(f"  muziek #{len(_music)}: {attr['artist']} - {attr['title']} ({attr['license']}, spraak {ratio:.1%})")
            _music.append((audio.astype(np.float32), attr))
    return _music[index]


def mix_background(buf: np.ndarray, music: np.ndarray, start_ms: int, end_ms: int, rms_dbfs: float) -> None:
    """Muziek (herhaald indien nodig) op buf mixen tussen start_ms en end_ms, met fades."""
    s, e = int(start_ms * SR / 1000), min(len(buf), int(end_ms * SR / 1000))
    if e <= s:
        return
    reps = int(np.ceil((e - s) / len(music)))
    bed = np.tile(music, reps)[: e - s].copy()
    rms = float(np.sqrt(np.mean(bed**2))) or 1e-9
    bed *= 10 ** (rms_dbfs / 20) / rms
    fade = min(int(0.2 * SR), len(bed) // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        bed[:fade] *= ramp
        bed[-fade:] *= ramp[::-1]
    buf[s:e] += bed


def load_clip(source: dict) -> tuple[np.ndarray, str]:
    repo, text_field, default_split = DATASETS[source["dataset"]]
    split = source.get("split", default_split)
    key = (repo, source["config"], split)
    if key not in _cache:
        print(f"  laad {repo} {source['config']} {split}...")
        ds = load_dataset(repo, source["config"], split=split)
        _cache[key] = ds.cast_column("audio", Audio(decode=False))
    sample = _cache[key][int(source["index"])]
    audio, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    audio = trim_to_speech(audio)
    rms = float(np.sqrt(np.mean(audio**2))) or 1e-9
    audio = audio * (10 ** (TARGET_RMS_DBFS / 20) / rms)
    return np.clip(audio, -1.0, 1.0), sample[text_field]


def build(recipe_path: Path) -> None:
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    name = recipe["name"]
    print(f"[{name}] bouwen...")
    channels = recipe["channels"]
    placed = {ch: [] for ch in channels}  # ch -> [(start_sample, audio, text)]
    cursor = LEAD_IN_MS
    marks = {"begin": 0}  # benoemde tijdstippen voor achtergrondmuziek
    for item in recipe["timeline"]:
        if "mark" in item:
            marks[item["mark"]] = cursor
            continue
        if "silence_ms" in item:
            cursor += int(item["silence_ms"])
            continue
        ch = item["channel"]
        if ch not in channels:
            raise ValueError(f"{name}: kanaal {ch} staat niet in 'channels'")
        audio, text = load_clip(item["source"])
        start = max(0, cursor + int(item.get("gap_before_ms", DEFAULT_GAP_MS)))
        placed[ch].append((start, audio, text))
        cursor = start + int(len(audio) * 1000 / SR)

    total_ms = cursor + int(recipe.get("tail_ms", TAIL_MS))
    total = int(total_ms * SR / 1000)
    marks["end"] = total_ms
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = {
        "name": name,
        "description": recipe.get("description", ""),
        "sample_rate": SR,
        "duration_ms": total_ms,
        "expect": recipe.get("expect", {}),
        "channels": {},
        "background": [],
    }
    for ch, cfg in channels.items():
        buf = np.zeros(total, dtype=np.float32)
        utts = []
        for start_ms, audio, text in placed[ch]:
            s = int(start_ms * SR / 1000)
            buf[s:s + len(audio)] += audio[: total - s]
            utts.append({"start_ms": start_ms, "end_ms": start_ms + int(len(audio) * 1000 / SR), "text": text})
        for bg in recipe.get("background", []):
            if bg["channel"] != ch:
                continue
            music, attr = load_music(int(bg["source"]["index"]))
            b_start, b_end = marks[bg.get("from", "begin")], marks[bg.get("to", "end")]
            mix_background(buf, music, b_start, b_end, float(bg.get("rms_dbfs", -32.0)))
            ref["background"].append({"channel": ch, "start_ms": b_start, "end_ms": b_end,
                                      "rms_dbfs": bg.get("rms_dbfs", -32.0), "attribution": attr})
        sf.write(out_dir / f"{ch}.wav", np.clip(buf, -1, 1), SR, subtype="PCM_16")
        ref["channels"][ch] = {"lang": cfg["lang"], "lang2": cfg.get("lang2"), "utterances": utts}
    (out_dir / "reference.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{name}] klaar: {total_ms / 1000:.1f}s, kanalen={list(channels)} -> {out_dir}")


def main() -> None:
    wanted = set(sys.argv[1:])
    recipes = sorted(RECIPES.glob("*.json"))
    for path in recipes:
        if wanted and path.stem not in wanted:
            continue
        try:
            build(path)
        except Exception as e:  # één kapot recept mag de rest niet blokkeren
            print(f"[{path.stem}] FOUT: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    # Streaming-datasets laten soms threads achter die bij het afsluiten van de
    # interpreter een core dump geven (gezien op de pod); hard afsluiten na flush.
    os._exit(0)
