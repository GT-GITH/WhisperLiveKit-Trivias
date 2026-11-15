"""Bouwt testscenario's (per-kanaal-WAV's + referentie) uit recepten.

Draait op de pod (internet nodig voor de datasets). Eenmalige extra deps,
niet in pyproject.toml:
    pip install datasets soundfile librosa

Gebruik:
    python tests/e2e/build_scenarios.py              # alle recepten
    python tests/e2e/build_scenarios.py nl_basis     # één recept

Recept (tests/e2e/recipes/<naam>.json):
{
  "name": "...", "description": "...",
  "channels": {"<channel_id>": {"lang": "nl", "lang2": null}},
  "timeline": [
    {"channel": "<channel_id>", "source": {"dataset": "fleurs", "config": "nl_nl", "index": 0},
     "gap_before_ms": 1500},          # optioneel, default 1500; negatief = overlap
    {"silence_ms": 45000}             # pauze waarin niemand spreekt
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
    # Stilte aan begin/eind eraf, zodat start_ms/end_ms in de referentie echt
    # het begin/einde van de spraak zijn (anders meten we datasetpadding).
    audio, _ = librosa.effects.trim(audio, top_db=35)
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
    for item in recipe["timeline"]:
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

    total_ms = cursor + TAIL_MS
    total = int(total_ms * SR / 1000)
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = {
        "name": name,
        "description": recipe.get("description", ""),
        "sample_rate": SR,
        "duration_ms": total_ms,
        "expect": recipe.get("expect", {}),
        "channels": {},
    }
    for ch, cfg in channels.items():
        buf = np.zeros(total, dtype=np.float32)
        utts = []
        for start_ms, audio, text in placed[ch]:
            s = int(start_ms * SR / 1000)
            buf[s:s + len(audio)] += audio[: total - s]
            utts.append({"start_ms": start_ms, "end_ms": start_ms + int(len(audio) * 1000 / SR), "text": text})
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
