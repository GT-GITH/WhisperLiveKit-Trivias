"""Eenmalig evaluatiescript, GEEN onderdeel van de reguliere pijplijn.

Doel: objectieve, native-speaker-onafhankelijke ASR-kwaliteitsmeting voor
Somalisch. Gebruikt google/fleurs (Google's meertalige ASR-benchmark --
102 talen, elk fragment met een geverifieerd correct referentietranscript)
en berekent Word Error Rate (WER) / Character Error Rate (CER) voor zowel
het huidige PoC-kandidaatmodel (Paza) als het server-brede standaardmodel
(large-v3), op DEZELFDE fragmenten, met dezelfde instellingen als de
productiepijplijn (beam_size=7, temperature=[0.0, 0.2], vad_filter=True,
no_speech_threshold=0.9 -- zie simul_whisper/config.py's batch-defaults).

LET OP: FLEURS is schone, ingesproken (gescripte) studio-spraak -- makkelijker
dan een spontaan, geaccentueerd interview. Een goede score hier is een
best-case bovengrens, geen garantie voor prestaties op echte gehoor-audio
(zie de eerdere bevindingen op de YouTube-testclips: herhalings-hallucinatie,
implausibel korte tekst, Swahili-woordintrusies). Dit script beantwoordt puur
de vraag: "kan dit model Somalisch principieel aan, onder ideale
omstandigheden" -- niet "presteert het goed genoeg voor een echt gehoor".

Vereist (eenmalig, niet in pyproject.toml -- dit is een wegwerpscript):
    pip install datasets jiwer

Gebruik op de RunPod-pod (venv actief, internet nodig voor de FLEURS- en
large-v3-download):
    python scripts/eval_somali_wer_fleurs.py
"""

import io
import re

import soundfile as sf
from datasets import Audio, get_dataset_config_names, load_dataset
from faster_whisper import WhisperModel
from jiwer import cer, wer

PAZA_MODEL_DIR = "/workspace/models/paza-whisper-large-v3-turbo-ct2"
STOCK_MODEL_NAME = "large-v3"
N_SAMPLES = 10
SPLIT = "test"


def resolve_somali_config() -> str:
    """FLEURS-configs volgen <taal>_<land> (ISO 639-1 + ISO 3166-1), dus
    waarschijnlijk 'so_so' -- val terug op auto-detectie als dat niet klopt,
    zodat dit script niet breekt op een verkeerd gegokte exacte naam."""
    candidate = "so_so"
    try:
        configs = get_dataset_config_names("google/fleurs")
    except Exception as e:
        print(f"Kon configlijst niet ophalen ({e}), probeer '{candidate}' direct.")
        return candidate
    if candidate in configs:
        return candidate
    matches = [c for c in configs if c.startswith("so_")]
    if not matches:
        raise RuntimeError(
            f"Geen Somalisch-config gevonden in FLEURS. Beschikbare configs: {configs}"
        )
    print(f"'{candidate}' niet gevonden in configlijst, gebruik in plaats daarvan: {matches[0]}")
    return matches[0]


def normalize(text: str) -> str:
    """Lichte normalisatie voor een eerlijke WER-vergelijking: kleine letters,
    interpunctie weg, dubbele spaties weg. Geen taalspecifieke stemming --
    dat zou de score kunstmatig kunnen verbeteren/verslechteren."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    config = resolve_somali_config()
    print(f"FLEURS-config: {config}, split={SPLIT}, n={N_SAMPLES}")
    ds = load_dataset("google/fleurs", config, split=f"{SPLIT}[:{N_SAMPLES}]")
    # Nieuwere datasets-versies decoderen audio via torchcodec (eigen torch-versie-eis,
    # zelfde soort valkuil als de eerdere transformers/torch-mismatch) -- decode=False
    # geeft de ruwe bestandsbytes terug, die we hieronder zelf met soundfile inlezen
    # (al een bestaande dependency, geen nieuwe torch-gevoelige package nodig).
    ds = ds.cast_column("audio", Audio(decode=False))

    print("Laad modellen (eenmalig, hergebruikt voor alle samples)...")
    models = {
        "paza": WhisperModel(PAZA_MODEL_DIR, device="cuda", compute_type="float16"),
        "large-v3": WhisperModel(STOCK_MODEL_NAME, device="cuda", compute_type="float16"),
    }

    results = {name: {"refs": [], "hyps": []} for name in models}

    for i, sample in enumerate(ds):
        audio, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            raise RuntimeError(f"Onverwachte samplerate {sr}, verwacht 16000")
        reference = sample.get("transcription") or sample.get("raw_transcription")
        if not reference:
            print(f"[{i}] geen referentietekst gevonden, velden: {list(sample.keys())}")
            continue

        print(f"\n--- sample {i} ---")
        print("REF      :", reference)

        for label, model in models.items():
            segments, _info = model.transcribe(
                audio,
                language="so",
                beam_size=7,
                temperature=[0.0, 0.2],
                condition_on_previous_text=False,
                vad_filter=True,
                no_speech_threshold=0.9,
            )
            hyp = " ".join(seg.text.strip() for seg in segments).strip()
            print(f"{label.upper():9s}:", hyp)
            results[label]["refs"].append(normalize(reference))
            results[label]["hyps"].append(normalize(hyp) or " ")  # jiwer verslikt zich in lege hyp

    print("\n=== Aggregaat (lager = beter) ===")
    for label, data in results.items():
        if not data["refs"]:
            continue
        w = wer(data["refs"], data["hyps"])
        c = cer(data["refs"], data["hyps"])
        print(f"{label:10s} WER={w:.3f}  CER={c:.3f}  (n={len(data['refs'])})")


if __name__ == "__main__":
    main()
