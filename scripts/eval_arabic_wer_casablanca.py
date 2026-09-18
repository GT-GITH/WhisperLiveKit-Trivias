"""Eenmalig evaluatiescript, GEEN onderdeel van de reguliere pijplijn.

Doel: objectieve baseline-meting van het server-brede standaardmodel (large-v3)
op Arabisch, per dialect -- vóór er wordt gezocht naar een gespecialiseerd
model. Gebruikt UBC-NLP/Casablanca (EMNLP 2024, handmatig getranscribeerd,
acht dialecten) i.p.v. FLEURS: FLEURS heeft voor Arabisch alleen een
Egyptisch-dialect-config, niet representatief voor de daadwerkelijke IND-
doelgroep (zie Gehoren/Rapport complexiteit... .pdf, tabel B3.1: Syrisch,
Iraaks, Marokkaans, Algerijns, Jemenitisch structureel in de top-10-
nationaliteiten 2013-2022).

Test de dialecten die het dichtst bij die doelgroep liggen:
  - jordan / palestine  -> proxy voor Levantijns/Syrisch (Syrië zelf zit niet
    apart in Casablanca, Jordaans/Palestijns is dialectisch het dichtst bij)
  - morocco / algeria   -> Maghrebijns (per de Interspeech-2025-leaderboard en
    oddadmix's eigen modelkaart het lastigste dialect voor elk getest model)
  - yemen               -> Jemenitisch

Egypt/mauritania/uae worden overgeslagen -- minder relevant voor de IND-
doelgroep, en dit script gaat om een gerichte baseline, niet volledigheid.

Vereist (eenmalig, niet in pyproject.toml):
    pip install datasets jiwer

Gebruik op de RunPod-pod (venv actief, internet nodig voor Casablanca-download,
~1GB):
    python scripts/eval_arabic_wer_casablanca.py
"""

import io
import os
import re

# Vóór elke import die HF Hub kan aanroken -- zie eerdere Somalisch-evaluaties
# voor de achtergrond (HF_HUB_ENABLE_HF_TRANSFER=1 zonder hf_transfer-pakket in
# deze RunPod-omgeving).
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import librosa
import soundfile as sf
from datasets import Audio, get_dataset_config_names, load_dataset
from faster_whisper import WhisperModel
from jiwer import cer, wer

MODEL_NAME = "large-v3"
DIALECT_CONFIGS = ["jordan", "palestine", "morocco", "algeria", "yemen"]
SPLIT = "test"
N_SAMPLES_PER_DIALECT = 20
TARGET_SR = 16000


def resolve_configs() -> list[str]:
    """Casablanca's exacte config-namen (hoofdlettergebruik e.d.) zijn niet
    100% zeker vooraf -- val terug op case-insensitive matching tegen de
    daadwerkelijke configlijst i.p.v. te crashen op een verkeerd gegokte naam."""
    try:
        available = get_dataset_config_names("UBC-NLP/Casablanca")
    except Exception as e:
        print(f"Kon configlijst niet ophalen ({e}), probeer de namen direct.")
        return DIALECT_CONFIGS
    lower_map = {c.lower(): c for c in available}
    resolved = []
    for wanted in DIALECT_CONFIGS:
        match = lower_map.get(wanted.lower())
        if match:
            resolved.append(match)
        else:
            print(f"WAARSCHUWING: config '{wanted}' niet gevonden in {available}, overgeslagen.")
    return resolved


def normalize(text: str) -> str:
    """Lichte normalisatie voor een eerlijke WER-vergelijking. Geen Arabisch-
    specifieke normalisatie (bv. hamza/alef-varianten, diakrieten) -- dat zou
    de score kunstmatig kunnen verbeteren; we willen zien wat het model kaal
    teruggeeft."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_audio(raw_bytes: bytes) -> "tuple[list[float], int]":
    audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    return audio, sr


def main() -> None:
    configs = resolve_configs()
    print(f"Casablanca-dialecten: {configs}, split={SPLIT}, n/dialect={N_SAMPLES_PER_DIALECT}")

    print(f"Laad {MODEL_NAME}...")
    model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")

    all_refs, all_hyps = [], []
    per_dialect_results = {}

    for dialect in configs:
        print(f"\n=== Dialect: {dialect} ===")
        try:
            ds = load_dataset("UBC-NLP/Casablanca", dialect, split=f"{SPLIT}[:{N_SAMPLES_PER_DIALECT}]")
        except Exception as e:
            print(f"[{dialect}] kon dataset niet laden, overgeslagen: {e}")
            continue
        ds = ds.cast_column("audio", Audio(decode=False))

        refs, hyps = [], []
        for i, sample in enumerate(ds):
            audio, sr = load_audio(sample["audio"]["bytes"])
            reference = sample.get("transcription")
            if not reference:
                print(f"[{dialect}][{i}] geen referentietekst, velden: {list(sample.keys())}")
                continue

            segments, _info = model.transcribe(
                audio,
                language="ar",
                beam_size=7,
                temperature=[0.0, 0.2],
                condition_on_previous_text=False,
                vad_filter=True,
                no_speech_threshold=0.9,
            )
            hyp = " ".join(seg.text.strip() for seg in segments).strip()

            print(f"[{dialect}][{i}] REF: {reference}")
            print(f"[{dialect}][{i}] HYP: {hyp}")

            refs.append(normalize(reference))
            hyps.append(normalize(hyp) or " ")

        if refs:
            d_wer, d_cer = wer(refs, hyps), cer(refs, hyps)
            per_dialect_results[dialect] = (d_wer, d_cer, len(refs))
            all_refs.extend(refs)
            all_hyps.extend(hyps)

    print(f"\n=== Resultaat per dialect ({MODEL_NAME}) ===")
    for dialect, (d_wer, d_cer, n) in per_dialect_results.items():
        print(f"{dialect:12s} WER={d_wer:.3f}  CER={d_cer:.3f}  (n={n})")

    if all_refs:
        print(f"\n=== Gemiddeld over alle dialecten ===")
        print(f"WER={wer(all_refs, all_hyps):.3f}  CER={cer(all_refs, all_hyps):.3f}  (n={len(all_refs)})")


if __name__ == "__main__":
    main()
