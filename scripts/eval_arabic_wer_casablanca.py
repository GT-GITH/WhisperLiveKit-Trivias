"""Eenmalig evaluatiescript, GEEN onderdeel van de reguliere pijplijn.

Doel: large-v3 (server-brede standaard) vergelijken met een Maghrebijns-
gerichte kandidaat over ALLE vijf geteste dialecten (Jordaans/Palestijns als
proxy voor Levantijns/Syrisch, Marokkaans, Algerijns, Jemenitisch) -- niet
alleen op Marokkaans/Algerijns, waar large-v3 het eerder het slechtst deed
(baseline: Morocco WER=0.904/CER=0.432, Algeria WER=0.788/CER=0.674,
inclusief concrete hallucinaties zoals een letterlijke "abonneer je op het
kanaal"-YouTube-outro-hallucinatie op Algeria[9]; op oddadmix daarna gemeten:
Morocco WER=0.640/CER=0.250, Algeria WER=0.743/CER=0.295 -- forse verbetering,
onafhankelijk bevestigd, geen hallucinaties meer op dezelfde fragmenten).

Openstaande vraag die deze volledige run beantwoordt: oddadmix is getraind op
Levantijns/Maghrebijns/Egyptisch/Golf/Soedanees/Iraaks/MSA tegelijk -- als het
óók minstens even goed is op Jordaans/Palestijns/Jemenitisch (waar large-v3
al prima scoorde), hoeft er geen dialect-specifieke routering gebouwd te
worden en kan gewoon AL het Arabisch naar oddadmix. Scoort het daar juist
slechter (specialisatie-afruil), dan is dialect-routering wél nodig.

Kandidaat: oddadmix/whisper-large-v3-turbo-arabic-dialectal. Eigen modelkaart:
WER 0.344/CER 0.115 op een eigen testset (932 clips), met de expliciete
kanttekening "real-world dialect coverage varies (Maghrebi is the hardest)"
en "Private / internal model. Evaluate on your own data before production
use." -- vandaar deze eigen, onafhankelijke meting i.p.v. dat cijfer over te
nemen (zie feedback-memory verify-asr-model-claims-independently).

Gebruikt UBC-NLP/Casablanca (EMNLP 2024, handmatig getranscribeerd) i.p.v.
FLEURS: FLEURS heeft voor Arabisch alleen een Egyptisch-dialect-config, niet
representatief voor de IND-doelgroep (zie Gehoren/Rapport complexiteit...pdf,
tabel B3.1).

Vereist (eenmalig, niet in pyproject.toml):
    pip install datasets jiwer librosa

Gebruik op de RunPod-pod (venv actief, internet nodig voor Casablanca +
oddadmix-download/conversie, ~3GB):
    python scripts/eval_arabic_wer_casablanca.py
"""

import io
import json
import os
import re
import shutil
import subprocess

# Vóór elke import die HF Hub kan aanroken -- zie eerdere Somalisch-evaluaties
# voor de achtergrond (HF_HUB_ENABLE_HF_TRANSFER=1 zonder hf_transfer-pakket in
# deze RunPod-omgeving).
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import librosa
import soundfile as sf
from datasets import Audio, get_dataset_config_names, load_dataset
from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download, snapshot_download
from jiwer import cer, wer

STOCK_MODEL_NAME = "large-v3"
ODDADMIX_HF_REPO = "oddadmix/whisper-large-v3-turbo-arabic-dialectal"
ODDADMIX_CT2_DIR = "/workspace/models/oddadmix-arabic-dialectal-ct2"

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


def _fix_extra_special_tokens(staging_dir: str) -> None:
    """Twee onafhankelijke Whisper-finetune-auteurs (Sunbird, oddadmix) bleken
    dezelfde tokenizer_config.json-eigenaardigheid te publiceren: extra_special_tokens
    als lege lijst i.p.v. dict. transformers' PreTrainedTokenizerBase roept hier
    altijd .keys() op aan en crasht met AttributeError -- vermoedelijk een bug in
    een veelgebruikt Whisper-finetune-exportscript, dus deze patch is breed
    herbruikbaar voor toekomstige kandidaten."""
    path = os.path.join(staging_dir, "tokenizer_config.json")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if isinstance(config.get("extra_special_tokens"), list):
        config["extra_special_tokens"] = {}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f)


def ensure_ct2_model(hf_repo: str, ct2_dir: str, preprocessor_fallback_repo: str | None = None) -> str:
    """Download het HF-model eerst zelf lokaal (i.p.v. de repo-naam rechtstreeks
    aan ct2-transformers-converter te geven), zodat een kapotte tokenizer_config.json
    gepatcht kan worden vóór de conversie. Zelfde twee vervolgstappen als
    scripts/prepare_batch_model_registry.py: CT2-conversie + preprocessor_config.json-
    aanvulling (ct2-transformers-converter neemt dat bestand nooit vanzelf mee)."""
    os.makedirs(ct2_dir, exist_ok=True)
    if not os.path.isfile(os.path.join(ct2_dir, "model.bin")):
        staging_dir = ct2_dir + "-hf-src"
        print(f"Download {hf_repo} -> {staging_dir}...")
        snapshot_download(hf_repo, local_dir=staging_dir, ignore_patterns=["*.h5", "*.msgpack"])
        _fix_extra_special_tokens(staging_dir)
        print(f"Converteer {staging_dir} -> {ct2_dir} (CT2, float16, kan even duren)...")
        subprocess.run(
            ["ct2-transformers-converter", "--model", staging_dir, "--output_dir", ct2_dir,
             "--quantization", "float16", "--force"],
            check=True,
        )
        shutil.rmtree(staging_dir, ignore_errors=True)
    preproc_path = os.path.join(ct2_dir, "preprocessor_config.json")
    if not os.path.isfile(preproc_path):
        print(f"Haal preprocessor_config.json op voor {hf_repo}...")
        try:
            src = hf_hub_download(hf_repo, "preprocessor_config.json")
        except Exception as e:
            if not preprocessor_fallback_repo:
                raise
            # oddadmix publiceert alleen processor_config.json (generieke wrapper,
            # geen feature-extractie-parameters) -- mel-bank-parameters zijn
            # architectuurbepaald, niet finetune-specifiek, dus val terug op het
            # basismodel (zelfde patroon als eerder bij Sunbird).
            print(f"{hf_repo} publiceert geen preprocessor_config.json ({e}) -- val terug op {preprocessor_fallback_repo}")
            src = hf_hub_download(preprocessor_fallback_repo, "preprocessor_config.json")
        shutil.copy(src, preproc_path)
    return ct2_dir


def normalize(text: str) -> str:
    """Lichte normalisatie voor een eerlijke WER-vergelijking. Geen Arabisch-
    specifieke normalisatie (bv. hamza/alef-varianten, diakrieten) -- dat zou
    de score kunstmatig kunnen verbeteren; we willen zien wat het model kaal
    teruggeeft."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_audio(raw_bytes: bytes):
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

    print("Laad modellen...")
    models = {
        "large-v3": WhisperModel(STOCK_MODEL_NAME, device="cuda", compute_type="float16"),
    }
    try:
        oddadmix_dir = ensure_ct2_model(
            ODDADMIX_HF_REPO, ODDADMIX_CT2_DIR, preprocessor_fallback_repo="openai/whisper-large-v3-turbo"
        )
        models["oddadmix-dialectal"] = WhisperModel(oddadmix_dir, device="cuda", compute_type="float16")
    except Exception as e:
        print(f"[oddadmix-dialectal] kon niet geladen worden, sla over: {e}")

    # Vooraf alle dialect-datasets laden (één keer per dialect, hergebruikt over modellen).
    dialect_samples = {}
    for dialect in configs:
        try:
            ds = load_dataset("UBC-NLP/Casablanca", dialect, split=f"{SPLIT}[:{N_SAMPLES_PER_DIALECT}]")
        except Exception as e:
            print(f"[{dialect}] kon dataset niet laden, overgeslagen: {e}")
            continue
        dialect_samples[dialect] = ds.cast_column("audio", Audio(decode=False))

    results = {label: {} for label in models}

    for dialect, ds in dialect_samples.items():
        print(f"\n=== Dialect: {dialect} ===")
        for label, model in models.items():
            refs, hyps = [], []
            for i, sample in enumerate(ds):
                audio, sr = load_audio(sample["audio"]["bytes"])
                reference = sample.get("transcription")
                if not reference:
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

                print(f"[{dialect}][{i}][{label}] REF: {reference}")
                print(f"[{dialect}][{i}][{label}] HYP: {hyp}")

                refs.append(normalize(reference))
                hyps.append(normalize(hyp) or " ")

            if refs:
                results[label][dialect] = (wer(refs, hyps), cer(refs, hyps), len(refs))

    print("\n=== Resultaat per dialect en model ===")
    for label, per_dialect in results.items():
        for dialect, (d_wer, d_cer, n) in per_dialect.items():
            print(f"{label:20s} {dialect:10s} WER={d_wer:.3f}  CER={d_cer:.3f}  (n={n})")


if __name__ == "__main__":
    main()
