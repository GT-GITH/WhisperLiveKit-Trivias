"""Eenmalig evaluatiescript, GEEN onderdeel van de reguliere pijplijn.

Doel: objectieve, native-speaker-onafhankelijke ASR-kwaliteitsmeting voor
Somalisch. Gebruikt google/fleurs (Google's meertalige ASR-benchmark --
102 talen, elk fragment met een geverifieerd correct referentietranscript)
en berekent Word Error Rate (WER) / Character Error Rate (CER) voor vier
kandidaten op DEZELFDE fragmenten, met dezelfde instellingen als de
productiepijplijn (beam_size=7, temperature=[0.0, 0.2], vad_filter=True,
no_speech_threshold=0.9 -- zie simul_whisper/config.py's batch-defaults):
  - paza     microsoft/paza-whisper-large-v3-turbo (eerste PoC-kandidaat;
             FLEURS-test 2026-09-18: WER 3.41 -- systematische herhalings-
             hallucinatie op alle 10 samples, ongeschikt)
  - large-v3 het server-brede standaardmodel (FLEURS-test 2026-09-18:
             WER 0.92 / CER 0.34 -- de WER is kunstmatig hoog door
             woordgrens-verschillen, CER geeft een eerlijker beeld)
  - steja    steja/whisper-large-somali, fine-tune van whisper-large-v2 OP
             de FLEURS so_so-trainingsset (dus mogelijk optimistisch op de
             testset t.o.v. audio in het wild) -- eigen modelkaart claimt
             WER 55.0 op de FLEURS-testset
  - sunbird  Sunbird/asr-whisper-51-african-languages, fine-tune van
             whisper-large-v3 op 51 Afrikaanse talen (Sunbird AI, juli 2026)
             -- eigen modelkaart claimt WER 0.381 / CER 0.127 voor Somalisch.
             LET OP: modelkaart labelt dit expliciet als "preview release,
             gewichten kunnen nog wijzigen"

steja en sunbird worden bij de eerste run automatisch naar CTranslate2
geconverteerd (zelfde twee stappen als scripts/init.sh's
prepare_somali_batch_model(): transformers<5 i.v.m. de torch-pin, en
preprocessor_config.json omdat ct2-transformers-converter die zelf niet
meeneemt -- zie commits a264f78/6ec48a7 voor de achtergrond), daarna
hergebruikt uit /workspace/models/.

LET OP: FLEURS is schone, ingesproken (gescripte) studio-spraak -- makkelijker
dan een spontaan, geaccentueerd interview. Een goede score hier is een
best-case bovengrens, geen garantie voor prestaties op echte gehoor-audio
(zie de eerdere bevindingen op de YouTube-testclips: herhalings-hallucinatie,
implausibel korte tekst, Swahili-woordintrusies). Dit script beantwoordt puur
de vraag: "kan dit model Somalisch principieel aan, onder ideale
omstandigheden" -- niet "presteert het goed genoeg voor een echt gehoor".

Vereist (eenmalig, niet in pyproject.toml -- dit is een wegwerpscript):
    pip install datasets jiwer

Gebruik op de RunPod-pod (venv actief, internet nodig voor FLEURS + de
model-downloads -- steja/sunbird zijn elk ~3GB, reken op wat tijd):
    python scripts/eval_somali_wer_fleurs.py
"""

import io
import json
import os
import re
import shutil
import subprocess

# Vóór elke import die HF Hub kan aanroken: deze RunPod-omgeving heeft
# HF_HUB_ENABLE_HF_TRANSFER=1 zonder het bijbehorende hf_transfer-pakket
# geinstalleerd (zelfde val als scripts/init.sh's startlive() al ondervangt
# voor de servercontext, maar dat gold niet voor los uitgevoerde scripts --
# hier nu ook hardcoded i.p.v. per keer een env-var-prefix te moeten
# onthouden, geconstateerd na 2x dezelfde fout op de RunPod-pod).
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import soundfile as sf
from datasets import Audio, get_dataset_config_names, load_dataset
from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download, snapshot_download
from jiwer import cer, wer

PAZA_MODEL_DIR = "/workspace/models/paza-whisper-large-v3-turbo-ct2"
STOCK_MODEL_NAME = "large-v3"
STEJA_HF_REPO = "steja/whisper-large-somali"
STEJA_MODEL_DIR = "/workspace/models/steja-whisper-large-somali-ct2"
SUNBIRD_HF_REPO = "Sunbird/asr-whisper-51-african-languages"
SUNBIRD_MODEL_DIR = "/workspace/models/sunbird-asr-whisper-51-african-ct2"
N_SAMPLES = 10
SPLIT = "test"

# Whisper's byte-level BPE-tokenizer is identiek over alle checkpoints/finetunes
# heen -- sommige oudere/incomplete fine-tune-repo's (bv. steja/whisper-large-somali)
# uploaden daarom geen eigen tokenizer-bestanden, wat ct2-transformers-converter
# laat crashen op een ontbrekend vocab_file. Vul aan vanuit een bekend-compleet
# basismodel als het lokaal ontbreekt.
WHISPER_TOKENIZER_FALLBACK_REPO = "openai/whisper-large-v2"
WHISPER_TOKENIZER_FILES = [
    "vocab.json", "merges.txt", "tokenizer_config.json",
    "normalizer.json", "added_tokens.json", "special_tokens_map.json",
]


def _ensure_tokenizer_files(staging_dir: str) -> None:
    for fname in WHISPER_TOKENIZER_FILES:
        if os.path.isfile(os.path.join(staging_dir, fname)):
            continue
        try:
            src = hf_hub_download(WHISPER_TOKENIZER_FALLBACK_REPO, fname)
        except Exception:
            continue  # niet elk Whisper-checkpoint publiceert alle 6 bestanden, prima
        shutil.copy(src, os.path.join(staging_dir, fname))


def _fix_extra_special_tokens(staging_dir: str) -> None:
    """Sunbird/asr-whisper-51-african-languages serialiseert tokenizer_config.json's
    extra_special_tokens als lege lijst i.p.v. dict -- transformers' PreTrainedTokenizerBase
    roept hier altijd .keys() op aan en crasht dan met AttributeError. Upgraden van
    transformers loste dit niet op (geen versieregressie, een eigenaardigheid van dit
    specifieke bestand), dus patch het bestand hier gericht."""
    path = os.path.join(staging_dir, "tokenizer_config.json")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if isinstance(config.get("extra_special_tokens"), list):
        if config["extra_special_tokens"]:
            print(
                f"WAARSCHUWING: extra_special_tokens in {path} is een niet-lege lijst "
                f"({config['extra_special_tokens']}) -- automatisch geleegd naar {{}} "
                f"i.p.v. omgezet, mogelijk verlies van info."
            )
        config["extra_special_tokens"] = {}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f)


def ensure_ct2_model(hf_repo: str, local_dir: str, preprocessor_fallback_repo: str | None = None) -> str:
    """Idempotente CT2-conversie + preprocessor_config.json-aanvulling --
    zelfde twee stappen als prepare_somali_batch_model() in scripts/init.sh.

    Download het HF-model eerst zelf lokaal (i.p.v. de repo-naam rechtstreeks
    aan ct2-transformers-converter te geven): recente transformers-versies
    weigeren pickle-.bin-checkpoints te laden tenzij torch>=2.6
    (CVE-2025-32434). steja/whisper-large-somali publiceert ALLEEN zo'n
    .bin-bestand (geen .safetensors-alternatief) -- dus lossen we dat hier
    zelf op met een rechtstreekse torch.load() (torch zelf blokkeert dit
    niet, alleen transformers' eigen from_pretrained-guard) gevolgd door een
    safetensors-export, i.p.v. het bestand simpelweg uit te sluiten."""
    os.makedirs(local_dir, exist_ok=True)
    if not os.path.isfile(os.path.join(local_dir, "model.bin")):
        staging_dir = local_dir + "-hf-src"
        print(f"Download {hf_repo} -> {staging_dir}...")
        snapshot_download(hf_repo, local_dir=staging_dir, ignore_patterns=["*.h5", "*.msgpack"])
        _ensure_tokenizer_files(staging_dir)
        _fix_extra_special_tokens(staging_dir)

        has_safetensors = any(f.endswith(".safetensors") for f in os.listdir(staging_dir))
        if not has_safetensors:
            bin_path = os.path.join(staging_dir, "pytorch_model.bin")
            if not os.path.isfile(bin_path):
                raise RuntimeError(
                    f"Geen .safetensors en geen pytorch_model.bin gevonden voor "
                    f"{hf_repo} in {staging_dir}"
                )
            print(f"Geen safetensors voor {hf_repo} -- converteer {bin_path} lokaal...")
            import torch
            from safetensors.torch import save_file

            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
            # .clone() breekt gedeeld geheugen (bv. getiede embeddings, gangbaar bij
            # Whisper) -- safetensors weigert anders te serialiseren; .contiguous()
            # is een vereiste van het safetensors-formaat.
            state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
            save_file(state_dict, os.path.join(staging_dir, "model.safetensors"))
            os.remove(bin_path)

        print(f"Converteer {staging_dir} -> {local_dir} (CT2, float16, kan even duren)...")
        subprocess.run(
            [
                "ct2-transformers-converter",
                "--model", staging_dir,
                "--output_dir", local_dir,
                "--quantization", "float16",
                "--force",
            ],
            check=True,
        )
        # De originele HF-gewichten (~3GB per model) zijn na een geslaagde conversie
        # overbodig -- alleen de CT2-output in local_dir wordt hergebruikt. Zonder
        # deze cleanup liepen we op de RunPod-pod vast op schijfruimte (FLEURS +
        # meerdere modellen x tijdelijk dubbele opslag tijdens de .bin->safetensors-
        # stap hierboven).
        shutil.rmtree(staging_dir, ignore_errors=True)
    else:
        print(f"{local_dir} al aanwezig -> conversie skip")

    preproc_path = os.path.join(local_dir, "preprocessor_config.json")
    if not os.path.isfile(preproc_path):
        print(f"Haal preprocessor_config.json op voor {hf_repo}...")
        try:
            src = hf_hub_download(hf_repo, "preprocessor_config.json")
        except Exception as e:
            if not preprocessor_fallback_repo:
                raise
            # Sunbird publiceert alleen processor_config.json (een generieke wrapper
            # zonder de feature-extractie-parameters die faster-whisper nodig heeft),
            # geen preprocessor_config.json. Mel-bank-parameters zijn architectuur-
            # bepaald, niet finetune-specifiek -- val terug op het basismodel.
            print(f"{hf_repo} publiceert geen preprocessor_config.json ({e}) -- val terug op {preprocessor_fallback_repo}")
            src = hf_hub_download(preprocessor_fallback_repo, "preprocessor_config.json")
        shutil.copy(src, preproc_path)
    return local_dir


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
    model_sources = {
        "paza": lambda: PAZA_MODEL_DIR,
        "large-v3": lambda: STOCK_MODEL_NAME,
        "steja": lambda: ensure_ct2_model(STEJA_HF_REPO, STEJA_MODEL_DIR),
        "sunbird": lambda: ensure_ct2_model(
            SUNBIRD_HF_REPO, SUNBIRD_MODEL_DIR, preprocessor_fallback_repo="openai/whisper-large-v3"
        ),
    }
    models = {}
    for label, get_source in model_sources.items():
        try:
            models[label] = WhisperModel(get_source(), device="cuda", compute_type="float16")
        except Exception as e:
            # Eén ontoegankelijke/kapotte kandidaat (bv. Sunbird is een gated repo --
            # vereist handmatig toegang aanvragen op de modelpagina, een HF-token
            # alleen is niet genoeg) mag de vergelijking voor de rest niet blokkeren.
            print(f"[{label}] kon niet geladen worden, sla over: {e}")

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
