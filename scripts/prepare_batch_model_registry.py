"""Converteert elk in batch_model_registry.json vermeld taalmodel naar CTranslate2,
idempotent (slaat een taal over zodra ct2_dir/model.bin al bestaat).

Aangeroepen door scripts/init.sh tijdens --start/--setup-start, VOOR de server
opstart -- zodat een geconfigureerde taal al klaarstaat op het moment dat een
operator een sessie start, in plaats van een live download te doen op het moment
dat "Start opname" geklikt wordt (zie CLAUDE.md: on-premises, geen afhankelijkheid
van een netwerkdownload op het moment dat een verhoor zou moeten beginnen).

Registry-formaat (batch_model_registry.json, projectroot, git-getrackt):
{
  "<taalcode>": {
    "hf_repo": "<HuggingFace-repo, gewoon PyTorch/HF-checkpoint>",
    "ct2_dir": "<lokaal pad waar het geconverteerde model moet komen>",
    "preprocessor_fallback_repo": "<optioneel -- alleen nodig als hf_repo zelf
       geen preprocessor_config.json publiceert; bv. 'openai/whisper-large-v3-
       turbo' voor een large-v3-turbo-finetune>"
  }
}

Leeg ({}) = geen enkele taal krijgt een specialisatie, dit script doet dan niets.
Elke vermelding hier moet eerst apart gevalideerd zijn (WER/CER tegen een
referentietranscriptset, zie feedback-memory "verify-asr-model-claims-independently")
voor hij hier wordt toegevoegd -- dit script converteert alleen, het beoordeelt niet.

Fail-safe per taal: als één vermelding faalt (netwerk, kapot bestand in het
bronmodel, etc.) wordt dat gelogd en gaat het script door met de overige
vermeldingen -- TranscriptionEngine valt vanzelf terug op het standaardmodel
voor een taal waarvan de conversie niet gelukt is.
"""

import json
import os
import shutil
import subprocess
import sys

from huggingface_hub import hf_hub_download, snapshot_download

REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "batch_model_registry.json"
)


def _fix_extra_special_tokens(staging_dir: str) -> None:
    """Meerdere Whisper-finetunes (Sunbird, oddadmix) publiceren in
    tokenizer_config.json `extra_special_tokens` als lijst i.p.v. dict;
    transformers roept daar .keys() op aan en crasht (AttributeError: 'list'
    object has no attribute 'keys'). Zelfde patch als in
    scripts/eval_arabic_wer_casablanca.py -- die stond daar wel, maar ontbrak
    hier, waardoor de conversie op een schone pod (2026-11-15) faalde: op de
    oude pod had het evaluatiescript het model al geconverteerd."""
    path = os.path.join(staging_dir, "tokenizer_config.json")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if isinstance(config.get("extra_special_tokens"), list):
        print(f"patch {path}: extra_special_tokens lijst -> dict")
        config["extra_special_tokens"] = {}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f)


def ensure_ct2_model(lang: str, hf_repo: str, ct2_dir: str, preprocessor_fallback_repo: str = None) -> None:
    os.makedirs(ct2_dir, exist_ok=True)
    if os.path.isfile(os.path.join(ct2_dir, "model.bin")):
        print(f"[{lang}] {ct2_dir} al aanwezig -> conversie skip")
    else:
        # Eerst zelf lokaal downloaden (i.p.v. de repo-naam rechtstreeks aan de
        # converter te geven), zodat een kapotte tokenizer_config.json gepatcht
        # kan worden vóór de conversie.
        staging_dir = ct2_dir.rstrip("/") + "-hf-src"
        print(f"[{lang}] download {hf_repo} -> {staging_dir}...")
        snapshot_download(hf_repo, local_dir=staging_dir, ignore_patterns=["*.h5", "*.msgpack"])
        _fix_extra_special_tokens(staging_dir)
        print(f"[{lang}] converteer {staging_dir} -> {ct2_dir} (CT2, float16, kan even duren)...")
        subprocess.run(
            [
                "ct2-transformers-converter",
                "--model", staging_dir,
                "--output_dir", ct2_dir,
                "--quantization", "float16",
                "--force",
            ],
            check=True,
        )
        shutil.rmtree(staging_dir, ignore_errors=True)

    # ct2-transformers-converter neemt nooit preprocessor_config.json mee (alleen
    # gewichten + tokenizer) -- zonder dat bestand valt faster-whisper terug op
    # feature_size=80 i.p.v. wat de architectuur (bv. 128 voor large-v3) vereist.
    # Buiten de "al geconverteerd"-tak om: een eerdere run kan dit bestand gemist
    # hebben terwijl model.bin al wel bestond. Zie project-memory "modelroutering-poc".
    preproc_path = os.path.join(ct2_dir, "preprocessor_config.json")
    if not os.path.isfile(preproc_path):
        print(f"[{lang}] haal preprocessor_config.json op voor {hf_repo}...")
        try:
            src = hf_hub_download(hf_repo, "preprocessor_config.json")
        except Exception as e:
            if not preprocessor_fallback_repo:
                raise
            # Sommige finetunes (bv. oddadmix/whisper-large-v3-turbo-arabic-dialectal)
            # publiceren alleen processor_config.json (generieke wrapper, geen
            # feature-extractie-parameters). Mel-bank-parameters zijn architectuur-
            # bepaald, niet finetune-specifiek -- val terug op het in de registry
            # opgegeven basismodel i.p.v. te crashen.
            print(f"[{lang}] {hf_repo} publiceert geen preprocessor_config.json ({e}) -- val terug op {preprocessor_fallback_repo}")
            src = hf_hub_download(preprocessor_fallback_repo, "preprocessor_config.json")
        shutil.copy(src, preproc_path)


def main() -> None:
    if not os.path.isfile(REGISTRY_PATH):
        print(f"Geen batch_model_registry.json gevonden op {REGISTRY_PATH}, niets te doen.")
        return
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)
    if not registry:
        print("batch_model_registry.json is leeg -- geen taalspecifieke modellen om voor te bereiden.")
        return

    for lang, entry in registry.items():
        hf_repo = entry.get("hf_repo")
        ct2_dir = entry.get("ct2_dir")
        if not hf_repo or not ct2_dir:
            print(f"[{lang}] WAARSCHUWING: vermelding mist 'hf_repo' of 'ct2_dir', overgeslagen: {entry}")
            continue
        try:
            ensure_ct2_model(lang, hf_repo, ct2_dir, entry.get("preprocessor_fallback_repo"))
        except Exception as e:
            print(f"[{lang}] FOUT bij voorbereiden ({hf_repo} -> {ct2_dir}): {e}", file=sys.stderr)
            print(f"[{lang}] blijft op het server-brede standaardmodel (fail-safe).")


if __name__ == "__main__":
    main()
