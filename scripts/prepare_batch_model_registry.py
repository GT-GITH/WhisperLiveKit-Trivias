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
    "ct2_dir": "<lokaal pad waar het geconverteerde model moet komen>"
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

from huggingface_hub import hf_hub_download

REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "batch_model_registry.json"
)


def ensure_ct2_model(lang: str, hf_repo: str, ct2_dir: str) -> None:
    os.makedirs(ct2_dir, exist_ok=True)
    if os.path.isfile(os.path.join(ct2_dir, "model.bin")):
        print(f"[{lang}] {ct2_dir} al aanwezig -> conversie skip")
        return
    print(f"[{lang}] converteer {hf_repo} -> {ct2_dir} (CT2, float16, kan even duren)...")
    subprocess.run(
        [
            "ct2-transformers-converter",
            "--model", hf_repo,
            "--output_dir", ct2_dir,
            "--quantization", "float16",
            "--force",
        ],
        check=True,
    )
    # ct2-transformers-converter neemt nooit preprocessor_config.json mee (alleen
    # gewichten + tokenizer) -- zonder dat bestand valt faster-whisper terug op
    # feature_size=80 i.p.v. wat de architectuur (bv. 128 voor large-v3) vereist.
    # Zie project-memory "modelroutering-poc" voor de volledige achtergrond.
    preproc_path = os.path.join(ct2_dir, "preprocessor_config.json")
    if not os.path.isfile(preproc_path):
        print(f"[{lang}] haal preprocessor_config.json op voor {hf_repo}...")
        src = hf_hub_download(hf_repo, "preprocessor_config.json")
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
            ensure_ct2_model(lang, hf_repo, ct2_dir)
        except Exception as e:
            print(f"[{lang}] FOUT bij voorbereiden ({hf_repo} -> {ct2_dir}): {e}", file=sys.stderr)
            print(f"[{lang}] blijft op het server-brede standaardmodel (fail-safe).")


if __name__ == "__main__":
    main()
