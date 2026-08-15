#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# RunPod init script (idempotent)
#
# Defaults:
#   REMOTE_BRANCH=main
#   LOCAL_BRANCH=main
#   LLM_ENABLED=1 LLM_MODEL=llama3.1:8b LLM_BACKEND_URL=http://localhost:11434/v1
#     -> on-prem Ollama voor gehoorverslag-sectieclassificatie, altijd lokaal
#        op de pod zelf (nooit een cloud-endpoint). LLM_ENABLED=0 om uit te zetten.
#
# Usage:
#   bash scripts/init.sh --setup         # deps + git + venv + pip
#   bash scripts/init.sh --setup-start   # setup + start server
#   bash scripts/init.sh --start         # start server ONLY (no setup)
#   bash scripts/init.sh --update        # git update only
#   bash scripts/init.sh --deps          # apt deps only
#   bash scripts/init.sh --venv          # venv + pip only
# ------------------------------------------------------------

INIT_VERSION="main-2025-12-19-new"
echo "[init] init.sh version: $INIT_VERSION"

# --- helpers ---
log() { echo -e "[init] $*"; }
die() { echo -e "[init] ❌ $*" >&2; exit 1; }

IS_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  IS_SOURCED=1
fi

# --- config ---
REPO_URL="${REPO_URL:-https://github.com/GT-GITH/WhisperLiveKit-Trivias.git}"
WORKSPACE="${WORKSPACE:-/workspace}"
APP_DIR="${APP_DIR:-$WORKSPACE/WhisperLiveKit-Trivias}"
VENV_DIR="${VENV_DIR:-$APP_DIR/.venv}"

REMOTE_BRANCH="${REMOTE_BRANCH:-main}"
LOCAL_BRANCH="${LOCAL_BRANCH:-main}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-large-v3}"
LANGUAGE="${LANGUAGE:-nl}"
FRAME_THRESHOLD="${FRAME_THRESHOLD:-25}"
AUDIO_MIN_LEN="${AUDIO_MIN_LEN:-0.0}"
AUDIO_MAX_LEN="${AUDIO_MAX_LEN:-30.0}"
BEAMS="${BEAMS:-1}"
DIARIZATION="${DIARIZATION:-0}"
PCM_INPUT="${PCM_INPUT:-1}"  # altijd PCM voor multi-channel stabiliteit
DIARIZATION_BACKEND="${DIARIZATION_BACKEND:-sortformer}"

# On-prem LLM (gehoorverslag-sectieclassificatie, zie llm_backend.py) --
# NOOIT een cloud-endpoint, altijd lokaal op de pod zelf (localhost, geen
# externe port-forward nodig). Default AAN: L40S 48GB heeft ~30GB vrij naast
# Whisper large-v3, ruim genoeg voor een 8B-model. LLM_ENABLED=0 schakelt
# dit volledig uit (Trivias werkt dan gewoon fail-safe verder zonder
# classificatie, zoals altijd zonder --llm-backend-url).
LLM_ENABLED="${LLM_ENABLED:-1}"
LLM_MODEL="${LLM_MODEL:-llama3.1:8b}"
LLM_BACKEND_URL="${LLM_BACKEND_URL:-http://localhost:11434/v1}"
# Ollama's default modellocatie is /root/.ollama/models, op de kleine
# root-schijf van de pod (niet de grote /workspace-volume waar de rest van
# dit project al staat) -- expliciet naar /workspace verplaatst, anders loopt
# een model van een paar GB de root-schijf vol.
OLLAMA_MODELS="${OLLAMA_MODELS:-$WORKSPACE/.ollama-models}"


# --- ensure bash ---
[[ -n "${BASH_VERSION:-}" ]] || die "Dit script vereist bash. Run: bash $0 ..."

# --- git identity (to prevent prompts) ---
git_identity() {
  git config --global user.email "topcug1975@gmail.com" >/dev/null 2>&1 || true
  git config --global user.name "Gokhan Topcu" >/dev/null 2>&1 || true
}

# --- ~/.cache van de kleine root-schijf af ---
redirect_home_cache_to_workspace() {
  # ~/.cache (HuggingFace-modellen, pip-cache, torch-hub, etc. -- allemaal
  # dezelfde XDG-conventie) staat standaard op de kleine, ephemere root-
  # overlay van de pod (hier gezien: 5GB, niet de grote /workspace-volume).
  # Die loopt vroeg of laat vol zodra er modellen gedownload worden -- exact
  # wat er gebeurde: root liep 100% vol, waardoor zelfs "apt-get install"
  # geen dpkg-lock-bestand meer kon wegschrijven. Eenmalig verplaatsen naar
  # /workspace en terugsymlinken lost dit voorgoed op, voor elke tool die
  # naar ~/.cache/... schrijft, zonder per-tool env vars te hoeven zetten.
  if [[ -L "$HOME/.cache" ]]; then
    log "~/.cache is al een symlink → skip"
    return 0
  fi
  mkdir -p "$WORKSPACE/.cache"
  if [[ -d "$HOME/.cache" ]]; then
    log "Verplaats bestaande ~/.cache (kan even duren) naar $WORKSPACE/.cache..."
    shopt -s dotglob nullglob
    mv "$HOME"/.cache/* "$WORKSPACE/.cache/" 2>/dev/null || true
    shopt -u dotglob nullglob
    rmdir "$HOME/.cache" 2>/dev/null || true
  fi
  ln -s "$WORKSPACE/.cache" "$HOME/.cache"
  log "~/.cache → $WORKSPACE/.cache (symlink)"
}

# --- deps ---
install_deps() {
  local marker="$WORKSPACE/.os_deps_installed"
  if [[ -f "$marker" ]]; then
    log "OS deps al gedaan → skip"
    return 0
  fi

  log "Install OS deps..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y >/dev/null
  apt-get install -y git curl ffmpeg python3-venv >/dev/null
  apt-get install -y python3.11 python3.11-venv >/dev/null
  # pciutils (lspci) -- de Ollama-installer gebruikt dit om de NVIDIA-GPU te
  # detecteren en de CUDA-runtime te installeren. Zonder lspci installeert
  # Ollama stilzwijgend CPU-only, ook als nvidia-smi prima werkt.
  apt-get install -y pciutils >/dev/null

  touch "$marker"
}

# --- repo ---
setup_repo() {
  mkdir -p "$WORKSPACE"

  if [[ -e "$APP_DIR" && ! -d "$APP_DIR/.git" ]]; then
    die "APP_DIR bestaat maar is geen git repo: $APP_DIR (verwijder/maak leeg of zet APP_DIR anders)"
  fi

  if [[ ! -d "$APP_DIR/.git" ]]; then
    log "Clone repo → $APP_DIR"
    git clone "$REPO_URL" "$APP_DIR"
  fi

  log "Update repo (fetch/prune)..."
  cd "$APP_DIR"
  git remote set-url origin "$REPO_URL" >/dev/null 2>&1 || true
  git fetch --all --prune >/dev/null

  # verify remote branch exists
  if ! git show-ref --verify --quiet "refs/remotes/origin/$REMOTE_BRANCH"; then
    die "Remote branch bestaat niet: origin/$REMOTE_BRANCH"
  fi

  # Idempotent: create or overwrite local branch to point to remote branch
  log "Checkout local '$LOCAL_BRANCH' ← origin/$REMOTE_BRANCH"
  git checkout -B "$LOCAL_BRANCH" "origin/$REMOTE_BRANCH" >/dev/null

  log "Hard reset working tree → origin/$REMOTE_BRANCH"
  git reset --hard "origin/$REMOTE_BRANCH" >/dev/null
}

install_pytorch_compatible() {
  local want_torch="2.4.1+cu121"
  local want_audio="2.4.1+cu121"
  local want_vision="0.19.1+cu121"
  local cur_torch cur_audio cur_vision


  cur_torch=$(python - <<'PY' 2>/dev/null || true
import torch; print(torch.__version__)
PY
)

  cur_audio=$(python - <<'PY' 2>/dev/null || true
import torchaudio; print(torchaudio.__version__)
PY
)

  cur_vision=$(python - <<'PY' 2>/dev/null || true
import torchvision; print(torchvision.__version__)
PY
)


if [[ "$cur_torch" == "$want_torch" && "$cur_audio" == "$want_audio" && "$cur_vision" == "$want_vision" ]]; then
  log "Torch stack OK (${cur_torch}/${cur_audio}/${cur_vision}) — skip install"
  return 0
fi


  log "Torch stack mismatch → enforcing ${want_torch}/${want_audio}/${want_vision}"
  pip uninstall -y torch torchaudio torchvision >/dev/null 2>&1 || true
  pip install  --index-url https://download.pytorch.org/whl/cu121 \
    "torch==${want_torch}" "torchaudio==${want_audio}" "torchvision==${want_vision}"
}


install_nemo_sortformer() {
  # Installeer NeMo alleen als diarization=1 en backend=sortformer
  if [[ "${DIARIZATION:-0}" != "1" ]]; then
    log "DIARIZATION=0 → NeMo skip"
    return 0
  fi

  # Als jij later een env var wilt toevoegen voor backend: DIARIZATION_BACKEND
  if [[ "${DIARIZATION_BACKEND:-sortformer}" != "sortformer" ]]; then
    log "DIARIZATION_BACKEND!=sortformer → NeMo skip"
    return 0
  fi

  # Idempotent: als import werkt, niets doen
  python - <<'PY' >/dev/null 2>&1 && { log "NeMo (SortFormer) al aanwezig"; return 0; }
from nemo.collections.asr.models import SortformerEncLabelModel
PY

  log "Install NeMo toolkit (ASR) voor SortFormer..."
  pip install "nemo_toolkit[asr]@git+https://github.com/NVIDIA/NeMo.git@v2.6.1"

  # Verify
  python - <<'PY' || die "NeMo install faalde (SortFormer import lukt niet)"
from nemo.collections.asr.models import SortformerEncLabelModel
print("NeMo SortFormer OK")
PY
}

install_ollama() {
  # Ollama = de on-prem LLM-runtime voor gehoorverslag-sectieclassificatie.
  # OpenAI-compatibele API op localhost:11434 -- zelfde interface als lokaal
  # getest tijdens ontwikkeling, dus geen gedragsverschil dev vs. runpod.
  if [[ "${LLM_ENABLED}" != "1" ]]; then
    log "LLM_ENABLED=0 → Ollama-install skip"
    return 0
  fi
  # BEWUST geen "al geïnstalleerd → skip": het installer-script is zelf al
  # idempotent (veilig opnieuw te draaien, dat is ook hoe Ollama zelf updaten
  # wil dat je het doet) en moet hier altijd draaien zodat een eerdere
  # installatie zonder lspci (dus zonder GPU-detectie, CPU-only) alsnog de
  # ontbrekende CUDA-ondersteuning krijgt zodra pciutils/lspci beschikbaar is.
  log "Install/update Ollama (detecteert de NVIDIA-GPU via lspci voor CUDA-versnelling)..."
  curl -fsSL https://ollama.com/install.sh | sh
}

# --- venv + pip ---
setup_venv_pip() {
  cd "$APP_DIR"

  if [[ ! -d "$VENV_DIR" ]]; then
    log "Maak venv: $VENV_DIR"
    python3.11 -m venv "$VENV_DIR"
  fi

  # activate venv
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  log "Venv actief: $VIRTUAL_ENV"

  log "Python: $(python -V)"
  log "Python path: $(which python)"

  PYV=$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
  [[ "$PYV" == "3.11" ]] || die "Verkeerde Python in venv: $PYV (verwacht 3.11)"

  log "Upgrade pip tooling..."
  pip install -U pip setuptools wheel
  
  install_pytorch_compatible
  
  # HuggingFace fast transfer hardening
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
  pip install -U huggingface_hub

  log "Install project + deps (editable) via pyproject.toml..."
  pip install -e .
  
  install_nemo_sortformer
  
  log "Sanity import checks..."
  python -c "import torch; print('torch', torch.__version__)" || die "torch import faalde"
  python -c "import torchaudio; print('torchaudio', torchaudio.__version__)" || die "torchaudio import faalde"
  python -c "import faster_whisper; print('faster_whisper OK')" || die "faster-whisper import faalde"
  python -c "import onnxruntime; print('onnxruntime OK')" || die "onnxruntime import faalde"

  if [[ "${DIARIZATION}" == "1" ]]; then
    python - <<'PY' || die "pyannote.audio import faalde (zie traceback hierboven)"
import traceback
try:
    import pyannote.audio
    print("pyannote.audio OK")
except Exception as e:
    print("pyannote.audio FAIL:", repr(e))
    traceback.print_exc()
    raise
PY
  else
    log "DIARIZATION=0 → pyannote check skip"
  fi
}



# --- run ---
ensure_ollama_running() {
  [[ "${LLM_ENABLED}" == "1" ]] || return 0
  if curl -fsS "http://localhost:11434/api/version" >/dev/null 2>&1; then
    log "Ollama draait al"
    return 0
  fi
  command -v ollama >/dev/null 2>&1 || die "LLM_ENABLED=1 maar 'ollama' niet gevonden. Run eerst: bash scripts/init.sh --setup"
  mkdir -p "$OLLAMA_MODELS"
  export OLLAMA_MODELS
  log "Start Ollama-service (achtergrond, modellen in $OLLAMA_MODELS, logt naar $WORKSPACE/ollama.log)..."
  nohup ollama serve > "$WORKSPACE/ollama.log" 2>&1 &
  disown
  for _ in $(seq 1 30); do
    curl -fsS "http://localhost:11434/api/version" >/dev/null 2>&1 && { log "Ollama draait"; return 0; }
    sleep 1
  done
  die "Ollama startte niet binnen 30s -- zie $WORKSPACE/ollama.log"
}

pull_llm_model() {
  [[ "${LLM_ENABLED}" == "1" ]] || return 0
  # Idempotent: 'ollama pull' zelf is al een no-op als het model al aanwezig is.
  log "Zorg dat LLM-model aanwezig is: $LLM_MODEL (eerste keer kan even duren)..."
  ollama pull "$LLM_MODEL"
}

startlive() {
  # IMPORTANT: --start should NOT do setup. It assumes repo+venv are already ready.
  [[ -d "$APP_DIR/.git" ]] || die "Repo niet gevonden in $APP_DIR. Run eerst: bash scripts/init.sh --setup"
  [[ -d "$VENV_DIR" ]] || die "Venv niet gevonden in $VENV_DIR. Run eerst: bash scripts/init.sh --setup"
  DIAR_ARGS=()
  if [[ "$DIARIZATION" == "1" ]]; then
    DIAR_ARGS+=(--diarization --diarization-backend "$DIARIZATION_BACKEND")

  fi

  LLM_ARGS=()
  if [[ "$LLM_ENABLED" == "1" ]]; then
    ensure_ollama_running
    pull_llm_model
    LLM_ARGS+=(--llm-backend-url "$LLM_BACKEND_URL" --llm-model "$LLM_MODEL")
  else
    log "LLM_ENABLED=0 → gehoorverslag draait zonder sectieclassificatie (platte fallback)"
  fi

  cd "$APP_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  # HuggingFace: voorkom impliciete hf_transfer dependency op RunPod images
  export HF_HUB_ENABLE_HF_TRANSFER=0

  log "Start TriviasServer: host=$HOST port=$PORT model=$MODEL lang=$LANGUAGE llm_enabled=$LLM_ENABLED"
  exec python -m whisperlivekit.TriviasServer \
    --host "$HOST" --port "$PORT" \
    --model "$MODEL" --language "$LANGUAGE" \
    --frame-threshold "$FRAME_THRESHOLD" \
    --audio-min-len "$AUDIO_MIN_LEN" \
    --audio-max-len "$AUDIO_MAX_LEN" \
    --beams "$BEAMS" \
    --pcm-input \
    "${DIAR_ARGS[@]}" \
    "${LLM_ARGS[@]}"
}

gpustat() {
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
}

# --- orchestrators ---
do_setup() {
  redirect_home_cache_to_workspace
  git_identity
  install_deps
  setup_repo

  # setup_repo() kan dit scriptbestand zelf net hebben bijgewerkt (git reset
  # --hard). Bash heeft de REST van dit script echter al in het geheugen
  # geladen op het moment dat het startte -- zonder re-exec zou alles NA dit
  # punt (setup_venv_pip, install_ollama, en straks startlive) nog de OUDE
  # versie draaien, ook al staat de nieuwe allang op schijf. Dit heeft
  # concreet een keer een hele --setup-start-run laten doorlopen zonder de
  # net toegevoegde Ollama/LLM-stappen. _INIT_REEXECED voorkomt een lus.
  if [[ "${_INIT_REEXECED:-0}" != "1" ]]; then
    log "scripts/init.sh is bijgewerkt door setup_repo() -- herstart mezelf met de nieuwste versie..."
    export _INIT_REEXECED=1
    exec bash "$APP_DIR/scripts/init.sh" "$MODE"
  fi

  setup_venv_pip
  install_ollama
}

do_update() {
  git_identity
  setup_repo
}

# --- CLI ---
MODE="${1:-}"

case "$MODE" in
  --deps)
    git_identity
    install_deps
    ;;
  --update)
    do_update
    ;;
  --venv)
    git_identity
    setup_venv_pip
    ;;
  --setup)
    do_setup
    ;;
  --setup-start)
    do_setup
    startlive
    ;;
  --start)
    startlive
    ;;
  *)
    echo "Usage:"
    echo "  bash scripts/init.sh --setup         # deps + git + venv + pip"
    echo "  bash scripts/init.sh --setup-start   # setup + start server"
    echo "  bash scripts/init.sh --start         # start server only (no setup)"
    echo "  bash scripts/init.sh --update        # git update only"
    echo "  bash scripts/init.sh --deps          # apt deps only"
    echo "  bash scripts/init.sh --venv          # venv + pip only"
    exit 2
    ;;
esac

if [[ "$IS_SOURCED" -eq 1 ]]; then
  log ""
  log "Functies geladen in huidige sessie:"
  log "  ▶ startlive   → start server"
  log "  ▶ gpustat     → GPU status"
  log ""
  log "✅ Setup voltooid. Actieve Python: $(command -v python)"
  log "Tip: startlive"
fi
