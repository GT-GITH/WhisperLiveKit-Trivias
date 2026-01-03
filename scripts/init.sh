#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# RunPod init script (idempotent)
#
# Defaults:
#   REMOTE_BRANCH=main
#   LOCAL_BRANCH=main
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


# --- ensure bash ---
[[ -n "${BASH_VERSION:-}" ]] || die "Dit script vereist bash. Run: bash $0 ..."

# --- git identity (to prevent prompts) ---
git_identity() {
  git config --global user.email "topcug1975@gmail.com" >/dev/null 2>&1 || true
  git config --global user.name "Gokhan Topcu" >/dev/null 2>&1 || true
}

# --- deps ---
install_deps() {
  log "Install OS deps (git, curl, ffmpeg, python3-venv)..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y >/dev/null
  apt-get install -y git curl ffmpeg python3-venv >/dev/null
  apt-get install -y python3.11 python3.11-venv
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
  log "Forceer compatibele PyTorch/torchaudio versies voor pyannote..."
  # Zorg dat we niet blijven hangen op die 2.9.x builds
  pip uninstall -y torch torchaudio torchvision >/dev/null 2>&1 || true

  # Installeer stabiele combo (cu121 is doorgaans ok op CUDA 12.x machines)
  pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1+cu121 torchaudio==2.4.1+cu121
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
  
  log "Install project + deps (editable) via pyproject.toml..."
  pip install -e .

  log "Sanity import checks..."
  python -c "import torch; print('torch', torch.__version__)" || die "torch import faalde"
  python -c "import torchaudio; print('torchaudio', torchaudio.__version__)" || die "torchaudio import faalde"
  python -c "import faster_whisper; print('faster_whisper OK')" || die "faster-whisper import faalde"
  python -c "import onnxruntime; print('onnxruntime OK')" || die "onnxruntime import faalde"

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
}


# --- run ---
startlive() {
  # IMPORTANT: --start should NOT do setup. It assumes repo+venv are already ready.
  [[ -d "$APP_DIR/.git" ]] || die "Repo niet gevonden in $APP_DIR. Run eerst: bash scripts/init.sh --setup"
  [[ -d "$VENV_DIR" ]] || die "Venv niet gevonden in $VENV_DIR. Run eerst: bash scripts/init.sh --setup"
  DIAR_ARGS=()
  if [[ "$DIARIZATION" == "1" ]]; then
    DIAR_ARGS+=(--diarization --diarization-backend sortformer)
  fi


  cd "$APP_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  log "Start TriviasServer: host=$HOST port=$PORT model=$MODEL lang=$LANGUAGE"
  exec python -m whisperlivekit.TriviasServer \
    --host "$HOST" --port "$PORT" \
    --model "$MODEL" --language "$LANGUAGE" \
    --frame-threshold "$FRAME_THRESHOLD" \
    --audio-min-len "$AUDIO_MIN_LEN" \
    --audio-max-len "$AUDIO_MAX_LEN" \
    --beams "$BEAMS" \
    "${DIAR_ARGS[@]}"
}

gpustat() {
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
}

# --- orchestrators ---
do_setup() {
  git_identity
  install_deps
  setup_repo
  setup_venv_pip
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
