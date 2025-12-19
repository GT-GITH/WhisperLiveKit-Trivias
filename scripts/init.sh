#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# RunPod init script (idempotent)
# - Repo: https://github.com/GT-GITH/WhisperLiveKit-Trivias.git
# - Target remote branch: origin/feat-batch-tuning-copt
# - Local working branch: stable-segment-batch-v1 (points to remote branch)
#
# Usage:
#   source scripts/init.sh                # load functions + run default setup
#   bash scripts/init.sh --all            # run full setup (no functions kept)
#   bash scripts/init.sh --start          # setup + start server
#   bash scripts/init.sh --update         # git update only
#   bash scripts/init.sh --deps           # apt deps only
#   bash scripts/init.sh --venv           # venv + poetry install only
# ------------------------------------------------------------

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

REMOTE_BRANCH="${REMOTE_BRANCH:-feat-batch-tuning-copt}"
LOCAL_BRANCH="${LOCAL_BRANCH:-stable-segment-batch-v1}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-large-v3}"
LANGUAGE="${LANGUAGE:-nl}"
FRAME_THRESHOLD="${FRAME_THRESHOLD:-25}"
AUDIO_MIN_LEN="${AUDIO_MIN_LEN:-0.0}"
AUDIO_MAX_LEN="${AUDIO_MAX_LEN:-30.0}"
BEAMS="${BEAMS:-1}"

export PATH="$PATH:/root/.local/bin"

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
}

ensure_poetry() {
  if command -v poetry >/dev/null 2>&1; then
    return 0
  fi
  log "Poetry niet gevonden → installeren..."
  # official installer puts poetry under /root/.local/bin
  curl -sSL https://install.python-poetry.org | python3 - >/dev/null
  command -v poetry >/dev/null 2>&1 || die "Poetry installatie faalde."
}

# --- repo ---
setup_repo() {
  mkdir -p "$WORKSPACE"

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

# --- venv + poetry ---
setup_venv_poetry() {
  cd "$APP_DIR"

  if [[ ! -d "$VENV_DIR" ]]; then
    log "Maak venv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi

  # activate venv
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  log "Venv actief: $VIRTUAL_ENV"

  ensure_poetry

  # Force poetry to use this in-project venv
  poetry config virtualenvs.in-project true >/dev/null
  poetry env use "$VENV_DIR/bin/python" >/dev/null || true

  log "Poetry install (no-interaction)..."
  poetry install --no-interaction >/dev/null
}

# --- run ---
startlive() {
  cd "$APP_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  log "Start TriviasServer: host=$HOST port=$PORT model=$MODEL lang=$LANGUAGE"
  exec poetry run python -m whisperlivekit.TriviasServer \
    --host "$HOST" --port "$PORT" \
    --model "$MODEL" --language "$LANGUAGE" \
    --frame-threshold "$FRAME_THRESHOLD" \
    --audio-min-len "$AUDIO_MIN_LEN" \
    --audio-max-len "$AUDIO_MAX_LEN" \
    --beams "$BEAMS"
}

gpustat() {
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
}

# --- orchestrators ---
do_all() {
  git_identity
  install_deps
  setup_repo
  setup_venv_poetry
}

do_update() {
  git_identity
  setup_repo
}

# --- CLI ---
MODE="${1:-}"

case "$MODE" in
  --deps)   git_identity; install_deps ;;
  --update) do_update ;;
  --venv)   git_identity; ensure_poetry; setup_venv_poetry ;;
  --start)  do_all; startlive ;;
  --all|"") do_all ;;
  *)
    echo "Usage: source scripts/init.sh  |  bash scripts/init.sh [--all|--start|--update|--deps|--venv]"
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
