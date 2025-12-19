#!/bin/sh
set -e

# --- Force Bash when executed via Dash ---
[ -z "$BASH_VERSION" ] && exec bash "$0" "$@"

# === Git identity instellen zodat git pull nooit vraagt ===
git config --global user.email "topcug1975@gmail.com"
git config --global user.name "Gokhan Topcu"

# === Config ===
REPO_URL="https://github.com/GT-GITH/WhisperLiveKit-Trivias.git"
WORKSPACE="/workspace"
APP_DIR="$WORKSPACE/WhisperLiveKit-Trivias"
VENV_DIR="$APP_DIR/.venv"
POETRY_BIN="/root/.local/bin/poetry"

echo "------------------------------------------"
echo " 🧠 WhisperLiveKit-Trivias setup starten..."
echo "------------------------------------------"

# === 1) Cache directories ===
export CACHE_BASE="$WORKSPACE/cache"
export TMPDIR="$CACHE_BASE/tmp"
export PIP_CACHE_DIR="$CACHE_BASE/pip"
export POETRY_CACHE_DIR="$CACHE_BASE/poetry"
mkdir -p "$CACHE_BASE" "$TMPDIR" "$PIP_CACHE_DIR" "$POETRY_CACHE_DIR"

# === 2) Basis packages ===
apt update -y >/dev/null
apt install -y git curl ffmpeg python3-venv >/dev/null

# === 3) Poetry installeren ===
export PATH="/root/.local/bin:$PATH"
if ! command -v poetry >/dev/null 2>&1; then
  echo "📦 Installeer Poetry..."
  curl -sSL https://install.python-poetry.org | python3 - >/dev/null
  export PATH="/root/.local/bin:$PATH"
fi

# === 4) Repo klonen of updaten ===
if [ ! -d "$APP_DIR/.git" ]; then
  echo "⬇️  Clone repo..."
  git clone "$REPO_URL" "$APP_DIR"
else
  echo "🔄  Update bestaande repo..."
  cd "$APP_DIR"
  git fetch --all >/dev/null
  #git reset --hard origin/main >/dev/null
  git checkout -b stable-segment-batch-v1 origin/feat-batch-tuning-copt
fi

# === 5) Virtuele omgeving ===
if [ -z "$VIRTUAL_ENV" ]; then
  if [ ! -d "$VENV_DIR" ]; then
    echo "[init] Maak nieuwe venv aan: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  . "$VENV_DIR/bin/activate"
  echo "[init] venv geactiveerd: $VENV_DIR"
else
  echo "[init] venv al actief: $VIRTUAL_ENV"
fi

# === 6) Dependencies via Poetry ===
cd "$APP_DIR"
poetry config virtualenvs.in-project true
echo "[init] Installeer dependencies..."
poetry install --no-interaction --no-root

# === 7) Hugging Face hf_transfer fix ===
echo "[init] Controleer Hugging Face-transferondersteuning..."
if [ "${HF_HUB_ENABLE_HF_TRANSFER:-1}" = "1" ]; then
  "$VENV_DIR/bin/pip" install -q hf_transfer || export HF_HUB_ENABLE_HF_TRANSFER=0
fi

# === 8) Helperfuncties ===
startlive() {
  cd "$APP_DIR" || return
  export PATH="/root/.local/bin:$PATH"
  export PYTHONPATH="$APP_DIR"

  # Detecteer RunPod-host IP voor externe toegang
  HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
  echo "🌐 RunPod-host gedetecteerd: $HOST_IP"

  # Start server bindend aan 0.0.0.0 zodat hij buiten RunPod bereikbaar is
  "$POETRY_BIN" run python -m whisperlivekit.basic_server --host 0.0.0.0 --port 8000
}

gpuprep() {
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
}

echo ""
echo "[init] Functies geladen in huidige sessie:"
echo "  ▶ startlive   → Start de live server (extern bereikbaar)"
echo "  ▶ gpuprep     → Bekijk GPU-status"
echo ""
echo "✅ Setup voltooid en venv actief!"
echo "Actieve Python: $(which python)"
echo "Gebruik nu: startlive"
echo "------------------------------------------"
