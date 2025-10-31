#!/bin/bash
set -e

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
  git reset --hard origin/main >/dev/null
fi

# === 5) Virtuele omgeving ===
if [ -z "$VIRTUAL_ENV" ]; then
  if [ ! -d "$VENV_DIR" ]; then
    echo "[init] Maak nieuwe venv aan: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
  source "$VENV_DIR/bin/activate"
  echo "[init] venv geactiveerd: $VENV_DIR"
else
  echo "[init] venv al actief: $VIRTUAL_ENV"
fi

# === 6) Dependencies via Poetry ===
cd "$APP_DIR"
poetry config virtualenvs.in-project true
echo "[init] Installeer dependencies..."
poetry install --no-interaction --no-root

# === 7) Helperfuncties ===
startlive() {
  cd "$APP_DIR" || return
  "$POETRY_BIN" run python whisperlivekit/basic_server.py
}

gpuprep() {
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
}

# functies gewoon definiëren, niet exporteren
echo ""
echo "[init] Functies geladen in huidige sessie:"
echo "  ▶ startlive   → Start de live server"
echo "  ▶ gpuprep     → Bekijk GPU-status"
echo ""

# === 8) Samenvatting ===
echo "✅ Setup voltooid en venv actief!"
echo "Actieve Python: $(which python)"
echo "Gebruik nu: startlive"
echo "------------------------------------------"
