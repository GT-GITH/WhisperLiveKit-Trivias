#!/bin/bash
set -e

REPO_URL="https://github.com/GT-GITH/WhisperLiveKit-Trivias.git"
WORKSPACE="/workspace"
APP_DIR="$WORKSPACE/WhisperLiveKit-Trivias"
VENV_DIR="$WORKSPACE/venv"

echo "------------------------------------------"
echo " 🧠 WhisperLiveKit-Trivias setup starten..."
echo "------------------------------------------"

# 🧩 Cache directories (voorkomt 'no space left')
export CACHE_BASE="$WORKSPACE/cache"
export TMPDIR="$CACHE_BASE/tmp"
export PIP_CACHE_DIR="$CACHE_BASE/pip"
export POETRY_CACHE_DIR="$CACHE_BASE/poetry"
mkdir -p "$CACHE_BASE" "$TMPDIR" "$PIP_CACHE_DIR" "$POETRY_CACHE_DIR"

# Basis packages
apt update -y && apt install -y git curl ffmpeg python3-venv

# Poetry installatie
if ! command -v poetry &> /dev/null; then
  echo "📦 Installeer Poetry..."
  curl -sSL https://install.python-poetry.org | python3 -
  export PATH="/root/.local/bin:$PATH"
fi

# Repo klonen of updaten
if [ ! -d "$APP_DIR/.git" ]; then
  echo "⬇️  Clone repo vanuit GitHub..."
  git clone "$REPO_URL" "$APP_DIR"
else
  echo "🔄  Update bestaande repo..."
  cd "$APP_DIR"
  git fetch --all
  git reset --hard origin/main
fi

# Virtuele omgeving
if [ ! -d "$VENV_DIR" ]; then
  echo "🐍 Maak virtuele omgeving..."
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Poetry setup
cd "$APP_DIR"
if [ ! -d "$APP_DIR/.venv" ]; then
  echo "⚙️  Installeer dependencies..."
  poetry config virtualenvs.in-project true
  poetry install --no-interaction --no-root
else
  echo "🚀 Dependencies al aanwezig – overslaan"
fi

# Permanente aliassen
ALIASES_FILE="$HOME/.bash_aliases"
if ! grep -q "startlive" "$ALIASES_FILE" 2>/dev/null; then
  echo "alias startlive='cd /workspace/WhisperLiveKit-Trivias && poetry run python whisperlivekit/basic_server.py'" >> "$ALIASES_FILE"
  echo "alias gpuprep='nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader'" >> "$ALIASES_FILE"
  echo "alias wssstart='cd /workspace/WhisperLiveKit-Trivias && poetry run python whisperlivekit/basic_server.py'" >> "$ALIASES_FILE"
  echo "source $ALIASES_FILE" >> ~/.bashrc
  source ~/.bashrc
  echo "✅ Aliassen permanent toegevoegd"
fi

echo ""
echo "✅ Setup voltooid!"
echo "Gebruik nu:  startlive"
echo "------------------------------------------"
