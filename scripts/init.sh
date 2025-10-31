#!/bin/bash
set -e

REPO_URL="https://github.com/GT-GITH/WhisperLiveKit-Trivias.git"
WORKSPACE="/workspace"
APP_DIR="$WORKSPACE/WhisperLiveKit-Trivias"

echo "------------------------------------------"
echo " 🧠 WhisperLiveKit-Trivias setup starten..."
echo "------------------------------------------"

# 🧩 Cache directories
export CACHE_BASE="$WORKSPACE/cache"
export TMPDIR="$CACHE_BASE/tmp"
export PIP_CACHE_DIR="$CACHE_BASE/pip"
export POETRY_CACHE_DIR="$CACHE_BASE/poetry"
mkdir -p "$CACHE_BASE" "$TMPDIR" "$PIP_CACHE_DIR" "$POETRY_CACHE_DIR"

apt update -y && apt install -y git curl ffmpeg python3-venv

# 📦 Poetry installatie
if ! command -v poetry &> /dev/null; then
  echo "📦 Installeer Poetry..."
  curl -sSL https://install.python-poetry.org | python3 -
  export PATH="/root/.local/bin:$PATH"
fi

# 📁 Repo klonen of updaten
if [ ! -d "$APP_DIR/.git" ]; then
  echo "⬇️  Clone repo vanuit GitHub..."
  git clone "$REPO_URL" "$APP_DIR"
else
  echo "🔄  Update bestaande repo..."
  cd "$APP_DIR"
  git fetch --all
  git reset --hard origin/main
fi

# ⚙️ Poetry configuratie
cd "$APP_DIR"
poetry config virtualenvs.in-project true
poetry config cache-dir "$POETRY_CACHE_DIR"

# 🐍 Lokale venv
if [ ! -d "$APP_DIR/.venv" ]; then
  echo "🐍 Maak lokale venv aan..."
  poetry env use python3
  PIP_CACHE_DIR="$PIP_CACHE_DIR" TMPDIR="$TMPDIR" poetry install --no-interaction --no-root
else
  echo "🚀 Lokale venv al aanwezig – overslaan"
fi

# 🧠 Controle
echo "Actieve Poetry venv: $(poetry env info --path)"

# 🔗 Alias (voor directe start)
ALIASES_FILE="$HOME/.bash_aliases"
if ! grep -q "startlive" "$ALIASES_FILE" 2>/dev/null; then
  echo "alias startlive='cd /workspace/WhisperLiveKit-Trivias && /workspace/WhisperLiveKit-Trivias/.venv/bin/python whisperlivekit/basic_server.py'" >> "$ALIASES_FILE"
  echo "✅ Alias toegevoegd: startlive"
  echo "source $ALIASES_FILE" >> ~/.bashrc
  source ~/.bashrc
fi

echo ""
echo "✅ Setup voltooid!"
echo "Gebruik nu:  startlive"
echo "------------------------------------------"
