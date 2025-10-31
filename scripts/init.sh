#!/bin/bash
set -e

REPO_URL="https://github.com/GT-GITH/WhisperLiveKit-Trivias.git"
WORKSPACE="/workspace"
APP_DIR="$WORKSPACE/WhisperLiveKit-Trivias"
VENV_DIR="$APP_DIR/.venv"

echo "------------------------------------------"
echo " 🧠 WhisperLiveKit-Trivias setup starten..."
echo "------------------------------------------"

# === 1) Cache directories instellen ===
export CACHE_BASE="$WORKSPACE/cache"
export TMPDIR="$CACHE_BASE/tmp"
export PIP_CACHE_DIR="$CACHE_BASE/pip"
export POETRY_CACHE_DIR="$CACHE_BASE/poetry"
mkdir -p "$CACHE_BASE" "$TMPDIR" "$PIP_CACHE_DIR" "$POETRY_CACHE_DIR"
echo "[init] Caches ingesteld onder $CACHE_BASE"

# === 2) Basis packages ===
apt update -y && apt install -y git curl ffmpeg python3-venv

# === 3) Virtuele omgeving (aanmaken of activeren) ===
if [ -d "$VENV_DIR" ]; then
  if [ -z "$VIRTUAL_ENV" ]; then
    source "$VENV_DIR/bin/activate"
    echo "[init] venv geactiveerd: $VENV_DIR"
  else
    echo "[init] venv al actief: $VIRTUAL_ENV"
  fi
else
  echo "[init] Geen venv gevonden — nieuwe aanmaken in $VENV_DIR"
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  echo "[init] venv aangemaakt en geactiveerd: $VENV_DIR"
fi

# === 4) Poetry installeren (binnen deze venv) ===
if ! command -v poetry &> /dev/null; then
  echo "[init] Installeer Poetry in venv..."
  curl -sSL https://install.python-poetry.org | python3 -
  export PATH="$VENV_DIR/bin:$PATH:/root/.local/bin"
fi

# === 5) Repo klonen of updaten ===
if [ ! -d "$APP_DIR/.git" ]; then
  echo "[init] Clone repo..."
  git clone "$REPO_URL" "$APP_DIR"
else
  echo "[init] Update bestaande repo..."
  cd "$APP_DIR"
  git fetch --all
  git reset --hard origin/main
fi

# === 6) Poetry configureren en dependencies installeren ===
cd "$APP_DIR"
poetry config virtualenvs.in-project true
poetry config cache-dir "$POETRY_CACHE_DIR"

echo "[init] Installeer dependencies..."
poetry install --no-interaction --no-root

# === 7) Aliassen ===
ALIASES_FILE="$HOME/.bash_aliases"
if ! grep -q "startlive" "$ALIASES_FILE" 2>/dev/null; then
  echo "alias startlive='cd /workspace/WhisperLiveKit-Trivias && poetry run python whisperlivekit/basic_server.py'" >> "$ALIASES_FILE"
  echo "source $ALIASES_FILE" >> ~/.bashrc
  source ~/.bashrc
  echo "[init] Alias toegevoegd: startlive"
fi

echo ""
echo "✅ Setup voltooid en venv actief!"
echo "Gebruik nu: startlive"
echo "------------------------------------------"
