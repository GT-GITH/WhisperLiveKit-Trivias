#!/bin/bash
set -e

REPO_URL="https://github.com/GT-GITH/WhisperLiveKit-Trivias.git"
WORKSPACE="/workspace"
APP_DIR="$WORKSPACE/WhisperLiveKit-Trivias"
VENV_DIR="$APP_DIR/.venv"

echo "------------------------------------------"
echo " 🧠 WhisperLiveKit-Trivias setup starten..."
echo "------------------------------------------"

# === 0) Zelfherstart in venv ===
if [ -z "$VIRTUAL_ENV" ]; then
  if [ -d "$VENV_DIR" ]; then
    echo "[init] Geen actieve venv gevonden — script herstart binnen venv..."
    exec bash --rcfile <(echo "source $VENV_DIR/bin/activate; bash /workspace/WhisperLiveKit-Trivias/scripts/init.sh")
  fi
fi

# === 1) Cache directories ===
export CACHE_BASE="$WORKSPACE/cache"
export TMPDIR="$CACHE_BASE/tmp"
export PIP_CACHE_DIR="$CACHE_BASE/pip"
export POETRY_CACHE_DIR="$CACHE_BASE/poetry"
mkdir -p "$CACHE_BASE" "$TMPDIR" "$PIP_CACHE_DIR" "$POETRY_CACHE_DIR"
echo "[init] Caches ingesteld onder $CACHE_BASE"

# === 2) Basis packages ===
apt update -y && apt install -y git curl ffmpeg python3-venv

# === 3) Virtuele omgeving aanmaken of activeren ===
if [ ! -d "$VENV_DIR" ]; then
  echo "[init] Geen venv gevonden — nieuwe aanmaken in $VENV_DIR"
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  echo "[init] venv aangemaakt en geactiveerd: $VENV_DIR"
else
  if [ -z "$VIRTUAL_ENV" ]; then
    source "$VENV_DIR/bin/activate"
    echo "[init] venv geactiveerd: $VENV_DIR"
  else
    echo "[init] venv al actief: $VIRTUAL_ENV"
  fi
fi

# === 4) Poetry installeren binnen venv ===
if ! command -v poetry &> /dev/null; then
  echo "[init] Installeer Poetry..."
  curl -sSL https://install.python-poetry.org | python3 -
  export PATH="$VENV_DIR/bin:$PATH:/root/.local/bin"
fi

# === 5) Repo klonen of updaten ===
if [ ! -d "$APP_DIR/.git" ]; then
  echo "[init] Clone repo vanuit GitHub..."
  git clone "$REPO_URL" "$APP_DIR"
else
  echo "[init] Update bestaande repo..."
  cd "$APP_DIR"
  git fetch --all
  git reset --hard origin/main
fi

# === 6) Dependencies installeren ===
cd "$APP_DIR"
poetry config virtualenvs.in-project true
poetry config cache-dir "$POETRY_CACHE_DIR"
poetry install --no-interaction --no-root

# === 7) Aliassen ===
ALIASES_FILE="$HOME/.bash_aliases"
if ! grep -q "startlive" "$ALIASES_FILE" 2>/dev/null; then
  echo "alias startlive='cd /workspace/WhisperLiveKit-Trivias && $VENV_DIR/bin/poetry run python whisperlivekit/basic_server.py'" >> "$ALIASES_FILE"
  echo "alias gpuprep='nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader'" >> "$ALIASES_FILE"
fi

# === 8) Forceer alias-load ook in deze sessie ===
if [ -f "$ALIASES_FILE" ]; then
  source "$ALIASES_FILE"
fi
if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi
echo "[init] Aliassen geladen: startlive, gpuprep"

echo ""
echo "✅ Setup voltooid en venv actief!"
echo "Actieve Python: $(which python)"
echo "Gebruik nu: startlive"
echo "------------------------------------------"
