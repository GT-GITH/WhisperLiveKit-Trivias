#!/usr/bin/env bash
set -e

# === 📁 Automatisch repo-root bepalen ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# === ⚙️ Configuratie ===
MODE=${1:-dev}  # gebruik "setup" voor eerste installatie
REPO_URL="https://github.com/GT-GITH/WhisperLiveKit-Trivias.git"
VENV_DIR="/workspace/venv"
CACHE_ROOT="/workspace/cache"
LOG_DIR="/workspace/logs"

echo "[init] Start in modus: $MODE"
mkdir -p "$CACHE_ROOT" "$LOG_DIR"

# === 🔐 Git SSH setup ===
mkdir -p ~/.ssh
for f in id_ed25519 id_ed25519.pub known_hosts; do
  [ -f "/workspace/.ssh/$f" ] && cp -f "/workspace/.ssh/$f" ~/.ssh/ || true
done
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519 2>/dev/null || true
chmod 644 ~/.ssh/id_ed25519.pub 2>/dev/null || true
chmod 644 ~/.ssh/known_hosts 2>/dev/null || true

# Start SSH-agent
if ! pgrep -u "$USER" ssh-agent >/dev/null 2>&1; then
  eval "$(ssh-agent -s)"
fi
[ -f ~/.ssh/id_ed25519 ] && ssh-add ~/.ssh/id_ed25519 >/dev/null 2>&1 || true

# === 🧾 Git identiteit ===
if ! git config --global user.email >/dev/null 2>&1; then
  git config --global user.name "Gokhan Topcu"
  git config --global user.email "topcug1975@gmail.com"
  echo "[init] Git identiteit ingesteld."
fi

# === 📦 Poetry & cache setup ===
export POETRY_HOME="/workspace/.poetry"
export POETRY_CACHE_DIR="$CACHE_ROOT/poetry"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export TMPDIR="$CACHE_ROOT/tmp"
mkdir -p $POETRY_HOME $POETRY_CACHE_DIR $PIP_CACHE_DIR $TMPDIR
export PATH="$POETRY_HOME/bin:$PATH"

echo "[init] Cache directories ingesteld in /workspace/cache"
echo "[init] Poetry-home ingesteld op $POETRY_HOME"

# === 🧰 Poetry installeren (indien nodig) ===
if ! command -v poetry &>/dev/null; then
  echo "[init] Poetry niet gevonden — installeren..."
  curl -sSL https://install.python-poetry.org | POETRY_HOME=$POETRY_HOME python3 -
else
  echo "[init] Poetry reeds aanwezig."
fi

# === 🐍 Virtuele omgeving (optioneel) ===
if [ -d "$VENV_DIR" ]; then
  source "$VENV_DIR/bin/activate"
  echo "[init] Virtuele omgeving geactiveerd: $VENV_DIR"
else
  echo "[init] Geen venv gevonden — nieuwe aanmaken"
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
fi

# === 🪄 Aliassen ===
alias gpuprep='pip install --upgrade pip setuptools wheel'
alias wsstart="poetry run python whisperlivekit/basic_server.py --host 0.0.0.0 --port 8000"
alias startlive="echo '🚀 Starting WhisperLiveKit-Trivias...' && cd $REPO_ROOT && wsstart"
echo "[init] Aliassen geladen: gpuprep, wsstart, startlive"

# === 🧩 Setup (alleen bij eerste keer) ===
if [ "$MODE" = "setup" ]; then
  echo "[setup] Setup-modus actief — installaties starten..."
  apt update && apt install -y ffmpeg
  poetry install
  echo "[setup] FFmpeg + dependencies via Poetry geïnstalleerd ✅"
fi

echo "[init] Klaar ✅"
