#!/bin/bash
set -e

REPO_URL="https://github.com/GT-GITH/WhisperLiveKit-Trivias.git"
WORKSPACE="/workspace"
APP_DIR="$WORKSPACE/WhisperLiveKit-Trivias"
VENV_PATH="$WORKSPACE/venv"

echo "------------------------------------------"
echo " 🧠 WhisperLiveKit-Trivias setup starten..."
echo "------------------------------------------"

# Zorg dat lokale bin-directory actief is
export PATH="/root/.local/bin:$PATH"

# Update systeem en installeer basispackages
apt update -y && apt install -y git curl ffmpeg python3-venv

# Poetry installeren als het nog niet bestaat
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

# Ga naar projectfolder
cd "$APP_DIR"

# Eventueel oude cache opruimen (optioneel)
rm -rf "$APP_DIR/__pycache__" || true
rm -rf "$APP_DIR/.pytest_cache" || true

# Installeer dependencies zonder root package
echo "⚙️  Poetry dependencies installeren..."
poetry install --no-root

# Symlink maken voor eenvoud (eenmalig)
if [ ! -f /usr/local/bin/startlive ]; then
  echo "🪄  Maak snelstartcommando 'startlive'..."
  echo "cd $APP_DIR && poetry run python whisperlivekit/basic_server.py" > /usr/local/bin/startlive
  chmod +x /usr/local/bin/startlive
fi

echo ""
echo "✅ Setup voltooid!"
echo "Gebruik nu:  startlive"
echo "of handmatig: poetry run python whisperlivekit/basic_server.py"
echo "------------------------------------------"
