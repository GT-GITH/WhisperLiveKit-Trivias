#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# RunPod init script (idempotent)
#
# Defaults:
#   REMOTE_BRANCH=main
#   LOCAL_BRANCH=main
#   LLM_ENABLED=1 LLM_MODEL=llama3.1:8b LLM_BACKEND_URL=http://localhost:11434/v1
#     -> on-prem Ollama, gereserveerd voor toekomstige features (geen actieve
#        aanroeper meer -- /translate gebruikt sinds 2026-08-16 NLLB, zie
#        onder). Altijd lokaal op de pod zelf. LLM_ENABLED=0 om uit te zetten.
#   NLLB_ENABLED=1 NLLB_MODEL=entai2965/nllb-200-distilled-600M-ctranslate2 NLLB_DEVICE=auto
#     -> on-prem NLLB-200 (via ctranslate2) voor het /translate-endpoint. Een
#        chatmodel (Ollama) bleek onbetrouwbaar voor vertaling van complexe
#        brontalen -- NLLB is uitsluitend op vertalen getraind. NLLB_ENABLED=0
#        om uit te zetten (dan draait /translate in fallback, 503).
#   SOMALI_BATCH_MODEL_ENABLED=0 SOMALI_BATCH_MODEL_SRC=microsoft/paza-whisper-large-v3-turbo
#     -> modelroutering per kanaal (fase 1, batch-only, PoC -- zie het voorstel).
#        Default UIT: opt-in voor het foreign_so-kanaal. =1 converteert het
#        bronmodel één keer naar CTranslate2 (SOMALI_BATCH_MODEL_DIR) en geeft
#        dat pad door als --foreign-so-batch-model. Elk ander kanaal blijft
#        ongewijzigd op het server-brede standaardmodel.
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
PCM_INPUT="${PCM_INPUT:-1}"  # altijd PCM voor multi-channel stabiliteit
DIARIZATION_BACKEND="${DIARIZATION_BACKEND:-sortformer}"

# On-prem LLM (Ollama) -- gereserveerd voor toekomstige, nog niet gebouwde
# features (zie llm_backend.py). NOOIT een cloud-endpoint, altijd lokaal op
# de pod zelf (localhost, geen externe port-forward nodig). Default AAN:
# L40S 48GB heeft ~30GB vrij naast Whisper large-v3, ruim genoeg voor een
# 8B-model. LLM_ENABLED=0 schakelt dit volledig uit.
LLM_ENABLED="${LLM_ENABLED:-1}"
LLM_MODEL="${LLM_MODEL:-llama3.1:8b}"
LLM_BACKEND_URL="${LLM_BACKEND_URL:-http://localhost:11434/v1}"
# Ollama's default modellocatie is /root/.ollama/models, op de kleine
# root-schijf van de pod (niet de grote /workspace-volume waar de rest van
# dit project al staat) -- expliciet naar /workspace verplaatst, anders loopt
# een model van een paar GB de root-schijf vol.
OLLAMA_MODELS="${OLLAMA_MODELS:-$WORKSPACE/.ollama-models}"

# On-prem NLLB-200-vertaalmodel (zie nllb_backend.py), gebruikt door het
# /translate-endpoint. Draait via ctranslate2 (geen aparte serverservice
# zoals Ollama nodig -- het model wordt gewoon in het TriviasServer-proces
# geladen). Default AAN, met het kleine gedistilleerde 600M-checkpoint
# (past ruim, ook naast Whisper + eventueel Ollama). NLLB_ENABLED=0
# schakelt dit volledig uit (dan draait /translate in fallback, 503).
# Modelgewichten komen via huggingface_hub in ~/.cache/huggingface terecht --
# dat is via redirect_home_cache_to_workspace() (zie hierboven) al naar
# /workspace verplaatst, dus geen aparte disk-fix nodig zoals bij Ollama.
NLLB_ENABLED="${NLLB_ENABLED:-1}"
NLLB_MODEL="${NLLB_MODEL:-entai2965/nllb-200-distilled-600M-ctranslate2}"
NLLB_DEVICE="${NLLB_DEVICE:-auto}"

# Modelroutering per kanaal (fase 1, batch-only, PoC -- zie het voorstel voor
# modelroutering per kanaal). Default UIT: dit is een opt-in PoC voor het
# foreign_so-kanaal, geen ander kanaal wordt hierdoor geraakt. Zet
# SOMALI_BATCH_MODEL_ENABLED=1 om bij --setup/--setup-start het bronmodel
# (PyTorch/HF-formaat) één keer te converteren naar CTranslate2 (nodig --
# faster-whisper/BatchFasterWhisperASR kan geen ruwe HF-checkpoints laden) en
# het resulterende pad als --foreign-so-batch-model aan de server mee te geven.
SOMALI_BATCH_MODEL_ENABLED="${SOMALI_BATCH_MODEL_ENABLED:-0}"
SOMALI_BATCH_MODEL_SRC="${SOMALI_BATCH_MODEL_SRC:-microsoft/paza-whisper-large-v3-turbo}"
SOMALI_BATCH_MODEL_DIR="${SOMALI_BATCH_MODEL_DIR:-$WORKSPACE/models/paza-whisper-large-v3-turbo-ct2}"


# --- ensure bash ---
[[ -n "${BASH_VERSION:-}" ]] || die "Dit script vereist bash. Run: bash $0 ..."

# --- git identity (to prevent prompts) ---
git_identity() {
  git config --global user.email "topcug1975@gmail.com" >/dev/null 2>&1 || true
  git config --global user.name "Gokhan Topcu" >/dev/null 2>&1 || true
}

# --- ~/.cache van de kleine root-schijf af ---
redirect_home_cache_to_workspace() {
  # ~/.cache (HuggingFace-modellen, pip-cache, torch-hub, etc. -- allemaal
  # dezelfde XDG-conventie) staat standaard op de kleine, ephemere root-
  # overlay van de pod (hier gezien: 5GB, niet de grote /workspace-volume).
  # Die loopt vroeg of laat vol zodra er modellen gedownload worden -- exact
  # wat er gebeurde: root liep 100% vol, waardoor zelfs "apt-get install"
  # geen dpkg-lock-bestand meer kon wegschrijven. Eenmalig verplaatsen naar
  # /workspace en terugsymlinken lost dit voorgoed op, voor elke tool die
  # naar ~/.cache/... schrijft, zonder per-tool env vars te hoeven zetten.
  if [[ -L "$HOME/.cache" ]]; then
    log "~/.cache is al een symlink → skip"
    return 0
  fi
  mkdir -p "$WORKSPACE/.cache"
  if [[ -d "$HOME/.cache" ]]; then
    log "Verplaats bestaande ~/.cache (kan even duren) naar $WORKSPACE/.cache..."
    shopt -s dotglob nullglob
    mv "$HOME"/.cache/* "$WORKSPACE/.cache/" 2>/dev/null || true
    shopt -u dotglob nullglob
    rmdir "$HOME/.cache" 2>/dev/null || true
  fi
  ln -s "$WORKSPACE/.cache" "$HOME/.cache"
  log "~/.cache → $WORKSPACE/.cache (symlink)"
}

# --- deps ---
install_deps() {
  local marker="$WORKSPACE/.os_deps_installed"
  if [[ -f "$marker" ]]; then
    log "OS deps al gedaan → skip"
    return 0
  fi

  log "Install OS deps..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y >/dev/null
  apt-get install -y git curl ffmpeg python3-venv >/dev/null
  apt-get install -y python3.11 python3.11-venv >/dev/null
  # pciutils (lspci) -- de Ollama-installer gebruikt dit om de NVIDIA-GPU te
  # detecteren en de CUDA-runtime te installeren. Zonder lspci installeert
  # Ollama stilzwijgend CPU-only, ook als nvidia-smi prima werkt.
  apt-get install -y pciutils >/dev/null

  touch "$marker"
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
  local want_torch="2.4.1+cu121"
  local want_audio="2.4.1+cu121"
  local want_vision="0.19.1+cu121"
  local cur_torch cur_audio cur_vision


  cur_torch=$(python - <<'PY' 2>/dev/null || true
import torch; print(torch.__version__)
PY
)

  cur_audio=$(python - <<'PY' 2>/dev/null || true
import torchaudio; print(torchaudio.__version__)
PY
)

  cur_vision=$(python - <<'PY' 2>/dev/null || true
import torchvision; print(torchvision.__version__)
PY
)


if [[ "$cur_torch" == "$want_torch" && "$cur_audio" == "$want_audio" && "$cur_vision" == "$want_vision" ]]; then
  log "Torch stack OK (${cur_torch}/${cur_audio}/${cur_vision}) — skip install"
  return 0
fi


  log "Torch stack mismatch → enforcing ${want_torch}/${want_audio}/${want_vision}"
  pip uninstall -y torch torchaudio torchvision >/dev/null 2>&1 || true
  pip install  --index-url https://download.pytorch.org/whl/cu121 \
    "torch==${want_torch}" "torchaudio==${want_audio}" "torchvision==${want_vision}"
}


install_nemo_sortformer() {
  # Installeer NeMo alleen als diarization=1 en backend=sortformer
  if [[ "${DIARIZATION:-0}" != "1" ]]; then
    log "DIARIZATION=0 → NeMo skip"
    return 0
  fi

  # Als jij later een env var wilt toevoegen voor backend: DIARIZATION_BACKEND
  if [[ "${DIARIZATION_BACKEND:-sortformer}" != "sortformer" ]]; then
    log "DIARIZATION_BACKEND!=sortformer → NeMo skip"
    return 0
  fi

  # Idempotent: als import werkt, niets doen
  python - <<'PY' >/dev/null 2>&1 && { log "NeMo (SortFormer) al aanwezig"; return 0; }
from nemo.collections.asr.models import SortformerEncLabelModel
PY

  log "Install NeMo toolkit (ASR) voor SortFormer..."
  pip install "nemo_toolkit[asr]@git+https://github.com/NVIDIA/NeMo.git@v2.6.1"

  # Verify
  python - <<'PY' || die "NeMo install faalde (SortFormer import lukt niet)"
from nemo.collections.asr.models import SortformerEncLabelModel
print("NeMo SortFormer OK")
PY
}

install_ollama() {
  # Ollama = de on-prem LLM-runtime voor gehoorverslag-sectieclassificatie.
  # OpenAI-compatibele API op localhost:11434 -- zelfde interface als lokaal
  # getest tijdens ontwikkeling, dus geen gedragsverschil dev vs. runpod.
  if [[ "${LLM_ENABLED}" != "1" ]]; then
    log "LLM_ENABLED=0 → Ollama-install skip"
    return 0
  fi
  # BEWUST geen "al geïnstalleerd → skip": het installer-script is zelf al
  # idempotent (veilig opnieuw te draaien, dat is ook hoe Ollama zelf updaten
  # wil dat je het doet) en moet hier altijd draaien zodat een eerdere
  # installatie zonder lspci (dus zonder GPU-detectie, CPU-only) alsnog de
  # ontbrekende CUDA-ondersteuning krijgt zodra pciutils/lspci beschikbaar is.
  log "Install/update Ollama (detecteert de NVIDIA-GPU via lspci voor CUDA-versnelling)..."
  curl -fsSL https://ollama.com/install.sh | sh
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
  
  # HuggingFace fast transfer hardening
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
  pip install -U huggingface_hub

  log "Install project + deps (editable) via pyproject.toml..."
  pip install -e .

  if [[ "${NLLB_ENABLED}" == "1" || "${SOMALI_BATCH_MODEL_ENABLED}" == "1" ]]; then
    # transformers is nodig voor de NLLB-tokenizer én voor ct2-transformers-converter
    # (modelroutering-PoC hieronder) -- ctranslate2 zelf is al een dep via faster-whisper.
    log "Install transformers (NLLB-tokenizer en/of ct2-transformers-converter)..."
    pip install transformers
  fi
  if [[ "${NLLB_ENABLED}" == "1" ]]; then
    log "Install langid (al-Nederlands-check bij tolk-vertaling)..."
    pip install langid
  fi
  if [[ "${NLLB_ENABLED}" != "1" && "${SOMALI_BATCH_MODEL_ENABLED}" != "1" ]]; then
    log "NLLB_ENABLED=0 en SOMALI_BATCH_MODEL_ENABLED=0 → transformers/langid-install skip"
  fi

  install_nemo_sortformer

  log "Sanity import checks..."
  python -c "import torch; print('torch', torch.__version__)" || die "torch import faalde"
  python -c "import torchaudio; print('torchaudio', torchaudio.__version__)" || die "torchaudio import faalde"
  python -c "import faster_whisper; print('faster_whisper OK')" || die "faster-whisper import faalde"
  python -c "import onnxruntime; print('onnxruntime OK')" || die "onnxruntime import faalde"
  if [[ "${NLLB_ENABLED}" == "1" ]]; then
    python -c "import ctranslate2, transformers, langid; print('ctranslate2', ctranslate2.__version__, '/ transformers', transformers.__version__)" || die "ctranslate2/transformers/langid import faalde"
  elif [[ "${SOMALI_BATCH_MODEL_ENABLED}" == "1" ]]; then
    python -c "import ctranslate2, transformers; print('ctranslate2', ctranslate2.__version__, '/ transformers', transformers.__version__)" || die "ctranslate2/transformers import faalde (nodig voor modelroutering-PoC conversie)"
  fi

  if [[ "${DIARIZATION}" == "1" ]]; then
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
  else
    log "DIARIZATION=0 → pyannote check skip"
  fi
}



# --- run ---
ensure_ollama_running() {
  [[ "${LLM_ENABLED}" == "1" ]] || return 0
  # --max-time 5 (2026-08-24): zonder een expliciete timeout hangt curl hier
  # onbeperkt als de poort wel openstaat maar het proces erachter niet meer
  # reageert (bv. per ongeluk gestopt door een Ctrl+Z die een losstaand,
  # genohup'd achtergrondproces raakte -- geobserveerd: `ollama serve` in
  # STAT=Tl, TCP accepteert de connectie maar levert nooit een response).
  # Zo'n hang hier blokkeert de hele --start-herstartsequentie stil, zonder
  # enige foutmelding.
  if curl -fsS --max-time 5 "http://localhost:11434/api/version" >/dev/null 2>&1; then
    log "Ollama draait al"
    return 0
  fi
  command -v ollama >/dev/null 2>&1 || die "LLM_ENABLED=1 maar 'ollama' niet gevonden. Run eerst: bash scripts/init.sh --setup"
  mkdir -p "$OLLAMA_MODELS"
  export OLLAMA_MODELS
  log "Start Ollama-service (achtergrond, modellen in $OLLAMA_MODELS, logt naar $WORKSPACE/ollama.log)..."
  nohup ollama serve > "$WORKSPACE/ollama.log" 2>&1 &
  disown
  for _ in $(seq 1 30); do
    curl -fsS --max-time 5 "http://localhost:11434/api/version" >/dev/null 2>&1 && { log "Ollama draait"; return 0; }
    sleep 1
  done
  die "Ollama startte niet binnen 30s -- zie $WORKSPACE/ollama.log"
}

pull_llm_model() {
  [[ "${LLM_ENABLED}" == "1" ]] || return 0
  # Idempotent: 'ollama pull' zelf is al een no-op als het model al aanwezig is.
  log "Zorg dat LLM-model aanwezig is: $LLM_MODEL (eerste keer kan even duren)..."
  ollama pull "$LLM_MODEL"
}

download_nllb_model() {
  [[ "${NLLB_ENABLED}" == "1" ]] || return 0
  # Idempotent: huggingface_hub.snapshot_download() is zelf al een no-op als
  # het model al in de HF-cache staat. Een lokaal pad (i.p.v. een HF-repo-id)
  # hoeft niet gedownload te worden.
  if [[ -d "$NLLB_MODEL" ]]; then
    log "NLLB_MODEL is een lokaal pad ($NLLB_MODEL) → download skip"
    return 0
  fi
  log "Zorg dat NLLB-vertaalmodel aanwezig is: $NLLB_MODEL (eerste keer kan even duren)..."
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$NLLB_MODEL")
PY
}

prepare_somali_batch_model() {
  # Modelroutering per kanaal (fase 1, batch-only, PoC -- zie het voorstel).
  # faster-whisper/BatchFasterWhisperASR kan alleen CTranslate2-formaat laden,
  # SOMALI_BATCH_MODEL_SRC is een gewoon HF/PyTorch-checkpoint -- daarom hier
  # één keer converteren, net als download_nllb_model() hierboven idempotent.
  [[ "${SOMALI_BATCH_MODEL_ENABLED:-0}" == "1" ]] || return 0
  if [[ -f "$SOMALI_BATCH_MODEL_DIR/model.bin" ]]; then
    log "Somalisch batch-model (CT2) al aanwezig → conversie skip: $SOMALI_BATCH_MODEL_DIR"
    return 0
  fi
  command -v ct2-transformers-converter >/dev/null 2>&1 \
    || die "ct2-transformers-converter niet gevonden (verwacht via het ctranslate2-pakket). Run eerst: bash scripts/init.sh --setup (met SOMALI_BATCH_MODEL_ENABLED=1)"
  log "Converteer Somalisch batch-model naar CTranslate2: $SOMALI_BATCH_MODEL_SRC → $SOMALI_BATCH_MODEL_DIR (eerste keer kan een tijd duren, download + conversie)..."
  mkdir -p "$(dirname "$SOMALI_BATCH_MODEL_DIR")"
  ct2-transformers-converter \
    --model "$SOMALI_BATCH_MODEL_SRC" \
    --output_dir "$SOMALI_BATCH_MODEL_DIR" \
    --quantization float16 \
    --force
}

startlive() {
  # IMPORTANT: --start should NOT do setup. It assumes repo+venv are already ready.
  [[ -d "$APP_DIR/.git" ]] || die "Repo niet gevonden in $APP_DIR. Run eerst: bash scripts/init.sh --setup"
  [[ -d "$VENV_DIR" ]] || die "Venv niet gevonden in $VENV_DIR. Run eerst: bash scripts/init.sh --setup"

  # RunPod's "Open Web Terminal" heeft een eigen idle-timeout, los van de pod
  # zelf -- valt die verbinding weg, dan verdween voorheen ook meteen de hele
  # server (was direct via `exec` aan die ene terminal-sessie gekoppeld, zie
  # onderin deze functie). Server draait nu op de achtergrond (nohup+disown,
  # zelfde bewezen patroon als ensure_ollama_running() hierboven) en overleeft
  # een weggevallen terminal. Deze PID-check zorgt dat een `--start` ná het
  # herverbinden niet blind een tweede server/poort-conflict/model-herlaad
  # veroorzaakt, maar gewoon weer aanhaakt bij de al-draaiende server.
  PID_FILE="$WORKSPACE/trivias.pid"
  LOG_FILE="$WORKSPACE/trivias.log"
  COMMIT_FILE="$WORKSPACE/trivias.commit"
  cd "$APP_DIR"
  current_commit="$(git rev-parse HEAD 2>/dev/null || echo "")"
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    running_commit="$(cat "$COMMIT_FILE" 2>/dev/null || echo "")"
    if [[ -n "$current_commit" && "$running_commit" == "$current_commit" ]]; then
      log "TriviasServer draait al (PID $(cat "$PID_FILE"), zelfde code) -- niet opnieuw gestart. Logs volgen ($LOG_FILE):"
      exec tail -f "$LOG_FILE"
    fi
    # Draait nog, maar met oudere code dan wat nu op schijf staat (bv. na een
    # git pull) -- zonder deze check zou "--start" stilzwijgend blijven
    # aanhaken bij een server die de nieuwste wijzigingen nooit geladen heeft.
    old_pid="$(cat "$PID_FILE")"
    log "TriviasServer draait (PID $old_pid) met oudere code (${running_commit:-onbekend} i.p.v. $current_commit) -- herstart..."
    kill "$old_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$old_pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -0 "$old_pid" 2>/dev/null && kill -9 "$old_pid" 2>/dev/null || true
  fi
  # shellcheck disable=SC1091
  # Venv nu al actief (i.p.v. pas vlak voor exec) -- download_nllb_model()
  # hieronder heeft de venv's python (met huggingface_hub) nodig, anders
  # gebruikt het per ongeluk het systeem-python zonder die dependency.
  source "$VENV_DIR/bin/activate"

  # HuggingFace: voorkom impliciete hf_transfer dependency op RunPod images --
  # ook nodig vóór download_nllb_model() hieronder, niet pas vlak voor exec.
  export HF_HUB_ENABLE_HF_TRANSFER=0

  DIAR_ARGS=()
  if [[ "$DIARIZATION" == "1" ]]; then
    DIAR_ARGS+=(--diarization --diarization-backend "$DIARIZATION_BACKEND")

  fi

  LLM_ARGS=()
  if [[ "$LLM_ENABLED" == "1" ]]; then
    ensure_ollama_running
    pull_llm_model
    LLM_ARGS+=(--llm-backend-url "$LLM_BACKEND_URL" --llm-model "$LLM_MODEL")
  else
    log "LLM_ENABLED=0 → geen actieve feature gebruikt dit momenteel (gereserveerd)"
  fi

  NLLB_ARGS=()
  if [[ "$NLLB_ENABLED" == "1" ]]; then
    download_nllb_model
    NLLB_ARGS+=(--nllb-model "$NLLB_MODEL" --nllb-device "$NLLB_DEVICE")
  else
    log "NLLB_ENABLED=0 → /translate draait in fallback-modus (geen vertaling)"
  fi

  ROUTING_ARGS=()
  if [[ "$SOMALI_BATCH_MODEL_ENABLED" == "1" ]]; then
    prepare_somali_batch_model
    ROUTING_ARGS+=(--foreign-so-batch-model "$SOMALI_BATCH_MODEL_DIR")
  else
    log "SOMALI_BATCH_MODEL_ENABLED=0 → foreign_so blijft op het server-brede standaardmodel (ongewijzigd gedrag)"
  fi

  log "Start TriviasServer: host=$HOST port=$PORT model=$MODEL lang=$LANGUAGE llm_enabled=$LLM_ENABLED nllb_enabled=$NLLB_ENABLED somali_batch_model_enabled=$SOMALI_BATCH_MODEL_ENABLED"
  nohup python -m whisperlivekit.TriviasServer \
    --host "$HOST" --port "$PORT" \
    --model "$MODEL" --language "$LANGUAGE" \
    --frame-threshold "$FRAME_THRESHOLD" \
    --audio-min-len "$AUDIO_MIN_LEN" \
    --audio-max-len "$AUDIO_MAX_LEN" \
    --beams "$BEAMS" \
    --pcm-input \
    "${DIAR_ARGS[@]}" \
    "${LLM_ARGS[@]}" \
    "${NLLB_ARGS[@]}" \
    "${ROUTING_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "$current_commit" > "$COMMIT_FILE"
  disown
  log "TriviasServer gestart op de achtergrond (PID $(cat "$PID_FILE"), commit ${current_commit:0:12}, log: $LOG_FILE) -- blijft nu draaien ook als deze terminal wegvalt. Logs volgen:"
  exec tail -f "$LOG_FILE"
}

gpustat() {
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
}

# --- orchestrators ---
do_setup() {
  redirect_home_cache_to_workspace
  git_identity
  install_deps
  setup_repo

  # setup_repo() kan dit scriptbestand zelf net hebben bijgewerkt (git reset
  # --hard). Bash heeft de REST van dit script echter al in het geheugen
  # geladen op het moment dat het startte -- zonder re-exec zou alles NA dit
  # punt (setup_venv_pip, install_ollama, en straks startlive) nog de OUDE
  # versie draaien, ook al staat de nieuwe allang op schijf. Dit heeft
  # concreet een keer een hele --setup-start-run laten doorlopen zonder de
  # net toegevoegde Ollama/LLM-stappen (en nogmaals met de NLLB-stappen,
  # 2026-08-16) -- REEXEC_MARKER (2e positieargument) voorkomt een lus.
  #
  # BEWUST GEEN environment-variabele (was eerder _INIT_REEXECED via
  # `export` + `exec`) -- dat bleek onbetrouwbaar: als deze guard ooit
  # ergens leeft in de omgeving (bv. via een eerdere `source`, of een
  # ge-exporteerde var die op een of andere manier al gezet was) wordt de
  # herstart stilzwijgend overgeslagen, óók bij een verse, losstaande
  # `bash scripts/init.sh ...`-aanroep -- exact het geconstateerde gedrag:
  # geen "herstart mezelf"-regel in de output, oude functiedefinities
  # (zonder --nllb-model) bleven draaien ondanks een geslaagde git pull.
  # Een positieargument kan nooit tussen twee losse `bash`-aanroepen lekken.
  if [[ "${REEXEC_MARKER:-}" != "--reexeced" ]]; then
    log "scripts/init.sh is bijgewerkt door setup_repo() -- herstart mezelf met de nieuwste versie..."
    exec bash "$APP_DIR/scripts/init.sh" "$MODE" --reexeced
  fi

  setup_venv_pip
  install_ollama
}

do_update() {
  git_identity
  setup_repo
}

# --- CLI ---
MODE="${1:-}"
REEXEC_MARKER="${2:-}"

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
