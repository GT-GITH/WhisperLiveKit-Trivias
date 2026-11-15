#!/usr/bin/env bash
# Housekeeping voor /workspace (netwerkvolume). Aangeroepen door init.sh bij elke
# serverstart (startlive), ook los te draaien: bash scripts/housekeeping.sh
#
# 1) De vorige serverlog (trivias.log) wordt BEWAARD i.p.v. overschreven: gecomprimeerd
#    naar $WORKSPACE/logs/trivias-<tijd>.log.gz, laatste $KEEP_SERVER_LOGS blijven.
#    Aanleiding (2026-11-15): de log van een experiment werd bij een herstart
#    overschreven, waardoor het bewijs voor een gevonden fout weg was.
# 2) Losse logbestanden in $WORKSPACE (start*.log, e2e*.log, ...) -> logs/archive/*.gz.
# 3) E2E-resultaten: laatste $KEEP_E2E_RESULTS runs bewaren.
# 4) Omvang /workspace (op de achtergrond, du is traag op het netwerkvolume; df toont
#    hier het hele opslagcluster, niet het volume) -> logs/disk_usage.txt, met
#    waarschuwing boven $DISK_WARN_PCT% van $VOLUME_SIZE_GB GB.
# Opnames (recordings/) worden NOOIT automatisch verwijderd: daar staan ook echte
# testopnames van de gebruiker tussen. Alleen de omvang wordt gemeld.
set -u
WORKSPACE="${WORKSPACE:-/workspace}"
APP_DIR="${APP_DIR:-$WORKSPACE/WhisperLiveKit-Trivias}"
KEEP_SERVER_LOGS="${KEEP_SERVER_LOGS:-20}"
KEEP_E2E_RESULTS="${KEEP_E2E_RESULTS:-30}"
DISK_WARN_PCT="${DISK_WARN_PCT:-85}"
VOLUME_SIZE_GB="${VOLUME_SIZE_GB:-75}"

log() { echo "[housekeeping] $*"; }

mkdir -p "$WORKSPACE/logs/archive"
stamp="$(date -u +%Y%m%d-%H%M%S)"

# 1) serverlog bewaren
if [[ -s "$WORKSPACE/trivias.log" ]]; then
  dest="$WORKSPACE/logs/trivias-$stamp.log"
  mv "$WORKSPACE/trivias.log" "$dest" && gzip -f "$dest" \
    && log "vorige serverlog bewaard: ${dest}.gz"
fi
ls -1t "$WORKSPACE"/logs/trivias-*.log.gz 2>/dev/null | tail -n +$((KEEP_SERVER_LOGS + 1)) | xargs -r rm -f

# 2) losse logs archiveren -- alleen als ze >= 10 min niet meer veranderd zijn, zodat
#    een log waar nog naar geschreven wordt (bv. de start-log van deze run) blijft staan.
find "$WORKSPACE" -maxdepth 1 -name '*.log' ! -name 'trivias.log' -mmin +10 -print0 2>/dev/null |
  while IFS= read -r -d '' f; do
    gzip -c "$f" > "$WORKSPACE/logs/archive/$(basename "$f" .log)-$stamp.log.gz" && rm -f "$f"
  done
n_arch=$(ls -1 "$WORKSPACE"/logs/archive/*.gz 2>/dev/null | wc -l)
[[ "$n_arch" -gt 0 ]] && log "losse logs gearchiveerd (totaal $n_arch in logs/archive)"

# 3) e2e-resultaten
if [[ -d "$APP_DIR/tests/e2e/results" ]]; then
  ls -1dt "$APP_DIR"/tests/e2e/results/*/ 2>/dev/null | tail -n +$((KEEP_E2E_RESULTS + 1)) | xargs -r rm -rf
fi

# 4) omvang op de achtergrond meten
(
  used_kb=$(du -sk "$WORKSPACE" 2>/dev/null | cut -f1)
  rec=$(du -sh "$APP_DIR/recordings" 2>/dev/null | cut -f1)
  pct=$(( used_kb * 100 / (VOLUME_SIZE_GB * 1024 * 1024) ))
  {
    echo "$(date -u +%FT%TZ) /workspace: $((used_kb / 1024 / 1024)) GB van ${VOLUME_SIZE_GB} GB (${pct}%); opnames: ${rec}"
    [[ "$pct" -ge "$DISK_WARN_PCT" ]] && echo "WAARSCHUWING: volume >= ${DISK_WARN_PCT}% vol -- ruim op of vergroot het volume"
  } >> "$WORKSPACE/logs/disk_usage.txt"
) >/dev/null 2>&1 &
log "omvangmeting loopt op de achtergrond -> $WORKSPACE/logs/disk_usage.txt"
exit 0
