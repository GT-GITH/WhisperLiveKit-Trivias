import asyncio
import logging
import time
import uuid
import json
import wave

import numpy as np

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional, Iterable
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

from whisperlivekit import AudioProcessor, TranscriptionEngine, parse_args
from whisperlivekit.simul_whisper.backend import evaluate_batch_segment
from whisperlivekit.simul_whisper.config import get_channel_config
from whisperlivekit.cross_channel_gate import compute_cross_channel_gate_masks
from whisperlivekit.gehoorverslag import build_gehoorverslag_docx
from whisperlivekit.llm_backend import LLMBackend, build_llm_backend
from whisperlivekit.nllb_backend import NLLBBackend, build_nllb_backend
from whisperlivekit.translate import translate_text

from whisperlivekit.web_trivias.web_interface import get_inline_ui_html


# ====== Logging setup ======
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

logger = logging.getLogger("trivias.server")
logger.setLevel(logging.DEBUG)

logging.getLogger("whisperlivekit.tokens_alignment").setLevel(logging.DEBUG)
logging.getLogger("whisperlivekit.audio_processor").setLevel(logging.DEBUG)
logging.getLogger("whisperlivekit.backend").setLevel(logging.DEBUG)
logging.getLogger("whisperlivekit.simul_whisper").setLevel(logging.DEBUG)

# voorkom enorme numba debug spam
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)

LOG_FILE = "trivias_stt.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

# voorkom dubbele file handlers op herimport / herstart in dezelfde process
_existing_file_handler = None
for h in root_logger.handlers:
    if isinstance(h, RotatingFileHandler):
        try:
            if getattr(h, "baseFilename", "").endswith(LOG_FILE):
                _existing_file_handler = h
                break
        except Exception:
            pass

if _existing_file_handler is None:
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)

# ====== CLI args (zelfde als basic_server) ======
args = parse_args()


# ====== Session manager (v0.1: alleen in-memory + logging) ======
class SessionManager:
    """Eenvoudige in-memory session registry voor debug/doeleinden.

    Later kun je hier:
    - persistente opslag (DB, S3, etc.) aan koppelen
    - metadata uitbreiden (tolk, vreemdeling, gehoormedewerker, enz.)
    - transcript / diarization / inconsistency resultaten aan vastmaken
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create_or_update(
        self,
        session_id: str,
        source_system: Optional[str],
        external_references: Dict[str, Optional[str]],
        user_id: Optional[str],
        channel_id: Optional[str] = None,
        language: Optional[str] = None,
        language2: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = datetime.utcnow().isoformat() + "Z"
        meta = self._sessions.get(session_id, {})
        meta.update(
            {
                "session_id": session_id,
                "source_system": source_system or meta.get("source_system"),
                "external_references": {
                    **meta.get("external_references", {}),
                    **external_references,
                },
                "user_id": user_id or meta.get("user_id"),
                "last_seen": now,
                "created_at": meta.get("created_at", now),
            }
        )
        # Per-kanaal taalconfiguratie zoals meegegeven bij WS-connect (lang/lang2
        # query-params) -- vooral relevant voor de tolk (Taal 2), die anders
        # nergens buiten de levende AudioProcessor-instance bekend is. Puur
        # in-memory (zelfde levenscyclus als de rest van deze registry, zie
        # klasse-docstring) -- gaat verloren bij een serverherstart, maar dat is
        # een acceptabele v1-beperking (fail-safe: /translate valt dan terug op
        # "brontaal onbekend", niet op een gok).
        if channel_id:
            channels = meta.setdefault("channels", {})
            channels[channel_id] = {"language": language, "language2": language2}
        self._sessions[session_id] = meta
        logger.info(f"[SESSION] {session_id} ÔåÆ {meta}")
        return meta

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    def get_channel_language2(self, session_id: str, channel_id: str) -> Optional[str]:
        """De daadwerkelijk geconfigureerde Taal 2 (brontaal-hint bij een
        taalpaar-kanaal zoals de tolk) voor deze sessie/kanaal, indien bekend
        (alleen als dat kanaal ooit in déze serverlevensduur is verbonden)."""
        meta = self._sessions.get(session_id)
        if not meta:
            return None
        return (meta.get("channels", {}).get(channel_id) or {}).get("language2")

    def all(self) -> Dict[str, Dict[str, Any]]:
        return self._sessions


session_manager = SessionManager()

# Registry: session_id + channel_id → wav_path
# Gevuld door AudioProcessor via callback bij sessie-afsluiting
_wav_registry: Dict[str, str] = {}  # key = f"{session_id}:{channel_id}"

def register_wav_path(session_id: str, channel_id: str, wav_path: str) -> None:
    key = f"{session_id}:{channel_id}"
    _wav_registry[key] = wav_path
    logger.info(f"[WAV REGISTRY] {key} → {wav_path}")

def get_wav_path(session_id: str, channel_id: str) -> Optional[str]:
    return _wav_registry.get(f"{session_id}:{channel_id}")

# ====== Shared transcription engine ======
transcription_engine: Optional[TranscriptionEngine] = None

# On-prem LLM-backend (optioneel, zie llm_backend.py) -- None zolang
# --llm-backend-url niet gezet is. Nog niet aangeroepen door een actieve
# feature (het /translate-endpoint gebruikt sinds 2026-08-16 nllb_backend
# hieronder, niet dit chatmodel -- zie translate.py's moduledocstring voor
# waarom) -- gereserveerd voor de volgende roadmap-fase.
llm_backend: Optional[LLMBackend] = None

# On-prem NLLB-vertaalmodel (optioneel, zie nllb_backend.py) -- None zolang
# --nllb-model niet gezet is; /translate moet hier altijd fail-safe mee omgaan.
nllb_backend: Optional[NLLBBackend] = None

def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _safe_call(obj: Any, method_name: str, default: Any = None) -> Any:
    try:
        method = getattr(obj, method_name, None)
        if callable(method):
            return method()
    except Exception:
        pass
    return default


def _log_kv_block(title: str, values: Dict[str, Any]) -> None:
    logger.info(f"=== {title} ===")
    for k, v in values.items():
        logger.info(f"{k}: {v}")
    logger.info(f"=== END {title} ===")


def _extract_channel_cfg_snapshot(engine: Any) -> Dict[str, Any]:
    """
    Best effort snapshot van de bedoelde channel-aware config.
    Ook als deze niet direct aan de engine hangt, loggen we expliciet wat we kunnen vinden.
    """
    snapshot: Dict[str, Any] = {}

    # Mogelijke plekken waar channel config of args hangen
    args_obj = _safe_get(engine, "args", None)
    asr_obj = _safe_get(engine, "asr", None)
    batch_asr_obj = _safe_get(engine, "batch_asr", None)

    snapshot["channel_id"] = _safe_get(args_obj, "channel_id", "default")
    snapshot["args.frame_threshold"] = _safe_get(args_obj, "frame_threshold", None)
    snapshot["args.audio_min_len"] = _safe_get(args_obj, "audio_min_len", None)
    snapshot["args.audio_max_len"] = _safe_get(args_obj, "audio_max_len", None)
    snapshot["args.beams"] = _safe_get(args_obj, "beams", None)
    snapshot["args.decoder_type"] = _safe_get(args_obj, "decoder_type", None)
    snapshot["args.lan"] = _safe_get(args_obj, "lan", None)
    snapshot["args.task"] = _safe_get(args_obj, "task", None)

    # Wat batch/live objecten al concreet dragen
    snapshot["engine.batch_asr.language"] = _safe_get(batch_asr_obj, "language", None)
    snapshot["engine.batch_asr.task"] = _safe_get(batch_asr_obj, "task", None)
    snapshot["engine.batch_asr.beam_size"] = _safe_get(batch_asr_obj, "beam_size", None)

    snapshot["engine.asr.language"] = _safe_get(asr_obj, "language", None)
    snapshot["engine.asr.task"] = _safe_get(asr_obj, "task", None)

    return snapshot


def _extract_resolved_asr_snapshot(engine: Any) -> Dict[str, Any]:
    """
    Snapshot van de daadwerkelijke live-ASR runtime config zoals de engine hem gebruikt.
    Dit is de belangrijkste bron van waarheid voor je tests.
    """
    result: Dict[str, Any] = {}
    asr_obj = _safe_get(engine, "asr", None)
    cfg = _safe_get(asr_obj, "cfg", None)

    result["asr.class"] = type(asr_obj).__name__ if asr_obj is not None else None
    result["cfg.class"] = type(cfg).__name__ if cfg is not None else None

    if cfg is not None:
        for name in (
            "decoder_type",
            "beam_size",
            "frame_threshold",
            "rewind_threshold",
            "audio_min_len",
            "audio_max_len",
            "segment_length",
            "language",
            "task",
            "never_fire",
            "init_prompt",
            "static_init_prompt",
            "max_context_tokens",
            "cif_ckpt_path",
        ):
            result[f"cfg.{name}"] = _safe_get(cfg, name, None)

    return result


def _extract_batch_snapshot(engine: Any) -> Dict[str, Any]:
    """
    Snapshot van batch instellingen / batch decoder object.
    """
    result: Dict[str, Any] = {}
    batch_asr_obj = _safe_get(engine, "batch_asr", None)

    result["batch_asr.class"] = type(batch_asr_obj).__name__ if batch_asr_obj is not None else None

    if batch_asr_obj is not None:
        for name in (
            "language",
            "task",
            "beam_size",
            "temperature",
            "condition_on_previous_text",
            "initial_prompt",
        ):
            result[f"batch_asr.{name}"] = _safe_get(batch_asr_obj, name, None)

    return result

@asynccontextmanager
async def lifespan(app: FastAPI):
    _log_kv_block("TRIVIAS SERVER STARTUP PARAMETERS (RAW ARGS)", vars(args))

    global transcription_engine, llm_backend, nllb_backend
    logger.info("Initialising TranscriptionEngine for TriviasServer...")
    transcription_engine = TranscriptionEngine(**vars(args))
    logger.info("TranscriptionEngine ready.")

    llm_backend = build_llm_backend(args)
    if llm_backend is None:
        logger.info("[LLM] geen --llm-backend-url geconfigureerd (gereserveerd voor toekomstige features).")

    nllb_backend = build_nllb_backend(args)
    if nllb_backend is None:
        logger.info("[NLLB] geen --nllb-model geconfigureerd -- /translate draait in fallback-modus (503).")

    # 1) Snapshot van bedoelde input / channel-context
    try:
        _log_kv_block(
            "RESOLVED CHANNEL / ENGINE INPUT SNAPSHOT",
            _extract_channel_cfg_snapshot(transcription_engine),
        )
    except Exception as e:
        logger.warning(f"Could not log channel/engine input snapshot: {e}")

    # 2) Snapshot van de daadwerkelijke live ASR runtime config
    try:
        _log_kv_block(
            "RESOLVED LIVE ASR RUNTIME CONFIG",
            _extract_resolved_asr_snapshot(transcription_engine),
        )
    except Exception as e:
        logger.warning(f"Could not log resolved live ASR runtime config: {e}")

    # 3) Snapshot van batch settings/object
    try:
        _log_kv_block(
            "RESOLVED BATCH CONFIG",
            _extract_batch_snapshot(transcription_engine),
        )
    except Exception as e:
        logger.warning(f"Could not log resolved batch config: {e}")

    try:
        yield
    finally:
        logger.info("Shutting down TriviasServer lifespan...")
        logger.info("Lifespan cleanup done.")
        
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev-friendly; later strakker maken per domein
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Basic endpoints ======


@app.get("/health")
async def health():
    """Eenvoudige healthcheck voor monitoring en debugging."""
    return JSONResponse(
        {
            "status": "ok",
            "model": getattr(args, "model", None),
            "language": getattr(args, "language", None),
            "pcm_input": bool(getattr(args, "pcm_input", False)),
        }
    )

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve de inline Trivias STT webinterface."""
    return HTMLResponse(get_inline_ui_html())

def _parse_session_wav_name(stem: str) -> Optional[tuple[str, str, str]]:
    """Parse 'session_{uuid}_{channel}_{timestamp}' -> (uuid, channel, timestamp).

    De UUID zelf bevat NOOIT underscores (alleen hyphens), dus parts[1] is altijd
    exact de UUID. De channel-naam kan wél underscores bevatten (bv. "foreign_tr",
    "foreign_ar") -- dat is alles tussen de UUID en de timestamp, ongeacht hoeveel
    underscores erin zitten. Voorheen werd channel = parts[-2] aangenomen (altijd
    precies één segment), wat "foreign_tr" fout naar "tr" knipte en de UUID
    daardoor ook nog eens fout naar "{uuid}_foreign" verlengde -- resultaat: een
    kapotte, dubbel-geregistreerde sessie-entry en bij het terugkijken een
    onherkende channel_id die terugvalt op het generieke "Spreker"-label i.p.v.
    de echte rol (bv. "Vreemdeling"). Gedeeld door /sessions/list en
    /sessions/{id}/transcript zodat deze aanname maar op één plek staat."""
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    timestamp_part = parts[-1]
    session_uuid = parts[1]
    channel_part = "_".join(parts[2:-1])
    return session_uuid, channel_part, timestamp_part


def _resolve_channel_language(channel_id: str) -> str:
    """Bepaal de taal voor een kanaal zonder levende sessie nodig te hebben.

    De live WS-verbinding krijgt zijn taal via een los `lang=`-queryparam (zie
    websocket_endpoint) -- die waarde wordt nergens gepersisteerd, dus "Ververs
    Transcriptie" (draait achteraf, buiten elke levende verbinding om) kan er niet
    bij. Voor het "foreign"-kanaal codeert de client de taal echter al in de
    channel_id zelf (bv. "foreign_tr", zie getChannelId() in app.js) -- dat IS wel
    persistent (staat letterlijk in de WAV-bestandsnaam). Voor overige kanalen is de
    taal een vast rol-preset, dus get_channel_config() volstaat daar."""
    if channel_id and channel_id.startswith("foreign_"):
        return channel_id[len("foreign_"):] or "nl"
    return get_channel_config(channel_id).language


def _load_wav_f32(wav_path: Path) -> np.ndarray:
    """Lees een s16le-mono WAV volledig in als float32 in [-1, 1]. Gedeeld door
    rebuild_channel_transcript() (single-channel pad) en refresh_transcript()
    (multi-channel upfront-load voor compute_cross_channel_gate_masks), zodat
    de int16->float32-conversielogica maar op één plek staat."""
    with wave.open(str(wav_path), "rb") as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def rebuild_channel_transcript(
    session_id: str, channel_id: str, wav_path: Path, engine: TranscriptionEngine,
    audio_f32: Optional[np.ndarray] = None,
    gate_mask: Optional[np.ndarray] = None,
) -> list[dict]:
    """"Ververs Transcriptie": herbouw het transcript van één kanaal helemaal
    opnieuw, rechtstreeks vanaf de opgenomen WAV -- los van elke incrementele
    live-decoder-boekhouding (geen state.end_buffer, geen cumulative_time_offset,
    geen pauze-resets). Dat sluit de hele klasse aan drift-/spooklijn-bugs van
    vandaag uit door constructie, en gate elk faster-whisper-segment individueel
    i.p.v. per (tot 40s) venster zoals de incrementele batch-worker doet -- fijner
    en dus preciezer.

    audio_f32: optioneel al-ingeladen audio (het multi-channel refresh-pad laadt
    alle kanalen van de sessie al vooraf voor compute_cross_channel_gate_masks()
    en geeft ze hier door om een dubbele disk-read + conversie te vermijden).
    None -> laad zelf via _load_wav_f32(wav_path) (single-channel pad, ongewijzigd).

    gate_mask: optionele boolean keep-mask van compute_cross_channel_gate_masks()
    (cross-kanaal akoestisch-lek-onderdrukking), zelfde lengte als audio_f32.
    Exact dezelfde kopieer-en-nul-aanpak als de live gate
    (audio_processor.py _consume_gate_mask() + de zero-ing rond ~1908-1921): de
    WAV is hier sowieso alleen leesend geopend, dus dit raakt het bronbestand
    nooit -- de mask beïnvloedt alleen wat naar transcribe_full() gaat.

    Vervangt <wav_stem>.json volledig (gebruiker bevestigd: transcript-bestand is
    altijd vervangbaar, de WAV blijft de enige bron van waarheid)."""
    if audio_f32 is None:
        audio_f32 = _load_wav_f32(wav_path)

    if gate_mask is not None and len(gate_mask) == len(audio_f32) and not gate_mask.all():
        n_suppressed = int((~gate_mask).sum())
        logger.info(
            f"[REFRESH][XGATE] session={session_id} channel={channel_id} suppressed "
            f"{n_suppressed}/{len(audio_f32)} samples ({100.0 * n_suppressed / len(audio_f32):.1f}%) "
            f"before decode (cross-channel anti-leak, WAV unaffected)"
        )
        asr_audio_f32 = audio_f32.copy()
        asr_audio_f32[~gate_mask] = 0.0
    else:
        if gate_mask is not None and len(gate_mask) != len(audio_f32):
            logger.warning(
                f"[REFRESH][XGATE] session={session_id} channel={channel_id} mask length "
                f"{len(gate_mask)} != audio length {len(audio_f32)}, ignoring mask (fail-safe, no suppression)"
            )
        asr_audio_f32 = audio_f32

    lang = _resolve_channel_language(channel_id)
    segments = engine.batch_asr.transcribe_full(asr_audio_f32, language_override=lang)

    entries: list[dict] = []
    n_accepted = 0
    for seg in segments:
        start_ms = int(round(seg["start"] * 1000.0))
        end_ms = int(round(seg["end"] * 1000.0))
        # no_speech_threshold=None: zie evaluate_batch_segment() -- faster-whisper's
        # no_speech_prob is per intern ~30s-decodeerblok, niet per zin, en kan een
        # heel blok vol prima zinnen onterecht laten afkeuren als een blokgrens
        # toevallig net na een korte pauze valt. avg_logprob/compression_ratio/
        # HALLUCINATION_PATTERNS zijn hier de betrouwbare signalen.
        accepted, reason = evaluate_batch_segment(
            seg["avg_logprob"], seg["compression_ratio"], seg["no_speech_prob"], seg["text"],
            no_speech_threshold=None,
        )
        if accepted:
            n_accepted += 1
        else:
            logger.info(
                f"[REFRESH][REJECTED] session={session_id} channel={channel_id} "
                f"ms={start_ms}..{end_ms} reason={reason} text='{seg['text'][:80]}'"
            )
            if reason == "hallucination_pattern":
                # Net als _batch_worker()'s [BATCH][FILTER] (audio_processor.py): een
                # bekend hallucinatiepatroon is per definitie geen spraak, dus -- anders
                # dan bij een laag-confidence maar wél mogelijk echte zin (80c6a09) --
                # helemaal overslaan i.p.v. onbevestigd te bewaren. Zonder deze skip
                # bleef de hallucinatietekst (zonder vinkje) toch zichtbaar in het
                # ververste transcript, terwijl live 'm nooit liet zien.
                continue
        entries.append({
            "type": "segment_update",
            "id": f"refresh_{start_ms}",
            "text_batch": seg["text"] if accepted else None,
            "text_final": seg["text"],
            "state": "FINAL",
            "start_ms": start_ms,
            "end_ms": end_ms,
        })

    json_path = wav_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)

    logger.info(
        f"[REFRESH] session={session_id} channel={channel_id} lang={lang} "
        f"herbouwd: {len(entries)} segmenten, {n_accepted} bevestigd -> {json_path}"
    )
    return entries


@app.get("/sessions/list")
async def list_sessions_from_disk():
    """Lijst van alle sessies op basis van WAV bestanden op disk."""
    recordings_dir = Path("recordings")
    if not recordings_dir.exists():
        return JSONResponse({"sessions": []})

    sessions = {}
    for wav_file in sorted(recordings_dir.glob("session_*.wav"),
                           key=lambda f: f.stat().st_mtime, reverse=True):
        parsed = _parse_session_wav_name(wav_file.stem)
        if parsed is None:
            continue
        session_uuid, channel_part, timestamp_part = parsed

        key = session_uuid
        if key not in sessions:
            sessions[key] = {
                "session_id": session_uuid,
                "channels": [],
                "created_at": timestamp_part,
                "wav_size_mb": round(wav_file.stat().st_size / 1024 / 1024, 2),
                "has_transcript": wav_file.with_suffix(".json").exists(),
            }
        sessions[key]["channels"].append(channel_part)

    return JSONResponse({"sessions": list(sessions.values())})


def _load_merged_transcript(session_id: str) -> Optional[Dict[str, Any]]:
    """Laad en merge alle kanaal-transcripten van een sessie, chronologisch
    gesorteerd en elk segment getagd met channel_id. Gedeeld tussen
    GET /sessions/{id}/transcript?channel_id=all en
    GET /sessions/{id}/gehoorverslag, zodat beide altijd van dezelfde
    brontekst uitgaan. Retourneert None als er geen transcript-bestanden zijn."""
    recordings_dir = Path("recordings")
    by_channel: Dict[str, Path] = {}
    earliest_ts: Optional[str] = None
    for json_path in recordings_dir.glob(f"session_{session_id}_*.json"):
        parsed = _parse_session_wav_name(json_path.stem)
        if parsed is None:
            continue
        uuid_part, ch, ts = parsed
        if uuid_part != session_id:
            continue
        if ch not in by_channel or json_path.stat().st_mtime > by_channel[ch].stat().st_mtime:
            by_channel[ch] = json_path
        if earliest_ts is None or ts < earliest_ts:
            earliest_ts = ts

    if not by_channel:
        return None

    date_str: Optional[str] = None
    if earliest_ts:
        try:
            date_str = datetime.strptime(earliest_ts, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            date_str = None

    merged_segments = []
    for ch, json_path in by_channel.items():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                segs = json.load(f)
        except Exception:
            continue
        for seg in segs:
            seg = dict(seg)
            seg["channel_id"] = ch
            merged_segments.append(seg)
    merged_segments.sort(key=lambda s: s.get("start_ms") or 0)

    return {"channels": sorted(by_channel.keys()), "segments": merged_segments, "date": date_str}


@app.get("/sessions/{session_id}/transcript")
async def get_session_transcript(session_id: str, channel_id: str = Query(default="default")):
    """Haal het opgeslagen transcript op voor een sessie.

    channel_id="all" -> gemergd transcript van alle kanalen van deze sessie,
    chronologisch gesorteerd en elk segment getagd met channel_id, zodat de
    frontend kan tonen wie wat zei (en achteraf per kanaal kan filteren) --
    net zoals de live-weergave kanalen al door elkaar toont."""
    recordings_dir = Path("recordings")

    if channel_id == "all":
        merged = _load_merged_transcript(session_id)
        if merged is None:
            return JSONResponse({"error": "transcript not found"}, status_code=404)

        return JSONResponse({
            "session_id": session_id,
            "channel_id": "all",
            "channels": merged["channels"],
            "segments": merged["segments"],
        })

    # Zoek JSON bestand voor deze sessie + channel
    pattern = f"session_{session_id}_{channel_id}_*.json"
    matches = list(recordings_dir.glob(pattern))
    if not matches:
        return JSONResponse({"error": "transcript not found"}, status_code=404)

    # Nieuwste bestand
    json_path = sorted(matches, key=lambda f: f.stat().st_mtime)[-1]
    wav_path = json_path.with_suffix(".wav")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
        for seg in segments:
            seg["channel_id"] = channel_id
        return JSONResponse({
            "session_id": session_id,
            "channel_id": channel_id,
            "wav_available": wav_path.exists(),
            "segments": segments,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/sessions/{session_id}/refresh_transcript")
async def refresh_transcript(session_id: str):
    """"Ververs Transcriptie": vervang het transcript van elk kanaal in deze sessie
    door een verse batch-herbouw vanaf de tot-nu-toe opgenomen WAV.

    Gaat ervan uit dat de WAV-bestanden actueel zijn: bij een gestopte sessie is dat
    altijd zo (WAV al gesloten), bij een gepauzeerde sessie moet de client eerst een
    flag=3 flush-frame over de levende WebSocket sturen (zie audio_processor.py) voor
    hij dit endpoint aanroept."""
    recordings_dir = Path("recordings")
    wav_files: Dict[str, Path] = {}
    for wav_file in recordings_dir.glob(f"session_{session_id}_*.wav"):
        parsed = _parse_session_wav_name(wav_file.stem)
        if parsed is None:
            continue
        uuid_part, channel_part, _ts = parsed
        if uuid_part != session_id:
            continue
        if channel_part not in wav_files or wav_file.stat().st_mtime > wav_files[channel_part].stat().st_mtime:
            wav_files[channel_part] = wav_file

    if not wav_files:
        return JSONResponse({"error": "session not found"}, status_code=404)

    if transcription_engine is None or not getattr(transcription_engine, "batch_asr", None):
        return JSONResponse({"error": "batch model not available"}, status_code=503)

    # Anti-lek-gate: laad ALLE kanalen vooraf (cross-kanaal-arbitrage kan niet
    # per kanaal los bepaald worden) en bereken de gate-masks, VOORDAT de
    # sequentiële decode-loop hieronder start. Ook bij een sessie met maar 1
    # kanaal de moeite waard: compute_cross_channel_gate_masks() past dan nog
    # wel de eigen-kanaal-ruisdrempel toe (net als live altijd doet), slaat
    # alleen de cross-kanaal-arbitrage over. Elke fout hier valt terug op het
    # bestaande, ongewijzigde gedrag (de hele refresh faalt hier nooit door)
    # -- consistent fail-safe.
    audio_by_channel: Dict[str, np.ndarray] = {}
    gate_masks: Dict[str, np.ndarray] = {}
    for channel_id, wav_file in wav_files.items():
        try:
            audio_by_channel[channel_id] = await asyncio.to_thread(_load_wav_f32, wav_file)
        except Exception as e:
            logger.warning(
                f"[REFRESH][XGATE] session={session_id} channel={channel_id} "
                f"kon WAV niet laden voor anti-lek-gate: {e}"
            )
    if audio_by_channel:
        try:
            t0 = time.monotonic()
            gate_masks = await asyncio.to_thread(
                compute_cross_channel_gate_masks, audio_by_channel, 16000, session_id=session_id,
            )
            logger.info(
                f"[REFRESH][XGATE] session={session_id} channels={sorted(audio_by_channel)} "
                f"gate-berekening compleet in {(time.monotonic() - t0) * 1000:.0f}ms"
            )
        except Exception as e:
            logger.warning(
                f"[REFRESH][XGATE] session={session_id} anti-lek-gate mislukt, "
                f"ga verder zonder onderdrukking: {e}"
            )
            gate_masks = {}

    # Sequentieel (niet parallel): batch_asr deelt één GPU-model-instance, en de
    # incrementele batch-worker verwerkt zijn jobs ook al sequentieel vanuit een
    # queue -- concurrent aanroepen op dezelfde WhisperModel is niet iets waar dit
    # project op vertrouwt.
    merged_segments = []
    channels_done = []
    for channel_id, wav_file in wav_files.items():
        try:
            entries = await asyncio.to_thread(
                rebuild_channel_transcript, session_id, channel_id, wav_file, transcription_engine,
                audio_by_channel.get(channel_id), gate_masks.get(channel_id),
            )
        except Exception as e:
            logger.warning(f"[REFRESH] channel={channel_id} mislukt: {e}")
            continue
        for entry in entries:
            entry = dict(entry)
            entry["channel_id"] = channel_id
            merged_segments.append(entry)
        channels_done.append(channel_id)

    if not channels_done:
        return JSONResponse({"error": "refresh failed for all channels"}, status_code=500)

    merged_segments.sort(key=lambda s: s.get("start_ms") or 0)

    return JSONResponse({
        "session_id": session_id,
        "channel_id": "all",
        "channels": channels_done,
        "segments": merged_segments,
    })


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    meta = session_manager.get(session_id)
    if not meta:
        return JSONResponse({"error": "unknown session_id"}, status_code=404)
    return JSONResponse(meta)

from fastapi.responses import StreamingResponse
import io

@app.get("/audio/{session_id}/{channel_id}")
async def serve_audio_slice(
    session_id: str,
    channel_id: str,
    start_ms: int = Query(default=0),
    end_ms: int = Query(default=0),
):
    """Serveer een WAV-slice voor terugluisteren."""
    wav_path = get_wav_path(session_id, channel_id)
    if not wav_path:
        # Fallback: zoek op disk
        recordings_dir = Path("recordings")
        pattern = f"session_{session_id}_{channel_id}_*.wav"
        matches = list(recordings_dir.glob(pattern))
        if not matches:
            return JSONResponse({"error": "session or channel not found"}, status_code=404)
        wav_path = str(sorted(matches, key=lambda f: f.stat().st_mtime)[-1])   
    path = Path(wav_path)
    if not path.exists():
        return JSONResponse({"error": "wav file not found"}, status_code=404)

    try:
        with wave.open(str(path), "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            total_frames = wf.getnframes()

            start_frame = int((start_ms / 1000.0) * sample_rate)
            end_frame = int((end_ms / 1000.0) * sample_rate) if end_ms > 0 else total_frames
            end_frame = min(end_frame, total_frames)
            n_frames = max(0, end_frame - start_frame)

            wf.setpos(start_frame)
            raw = wf.readframes(n_frames)

        # Schrijf slice als WAV naar buffer
        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(n_channels)
            out.setsampwidth(sampwidth)
            out.setframerate(sample_rate)
            out.writeframes(raw)
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline"},
        )
    except Exception as e:
        logger.warning(f"[AUDIO SERVE] error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/sessions/{session_id}/gehoorverslag")
async def generate_gehoorverslag(session_id: str):
    """Genereer het gehoorverslag-export (v1) als .docx-download: een
    letterlijke, chronologische, sprekergelabelde weergave, geen structuur
    of classificatie -- de hoormedewerker plakt dit zelf in het door INDiGO
    gegenereerde "Rapport nader gehoor" en werkt het verder af (zie
    features/gehoorverslag-automatisering.md in de projectrepo).

    Bouwt op dezelfde gemergde, chronologische transcript-data als
    GET /sessions/{id}/transcript?channel_id=all (_load_merged_transcript())
    -- geen aparte databron, dus nooit inconsistent met wat in de UI te zien
    is. Werkt op wat er nu ligt, incrementeel of net ververst -- de
    medewerker kiest zelf wanneer te exporteren."""
    merged = _load_merged_transcript(session_id)
    if merged is None:
        return JSONResponse({"error": "transcript not found"}, status_code=404)

    session_meta = {
        "date": merged.get("date"),
        "languages": {ch: _resolve_channel_language(ch) for ch in merged["channels"]},
    }

    try:
        document = build_gehoorverslag_docx(session_id, merged["segments"], session_meta)
        buf = io.BytesIO()
        document.save(buf)
        buf.seek(0)
    except Exception as e:
        logger.warning(f"[GEHOORVERSLAG] session={session_id} generatie mislukt: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

    logger.info(f"[GEHOORVERSLAG] session={session_id} gegenereerd, {len(merged['segments'])} segmenten in bron")
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="gehoorverslag_{session_id}.docx"'},
    )


class TranslateRequest(BaseModel):
    text: str
    channel_id: str
    session_id: Optional[str] = None


@app.post("/translate")
async def translate_segment(payload: TranslateRequest):
    """Vertaalt een los tekstfragment naar het Nederlands, on-demand (geen
    persistente opslag -- zie features/vertaling-niet-nl-tekst.md).

    Draait op NLLB-200 (nllb_backend.py), niet op het LLM-chatmodel -- zie
    translate.py's moduledocstring voor waarom (chatmodel bleek onbetrouwbaar
    op complexe brontalen). Belangrijk verschil met de eerdere aanpak: NLLB
    kan de brontaal niet zelf detecteren/raden, dus is een expliciete,
    betrouwbare bron-taal-hint verplicht.

    Twee bronnen voor die hint, op volgorde geprobeerd:
    1. "foreign_*"-kanalen: taal staat letterlijk (en per-sessie correct) in
       de channel_id zelf (zie _resolve_channel_language()).
    2. Overige kanalen (bv. tolk): get_channel_config().language is een vast
       rol-preset ("nl"), niet per se wat er in dít fragment gezegd is --
       daarom wordt i.p.v. dat preset de daadwerkelijk bij WS-connect
       meegegeven Taal 2 opgezocht via session_manager (zie
       SessionManager.get_channel_language2()). Alleen bekend als dat kanaal
       ooit in déze serverlevensduur verbonden is geweest (in-memory, geen
       DB) -- anders faalt dit bewust i.p.v. te gokken (dat gaf eerder
       precies de onbetrouwbare vertalingen die deze herbouw moest
       oplossen)."""
    if nllb_backend is None:
        return JSONResponse({"error": "vertaalfunctie niet geconfigureerd (geen --nllb-model)"}, status_code=503)

    channel_id = payload.channel_id or ""
    if channel_id.startswith("foreign_"):
        source_language = _resolve_channel_language(channel_id)
    else:
        source_language = (
            session_manager.get_channel_language2(payload.session_id, channel_id)
            if payload.session_id else None
        )
        if not source_language:
            return JSONResponse(
                {"error": "brontaal van dit kanaal niet bekend (sessie niet meer actief op deze server, "
                          "of geen taalpaar geconfigureerd) -- vertalen niet mogelijk"},
                status_code=502,
            )

    translation = translate_text(payload.text, source_language, nllb_backend)
    if translation is None:
        return JSONResponse({"error": "vertalen mislukt"}, status_code=502)

    return JSONResponse({"translation": translation})


# ====== WebSocket result handler ======
async def handle_websocket_results(websocket: WebSocket, results_generator):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            # WhisperLiveKit geeft een object met .to_dict()
            await websocket.send_json(response.to_dict())
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results (client closed connection?).")
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


# ====== WebSocket ASR endpoint ======


@app.websocket("/asr")
async def websocket_endpoint(
    websocket: WebSocket,
    # optionele query parameters voor integratie met klantapplicaties:
    session_id: Optional[str] = Query(default=None),
    source_system: Optional[str] = Query(default=None),
    case_ref: Optional[str] = Query(default=None),
    person_ref: Optional[str] = Query(default=None),
    user_id: Optional[str] = Query(default=None),
    channel_id: Optional[str] = Query(default=None),
    lang: Optional[str] = Query(default=None),
    lang2: Optional[str] = Query(default=None),
    gate_framed: bool = Query(default=False),
):
    """Hoofdstream voor audio ÔåÆ ASR (exactzelfde kern als basic_server, maar met session-metadata).

    gate_framed=1: elk binair audio-bericht begint met 1 vlag-byte (0x00/0x01) dat aangeeft
    of dit fragment naar ASR mag (cross-kanaal anti-lek, alleen gebruikt door web_trivias/app.js).
    De WAV-opname krijgt altijd de volledige audio, ongeacht deze vlag. Zie docs/API.md.
    """
    global transcription_engine
    if transcription_engine is None:
        logger.error("TranscriptionEngine is not initialized.")
        await websocket.close(code=1011)
        return

    # Sessiesleutel bepalen
    sid = session_id or str(uuid.uuid4())
    session_meta = session_manager.create_or_update(
        session_id=sid,
        source_system=source_system,
        external_references={"case_ref": case_ref, "person_ref": person_ref},
        user_id=user_id,
        channel_id=channel_id,
        language=lang,
        language2=lang2,
    )

    #audio_processor = AudioProcessor(transcription_engine=transcription_engine)
    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
        session_id=sid,
        channel_id=channel_id or "default",
        language=lang,
        language2=lang2,
        gate_framed=gate_framed,
    )

    await websocket.accept()
    logger.info(f"WebSocket connection opened for session {sid}.")

    # Config naar client sturen (zelfde semantics als basic_server)
    try:
        await websocket.send_json({"type": "config", "useAudioWorklet": bool(args.pcm_input)})
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")

    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

    # Registreer WAV zodra die aangemaakt wordt (binnen 0.5s na eerste audio)
    async def _watch_wav_path():
        for _ in range(60):
            await asyncio.sleep(0.5)
            if getattr(audio_processor, "_wav_path", None):
                register_wav_path(sid, channel_id or "default", str(audio_processor._wav_path))
                return
    asyncio.create_task(_watch_wav_path())
    
    try:
        while True:
            message = await websocket.receive_bytes()
            # Hier kun je later per kanaal/session extra metadata koppelen
            await audio_processor.process_audio(message)
    except KeyError as e:
        if "bytes" in str(e):
            logger.warning("Client has closed the connection (KeyError on 'bytes').")
        else:
            logger.error(f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected by client during main loop (session_id={sid}).")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True)
    finally:
        logger.info(f"Cleaning up WebSocket endpoint for session {sid}...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}") 
            # Registreer WAV pad voor terugluisteren
        if audio_processor._wav_path:
            register_wav_path(sid, channel_id or "default", str(audio_processor._wav_path))
        
        logger.info(f"Cleaning up WebSocket endpoint for session {sid}...")
                
        await audio_processor.cleanup()
        logger.info(f"WebSocket endpoint cleaned up successfully for session {sid}.")

@app.websocket("/ws")
async def websocket_ws(
    websocket: WebSocket,
    session_id: Optional[str] = Query(default=None),
    source_system: Optional[str] = Query(default=None),
    case_ref: Optional[str] = Query(default=None),
    person_ref: Optional[str] = Query(default=None),
    user_id: Optional[str] = Query(default=None),
    channel_id: Optional[str] = Query(default=None),
    lang: Optional[str] = Query(default=None),
    lang2: Optional[str] = Query(default=None),
    gate_framed: bool = Query(default=False),
):
    """Compat-endpoint voor clients die nog /ws gebruiken."""
    return await websocket_endpoint(
        websocket=websocket,
        session_id=session_id,
        source_system=source_system,
        case_ref=case_ref,
        person_ref=person_ref,
        user_id=user_id,
        channel_id=channel_id,
        lang=lang,
        lang2=lang2,
        gate_framed=gate_framed,
    )

def main():
    """CLI entry point voor TriviasServer.

    Gebruik:
      python TriviasServer.py --model large-v3 --language nl --frame-threshold 4 --audio-max-len 30.0 ...
    """
    import uvicorn

    uvicorn_kwargs = {
        "app": "whisperlivekit.TriviasServer:app",  # module:object (bestandsnaam = TriviasServer.py)
        "host": args.host,
        "port": args.port,
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }

    ssl_kwargs = {}
    if getattr(args, "ssl_certfile", None) or getattr(args, "ssl_keyfile", None):
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile,
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}
    if getattr(args, "forwarded_allow_ips", None):
        uvicorn_kwargs = {**uvicorn_kwargs, "forwarded_allow_ips": args.forwarded_allow_ips}

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
