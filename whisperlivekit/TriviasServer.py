import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional, Iterable
from logging.handlers import RotatingFileHandler
 
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from whisperlivekit import AudioProcessor, TranscriptionEngine, parse_args

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
        self._sessions[session_id] = meta
        logger.info(f"[SESSION] {session_id} ÔåÆ {meta}")
        return meta

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    def all(self) -> Dict[str, Dict[str, Any]]:
        return self._sessions


session_manager = SessionManager()

# ====== Shared transcription engine ======
transcription_engine: Optional[TranscriptionEngine] = None

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

    global transcription_engine
    logger.info("Initialising TranscriptionEngine for TriviasServer...")
    transcription_engine = TranscriptionEngine(**vars(args))
    logger.info("TranscriptionEngine ready.")

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

@app.get("/sessions")
async def list_sessions():
    """Debug endpoint: toon alle actieve / bekende sessies."""
    return JSONResponse({"sessions": session_manager.all()})


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    meta = session_manager.get(session_id)
    if not meta:
        return JSONResponse({"error": "unknown session_id"}, status_code=404)
    return JSONResponse(meta)


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
):
    """Hoofdstream voor audio ÔåÆ ASR (exactzelfde kern als basic_server, maar met session-metadata)."""
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
    )

    audio_processor = AudioProcessor(transcription_engine=transcription_engine)

    await websocket.accept()
    logger.info(f"WebSocket connection opened for session {sid}.")

    # Config naar client sturen (zelfde semantics als basic_server)
    try:
        await websocket.send_json({"type": "config", "useAudioWorklet": bool(args.pcm_input)})
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")

    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

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
):
    """
    Compat-endpoint voor clients die nog /ws gebruiken.
    Roept intern dezelfde logica aan als /asr.
    """
    return await websocket_endpoint(
        websocket=websocket,
        session_id=session_id,
        source_system=source_system,
        case_ref=case_ref,
        person_ref=person_ref,
        user_id=user_id,
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
