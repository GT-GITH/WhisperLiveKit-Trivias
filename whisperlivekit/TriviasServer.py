import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional
 
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from starlette.staticfiles import StaticFiles
import pathlib
import whisperlivekit.web_trivias as webpkg

from whisperlivekit import AudioProcessor, TranscriptionEngine, parse_args

from whisperlivekit.web_trivias.web_interface import get_inline_ui_html

# ====== Logging setup ======
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)

logger = logging.getLogger("trivias.server")
logger.setLevel(logging.DEBUG)

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
        logger.info(f"[SESSION] {session_id} → {meta}")
        return meta

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    def all(self) -> Dict[str, Dict[str, Any]]:
        return self._sessions


session_manager = SessionManager()

# ====== Shared transcription engine ======
transcription_engine: Optional[TranscriptionEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== TRIVIAS SERVER STARTUP PARAMETERS (RAW ARGS) ===")
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")
    logger.info("=== END RAW ARGS ===")
    global transcription_engine
    logger.info("Initialising TranscriptionEngine for TriviasServer...")
    transcription_engine = TranscriptionEngine(**vars(args))
    logger.info("TranscriptionEngine ready.")
    try:
        yield
    finally:
        logger.info("Shutting down TriviasServer lifespan...")
        # Als er ooit een nette shutdown op TranscriptionEngine komt, kun je die hier aanroepen.
        # bijv: await transcription_engine.aclose()  (afhankelijk van library)
        logger.info("Lifespan cleanup done.")


app = FastAPI(lifespan=lifespan)

# ====== Static files for PCM AudioWorklet / Worker ======
web_dir = pathlib.Path(webpkg.__file__).parent
app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")

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
    """Hoofdstream voor audio → ASR (exactzelfde kern als basic_server, maar met session-metadata)."""
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
