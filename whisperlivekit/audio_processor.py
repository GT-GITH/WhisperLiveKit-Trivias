import asyncio
import logging
import traceback
import weakref

from collections import deque

import uuid

import os
import wave
from pathlib import Path
from datetime import datetime, timezone

from time import time
from typing import Any, AsyncGenerator, List, Optional, Tuple, Union
from whisperlivekit.timed_objects import SegmentUpdate

import numpy as np
import hashlib  
import json

from whisperlivekit.core import (TranscriptionEngine,
                                 online_diarization_factory, online_factory,
                                 online_translation_factory)
from whisperlivekit.ffmpeg_manager import FFmpegManager, FFmpegState
from whisperlivekit.silero_vad_iterator import FixedVADIterator, OnnxWrapper, load_jit_vad
from whisperlivekit.simul_whisper.backend import HALLUCINATION_PATTERNS, evaluate_batch_segment
from whisperlivekit.timed_objects import (ASRToken, ChangeSpeaker, FrontData,
                                          Segment, Silence, State, Transcript)
from whisperlivekit.tokens_alignment import TokensAlignment

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SENTINEL = object() # unique sentinel object for end of stream marker
# Per-queue pushback slot used by get_all_from_queue() to preserve ordering without peeking.
_QUEUE_PUSHBACK: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# Stilte die we als "segment boundary" gebruiken (dus FINAL/segment-close).
# Dit heeft niets met decoder reset te maken.
SILENCE_TOKEN_MIN_DURATION = 0.10  # 100ms (mag 0.0 als je alles wil)

# Vanaf hoeveel seconden stilte we de decoder (AlignAtt) resetten
SILENCE_RESET_THRESHOLD = 8.0  # sneller resetten om vervuilde live-state kort te houden

ENABLE_HARD_CAP = False

#BATCH_CONTEXT_PAD_MS = 600  # FO default pre/post padding
BATCH_CONTEXT_PAD_LEFT_MS = 600
BATCH_CONTEXT_PAD_RIGHT_MS = 0

# Batch windowing (Stap 1)
BATCH_TARGET_WINDOW_MS = 30_000   # 30s
BATCH_MIN_WINDOW_MS    = 15_000   # (nu nog niet gebruikt, maar handig)
BATCH_HARD_CAP_MS      = 45_000   # (nu nog niet gebruikt, maar handig)

_STATE_TOKENS_MAX_DURATION_S: float = 600.0  # maximaal 10 minuten tokens in state.tokens
_STATE_TOKENS_PRUNE_TRIGGER: int = 13_000    # alleen prunen boven deze grens

# state.end_buffer mag nooit verder vooruitlopen dan current_audio_processed_upto (de
# zuivere PCM-teller) + deze marge. Voorkomt dat een incidenteel foute attention-afgeleide
# tijdstempel (bv. van een hallucinerend " *"-token, zie [DIAG][END_BUFFER_JUMP]) de
# gedeelde end_buffer-status vergiftigt en batch-vensters buiten de opgenomen WAV laat
# vallen. Marge > 0 omdat legitieme tokentijdstempels soms een fractie voor de exacte
# chunk-grens kunnen liggen (woord-uitlijning is geen exacte wetenschap).
END_BUFFER_MAX_LOOKAHEAD_S: float = 3.0

def _sha1_pcm16(audio_f32: Optional[np.ndarray]) -> str:
    if audio_f32 is None or audio_f32.size == 0:
        return ""
    pcm16 = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    return hashlib.sha1(pcm16.tobytes()).hexdigest()


async def get_all_from_queue(queue: asyncio.Queue) -> Union[object, Silence, np.ndarray, List[Any]]:
    """
    Get one logical item from an asyncio.Queue.

    - For audio queues: coalesce consecutive np.ndarray chunks into a single np.concatenate() result.
    - For non-audio queues (e.g. translation): return a list of consecutive non-sentinel/non-silence items.

    We DO NOT peek into private queue internals (queue._queue). To preserve FIFO ordering when we
    encounter SENTINEL or Silence while draining, we use a per-queue pushback slot.
    """
    items: List[Any] = []

    # 0) If we have a pushed-back item, consume it first (preserves original order).
    if queue in _QUEUE_PUSHBACK:
        first_item = _QUEUE_PUSHBACK.pop(queue)
    else:
        first_item = await queue.get()
        queue.task_done()

    # 1) Sentinels / silence pass through directly
    if first_item is SENTINEL:
        return first_item
    if isinstance(first_item, Silence):
        return first_item

    items.append(first_item)

    # 2) Drain immediately-available items without blocking
    while True:
        try:
            nxt = queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            break

        # Stop at boundary items; push back so it will be returned next call (order preserved)
        if nxt is SENTINEL or isinstance(nxt, Silence):
            _QUEUE_PUSHBACK[queue] = nxt
            break

        # Coalesce only homogeneous audio chunks
        if isinstance(first_item, np.ndarray) and not isinstance(nxt, np.ndarray):
            # This should not happen in a well-formed audio queue; don't mix types.
            _QUEUE_PUSHBACK[queue] = nxt
            break

        items.append(nxt)

    # 3) Return coalesced audio or list of items (translation etc.)
    if isinstance(first_item, np.ndarray):
        return np.concatenate(items)
    return items


class AudioProcessor:
    """
    Processes audio streams for transcription and diarization.
    Handles audio processing, state management, and result formatting.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the audio processor with configuration, models, and state."""

        self.channel_id = kwargs.get("channel_id", "default")
        self.channel_language  = kwargs.get("language",  None)
        self.channel_language2 = kwargs.get("language2", None)

        if 'transcription_engine' in kwargs and isinstance(kwargs['transcription_engine'], TranscriptionEngine):
            models = kwargs['transcription_engine']
        else:
            models = TranscriptionEngine(**kwargs)
        
        # Batch window state (Stap 1)
        self._batch_window_start_ms: Optional[int] = None
        # Guard: voorkomt dubbele enqueue voor dezelfde window-close
        self._batch_last_close_end_ms: Optional[int] = None
        # Batch window control (Stap 1 refined)
        self._batch_ready_to_close: bool = False   # "30s gehaald" -> wacht nu op eerstvolgende silence-end
        self._batch_ready_at_ms: Optional[int] = None

        # Batch refinement (Stap 3)
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._batch_worker_task: Optional[asyncio.Task] = None
        
        # nieuwe asyncio.Queue voor tweede poging batch transscripttie
        self._ws_update_queue: asyncio.Queue = asyncio.Queue()

        # Audio processing settings
        self.args = models.args
        self.batch_asr = getattr(models, "batch_asr", None)
        _live_lang_effective = self.channel_language or getattr(self.args, "lan", None)
        _batch_lang_init     = getattr(self.batch_asr, "language", None) if self.batch_asr else None
        _batch_lang_override = self.channel_language  # van URL-param lang=
        logger.info(
            "AudioProcessor engine bound: channel_id=%s "
            "live_lang=%s batch_lang_init=%s batch_lang_override=%s (batch effectief: %s)",
            self.channel_id,
            _live_lang_effective,
            _batch_lang_init,
            _batch_lang_override,
            _batch_lang_override if _batch_lang_override else _batch_lang_init,
        )
        self.sample_rate = 16000
        self.channels = 1
        self.samples_per_sec = int(self.sample_rate * self.args.min_chunk_size)
        self.bytes_per_sample = 2
        self.bytes_per_sec = self.samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = 32000 * 5  # 5 seconds of audio at 32 kHz
        self.is_pcm_input = self.args.pcm_input

        # Cross-kanaal anti-lek: client stuurt per bericht 1 vlag-byte (open/dicht voor ASR).
        # De WAV-opname is hier nooit van afhankelijk (zie handle_pcm_data). Zie docs/API.md.
        self._gate_framed: bool = bool(kwargs.get("gate_framed", False))
        # deque van (n_samples, is_open)-segmenten, in lockstep met self.pcm_buffer gevuld
        self._gate_segments: "deque" = deque()

        # State management
        self.is_stopping: bool = False
        self.current_silence: Optional[Silence] = None
        self.state: State = State()
        self.lock: asyncio.Lock = asyncio.Lock()
        self.sep: str = " "  # Default separator
        self.last_response_content: FrontData = FrontData()
        # Status-log throttling (voorkomt spam per chunk)
        self._status_last_log_t: float = 0.0
        self._status_last_msg: Optional[str] = None
        self._status_min_interval_s: float = 1.0  # max 1x per seconde

        self.tokens_alignment: TokensAlignment = TokensAlignment(self.state, self.args, self.sep)
        self.beg_loop: Optional[float] = None
        # True zolang process_iter() (de decode-stap, draait via asyncio.to_thread op een
        # achtergrond-thread) bezig is. Zie _pause_flush(): queue.join() alleen garandeert
        # dat audio-chunks zijn OPGEHAALD uit transcription_queue (get_all_from_queue roept
        # task_done() al bij het ophalen aan, niet na verwerking) -- niet dat een lopende
        # decode-stap ook echt klaar is vóór _hard_reset_live_decoder() de decoder-context
        # reset.
        self._decode_in_progress: bool = False
        # [DIAG][LIVE_STALL] onderzoek (2026-08-23): wanneer gezet, kan results_formatter()
        # (een aparte asyncio-task die onafhankelijk van transcription_processor() blijft
        # tikken) detecteren of een decode-aanroep verdacht lang bezig is -- inclusief een
        # eventuele hang in start_silence() (roept process_iter(is_last=True) aan, maar
        # heeft -- anders dan het hoofdpad -- geen asyncio.wait_for-timeout-guard).
        self._decode_in_progress_since: Optional[float] = None
        self._live_stall_last_logged_t: float = 0.0

        # Models and processing
        self.asr: Any = models.asr
        self.vac: Optional[FixedVADIterator] = None
        
        if self.args.vac:
            if models.vac_session is not None:
                vac_model = OnnxWrapper(session=models.vac_session)
                self.vac = FixedVADIterator(vac_model)
            else:
                self.vac = FixedVADIterator(load_jit_vad())    
        self.ffmpeg_manager: Optional[FFmpegManager] = None
        self.ffmpeg_reader_task: Optional[asyncio.Task] = None
        self._ffmpeg_error: Optional[str] = None

        if not self.is_pcm_input:
            self.ffmpeg_manager = FFmpegManager(
                sample_rate=self.sample_rate,
                channels=self.channels
            )
            async def handle_ffmpeg_error(error_type: str):
                logger.error(f"FFmpeg error: {error_type}")
                self._ffmpeg_error = error_type
            self.ffmpeg_manager.on_error_callback = handle_ffmpeg_error
             
        self.transcription_queue: Optional[asyncio.Queue] = asyncio.Queue() if self.args.transcription else None
        self.diarization_queue: Optional[asyncio.Queue] = asyncio.Queue() if self.args.diarization else None
        self.translation_queue: Optional[asyncio.Queue] = asyncio.Queue() if self.args.target_language else None
        self.pcm_buffer: bytearray = bytearray()
        self.total_pcm_samples: int = 0
        self.transcription_task: Optional[asyncio.Task] = None
        self.diarization_task: Optional[asyncio.Task] = None
        self.translation_task: Optional[asyncio.Task] = None
        self.watchdog_task: Optional[asyncio.Task] = None
        self.all_tasks_for_cleanup: List[asyncio.Task] = []
        
        self.transcription: Optional[Any] = None
        self.translation: Optional[Any] = None
        self.diarization: Optional[Any] = None

        if self.args.transcription:
            self.transcription = online_factory(self.args, models.asr, language=self.channel_language)
            self.sep = self.transcription.asr.sep   
        if self.args.diarization:
            self.diarization = online_diarization_factory(self.args, models.diarization_model)
        if models.translation_model:
            self.translation = online_translation_factory(self.args, models.translation_model)

        # ====== Session WAV recording (1 file per session) ======
        self.session_id: str = str(kwargs.get("session_id") or uuid.uuid4())
        self.recordings_dir: Path = Path(kwargs.get("recordings_dir") or "recordings")
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

        self._wav_writer: Optional[wave.Wave_write] = None
        self._wav_path: Optional[Path] = None

    def _log_status_throttled(self, msg: str) -> None:
        # Max 1x per interval loggen (msg verandert steeds door lag, dus niet op msg vergelijken)
        now = time()
        if (now - self._status_last_log_t) < self._status_min_interval_s:
            return
        logger.debug(msg)
        self._status_last_log_t = now

    async def emit_segment_update(self, upd: SegmentUpdate) -> None:
        """Queue a WS update that will be yielded by results_formatter."""
        await self._ws_update_queue.put(upd)
        # Persisteer naar JSON naast de WAV
        self._persist_segment_update(upd)

    def _persist_segment_update(self, upd: SegmentUpdate) -> None:
        """Schrijf segment update naar JSON transcript bestand."""
        if not self._wav_path:
            return
        try:
            json_path = Path(str(self._wav_path).replace(".wav", ".json"))
            # Lees bestaande entries
            entries = []
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        entries = json.load(f)
                except Exception:
                    entries = []
            # Voeg toe of update bestaande entry
            entry = upd.to_dict()
            existing_ids = {e["id"]: i for i, e in enumerate(entries)}
            if upd.id in existing_ids:
                entries[existing_ids[upd.id]] = entry
            else:
                entries.append(entry)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[TRANSCRIPT][PERSIST] failed: {e}")

    async def _push_silence_event(self) -> None:
        if self.transcription_queue:
            await self.transcription_queue.put(self.current_silence)
        if self.args.diarization and self.diarization_queue:
            await self.diarization_queue.put(self.current_silence)
        if self.translation_queue:
            await self.translation_queue.put(self.current_silence)

    async def _begin_silence(self) -> None:
        if self.current_silence:
            return
        # [DIAG] spooklijn/ontbrekend-vinkje-onderzoek (2026-07-19): stond hier voorheen
        # op `time() - self.beg_loop` (wall-clock). Bij verwerkingsachterstand (gezien:
        # transcription_lag_s tot 49s) loopt wall-clock ver voor op wat er daadwerkelijk
        # aan PCM ontvangen/naar WAV geschreven is -- deze silence-tijdstempel voedt via
        # de max-ratchet uiteindelijk state.end_buffer, buiten de bestaande token-klemmen
        # om (die zitten alleen in transcription_processor()'s decode-tak, niet hier).
        # Reproductie: [DIAG][WAV_CLIP] ~10s afgeknipt op een silence_end-grens, en bij
        # Stop zelfs een venster dat volledig buiten de WAV viel (nul vinkje op de laatste
        # zin). self.total_pcm_samples is de bewezen betrouwbare grond -- puur een
        # optelsom van daadwerkelijk ontvangen/weggeschreven PCM-bytes, nooit voor de WAV
        # uit.
        now = self.total_pcm_samples / self.sample_rate
        self.current_silence = Silence(
            is_starting=True, start=now
        )
        await self._push_silence_event()

    async def _end_silence(self) -> None:
        if not self.current_silence:
            return
        # Zie toelichting in _begin_silence() hierboven.
        now = self.total_pcm_samples / self.sample_rate
        self.current_silence.end = now
        self.current_silence.is_starting=False
        self.current_silence.has_ended=True
        self.current_silence.compute_duration()
        if self.current_silence.duration >  SILENCE_TOKEN_MIN_DURATION:
            self.state.new_tokens.append(self.current_silence)
        await self._push_silence_event()
        self.current_silence = None

    def _hard_reset_live_decoder(self, reason: str) -> None:
        """
        Forceren van een schone reset van de live decoder + live buffers.
        Dit voorkomt dat vervuilde hypothesen zichtbaar blijven terugkomen.
        """
        logger.warning(f"[LIVE][HARD_RESET] reason={reason}")

        # 1) reset decoder via transcription.{asr|model}
        base_asr = None
        for attr_name in ("asr", "model"):
            candidate = getattr(self.transcription, attr_name, None)
            if candidate is not None and hasattr(candidate, "refresh_segment"):
                base_asr = candidate
                logger.info(
                    f"[LIVE][HARD_RESET] using self.transcription.{attr_name}.refresh_segment(complete=True)"
                )
                break

        if base_asr is not None:
            try:
                # [DIAG] end_buffer-drift onderzoek (2026-07-19): vastleggen wat de offset
                # was vlak vóór de reset, en wat state.end_buffer is op het moment dat het
                # zo dadelijk als nieuwe offset gebruikt wordt -- als state.end_buffer hier
                # al te hoog is, is de bron van de drift NIET deze restauratiestap zelf, maar
                # iets dat eerder al end_buffer heeft opgeblazen (zie [DIAG]-logs in
                # simul_whisper.py voor de kandidaten daar).
                _diag_offset_before = float(
                    getattr(getattr(base_asr, "state", None), "cumulative_time_offset", 0.0) or 0.0
                )
                base_asr.refresh_segment(complete=True)
                # refresh_segment() zet cumulative_time_offset terug naar 0.0, alsof de
                # audio opnieuw bij het begin start. Zonder correctie krijgen alle tokens
                # na een reset een tijdstempel die weer bij ~0 begint i.p.v. doortelt vanaf
                # het huidige punt in de sessie -- _get_last_speech_end_ms() (gebruikt door
                # de stilte-gestuurde batch-venster-sluiting) raakt daardoor voorgoed in de
                # war (window_end < window_start), en er sluit geen venster meer tussentijds
                # tot de eind-flush bij Stop (die wél een eigen fallback heeft en daarom
                # per ongeluk goed bleef gaan). Herstel de offset naar de echte absolute
                # positie in de sessie zodat tijdstempels doorlopend blijven.
                if hasattr(base_asr, "state") and hasattr(base_asr.state, "cumulative_time_offset"):
                    _diag_restore_value = float(getattr(self.state, "end_buffer", 0.0) or 0.0)
                    base_asr.state.cumulative_time_offset = _diag_restore_value
                    logger.info(
                        f"[DIAG][RESET_OFFSET] channel={self.channel_id} reason={reason} "
                        f"cumulative_time_offset {_diag_offset_before:.2f}s -> 0.0 (refresh_segment) "
                        f"-> {_diag_restore_value:.2f}s (hersteld uit state.end_buffer)"
                    )
            except Exception as e:
                logger.warning(f"[LIVE][HARD_RESET] refresh_segment failed: {e}")
        else:
            logger.warning("[LIVE][HARD_RESET] no decoder with refresh_segment found")

        # 2) wis live buffer in online processor
        try:
            if hasattr(self.transcription, "buffer"):
                self.transcription.buffer = []
        except Exception:
            pass

        # 3) wis zichtbare live tekstbuffer in state
        try:
            self.state.buffer_transcription = Transcript()
            self.state.new_tokens_buffer = Transcript()
        except Exception:
            pass
        
    async def _enqueue_active_audio(self, pcm_chunk: np.ndarray) -> None:
        if pcm_chunk is None or pcm_chunk.size == 0:
            return
        if self.transcription_queue:
            await self.transcription_queue.put(pcm_chunk.copy())
        if self.args.diarization and self.diarization_queue:
            await self.diarization_queue.put(pcm_chunk.copy())

    def _slice_before_silence(self, pcm_array: np.ndarray, chunk_sample_start: int, silence_sample: Optional[int]) -> Optional[np.ndarray]:
        if silence_sample is None:
            return None
        relative_index = int(silence_sample - chunk_sample_start)
        if relative_index <= 0:
            return None
        split_index = min(relative_index, len(pcm_array))
        if split_index <= 0:
            return None
        return pcm_array[:split_index]

    def convert_pcm_to_float(self, pcm_buffer: Union[bytes, bytearray]) -> np.ndarray:
        """Convert PCM buffer in s16le format to normalized NumPy array."""
        return np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            
    async def get_current_state(self) -> State:
        """Get current state."""
        async with self.lock:
            current_time = time()
            
            remaining_transcription = 0
            if self.state.end_buffer > 0:
                remaining_transcription = max(0, round(current_time - self.beg_loop - self.state.end_buffer, 1))
                
            remaining_diarization = 0
            if self.state.tokens:
                latest_end = max(self.state.end_buffer, self.state.tokens[-1].end if self.state.tokens else 0)
                remaining_diarization = max(0, round(latest_end - self.state.end_attributed_speaker, 1))
                
            self.state.remaining_time_transcription = remaining_transcription
            self.state.remaining_time_diarization = remaining_diarization
            
            return self.state
        
    def _now_ms(self) -> int:
        if not self.beg_loop:
            return 0
        return int((time() - self.beg_loop) * 1000)
  
    async def ffmpeg_stdout_reader(self) -> None:
        """Read audio data from FFmpeg stdout and process it into the PCM pipeline."""
        beg = time()
        while True:
            try:
                if self.is_stopping:
                    logger.info("Stopping ffmpeg_stdout_reader due to stopping flag.")
                    break

                state = await self.ffmpeg_manager.get_state() if self.ffmpeg_manager else FFmpegState.STOPPED
                if state == FFmpegState.FAILED:
                    logger.error("FFmpeg is in FAILED state, cannot read data")
                    break
                elif state == FFmpegState.STOPPED:
                    logger.info("FFmpeg is stopped")
                    break
                elif state != FFmpegState.RUNNING:
                    await asyncio.sleep(0.1)
                    continue

                current_time = time()
                elapsed_time = max(0.0, current_time - beg)
                buffer_size = max(int(32000 * elapsed_time), 4096)  # dynamic read
                beg = current_time

                chunk = await self.ffmpeg_manager.read_data(buffer_size)
                if not chunk:
                    # No data currently available
                    await asyncio.sleep(0.05)
                    continue

                self.pcm_buffer.extend(chunk)
                await self.handle_pcm_data()

            except asyncio.CancelledError:
                logger.info("ffmpeg_stdout_reader cancelled.")
                break
            except Exception as e:
                logger.warning(f"Exception in ffmpeg_stdout_reader: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.2)

        logger.info("FFmpeg stdout processing finished. Signaling downstream processors if needed.")
        if self.transcription_queue:
            await self.transcription_queue.put(SENTINEL)
        if self.diarization:
            await self.diarization_queue.put(SENTINEL)
        if self.translation:
            await self.translation_queue.put(SENTINEL)

    async def _enqueue_batch_window(self, window_start_ms: int, window_end_ms: int, reason: str) -> None:
        """Stap 1: enqueue alleen metadata (nog geen decode)."""
        job = {
            "job_id": str(uuid.uuid4()),
            "start_ms": int(window_start_ms),
            "end_ms": int(window_end_ms),
            "reason": reason,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        
        state_end_buffer_ms = int(round(float(self.state.end_buffer) * 1000.0)) if getattr(self, "state", None) else None
        last_tok_end_ms = None
        try:
            if self.state and self.state.tokens:
                last_tok_end_ms = int(round(float(self.state.tokens[-1].end) * 1000.0))
        except Exception:
            pass

        await self._batch_queue.put(job)

        logger.info(
            f"[BATCH][ENQUEUE][DBG] job_id={job['job_id']} "
            f"window={job['start_ms']}..{job['end_ms']}ms "
            f"len={(job['end_ms']-job['start_ms'])/1000.0:.2f}s reason={reason} "
            f"state_end_buffer_ms={state_end_buffer_ms} last_tok_end_ms={last_tok_end_ms}"
        )
        logger.info(f"[BATCH][QUEUE] size={self._batch_queue.qsize()}")


    def _get_last_speech_end_ms(self, fallback_ms: int) -> int:
        """Window-end moet einde van spraak zijn (niet einde van stilte)."""
        if self.state.tokens:
            try:
                return int(round(float(self.state.tokens[-1].end) * 1000.0))
            except Exception:
                pass
        return int(fallback_ms)

    async def _batch_on_silence_boundary(self, stream_time_ms: int, boundary: str) -> None:
        """
        Stap 1: 30s hard-grens, stilte is afrondingshint.
        We proberen te sluiten op een stilte-boundary:
        - boundary = "silence_start"  (preferred)
        - boundary = "silence_end"    (fallback)
        """

        if self._batch_window_start_ms is None:
            # Batch window starts at t=0 for this session (wordt ook gezet in process_audio)
            self._batch_window_start_ms = 0
            self._batch_last_close_end_ms = None
            self._batch_ready_to_close = False
            self._batch_ready_at_ms = None

        window_start_ms = int(self._batch_window_start_ms)

        # Einde van spraak is leidend
        window_end_ms = self._get_last_speech_end_ms(fallback_ms=stream_time_ms)
        window_len_ms = window_end_ms - window_start_ms

        # 1) Markeer ready zodra target gehaald is
        if (not self._batch_ready_to_close) and (window_len_ms >= BATCH_TARGET_WINDOW_MS):
            self._batch_ready_to_close = True
            self._batch_ready_at_ms = window_end_ms
            logger.info(
                f"[BATCH][WINDOW] ready_to_close=True at {window_end_ms}ms "
                f"(start={window_start_ms}ms len={window_len_ms/1000.0:.2f}s)"
            )

        # 2) Sluit alleen als ready_to_close
        if not self._batch_ready_to_close:
            return

        # Guard: voorkom dubbele close op zelfde end
        if self._batch_last_close_end_ms is not None and window_end_ms <= self._batch_last_close_end_ms:
            logger.info(
                f"[BATCH][WINDOW] skip duplicate close at {window_end_ms}ms "
                f"(last_close_end_ms={self._batch_last_close_end_ms})"
            )
            return

        # Enqueue alleen als window ook echt lang genoeg is
        if window_len_ms < BATCH_TARGET_WINDOW_MS:
            logger.info(
                f"[BATCH][WINDOW] ready_to_close but len too short: {window_len_ms/1000.0:.2f}s"
            )
            return

        self._batch_last_close_end_ms = window_end_ms

        await self._enqueue_batch_window(
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            reason=boundary
        )

        # Advance window
        self._batch_window_start_ms = window_end_ms
        self._batch_ready_to_close = False
        self._batch_ready_at_ms = None

    async def _flush_final_batch_tail(self, reason: str = "end_of_stream") -> None:
        """
        Forceer bij EOF/stop nog één laatste batch-window voor resterende audio
        die nog niet via silence_start is afgesloten.
        """
        try:
            if self._batch_window_start_ms is None:
                return

            window_start_ms = int(self._batch_window_start_ms)

            # Neem het meest betrouwbare einde van de al verwerkte audio
            state_end_buffer_ms = int(round(float(self.state.end_buffer) * 1000.0)) if getattr(self, "state", None) else 0
            last_speech_end_ms = self._get_last_speech_end_ms(fallback_ms=state_end_buffer_ms)

            # Kies speech-end als die er is, anders de buffer-end
            window_end_ms = max(last_speech_end_ms, state_end_buffer_ms)

            if window_end_ms <= window_start_ms:
                logger.info(
                    f"[BATCH][FINALFLUSH] skip empty tail: "
                    f"start={window_start_ms} end={window_end_ms}"
                )
                return

            # Guard tegen dubbel enqueue van exact hetzelfde einde
            if self._batch_last_close_end_ms is not None and window_end_ms <= self._batch_last_close_end_ms:
                logger.info(
                    f"[BATCH][FINALFLUSH] skip duplicate tail: "
                    f"start={window_start_ms} end={window_end_ms} "
                    f"last_close_end_ms={self._batch_last_close_end_ms}"
                )
                return

            window_len_ms = window_end_ms - window_start_ms

            # Voor EOF niet te streng zijn: liever een korte tail meenemen dan verliezen
            if window_len_ms < 300:
                logger.info(
                    f"[BATCH][FINALFLUSH] skip very short tail: "
                    f"start={window_start_ms} end={window_end_ms} len={window_len_ms}ms"
                )
                return

            self._batch_last_close_end_ms = window_end_ms

            await self._enqueue_batch_window(
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                reason=reason,
            )

            logger.info(
                f"[BATCH][FINALFLUSH] enqueued tail window "
                f"{window_start_ms}..{window_end_ms}ms len={window_len_ms/1000.0:.2f}s"
            )

            # Advance zodat cleanup/latere stop niet opnieuw hetzelfde stukje pakt
            self._batch_window_start_ms = window_end_ms
            self._batch_ready_to_close = False
            self._batch_ready_at_ms = None

        except Exception as e:
            logger.warning(f"[BATCH][FINALFLUSH] failed: {e}")

    async def _pause_flush(self) -> None:
        """Client-side Pauze: sluit het huidige live-segment + batch-venster netjes
        af (zelfde bouwstenen als de stop-sequentie in process_audio), maar zonder
        is_stopping te zetten, de WAV te sluiten of de queues/ffmpeg te stoppen --
        de sessie blijft volledig leven voor Hervatten.

        Reset ook de live-decoder: een pauze is een echte discontinuïteit in de
        audio (de microfoon staat écht stil), en zonder reset probeert de decoder
        na hervatten door te decoderen met verouderde interne context alsof de
        audio doorlopend was -- geobserveerd als een woordherhaling-lus ("Evet
        Evet Evet...") die kort na hervatten begint. batch_groups (de al
        opgebouwde transcript-historie) blijven ongemoeid; de in-flight
        decoder-hypothese wordt gewist, en de nog niet gevalideerde live-regel
        (current_line_tokens) wordt afgesloten in validated_segments in plaats
        van losjes te blijven hangen -- zie flush_current_line()."""
        if self.pcm_buffer:
            await self.handle_pcm_data()
        if self.current_silence:
            try:
                await self._end_silence()
            except Exception as e:
                logger.warning(f"[PAUSE][FLUSH] ending current silence failed: {e}")

        # Wacht tot transcription_processor() alle reeds ingeklede audio daadwerkelijk
        # heeft verwerkt, vóórdat we hieronder state.end_buffer/state.tokens lezen.
        # transcription_processor draait als losstaande achtergrondtaak en kan
        # achterlopen (geobserveerd: 40+ seconden vertraging na meerdere resets in
        # één sessie) -- zonder deze wachtstap lazen _flush_final_batch_tail en de
        # tijdstempel-correctie in _hard_reset_live_decoder een verouderde positie,
        # met als gevolg dat er geen batch-venster meer tussentijds sloot (pas de
        # eind-flush bij Stop ving het op). Timeout als noodrem: liever doorgaan met
        # een mogelijk-verouderde waarde dan de sessie laten vastlopen.
        #
        # LET OP: get_all_from_queue() roept task_done() al aan zodra een item van de
        # queue is AFGEHAALD, niet nadat het daadwerkelijk verwerkt is (process_iter()
        # loopt daarna nog, op een asyncio.to_thread-achtergrondthread) -- join() alleen
        # garandeert dus NIET dat een lopende decode-stap al klaar is. Bevestigd via
        # [DIAG][END_BUFFER_JUMP]-logging (2026-07-19): een tokentijdstempel kon 6+
        # seconden hoger uitvallen dan de echte audio-positie, exact enkele seconden na
        # een pauze-reset -- consistent met een in-flight decode-stap die nog met de OUDE
        # cumulative_time_offset bezig was toen _hard_reset_live_decoder() hieronder al
        # resette, en zijn resultaat er pas ná de reset kwam invallen. Vandaar de losse
        # wachtstap op _decode_in_progress hieronder, specifiek voor deze race.
        try:
            if self.transcription_queue:
                await asyncio.wait_for(self.transcription_queue.join(), timeout=20.0)
        except asyncio.TimeoutError:
            logger.warning(
                f"[PAUSE][FLUSH] channel={self.channel_id} transcription_queue.join() "
                f"timed out na 20s, ga door met mogelijk verouderde state"
            )

        _decode_wait_start = time()
        while self._decode_in_progress and (time() - _decode_wait_start) < 5.0:
            await asyncio.sleep(0.02)
        if self._decode_in_progress:
            logger.warning(
                f"[PAUSE][FLUSH] channel={self.channel_id} in-flight decode-stap na 5s "
                f"nog niet klaar, ga door (mogelijk resterende drift)"
            )

        # Sluit de lopende, nog niet gevalideerde live-regel af als eigen FINAL
        # segment -- anders overleeft current_line_tokens de hard reset hieronder
        # (die raakt alleen state.buffer_transcription) en plakken tokens van na
        # het hervatten er zonder afsluiting achteraan, sessie na sessie verder
        # groeiend. Zie flush_current_line() in tokens_alignment.py.
        try:
            self.tokens_alignment.flush_current_line()
        except Exception as e:
            logger.warning(f"[PAUSE][FLUSH] flush_current_line failed: {e}")

        await self._flush_final_batch_tail(reason="pause")
        try:
            self._hard_reset_live_decoder(reason="pause")
        except Exception as e:
            logger.warning(f"[PAUSE][FLUSH] live decoder reset failed: {e}")
        logger.info(f"[PAUSE][FLUSH] channel={self.channel_id} session paused, segment/window flushed")

    async def transcription_processor(self) -> None:
        """Process audio chunks for transcription."""
        cumulative_pcm_duration_stream_time = 0.0
        
        while True:
            try:
                # item = await self.transcription_queue.get()
                item = await get_all_from_queue(self.transcription_queue)
                if item is SENTINEL:
                    logger.debug("Transcription processor received sentinel. Finishing.")
                    break

                asr_internal_buffer_duration_s = len(getattr(self.transcription, 'audio_buffer', [])) / self.transcription.SAMPLING_RATE
                transcription_lag_s = max(0.0, time() - self.beg_loop - self.state.end_buffer)
                asr_processing_logs = f"internal_buffer={asr_internal_buffer_duration_s:.2f}s | lag={transcription_lag_s:.2f}s |"
                stream_time_end_of_current_pcm = cumulative_pcm_duration_stream_time
                new_tokens = []
                current_audio_processed_upto = self.state.end_buffer

                if isinstance(item, Silence):
                    if item.is_starting:
                        # Begin van stilte → ASR informeren
                        try:
                            # start_silence() roept intern process_iter(is_last=True) aan --
                            # dezelfde potentieel trage GPU-decode als het hoofdpad hieronder,
                            # maar zonder diens asyncio.wait_for(timeout=15.0)-guard. Zelfde
                            # _decode_in_progress(_since)-markering zodat results_formatter()
                            # (blijft onafhankelijk tikken) een hang hier ook kan signaleren.
                            self._decode_in_progress = True
                            self._decode_in_progress_since = time()
                            _diag_start_silence_t0 = time()
                            try:
                                new_tokens, current_audio_processed_upto = await asyncio.to_thread(
                                    self.transcription.start_silence
                                )
                                logger.info(
                                    f"[DIAG][LIVE_PERF] start_silence_duration="
                                    f"{time() - _diag_start_silence_t0:.3f}s"
                                )
                            finally:
                                self._decode_in_progress = False
                                self._decode_in_progress_since = None
                            asr_processing_logs += f" + Silence starting"
                        except Exception as e:
                            logger.warning(f"[LIVE] start_silence failed: {e}")
                            new_tokens = []
                        # ===== Stap 1: batch windowing proberen te sluiten op silence-start =====
                        # Loopt altijd, ook als start_silence hierboven faalde.
                        try:
                            stream_time_ms = int(round(cumulative_pcm_duration_stream_time * 1000.0))
                            await self._batch_on_silence_boundary(
                                stream_time_ms=stream_time_ms,
                                boundary="silence_start"
                            )
                        except Exception as e:
                            logger.warning(f"[BATCH][WINDOW] silence-start handler failed: {e}")

                    if item.has_ended:
                        # Einde van stilte
                        asr_processing_logs += f" + Silence of = {item.duration:.2f}s"
                        cumulative_pcm_duration_stream_time += item.duration
                        current_audio_processed_upto = cumulative_pcm_duration_stream_time

                        # Laat het lopende segment netjes afsluiten
                        self.transcription.end_silence(
                            item.duration,
                            self.state.tokens[-1].end if self.state.tokens else 0
                        )

                        # ===== Stap 1: batch windowing sluiten op silence-end (los van decoder reset) ===== 
                        try:
                            stream_time_ms = int(round(cumulative_pcm_duration_stream_time * 1000.0))
                            await self._batch_on_silence_boundary(
                                stream_time_ms=stream_time_ms,
                                boundary="silence_end"
                           )
                        except Exception as e:
                           logger.warning(f"[BATCH][WINDOW] silence-end handler failed: {e}")

                        # Voor nu géén harde live reset op normale stilte.
                        # We willen de doorlopende live transcriptie intact houden
                        # en batch later laten corrigeren.
                        if item.duration >= SILENCE_RESET_THRESHOLD:
                            logger.info(
                                f"[Decoder reset] skip hard reset after silence "
                                f"{item.duration:.2f}s (threshold={SILENCE_RESET_THRESHOLD}s)"
                            )

                    if self.state.tokens:
                        asr_processing_logs += f" | last_end = {self.state.tokens[-1].end} |"

                    self._log_status_throttled(asr_processing_logs)
                    new_tokens = new_tokens or []
                    current_audio_processed_upto = max(
                        current_audio_processed_upto,
                        stream_time_end_of_current_pcm
                    )
                elif isinstance(item, ChangeSpeaker):
                    self.transcription.new_speaker(item)
                    continue
                elif isinstance(item, np.ndarray):
                    pcm_array = item
                    self._log_status_throttled(asr_processing_logs)
                    cumulative_pcm_duration_stream_time += len(pcm_array) / self.sample_rate
                    stream_time_end_of_current_pcm = cumulative_pcm_duration_stream_time

                    # 🔹 HARD CAP = alleen windowing, nooit ASR-gating
                    if ENABLE_HARD_CAP:
                        try:
                            if self._batch_window_start_ms is not None:
                                now_ms = int(round(stream_time_end_of_current_pcm * 1000.0))
                                if (now_ms - int(self._batch_window_start_ms)) >= BATCH_HARD_CAP_MS:
                                    window_start_ms = int(self._batch_window_start_ms)
                                    window_end_ms = now_ms

                                    if self._batch_last_close_end_ms is None or window_end_ms > self._batch_last_close_end_ms:
                                        self._batch_last_close_end_ms = window_end_ms
                                        await self._enqueue_batch_window(
                                            window_start_ms=window_start_ms,
                                            window_end_ms=window_end_ms,
                                            reason="hard_cap_close"
                                        )

                                    self._batch_window_start_ms = window_end_ms
                                    self._batch_ready_to_close = False
                                    self._batch_ready_at_ms = None
                        except Exception as e:
                            logger.warning(f"[BATCH][WINDOW] hard-cap handler failed: {e}")

                    # ✅ ASR MAG ALLEEN HIER
                    self.transcription.insert_audio_chunk(pcm_array, stream_time_end_of_current_pcm)

                    self._decode_in_progress = True
                    self._decode_in_progress_since = time()
                    _diag_process_iter_start = time()  # [DIAG][LIVE_PERF], zie live-lag-onderzoek
                    try:
                        new_tokens, current_audio_processed_upto = await asyncio.wait_for(
                            asyncio.to_thread(self.transcription.process_iter),
                            timeout=15.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("[LIVE][TIMEOUT] process_iter exceeded 15s — forcing decoder reset")
                        await asyncio.to_thread(self._hard_reset_live_decoder, "process_iter_timeout")
                        new_tokens = []
                        current_audio_processed_upto = self.state.end_buffer
                    finally:
                        self._decode_in_progress = False
                        self._decode_in_progress_since = None
                        # Puur observerend (2026-08-23, onderzoek groeiende live-lag) --
                        # rechtstreeks te correleren met de al bestaande lag-regel
                        # hierboven (transcription_lag_s) om te zien of de lag-groei
                        # samenvalt met een groeiende process_iter-duur.
                        logger.info(
                            f"[DIAG][LIVE_PERF] process_iter_duration={time() - _diag_process_iter_start:.3f}s "
                            f"lag={transcription_lag_s:.2f}s"
                        )

                    new_tokens = new_tokens or []

                # Tweede lek gevonden (2026-07-19, na eerste end_buffer-klem):
                # _get_last_speech_end_ms() (gebruikt door _flush_final_batch_tail/
                # _batch_on_silence_boundary om batch-vensters af te bakenen) leest
                # self.state.tokens[-1].end RECHTSTREEKS, buiten end_buffer om. Een
                # hallucinerend token met een te hoge attention-afgeleide tijdstempel
                # (zelfde fenomeen als de end_buffer-klem verderop afvangt) kon zo
                # alsnog een batch-venster ver voorbij de echte audio laten beginnen --
                # bevestigd via [DIAG][WAV_DRIFT] vlak na een reeks succesvolle
                # [DIAG][END_BUFFER_CLAMP]-regels. Klem daarom hier al, per token, vóór
                # ze aan state.tokens worden toegevoegd -- op het gedeelde punt na beide
                # takken (Silence en np.ndarray kunnen allebei new_tokens opleveren).
                _token_ceiling = current_audio_processed_upto + END_BUFFER_MAX_LOOKAHEAD_S
                for _tok in new_tokens:
                    _tok_end = getattr(_tok, "end", None)
                    if _tok_end is not None and _tok_end > _token_ceiling:
                        logger.warning(
                            f"[DIAG][TOKEN_CLAMP] channel={self.channel_id} "
                            f"token.end {_tok_end:.2f}s geklemd naar {_token_ceiling:.2f}s "
                            f"(text={getattr(_tok, 'text', '?')!r})"
                        )
                        _tok.end = _token_ceiling
                        _tok_start = getattr(_tok, "start", None)
                        if _tok_start is not None and _tok_start > _token_ceiling:
                            _tok.start = _token_ceiling

                _buffer_transcript = self.transcription.get_buffer()
                buffer_text = (_buffer_transcript.text or "").strip()

                # Laat buffer_transcription intact.
                # We willen de volledige lopende hypothese in de GUI tonen,
                # niet alleen het restant na aftrek van already-committed tokens.

                # Live zichtbaar houden:
                # alleen pure rommel/single-char junk onderdrukken,
                # niet normale voorlopige partials die batch later toch overschrijft.
                if buffer_text:
                    junk_only = (
                        len(buffer_text) <= 1 or
                        buffer_text in {'"', "'", ",", ".", "?", "!"}
                    )
                    if junk_only:
                        _buffer_transcript.text = ""
                        _buffer_transcript.start = None
                        _buffer_transcript.end = None

                # [DIAG] end_buffer-drift onderzoek (2026-07-19): kandidaten gelabeld
                # opbouwen (i.p.v. een kale lijst waarden) zodat bij een sprong meteen
                # zichtbaar is WELKE bron 'm veroorzaakte -- new_tokens[-1].end (net
                # gecommit token) en _buffer_transcript.end (nog-lopende live-hypothese,
                # via get_buffer()) zijn twee wezenlijk andere routes naar een tijdstempel.
                _diag_labeled_candidates = [("state.end_buffer", self.state.end_buffer)]
                if new_tokens:
                    _diag_labeled_candidates.append(("new_tokens[-1].end", new_tokens[-1].end))
                if _buffer_transcript.end is not None:
                    _diag_labeled_candidates.append(("buffer_transcript.end", _buffer_transcript.end))
                _diag_labeled_candidates.append(("current_audio_processed_upto", current_audio_processed_upto))

                candidate_end_times = [v for _, v in _diag_labeled_candidates]

                # end_buffer is een eenrichtings-max -- als een kandidaat hier significant
                # hoger uitvalt dan current_audio_processed_upto (de zuivere, betrouwbare
                # PCM-teller), is dát de kandidaat die de drift injecteert. Alleen loggen
                # bij een echte sprong om logspam te vermijden.
                _diag_new_end = max(candidate_end_times)
                if _diag_new_end - self.state.end_buffer > 5.0 and _diag_new_end > current_audio_processed_upto + 5.0:
                    _diag_winner_name, _diag_winner_val = max(_diag_labeled_candidates, key=lambda kv: kv[1])
                    _diag_last_tok_info = "n/a"
                    if new_tokens:
                        _lt = new_tokens[-1]
                        _diag_last_tok_info = (
                            f"start={getattr(_lt, 'start', '?')} end={getattr(_lt, 'end', '?')} "
                            f"text={getattr(_lt, 'text', '?')!r}"
                        )
                    logger.warning(
                        f"[DIAG][END_BUFFER_JUMP] channel={self.channel_id} "
                        f"end_buffer {self.state.end_buffer:.2f}s -> {_diag_new_end:.2f}s "
                        f"BRON={_diag_winner_name}={_diag_winner_val:.2f}s "
                        f"(alle kandidaten={[(n, round(v, 2)) for n, v in _diag_labeled_candidates]}, "
                        f"buffer_text={buffer_text[:60]!r}, new_tokens_count={len(new_tokens)}, "
                        f"laatste_new_token=[{_diag_last_tok_info}])"
                    )

                # Begrens end_buffer tot wat er daadwerkelijk aan PCM is aangeboden (+ marge).
                # new_tokens[-1].end/buffer_transcript.end zijn attention-afgeleide tijdstempels
                # die bij een laag-vertrouwen/hallucinerend token (bv. een los " *"-token,
                # gezien in [DIAG][END_BUFFER_JUMP]-logs) een willekeurig verkeerd encoder-frame
                # kunnen kiezen -- current_audio_processed_upto is een zuivere PCM-teller
                # (letterlijk hoeveel audio is aangeboden) en kan dat per constructie niet.
                # We proberen de hallucinatie zelf niet te voorkomen (blijft, zoals bij elk
                # Whisper-model, incidenteel gebeuren) -- alleen te voorkomen dat zo'n
                # incidenteel fout token de gedeelde end_buffer-status vergiftigt, wat
                # bevestigd tot batch-vensters buiten de WAV leidde ([DIAG][WAV_DRIFT]) en
                # daarmee tot permanent verlies van transcript-inhoud.
                _end_buffer_ceiling = current_audio_processed_upto + END_BUFFER_MAX_LOOKAHEAD_S
                if _diag_new_end > _end_buffer_ceiling:
                    logger.warning(
                        f"[DIAG][END_BUFFER_CLAMP] channel={self.channel_id} "
                        f"{_diag_new_end:.2f}s geklemd naar {_end_buffer_ceiling:.2f}s "
                        f"(current_audio_processed_upto={current_audio_processed_upto:.2f}s)"
                    )
                    _diag_new_end = _end_buffer_ceiling

                async with self.lock:
                    self.state.tokens.extend(new_tokens)
                    self.state.buffer_transcription = _buffer_transcript
                    self.state.end_buffer = _diag_new_end
                    self.state.new_tokens.extend(new_tokens)
                    self.state.new_tokens_buffer = _buffer_transcript

                    if len(self.state.tokens) > _STATE_TOKENS_PRUNE_TRIGGER:
                        _cutoff = self.state.tokens[-1].end - _STATE_TOKENS_MAX_DURATION_S
                        if _cutoff > 0.0:
                            self.state.tokens = [
                                t for t in self.state.tokens if t.end >= _cutoff
                            ]

                if self.translation_queue:
                    for token in new_tokens:
                        await self.translation_queue.put(token)                
            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")

                # HARD recovery voor live decoder zodat de pipeline niet in corrupte state blijft hangen
                try:
                    self._hard_reset_live_decoder(reason=f"transcription_exception:{type(e).__name__}")
                except Exception as reset_e:
                    logger.warning(f"[LIVE][HARD_RESET] failed after transcription exception: {reset_e}")

                new_tokens = []
                current_audio_processed_upto = max(
                    self.state.end_buffer,
                    cumulative_pcm_duration_stream_time
                )

                continue
        
        if self.is_stopping:
            logger.info("Transcription processor finishing due to stopping flag.")
            if self.diarization_queue:
                await self.diarization_queue.put(SENTINEL)
            if self.translation_queue:
                await self.translation_queue.put(SENTINEL)

        logger.info("Transcription processor task finished.")

    def assign_speaker_to_tokens(self, speaker_segments, tokens):
        """
        Assigns speaker labels to ASR tokens based on maximum time overlap.
        Mutates tokens in-place.
        """
        MIN_OVERLAP = 0.08  # 80ms, tweakbaar

        for tok in tokens:
            best_speaker = None
            best_overlap = 0.0
            
            for seg in speaker_segments:
                overlap_start = max(tok.start, seg.start)
                overlap_end = min(tok.end, seg.end)
                overlap = overlap_end - overlap_start

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = seg.speaker

            if best_speaker is not None and best_overlap >= MIN_OVERLAP:
                tok.speaker = best_speaker

    async def diarization_processor(self) -> None:
        while True:
            try:
                item = await get_all_from_queue(self.diarization_queue)
                if item is SENTINEL:
                    break
                elif type(item) is Silence:
                    if item.has_ended:
                        self.diarization.insert_silence(item.duration)
                    continue

                self.diarization.insert_audio_chunk(item)
                diarization_segments = await self.diarization.diarize()

                async with self.lock:
                    self.state.new_diarization = diarization_segments

                    if diarization_segments:
                        asr_tokens = [t for t in self.state.tokens if hasattr(t, "start")]
                        if asr_tokens:
                            self.assign_speaker_to_tokens(diarization_segments, asr_tokens)

                        self.state.end_attributed_speaker = max(
                            self.state.end_attributed_speaker,
                            max(seg.end for seg in diarization_segments)
                        )

                
            except Exception as e:
                logger.warning(f"Exception in diarization_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
        logger.info("Diarization processor task finished.")

    async def translation_processor(self) -> None:
        # the idea is to ignore diarization for the moment. We use only transcription tokens. 
        # And the speaker is attributed given the segments used for the translation
        # in the future we want to have different languages for each speaker etc, so it will be more complex.
        while True:
            try:
                item = await get_all_from_queue(self.translation_queue)
                if item is SENTINEL:
                    logger.debug("Translation processor received sentinel. Finishing.")
                    break
                elif type(item) is Silence:
                    if item.is_starting:
                        new_translation, new_translation_buffer = self.translation.validate_buffer_and_reset()
                    if item.has_ended:
                        self.translation.insert_silence(item.duration)
                        continue
                elif isinstance(item, ChangeSpeaker):
                    new_translation, new_translation_buffer = self.translation.validate_buffer_and_reset()
                    pass
                else:
                    self.translation.insert_tokens(item)
                    new_translation, new_translation_buffer = await asyncio.to_thread(self.translation.process)
                async with self.lock:
                    self.state.new_translation.append(new_translation)
                    self.state.new_translation_buffer = new_translation_buffer
            except Exception as e:
                logger.warning(f"Exception in translation_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
        logger.info("Translation processor task finished.")

    async def results_formatter(self) -> AsyncGenerator[Union[FrontData, SegmentUpdate], None]:
        """Format processing results for output."""
        LIVE_STALL_THRESHOLD_S = 8.0
        LIVE_STALL_LOG_INTERVAL_S = 5.0
        while True:
            try:
                # [DIAG][LIVE_STALL] onderzoek (2026-08-23): results_formatter draait als
                # eigen asyncio-task en blijft -- zoals bevestigd in een eerder log, waar
                # deze FRONTDATA_PUSH-regels bleven doorlopen terwijl de GUI volledig
                # bevroor -- onafhankelijk tikken ook als transcription_processor() zelf
                # muurvast zit op een niet van een timeout voorziene decode-aanroep (bv.
                # start_silence()). Precies daarom is dit de juiste plek om zo'n hang te
                # signaleren: alleen deze task kan het nog waarnemen.
                _since = self._decode_in_progress_since
                if _since is not None:
                    _stalled_for = time() - _since
                    if _stalled_for >= LIVE_STALL_THRESHOLD_S and \
                       (time() - self._live_stall_last_logged_t) >= LIVE_STALL_LOG_INTERVAL_S:
                        self._live_stall_last_logged_t = time()
                        logger.warning(
                            f"[DIAG][LIVE_STALL] channel={self.channel_id} decode al "
                            f"{_stalled_for:.1f}s bezig zonder terug te keren "
                            f"(process_iter of start_silence hangt mogelijk)"
                        )

                # results_formatter loop - first flush pending segment updates
                while not self._ws_update_queue.empty():
                    upd = await self._ws_update_queue.get()
                    self._ws_update_queue.task_done()
                    yield upd

                if self._ffmpeg_error:
                    yield FrontData(status="error", error=f"FFmpeg error: {self._ffmpeg_error}")
                    self._ffmpeg_error = None
                    await asyncio.sleep(1)
                    continue

                self.tokens_alignment.update()
                lines, buffer_diarization_text, buffer_translation_text = self.tokens_alignment.get_lines(
                    diarization=self.args.diarization,
                    translation=bool(self.translation),
                    current_silence=self.current_silence
                )
                state = await self.get_current_state()

                buffer_transcription_text = state.buffer_transcription.text if state.buffer_transcription else ''

                response_status = "active_transcription"
                if not lines and not buffer_transcription_text and not buffer_diarization_text:
                    response_status = "no_audio_detected"

                response = FrontData(
                    status=response_status,
                    lines=lines,
                    buffer_transcription=buffer_transcription_text,
                    buffer_diarization=buffer_diarization_text,
                    buffer_translation=buffer_translation_text,
                    remaining_time_transcription=state.remaining_time_transcription,
                    remaining_time_diarization=state.remaining_time_diarization if self.args.diarization else 0,
                    session_id=self.session_id,
                    channel_id=self.channel_id,
                )
                                
                should_push = (response != self.last_response_content)
                if should_push:
                    if logger.isEnabledFor(logging.DEBUG):
                        # [DIAG] client-side "spooklijn"-onderzoek (2026-07-19): bevestigt of een
                        # bijgewerkte front_data-snapshot (bv. na apply_batch_group() dat een
                        # gehallucineerde validated_segment liet vallen) daadwerkelijk de WS uit
                        # gaat, en met welke line-ids -- zonder dit is niet te zien of een
                        # verdwenen regel de server nooit verliet, of alsnog client-side bleef
                        # hangen ondanks een correcte push.
                        _diag_line_ids = [getattr(l, "id", None) for l in lines]
                        logger.debug(
                            f"[DIAG][FRONTDATA_PUSH] channel={self.channel_id} n_lines={len(lines)} "
                            f"ids={_diag_line_ids}"
                        )
                    yield response
                    self.last_response_content = response
                
                if self.is_stopping and self._processing_tasks_done():
                    logger.info("Results formatter: All upstream processors are done and in stopping state. Terminating.")
                    return
                
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.warning(f"Exception in results_formatter. Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.5)
        
    async def create_tasks(self) -> AsyncGenerator[FrontData, None]:
        """Create and start processing tasks."""
        self.all_tasks_for_cleanup = []
        processing_tasks_for_watchdog: List[asyncio.Task] = []

        # If using FFmpeg (non-PCM input), start it and spawn stdout reader
        if not self.is_pcm_input:
            success = await self.ffmpeg_manager.start()
            if not success:
                logger.error("Failed to start FFmpeg manager")
                async def error_generator() -> AsyncGenerator[FrontData, None]:
                    yield FrontData(
                        status="error",
                        error="FFmpeg failed to start. Please check that FFmpeg is installed."
                    )
                return error_generator()
            self.ffmpeg_reader_task = asyncio.create_task(self.ffmpeg_stdout_reader())
            self.all_tasks_for_cleanup.append(self.ffmpeg_reader_task)
            processing_tasks_for_watchdog.append(self.ffmpeg_reader_task)

        if self.transcription:
            self.transcription_task = asyncio.create_task(self.transcription_processor())
            self.all_tasks_for_cleanup.append(self.transcription_task)
            processing_tasks_for_watchdog.append(self.transcription_task)
            
        if self.diarization:
            self.diarization_task = asyncio.create_task(self.diarization_processor())
            self.all_tasks_for_cleanup.append(self.diarization_task)
            processing_tasks_for_watchdog.append(self.diarization_task)
        
        if self.translation:
            self.translation_task = asyncio.create_task(self.translation_processor())
            self.all_tasks_for_cleanup.append(self.translation_task)
            processing_tasks_for_watchdog.append(self.translation_task)
        
        # ===== Batch worker (Stap 2) =====
        if self._batch_worker_task is None or self._batch_worker_task.done():
            self._batch_worker_task = asyncio.create_task(self._batch_worker())
            self.all_tasks_for_cleanup.append(self._batch_worker_task)
            # Zonder dit blijft een onverwacht gestorven batch-worker (bv. stilzwijgend
            # gecancelled) volledig onopgemerkt: nieuwe batch-taken stapelen zich dan op
            # in _batch_queue zonder ooit verwerkt te worden, en zonder één regel log die
            # dat verklaart. De waakhond bestond al voor de andere achtergrondtaken
            # (transcriptie/diarisatie/vertaling, zie hierboven) -- de batch-worker hoorde
            # daar ook al bij.
            processing_tasks_for_watchdog.append(self._batch_worker_task)

        # Monitor overall system health
        self.watchdog_task = asyncio.create_task(self.watchdog(processing_tasks_for_watchdog))
        self.all_tasks_for_cleanup.append(self.watchdog_task)

        return self.results_formatter()

    async def watchdog(self, tasks_to_monitor: List[asyncio.Task]) -> None:
        """Monitors the health of critical processing tasks."""
        tasks_remaining: List[asyncio.Task] = [task for task in tasks_to_monitor if task]
        while True:
            try:
                if not tasks_remaining:
                    logger.info("Watchdog task finishing: all monitored tasks completed.")
                    return

                await asyncio.sleep(10)
                
                for i, task in enumerate(list(tasks_remaining)):
                    if task.done():
                        task_name = task.get_name() if hasattr(task, 'get_name') else f"Monitored Task {i}"
                        # task.exception() gooit zelf CancelledError als de taak
                        # geannuleerd was -- dat zou hier ongevangen naar de
                        # except asyncio.CancelledError hieronder lekken, die dan
                        # denkt dat de WAAKHOND zelf geannuleerd is (en stopt met
                        # alles monitoren, stilzwijgend). Eerst cancelled() checken
                        # (nooit-gooiend) voorkomt die misattributie.
                        if task.cancelled():
                            logger.error(f"{task_name} was cancelled unexpectedly.")
                        else:
                            exc = task.exception()
                            if exc:
                                logger.error(f"{task_name} unexpectedly completed with exception: {exc}")
                            else:
                                logger.info(f"{task_name} completed normally.")
                        tasks_remaining.remove(task)
                    
            except asyncio.CancelledError:
                logger.info("Watchdog task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in watchdog task: {e}", exc_info=True)

    def _ensure_wav_open(self) -> None:
        if self._wav_writer is not None:
            return

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"session_{self.session_id}_{self.channel_id}_{ts}.wav"
        self._wav_path = self.recordings_dir / filename

        wf = wave.open(str(self._wav_path), "wb")
        wf.setnchannels(self.channels)        # 1
        wf.setsampwidth(self.bytes_per_sample) # 2 bytes (int16)
        wf.setframerate(self.sample_rate)     # 16000 Hz
        self._wav_writer = wf

        logger.info(f"[SESSION WAV] Recording to {self._wav_path}")

    def _close_wav(self) -> None:
        if self._wav_writer is None:
            return
        try:
            self._wav_writer.close()
            logger.info(f"[SESSION WAV] Closed {self._wav_path}")
        except Exception as e:
            logger.warning(f"[SESSION WAV] Error closing WAV: {e}")
        finally:
            self._wav_writer = None

    def _flush_wav(self) -> None:
        try:
            wf = self._wav_writer
            if wf is None:
                return
            f = getattr(wf, "_file", None)
            if f:
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
        except Exception:
            pass

    def _read_wav_slice_float32(
        self, start_ms: int, end_ms: int
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Returns (audio_f32, actual_end_ms). actual_end_ms is the real end of the
        slice that was read, in the same ms timebase as the requested start_ms/end_ms --
        equal to end_ms normally, but less than end_ms when the WAV on disk didn't yet
        reach that far (see the [DIAG][WAV_CLIP] branch below). Callers MUST use
        actual_end_ms (not the requested end_ms) for any downstream bookkeeping that
        claims "this range has been covered" -- otherwise a window that got silently
        clipped here still gets marked as fully processed, and the un-decoded tail is
        later pruned from the live transcript with nothing to replace it."""
        if not getattr(self, "_wav_path", None):
            return None, None
        if end_ms <= start_ms:
            return None, None

        self._flush_wav()

        start_frame = int((start_ms / 1000.0) * self.sample_rate)
        end_frame   = int((end_ms   / 1000.0) * self.sample_rate)
        n_frames    = max(0, end_frame - start_frame)
        if n_frames <= 0:
            return None, None

        try:
            with wave.open(str(self._wav_path), "rb") as rf:
                actual_frames = rf.getnframes()

                if start_frame >= actual_frames:
                    # Venster begint volledig voorbij wat er daadwerkelijk is opgenomen --
                    # hier is echt niets te lezen, ongeacht welke bovenstroomse teller
                    # (end_buffer, tokentijdstempel, wat dan ook) dit venster aanvroeg.
                    logger.warning(
                        f"[DIAG][WAV_DRIFT] channel={self.channel_id} gevraagd venster begint "
                        f"bij frame {start_frame} ({start_frame / self.sample_rate:.2f}s) maar "
                        f"WAV bevat slechts {actual_frames} frames "
                        f"({actual_frames / self.sample_rate:.2f}s) -- drift van "
                        f"{(start_frame - actual_frames) / self.sample_rate:.2f}s, niets te lezen"
                    )
                    return None, None

                if end_frame > actual_frames:
                    # De kraan structureel dicht i.p.v. dweilen per bovenstroomse teller
                    # (end_buffer-klem, token-klem, ...): ongeacht WAAROM het gevraagde
                    # venster verder reikt dan wat er nu op schijf staat (hallucinatie,
                    # write-buffering-vertraging, een toekomstige, nu nog onbekende
                    # bron), knip hier gewoon af op de daadwerkelijke bestandslengte --
                    # de enige plek die nooit kan liegen, want die leest de rauwe bytes.
                    # Resultaat: een iets korter venster i.p.v. de hele batch-job
                    # verliezen (voorheen: [BATCH][SKIP], job volledig overgeslagen).
                    logger.warning(
                        f"[DIAG][WAV_CLIP] channel={self.channel_id} venster-einde "
                        f"{end_frame} ({end_frame / self.sample_rate:.2f}s) afgeknipt naar "
                        f"WAV-einde {actual_frames} ({actual_frames / self.sample_rate:.2f}s)"
                    )
                    end_frame = actual_frames
                    n_frames = max(0, end_frame - start_frame)
                    if n_frames <= 0:
                        return None, None

                rf.setpos(start_frame)
                raw = rf.readframes(n_frames)
            if not raw:
                return None, None

            audio_f32 = self.convert_pcm_to_float(raw)
            sha1 = _sha1_pcm16(audio_f32)
            actual_end_ms = int(round((end_frame / self.sample_rate) * 1000.0))

            logger.info(
                f"[BATCH][SLICE][DBG] wav={Path(self._wav_path).name} "
                f"ms={start_ms}..{end_ms} "
                f"frames={start_frame}..{end_frame} n={n_frames} "
                f"sr={self.sample_rate} sha1={sha1}"
            )

            return audio_f32, actual_end_ms
        except Exception as e:
            logger.warning(f"[BATCH] WAV slice read failed: {e}")
            return None, None

    def _batch_transcribe_text(self, audio_f32: np.ndarray) -> Optional[str]:
        try:
            if not self.batch_asr:
                return None
            txt = self.batch_asr.transcribe_text(audio_f32)
            return txt or None
        except Exception as e:
            logger.warning(f"[BATCH] transcribe failed: {e}")
            return None
        
    async def _batch_worker(self) -> None:
        """Stap 2: worker die 1 FINAL segment in window markeert met 'BATCH OK'."""
        logger.info("[BATCH][WORKER] worker started")
        while True:
            job = await self._batch_queue.get()
            try:
                if job is SENTINEL:
                    logger.info("[BATCH][WORKER] received sentinel, stopping")
                    return

                start_ms = int(job["start_ms"])
                end_ms = int(job["end_ms"])
                reason = job.get("reason", "?")
                
                attempt = int(job.get("_attempt", 0))

                # 1) Neem een consistente snapshot van 'lines' onder lock (kort!)
                async with self.lock:
                    self.tokens_alignment.update()
                    lines, _, _ = self.tokens_alignment.get_lines(
                        diarization=self.args.diarization,
                        translation=bool(self.translation),
                        current_silence=self.current_silence
                    )
                    # kopieer alleen referenties (voldoende); we muteren 'lines' hieronder niet
                    lines = list(lines)

                # 2) Vanaf hier ALLES buiten lock (loops / checks / samenvoegen)

                # We wachten alleen tot er in dit window überhaupt FINAL content is
                has_any_final = False
                for seg in lines:
                    if hasattr(seg, "is_silence") and seg.is_silence():
                        continue
                    if getattr(seg, "state", "") != "FINAL":
                        continue

                    # Determine overlap with window
                    seg_start_s = getattr(seg, "start", None)
                    seg_end_s = getattr(seg, "end", None)
                    seg_start_ms = int(round(float(seg_start_s) * 1000.0)) if seg_start_s is not None else 0
                    seg_end_ms = int(round(float(seg_end_s) * 1000.0)) if seg_end_s is not None else seg_start_ms

                    if seg_end_ms > start_ms and seg_start_ms < end_ms:
                        has_any_final = True
                        break

                if not has_any_final:
                    # We wachten NIET meer op FINAL; we gaan batch gewoon doen op het window.
                    # Live fallback blijft dan leeg / best-effort.
                    logger.warning(
                        f"[BATCH][WORKER] no FINAL content yet for job {job['job_id']} ({reason}) "
                        f"-> proceeding with batch-only (live fallback empty)"
                    )

                
                # ====== Stap 2: ECHTE batch decode + SAFE overwrite (FO) ======
                # Fallback live text: best effort window transcript from current lines (FINAL only)
                live_parts: List[str] = []
                for seg in lines:
                    if hasattr(seg, "is_silence") and seg.is_silence():
                        continue
                    if getattr(seg, "state", "") != "FINAL":
                        continue

                    seg_start_s = getattr(seg, "start", None)
                    seg_end_s = getattr(seg, "end", None)
                    seg_start_ms = int(round(float(seg_start_s) * 1000.0)) if seg_start_s is not None else 0
                    seg_end_ms = int(round(float(seg_end_s) * 1000.0)) if seg_end_s is not None else seg_start_ms

                    if not (seg_end_ms > start_ms and seg_start_ms < end_ms):
                        continue

                    t = (getattr(seg, "text", None) or "").strip()
                    if t:
                        live_parts.append(t)

                live_text = " ".join(live_parts).strip()

                # 2) Decode audio slice met padding (FO)
                decode_start_ms = max(0, start_ms - BATCH_CONTEXT_PAD_LEFT_MS)
                decode_end_ms   = end_ms + BATCH_CONTEXT_PAD_RIGHT_MS

                # Synchrone I/O (flush+fsync+wave-read) hoort niet rechtstreeks op de
                # gedeelde event loop -- _batch_worker() draait als asyncio.create_task
                # op DEZELFDE loop als transcription_processor() (live) en watchdog().
                # Een trage/hangende aanroep hier bevriest dus ALLES, niet alleen batch
                # (bevestigd: [DIAG][WAV_CLIP]-kloof liep sessielang op van 2s naar 17s,
                # gevolgd door een totale stilstand zonder exception/traceback in het log).
                _batch_perf_t0 = time()
                audio_f32, decoded_end_ms = await asyncio.to_thread(
                    self._read_wav_slice_float32, decode_start_ms, decode_end_ms
                )
                logger.info(
                    f"[DIAG][BATCH_PERF] job={job['job_id']} wav_read_duration="
                    f"{time() - _batch_perf_t0:.3f}s"
                )
                # Wat er echt gedecodeerd werd, kan korter zijn dan het geplande end_ms
                # (WAV op schijf liep nog niet bij -- zie [DIAG][WAV_CLIP]). Downstream
                # boekhouding (suppressed_ranges_ms, geëmitteerde SegmentUpdates) moet
                # hierop gebaseerd zijn, niet op het geplande end_ms, anders wordt een
                # niet-gedecodeerd staartstuk toch als "afgehandeld" geboekt en later uit
                # de live-tekst gesnoeid zonder vervanging.
                real_window_end_ms = min(end_ms, decoded_end_ms) if decoded_end_ms is not None else end_ms
                logger.info(
                    f"[BATCH][DECODE][DBG] job={job['job_id']} "
                    f"window={start_ms}..{end_ms} decode={decode_start_ms}..{decode_end_ms} "
                    f"samples={(0 if audio_f32 is None else audio_f32.size)} reason={reason}"
                )

                if audio_f32 is None or getattr(audio_f32, "size", 0) == 0:
                    logger.warning(
                        f"[BATCH][SKIP] job={job['job_id']} empty/invalid wav slice "
                        f"decode={decode_start_ms}..{decode_end_ms}"
                    )
                    # Geen batch-decode mogelijk (venster wijst voorbij de daadwerkelijk
                    # opgenomen WAV, zie [DIAG][WAV_DRIFT]). Dit venster verdween voorheen
                    # volledig uit het opgeslagen transcript, zelfs als er wél live-tekst
                    # voor bestond (zichtbaar in de GUI met [1], maar nooit gepersisteerd
                    # -- de gebruiker zag het live, maar het was bij het opvragen van de
                    # sessie later spoorloos). Val net als bij een afgewezen batch terug op
                    # live_text -- zonder text_batch, dus zonder vinkje in de UI, maar wél
                    # bewaard en terugluisterbaar (start_ms/end_ms verwijzen naar de altijd
                    # volledige WAV, onafhankelijk van batch-status).
                    if live_text:
                        async with self.lock:
                            group_id = self.tokens_alignment.apply_batch_group(
                                window_start_ms=start_ms,
                                window_end_ms=end_ms,
                                text_final=live_text,
                                text_batch=None,
                                speaker=-1,
                            )
                        logger.info(
                            f"[BATCH][SKIP][LIVE_FALLBACK] job={job['job_id']} "
                            f"group_id={group_id} persisted zonder batch-bevestiging"
                        )
                        try:
                            upd = SegmentUpdate(
                                id=str(group_id),
                                state="FINAL",
                                start_ms=start_ms,
                                end_ms=end_ms,
                                text_batch=None,
                                text_final=live_text,
                                is_final=(reason == "end_of_stream"),
                            )
                            await self.emit_segment_update(upd)
                        except Exception as emit_err:
                            logger.warning(
                                f"[BATCH][SKIP][EMIT] live-fallback update mislukt voor "
                                f"group={group_id} job={job['job_id']}: {emit_err}"
                            )
                    continue

                # Tolk met taalpaar → auto-detectie; overige kanalen → channel_language als hint
                _lang_override = (
                    "auto" if self.channel_language2
                    else self.channel_language
                )
                logger.info(
                    "[BATCH][LANG] job=%s channel=%s lang_used=%s (override=%s init=%s)",
                    job["job_id"], self.channel_id,
                    _lang_override if _lang_override else getattr(self.batch_asr, "language", None),
                    _lang_override,
                    getattr(self.batch_asr, "language", None),
                )
                _batch_perf_t0 = time()
                result = await asyncio.to_thread(
                    self.batch_asr.transcribe,
                    audio_f32,
                    word_timestamps=True,
                    language_override=_lang_override,
                )
                logger.info(
                    f"[DIAG][BATCH_PERF] job={job['job_id']} transcribe_duration="
                    f"{time() - _batch_perf_t0:.3f}s"
                )
                batch_txt = result["text"]
                sentence_segments = result.get("sentence_segments", [])
                logger.info(
                    f"[BATCH][SENTENCES] job={job['job_id']} "
                    f"num_sentences={len(sentence_segments)} "
                    f"segments={[(s['text'][:30], s['start'], s['end']) for s in sentence_segments[:10]]}"
                )
                batch_avg_logprob = result["avg_logprob"]
                batch_compression = result["compression_ratio"]

                batch_txt = (batch_txt or "").strip()

                # --- Fix 1: filter hallucination-zinnen per sentence, niet op de volledige tekst ---
                # Een enkel ***-artefact aan de stilte-grens mag niet het hele venster afkeuren.
                # HALLUCINATION_PATTERNS komt uit simul_whisper.backend -- gedeeld met
                # evaluate_batch_segment() hieronder en met "Ververs Transcriptie", zodat
                # deze lijst nooit tussen de twee paden uiteen kan lopen.
                if sentence_segments:
                    clean_sentences = [
                        s for s in sentence_segments
                        if not any(p in (s.get("text") or "") for p in HALLUCINATION_PATTERNS)
                    ]
                    if len(clean_sentences) < len(sentence_segments):
                        removed = len(sentence_segments) - len(clean_sentences)
                        logger.info(
                            f"[BATCH][FILTER] job={job['job_id']} "
                            f"removed {removed} hallucination sentence(s) "
                            f"from {len(sentence_segments)} → {len(clean_sentences)} clean"
                        )
                        sentence_segments = clean_sentences
                        batch_txt = " ".join(s["text"] for s in clean_sentences).strip()

                logger.info(
                    f"[BATCH][TEXT][DBG] job={job['job_id']} "
                    f"len_live={len(live_text)} len_batch={len(batch_txt)}"
                )

                # 3) CONFIDENCE-based accept policy -- kernbeslissing gedeeld via
                # evaluate_batch_segment() (simul_whisper/backend.py), zodat dit pad en
                # "Ververs Transcriptie" nooit uiteenlopende drempels kunnen krijgen.
                use_batch_as_final = False

                if batch_txt:
                    # --- Fix 2: ruimere no_speech_prob-drempel voor korte end-of-stream fragmenten ---
                    # FasterWhisper rapporteert structureel hoge no_speech_prob (0.85+) voor
                    # korte clips aan het einde van een sessie, ook bij echte spraak. Dit is
                    # een eigenaardigheid van het venster-getriggerde (incrementele) pad --
                    # geen onderdeel van de gedeelde kernfunctie.
                    audio_duration_s = (end_ms - start_ms) / 1000.0
                    no_speech_threshold = (
                        0.95 if (reason == "end_of_stream" and audio_duration_s < 8.0) else 0.6
                    )
                    use_batch_as_final, _reject_reason = evaluate_batch_segment(
                        batch_avg_logprob, batch_compression, result.get("no_speech_prob"),
                        batch_txt, no_speech_threshold=no_speech_threshold,
                    )
                    if not use_batch_as_final:
                        logger.warning(
                            f"[BATCH][REJECTED] job={job['job_id']} "
                            f"logprob={batch_avg_logprob} "
                            f"compr={batch_compression} "
                            f"no_speech_prob={result.get('no_speech_prob')} "
                            f"threshold={no_speech_threshold} "
                            f"reason={_reject_reason} "
                            f"text='{batch_txt[:80]}'"
                        )
                # Als batch niet vertrouwd wordt, val terug op live -- maar NOOIT op de
                # net-afgewezen batch_txt zelf. "Afgewezen" moet ook echt niets betekenen:
                # als er dan ook geen live-tekst is (bv. omdat de cross-kanaal anti-lek-gate
                # dit venster grotendeels onderdrukte), is er simpelweg niets betrouwbaars
                # om te tonen. De oude "anders batch"-fallback liet precies het net
                # afgewezen, gehallucineerde venster alsnog zien -- het tegenovergestelde
                # van wat een confidence-gate hoort te doen.
                if use_batch_as_final:
                    final_txt = batch_txt
                else:
                    final_txt = live_text

                logger.info(
                    f"[BATCH][GATE][DBG] job={job['job_id']} "
                    f"use_batch={use_batch_as_final} "
                    f"avg_logprob={batch_avg_logprob} "
                    f"compression_ratio={batch_compression} "
                    f"len_live={len(live_text)} len_batch={len(batch_txt)} "
                    f"final_len={len(final_txt)}"
                )

                # 4) Mutate segment
                async with self.lock:
                    group_id = self.tokens_alignment.apply_batch_group(
                        window_start_ms=start_ms,
                        window_end_ms=real_window_end_ms,
                        text_final=final_txt,
                        text_batch=(batch_txt if (use_batch_as_final and batch_txt) else None),
                        speaker=-1,
                    )
                    # Het geplande end_ms lag verder dan wat er echt gedecodeerd is
                    # (WAV liep nog niet bij op decodeermoment). Het volgende venster
                    # stond al klaar om vanaf het optimistische end_ms te beginnen
                    # (_batch_on_silence_boundary/_flush_final_batch_tail zetten dat
                    # synchroon bij enqueue) -- zet dat terug naar wat echt gedekt is,
                    # zodat het gemiste staartstuk in het eerstvolgende venster alsnog
                    # wordt opgepikt i.p.v. voor altijd overgeslagen. Alleen als er
                    # ondertussen niet al een nieuwer venster overheen is gegaan.
                    if real_window_end_ms < end_ms and self._batch_window_start_ms == end_ms:
                        logger.info(
                            f"[BATCH][WINDOW_REWIND] job={job['job_id']} "
                            f"start teruggezet van {end_ms}ms naar {real_window_end_ms}ms "
                            f"(WAV bevatte nog niet alles op decodeermoment)"
                        )
                        self._batch_window_start_ms = real_window_end_ms

                logger.info(
                    f"[BATCH][APPLY][DBG] job={job['job_id']} group_id={group_id} "
                    f"applied_text_batch={'yes' if batch_txt else 'no'} use_batch={use_batch_as_final}"
                )

                # 5) Emit SegmentUpdate naar UI
                # apply_batch_group() is al uitgevoerd; state is de bron van waarheid.
                # Als de emit mislukt (bijv. WebSocket disconnect) wordt de batch_group
                # alsnog opgepikt door de eerstvolgende reguliere FrontData-emissie.
                try:
                    if use_batch_as_final and sentence_segments:
                        # Meerdere klikbare zinnen met eigen tijdstempels
                        for i, sent in enumerate(sentence_segments):
                            sent_start_ms = int(round(sent["start"] * 1000)) + decode_start_ms
                            sent_end_ms = int(round(sent["end"] * 1000)) + decode_start_ms + 300  # 300ms buffer
                            sent_id = f"{group_id}_s{i}"
                            upd = SegmentUpdate(
                                id=sent_id,
                                state="FINAL",
                                start_ms=sent_start_ms,
                                end_ms=sent_end_ms,
                                text_batch=sent["text"],
                                text_final=sent["text"],
                                is_final=(reason == "end_of_stream"),
                            )
                            await self.emit_segment_update(upd)
                    else:
                        # Fallback: één blok voor het hele window
                        upd = SegmentUpdate(
                            id=str(group_id),
                            state="FINAL",
                            start_ms=start_ms,
                            end_ms=real_window_end_ms,
                            text_batch=(batch_txt if (use_batch_as_final and batch_txt) else None),
                            text_final=final_txt,
                            is_final=(reason == "end_of_stream"),
                        )
                        await self.emit_segment_update(upd)
                except Exception as emit_err:
                    logger.warning(
                        f"[BATCH][EMIT] segment update failed for group={group_id} "
                        f"job={job['job_id']}: {emit_err} — state correct, UI synct bij volgende FrontData"
                    )

                logger.info(
                    f"[BATCH][WORKER] BATCHGROUP overwrite id={group_id} job={job['job_id']} "
                    f"window={start_ms}..{end_ms} "
                    f"pad_left={BATCH_CONTEXT_PAD_LEFT_MS}ms "
                    f"pad_right={BATCH_CONTEXT_PAD_RIGHT_MS}ms "
                    f"use_batch={use_batch_as_final} reason={reason}"
                )

            except Exception as e:
                logger.warning(f"[BATCH][WORKER] error: {e}")
            finally:
                self._batch_queue.task_done()

    async def cleanup(self) -> None:
        """Clean up resources when processing is complete."""
        logger.info("Starting cleanup of AudioProcessor resources.")
        self.is_stopping = True
        # 1) Geef batch worker kans om clean te stoppen
        try:
            await self._batch_queue.put(SENTINEL)
        except Exception:
            pass

        # 2) Geef event loop 1 tick om sentinel te verwerken
        await asyncio.sleep(0)

        # 3) Cancel alleen als fallback (geef worker kans om SENTINEL te consumeren).
        # 10s i.p.v. de eerdere 0.25s: sinds batch_asr.transcribe()/de WAV-lees via
        # asyncio.to_thread lopen (niet meer synchroon op de event loop), duurt een
        # regulier venster gemeten 2.4-5.8s ([DIAG][BATCH_PERF] transcribe_duration).
        # Bij 0.25s werd de allerlaatste batch-taak na Stop (_flush_final_batch_tail,
        # reason="end_of_stream") daardoor zo goed als altijd halverwege geannuleerd,
        # vóór apply_batch_group() de batch-bevestigde slotzin kon toepassen -- de
        # Stop-knop zelf wordt hier niet trager van, de frontend wacht al niet op
        # cleanup() (zie stopRecording() in app.js).
        if self._batch_worker_task and not self._batch_worker_task.done():
            try:
                await asyncio.wait_for(self._batch_worker_task, timeout=10.0)
            except Exception:
                self._batch_worker_task.cancel()

        for task in self.all_tasks_for_cleanup:
            if task and not task.done():
                task.cancel()
            
        created_tasks = [t for t in self.all_tasks_for_cleanup if t]
        if created_tasks:
            await asyncio.gather(*created_tasks, return_exceptions=True)
        logger.info("All processing tasks cancelled or finished.")

        if not self.is_pcm_input and self.ffmpeg_manager:
            try:
                await self.ffmpeg_manager.stop()
                logger.info("FFmpeg manager stopped.")
            except Exception as e:
                logger.warning(f"Error stopping FFmpeg manager: {e}")
        try:
            if self.diarization:
                self.diarization.close()
        finally:
            self._close_wav()
        logger.info("AudioProcessor cleanup complete.")

    def _processing_tasks_done(self) -> bool:
        """Return True when all active processing tasks have completed."""
        tasks_to_check = [
            self.transcription_task,
            self.diarization_task,
            self.translation_task,
            self.ffmpeg_reader_task,
        ]
        return all(task.done() for task in tasks_to_check if task)


    async def process_audio(self, message: Optional[bytes]) -> None:
        """Process incoming audio data."""

        if not self.beg_loop:
            self.beg_loop = time()
            self.current_silence = Silence(start=0.0, is_starting=True)
            self.tokens_alignment.beg_loop = self.beg_loop
            # Batch window starts at t=0 for this session
            self._batch_window_start_ms = 0

        if not message:
            logger.info("Empty audio message received, initiating stop sequence.")

            # Zorg dat eventueel resterende PCM eerst nog naar de pipeline én WAV gaat
            if self.pcm_buffer:
                await self.handle_pcm_data()

            # Markeer stop pas hierna
            self.is_stopping = True

            # Sluit een open stilte netjes af zodat state.end_buffer / tokens bijgewerkt zijn
            if self.current_silence:
                try:
                    await self._end_silence()
                except Exception as e:
                    logger.warning(f"[BATCH][FINALFLUSH] ending current silence failed: {e}")

            # Sluit de lopende, nog niet gevalideerde live-regel af als eigen FINAL
            # segment -- zelfde reden als bij _pause_flush(): zonder dit blijft de
            # allerlaatste, nog niet door een stilte-token afgesloten live-tekst buiten
            # live_text (die alleen al-FINAL segmenten meeneemt, zie hieronder), en dus
            # ook buiten wat _flush_final_batch_tail() eventueel als live-fallback kan
            # persisteren als de batch-job voor de staart skipt of wordt afgewezen.
            try:
                self.tokens_alignment.flush_current_line()
            except Exception as e:
                logger.warning(f"[BATCH][FINALFLUSH] flush_current_line failed: {e}")

            # Forceer nog één laatste batch-window voor de resterende tail
            await self._flush_final_batch_tail(reason="end_of_stream")

            # Nu pas WAV sluiten, zodat de batch worker de slice nog uit bestand kan lezen
            self._close_wav()

            # Laat transcription processor stoppen
            if self.transcription_queue:
                await self.transcription_queue.put(SENTINEL)

            if not self.is_pcm_input and self.ffmpeg_manager:
                await self.ffmpeg_manager.stop()

            return
        
        if self.is_stopping:
            logger.warning("AudioProcessor is stopping. Ignoring incoming audio.")
            return

        if self.is_pcm_input:
            if self._gate_framed:
                # Frame: [1 byte gate-vlag][s16le PCM]. De vlag bepaalt alleen of dit
                # fragment straks naar VAD/ASR mag (zie handle_pcm_data) — de audio zelf
                # wordt hieronder altijd volledig in pcm_buffer gezet en dus altijd opgenomen.
                flag = message[0]
                payload = message[1:]
                if flag == 2 and not payload:
                    # Pauze-flush (client-side Pauze-knop): sluit het huidige live-
                    # segment/batch-venster netjes af, zonder de sessie/WAV te
                    # beëindigen. Geen audio-payload voor dit bericht.
                    await self._pause_flush()
                    return
                if flag == 3 and not payload:
                    # "Ververs Transcriptie" tijdens Pauze: zorg dat wat op schijf
                    # staat actueel is vóórdat de client het REST-endpoint
                    # /sessions/{id}/refresh_transcript aanroept (die leest de WAV
                    # rechtstreeks, buiten deze levende AudioProcessor om, en heeft
                    # dus geen andere manier om een pending write-buffer te zien).
                    # Geen transcriptielogica hier -- puur de bestaande WAV-flush.
                    self._flush_wav()
                    return
                gate_open = flag != 0
                n_samples = len(payload) // self.bytes_per_sample
                if n_samples > 0:
                    self._gate_segments.append((n_samples, gate_open))
                self.pcm_buffer.extend(payload)
            else:
                self.pcm_buffer.extend(message)
            await self.handle_pcm_data()
        else:
            if not self.ffmpeg_manager:
                logger.error("FFmpeg manager not initialized for non-PCM input.")
                return
            success = await self.ffmpeg_manager.write_data(message)
            if not success:
                ffmpeg_state = await self.ffmpeg_manager.get_state()
                if ffmpeg_state == FFmpegState.FAILED:
                    logger.error("FFmpeg is in FAILED state, cannot process audio")
                else:
                    logger.warning("Failed to write audio data to FFmpeg")

    def _consume_gate_mask(self, n_samples: int) -> Optional[np.ndarray]:
        """Bouwt een boolean-mask (True = naar ASR mag) voor de eerstvolgende n_samples,
        door self._gate_segments te consumeren (evt. splitsen van het kopsegment).
        Retourneert None als er geen framing actief is -> huidig gedrag (alles open)."""
        if not self._gate_framed:
            return None
        mask = np.ones(n_samples, dtype=bool)
        filled = 0
        while filled < n_samples and self._gate_segments:
            seg_samples, seg_open = self._gate_segments[0]
            take = min(seg_samples, n_samples - filled)
            if not seg_open:
                mask[filled:filled + take] = False
            filled += take
            remaining = seg_samples - take
            if remaining > 0:
                self._gate_segments[0] = (remaining, seg_open)
            else:
                self._gate_segments.popleft()
        # Als segmenten opraken (bv. eerste chunk vóór het eerste bericht verwerkt is),
        # blijft de rest van de mask True: fail-open, nooit fail-closed.
        return mask

    async def handle_pcm_data(self) -> None:
        # Process when enough data
        if len(self.pcm_buffer) < self.bytes_per_sec:
            return

        if len(self.pcm_buffer) > self.max_bytes_per_sec:
            logger.warning(
                f"Audio buffer too large: {len(self.pcm_buffer) / self.bytes_per_sec:.2f}s. "
                f"Consider using a smaller model."
            )

        chunk_size = min(len(self.pcm_buffer), self.max_bytes_per_sec)
        aligned_chunk_size = (chunk_size // self.bytes_per_sample) * self.bytes_per_sample
        
        if aligned_chunk_size == 0:
            return
        raw_pcm = bytes(self.pcm_buffer[:aligned_chunk_size])

        # Session WAV opnemen (bronbestand)
        self._ensure_wav_open()
        if self._wav_writer is not None:
            self._wav_writer.writeframes(raw_pcm)

        pcm_array = self.convert_pcm_to_float(raw_pcm)
        self.pcm_buffer = self.pcm_buffer[aligned_chunk_size:]


        num_samples = len(pcm_array)
        chunk_sample_start = self.total_pcm_samples
        chunk_sample_end = chunk_sample_start + num_samples

        # Cross-kanaal anti-lek: alleen de kopie die naar VAD/ASR gaat wordt voor
        # dicht-gemarkeerde samples op nul gezet. pcm_array zelf (en dus de WAV, hierboven
        # al weggeschreven) blijft altijd de volledige, ongewijzigde audio.
        gate_mask = self._consume_gate_mask(num_samples)
        if gate_mask is not None and not gate_mask.all():
            n_closed = int((~gate_mask).sum())
            logger.debug(
                f"[GATE] channel={self.channel_id} suppressed {n_closed}/{num_samples} "
                f"samples for ASR (cross-channel anti-leak, WAV unaffected)"
            )
            asr_pcm_array = pcm_array.copy()
            asr_pcm_array[~gate_mask] = 0.0
        else:
            asr_pcm_array = pcm_array

        res = None
        if self.args.vac:
            res = self.vac(asr_pcm_array)

        if res is not None:
            if "start" in res and self.current_silence:
                await self._end_silence()

            if "end" in res and not self.current_silence:
                pre_silence_chunk = self._slice_before_silence(
                    asr_pcm_array, chunk_sample_start, res.get("end")
                )
                if pre_silence_chunk is not None and pre_silence_chunk.size > 0:
                    await self._enqueue_active_audio(pre_silence_chunk)
                await self._begin_silence()

        if not self.current_silence:
            await self._enqueue_active_audio(asr_pcm_array)


        self.total_pcm_samples = chunk_sample_end

        if not self.args.transcription and not self.args.diarization:
            await asyncio.sleep(0.1)
