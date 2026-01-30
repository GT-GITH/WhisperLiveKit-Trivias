import asyncio
import logging
import traceback

import uuid

import os
import wave
from pathlib import Path
from datetime import datetime

from time import time
from typing import Any, AsyncGenerator, List, Optional, Union
from whisperlivekit.timed_objects import SegmentUpdate

import numpy as np
from types import SimpleNamespace   

from whisperlivekit.core import (TranscriptionEngine,
                                 online_diarization_factory, online_factory,
                                 online_translation_factory)
from whisperlivekit.ffmpeg_manager import FFmpegManager, FFmpegState
from whisperlivekit.silero_vad_iterator import FixedVADIterator, OnnxWrapper, load_jit_vad
from whisperlivekit.timed_objects import (ASRToken, ChangeSpeaker, FrontData,
                                          Segment, Silence, State, Transcript)
from whisperlivekit.tokens_alignment import TokensAlignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SENTINEL = object() # unique sentinel object for end of stream marker
# Per-queue pushback slot used by get_all_from_queue() to preserve ordering without peeking.
_QUEUE_PUSHBACK: dict[int, Any] = {}

# Stilte die we als "segment boundary" gebruiken (dus FINAL/segment-close).
# Dit heeft niets met decoder reset te maken.
SILENCE_TOKEN_MIN_DURATION = 0.10  # 100ms (mag 0.0 als je alles wil)

# Vanaf hoeveel seconden stilte we de decoder (AlignAtt) resetten
SILENCE_RESET_THRESHOLD = 3.0  # kun je later tweaken (2–5s)

ENABLE_HARD_CAP = False

# Batch windowing (Stap 1)
BATCH_TARGET_WINDOW_MS = 30_000   # 30s
BATCH_MIN_WINDOW_MS    = 15_000   # (nu nog niet gebruikt, maar handig)
BATCH_HARD_CAP_MS      = 45_000   # (nu nog niet gebruikt, maar handig)

async def get_all_from_queue(queue: asyncio.Queue) -> Union[object, Silence, np.ndarray, List[Any]]:
    """
    Get one logical item from an asyncio.Queue.

    - For audio queues: coalesce consecutive np.ndarray chunks into a single np.concatenate() result.
    - For non-audio queues (e.g. translation): return a list of consecutive non-sentinel/non-silence items.

    We DO NOT peek into private queue internals (queue._queue). To preserve FIFO ordering when we
    encounter SENTINEL or Silence while draining, we use a per-queue pushback slot.
    """
    items: List[Any] = []
    qid = id(queue)

    # 0) If we have a pushed-back item, consume it first (preserves original order).
    if qid in _QUEUE_PUSHBACK:
        first_item = _QUEUE_PUSHBACK.pop(qid)
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
            _QUEUE_PUSHBACK[qid] = nxt
            break

        # Coalesce only homogeneous audio chunks
        if isinstance(first_item, np.ndarray) and not isinstance(nxt, np.ndarray):
            # This should not happen in a well-formed audio queue; don't mix types.
            _QUEUE_PUSHBACK[qid] = nxt
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
        self.sample_rate = 16000
        self.channels = 1
        self.samples_per_sec = int(self.sample_rate * self.args.min_chunk_size)
        self.bytes_per_sample = 2
        self.bytes_per_sec = self.samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = 32000 * 5  # 5 seconds of audio at 32 kHz
        self.is_pcm_input = self.args.pcm_input

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
            self.transcription = online_factory(self.args, models.asr)        
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
        # Alleen loggen als DEBUG aan staat of msg verandert, en max 1x per interval.
        now = time()
        if msg == self._status_last_msg and (now - self._status_last_log_t) < self._status_min_interval_s:
            return
        logger.debug(msg)
        self._status_last_msg = msg
        self._status_last_log_t = now

    async def emit_segment_update(self, upd: SegmentUpdate) -> None:
        """Queue a WS update that will be yielded by results_formatter."""
        await self._ws_update_queue.put(upd)

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
        now = time() - self.beg_loop
        self.current_silence = Silence(
            is_starting=True, start=now
        )
        await self._push_silence_event()

    async def _end_silence(self) -> None:
        if not self.current_silence:
            return
        now = time() - self.beg_loop
        self.current_silence.end = now
        self.current_silence.is_starting=False
        self.current_silence.has_ended=True
        self.current_silence.compute_duration()
        if self.current_silence.duration >  SILENCE_TOKEN_MIN_DURATION:
            self.state.new_tokens.append(self.current_silence)
        await self._push_silence_event()
        self.current_silence = None

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
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        await self._batch_queue.put(job)
        logger.info(
            f"[BATCH][ENQUEUE] job_id={job['job_id']} "
            f"window={job['start_ms']}..{job['end_ms']}ms "
            f"len={(job['end_ms']-job['start_ms'])/1000.0:.2f}s reason={reason}"
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
                        new_tokens, current_audio_processed_upto = await asyncio.to_thread(
                            self.transcription.start_silence
                        )
                        asr_processing_logs += f" + Silence starting"
                        # ===== Stap 1: batch windowing proberen te sluiten op silence-start =====
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
                        #try:
                        #    stream_time_ms = int(round(cumulative_pcm_duration_stream_time * 1000.0))
                        #    await self._batch_on_silence_boundary(
                        #        stream_time_ms=stream_time_ms,
                        #        boundary="silence_end"
                        #   )
                        #except Exception as e:
                         #   logger.warning(f"[BATCH][WINDOW] silence-end handler failed: {e}")

                        # 🔸 En nu de *enige* echte decoder-reset na lange stilte
                        if item.duration >= SILENCE_RESET_THRESHOLD:
                            logger.info(
                                f"[Decoder reset] refresh_segment(complete=True) "
                                f"na {item.duration:.2f}s stilte (threshold={SILENCE_RESET_THRESHOLD}s)"
                            )

                            # Zoek de decoder met refresh_segment (asr of model)
                            base_asr = None
                            for attr_name in ("asr", "model"):
                                candidate = getattr(self.transcription, attr_name, None)
                                if candidate is not None and hasattr(candidate, "refresh_segment"):
                                    base_asr = candidate
                                    logger.info(
                                        f"[Decoder reset] gebruik decoder via "
                                        f"self.transcription.{attr_name}.refresh_segment(complete=True)"
                                    )
                                    break

                            if base_asr is not None:
                                # Dit roept onder water al aan:
                                # - init_tokens()
                                # - init_context()
                                # - pending_incomplete_tokens = []
                                # - segments leegmaken
                                base_asr.refresh_segment(complete=True)
                               
                            else:
                                logger.warning(
                                    "[Decoder reset] geen decoder met refresh_segment gevonden "
                                    "(asr/model) – geen reset uitgevoerd"
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

                if ENABLE_HARD_CAP:
                    # ===== Stap 1: HARD CAP (force-close) ALS ENABLE_HARD_CAP = True =====
                    try:
                        if self._batch_window_start_ms is not None:
                            now_ms = int(round(stream_time_end_of_current_pcm * 1000.0))
                            if (now_ms - int(self._batch_window_start_ms)) >= BATCH_HARD_CAP_MS:
                                # Force enqueue op laatste speech-end
                                window_start_ms = int(self._batch_window_start_ms)
                                # HARD CAP moet ALTIJD vooruitgang forceren in tijd.
                                # Gebruik daarom now_ms als window_end (niet tokens[-1].end), anders krijg je micro-windows als tokens achterlopen.
                                window_end_ms = now_ms

                                if self._batch_last_close_end_ms is None or window_end_ms > self._batch_last_close_end_ms:
                                    self._batch_last_close_end_ms = window_end_ms
                                    await self._enqueue_batch_window(
                                        window_start_ms=window_start_ms,
                                        window_end_ms=window_end_ms,
                                        reason="hard_cap_close"
                                    )

                                    # Advance window naar echte tijd, zodat hard-cap niet direct opnieuw triggert
                                    self._batch_window_start_ms = window_end_ms
                                    self._batch_ready_to_close = False
                                    self._batch_ready_at_ms = None
                    except Exception as e:
                        logger.warning(f"[BATCH][WINDOW] hard-cap handler failed: {e}")

                    self.transcription.insert_audio_chunk(pcm_array, stream_time_end_of_current_pcm)
                    new_tokens, current_audio_processed_upto = await asyncio.to_thread(self.transcription.process_iter)
                    new_tokens = new_tokens or []

                _buffer_transcript = self.transcription.get_buffer()
                buffer_text = _buffer_transcript.text

                if new_tokens:
                    validated_text = self.sep.join([t.text for t in new_tokens])
                    if buffer_text.startswith(validated_text):
                        _buffer_transcript.text = buffer_text[len(validated_text):].lstrip()

                candidate_end_times = [self.state.end_buffer]

                if new_tokens:
                    candidate_end_times.append(new_tokens[-1].end)
                
                if _buffer_transcript.end is not None:
                    candidate_end_times.append(_buffer_transcript.end)
                
                candidate_end_times.append(current_audio_processed_upto)
                
                async with self.lock:
                    self.state.tokens.extend(new_tokens)
                    self.state.buffer_transcription = _buffer_transcript
                    self.state.end_buffer = max(candidate_end_times)
                    self.state.new_tokens.extend(new_tokens)
                    self.state.new_tokens_buffer = _buffer_transcript

                if self.translation_queue:
                    for token in new_tokens:
                        await self.translation_queue.put(token)                
            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if 'pcm_array' in locals() and pcm_array is not SENTINEL : # Check if pcm_array was assigned from queue
                    self.transcription_queue.task_done()
        
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
        while True:
            try:
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
                    remaining_time_diarization=state.remaining_time_diarization if self.args.diarization else 0
                )
                                
                should_push = (response != self.last_response_content)
                if should_push:
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
        
        # ===== Batch dummy worker (Stap 2) =====
        if self._batch_worker_task is None or self._batch_worker_task.done():
            self._batch_worker_task = asyncio.create_task(self._batch_worker_dummy())
            self.all_tasks_for_cleanup.append(self._batch_worker_task)

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
                        exc = task.exception()
                        task_name = task.get_name() if hasattr(task, 'get_name') else f"Monitored Task {i}"
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
        filename = f"session_{self.session_id}_{ts}.wav"
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

    def _read_wav_slice_float32(self, start_ms: int, end_ms: int) -> Optional[np.ndarray]:
        if not getattr(self, "_wav_path", None):
            return None
        if end_ms <= start_ms:
            return None

        self._flush_wav()

        start_frame = int((start_ms / 1000.0) * self.sample_rate)
        end_frame   = int((end_ms   / 1000.0) * self.sample_rate)
        n_frames    = max(0, end_frame - start_frame)
        if n_frames <= 0:
            return None

        try:
            with wave.open(str(self._wav_path), "rb") as rf:
                rf.setpos(min(start_frame, rf.getnframes()))
                raw = rf.readframes(n_frames)
            if not raw:
                return None
            # raw is s16le mono
            return self.convert_pcm_to_float(raw)
        except Exception as e:
            logger.warning(f"[BATCH] WAV slice read failed: {e}")
            return None

    def _batch_transcribe_text(self, audio_f32: np.ndarray) -> Optional[str]:
        try:
            if not self.batch_asr:
                return None
            txt = self.batch_asr.transcribe_text(audio_f32)
            return txt or None
        except Exception as e:
            logger.warning(f"[BATCH] transcribe failed: {e}")
            return None
        
    async def _batch_worker_dummy(self) -> None:
        """Stap 2: dummy worker die 1 FINAL segment in window markeert met 'BATCH OK'."""
        logger.info("[BATCH][WORKER] dummy worker started")
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

                # Pak één FINAL segment dat binnen dit window valt
                chosen = None

                async with self.lock:
                    # Zorg dat alignment up-to-date is (minimaal drain van state)
                    self.tokens_alignment.update()
                    lines, _, _ = self.tokens_alignment.get_lines(
                        diarization=self.args.diarization,
                        translation=bool(self.translation),
                        current_silence=self.current_silence
                    )

                # Kies: laatste FINAL segment dat (grotendeels) binnen window valt en geen silence is
                chosen = None

                for seg in reversed(lines):
                    # Skip silence segments: we willen een speech segment kiezen
                    if hasattr(seg, "is_silence") and seg.is_silence():
                        continue

                    # Segmenten kunnen óf ms-velden hebben, óf start/end in seconden.
                    seg_start_ms = getattr(seg, "start_ms", None)
                    seg_end_ms   = getattr(seg, "end_ms", None)

                    if seg_start_ms is None:
                        seg_start_s = getattr(seg, "start", None)
                        seg_start_ms = int(round(float(seg_start_s) * 1000.0)) if seg_start_s is not None else 0
                    else:
                        seg_start_ms = int(seg_start_ms)

                    if seg_end_ms is None:
                        seg_end_s = getattr(seg, "end", None)
                        # LIVE kan end=None hebben; dan behandelen we end als window-einde
                        if seg_end_s is None:
                            seg_end_ms = end_ms
                        else:
                            seg_end_ms = int(round(float(seg_end_s) * 1000.0))
                    else:
                        seg_end_ms = int(seg_end_ms)

                    if seg_end_ms <= seg_start_ms:
                        continue

                    # kies segment dat overlapt met window
                    if (
                        seg_end_ms > start_ms
                        and seg_start_ms < end_ms
                        and getattr(seg, "state", "") in ("LIVE", "FINAL")
                    ):
                        chosen = seg
                        break


                if not chosen:
                    # Nog geen FINAL segment beschikbaar -> retry kort, bounded (geen moeras)
                    if attempt < 20:
                        job["_attempt"] = attempt + 1
                        await asyncio.sleep(0.2)
                        await self._batch_queue.put(job)
                        logger.info(
                            f"[BATCH][WORKER] no FINAL segment yet for job {job['job_id']} ({reason}) "
                            f"retry {job['_attempt']}/20"
                        )
                        continue

                    logger.warning(
                        f"[BATCH][WORKER] giving up: no FINAL segment for job {job['job_id']} ({reason}) after {attempt} retries"
                    )
                    continue
                
                # --- HARD OVERRIDE IN STATE (dit is de kern) ---
                async with self.lock:
                    # chosen is al een Segment uit tokens_alignment output
                    # We forceren hem FINAL in de state
                    chosen.state = "FINAL"
                    chosen.text = "BATCH OK"
                    chosen.text_live = None
                    chosen.text_batch = "BATCH OK"

                    # Zorg dat tijden kloppen
                    chosen.start = start_ms / 1000.0
                    chosen.end   = end_ms   / 1000.0

                upd = SegmentUpdate(
                    id=str(chosen.id),
                    text_final="BATCH OK",
                    # geen state forceren in dummy
                    start_ms=int(getattr(chosen, "start_ms", 0) or 0),
                    end_ms=int(getattr(chosen, "end_ms", 0) or 0),
                )

                await self.emit_segment_update(upd)
                logger.info(
                    f"[BATCH][WORKER] updated segment id={chosen.id} for job={job['job_id']} reason={reason}"
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

        # 3) Cancel alleen als fallback (geef worker kans om SENTINEL te consumeren)
        if self._batch_worker_task and not self._batch_worker_task.done():
            try:
                await asyncio.wait_for(self._batch_worker_task, timeout=0.25)
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
        if self.diarization:
            self.diarization.close()

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
            self.is_stopping = True
           
            # NEW: close session WAV immediately on stop
            self._close_wav()
            
            if self.transcription_queue:
                await self.transcription_queue.put(SENTINEL)

            if not self.is_pcm_input and self.ffmpeg_manager:
                await self.ffmpeg_manager.stop()

            return

        if self.is_stopping:
            logger.warning("AudioProcessor is stopping. Ignoring incoming audio.")
            return

        if self.is_pcm_input:
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

        res = None
        if self.args.vac:
            res = self.vac(pcm_array)

        if res is not None:
            if "start" in res and self.current_silence:
                await self._end_silence()
            
            if "end" in res and not self.current_silence:
                pre_silence_chunk = self._slice_before_silence(
                    pcm_array, chunk_sample_start, res.get("end")
                )
                if pre_silence_chunk is not None and pre_silence_chunk.size > 0:
                    await self._enqueue_active_audio(pre_silence_chunk)
                await self._begin_silence()

        if not self.current_silence:
            await self._enqueue_active_audio(pcm_array)


        self.total_pcm_samples = chunk_sample_end

        if not self.args.transcription and not self.args.diarization:
            await asyncio.sleep(0.1)
