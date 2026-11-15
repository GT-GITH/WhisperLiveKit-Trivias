import copy
import gc
import logging
import os
import platform
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
try:
    import torch
except Exception:
    torch = None

from whisperlivekit.backend_support import (faster_backend_available,
                                            mlx_backend_available)
from whisperlivekit.model_paths import detect_model_format, resolve_model_path
from whisperlivekit.simul_whisper.config import AlignAttConfig
from whisperlivekit.simul_whisper.chunked_vad import speech_clips, to_clip_timestamps
from whisperlivekit.simul_whisper.segment_split import split_on_word_gaps
from whisperlivekit.simul_whisper.simul_whisper import AlignAtt
from whisperlivekit.timed_objects import ASRToken, ChangeSpeaker, Transcript
from whisperlivekit.warmup import load_file
from whisperlivekit.whisper import load_model, tokenizer
from whisperlivekit.whisper.audio import TOKENS_PER_SECOND

logger = logging.getLogger(__name__)


HAS_MLX_WHISPER = mlx_backend_available(warn_on_missing=True)
if HAS_MLX_WHISPER:
    from .mlx_encoder import load_mlx_encoder, load_mlx_model, mlx_model_mapping
    from .mlx import MLXAlignAtt
else:
    mlx_model_mapping = {}
    MLXAlignAtt = None
HAS_FASTER_WHISPER = faster_backend_available(warn_on_missing=not HAS_MLX_WHISPER)
if HAS_FASTER_WHISPER:
    from faster_whisper import WhisperModel
else:
    WhisperModel = None

MIN_DURATION_REAL_SILENCE = 5


def resolve_batch_precision() -> Tuple[str, str]:
    """Device + compute_type voor het batch-model: ALTIJD volle precisie.

    GPU -> float16, CPU -> float32. Bewust niet instelbaar en nooit "auto":
    met compute_type="auto" koos CTranslate2 op een RTX 4090 stilzwijgend
    int8_float16, en in die modus viel een complete Turkse uiting weg (6/6
    keer; in float16 6/6 keer aanwezig, zelfde audio en instellingen --
    e2e-suite, 2026-11-15). De batch-pass is de juridisch citeerbare bron
    (CLAUDE.md); een kleine VRAM-besparing weegt daar niet tegen op. Op CPU
    is float16 niet efficiënt, dus float32: trager, maar even nauwkeurig --
    batch hoeft niet realtime te zijn.
    """
    try:
        import ctranslate2
        has_cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        has_cuda = False
    return ("cuda", "float16") if has_cuda else ("cpu", "float32")


class BatchFasterWhisperASR:
    """
    Offline/batch ASR op basis van faster-whisper WhisperModel.
    Los van SimulStreaming/AlignAtt, zodat batch andere decode settings kan hebben.
    """
    def __init__(
        self,
        model: str,
        language: str = "nl",
        beam_size: int = 5,
        condition_on_previous_text: bool = False,
        temperature: float = 0.0,
        initial_prompt: str | None = None,
        #best_of=None,   
        #patience=None,
    ):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not available (WhisperModel import failed).")
        #self.best_of = best_of
        #self.patience = patience
        self.language = language
        self.beam_size = beam_size
        self.condition_on_previous_text = condition_on_previous_text
        self.temperature = temperature
        self.initial_prompt = initial_prompt
        self.device, self.compute_type = resolve_batch_precision()
        self.model = WhisperModel(
            model,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info(
            f"[BATCH][PRECISION] model={model} device={self.device} "
            f"compute_type={self.compute_type} (volle precisie, vast -- zie resolve_batch_precision)"
        )

    def _decode(self, audio_f32: np.ndarray, lang: str | None, word_timestamps: bool):
        """Decodeer alleen de spraakstukken, bepaald met chunked_vad (frisse VAD-toestand
        per kort stuk) i.p.v. faster-whisper's vad_filter (één VAD-pass met opgebouwde
        toestand die na lange muziek/stilte spraak mist -- zie chunked_vad.py).
        clip_timestamps: faster-whisper decodeert de stukken op hun echte plek, zonder
        ze aan elkaar te plakken. Geen spraak gevonden -> geen segmenten."""
        clips = speech_clips(audio_f32)
        if not clips:
            logger.info("[BATCH][VAD] geen spraak gevonden in dit fragment -> niets te decoderen")
            return []
        logger.info(
            f"[BATCH][VAD] {len(clips)} spraakstuk(ken), "
            f"{sum(e - s for s, e in clips):.1f}s van {len(audio_f32) / 16000:.1f}s"
        )
        segments, _info = self.model.transcribe(
            audio_f32,
            language=lang if lang and lang != "auto" else None,
            beam_size=self.beam_size,
            condition_on_previous_text=self.condition_on_previous_text,
            temperature=self.temperature,
            initial_prompt=self.initial_prompt,
            vad_filter=False,
            clip_timestamps=to_clip_timestamps(clips),
            word_timestamps=word_timestamps,
            no_speech_threshold=0.9,
        )
        return segments

    def transcribe(self, audio_f32: np.ndarray, word_timestamps: bool = False, language_override: str | None = None) -> dict:
        lang = language_override if language_override is not None else self.language
        segments = self._decode(audio_f32, lang, word_timestamps)

        texts = []
        avg_logprobs = []
        compression_ratios = []
        no_speech_probs = []
        sentence_segments = []  # lijst van {text, start, end} per faster-whisper segment

        for s in segments:
            if not getattr(s, "text", None):
                continue
            texts.append(s.text.strip())
            if hasattr(s, "avg_logprob"):
                avg_logprobs.append(s.avg_logprob)
            if hasattr(s, "compression_ratio"):
                compression_ratios.append(s.compression_ratio)
            if hasattr(s, "no_speech_prob"):
                no_speech_probs.append(s.no_speech_prob)
            if word_timestamps:
                # Splits op grote woordgaten -- zie segment_split.py (segment over
                # een VAD-stilte heen / één segment voor meerdere uitingen).
                parts = split_on_word_gaps(getattr(s, "words", None)) or [(s.text.strip(), s.start, s.end)]
                for text, start, end in parts:
                    # Vertrouwenswaarden per zin (geërfd van het faster-whisper-
                    # segment), zodat de batch-worker PER ZIN kan keuren i.p.v.
                    # één oordeel over het hele venster -- zie gate_batch_sentence().
                    sentence_segments.append({
                        "text": text, "start": start, "end": end,
                        "avg_logprob": getattr(s, "avg_logprob", None),
                        "compression_ratio": getattr(s, "compression_ratio", None),
                        "no_speech_prob": getattr(s, "no_speech_prob", None),
                    })

        full_text = " ".join([t for t in texts if t]).strip()

        return {
            "text": full_text,
            "avg_logprob": float(np.mean(avg_logprobs)) if avg_logprobs else None,
            "compression_ratio": float(np.mean(compression_ratios)) if compression_ratios else None,
            "no_speech_prob": float(np.mean(no_speech_probs)) if no_speech_probs else None,
            "num_segments": len(texts),
            "sentence_segments": sentence_segments,  # alleen gevuld als word_timestamps=True
        }

    def transcribe_full(self, audio_f32: np.ndarray, language_override: str | None = None) -> list[dict]:
        """Transcribeer een heel bestand in één keer en geef PER SEGMENT het eigen
        vertrouwen terug, i.p.v. het over het hele venster gemiddelde `transcribe()`
        hierboven levert.

        Gebouwd voor "Ververs Transcriptie" (2026-07-19): de incrementele
        pauze/stilte-getriggerde batch-vensters wijzen één vertrouwensoordeel toe aan
        een heel venster (tot 40s) -- één laag-vertrouwen fragment daarin kan zo een
        verder prima venster laten afkeuren (gezien: een 42s-blok volledig onbevestigd
        door één zwak stukje). faster-whisper's eigen VAD hakt een lang bestand al
        intern op in natuurlijke segmenten, elk met een EIGEN avg_logprob/
        compression_ratio/no_speech_prob -- die per-segment waarden hier gewoon
        doorgeven (i.p.v. ze te middelen) maakt een latere gate-beslissing per zin
        mogelijk, preciezer dan de huidige per-venster aanpak."""
        lang = language_override if language_override is not None else self.language
        segments = self._decode(audio_f32, lang, word_timestamps=True)

        result = []
        for s in segments:
            text = (getattr(s, "text", None) or "").strip()
            if not text:
                continue
            # Splits op grote woordgaten (zie segment_split.py). De vertrouwens-
            # waarden zijn per faster-whisper-segment bekend, niet per deel: elk
            # deel erft die van zijn oorspronkelijke segment.
            parts = split_on_word_gaps(getattr(s, "words", None)) or [(text, s.start, s.end)]
            for part_text, start, end in parts:
                result.append({
                    "text": part_text,
                    "start": start,
                    "end": end,
                    "avg_logprob": getattr(s, "avg_logprob", None),
                    "compression_ratio": getattr(s, "compression_ratio", None),
                    "no_speech_prob": getattr(s, "no_speech_prob", None),
                })
        return result


# Kernlogica van de confidence-gate, gedeeld tussen de incrementele batch-worker
# (audio_processor.py, _batch_worker()) en "Ververs Transcriptie" (TriviasServer.py) --
# zodat de drempelwaarden nooit uiteen kunnen lopen tussen de twee paden.
HALLUCINATION_PATTERNS = [
    "***",
    "Ondertiteling",
    "ondertiteling",
    "Ondertitels",
    "ondertitels",
    "www.",
    ".com",
    "Abonneer",
    "abonneer",
    "Subtitles by",
    "Subscribe",
    "subscribe",
    "Altyazı",
    "altyazı",
    "Altyazi",
    "altyazi",
    # Turkse YouTube-outro-hallucinaties (waargenomen 2026-07-19 en herbevestigd
    # 2026-08-15, zie CLAUDE.md/refresh transcript-context) -- specifiek genoeg
    # om echte spraak niet te raken: "teşekkür ederim" alleen (gewoon "dankjewel")
    # zou wél te breed zijn en legitieme bedankjes afkeuren.
    "İzlediğiniz için",
    "izlediğiniz için",
    "abone olmayı",
]


def gate_batch_sentence(seg: dict) -> tuple[bool, str]:
    """De ENIGE keuring per batch-zin, gedeeld door de incrementele batch-worker
    (audio_processor._batch_worker) en "Ververs Transcriptie" (TriviasServer), zodat
    beide paden nooit meer uiteen kunnen lopen.

    Per zin, nooit per venster: in een live-test (2026-11-15) keurde de incrementele
    worker een heel venster van 31 s met 10 prima zinnen af omdat het venster-
    gemiddelde no_speech_prob 0.66 was (> 0.6), opgetrokken door een paar korte,
    onduidelijke fragmenten. Eén zwakke zin mag nooit de rest meeslepen.
    no_speech_threshold=None: faster-whisper berekent no_speech_prob per intern
    ~30s-decodeerblok, niet per zin (zie evaluate_batch_segment) -- dus geen
    zinsniveau-signaal. seg: dict met text/start/end/avg_logprob/compression_ratio/
    no_speech_prob (start/end in seconden)."""
    return evaluate_batch_segment(
        seg.get("avg_logprob"), seg.get("compression_ratio"), seg.get("no_speech_prob"),
        seg.get("text") or "",
        no_speech_threshold=None,
        duration_s=(seg["end"] - seg["start"]) if seg.get("end") is not None and seg.get("start") is not None else None,
    )


def evaluate_batch_segment(
    avg_logprob: float | None,
    compression_ratio: float | None,
    no_speech_prob: float | None,
    text: str,
    no_speech_threshold: float | None = 0.6,
    duration_s: float | None = None,
) -> tuple[bool, str]:
    """Kernbeslissing: is dit batch-resultaat betrouwbaar genoeg om als bevestigd
    (met vinkje) te tonen? Retourneert (geaccepteerd, reden-voor-logging).

    duration_s (optioneel, lengte van het gedecodeerde audiofragment in seconden):
    vangt een ander faalpatroon dan de bestaande checks hierboven. Geconstateerd
    2026-09-18 (foreign_so-modelroutering-PoC): na uitputting van de temperatuur-
    fallback-ladder op repetitie kan faster-whisper voor een venster van 25+
    seconden spraak een tekst van een paar tekens teruggeven (bv. "baki nda") --
    en omdat zo'n triviaal korte tekst per definitie een lage compression_ratio
    heeft, kwam die ondanks de mislukking gewoon door de bestaande checks heen.
    Alleen toegepast vanaf 3s (kortere fragmenten zijn te ruisgevoelig om op
    tekens/seconde te beoordelen) en met een bewust lage vloer (1 teken/seconde
    -- ruim onder normale spreeksnelheid van 10-15 tekens/seconde) zodat dit
    geen echte, beknopte uitspraken afkeurt.

    no_speech_threshold=None schakelt de no_speech_prob-check helemaal uit. Nodig
    voor "Ververs Transcriptie" (2026-07-19): faster-whisper berekent no_speech_prob
    ÉÉN keer per intern decodeer-blok (~30s), niet per zin -- als zo'n blok later in
    meerdere tekstzinnen wordt opgesplitst, erven ze allemaal diezelfde waarde.
    Bevestigd via reproductie: 19 zinnen in twee blokken kregen elk exact dezelfde
    no_speech_prob (0.921875 resp. 0.9306640625) ondanks compleet verschillende,
    prima tekst -- en in alle 19 gevallen was avg_logprob/compression_ratio wél
    goed. De incrementele batch-worker heeft dit lek niet: zijn vensters starten
    altijd op een stilte-grens (daar worden ze juist door getriggerd), dus hun
    no_speech_prob is al van nature representatief. transcribe_full()'s eigen
    interne 30s-hakking is niet stilte-uitgelijnd, dus een blokgrens kan toevallig
    net na een korte pauze vallen en een heel blok vol prima zinnen onterecht
    afkeuren op deze ene metriek."""
    if not text:
        return False, "empty_text"

    ok_logprob = (avg_logprob is None) or (avg_logprob > -1.3)
    ok_compr = (compression_ratio is None) or (compression_ratio < 2.4)
    ok_no_speech = (
        no_speech_threshold is None or
        no_speech_prob is None or
        no_speech_prob < no_speech_threshold or
        # zie _batch_worker(): hoge-kwaliteit audio (bv. Turks materiaal) scoort
        # structureel hoger op no_speech_prob ondanks uitstekende logprob.
        (no_speech_prob < 0.92 and avg_logprob is not None and avg_logprob > -0.3)
    )
    has_hallucination_pattern = any(p in text for p in HALLUCINATION_PATTERNS)
    MIN_CHARS_PER_SECOND = 1.0
    MIN_DURATION_FOR_DENSITY_CHECK_S = 3.0
    ok_density = (
        duration_s is None or
        duration_s < MIN_DURATION_FOR_DENSITY_CHECK_S or
        (len(text) / duration_s) >= MIN_CHARS_PER_SECOND
    )

    if ok_logprob and ok_compr and ok_no_speech and ok_density and not has_hallucination_pattern:
        return True, "accepted"

    reasons = []
    if not ok_logprob:
        reasons.append(f"logprob={avg_logprob}")
    if not ok_compr:
        reasons.append(f"compr={compression_ratio}")
    if not ok_no_speech:
        reasons.append(f"no_speech_prob={no_speech_prob}")
    if not ok_density:
        reasons.append(f"implausibly_short_for_duration=len={len(text)}/{duration_s:.1f}s")
    if has_hallucination_pattern:
        reasons.append("hallucination_pattern")
    return False, ",".join(reasons)


class SimulStreamingOnlineProcessor:
    """Online processor for SimulStreaming ASR."""
    SAMPLING_RATE = 16000

    def __init__(self, asr, logfile=sys.stderr, language: Optional[str] = None):
        self.logger = logging.getLogger("whisperlivekit.backend.SimulStreamingOnlineProcessor")
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("🔥 SimulStreamingOnlineProcessor logger ACTIVE, DEBUG level 🔥")
        self.asr = asr
        self.logfile = logfile
        self.end = 0.0
        self.buffer = []
        self.committed: List[ASRToken] = []
        self.last_result_tokens: List[ASRToken] = []
        # Per-sessie taaloverride: kopieer cfg zodat de singleton niet wordt gewijzigd
        self._session_cfg = copy.copy(asr.cfg)
        if language and language != self._session_cfg.language:
            self._session_cfg.language = language
        self.model = self._create_alignatt()

        # GT Added for debug
        self.logger.debug("=== INITIALIZING STREAMING DECODER ===")
        self.logger.debug(f"Decoder type: {self._session_cfg.decoder_type}")
        self.logger.debug(f"Beam size: {self._session_cfg.beam_size}")
        self.logger.debug(f"Language: {self._session_cfg.language} (singleton was: {asr.cfg.language})")
        self.logger.debug(f"Task: {self._session_cfg.task}")
        self.logger.debug(f"Tokenizer multilingual: {self._session_cfg.tokenizer_is_multilingual}")
        self.logger.debug(f"Audio min length: {self._session_cfg.audio_min_len}")
        self.logger.debug(f"Audio max length: {self._session_cfg.audio_max_len}")
        self.logger.debug("=== END STREAMING DECODER INIT ===")

        if asr.tokenizer and language is None:
            # Alleen singleton-tokenizer hergebruiken als taal niet is overschreven
            self.model.tokenizer = asr.tokenizer
            self.model.state.tokenizer = asr.tokenizer

    def _create_alignatt(self):
        """Create the AlignAtt decoder instance based on ASR mode."""
        if self.asr.use_full_mlx and HAS_MLX_WHISPER:
            return MLXAlignAtt(cfg=self._session_cfg, mlx_model=self.asr.mlx_model)
        else:
            return AlignAtt(
                cfg=self._session_cfg,
                loaded_model=self.asr.shared_model,
                mlx_encoder=self.asr.mlx_encoder,
                fw_encoder=self.asr.fw_encoder,
            )

    def start_silence(self):
        tokens, processed_upto = self.process_iter(is_last=True)
        return tokens, processed_upto

    def end_silence(self, silence_duration, offset):
        """Handle silence period."""
        self.end += silence_duration
        long_silence = silence_duration >= MIN_DURATION_REAL_SILENCE
        if not long_silence:
            gap_len = int(16000 * silence_duration)
            if gap_len > 0:
                if self.asr.use_full_mlx:
                    gap_silence = np.zeros(gap_len, dtype=np.float32)
                else:
                    gap_silence = torch.zeros(gap_len)
                self.model.insert_audio(gap_silence)
        if long_silence:
            self.model.refresh_segment(complete=True)
            self.model.global_time_offset = silence_duration + offset

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time):
        """Append an audio chunk to be processed by SimulStreaming."""
        self.end = audio_stream_end_time
        if self.asr.use_full_mlx:
            self.model.insert_audio(audio)
        else:
            audio_tensor = torch.from_numpy(audio).float()
            self.model.insert_audio(audio_tensor)

    def new_speaker(self, change_speaker: ChangeSpeaker):
        """Handle speaker change event."""
        self.process_iter(is_last=True)
        self.model.refresh_segment(complete=True)
        self.model.speaker = change_speaker.speaker
        self.model.global_time_offset = change_speaker.start
            
    def get_buffer(self):
        concat_buffer = Transcript.from_tokens(tokens= self.buffer, sep='')
        return concat_buffer

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        """
        Process accumulated audio chunks using SimulStreaming.
        
        Returns a tuple: (list of committed ASRToken objects, float representing the audio processed up to time).
        """
        try:
            timestamped_words = self.model.infer(is_last=is_last)
            
            if not timestamped_words:
                return [], self.end
            
            if self.model.cfg.language == "auto" and timestamped_words[0].detected_language is None:
                self.buffer.extend(timestamped_words)
                return [], self.end
            
            self.committed.extend(timestamped_words)
            self.buffer = []
            return timestamped_words, self.end
        except RuntimeError as e:
            # Handle CUDA OOM gracefully
            msg = str(e).lower()
            if "out of memory" in msg or "cuda failed with error out of memory" in msg:
                logger.error(f"CUDA OOM in process_iter (is_last={is_last}). Resetting decoder and clearing cache.")
                try:
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                # Reset streaming decoder state so we can continue
                try:
                    self.model.refresh_segment(complete=True)
                except Exception:
                    pass

                # Drop current buffer content (avoid repeating the same huge batch)
                self.buffer = []
                return [], self.end

            # Not an OOM -> rethrow to be handled below/logged
            raise

        except Exception as e:
            logger.exception(f"SimulStreaming processing error: {e}")
            return [], self.end


    def warmup(self, audio, init_prompt=""):
        """Warmup the SimulStreaming model."""
        try:
            if self.asr.use_full_mlx:
                # MLX mode: ensure numpy array
                if hasattr(audio, 'numpy'):
                    audio = audio.numpy()
            self.model.insert_audio(audio)
            self.model.infer(True)
            self.model.refresh_segment(complete=True)
            logger.info("SimulStreaming model warmed up successfully")
        except Exception as e:
            logger.exception(f"SimulStreaming warmup failed: {e}")

    def __del__(self):
        gc.collect()
        if not getattr(self.asr, 'use_full_mlx', True) and torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


class SimulStreamingASR:
    """SimulStreaming backend with AlignAtt policy."""
    sep = ""

    def __init__(self, logfile=sys.stderr, **kwargs):
        self.logfile = logfile
        self.transcribe_kargs = {}
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.decoder_type is None:
            self.decoder_type = 'greedy' if self.beams == 1 else 'beam'

        self.fast_encoder = False
        self._resolved_model_path = None
        self.encoder_backend = "whisper"
        self.use_full_mlx = getattr(self, "use_full_mlx", False)
        preferred_backend = getattr(self, "backend", "auto")
        compatible_whisper_mlx, compatible_faster_whisper = True, True
        
        if self.model_path:
            resolved_model_path = resolve_model_path(self.model_path)
            self._resolved_model_path = resolved_model_path
            self.model_path = str(resolved_model_path)
            
            model_info = detect_model_format(resolved_model_path)
            compatible_whisper_mlx = model_info.compatible_whisper_mlx
            compatible_faster_whisper = model_info.compatible_faster_whisper
            
            if not self.use_full_mlx and not model_info.has_pytorch:
                raise FileNotFoundError(
                    f"No PyTorch checkpoint (.pt/.bin/.safetensors) found under {self.model_path}"
                )            
            self.model_name = resolved_model_path.name if resolved_model_path.is_dir() else resolved_model_path.stem
        elif self.model_size is not None:
            self.model_name = self.model_size
        else:
            raise ValueError("Either model_size or model_path must be specified for SimulStreaming.")

        is_multilingual = not self.model_name.endswith(".en")

        self.encoder_backend = self._resolve_encoder_backend(
            preferred_backend,
            compatible_whisper_mlx,
            compatible_faster_whisper,
        )
        self.fast_encoder = self.encoder_backend in ("mlx-whisper", "faster-whisper")
        if self.encoder_backend == "whisper":
            self.disable_fast_encoder = True
        
        if self.encoder_backend == "mlx-whisper" and platform.system() == "Darwin":
            if not hasattr(self, '_full_mlx_disabled'):
                self.use_full_mlx = True
                    
        self.cfg = AlignAttConfig(
                tokenizer_is_multilingual= is_multilingual,
                segment_length=self.min_chunk_size,
                frame_threshold=self.frame_threshold,
                language=self.lan,
                audio_max_len=self.audio_max_len,
                audio_min_len=self.audio_min_len,
                cif_ckpt_path=self.cif_ckpt_path,
                decoder_type="beam",
                beam_size=self.beams,
                task=self.task,
                never_fire=self.never_fire,
                init_prompt=self.init_prompt,
                max_context_tokens=self.max_context_tokens,
                static_init_prompt=self.static_init_prompt,
        )  
        
        # Set up tokenizer for translation if needed
        if self.direct_english_translation:
            self.tokenizer = self.set_translate_task()
        else:
            self.tokenizer = None

        self.mlx_encoder, self.fw_encoder, self.mlx_model = None, None, None
        self.shared_model = None
        
        if self.use_full_mlx and HAS_MLX_WHISPER:
            logger.info('MLX Whisper backend used.')
            if self._resolved_model_path is not None:
                mlx_model_path = str(self._resolved_model_path)
            else:
                mlx_model_path = mlx_model_mapping.get(self.model_name)
            if not mlx_model_path:
                raise FileNotFoundError(
                    f"MLX Whisper backend requested but no compatible weights found for model '{self.model_name}'."
                )
            self.mlx_model = load_mlx_model(path_or_hf_repo=mlx_model_path)
            self._warmup_mlx_model()
        elif self.encoder_backend == "mlx-whisper":
            # hybrid mode: mlx encoder + pytorch decoder
            logger.info('SimulStreaming will use MLX Whisper encoder with PyTorch decoder.')
            if self._resolved_model_path is not None:
                mlx_model_path = str(self._resolved_model_path)
            else:
                mlx_model_path = mlx_model_mapping.get(self.model_name)
            if not mlx_model_path:
                raise FileNotFoundError(
                    f"MLX Whisper backend requested but no compatible weights found for model '{self.model_name}'."
                )
            self.mlx_encoder = load_mlx_encoder(path_or_hf_repo=mlx_model_path)
            self.shared_model = self.load_model()
        elif self.encoder_backend == "faster-whisper":
            print('SimulStreaming will use Faster Whisper for the encoder.')
            if self._resolved_model_path is not None:
                fw_model = str(self._resolved_model_path)
            else:
                fw_model = self.model_name
            self.fw_encoder = WhisperModel(
                fw_model,
                device='auto',
                compute_type='auto',
            )
            self.shared_model = self.load_model()
        else:
            self.shared_model = self.load_model()
    
    def _warmup_mlx_model(self):
        """Warmup the full MLX model."""
        warmup_audio = load_file(self.warmup_file)
        if warmup_audio is not None:
            temp_model = MLXAlignAtt(
                cfg=self.cfg,
                mlx_model=self.mlx_model,
            )
            temp_model.warmup(warmup_audio)
            logger.info("Full MLX model warmed up successfully")


    def _resolve_encoder_backend(self, preferred_backend, compatible_whisper_mlx, compatible_faster_whisper):
        choice = preferred_backend or "auto"
        if self.disable_fast_encoder:
            return "whisper"
        if choice == "whisper":
            return "whisper"
        if choice == "mlx-whisper":
            if not self._can_use_mlx(compatible_whisper_mlx):
                raise RuntimeError("mlx-whisper backend requested but MLX Whisper is unavailable or incompatible with the provided model.")
            return "mlx-whisper"
        if choice == "faster-whisper":
            if not self._can_use_faster(compatible_faster_whisper):
                raise RuntimeError("faster-whisper backend requested but Faster-Whisper is unavailable or incompatible with the provided model.")
            return "faster-whisper"
        if choice == "openai-api":
            raise ValueError("openai-api backend is only supported with the LocalAgreement policy.")
        # auto mode
        if platform.system() == "Darwin" and self._can_use_mlx(compatible_whisper_mlx):
            return "mlx-whisper"
        if self._can_use_faster(compatible_faster_whisper):
            return "faster-whisper"
        return "whisper"

    def _has_custom_model_path(self):
        return self._resolved_model_path is not None

    def _can_use_mlx(self, compatible_whisper_mlx):
        if not HAS_MLX_WHISPER:
            return False
        if self._has_custom_model_path():
            return compatible_whisper_mlx
        return self.model_name in mlx_model_mapping

    def _can_use_faster(self, compatible_faster_whisper):
        if not HAS_FASTER_WHISPER:
            return False
        if self._has_custom_model_path():
            return compatible_faster_whisper
        return True

    def load_model(self):
        model_ref = str(self._resolved_model_path) if self._resolved_model_path else self.model_name
        lora_path = getattr(self, 'lora_path', None)
        whisper_model = load_model(
            name=model_ref,
            download_root=None,
            decoder_only=self.fast_encoder,
            custom_alignment_heads=self.custom_alignment_heads,
            lora_path=lora_path,
        )
        warmup_audio = load_file(self.warmup_file)
        if warmup_audio is not None:
            warmup_audio = torch.from_numpy(warmup_audio).float()
            if self.fast_encoder:
                temp_model = AlignAtt(
                    cfg=self.cfg,
                    loaded_model=whisper_model,
                    mlx_encoder=self.mlx_encoder,
                    fw_encoder=self.fw_encoder,
                )
                temp_model.warmup(warmup_audio)
            else:
                whisper_model.transcribe(warmup_audio, language=self.lan if self.lan != 'auto' else None)
        return whisper_model

    def set_translate_task(self):
        """Set up translation task."""
        if self.cfg.language == 'auto':
            raise Exception('Translation cannot be done with language = auto')
        return tokenizer.get_tokenizer(
            multilingual=True,
            language=self.cfg.language,
            num_languages=99,
            task="translate"
        )

    def transcribe(self, audio):
        """
        Warmup is done directly in load_model
        """
        pass
