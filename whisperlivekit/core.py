import json
import logging
import sys
from argparse import Namespace

from whisperlivekit.local_agreement.online_asr import OnlineASRProcessor
from whisperlivekit.local_agreement.whisper_online import backend_factory
from whisperlivekit.simul_whisper import SimulStreamingASR
from whisperlivekit.simul_whisper.backend import BatchFasterWhisperASR
from whisperlivekit.simul_whisper.config import get_channel_config, resolve_language_for_routing


def update_with_kwargs(_dict, kwargs):
    _dict.update({
        k: v for k, v in kwargs.items() if k in _dict
    })
    return _dict


logger = logging.getLogger(__name__)

class TranscriptionEngine:
    
    def __init__(self, **kwargs):

        global_params = {
            "host": "localhost",
            "port": 8000,
            "diarization": False,
            "punctuation_split": False,
            "target_language": "",
            "vac": True,
            "vac_chunk_size": 0.04,
            "log_level": "DEBUG",
            "ssl_certfile": None,
            "ssl_keyfile": None,
            "forwarded_allow_ips": None,
            "transcription": True,
            "vad": True,
            "pcm_input": False,
            "disable_punctuation_split" : False,
            "diarization_backend": "sortformer",
            "backend_policy": "simulstreaming",
            "backend": "auto",
            "channel_id": "default",
        }
        global_params = update_with_kwargs(global_params, kwargs)

        transcription_common_params = {
            "warmup_file": None,
            "min_chunk_size": 0.1,
            "model_size": "base",
            "model_cache_dir": None,
            "model_dir": None,
            "model_path": None,
            "lora_path": None,
            "lan": "auto",
            "direct_english_translation": False,
        }
        transcription_common_params = update_with_kwargs(transcription_common_params, kwargs)                                            

        if transcription_common_params['model_size'].endswith(".en"):
            transcription_common_params["lan"] = "en"
        if 'no_transcription' in kwargs:
            global_params['transcription'] = not kwargs['no_transcription']
        if 'no_vad' in kwargs:
            global_params['vad'] = not kwargs['no_vad']
        if 'no_vac' in kwargs:
            global_params['vac'] = not kwargs['no_vac']

        self.args = Namespace(**{**global_params, **transcription_common_params})
        self.channel_cfg = get_channel_config(getattr(self.args, "channel_id", "default"))
        logger.info("=== CHANNEL CONFIG ===")
        logger.info(f"live_frame_threshold={self.channel_cfg.live_frame_threshold}")
        logger.info(f"live_audio_min_len={self.channel_cfg.live_audio_min_len}")
        logger.info(f"live_audio_max_len={self.channel_cfg.live_audio_max_len}")
        logger.info(f"live_beams={self.channel_cfg.live_beams}")
        logger.info(f"live_decoder_type={self.channel_cfg.live_decoder_type}")
        logger.info("=== END CHANNEL CONFIG ===")
        if getattr(self.args, "lan", None) in (None, "", "auto"):
            self.args.lan = self.channel_cfg.language
        self.args.task = self.channel_cfg.task
        self.asr = None
        self.tokenizer = None
        self.diarization = None
        self.vac_session = None
        
        if self.args.vac:
            from whisperlivekit.silero_vad_iterator import is_onnx_available
            
            if is_onnx_available():
                from whisperlivekit.silero_vad_iterator import load_onnx_session
                self.vac_session = load_onnx_session()
            else:
                logger.warning(
                    "onnxruntime not installed. VAC will use JIT model which is loaded per-session. "
                    "For multi-user scenarios, install onnxruntime: pip install onnxruntime"
                )
        backend_policy = self.args.backend_policy
        if self.args.transcription:

            self.batch_asr = None
            # Modelroutering (generiek, per taal): registry van extra
            # BatchFasterWhisperASR-instances, lazily gebouwd per uniek modelpad.
            # Kanalen/talen zonder override blijven op self.batch_asr, exact het
            # bestaande gedrag.
            self._batch_asr_registry: dict = {}
            # {taalcode: ct2_model_pad}, handmatig bijgehouden na validatie (zie
            # project-memory "modelroutering-poc") -- geen taal krijgt een
            # specialisatie tenzij expliciet in dit bestand vermeld. Bewust geen
            # automatische mapnaam-detectie: een specialisatie is een bewuste,
            # geverifieerde beslissing, geen bijeffect van wat er toevallig op
            # schijf staat.
            self._language_batch_model_registry: dict = {}
            registry_path = kwargs.get("batch_model_registry")
            if registry_path:
                try:
                    with open(registry_path, "r", encoding="utf-8") as f:
                        self._language_batch_model_registry = json.load(f)
                    logger.info(
                        "Modelroutering: batch_model_registry geladen (%d taal/talen) uit %s: %s",
                        len(self._language_batch_model_registry), registry_path,
                        list(self._language_batch_model_registry.keys()),
                    )
                except Exception as e:
                    logger.warning(
                        "Modelroutering: kon batch_model_registry niet laden (%s) -- "
                        "ga verder zonder taalspecifieke routering: %s", registry_path, e,
                    )
            if backend_policy == "simulstreaming":               
                simulstreaming_params = {
                    "disable_fast_encoder": False,
                    "custom_alignment_heads": None,
                    "frame_threshold": self.channel_cfg.live_frame_threshold,
                    "beams": self.channel_cfg.live_beams,
                    "decoder_type": self.channel_cfg.live_decoder_type,
                    "audio_max_len": self.channel_cfg.live_audio_max_len,
                    "audio_min_len": self.channel_cfg.live_audio_min_len,
                    "cif_ckpt_path": self.channel_cfg.live_cif_ckpt_path,
                    "never_fire": self.channel_cfg.live_never_fire,
                    "init_prompt": self.channel_cfg.live_init_prompt,
                    "static_init_prompt": self.channel_cfg.live_static_init_prompt,
                    "max_context_tokens": self.channel_cfg.live_max_context_tokens,
                }
                # Alleen expliciete overrides toestaan, niet defaults
                allowed_live_overrides = {"custom_alignment_heads", "disable_fast_encoder"}

                for k, v in kwargs.items():
                    if k in allowed_live_overrides and v is not None:
                        simulstreaming_params[k] = v                          
                
                self.tokenizer = None    
                transcription_common_params["task"] = self.args.task    
                self.asr = SimulStreamingASR(
                    **transcription_common_params,
                    **simulstreaming_params,
                    backend=self.args.backend,
                )
                logger.info(
                    "Using SimulStreaming policy with %s backend",
                    getattr(self.asr, "encoder_backend", "whisper"),
                )

                # batch gebruikt dezelfde weights als je encoder/model keuze
                model_for_batch = self.args.model_path or self.args.model_size
                self.batch_asr = BatchFasterWhisperASR(
                    model=model_for_batch,
                    language=self.channel_cfg.language,
                    beam_size=self.channel_cfg.batch_beam_size,
                    condition_on_previous_text=self.channel_cfg.batch_condition_on_previous_text,
                    temperature=self.channel_cfg.batch_temperature,
                    initial_prompt=self.channel_cfg.batch_initial_prompt or None,
                    #best_of=5,
                    #patience=1.2,
                    #length_penalty=0.6,
                    #no_speech_threshold=0.6,
                    #log_prob_threshold=-1.0,
                    #compression_ratio_threshold=2.4,
                )
            else:
                
                whisperstreaming_params = {
                    "buffer_trimming": "segment",
                    "confidence_validation": False,
                    "buffer_trimming_sec": 15,
                }
                whisperstreaming_params = update_with_kwargs(whisperstreaming_params, kwargs)
                
                self.asr = backend_factory(
                    backend=self.args.backend,
                    **transcription_common_params,
                    **whisperstreaming_params,
                )
                logger.info(
                    "Using LocalAgreement policy with %s backend",
                    getattr(self.asr, "backend_choice", self.asr.__class__.__name__),
                )

        if self.args.diarization:
            if self.args.diarization_backend == "diart":
                from whisperlivekit.diarization.diart_backend import \
                    DiartDiarization
                diart_params = {
                    "segmentation_model": "pyannote/segmentation-3.0",
                    "embedding_model": "pyannote/embedding",
                }
                diart_params = update_with_kwargs(diart_params, kwargs)
                self.diarization_model = DiartDiarization(
                    block_duration=self.args.min_chunk_size,
                    **diart_params
                )
            elif self.args.diarization_backend == "sortformer":
                from whisperlivekit.diarization.sortformer_backend import \
                    SortformerDiarization
                self.diarization_model = SortformerDiarization()
        
        self.translation_model = None
        if self.args.target_language:
            if self.args.lan == 'auto' and backend_policy != "simulstreaming":
                raise Exception('Translation cannot be set with language auto when transcription backend is not simulstreaming')
            else:
                try:
                    from nllw import load_model
                except:
                    raise Exception('To use translation, you must install nllw: `pip install nllw`')
                translation_params = { 
                    "nllb_backend": "transformers",
                    "nllb_size": "600M"
                }
                translation_params = update_with_kwargs(translation_params, kwargs)
                self.translation_model = load_model([self.args.lan], **translation_params) #in the future we want to handle different languages for different speakers

    def get_batch_asr_for_channel(self, channel_id: str):
        """
        Modelroutering (generiek, per taal -- batch-only). Volgorde:
        1. channel_cfg.batch_model_path -- expliciete per-kanaal override, voor
           het zeldzame geval dat één specifiek kanaal (i.p.v. een hele taal)
           een eigen model nodig heeft.
        2. self._language_batch_model_registry -- de taal van dit kanaal
           (resolve_language_for_routing()) opgezocht in het via
           --batch-model-registry geladen JSON-bestand.
        3. self.batch_asr -- het server-brede standaardmodel, exact het gedrag
           van vóór deze methode bestond.
        Elk model wordt lazily gebouwd (eerste keer per uniek pad) en daarna
        hergebruikt. De live/AlignAtt-route blijft bewust ongemoeid.
        """
        if self.batch_asr is None:
            return None
        channel_cfg = get_channel_config(channel_id)
        model_path = channel_cfg.batch_model_path
        if not model_path:
            language = resolve_language_for_routing(channel_id)
            if language:
                model_path = self._language_batch_model_registry.get(language)
        if not model_path:
            return self.batch_asr
        if model_path not in self._batch_asr_registry:
            logger.info(
                "Modelroutering: laad apart batch-model voor channel_id=%s taal=%s model=%s",
                channel_id, resolve_language_for_routing(channel_id), model_path,
            )
            self._batch_asr_registry[model_path] = BatchFasterWhisperASR(
                model=model_path,
                language=channel_cfg.language,
                beam_size=channel_cfg.batch_beam_size,
                condition_on_previous_text=channel_cfg.batch_condition_on_previous_text,
                temperature=channel_cfg.batch_temperature,
                initial_prompt=channel_cfg.batch_initial_prompt or None,
            )
        return self._batch_asr_registry[model_path]


def online_factory(args, asr, language: str = None):
    if args.backend_policy == "simulstreaming":
        from whisperlivekit.simul_whisper import SimulStreamingOnlineProcessor
        return SimulStreamingOnlineProcessor(asr, language=language)
    return OnlineASRProcessor(asr)
  
  
def online_diarization_factory(args, diarization_backend):
    if args.diarization_backend == "diart":
        online = diarization_backend
        # Not the best here, since several user/instances will share the same backend, but diart is not SOTA anymore and sortformer is recommended
    
    if args.diarization_backend == "sortformer":
        from whisperlivekit.diarization.sortformer_backend import \
            SortformerDiarizationOnline
        online = SortformerDiarizationOnline(shared_model=diarization_backend)
    return online


def online_translation_factory(args, translation_model):
    #should be at speaker level in the future:
    #one shared nllb model for all speaker
    #one tokenizer per speaker/language
    from nllw import OnlineTranslation
    return OnlineTranslation(translation_model, [args.lan], [args.target_language])
