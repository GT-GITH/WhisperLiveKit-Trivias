from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class AlignAttConfig():
    eval_data_path: str = "tmp"
    segment_length: float = field(default=1.0, metadata = {"help": "in second"})
    frame_threshold: int = 4
    rewind_threshold: int = 120
    audio_max_len: float = 30.0
    cif_ckpt_path: str = ""
    never_fire: bool = False
    language: str = field(default="zh")
    nonspeech_prob: float = 0.5
    audio_min_len: float = 2.0
    decoder_type: Literal["greedy","beam"] = "beam"
    beam_size: int = 3
    task: Literal["transcribe","translate"] = "transcribe"
    tokenizer_is_multilingual: bool = True
    init_prompt: str = field(default=None)
    static_init_prompt: str = field(default=None)
    max_context_tokens: int = field(default=None)
    
# ------------------------------------------------------------
# Trivias STT channel-aware quality config (simple v1)
# ------------------------------------------------------------

TaskType = Literal["transcribe", "translate"]


@dataclass
class ChannelTranscriptionConfig:
    language: Optional[str] = "nl"   # None of "auto" mag ook
    task: TaskType = "transcribe"

    # Live / SimulStreaming
    live_frame_threshold: int = 25
    live_audio_min_len: float = 0.0
    live_audio_max_len: float = 30.0
    live_beams: int = 1
    live_decoder_type: Optional[Literal["greedy", "beam"]] = None
    live_init_prompt: Optional[str] = None
    live_static_init_prompt: Optional[str] = None
    live_max_context_tokens: Optional[int] = None
    live_never_fire: bool = False
    live_cif_ckpt_path: Optional[str] = None

    # Batch
    batch_beam_size: int = 7
    batch_temperature: list[float] = field(default_factory=lambda: [0.0, 0.2])
    batch_condition_on_previous_text: bool = False
    batch_initial_prompt: Optional[str] = None


CHANNEL_CONFIGS: dict[str, ChannelTranscriptionConfig] = {
    "default": ChannelTranscriptionConfig(
        language="nl",
        task="transcribe",
        live_frame_threshold=25,
        live_audio_min_len=0.0,
        live_audio_max_len=30.0,
        live_beams=1,
        live_decoder_type="beam",
        batch_beam_size=7,
        batch_temperature=[0.0, 0.2],
        batch_condition_on_previous_text=False,
        batch_initial_prompt=None,
    ),

    "interpreter": ChannelTranscriptionConfig(
        language="nl",
        task="transcribe",
        live_frame_threshold=25,
        live_audio_min_len=0.0,
        live_decoder_type="beam",
        batch_beam_size=7,
        batch_temperature=[0.0, 0.2],
        batch_condition_on_previous_text=False,
    ),
    "lawyer": ChannelTranscriptionConfig(
        language="nl",
        task="transcribe",
        live_frame_threshold=25,
        live_audio_min_len=0.0,
        live_decoder_type="beam",
        batch_beam_size=7,
        batch_temperature=[0.0, 0.2],
        batch_condition_on_previous_text=False,
    ),

    # Voorbeeldrollen / toekomstige uitbreiding
    "employee": ChannelTranscriptionConfig(
        language="nl",
        task="transcribe",
        live_frame_threshold=25,
        live_audio_min_len=0.0,
        live_decoder_type="greedy",
        batch_initial_prompt=None,
    ),
    "foreign_nl": ChannelTranscriptionConfig(
        language="nl",
        task="transcribe",
        live_frame_threshold=25,
        live_audio_min_len=0.0,
        live_decoder_type="greedy",
        batch_initial_prompt=None,
    ),
    "foreign_ar": ChannelTranscriptionConfig(
        language="ar",
        task="transcribe",
        live_frame_threshold=25,
        live_audio_min_len=0.0,
        live_decoder_type="greedy",
        batch_initial_prompt=None,
    ),
    "foreign_fa": ChannelTranscriptionConfig(
        language="fa",
        task="transcribe",
        live_frame_threshold=25,
        live_audio_min_len=0.0,
        live_decoder_type="greedy",
        batch_initial_prompt=None,
    ),
    "foreign_ru": ChannelTranscriptionConfig(
        language="ru",
        task="transcribe",
        live_frame_threshold=25,
        live_audio_min_len=0.0,
        live_decoder_type="greedy",
        batch_initial_prompt=None,
    ),
    "foreign_tr": ChannelTranscriptionConfig(
        language="tr",
        task="transcribe",
        live_frame_threshold=25,
        live_audio_min_len=0.0,
        live_decoder_type="greedy",
        batch_initial_prompt=None,
    ),
}


def get_channel_config(channel_id: Optional[str]) -> ChannelTranscriptionConfig:
    """
    Resolve a channel config with fallback to 'default'.
    """
    if not channel_id:
        return CHANNEL_CONFIGS["default"]
    return CHANNEL_CONFIGS.get(channel_id, CHANNEL_CONFIGS["default"])