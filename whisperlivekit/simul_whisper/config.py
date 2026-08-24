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
    # Modelroutering (fase 1 van het voorstel, batch-only): optioneel pad/HF-repo
    # naar een taalspecifiek CTranslate2-model voor de batch-pass van dit kanaal.
    # None (default, elk bestaand kanaal) = ongewijzigd gedrag: het server-brede
    # standaardmodel (--model/--model-path) wordt gebruikt, exact zoals vandaag.
    # Alleen kanalen die dit veld expliciet invullen krijgen een eigen, apart
    # geladen batch-model (zie TranscriptionEngine.get_batch_asr_for_channel()).
    batch_model_path: Optional[str] = None


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

    # PoC modelroutering (zie voorstel): expliciete vermelding zodat er een
    # concrete plek is om batch_model_path in te vullen. Vandaag nog None =
    # ongewijzigd gedrag, identiek aan de generieke "foreign"-fallback hieronder.
    # Vul in zodra een CTranslate2-conversie van bv.
    # microsoft/paza-whisper-large-v3-turbo beschikbaar is (lokaal pad of een
    # al als CT2 gepubliceerde HF-repo), bv.
    # batch_model_path="/models/paza-whisper-large-v3-turbo-ct2".
    "foreign_so": ChannelTranscriptionConfig(
        language="so",
        task="transcribe",
        live_frame_threshold=25,
        live_audio_min_len=0.0,
        live_decoder_type="greedy",
        batch_initial_prompt=None,
        batch_model_path=None,
    ),

    # Generieke fallback voor elk "foreign_<taalcode>"-kanaal zonder eigen,
    # specifiek afgestelde vermelding hierboven (zie get_channel_config()).
    # app.js's LANGUAGES-lijst biedt 13 talen voor de rol "Vreemdeling", maar
    # hierboven staan er maar 4 specifiek uitgewerkt (nl/ar/fa/ru, onderling nu
    # al identiek op de taalcode na) -- voor de overige 9 (bv. "so", "tr") viel
    # get_channel_config() stilzwijgend terug op "default", een rolconfig die
    # niet voor "Vreemdeling" bedoeld is. language=None: dit veld wordt voor
    # foreign_*-kanalen in de praktijk nooit gebruikt (TriviasServer.py's
    # _resolve_channel_language() haalt de taal al rechtstreeks uit de
    # channel_id zelf, vóórdat deze config geraadpleegd zou worden), dus een
    # gok als "nl" laten staan zou misleidend zijn.
    "foreign": ChannelTranscriptionConfig(
        language=None,
        task="transcribe",
        live_frame_threshold=25,
        live_audio_min_len=0.0,
        live_decoder_type="greedy",
        batch_initial_prompt=None,
    ),
}


def get_channel_config(channel_id: Optional[str]) -> ChannelTranscriptionConfig:
    """
    Volgorde:
    1. Exacte match in CHANNEL_CONFIGS -- een bewust per-taal afgestelde
       vermelding (zoals de bestaande foreign_nl/ar/fa/ru, of een toekomstige
       voor bv. foreign_so als iemand die ooit specifiek wil afstellen).
    2. Elk ander "foreign_<taalcode>"-kanaal (zie getChannelId() in app.js --
       de talenlijst daar staat los van deze backend-config en kan zonder
       hier iets aan te passen uitbreiden): de generieke "foreign"-rolconfig,
       i.p.v. stilzwijgend "default" (een andere rol) te gebruiken.
    3. Overal anders: "default".
    """
    if not channel_id:
        return CHANNEL_CONFIGS["default"]
    if channel_id in CHANNEL_CONFIGS:
        return CHANNEL_CONFIGS[channel_id]
    if channel_id.startswith("foreign_"):
        return CHANNEL_CONFIGS["foreign"]
    return CHANNEL_CONFIGS["default"]