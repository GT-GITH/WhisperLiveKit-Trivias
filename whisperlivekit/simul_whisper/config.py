# This code was originally in simul_whisper/transcriber/simul_whisper.py . It is adapted a lot for SimulStreaming.

from dataclasses import dataclass, field
from typing import Literal

@dataclass
class AlignAttConfig():
    eval_data_path: str = "tmp"
    segment_length: float = field(default=1.0, metadata = {"help": "in second"})
    frame_threshold: int = 8       #GT 4 > 8: commit pas later → vloeiender output
    rewind_threshold: int = 200
    audio_max_len: float = 30.0    # GT 20.0 > 30.0 meer context voor lange zinnen
    cif_ckpt_path: str = ""
    never_fire: bool = False
    language: str = field(default="nl")  #GT: Nederlands als standaard
    nonspeech_prob: float = 0.5
    audio_min_len: float = 0.5         # GT 1> 0.5 : sneller starten met transcriberen
    decoder_type: Literal["greedy","beam"] = "greedy"
    beam_size: int = 1     #GT 5 > 1 lage latency
    task: Literal["transcribe","translate"] = "transcribe"
    tokenizer_is_multilingual: bool = False
    init_prompt: str = field(default=None)
    static_init_prompt: str = field(default=None)
    max_context_tokens: int = field(default=None)
    