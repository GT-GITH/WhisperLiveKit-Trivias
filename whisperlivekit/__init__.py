import os as _os

# Begrens PyTorch/NumPy/BLAS CPU-threads (2026-08-23, CPU-piek-onderzoek): zonder
# dit probeert PyTorch op machines met veel cores voor elke kleine CPU-zijdige
# bewerking tijdens live-decoderen (feature-extractie, tensor-prep rond de GPU-
# forward-pass) standaard alle beschikbare cores te gebruiken -- gemeten tot
# 1300% CPU-gebruik, aanhoudend zolang er gesproken wordt. Moet vóór de torch/
# numpy-imports hieronder gezet worden, anders lezen OpenMP/MKL dit niet meer.
# Respecteert een expliciet door de operator gezette waarde.
_default_threads = str(min(4, _os.cpu_count() or 4))
for _env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_env_var, _default_threads)

from .audio_processor import AudioProcessor
from .core import TranscriptionEngine
from .parse_args import parse_args
from .web_trivias.web_interface import get_inline_ui_html, get_web_interface_html

try:
    import torch as _torch
    _torch.set_num_threads(int(_os.environ.get("OMP_NUM_THREADS", _default_threads)))
except Exception:
    pass  # nooit de server-start blokkeren op een tuning-instelling

__all__ = [
    "TranscriptionEngine",
    "AudioProcessor",
    "parse_args",
    "get_web_interface_html",
    "get_inline_ui_html",
    "download_simulstreaming_backend",
]
