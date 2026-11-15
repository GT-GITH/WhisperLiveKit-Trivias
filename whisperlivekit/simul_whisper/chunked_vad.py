"""Spraakdetectie voor de batch-paden, met frisse VAD-toestand per kort stuk.

Waarom (bewezen 2026-11-15, live YouTube-test + offline reproductie):
faster-whisper's vad_filter draait Silero-VAD één keer over de hele invoer, en
Silero heeft een recurrente toestand. Na een lange periode muziek of zacht geluid
raakt die "verdoofd" en mist hij de spraak die volgt. Zelfde audio (305-316 s):
spraakkans 0.0 met 200 s voorafgaande context, 0.3-0.75 met 10 s context. Een
lagere drempel hielp niet (0.25 vond nog niets). Het refresh-pad (VAD over het
hele bestand) miste daardoor meer dan het incrementele pad (VAD per venster).

Daarom: VAD per kort, overlappend stuk (elk stuk begint met een frisse toestand),
en een tijdstip telt als spraak zodra één stuk er spraak ziet (vereniging). De
uitkomst gaat als clip_timestamps naar faster-whisper i.p.v. vad_filter. Bijkomend:
faster-whisper plakt de spraakstukken dan niet meer aan elkaar -- dat aaneenplakken
was de oorzaak van tijdstempels die over stiltes heen liepen (zie segment_split.py).

De detector is injecteerbaar zodat de samenvoeglogica zonder model te testen is.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

SR = 16000
CHUNK_S = 10.0
HOP_S = 5.0
# Zelfde samenvoeggedrag als faster-whisper's eigen VadOptions-default
# (min_silence_duration_ms=2000): spraak met < 2 s ertussen is één clip.
MERGE_GAP_S = 2.0
# Silero heeft aan het begin van de invoer (frisse toestand) een fractie van een
# seconde nodig om spraak te herkennen. Begint de spraak direct bij de venstergrens,
# dan start de eerste clip midden in een woord en slaat Whisper het hele eerste stuk
# over (bewezen 2026-11-15: venster 238.05-253.78 s, VAD-start 0.53 s terwijl de spraak
# op 0 s begon -> 1 van 11 zinnen; zelfde venster vanaf 0 s -> alle 11). Midden in het
# venster speelt dit niet: daar heeft het overlappende vorige stuk de toestand al warm.
EDGE_SNAP_S = 1.0  # begint de eerste clip binnen deze tijd na het begin: vanaf 0

Detector = Callable[["object"], List[dict]]  # audio-stuk -> [{"start": sample, "end": sample}]


def _default_detector():
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    opts = VadOptions()  # zelfde drempel/padding als faster-whisper's vad_filter

    def detect(chunk):
        return get_speech_timestamps(chunk, opts)
    return detect


def merge_intervals(intervals: List[Tuple[float, float]], max_gap: float = MERGE_GAP_S) -> List[Tuple[float, float]]:
    """Vereniging van (start, eind)-intervallen; gaten < max_gap worden gedicht."""
    out: List[Tuple[float, float]] = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if out and s - out[-1][1] < max_gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def speech_clips(audio, sr: int = SR, chunk_s: float = CHUNK_S, hop_s: float = HOP_S,
                 detector: Optional[Detector] = None) -> List[Tuple[float, float]]:
    """Spraakintervallen in seconden (absoluut in `audio`), gesorteerd en disjunct."""
    detect = detector or _default_detector()
    n, hop = int(chunk_s * sr), int(hop_s * sr)
    total = len(audio)
    found: List[Tuple[float, float]] = []
    off = 0
    while off < total:
        for t in detect(audio[off:off + n]):
            found.append(((off + t["start"]) / sr, min(total, off + t["end"]) / sr))
        if off + n >= total:
            break
        off += hop
    clips = merge_intervals(found)
    # Spraak die bij de venstergrens begint: clip vanaf 0 (zie EDGE_SNAP_S).
    if clips and 0.0 < clips[0][0] < EDGE_SNAP_S:
        clips[0] = (0.0, clips[0][1])
    return clips


def to_clip_timestamps(clips: List[Tuple[float, float]]) -> List[float]:
    """[(s, e), ...] -> [s, e, s, e, ...] zoals faster-whisper's clip_timestamps verwacht."""
    flat: List[float] = []
    for s, e in clips:
        flat.extend([round(s, 3), round(e, 3)])
    return flat
