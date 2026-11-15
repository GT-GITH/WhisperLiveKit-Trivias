"""Tests voor whisperlivekit/simul_whisper/chunked_vad.py (zonder model: nep-detector).

Draai: `python tests/test_chunked_vad.py`.
"""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "whisperlivekit" / "simul_whisper" / "chunked_vad.py"
_spec = importlib.util.spec_from_file_location("chunked_vad", _PATH)
cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cv)

SR = 16000


class FakeAudio:
    """Alleen len() en slicing nodig; de nep-detector kijkt naar de offset."""
    def __init__(self, seconds):
        self.n = int(seconds * SR)
        self.offset = 0

    def __len__(self):
        return self.n

    def __getitem__(self, sl):
        a = FakeAudio(0)
        a.offset = sl.start or 0
        a.n = min(self.n, sl.stop) - a.offset
        return a


def make_detector(truth, deaf_after_s=None):
    """truth: echte spraak [(s,e)] in absolute seconden. deaf_after_s: simuleer
    Silero-verdoving -- spraak die later dan deaf_after_s na het BEGIN van het stuk
    valt, wordt niet gezien (zoals na lange muziek)."""
    def detect(chunk):
        c0, c1 = chunk.offset / SR, (chunk.offset + chunk.n) / SR
        out = []
        for s, e in truth:
            s2, e2 = max(s, c0), min(e, c1)
            if e2 <= s2:
                continue
            if deaf_after_s is not None and s2 - c0 > deaf_after_s:
                continue
            out.append({"start": int((s2 - c0) * SR), "end": int((e2 - c0) * SR)})
        return out
    return detect


def test_merge_intervals():
    assert cv.merge_intervals([(5, 6), (1, 2), (2.5, 3)], max_gap=1.0) == [(1, 3), (5, 6)]
    assert cv.merge_intervals([(1, 2), (1.5, 4)]) == [(1, 4)]
    assert cv.merge_intervals([(3, 3)]) == []


def test_overlapping_chunks_union():
    truth = [(12.0, 18.0), (40.0, 41.0)]
    clips = cv.speech_clips(FakeAudio(60), detector=make_detector(truth))
    assert clips == [(12.0, 18.0), (40.0, 41.0)], clips


def test_deafness_beyond_context_is_recovered_by_short_chunks():
    # Spraak op 305-316 s na 300 s "muziek": een detector die alleen de eerste 6 s
    # van elk stuk goed hoort, vindt het toch dankzij stukken van 10 s / hop 5 s.
    truth = [(305.0, 316.0)]
    clips = cv.speech_clips(FakeAudio(330), detector=make_detector(truth, deaf_after_s=6.0))
    assert clips == [(305.0, 316.0)], clips


def test_to_clip_timestamps():
    assert cv.to_clip_timestamps([(1.0, 2.5), (4.0, 5.0)]) == [1.0, 2.5, 4.0, 5.0]
    assert cv.to_clip_timestamps([]) == []


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"{len(tests)} tests geslaagd")
