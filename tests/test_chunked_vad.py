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


def test_first_clip_near_start_snaps_to_zero():
    # VAD mist het eerste woord (spraak vanaf 0, gedetecteerd vanaf 0.53 s): de clip
    # moet vanaf 0 beginnen, anders slaat Whisper het hele eerste stuk over.
    clips = cv.speech_clips(FakeAudio(16), detector=make_detector([(0.53, 15.7)]))
    assert clips[0][0] == 0.0, clips


def test_uncovered_tails():
    clips = [(0.0, 20.73), (30.0, 40.0), (50.0, 55.0)]
    speech = [(0.0, 3.0), (3.5, 20.73), (30.0, 39.8), (50.0, 55.0)]
    # clip 1: tekst tot 2.76, daarna nog 17 s spraak -> staart; clip 2: gedekt tot 39.5
    # (0.3 s spraak over); clip 3: helemaal geen tekst -> geen staart
    spans = [(0.0, 2.76), (30.5, 39.5)]
    assert cv.uncovered_tails(clips, spans, speech) == [(2.76, 20.73)]
    assert cv.uncovered_tails(clips, spans + [(3.86, 20.5)], speech) == []


def test_tail_of_silence_is_not_resumed():
    # Clip loopt door tot 20 s (samengevoegde pauze), maar de spraak stopt op 18.4 s
    # (laatste woord 18.0 + VAD-marge): stilte decoderen gaf hallucinaties -> niet hervatten.
    clips = [(0.0, 20.0)]
    speech = [(0.0, 9.0), (10.5, 18.4), (19.6, 20.0)]
    assert cv.uncovered_tails(clips, [(0.0, 18.16)], speech) == []


def make_open_end_detector(truth, min_silence_s=2.0):
    """Zoals Silero: een spraakstuk dat < min_silence_s voor het einde van het stuk
    audio eindigt, blijft open tot het einde van dat stuk."""
    inner = make_detector(truth)

    def detect(chunk):
        c1 = (chunk.offset + chunk.n) / SR
        out = []
        for t in inner(chunk):
            e = (chunk.offset + t["end"]) / SR
            if c1 - e < min_silence_s:
                t = {"start": t["start"], "end": chunk.n}
            out.append(t)
        return out
    return detect


def test_open_end_at_chunk_boundary_is_not_speech():
    # Echte spraak 11.3-18.66 (dan stilte). Stuk [10,20] ziet het einde niet -> "tot 20";
    # stuk [15,25] wel. Uitkomst moet het echte einde zijn, niet 20.0.
    got = cv.speech_intervals(FakeAudio(40), detector=make_open_end_detector([(11.3, 18.66)]))
    assert got == [(11.3, 18.66)], got


def test_open_end_at_final_chunk_is_kept():
    # Spraak loopt door tot het einde van de audio: geen volgend stuk -> niet inkorten.
    got = cv.speech_intervals(FakeAudio(20), detector=make_open_end_detector([(16.0, 20.0)]))
    assert got == [(16.0, 20.0)], got


def test_speech_intervals_not_merged_over_pauses():
    truth = [(1.0, 3.0), (4.0, 6.0)]  # 1 s pauze: één decodeer-clip, twee spraakstukken
    assert cv.speech_intervals(FakeAudio(10), detector=make_detector(truth)) == truth
    assert cv.speech_clips(FakeAudio(10), detector=make_detector(truth)) == [(1.0, 6.0)]


def test_uncovered_regions_begin_middle_tail():
    evidence = [(0.0, 20.0)]
    spans = [(4.8, 7.6), (12.0, 14.0)]  # tekst dekt 4.5-7.9 en 11.7-14.3 (tol 0.3)
    got = cv.uncovered_regions(evidence, spans)
    assert [(round(a, 2), round(b, 2)) for a, b in got] == [(0.0, 4.5), (7.9, 11.7), (14.3, 20.0)], got


def test_uncovered_regions_ignores_short_and_covered():
    evidence = [(10.0, 10.5), (20.0, 23.0)]   # 0.5 s < min 0.6 s; tweede helemaal gedekt
    assert cv.uncovered_regions(evidence, [(19.9, 23.1)]) == []


def test_uncovered_regions_no_text_at_all():
    # Standaard-VAD vond niets, ruimere VAD wel: het hele stuk is onbedekt.
    assert cv.uncovered_regions([(212.5, 213.9)], []) == [(212.5, 213.9)]


def test_to_clip_timestamps():
    assert cv.to_clip_timestamps([(1.0, 2.5), (4.0, 5.0)]) == [1.0, 2.5, 4.0, 5.0]
    assert cv.to_clip_timestamps([]) == []


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"{len(tests)} tests geslaagd")
