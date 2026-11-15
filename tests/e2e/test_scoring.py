"""Unit-tests voor tests/e2e/scoring.py. Draai: `python tests/e2e/test_scoring.py`.

Zelfde stijl als de andere bestanden in tests/: geen pytest-dependency, gewoon
asserts. Zet vast dat de meetlat zelf klopt -- een fout hier zou elke
e2e-uitkomst stilletjes vertekenen.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import cer, edit_distance, normalize, score_channel, score_live, wer  # noqa: E402


def _fd(t, buffer="", lines=()):
    return {"t": t, "msg": {"type": "front_data", "buffer_transcription": buffer,
                            "lines": [{"text": x, "text_batch": None} for x in lines]}}


def test_score_live_moving():
    utts = [{"start_ms": 1000, "end_ms": 5000, "text": "a"}, {"start_ms": 8000, "end_ms": 12000, "text": "b"}]
    msgs = [_fd(0.5), _fd(2.0, "goe"), _fd(3.0, "goede"), _fd(9.0, "", ["goede"]), _fd(10.0, "ho", ["goede"])]
    r = score_live(utts, msgs)
    assert r["live_missed"] == 0, r
    assert r["live_max_gap_s"] < 8.0, r


def test_score_live_stalled():
    # live stopt na 3 s, terwijl er tot 30 s spraak is (zoals de live-stilstand 2026-11-15)
    utts = [{"start_ms": 1000, "end_ms": 10000, "text": "a"}, {"start_ms": 11000, "end_ms": 30000, "text": "b"}]
    msgs = [_fd(0.5), _fd(2.0, "x"), _fd(3.0, "xy")] + [_fd(t, "xy") for t in range(4, 34)]
    r = score_live(utts, msgs)
    assert r["live_missed"] == 1 and r["live_missed_utterances"] == ["b"], r
    assert r["live_max_gap_s"] >= 30.0, r


def test_score_live_lag():
    # live-regel verschijnt op t=60 s maar eindigt (volgens zijn tijdstempel) op 10 s: 50 s achter
    utts = [{"start_ms": 1000, "end_ms": 9000, "text": "a"}]
    msgs = [{"t": 60.0, "msg": {"type": "front_data", "buffer_transcription": "",
                                "lines": [{"id": "seg_1", "text": "x", "text_batch": None, "start_ms": 8000, "end_ms": 10000}]}}]
    r = score_live(utts, msgs)
    assert r["live_lag_median_s"] == 50.0 and r["live_lag_p90_s"] == 50.0, r


def test_score_live_ignores_batch_replacement():
    # alleen een batch-regel die verschijnt telt niet als live-activiteit
    utts = [{"start_ms": 1000, "end_ms": 5000, "text": "a"}]
    msgs = [_fd(0.5), {"t": 3.0, "msg": {"type": "front_data", "buffer_transcription": "",
                                         "lines": [{"text": "a", "text_batch": "a"}]}}]
    assert score_live(utts, msgs)["live_missed"] == 1


def test_normalize():
    assert normalize("Hallo, Wereld!  Dit   is\tEEN test.") == "hallo wereld dit is een test"
    assert normalize("Hayır, her şey doğruydu.") == "hayır her şey doğruydu"
    assert normalize("") == ""


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], ["a"]) == 1
    assert edit_distance(["a", "b"], ["a", "b"]) == 0


def test_wer_cer():
    assert wer("de kat zit op de mat", "de kat zit op de mat") == 0.0
    assert abs(wer("de kat zit op de mat", "de hond zit op mat") - 2 / 6) < 1e-9
    assert wer("", "iets") is None
    assert cer("abc", "abd") == 1 / 3


def _seg(s, e, text, batch=True):
    return {"start_ms": s, "end_ms": e, "text_final": text, "text_batch": text if batch else None}


def test_score_channel_perfect():
    utts = [{"start_ms": 1000, "end_ms": 3000, "text": "goedemorgen"},
            {"start_ms": 5000, "end_ms": 7000, "text": "hoe gaat het"}]
    segs = [_seg(1100, 2900, "Goedemorgen."), _seg(5050, 6900, "Hoe gaat het?")]
    r = score_channel(utts, segs)
    assert r["wer"] == 0.0 and r["missed"] == 0 and r["spurious"] == 0
    assert r["ts_median_ms"] == 75 and r["ts_max_ms"] == 100
    assert r["live_fallback"] == 0


def test_score_channel_missed_spurious_fallback():
    utts = [{"start_ms": 1000, "end_ms": 3000, "text": "goedemorgen"},
            {"start_ms": 5000, "end_ms": 7000, "text": "hoe gaat het"}]
    # tweede uiting gemist, hallucinatie in stilte op 20s, eerste is live-fallback
    segs = [_seg(1000, 3000, "goedemorgen", batch=False), _seg(20000, 21000, "bedankt voor het kijken")]
    r = score_channel(utts, segs)
    assert r["missed"] == 1
    assert r["spurious"] == 1 and r["spurious_segments"][0]["start_ms"] == 20000
    assert r["live_fallback"] == 1


def test_split_utterance_is_not_a_timestamp_error():
    # Whisper knipt één lange uiting in twee segmenten: de start van het
    # tweede segment mag de tijdstempel-afwijking niet opblazen.
    utts = [{"start_ms": 1000, "end_ms": 12000, "text": "een heel lange zin die doorloopt"}]
    segs = [_seg(1100, 6000, "een heel lange zin"), _seg(6000, 11900, "die doorloopt")]
    r = score_channel(utts, segs)
    assert r["ts_median_ms"] == 100 and r["ts_max_ms"] == 100, r


def test_touching_next_segment_does_not_count_as_covered():
    # Uiting 71.6-79.8 s volledig weggevallen; het segment van de volgende
    # uiting begint op 80.32 s en raakt alleen de 500 ms-marge.
    utts = [{"start_ms": 71624, "end_ms": 79824, "text": "romantizm"},
            {"start_ms": 80424, "end_ms": 88112, "text": "japonya"}]
    r = score_channel(utts, [_seg(80320, 88000, "japonya")])
    assert r["missed"] == 1 and r["missed_utterances"] == ["romantizm"], r


def test_empty_segments_ignored():
    utts = [{"start_ms": 0, "end_ms": 1000, "text": "ja"}]
    r = score_channel(utts, [{"start_ms": 0, "end_ms": 500, "text_final": "  ", "text_batch": None}])
    assert r["n_segments"] == 0 and r["missed"] == 1 and r["wer"] == 1.0


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"{len(tests)} tests geslaagd")
