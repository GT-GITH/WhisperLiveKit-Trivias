"""Unit-tests voor tests/e2e/scoring.py. Draai: `python tests/e2e/test_scoring.py`.

Zelfde stijl als de andere bestanden in tests/: geen pytest-dependency, gewoon
asserts. Zet vast dat de meetlat zelf klopt -- een fout hier zou elke
e2e-uitkomst stilletjes vertekenen.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import cer, edit_distance, normalize, score_channel, wer  # noqa: E402


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
