"""Tests voor whisperlivekit/simul_whisper/segment_split.py.

Draai rechtstreeks: `python tests/test_segment_split.py` (geen pytest, zie
CLAUDE.md). De twee hoofdgevallen komen uit echte e2e-runs (2026-11-15).
"""

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "whisperlivekit" / "simul_whisper" / "segment_split.py"
_spec = importlib.util.spec_from_file_location("segment_split", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
split_on_word_gaps = _mod.split_on_word_gaps


def W(word, start, end):
    return {"word": word, "start": start, "end": end}


def test_no_gap_keeps_one_segment():
    words = [W(" Goede", 1.0, 1.4), W("morgen", 1.4, 1.9), W(" allemaal.", 2.1, 2.8)]
    assert split_on_word_gaps(words) == [("Goedemorgen allemaal.", 1.0, 2.8)]


def test_seam_artifact_across_long_silence():
    # nl_lange_pauze: "De" op de VAD-naad (36.09 s), de rest na 45 s stilte.
    words = [W(" De", 36.09, 36.30), W(" aankondiging", 82.90, 83.60), W(" kwam", 83.65, 83.90),
             W(" na", 83.95, 84.10), W(" overleg.", 84.20, 87.65)]
    # Alleen splitsen: "De" blijft los op zijn eigen plek, de echte uiting
    # krijgt haar echte start (82.90 i.p.v. 36.09).
    assert split_on_word_gaps(words) == [("De", 36.09, 36.30), ("aankondiging kwam na overleg.", 82.90, 87.65)]


def test_multiple_utterances_in_one_segment():
    # oddadmix: één segment voor meerdere uitingen met seconden stilte ertussen.
    words = [W(" a1", 10.8, 11.2), W(" a2", 11.3, 11.9), W(" a3", 12.0, 18.0),
             W(" b1", 25.4, 26.0), W(" b2", 26.1, 27.0), W(" b3", 27.1, 29.6),
             W(" c1", 31.2, 32.0), W(" c2", 32.1, 33.0), W(" c3", 33.1, 35.1)]
    assert split_on_word_gaps(words) == [
        ("a1 a2 a3", 10.8, 18.0), ("b1 b2 b3", 25.4, 29.6), ("c1 c2 c3", 31.2, 35.1)]


def test_short_answer_keeps_its_own_timestamp():
    # Een kort antwoord vóór een stilte mag NIET de starttijd van de volgende zin krijgen.
    words = [W(" Ja.", 1.0, 1.3), W(" Dat", 4.5, 4.8), W(" klopt.", 4.8, 5.3)]
    assert split_on_word_gaps(words) == [("Ja.", 1.0, 1.3), ("Dat klopt.", 4.5, 5.3)]


def test_short_trailing_group_is_kept_not_dropped():
    words = [W(" ja", 1.0, 1.3), W(" dat", 1.3, 1.5), W(" klopt.", 1.5, 2.0), W(" Ja.", 5.0, 5.3)]
    assert split_on_word_gaps(words) == [("ja dat klopt.", 1.0, 2.0), ("Ja.", 5.0, 5.3)]


def test_empty_or_missing_words():
    assert split_on_word_gaps([]) == []
    assert split_on_word_gaps(None) == []
    assert split_on_word_gaps([W("  ", 1.0, 1.1)]) == []


def test_object_words():
    class Obj:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end
    assert split_on_word_gaps([Obj(" Hoi", 0.0, 0.5)]) == [("Hoi", 0.0, 0.5)]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"{len(tests)} tests geslaagd")
