"""Tests voor whisperlivekit/tokens_alignment.py's suppressie-/snoeilogica.

Geen testsuite/pytest-dependency in dit project (zie CLAUDE.md) -- draai dit
bestand rechtstreeks: `python tests/test_tokens_alignment.py`. Zelfde stijl als
tests/test_cross_channel_gate.py en tests/test_translate.py.

Focus: apply_batch_group()'s window_end_ms is een contract, geen interne
implementatiedetail -- alles wat er overlapt met [window_start_ms, window_end_ms)
wordt permanent uit de live-weergave gesnoeid (get_lines(), via
suppressed_ranges_ms). Op 2026-08-23 bleek audio_processor.py's _batch_worker()
hier het GEPLANDE venster-einde aan doorgeven i.p.v. wat de WAV daadwerkelijk
bevatte -- een venster dat op schijf korter bleek dan gepland, suppressede toch
het volledige geplande bereik, en verwijderde zo een live-segment in het
niet-gedecodeerde staartstuk zonder vervanging (5+ minuten aan tekst in een
lange sessie). De fix zat in audio_processor.py (het écht gedecodeerde einde
doorgeven), maar het CONTRACT dat deze bug schond zit hier, in
apply_batch_group()/get_lines() -- dat is wat onderstaande tests vastzetten.
"""

import sys
import types
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _load_from_path(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, _ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# tokens_alignment.py doet `from whisperlivekit.timed_objects import ...` --
# een absoluut, pakket-gekwalificeerd import. Een dummy "whisperlivekit"-pakket
# vooraf in sys.modules zetten (plus de echte timed_objects.py er al in, onder
# zijn volledige naam) voorkomt dat Python daarvoor whisperlivekit/__init__.py
# uitvoert -- dat laadt de hele, zware ASR-stack (torch, faster-whisper, etc.),
# precies wat deze module-voor-module-aanpak (zie ook de andere tests/test_*.py)
# bewust vermijdt.
if "whisperlivekit" not in sys.modules:
    sys.modules["whisperlivekit"] = types.ModuleType("whisperlivekit")

_load_from_path("whisperlivekit.timed_objects", "whisperlivekit/timed_objects.py")
_ta = _load_from_path("whisperlivekit.tokens_alignment", "whisperlivekit/tokens_alignment.py")

TokensAlignment = _ta.TokensAlignment
Segment = sys.modules["whisperlivekit.timed_objects"].Segment


def _make_alignment() -> "TokensAlignment":
    args = types.SimpleNamespace(diarization=False)
    ta = TokensAlignment(state=None, args=args, sep=" ")
    ta.beg_loop = 0.0
    return ta


def _seg(start_s, end_s, text, seg_id):
    return Segment(start=start_s, end=end_s, text=text, speaker=-1, id=seg_id)


def test_suppressed_range_only_prunes_overlapping_segments():
    """Rechtstreekse test van get_lines()'s snoei-predicaat: een segment vóór en
    een segment ná een suppressed range moeten allebei overleven, alleen het
    overlappende segment mag verdwijnen."""
    ta = _make_alignment()
    ta.validated_segments = [
        _seg(1.0, 2.0, "voor het venster", "seg_before"),
        _seg(11.0, 12.0, "midden in het venster", "seg_inside"),
        _seg(21.0, 22.0, "na het venster", "seg_after"),
    ]
    ta.suppressed_ranges_ms = [(10_000, 20_000)]

    lines, _, _ = ta.get_lines()
    ids = {getattr(l, "id", None) for l in lines}

    assert "seg_before" in ids, "segment vóór de suppressed range mag niet gesnoeid worden"
    assert "seg_after" in ids, "segment ná de suppressed range mag niet gesnoeid worden"
    assert "seg_inside" not in ids, "segment binnen de suppressed range hoort wél gesnoeid te worden"
    print("OK test_suppressed_range_only_prunes_overlapping_segments")


def test_apply_batch_group_with_correct_window_end_preserves_undecoded_tail():
    """Reproduceert het scenario achter de content-verlies-bug (2026-08-23), maar
    dan met de FIX: audio_processor.py geeft nu real_window_end_ms (wat de WAV
    daadwerkelijk bevatte, hier 30000ms) door i.p.v. het oorspronkelijk geplande,
    te ruime venster-einde (hier 33000ms, waar een live-segment al in stond)."""
    ta = _make_alignment()
    ta.validated_segments = [
        _seg(5.0, 25.0, "binnen het echt gedecodeerde deel", "seg_decoded"),
        _seg(31.0, 33.0, "in de niet-gedecodeerde staart", "seg_tail"),
    ]

    group_id = ta.apply_batch_group(
        window_start_ms=0,
        window_end_ms=30_000,  # = real_window_end_ms, niet het geplande 33000
        text_final="batch-bevestigde tekst voor het echt gedecodeerde deel",
        speaker=-1,
    )
    assert group_id, "apply_batch_group() moet een group_id teruggeven voor een geldig venster"

    lines, _, _ = ta.get_lines()
    ids = {getattr(l, "id", None) for l in lines}

    assert group_id in ids, "de canonical batch-groep zelf moet aanwezig zijn"
    assert "seg_tail" in ids, (
        "een live-segment ná het echt gedecodeerde venster-einde mag NOOIT verdwijnen -- "
        "dit is precies wat er misging vóór de fix"
    )
    print("OK test_apply_batch_group_with_correct_window_end_preserves_undecoded_tail")


def test_apply_batch_group_with_inflated_window_end_drops_the_tail():
    """Dezelfde opstelling als hierboven, maar nu met het GEPLANDE (te ruime)
    venster-einde (33000ms i.p.v. de echte 30000ms) -- dit documenteert exact het
    kapotte gedrag van vóór de fix: apply_batch_group()/get_lines() doen precies
    wat er gevraagd wordt, dus een te ruim window_end_ms is voldoende om een
    live-segment stilletjes te laten verdwijnen. De eigenlijke fix (het juiste,
    geklemde window_end_ms doorgeven) zit in audio_processor.py, niet hier --
    deze test legt vast WAAROM die waarde zo belangrijk is."""
    ta = _make_alignment()
    ta.validated_segments = [
        _seg(5.0, 25.0, "binnen het echt gedecodeerde deel", "seg_decoded"),
        _seg(31.0, 33.0, "in de niet-gedecodeerde staart", "seg_tail"),
    ]

    ta.apply_batch_group(
        window_start_ms=0,
        window_end_ms=33_000,  # het (te ruime) oorspronkelijk geplande einde
        text_final="batch-bevestigde tekst",
        speaker=-1,
    )

    lines, _, _ = ta.get_lines()
    ids = {getattr(l, "id", None) for l in lines}

    assert "seg_tail" not in ids, (
        "met een te ruim window_end_ms verdwijnt het staart-segment inderdaad -- "
        "zo verloor de live-weergave vóór de fix stilletjes content"
    )
    print("OK test_apply_batch_group_with_inflated_window_end_drops_the_tail")


def test_apply_batch_group_rejects_empty_or_inverted_window():
    ta = _make_alignment()
    assert ta.apply_batch_group(window_start_ms=1000, window_end_ms=1000, text_final="x") == ""
    assert ta.apply_batch_group(window_start_ms=2000, window_end_ms=1000, text_final="x") == ""
    assert ta.suppressed_ranges_ms == [], "een leeg/omgekeerd venster mag niets suppressen"
    print("OK test_apply_batch_group_rejects_empty_or_inverted_window")


if __name__ == "__main__":
    tests = [
        test_suppressed_range_only_prunes_overlapping_segments,
        test_apply_batch_group_with_correct_window_end_preserves_undecoded_tail,
        test_apply_batch_group_with_inflated_window_end_drops_the_tail,
        test_apply_batch_group_rejects_empty_or_inverted_window,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    if failures:
        print(f"\n{failures}/{len(tests)} test(s) FAILED")
        sys.exit(1)
    print(f"\nAlle {len(tests)} tests geslaagd.")
