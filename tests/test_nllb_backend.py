"""Synthetische tests voor whisperlivekit/nllb_backend.py.

Geen testsuite/pytest-dependency in dit project (zie CLAUDE.md) -- draai dit
bestand rechtstreeks: `python tests/test_nllb_backend.py`. Test alleen wat
zonder echte NLLB-modelgewichten te testen is: de fail-safe `build_nllb_backend()`-
factory (nooit een exception, ook niet als het model niet geconfigureerd is)
en de ISO->NLLB-taalcode-mapping (moet alle talen dekken die de UI aanbiedt,
zie LANGUAGES in web_trivias/app.js). Het daadwerkelijke vertaalgedrag
(NLLBBackend.translate() met een geladen model) is alleen op de runpod met
echte modelgewichten te verifiëren.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

_MODULE_PATH = Path(__file__).resolve().parent.parent / "whisperlivekit" / "nllb_backend.py"
_spec = importlib.util.spec_from_file_location("nllb_backend", _MODULE_PATH)
_nb = importlib.util.module_from_spec(_spec)
sys.modules["nllb_backend"] = _nb
_spec.loader.exec_module(_nb)

ISO_TO_NLLB = _nb.ISO_TO_NLLB
build_nllb_backend = _nb.build_nllb_backend
_split_sentences = _nb._split_sentences


def test_no_model_configured_returns_none():
    args = SimpleNamespace(nllb_model=None, nllb_device="auto")
    assert build_nllb_backend(args) is None
    print("OK test_no_model_configured_returns_none")


def test_empty_string_model_returns_none():
    args = SimpleNamespace(nllb_model="", nllb_device="auto")
    assert build_nllb_backend(args) is None
    print("OK test_empty_string_model_returns_none")


def test_missing_nllb_model_attr_does_not_crash():
    # AudioProcessor/args-achtige objecten zonder dit attribuut moeten geen
    # AttributeError geven -- getattr(..., None) in build_nllb_backend().
    args = SimpleNamespace()
    assert build_nllb_backend(args) is None
    print("OK test_missing_nllb_model_attr_does_not_crash")


def test_iso_to_nllb_covers_all_ui_languages():
    # Zie LANGUAGES in whisperlivekit/web_trivias/app.js -- elke taal die de
    # gebruiker in de kanaalconfiguratie kan kiezen moet hier een NLLB-code hebben,
    # anders faalt vertaling stilzwijgend voor een taal die de UI wél aanbiedt.
    ui_languages = ["nl", "en", "ar", "fa", "ru", "fr", "de", "tr", "so", "ti", "ku", "sr", "bs"]
    for code in ui_languages:
        assert code in ISO_TO_NLLB, f"taal {code!r} uit de UI ontbreekt in ISO_TO_NLLB"
        assert ISO_TO_NLLB[code].count("_") == 1, f"NLLB-code voor {code!r} mist het scriptsuffix (bv. '_Latn')"
    print("OK test_iso_to_nllb_covers_all_ui_languages")


def test_split_sentences_multi_sentence():
    # Geconstateerd 2026-08-16 (testsessie): NLLB-200 genereert bij meerzinnige
    # input een voortijdig stop-token en laat alles na de eerste zin vallen --
    # translate() splitst daarom nu vooraf in losse zinnen (zie NLLBBackend.translate()).
    result = _split_sentences(
        "Kardesim gazeteciydi ve hukumeti elestiren yazilar yaziyordu. "
        "Bir gece evimize silahli kisiler geldi."
    )
    assert result == [
        "Kardesim gazeteciydi ve hukumeti elestiren yazilar yaziyordu.",
        "Bir gece evimize silahli kisiler geldi.",
    ]
    print("OK test_split_sentences_multi_sentence")


def test_split_sentences_single_sentence_unchanged():
    result = _split_sentences("Enkele zin zonder splitsing.")
    assert result == ["Enkele zin zonder splitsing."]
    print("OK test_split_sentences_single_sentence_unchanged")


def test_split_sentences_question_and_exclamation_marks():
    result = _split_sentences("Vraag? Antwoord! Nog een zin.")
    assert result == ["Vraag?", "Antwoord!", "Nog een zin."]
    print("OK test_split_sentences_question_and_exclamation_marks")


if __name__ == "__main__":
    tests = [
        test_no_model_configured_returns_none,
        test_empty_string_model_returns_none,
        test_missing_nllb_model_attr_does_not_crash,
        test_iso_to_nllb_covers_all_ui_languages,
        test_split_sentences_multi_sentence,
        test_split_sentences_single_sentence_unchanged,
        test_split_sentences_question_and_exclamation_marks,
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
