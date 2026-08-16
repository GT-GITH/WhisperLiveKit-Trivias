"""Synthetische tests voor whisperlivekit/translate.py.

Geen testsuite/pytest-dependency in dit project (zie CLAUDE.md) -- draai dit
bestand rechtstreeks: `python tests/test_translate.py`. Focus ligt op de
fail-safe-eis die het ontwerp bepaalt: translate_text() mag nooit een
exception naar de aanroeper laten lekken -- ontbrekende backend, lege tekst,
een falende vertaal-call, of een lege respons resulteren allemaal in None.

Gebruikt een testdubbel voor NLLBBackend (geen echte modelgewichten nodig,
die zijn hier niet beschikbaar -- zie tests/test_nllb_backend.py voor wat
wél zonder gewichten te testen is: de taalcode-mapping en de fail-safe
factory). Wat de "echte" vertaalkwaliteit betreft (NLLB vs. het eerder
gebruikte, onbetrouwbaar gebleken LLM-chatmodel) is alleen op de runpod met
een echt geladen model te verifiëren.
"""

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent.parent / "whisperlivekit" / "translate.py"
_spec = importlib.util.spec_from_file_location("translate", _MODULE_PATH)
_tr = importlib.util.module_from_spec(_spec)
sys.modules["translate"] = _tr
_spec.loader.exec_module(_tr)

translate_text = _tr.translate_text


class _FakeNLLBBackend:
    """Testdubbel voor whisperlivekit.nllb_backend.NLLBBackend."""

    def __init__(self, response=None, raise_exc=None):
        self.response = response
        self.raise_exc = raise_exc
        self.last_text = None
        self.last_source = None
        self.call_count = 0

    def translate(self, text, source_iso, target_iso="nl"):
        self.call_count += 1
        self.last_text = text
        self.last_source = source_iso
        if self.raise_exc:
            raise self.raise_exc
        return self.response


def test_no_backend_returns_none():
    assert translate_text("hello", "en", None) is None
    print("OK test_no_backend_returns_none")


def test_empty_text_returns_none_without_calling_backend():
    backend = _FakeNLLBBackend(response="zou nooit gebruikt moeten worden")
    assert translate_text("", "en", backend) is None
    assert translate_text("   ", "en", backend) is None
    assert backend.call_count == 0
    print("OK test_empty_text_returns_none_without_calling_backend")


def test_fail_safe_on_backend_exception():
    backend = _FakeNLLBBackend(raise_exc=RuntimeError("model niet geladen"))
    assert translate_text("merhaba", "tr", backend) is None
    print("OK test_fail_safe_on_backend_exception")


def test_fail_safe_on_none_response():
    # NLLBBackend.translate() geeft zelf al None terug bij een onbekende
    # brontaal of een mislukte call -- translate_text() moet dat gewoon
    # doorgeven, niet omzetten naar een lege string of exception.
    backend = _FakeNLLBBackend(response=None)
    assert translate_text("merhaba", "tr", backend) is None
    print("OK test_fail_safe_on_none_response")


def test_successful_translation_passthrough():
    backend = _FakeNLLBBackend(response="Hallo, hoe gaat het?")
    result = translate_text("merhaba, nasilsin?", "tr", backend)
    assert result == "Hallo, hoe gaat het?"
    assert backend.last_text == "merhaba, nasilsin?"
    assert backend.last_source == "tr"
    print("OK test_successful_translation_passthrough")


if __name__ == "__main__":
    tests = [
        test_no_backend_returns_none,
        test_empty_text_returns_none_without_calling_backend,
        test_fail_safe_on_backend_exception,
        test_fail_safe_on_none_response,
        test_successful_translation_passthrough,
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
