"""Synthetische tests voor whisperlivekit/translate.py.

Geen testsuite/pytest-dependency in dit project (zie CLAUDE.md) -- draai dit
bestand rechtstreeks: `python tests/test_translate.py`. Focus ligt op de
fail-safe-eis die het ontwerp bepaalt: translate_text() mag nooit een
exception naar de aanroeper laten lekken -- ontbrekende backend, lege tekst,
een falende LLM-call of een lege respons resulteren allemaal in None.
"""

import importlib.util
import sys
from pathlib import Path

# translate.py is verder onafhankelijk van de rest van het pakket -- rechtstreeks
# via bestandspad laden i.p.v. via `whisperlivekit.translate`, want dat laatste
# triggert whisperlivekit/__init__.py, dat de hele (zware) ASR-stack importeert.
_MODULE_PATH = Path(__file__).resolve().parent.parent / "whisperlivekit" / "translate.py"
_spec = importlib.util.spec_from_file_location("translate", _MODULE_PATH)
_tr = importlib.util.module_from_spec(_spec)
sys.modules["translate"] = _tr
_spec.loader.exec_module(_tr)

translate_text = _tr.translate_text


class _FakeLLMBackend:
    """Testdubbel voor LLMBackend -- zie tests/test_gehoorverslag.py voor
    hetzelfde patroon (nu daar verwijderd samen met classify_segments())."""

    def __init__(self, response=None, raise_exc=None):
        self.response = response
        self.raise_exc = raise_exc
        self.last_system_prompt = None
        self.last_user_prompt = None
        self.call_count = 0

    def chat(self, system_prompt, user_prompt):
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        if self.raise_exc:
            raise self.raise_exc
        return self.response


def test_no_backend_returns_none():
    assert translate_text("hello", "en", None) is None
    print("OK test_no_backend_returns_none")


def test_empty_text_returns_none_without_calling_backend():
    backend = _FakeLLMBackend(response="zou nooit gebruikt moeten worden")
    assert translate_text("", "en", backend) is None
    assert translate_text("   ", "en", backend) is None
    assert backend.call_count == 0
    print("OK test_empty_text_returns_none_without_calling_backend")


def test_fail_safe_on_backend_exception():
    backend = _FakeLLMBackend(raise_exc=RuntimeError("connection refused"))
    assert translate_text("merhaba", "tr", backend) is None
    print("OK test_fail_safe_on_backend_exception")


def test_fail_safe_on_empty_response():
    backend = _FakeLLMBackend(response="   ")
    assert translate_text("merhaba", "tr", backend) is None
    print("OK test_fail_safe_on_empty_response")


def test_successful_translation_strips_whitespace():
    backend = _FakeLLMBackend(response="  Hallo, hoe gaat het?  ")
    result = translate_text("merhaba, nasılsın?", "tr", backend)
    assert result == "Hallo, hoe gaat het?"
    print("OK test_successful_translation_strips_whitespace")


def test_language_code_mapped_to_readable_name_in_prompt():
    backend = _FakeLLMBackend(response="Hallo")
    translate_text("merhaba", "tr", backend)
    assert "Turks" in backend.last_system_prompt
    print("OK test_language_code_mapped_to_readable_name_in_prompt")


def test_unknown_language_code_falls_back_gracefully():
    backend = _FakeLLMBackend(response="Hallo")
    translate_text("hello", "xx", backend)
    assert "xx" in backend.last_system_prompt
    print("OK test_unknown_language_code_falls_back_gracefully")


def test_missing_language_falls_back_to_autodetect_prompt():
    backend = _FakeLLMBackend(response="Hallo")
    translate_text("bir şey söyledi", None, backend)
    assert "Detecteer zelf de brontaal" in backend.last_system_prompt
    print("OK test_missing_language_falls_back_to_autodetect_prompt")


if __name__ == "__main__":
    tests = [
        test_no_backend_returns_none,
        test_empty_text_returns_none_without_calling_backend,
        test_fail_safe_on_backend_exception,
        test_fail_safe_on_empty_response,
        test_successful_translation_strips_whitespace,
        test_language_code_mapped_to_readable_name_in_prompt,
        test_unknown_language_code_falls_back_gracefully,
        test_missing_language_falls_back_to_autodetect_prompt,
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
