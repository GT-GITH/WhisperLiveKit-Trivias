"""Vertaling van niet-Nederlandse transcript-tekst -- on-demand, per segment.

Zie features/vertaling-niet-nl-tekst.md in de projectrepo voor het volledige
ontwerp. Kern: alleen bevestigde (batch) tekst van "foreign_*"-kanalen wordt
op verzoek (klik op het 🌐-icoontje in de UI) vertaald naar het Nederlands,
via de generieke on-prem LLM-backend (zie llm_backend.py) -- geen NLLB, geen
automatische vertaling van elke regel, geen persistente opslag in
transcript.json (een vertaling is een AI-afleiding, geen onderdeel van de
autoritatieve transcriptie, zie CLAUDE.md "audio is authoritative").

Puur, van FastAPI losgekoppeld -- zelfde patroon als het (inmiddels
verwijderde) classify_segments() uit gehoorverslag.py: één functie,
VOLLEDIG fail-safe. Geen backend, een timeout/HTTP-fout, of een lege
respons resulteren allemaal in None, nooit een exception naar de
aanroeper.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger("whisperlivekit.translate")

# ISO 639-1 -> volledige naam, voor een leesbaardere promptformulering dan
# een kale taalcode. Onbekende/ontbrekende codes vallen terug op de code
# zelf (of een generieke omschrijving) -- nooit een KeyError.
_LANGUAGE_NAMES = {
    "ar": "Arabisch",
    "fa": "Farsi",
    "tr": "Turks",
    "ru": "Russisch",
    "en": "Engels",
    "nl": "Nederlands",
}

# Bekende brontaal (bv. het foreign_*-kanaal, waar de taal betrouwbaar in de
# channel_id zelf staat) -- expliciet benoemd in de prompt.
_SYSTEM_PROMPT_KNOWN = (
    "Je vertaalt een fragment uit een asielgehoor-transcript van het {lang} "
    "naar het Nederlands. Geef UITSLUITEND de vertaling terug -- geen "
    "uitleg, geen aanhalingstekens, geen markdown. Is de tekst al Nederlands, "
    "geef 'm dan ongewijzigd terug."
)
# Onbekende/onbetrouwbare brontaal (bv. het tolk-kanaal: meestal Nederlands,
# maar een tolk zegt soms iets in de brontaal van de vreemdeling) -- geen
# brontaal opdringen, het LLM detecteert zelf.
_SYSTEM_PROMPT_AUTODETECT = (
    "Je vertaalt een fragment uit een asielgehoor-transcript naar het "
    "Nederlands. Detecteer zelf de brontaal. Geef UITSLUITEND de vertaling "
    "terug -- geen uitleg, geen aanhalingstekens, geen markdown. Is de tekst "
    "al Nederlands, geef 'm dan ongewijzigd terug."
)


def _language_name(source_language: Optional[str]) -> str:
    code = (source_language or "").strip().lower()
    if not code:
        return "de brontaal"
    return _LANGUAGE_NAMES.get(code, code)


def _build_system_prompt(source_language: Optional[str]) -> str:
    code = (source_language or "").strip().lower()
    if not code:
        return _SYSTEM_PROMPT_AUTODETECT
    return _SYSTEM_PROMPT_KNOWN.format(lang=_language_name(code))


def translate_text(text: str, source_language: Optional[str], llm_backend: Optional[Any]) -> Optional[str]:
    """Vertaalt `text` naar het Nederlands via `llm_backend`.

    Retourneert None (nooit een exception) als: de tekst leeg is, er geen
    backend geconfigureerd is, de LLM-call faalt, of de respons leeg is na
    strippen. De aanroeper (het /translate-endpoint) is verantwoordelijk
    voor het omzetten van None naar een passende HTTP-fout -- deze functie
    doet zelf geen aannames over hoe dat gecommuniceerd moet worden."""
    text = (text or "").strip()
    if not text or not llm_backend:
        return None

    system_prompt = _build_system_prompt(source_language)

    try:
        raw = llm_backend.chat(system_prompt, text)
    except Exception as e:
        logger.warning(f"[TRANSLATE] vertalen mislukt (LLM-call): {e}")
        return None

    translation = (raw or "").strip()
    if not translation:
        logger.warning("[TRANSLATE] LLM gaf een lege respons terug")
        return None
    return translation
