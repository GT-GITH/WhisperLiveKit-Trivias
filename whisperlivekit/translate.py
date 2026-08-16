"""Vertaling van niet-Nederlandse transcript-tekst -- on-demand, per segment.

Zie features/vertaling-niet-nl-tekst.md in de projectrepo voor het volledige
ontwerp. Kern: alleen bevestigde (batch) tekst wordt op verzoek (klik op het
🌐-icoontje in de UI) vertaald naar het Nederlands, via het on-prem NLLB-200-
model (zie nllb_backend.py) -- geen persistente opslag in transcript.json
(een vertaling is een AI-afleiding, geen onderdeel van de autoritatieve
transcriptie, zie CLAUDE.md "audio is authoritative").

Draaide eerder op de generieke LLM-chat-backend (llm_backend.py/Ollama), maar
die bleek onbetrouwbaar voor morfologisch complexe brontalen (Turks): een
simpele testzin over een zieke moeder kwam terug met omgekeerde grammaticale
persoon. NLLB-200 is een model dat uitsluitend op vertalen getraind is en
heeft daardoor geen "raad-gedrag" -- vereist wel altijd een EXPLICIETE
brontaal (kan zelf niet detecteren, anders dan een chatmodel).

Puur, van FastAPI losgekoppeld -- fail-safe: geen backend, onbekende
brontaal, of een fout tijdens vertalen resulteren allemaal in None, nooit
een exception naar de aanroeper.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger("whisperlivekit.translate")

try:
    import langid
except ImportError:
    langid = None


def is_probably_dutch(text: str) -> bool:
    """Lichte, offline taalcheck (langid, geen download/internet nodig) om te
    voorkomen dat al-Nederlandse tekst tóch door NLLB gehaald wordt.

    Alleen zinnig voor kanalen met een GEGOKTE brontaal (bv. de tolk: de hint
    daar is de taal van de vreemdeling in dezelfde sessie, zie
    TriviasServer.py: _find_session_foreign_language() -- een tolk spreekt
    vaak ook gewoon Nederlands). NLLB kan, anders dan een chatmodel, niet zelf
    herkennen "dit is al de doeltaal, laat maar staan" -- forceer je toch een
    verkeerde brontaal (bv. "tr") op al-Nederlandse tekst, dan komt er
    betekenisloze onzin uit (geconstateerd 2026-08-16, testsessie).

    BEWUST NIET toegepast op "foreign_*"-kanalen: daar is de brontaal al
    betrouwbaar bekend (geconfigureerd, geen gok), en taalherkenning op korte
    fragmenten (een enkel woord als antwoord) is onbetrouwbaar genoeg om per
    ongeluk een echt niet-Nederlands fragment als Nederlands te verkeerd-
    classificeren en zo een eerder wél werkende vertaling te laten
    overslaan -- alleen de moeite waard waar het alternatief (blind
    vertrouwen op een gegokte brontaal) al slechter is.

    Fail-safe: als langid niet geïnstalleerd is, wordt hier NIET geblokkeerd
    (liever een keer een overbodige vertaalpoging dan de hele tolk-vertaling
    laten uitvallen op een ontbrekende dependency)."""
    if langid is None:
        return False
    try:
        lang, _score = langid.classify(text)
    except Exception:
        return False
    return lang == "nl"


def translate_text(text: str, source_language: Optional[str], nllb_backend: Optional[Any]) -> Optional[str]:
    """Vertaalt `text` naar het Nederlands via `nllb_backend` (whisperlivekit.nllb_backend.NLLBBackend).

    Retourneert None (nooit een exception) als: de tekst leeg is, er geen
    backend geconfigureerd is, de brontaal onbekend is (NLLB kan -- anders
    dan een chatmodel -- niet zelf raden), of de vertaal-call faalt. De
    aanroeper (het /translate-endpoint) is verantwoordelijk voor het omzetten
    van None naar een passende HTTP-fout, en voor het (waar relevant)
    vooraf checken van is_probably_dutch() -- zie die functie voor waarom
    dat niet hier, universeel, gebeurt."""
    text = (text or "").strip()
    if not text or not nllb_backend:
        return None

    try:
        translation = nllb_backend.translate(text, source_language, target_iso="nl")
    except Exception as e:
        logger.warning(f"[TRANSLATE] vertalen mislukt (NLLB-call): {e}")
        return None

    return translation or None
