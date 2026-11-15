"""Splits faster-whisper-segmenten op grote gaten tussen woorden.

Waarom (gemeten met de e2e-suite, 2026-11-15):
- Met vad_filter=True plakt faster-whisper spraakstukken aan elkaar en zet de
  tijdstempels daarna terug. Een segment kan dan over een weggefilterde stilte
  heen lopen: in `nl_lange_pauze` kreeg een uiting die op 82.8 s begint het
  segment 36.09–87.65 s, omdat het eerste woord ("De") op de naad tussen twee
  spraakstukken viel en vóór de stilte terechtkwam.
- Sommige finetunes (oddadmix/whisper-large-v3-turbo-arabic-dialectal) geven
  geen segmentgrenzen per uiting: één segment 10.8–49.1 s voor vijf losse
  uitingen met tot 8 s stilte ertussen.

In beide gevallen kloppen de WOORD-tijdstempels wel. Tijdstempels zijn de
primaire sleutel van dit project (CLAUDE.md), dus een segment wordt hier
opgesplitst waar twee opeenvolgende woorden meer dan `max_gap_s` uit elkaar
liggen.

Bewust ALLEEN splitsen, nooit een woord naar een andere tijd verschuiven: een
los naad-woord ("De" op 36.09 s) blijft dan een mini-segment op zijn eigen
(foute) plek, maar het alternatief -- korte groepjes aan de volgende groep
vastmaken -- zou een echt kort antwoord ("Ja.", stilte, verder praten) de
starttijd van de volgende zin geven. Een verzonnen tijdstempel is erger dan
een zichtbaar los woord.

Puur Python, geen faster-whisper-import: woorden zijn objecten of dicts met
`word`, `start`, `end`.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

DEFAULT_MAX_GAP_S = 1.2


def _get(w, key):
    return w.get(key) if isinstance(w, dict) else getattr(w, key, None)


def split_on_word_gaps(words: Iterable, max_gap_s: float = DEFAULT_MAX_GAP_S) -> List[Tuple[str, float, float]]:
    """Geef [(tekst, start, eind)] terug, één item per aaneengesloten woordgroep.

    Lege of ontbrekende woordenlijst -> []: de aanroeper houdt dan het
    oorspronkelijke segment."""
    ws = [w for w in (words or []) if (_get(w, "word") or "").strip() and _get(w, "start") is not None]
    if not ws:
        return []
    groups: List[list] = [[ws[0]]]
    for prev, cur in zip(ws, ws[1:]):
        if _get(cur, "start") - _get(prev, "end") > max_gap_s:
            groups.append([cur])
        else:
            groups[-1].append(cur)
    return [
        ("".join(_get(w, "word") for w in g).strip(), float(_get(g[0], "start")), float(_get(g[-1], "end")))
        for g in groups
    ]
