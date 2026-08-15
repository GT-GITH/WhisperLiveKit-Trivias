"""Gehoorverslag ("rapport van nader gehoor") -- docx-generatie.

Bouwt een Word-document uit een reeds samengevoegde, chronologisch
gesorteerde transcript-lijst (zelfde vorm als de `channel_id=all`-tak van
`GET /sessions/{id}/transcript` in `TriviasServer.py`), in lijn met de
officiële IND-werkinstructie WI 2021/13 "Nader gehoor": letterlijke
weergave van vragen en antwoorden, hoormedewerker-tekst cursief t.o.v.
overige rollen, nooit content weglaten.

v1 doet BEWUST geen automatische indeling in IND's fijnmazige subsecties
(2.1/2.2/3/4.1-4.6/5.1) en geen AI-samenvatting (§4.4) -- die vergen eigen,
voorzichtig ontworpen guardrails vanwege de harde "geen oordeel over
geloofwaardigheid"-eis uit dezelfde werkinstructie. Zie het bijbehorende
plan (feat-gehoorverslag) voor de volledige scope-afbakening.

Puur, van FastAPI losgekoppeld: elke functie is los te unit-testen met
synthetische segment-lijsten (zelfde patroon als cross_channel_gate.py).
Wijzigt nooit de input-lijst.
"""

from typing import Any, Dict, List, Optional

from docx import Document
from docx.shared import Pt

# Python-poort van web_trivias/app.js's ROLES/getRoleLabel() -- houd deze
# twee plekken bewust in sync als de frontend-rollen ooit wijzigen.
ROLE_LABELS: Dict[str, str] = {
    "employee": "Hoormedewerker",
    "interpreter": "Tolk",
    "lawyer": "Gemachtigde",
    "foreign": "Vreemdeling",
    "default": "Spreker",
}

DEFAULT_PAUSE_GAP_THRESHOLD_MS = 10_000
FONT_NAME = "Verdana"
FONT_SIZE_PT = 9


def channel_id_to_role(channel_id: str) -> str:
    """Python-poort van app.js's channelIdToRoleId(): alle foreign_<taal>-
    kanalen vallen samen tot rol "foreign", overige kanalen blijven zichzelf."""
    if not channel_id:
        return "default"
    if channel_id.startswith("foreign"):
        return "foreign"
    return channel_id


def role_label(channel_id: str) -> str:
    role = channel_id_to_role(channel_id)
    return ROLE_LABELS.get(role, role)


def _format_timestamp(ms: Optional[int]) -> str:
    if ms is None:
        return "??:??"
    total_seconds = int(ms) // 1000
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _set_base_style(document: "Document") -> None:
    style = document.styles["Normal"]
    style.font.name = FONT_NAME
    style.font.size = Pt(FONT_SIZE_PT)


def _add_styled_paragraph(document: "Document", text: str, *, bold: bool = False, italic: bool = False):
    p = document.add_paragraph()
    run = p.add_run(text)
    run.font.name = FONT_NAME
    run.font.size = Pt(FONT_SIZE_PT)
    run.bold = bold
    run.italic = italic
    return p


def _add_voorblad(
    document: "Document",
    session_id: str,
    ordered_segments: List[Dict[str, Any]],
    session_meta: Optional[Dict[str, Any]],
) -> None:
    _add_styled_paragraph(document, "Rapport van gehoor", bold=True)
    _add_styled_paragraph(document, "Automatisch gegenereerd door Trivias STT -- zie opmerking onderaan dit voorblad.")

    channels_seen: Dict[str, str] = {}
    for seg in ordered_segments:
        ch = seg.get("channel_id")
        if ch and ch not in channels_seen:
            channels_seen[ch] = role_label(ch)

    first_ms = ordered_segments[0].get("start_ms") if ordered_segments else None
    last_ms = ordered_segments[-1].get("end_ms") if ordered_segments else None
    session_meta = session_meta or {}
    languages = session_meta.get("languages", {})

    table = document.add_table(rows=0, cols=2)

    def _row(label: str, value: Optional[str]) -> None:
        cells = table.add_row().cells
        cells[0].text = label
        cells[1].text = value or "(niet vastgelegd -- handmatig aanvullen)"

    _row("Sessie-ID", session_id)
    _row("Datum", session_meta.get("date"))
    _row("Aanvangstijd (relatief)", _format_timestamp(first_ms))
    _row("Eindtijd (relatief)", _format_timestamp(last_ms))
    for ch, label in channels_seen.items():
        _row(f"Kanaal -- {label}", languages.get(ch, ch))
    _row("Naam hoormedewerker", None)
    _row("Naam tolk / registratieniveau", None)

    _add_styled_paragraph(
        document,
        "Let op: dit document is automatisch gegenereerd vanaf de opgenomen "
        "audio en is een letterlijke weergave, geen beoordeling van "
        "geloofwaardigheid en geen beslissing. Velden hierboven die niet "
        "automatisch zijn ingevuld ('handmatig aanvullen') dienen te worden "
        "aangevuld voordat dit rapport als officieel rapport van gehoor "
        "wordt gebruikt. Pauze-annotaties in dit document zijn een "
        "benadering (afgeleid uit tijdsgaten tussen segmenten), geen harde "
        "stilte-detectie.",
    )
    document.add_page_break()


def build_gehoorverslag_docx(
    session_id: str,
    merged_segments: List[Dict[str, Any]],
    session_meta: Optional[Dict[str, Any]] = None,
    pause_gap_threshold_ms: int = DEFAULT_PAUSE_GAP_THRESHOLD_MS,
) -> "Document":
    """Bouwt het Word-document: voorblad + chronologische, cursief/normaal
    geformatteerde vraag-antwoord-body + benaderde 'opmerking rapporteur'-
    regels bij tijdsgaten >= pause_gap_threshold_ms.

    merged_segments: lijst van dicts met minimaal channel_id, start_ms,
    end_ms, text_final (zelfde vorm als de merged-transcript-helper in
    TriviasServer.py). Segmenten zonder (niet-lege) text_final worden
    overgeslagen -- er is dan geen content om te rapporteren -- maar elk
    segment MET tekst wordt altijd opgenomen, nooit gefilterd op
    vertrouwen/rol: dat is een harde IND-eis (nooit een verklaring
    weglaten), niet iets waar deze functie een oordeel over mag hebben.
    """
    ordered = sorted(
        (s for s in merged_segments if (s.get("text_final") or "").strip()),
        key=lambda s: s.get("start_ms") if s.get("start_ms") is not None else 0,
    )

    document = Document()
    _set_base_style(document)

    _add_voorblad(document, session_id, ordered, session_meta)

    _add_styled_paragraph(document, "Verloop van het gehoor", bold=True)

    prev_end_ms: Optional[int] = None
    for seg in ordered:
        start_ms = seg.get("start_ms")
        if prev_end_ms is not None and start_ms is not None:
            gap = start_ms - prev_end_ms
            if gap >= pause_gap_threshold_ms:
                _add_styled_paragraph(
                    document,
                    f"Opmerking rapporteur: [pauze van ongeveer {gap // 1000}s "
                    "-- afgeleid uit tijdsverloop tussen segmenten, geen harde detectie]",
                    italic=True,
                )

        channel_id = seg.get("channel_id", "") or ""
        label = role_label(channel_id)
        text = (seg.get("text_final") or "").strip()
        is_hoormedewerker = channel_id_to_role(channel_id) == "employee"
        _add_styled_paragraph(
            document,
            f"[{_format_timestamp(start_ms)}] {label}: {text}",
            italic=is_hoormedewerker,
        )

        end_ms = seg.get("end_ms")
        prev_end_ms = end_ms if end_ms is not None else start_ms

    return document
