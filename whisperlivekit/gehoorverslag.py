"""Gehoorverslag ("rapport van nader gehoor") -- docx-generatie + classificatie.

Bouwt een Word-document uit een reeds samengevoegde, chronologisch
gesorteerde transcript-lijst (zelfde vorm als de `channel_id=all`-tak van
`GET /sessions/{id}/transcript` in `TriviasServer.py`), in lijn met de
officiële IND-werkinstructie WI 2021/13 "Nader gehoor": letterlijke
weergave van vragen en antwoorden, hoormedewerker-tekst cursief t.o.v.
overige rollen, nooit content weglaten, vaste sectiestructuur/voorblad.

Sectie-CLASSIFICATIE (welk segment hoort bij welke IND-sectie) is expliciet
onderscheiden van een geloofwaardigheids-BEOORDELING: classificeren is
structureel indelen, geen oordeel -- de hoormedewerker leest na en
corrigeert (FO §11 "ondersteunt, vervangt niet"). Classificatie gebeurt via
een optionele, on-prem LLM-backend (zie llm_backend.py) en is VOLLEDIG
fail-safe: zonder geconfigureerde backend, of bij elke fout tijdens
classificeren, valt dit document terug op een platte chronologische
weergave -- nooit een harde afhankelijkheid.

Nog steeds bewust BUITEN scope: AI-gegenereerde samenvatting (§4.4) en
betrouwbare pauze/onderbrekingsdetectie (alleen een tijdsgat-benadering).
Zie het bijbehorende plan voor de volledige scope-afbakening.

Puur, van FastAPI losgekoppeld: elke functie is los te unit-testen met
synthetische segment-lijsten (zelfde patroon als cross_channel_gate.py).
Wijzigt nooit de input-lijst.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from docx import Document
from docx.shared import Pt

logger = logging.getLogger("whisperlivekit.gehoorverslag")

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

# Officiële sectiecodes + titels uit IND-werkinstructie WI 2021/13 "Nader
# gehoor", in documentvolgorde. "overig" is geen IND-code maar een bewust
# toegevoegd vangnet: elk segment dat niet (betrouwbaar) geclassificeerd kon
# worden landt hier, zodat nooit content stilzwijgend verdwijnt.
IND_SECTIONS: List[Tuple[str, str]] = [
    ("2.1", "Correcties en aanvullingen"),
    ("2.2", "Bespreking onderzoek(en)"),
    ("3", "Reden asielaanvraag (asielrelaas)"),
    ("4.1", "Inleidende vragen"),
    ("4.2", "Verdere vragen"),
    ("4.3", "Overig asielrelaas"),
    ("4.5", "Bijzondere individuele omstandigheden"),
    (
        "4.6",
        "Getuigenissen van oorlogsmisdrijven, misdrijven tegen de menselijkheid, "
        "foltering, genocide en ernstige niet-politieke misdrijven (1F(b))",
    ),
    ("5.1", "Op- en aanmerkingen over de gang van zaken tijdens het gehoor"),
    ("overig", "Overig / niet geclassificeerd"),
]
_IND_SECTION_TITLES: Dict[str, str] = dict(IND_SECTIONS)
_VALID_SECTION_CODES = frozenset(code for code, _ in IND_SECTIONS)

# Korte, voor een klein lokaal LLM behapbare omschrijving per sectie --
# gebruikt in de classificatie-prompt, geen letterlijke kopie van de
# werkinstructie (die is voor een hoormedewerker geschreven, niet voor een
# classificatie-taak).
_SECTION_PROMPT_DESCRIPTIONS: Dict[str, str] = {
    "2.1": "correcties/aanvullingen op een eerder (aanmeld)gehoor worden besproken",
    "2.2": "resultaten van onderzoek (documenten, leeftijd, herkomst, taalanalyse) worden besproken, of nieuwe documenten",
    "3": "het asielrelaas zelf: de vreemdeling vertelt in eigen woorden, grotendeels ononderbroken, over de directe reden van vertrek",
    "4.1": "korte inleidende vragen van de hoormedewerker naar de reden van vertrek",
    "4.2": "gerichte vervolgvragen van de hoormedewerker over details van het asielrelaas",
    "4.3": "vragen over verblijfsmogelijkheden elders, vrouwenbesnijdenis, of ervaringen van kinderen/familie",
    "4.5": "bijzondere individuele omstandigheden los van het asielrelaas zelf",
    "4.6": "getuigenissen over oorlogsmisdrijven of andere ernstige misdrijven",
    "5.1": "klachten, opmerkingen over de gang van zaken tijdens het gehoor zelf, of de afsluiting van het gehoor",
    "overig": "past nergens goed bij, of je bent onzeker",
}


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
    # Velden die WI 2021/13 als vast onderdeel van het voorblad noemt, maar
    # die Trivias nergens vastlegt (client-side, of helemaal niet ingevoerd)
    # -- expliciet als invulplaceholder, nooit verzonnen.
    _row("Naam hoormedewerker (geslacht)", None)
    _row("Naam tolk (geslacht)", None)
    _row("Tolk-registratieniveau (of reden niet-registertolk)", None)
    _row("Naam/aanwezigheid gemachtigde of hulpverlener (VWN)", None)
    _row("Termijn voor correcties en aanvullingen", None)

    _add_styled_paragraph(
        document,
        "Let op: dit document is automatisch gegenereerd vanaf de opgenomen "
        "audio en is een letterlijke weergave, geen beoordeling van "
        "geloofwaardigheid en geen beslissing. Sectie-indeling hieronder (indien "
        "aanwezig) is een classificatie ter ondersteuning, geen oordeel -- "
        "controleer en corrigeer waar nodig. Velden hierboven die niet "
        "automatisch zijn ingevuld ('handmatig aanvullen') dienen te worden "
        "aangevuld voordat dit rapport als officieel rapport van gehoor "
        "wordt gebruikt. Pauze-annotaties in dit document zijn een "
        "benadering (afgeleid uit tijdsgaten tussen segmenten), geen harde "
        "stilte-detectie.",
    )
    document.add_page_break()


def _render_segment(document: "Document", seg: Dict[str, Any]) -> None:
    channel_id = seg.get("channel_id", "") or ""
    label = role_label(channel_id)
    text = (seg.get("text_final") or "").strip()
    is_hoormedewerker = channel_id_to_role(channel_id) == "employee"
    _add_styled_paragraph(
        document,
        f"[{_format_timestamp(seg.get('start_ms'))}] {label}: {text}",
        italic=is_hoormedewerker,
    )


def _render_flat_body(
    document: "Document",
    ordered_segments: List[Dict[str, Any]],
    pause_gap_threshold_ms: int,
) -> None:
    """Platte, puur chronologische weergave -- de fallback zonder
    classificatie (geen LLM-backend geconfigureerd, of classificatie leverde
    niets bruikbaars op). Dit was v1's enige rendering en blijft ongewijzigd
    beschikbaar."""
    _add_styled_paragraph(document, "Verloop van het gehoor", bold=True)

    prev_end_ms: Optional[int] = None
    for seg in ordered_segments:
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
        _render_segment(document, seg)
        end_ms = seg.get("end_ms")
        prev_end_ms = end_ms if end_ms is not None else start_ms


def _render_grouped_body(
    document: "Document",
    ordered_segments: List[Dict[str, Any]],
    section_labels: Dict[int, str],
) -> None:
    """Gegroepeerd per IND-sectie (documentvolgorde uit IND_SECTIONS),
    chronologisch binnen elke sectie. Geen pauze-annotaties hier: zodra
    content is heringedeeld op thema i.p.v. tijd, betekent een tijdsgat
    tussen twee ANGRENZENDE regels in dit document niet meer "hier viel een
    stilte in het gesprek" -- dat zou misleidend zijn. Elk segment landt in
    precies één sectie (nooit weggelaten): onbekend/laag-vertrouwen -> "overig"."""
    by_section: Dict[str, List[Dict[str, Any]]] = {code: [] for code, _ in IND_SECTIONS}
    for idx, seg in enumerate(ordered_segments):
        code = section_labels.get(idx, "overig")
        if code not in by_section:
            code = "overig"
        by_section[code].append(seg)

    for code, title in IND_SECTIONS:
        segs = by_section[code]
        if not segs:
            continue
        _add_styled_paragraph(document, f"{code} {title}", bold=True)
        for seg in segs:
            _render_segment(document, seg)


def order_segments_for_report(merged_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter (geen lege tekst) + chronologisch sorteren -- gedeeld tussen
    build_gehoorverslag_docx() en classify_segments(), zodat een
    section_labels-dict (list-index -> sectiecode) altijd naar precies
    dezelfde volgorde verwijst in beide functies. Nooit los opnieuw
    implementeren, anders lopen de indices uiteen."""
    return sorted(
        (s for s in merged_segments if (s.get("text_final") or "").strip()),
        key=lambda s: s.get("start_ms") if s.get("start_ms") is not None else 0,
    )


def build_gehoorverslag_docx(
    session_id: str,
    merged_segments: List[Dict[str, Any]],
    session_meta: Optional[Dict[str, Any]] = None,
    pause_gap_threshold_ms: int = DEFAULT_PAUSE_GAP_THRESHOLD_MS,
    section_labels: Optional[Dict[int, str]] = None,
) -> "Document":
    """Bouwt het Word-document: voorblad + hetzij een platte chronologische
    body (section_labels=None) hetzij een per-IND-sectie gegroepeerde body
    (section_labels gezet, zie classify_segments()).

    merged_segments: lijst van dicts met minimaal channel_id, start_ms,
    end_ms, text_final (zelfde vorm als de merged-transcript-helper in
    TriviasServer.py). Segmenten zonder (niet-lege) text_final worden
    overgeslagen -- er is dan geen content om te rapporteren -- maar elk
    segment MET tekst wordt altijd opgenomen, nooit gefilterd op
    vertrouwen/rol: dat is een harde IND-eis (nooit een verklaring
    weglaten), niet iets waar deze functie een oordeel over mag hebben.

    section_labels: {list-index in de gesorteerde/gefilterde segmentenlijst
    -> IND-sectiecode}, zoals geretourneerd door classify_segments(). Een
    ontbrekende of ongeldige code voor een index valt terug op "overig" --
    nooit een KeyError, nooit verloren content.
    """
    ordered = order_segments_for_report(merged_segments)

    document = Document()
    _set_base_style(document)

    _add_voorblad(document, session_id, ordered, session_meta)

    if section_labels:
        _render_grouped_body(document, ordered, section_labels)
    else:
        _render_flat_body(document, ordered, pause_gap_threshold_ms)

    return document


# --- Classificatie (optioneel, on-prem LLM) ---------------------------------

_CLASSIFY_SYSTEM_PROMPT = (
    "Je helpt een IND-hoormedewerker door delen van een asielgehoor-transcript "
    "te SORTEREN onder de juiste standaardsectie van het officiële rapport van "
    "nader gehoor. Dit is een structurele indeling, GEEN beoordeling van "
    "geloofwaardigheid en geen beslissing over de aanvraag -- daar doe je geen "
    "uitspraak over. De hoormedewerker controleert en corrigeert je indeling "
    "achteraf altijd zelf.\n\n"
    "Beschikbare secties:\n"
    + "\n".join(f"- {code}: {_SECTION_PROMPT_DESCRIPTIONS[code]}" for code, _ in IND_SECTIONS)
    + "\n\nAntwoord UITSLUITEND met geldige JSON: een object dat elk segmentnummer "
    '(als string) koppelt aan precies één sectiecode, bv. {"0": "3", "1": "3", "2": "4.2"}. '
    "Geen uitleg, geen markdown, geen tekst buiten de JSON."
)


def _build_classify_user_prompt(batch: List[Dict[str, Any]]) -> str:
    lines = ["Segmenten (nummer: rol: tekst):"]
    for i, seg in enumerate(batch):
        label = role_label(seg.get("channel_id", "") or "")
        text = (seg.get("text_final") or "").strip()
        lines.append(f"{i}: {label}: {text}")
    return "\n".join(lines)


def _extract_json_object(raw: str) -> Optional[dict]:
    """Defensieve JSON-extractie: kleine lokale modellen volgen 'alleen JSON'
    niet altijd perfect (bv. een ```json-codeblok eromheen). Probeer eerst
    de ruwe tekst, anders het eerste {...}-blok erin."""
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def classify_segments(
    ordered_segments: List[Dict[str, Any]],
    llm_backend: Optional[Any],
    batch_size: int = 10,
) -> Dict[int, str]:
    """Best-effort classificatie van elk segment (list-index in
    ordered_segments) naar een IND_SECTIONS-code.

    VOLLEDIG fail-safe: geen backend, een timeout/HTTP-fout, kapotte of
    ongeldige JSON-output, een onbekend sectielabel -- elk van deze gevallen
    resulteert in het overslaan van precies dat segment/die batch, nooit een
    exception naar de aanroeper en nooit de hele generatie blokkeren.

    batch_size=10 is empirisch bepaald (getest tegen lokale Ollama,
    llama3.1:8b): een batch van 20 liep tegen de default LLMBackend-timeout
    aan (generatietijd domineert, niet netwerk-overhead, dus grotere batches
    winnen weinig aan totale doorlooptijd en verhogen het risico op
    afgekapte/onvolledige JSON bij een klein lokaal model). Op trager
    on-prem hardware kan een volledig gehoor (honderden segmenten) alsnog
    traag classificeren -- dit is een bekende v1-beperking, geen bug: elk
    niet-tijdig geclassificeerd segment landt gewoon in "overig", nooit
    verloren, wel minder overzichtelijk voor de hoormedewerker."""
    if not llm_backend or not ordered_segments:
        return {}

    labels: Dict[int, str] = {}
    for batch_start in range(0, len(ordered_segments), batch_size):
        batch = ordered_segments[batch_start:batch_start + batch_size]
        try:
            raw = llm_backend.chat(_CLASSIFY_SYSTEM_PROMPT, _build_classify_user_prompt(batch))
        except Exception as e:
            logger.warning(
                f"[GEHOORVERSLAG][CLASSIFY] batch {batch_start}..{batch_start + len(batch)} "
                f"mislukt (LLM-call), overgeslagen: {e}"
            )
            continue

        parsed = _extract_json_object(raw)
        if not isinstance(parsed, dict):
            logger.warning(
                f"[GEHOORVERSLAG][CLASSIFY] batch {batch_start}..{batch_start + len(batch)} "
                f"gaf geen geldige JSON terug, overgeslagen. Ruwe output: {raw[:200]!r}"
            )
            continue

        n_accepted = 0
        for key, value in parsed.items():
            try:
                local_idx = int(key)
            except (TypeError, ValueError):
                continue
            if not (0 <= local_idx < len(batch)):
                continue
            code = str(value).strip()
            if code not in _VALID_SECTION_CODES:
                continue
            labels[batch_start + local_idx] = code
            n_accepted += 1
        logger.info(
            f"[GEHOORVERSLAG][CLASSIFY] batch {batch_start}..{batch_start + len(batch)}: "
            f"{n_accepted}/{len(batch)} segmenten geclassificeerd"
        )

    return labels
