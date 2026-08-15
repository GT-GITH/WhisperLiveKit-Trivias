"""Synthetische tests voor whisperlivekit/gehoorverslag.py.

Geen testsuite/pytest-dependency in dit project (zie CLAUDE.md) -- draai dit
bestand rechtstreeks: `python tests/test_gehoorverslag.py`. Focus ligt op de
IND-harde eisen uit WI 2021/13 die het ontwerp bepalen: nooit content
weglaten, hoormedewerker-tekst cursief t.o.v. overige rollen, chronologische
volgorde, en dat pauze-annotaties alleen als benadering verschijnen (nooit
als harde detectie gepresenteerd) en alleen boven de drempel.
"""

import importlib.util
import sys
from pathlib import Path

# gehoorverslag.py is verder onafhankelijk van de rest van het pakket (alleen
# python-docx nodig) -- rechtstreeks via bestandspad laden i.p.v. via
# `whisperlivekit.gehoorverslag`, want dat laatste triggert
# whisperlivekit/__init__.py, dat de hele (zware) ASR-stack importeert.
_MODULE_PATH = Path(__file__).resolve().parent.parent / "whisperlivekit" / "gehoorverslag.py"
_spec = importlib.util.spec_from_file_location("gehoorverslag", _MODULE_PATH)
_gv = importlib.util.module_from_spec(_spec)
sys.modules["gehoorverslag"] = _gv
_spec.loader.exec_module(_gv)

channel_id_to_role = _gv.channel_id_to_role
role_label = _gv.role_label
build_gehoorverslag_docx = _gv.build_gehoorverslag_docx
classify_segments = _gv.classify_segments
order_segments_for_report = _gv.order_segments_for_report
_extract_json_object = _gv._extract_json_object
IND_SECTIONS = _gv.IND_SECTIONS


def _body_paragraphs(document):
    """Alinea's ná de platte 'Verloop van het gehoor'-kop, d.w.z. alleen de
    segment-regels zelf -- gebruikt door de tests voor de platte (niet-
    gegroepeerde) weergave."""
    texts = [p.text for p in document.paragraphs]
    idx = texts.index("Verloop van het gehoor")
    return document.paragraphs[idx + 1:]


def _paragraphs_after_pagebreak(document):
    """Alles ná het voorblad (na de page-break-alinea, altijd leeg),
    INCLUSIEF eventuele sectiekoppen -- gebruikt door de gegroepeerde-
    weergave-test, die de koppen zelf juist wil controleren."""
    texts = [p.text for p in document.paragraphs]
    idx = texts.index("")  # de page-break-alinea
    return document.paragraphs[idx + 1:]


class _FakeLLMBackend:
    """Test-double voor llm_backend.LLMBackend -- geen echte HTTP-call."""

    def __init__(self, response=None, raise_exc=None):
        self._response = response
        self._raise_exc = raise_exc
        self.calls = 0

    def chat(self, system_prompt, user_prompt, *, temperature=0.0):
        self.calls += 1
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response


def test_role_mapping():
    assert channel_id_to_role("employee") == "employee"
    assert channel_id_to_role("interpreter") == "interpreter"
    assert channel_id_to_role("lawyer") == "lawyer"
    assert channel_id_to_role("foreign_ar") == "foreign"
    assert channel_id_to_role("foreign_tr") == "foreign"
    assert channel_id_to_role("default") == "default"
    assert channel_id_to_role("") == "default"

    assert role_label("employee") == "Hoormedewerker"
    assert role_label("interpreter") == "Tolk"
    assert role_label("foreign_ar") == "Vreemdeling"
    assert role_label("unknown_channel") == "unknown_channel"  # fail-safe: label = raw id, niet verzinnen
    print("OK test_role_mapping")


def test_no_content_dropped():
    segments = [
        {"channel_id": "employee", "start_ms": 0, "end_ms": 1000, "text_final": "Vraag een."},
        {"channel_id": "foreign_ar", "start_ms": 1000, "end_ms": 2000, "text_final": "Antwoord een."},
        {"channel_id": "employee", "start_ms": 2000, "end_ms": 3000, "text_final": "Vraag twee."},
        {"channel_id": "foreign_ar", "start_ms": 3000, "end_ms": 4000, "text_final": "Ik weet het niet."},
        {"channel_id": "interpreter", "start_ms": 4000, "end_ms": 4500, "text_final": "(vertaling)"},
    ]
    doc = build_gehoorverslag_docx("sess-1", segments)
    body_text = "\n".join(p.text for p in _body_paragraphs(doc))
    for seg in segments:
        assert seg["text_final"] in body_text, f"tekst ontbreekt in output: {seg['text_final']!r}"
    print("OK test_no_content_dropped")


def test_empty_text_segments_skipped():
    segments = [
        {"channel_id": "employee", "start_ms": 0, "end_ms": 1000, "text_final": "Echte vraag."},
        {"channel_id": "foreign_ar", "start_ms": 1000, "end_ms": 1500, "text_final": "   "},  # alleen whitespace
        {"channel_id": "foreign_ar", "start_ms": 1500, "end_ms": 2000, "text_final": ""},  # leeg
    ]
    doc = build_gehoorverslag_docx("sess-2", segments)
    body = _body_paragraphs(doc)
    non_empty_lines = [p.text for p in body if p.text.strip()]
    assert len(non_empty_lines) == 1, f"verwacht 1 regel (lege segmenten overgeslagen), kreeg {len(non_empty_lines)}"
    print("OK test_empty_text_segments_skipped")


def test_hoormedewerker_italic_others_normal():
    segments = [
        {"channel_id": "employee", "start_ms": 0, "end_ms": 1000, "text_final": "Kunt u dat toelichten?"},
        {"channel_id": "foreign_ar", "start_ms": 1000, "end_ms": 2000, "text_final": "Ja, dat kan."},
        {"channel_id": "interpreter", "start_ms": 2000, "end_ms": 2500, "text_final": "Vertaling van het antwoord."},
        {"channel_id": "lawyer", "start_ms": 2500, "end_ms": 3000, "text_final": "Ik heb een opmerking."},
    ]
    doc = build_gehoorverslag_docx("sess-3", segments)
    body = [p for p in _body_paragraphs(doc) if p.text.strip()]
    assert len(body) == 4
    expected_italic = [True, False, False, False]  # alleen employee (Hoormedewerker) cursief
    for p, expect in zip(body, expected_italic):
        actual = all(r.italic for r in p.runs if r.text.strip())
        assert actual == expect, f"cursief-status fout voor {p.text!r}: verwacht {expect}"
    print("OK test_hoormedewerker_italic_others_normal")


def test_chronological_ordering_regardless_of_input_order():
    # Bewust door elkaar aangeleverd (zoals een merge van meerdere kanaal-
    # bestanden dat kan opleveren vóór sortering).
    segments = [
        {"channel_id": "foreign_ar", "start_ms": 5000, "end_ms": 6000, "text_final": "Derde."},
        {"channel_id": "employee", "start_ms": 0, "end_ms": 1000, "text_final": "Eerste."},
        {"channel_id": "foreign_ar", "start_ms": 2000, "end_ms": 3000, "text_final": "Tweede."},
    ]
    doc = build_gehoorverslag_docx("sess-4", segments)
    body = [p.text for p in _body_paragraphs(doc) if p.text.strip()]
    assert body[0].endswith("Eerste.")
    assert body[1].endswith("Tweede.")
    assert body[2].endswith("Derde.")
    print("OK test_chronological_ordering_regardless_of_input_order")


def test_pause_marker_only_above_threshold():
    segments = [
        {"channel_id": "employee", "start_ms": 0, "end_ms": 1000, "text_final": "Vraag."},
        {"channel_id": "foreign_ar", "start_ms": 1500, "end_ms": 2000, "text_final": "Kort antwoord (klein gat)."},
        {"channel_id": "employee", "start_ms": 30000, "end_ms": 31000, "text_final": "Vervolgvraag (groot gat)."},
    ]
    doc = build_gehoorverslag_docx("sess-5", segments, pause_gap_threshold_ms=10_000)
    body = [p.text for p in _body_paragraphs(doc) if p.text.strip()]
    pause_lines = [t for t in body if t.startswith("Opmerking rapporteur")]
    assert len(pause_lines) == 1, f"verwacht precies 1 pauze-annotatie, kreeg {len(pause_lines)}: {pause_lines}"
    assert "benadering" not in pause_lines[0].lower() or "afgeleid" in pause_lines[0].lower(), (
        "pauze-annotatie moet zichzelf expliciet als afgeleid/benaderd labelen, niet als harde detectie"
    )
    print("OK test_pause_marker_only_above_threshold")


def test_voorblad_has_placeholders_for_unknown_fields():
    segments = [
        {"channel_id": "employee", "start_ms": 0, "end_ms": 1000, "text_final": "Iets."},
    ]
    doc = build_gehoorverslag_docx("sess-6", segments)
    table = doc.tables[0]
    rows = {row.cells[0].text: row.cells[1].text for row in table.rows}
    assert rows["Sessie-ID"] == "sess-6"
    for label in (
        "Naam hoormedewerker (geslacht)",
        "Naam tolk (geslacht)",
        "Tolk-registratieniveau (of reden niet-registertolk)",
        "Naam/aanwezigheid gemachtigde of hulpverlener (VWN)",
        "Termijn voor correcties en aanvullingen",
    ):
        assert "handmatig aanvullen" in rows[label], (
            f"onbekend veld {label!r} moet een expliciete invulplaceholder krijgen, nooit verzonnen data"
        )
    print("OK test_voorblad_has_placeholders_for_unknown_fields")


def test_empty_session_does_not_crash():
    doc = build_gehoorverslag_docx("sess-empty", [])
    # Moet gewoon een geldig (leeg) document opleveren, geen exception.
    assert doc is not None
    print("OK test_empty_session_does_not_crash")


def test_grouped_rendering_by_section():
    segments = [
        {"channel_id": "employee", "start_ms": 0, "end_ms": 1000, "text_final": "Vertelt u eens over uw vertrek."},
        {"channel_id": "foreign_ar", "start_ms": 1000, "end_ms": 5000, "text_final": "Lang asielrelaas hier."},
        {"channel_id": "employee", "start_ms": 5000, "end_ms": 6000, "text_final": "Klopt het dat u toen alleen was?"},
        {"channel_id": "foreign_ar", "start_ms": 6000, "end_ms": 7000, "text_final": "Ja, dat klopt."},
        {"channel_id": "employee", "start_ms": 7000, "end_ms": 8000, "text_final": "Iets wat niet paste."},
    ]
    # index 0,1 -> "3" (asielrelaas), 2,3 -> "4.2" (verdere vragen), 4 -> geen label (-> overig)
    section_labels = {0: "3", 1: "3", 2: "4.2", 3: "4.2"}
    doc = build_gehoorverslag_docx("sess-7", segments, section_labels=section_labels)
    body = [p.text for p in _paragraphs_after_pagebreak(doc) if p.text.strip()]

    idx_3 = body.index("3 Reden asielaanvraag (asielrelaas)")
    idx_42 = body.index("4.2 Verdere vragen")
    idx_overig = body.index("overig Overig / niet geclassificeerd")
    # IND_SECTIONS-documentvolgorde: 3 vóór 4.2 vóór overig.
    assert idx_3 < idx_42 < idx_overig
    assert "Vertelt u eens over uw vertrek." in body[idx_3 + 1]
    assert "Lang asielrelaas hier." in body[idx_3 + 2]
    assert "Klopt het dat u toen alleen was?" in body[idx_42 + 1]
    assert "Iets wat niet paste." in body[idx_overig + 1], "ongelabeld segment moet onder 'overig' landen, niet verdwijnen"
    print("OK test_grouped_rendering_by_section")


def test_extract_json_object_handles_markdown_fence():
    clean = _extract_json_object('{"0": "3", "1": "4.2"}')
    assert clean == {"0": "3", "1": "4.2"}

    fenced = _extract_json_object('```json\n{"0": "3"}\n```')
    assert fenced == {"0": "3"}

    garbage = _extract_json_object("dit is geen JSON")
    assert garbage is None
    print("OK test_extract_json_object_handles_markdown_fence")


def test_classify_segments_no_backend_returns_empty():
    segments = [{"channel_id": "employee", "start_ms": 0, "end_ms": 1000, "text_final": "Iets."}]
    ordered = order_segments_for_report(segments)
    assert classify_segments(ordered, None) == {}
    print("OK test_classify_segments_no_backend_returns_empty")


def test_classify_segments_fail_safe_on_exception():
    segments = [{"channel_id": "employee", "start_ms": 0, "end_ms": 1000, "text_final": "Iets."}]
    ordered = order_segments_for_report(segments)
    backend = _FakeLLMBackend(raise_exc=RuntimeError("connection refused"))
    result = classify_segments(ordered, backend)  # mag NOOIT raisen
    assert result == {}
    print("OK test_classify_segments_fail_safe_on_exception")


def test_classify_segments_fail_safe_on_garbage_response():
    segments = [{"channel_id": "employee", "start_ms": 0, "end_ms": 1000, "text_final": "Iets."}]
    ordered = order_segments_for_report(segments)
    backend = _FakeLLMBackend(response="Sorry, ik kan deze vraag niet beantwoorden.")
    result = classify_segments(ordered, backend)
    assert result == {}
    print("OK test_classify_segments_fail_safe_on_garbage_response")


def test_classify_segments_parses_valid_response_and_filters_invalid_codes():
    segments = [
        {"channel_id": "employee", "start_ms": 0, "end_ms": 1000, "text_final": "Vraag 0."},
        {"channel_id": "foreign_ar", "start_ms": 1000, "end_ms": 2000, "text_final": "Antwoord 1."},
        {"channel_id": "employee", "start_ms": 2000, "end_ms": 3000, "text_final": "Vraag 2."},
    ]
    ordered = order_segments_for_report(segments)
    # index 2 krijgt een niet-bestaande sectiecode -- moet genegeerd worden,
    # niet als geldig label doorsijpelen.
    backend = _FakeLLMBackend(response='{"0": "4.1", "1": "3", "2": "9.9-bestaat-niet"}')
    result = classify_segments(ordered, backend, batch_size=25)
    assert result == {0: "4.1", 1: "3"}
    assert 2 not in result
    assert backend.calls == 1
    print("OK test_classify_segments_parses_valid_response_and_filters_invalid_codes")


def test_classify_segments_batches_large_input():
    segments = [
        {"channel_id": "employee", "start_ms": i * 1000, "end_ms": i * 1000 + 500, "text_final": f"Zin {i}."}
        for i in range(30)
    ]
    ordered = order_segments_for_report(segments)
    backend = _FakeLLMBackend(response='{"0": "4.1"}')  # elke batch "ziet" alleen z'n eigen lokale index 0
    result = classify_segments(ordered, backend, batch_size=10)
    assert backend.calls == 3, f"30 segmenten / batch_size=10 moet 3 calls geven, kreeg {backend.calls}"
    # lokale index 0 van elke batch -> globale index 0, 10, 20
    assert set(result.keys()) == {0, 10, 20}
    print("OK test_classify_segments_batches_large_input")


if __name__ == "__main__":
    tests = [
        test_role_mapping,
        test_no_content_dropped,
        test_empty_text_segments_skipped,
        test_hoormedewerker_italic_others_normal,
        test_chronological_ordering_regardless_of_input_order,
        test_pause_marker_only_above_threshold,
        test_voorblad_has_placeholders_for_unknown_fields,
        test_empty_session_does_not_crash,
        test_grouped_rendering_by_section,
        test_extract_json_object_handles_markdown_fence,
        test_classify_segments_no_backend_returns_empty,
        test_classify_segments_fail_safe_on_exception,
        test_classify_segments_fail_safe_on_garbage_response,
        test_classify_segments_parses_valid_response_and_filters_invalid_codes,
        test_classify_segments_batches_large_input,
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
