"""Scoring voor de end-to-end-testsuite (laag 1).

Puur Python, geen dependencies (geen jiwer/numpy) -- zodat dit lokaal én op de
pod draait en los te unit-testen is (zie tests/e2e/test_scoring.py).

Wat er gemeten wordt, per kanaal en per transcriptpad:
- WER / CER: corpus-niveau (alle referentie-uitingen achter elkaar vs alle
  hypothese-segmenten op starttijd gesorteerd).
- missed: referentie-uitingen zonder enig overlappend segment (gemiste spraak).
- spurious: segmenten die met geen enkele referentie-uiting overlappen --
  tekst in stilte, dus hallucinatie (of lek van een ander kanaal).
- ts_*_ms: afwijking tussen segment-start en de start van de dichtstbijzijnde
  overlappende referentie-uiting. Tijdstempels zijn de primaire sleutel van dit
  project (CLAUDE.md), dus drift hier is een echte fout, geen cosmetiek.
- live_fallback: opgeslagen segmenten zonder batch-tekst (text_batch leeg).
  Die zijn niet citeerbaar (CLAUDE.md: alleen batch is citeerbaar).
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from typing import Dict, List, Optional, Sequence

OVERLAP_TOLERANCE_MS = 500


def normalize(text: str) -> str:
    """NFKC + lowercase + interpunctie weg + witruimte samenvouwen.

    Bewust geen taalspecifieke normalisatie (getallen uitschrijven, Arabische
    alef/hamza-varianten, ...): die verbetert de score kunstmatig en maakt
    runs minder vergelijkbaar. Zelfde keuze als scripts/eval_arabic_wer_casablanca.py.
    """
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Levenshtein-afstand (substitutie/invoeging/weglating kosten elk 1)."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(ref_text: str, hyp_text: str) -> Optional[float]:
    ref = normalize(ref_text).split()
    if not ref:
        return None
    return edit_distance(ref, normalize(hyp_text).split()) / len(ref)


def cer(ref_text: str, hyp_text: str) -> Optional[float]:
    ref = normalize(ref_text).replace(" ", "")
    if not ref:
        return None
    return edit_distance(ref, normalize(hyp_text).replace(" ", "")) / len(ref)


def segment_text(seg: dict) -> str:
    """Citeerbare tekst van een opgeslagen segment: batch indien aanwezig,
    anders wat er als final staat (live-fallback)."""
    return (seg.get("text_batch") or seg.get("text_final") or seg.get("text") or "").strip()


def is_live_fallback(seg: dict) -> bool:
    return not (seg.get("text_batch") or "").strip() and bool(segment_text(seg))


def _overlaps(seg: dict, utt: dict, tol: int = OVERLAP_TOLERANCE_MS) -> bool:
    s, e = seg.get("start_ms") or 0, seg.get("end_ms") or 0
    return s < utt["end_ms"] + tol and e > utt["start_ms"] - tol


MIN_COVERAGE = 0.3


def _coverage(utt: dict, segs: List[dict]) -> float:
    """Fractie van de uiting die door (de vereniging van) segmenten gedekt wordt."""
    dur = utt["end_ms"] - utt["start_ms"]
    if dur <= 0:
        return 1.0
    spans = sorted(
        (max(utt["start_ms"], s.get("start_ms") or 0), min(utt["end_ms"], s.get("end_ms") or 0)) for s in segs
    )
    covered, cur_s, cur_e = 0, None, None
    for a, b in spans:
        if b <= a:
            continue
        if cur_e is None or a > cur_e:
            if cur_e is not None:
                covered += cur_e - cur_s
            cur_s, cur_e = a, b
        else:
            cur_e = max(cur_e, b)
    if cur_e is not None:
        covered += cur_e - cur_s
    return covered / dur


def score_channel(utterances: List[dict], segments: List[dict]) -> Dict[str, object]:
    """utterances: [{start_ms, end_ms, text}] (referentie, op tijd gesorteerd).
    segments: opgeslagen transcriptsegmenten van de server voor dit kanaal."""
    segs = sorted((s for s in segments if segment_text(s)), key=lambda s: s.get("start_ms") or 0)
    utts = sorted(utterances, key=lambda u: u["start_ms"])

    ref_text = " ".join(u["text"] for u in utts)
    hyp_text = " ".join(segment_text(s) for s in segs)

    # Gemist = minder dan MIN_COVERAGE van de uiting wordt echt door segmenten
    # gedekt (zonder tolerantiemarge). Alleen "raakt binnen 500 ms" was te
    # mild: een volledig weggevallen Turkse uiting telde zo als gedekt omdat het
    # segment van de VOLGENDE uiting de marge net raakte (run 2026-11-15).
    missed = [i for i, u in enumerate(utts) if _coverage(u, segs) < MIN_COVERAGE]
    spurious = [s for s in segs if not any(_overlaps(s, u) for u in utts)]

    # Per referentie-uiting: klopt de tijdstempel van het BEGIN van die uiting?
    # = afstand tussen utt.start en de vroegste start van de segmenten die ermee
    # overlappen. Niet per segment meten: Whisper knipt een lange uiting vaak in
    # meerdere segmenten, en dan ligt de start van segment 2 terecht midden in de
    # uiting -- dat is geen tijdstempelfout (eerste run 2026-11-15 gaf zo een
    # vals ts_max van 4.9-7.3 s).
    offsets = []
    for u in utts:
        starts = [s.get("start_ms") or 0 for s in segs if _overlaps(s, u)]
        if starts:
            offsets.append(abs(min(starts) - u["start_ms"]))

    return {
        "wer": wer(ref_text, hyp_text),
        "cer": cer(ref_text, hyp_text),
        "n_ref_utterances": len(utts),
        "n_ref_words": len(normalize(ref_text).split()),
        "n_segments": len(segs),
        "missed": len(missed),
        "missed_utterances": [utts[i]["text"][:80] for i in missed],
        "spurious": len(spurious),
        "spurious_segments": [
            {"start_ms": s.get("start_ms"), "end_ms": s.get("end_ms"), "text": segment_text(s)[:120]}
            for s in spurious
        ],
        "ts_median_ms": int(statistics.median(offsets)) if offsets else None,
        "ts_max_ms": max(offsets) if offsets else None,
        "live_fallback": sum(1 for s in segs if is_live_fallback(s)),
        "hyp_text": hyp_text,
        "ref_text": ref_text,
    }
