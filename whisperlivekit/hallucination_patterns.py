"""Bekende Whisper-hallucinaties (tekst die het model verzint bij stilte of muziek).

Eén gedeelde lijst voor alle paden: de incrementele batch-worker, "Ververs
Transcriptie" (via simul_whisper.backend, dat deze naam doorgeeft) en de live-
weergave (tokens_alignment.get_lines). Een tekst die één van deze patronen bevat,
is per definitie geen spraak en wordt in zijn geheel weggelaten -- per zin
(batch) of per live-regel (live), nooit per los woord.

Bewust een lichte module zonder imports: tokens_alignment en de tests laden
hem zonder de zware ASR-stack.
"""

HALLUCINATION_PATTERNS = [
    "***",
    "Ondertiteling",
    "ondertiteling",
    "Ondertitels",
    "ondertitels",
    "www.",
    ".com",
    "Abonneer",
    "abonneer",
    "Subtitles by",
    "Subscribe",
    "subscribe",
    "Altyazı",
    "altyazı",
    "Altyazi",
    "altyazi",
    # Turkse YouTube-outro-hallucinaties (waargenomen 2026-07-19 en herbevestigd
    # 2026-08-15, zie CLAUDE.md/refresh transcript-context) -- specifiek genoeg
    # om echte spraak niet te raken: "teşekkür ederim" alleen (gewoon "dankjewel")
    # zou wél te breed zijn en legitieme bedankjes afkeuren.
    "İzlediğiniz için",
    "izlediğiniz için",
    "abone olmayı",
]


def contains_hallucination(text: str) -> bool:
    return any(p in (text or "") for p in HALLUCINATION_PATTERNS)
