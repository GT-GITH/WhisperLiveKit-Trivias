"""Cross-channel akoestisch-lek onderdrukking voor "Ververs Transcriptie".

Niet-causale tegenhanger van de live cross-kanaal arbitrage in
web_trivias/app.js (computeCombinedGate(), ~regel 112): daar wordt per
binnenkomend audio-chunk het RMS-niveau van het eigen kanaal vergeleken met het
luidste andere kanaal, causaal en met een korte hold-tijd om flikkeren te
voorkomen. "Ververs Transcriptie" leest de hele WAV in één keer terug, dus kan
achteraf (niet-causaal) werken: kijk zowel vooruit als achteruit, en gebruik
audio-cross-correlatie om kanalen die onafhankelijk gestart zijn (aparte
WebSocket-verbindingen, geen gedeelde starttijd op sub-seconde-precisie) eerst
uit te lijnen vóórdat RMS-arbitrage tussen ze zinvol is.

Puur numpy/scipy, geen FastAPI/sessie-koppeling: elke functie is los te
unit-testen met synthetische signalen. Wijzigt NOOIT de input-arrays (audio is
autoritatief, zie CLAUDE.md) -- retourneert alleen boolean keep-masks; het
zero-en van samples gebeurt pas in TriviasServer.rebuild_channel_transcript(),
op een kopie, exact zoals de live gate het doet (audio_processor.py
~1908-1921).
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as scipy_ndimage
from scipy import signal as scipy_signal

logger = logging.getLogger("whisperlivekit.cross_channel_gate")

# --- Tunable constants (startpunten, deels gespiegeld van app.js) ---
DEFAULT_WINDOW_MS = 40.0             # RMS-envelope analysevenster
DEFAULT_HOP_MS = 20.0                # envelope-hop (= resolutie van uitlijning + arbitrage)
DEFAULT_MAX_LAG_S = 5.0              # begrensd lag-zoekbereik: verbindings-jitter tussen
                                      # apart gestarte WebSocket-kanalen is orde seconden,
                                      # niet minuten (geen sub-seconde starttijd wordt ooit
                                      # gepersisteerd, zie project-CLAUDE.md-onderzoek)
DEFAULT_MIN_ALIGN_CONFIDENCE = 0.3   # Pearson-r op het beste-lag-overlapvenster; eronder
                                      # wordt het kanaalpaar als onuitlijnbaar beschouwd --
                                      # fail-safe, geen onderdrukking
DEFAULT_ARBITRATION_MARGIN = 0.4     # gespiegeld van app.js ARBITRATION_MARGIN
DEFAULT_SILENCE_FLOOR = 0.005        # RMS onder deze waarde telt nooit mee als PEER in de
                                      # cross-kanaal-arbitrage hieronder (voorkomt dat een
                                      # praktisch stil kanaal toch als "luidste peer" geldt).
                                      # Dit is UITDRUKKELIJK NIET hetzelfde als app.js's
                                      # gateOpen1 (eigen-kanaal-ruisdrempel, standaard 0.015,
                                      # drie keer zo hoog) -- zie DEFAULT_OWN_GATE_THRESHOLD.
DEFAULT_OWN_GATE_THRESHOLD = 0.015   # = app.js's gateThreshold-default ("Normaal"-preset).
                                      # Niet-causale tegenhanger van app.js's PCMForwarder-
                                      # ruisdrempel (stage 1, pcm_worklet.js): puur per kanaal,
                                      # geen ander kanaal of uitlijning nodig. Dit is wat een
                                      # ver-weg-microfoon-lek in live al onderdrukt vóórdat
                                      # cross-kanaal-arbitrage (stage 2, hieronder) er ooit aan
                                      # te pas komt.
                                      # BELANGRIJK (2026-08-23, echte sessie liet dit zien):
                                      # deze waarde is een live-UX/latency-drempel ("bespaar
                                      # onnodige ASR-aanroepen tijdens streamen"), nooit
                                      # gekalibreerd als "is dit wel/geen echte spraak"-drempel.
                                      # Toegepast als vaste absolute drempel op een HELE, in
                                      # één keer gedecodeerde batch-opname hakt hij doodgewone
                                      # spraakdynamiek (pauzes, wegstervende medeklinkers) net
                                      # zo goed weg als een echt lek -- die twee zijn met een
                                      # vaste drempel niet te onderscheiden. Daarom wordt deze
                                      # drempel nu pas toegepast NADAT is vastgesteld dat een
                                      # kanaal structureel stil is (zie
                                      # DEFAULT_OWN_GATE_ACTIVITY_PERCENTILE hieronder en
                                      # _channel_has_active_content()) -- nooit meer blind op
                                      # een kanaal met aantoonbaar actieve inhoud.
DEFAULT_OWN_GATE_ACTIVITY_PERCENTILE = 90.0
                                      # Kanaalniveau-vooraf-check: als het p90 van de eigen
                                      # RMS-envelope al boven DEFAULT_SILENCE_FLOOR ligt, heeft
                                      # dit kanaal aantoonbaar actieve inhoud (normale spraak
                                      # heeft altijd aanzienlijke dynamiek tussen klinkers/
                                      # medeklinkers) -- dan slaat OWNGATE (de per-frame
                                      # DEFAULT_OWN_GATE_THRESHOLD hierboven) voor dit hele
                                      # kanaal helemaal over. Een structureel stil/lekkend
                                      # kanaal (het scenario dat OWNGATE moet vangen, zie
                                      # tests/test_cross_channel_gate.py) heeft juist een vlakke,
                                      # laag-blijvende envelope: p90 blijft dan onder de vloer,
                                      # dus wordt nog steeds (frame-niveau) onderdrukt.
DEFAULT_MIN_SUPPRESS_RUN_MS = 150.0  # niet-causale tegenhanger van CLOSE_HOLD_MS (100ms);
                                      # iets ruimer omdat offline smoothing minder risico op
                                      # valse positieven heeft dan een causale hold
DEFAULT_BRIDGE_GAP_MS = 100.0        # overbrug korte "open"-gaten tussen twee onderdrukte
                                      # runs, om snel wisselen bij een wegstervende zin te
                                      # voorkomen
DEFAULT_DETREND_WINDOW_S = 1.5       # zie estimate_alignment(): een gesprek met beurtwisseling
                                      # (spreker A even dominant, dan spreker B) geeft op
                                      # RUWE RMS-envelopes een sterke NEGATIEVE globale
                                      # correlatie (luid/stil wisselt tegengesteld), die de
                                      # veel zwakkere, maar wel bruikbare, fijnmazige
                                      # lek-correlatie compleet overstemt. Een lokaal
                                      # voortschrijdend gemiddelde aftrekken vóór correlatie
                                      # verwijdert die trage "wie spreekt er nu"-trend en laat
                                      # de snellere mede-modulatie staan die de akoestische
                                      # pad-vertraging (en dus de uitlijning) echt verraadt.


def compute_rms_envelope(
    audio_f32: np.ndarray,
    sample_rate: int,
    window_ms: float = DEFAULT_WINDOW_MS,
    hop_ms: float = DEFAULT_HOP_MS,
) -> Tuple[np.ndarray, int]:
    """Per-frame RMS via een gestrided (zero-copy) venster.

    window_ms=40/hop_ms=20 (50% overlap) vangt foneemschaal-dynamiek (fonemen
    zijn doorgaans 50-200ms) terwijl een uur 16kHz-audio (57,6M samples)
    gecomprimeerd wordt naar ~180k frames -- goedkoop voor zowel correlatie
    als morfologische smoothing.

    Retourneert (envelope, hop_samples).
    """
    window_samples = max(1, int(round(window_ms / 1000.0 * sample_rate)))
    hop_samples = max(1, int(round(hop_ms / 1000.0 * sample_rate)))
    n = len(audio_f32)

    if n == 0:
        return np.zeros(0, dtype=np.float64), hop_samples
    if n < window_samples:
        rms = float(np.sqrt(np.mean(audio_f32.astype(np.float64) ** 2)))
        return np.array([rms], dtype=np.float64), hop_samples

    windows = np.lib.stride_tricks.sliding_window_view(audio_f32, window_samples)
    strided = windows[::hop_samples]
    envelope = np.sqrt(np.mean(strided.astype(np.float64) ** 2, axis=1))
    return envelope, hop_samples


def _channel_has_active_content(
    envelope: np.ndarray,
    silence_floor: float = DEFAULT_SILENCE_FLOOR,
    percentile: float = DEFAULT_OWN_GATE_ACTIVITY_PERCENTILE,
) -> bool:
    """Kanaalniveau-vooraf-check, zie DEFAULT_OWN_GATE_ACTIVITY_PERCENTILE hierboven:
    is het p90 van de eigen envelope al boven de stilte-vloer, dan heeft dit kanaal
    aantoonbaar actieve inhoud (spraak heeft altijd dynamiek), en mag de per-frame
    OWNGATE-drempel (DEFAULT_OWN_GATE_THRESHOLD, gekalibreerd voor iets heel anders --
    zie de comment daar) 'm niet aan flarden knippen. Een leeg envelope-array (lege
    audio) telt als geen actieve inhoud -- fail-safe naar de bestaande OWNGATE-weg."""
    if len(envelope) == 0:
        return False
    return float(np.percentile(envelope, percentile)) >= silence_floor


def _expand_frames_to_samples(keep_frames: np.ndarray, hop_samples: int, total_len: int) -> np.ndarray:
    """Frame-niveau boolean-array (compute_rms_envelope-hop-resolutie) uitvouwen
    naar sample-niveau, bijgeknipt/opgevuld (met True/open, fail-safe) op de
    exacte originele array-lengte."""
    keep_samples = np.repeat(keep_frames, hop_samples)
    if len(keep_samples) < total_len:
        pad = np.ones(total_len - len(keep_samples), dtype=bool)
        keep_samples = np.concatenate([keep_samples, pad])
    else:
        keep_samples = keep_samples[:total_len]
    return keep_samples


def compute_own_channel_gate_mask(
    audio_f32: np.ndarray,
    sample_rate: int,
    threshold: float = DEFAULT_OWN_GATE_THRESHOLD,
    window_ms: float = DEFAULT_WINDOW_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    min_suppress_run_ms: float = DEFAULT_MIN_SUPPRESS_RUN_MS,
    bridge_gap_ms: float = DEFAULT_BRIDGE_GAP_MS,
) -> np.ndarray:
    """Niet-causale tegenhanger van app.js's PCMForwarder-ruisdrempel (stage 1,
    pcm_worklet.js `_gateOpen`/`gateOpen1`): PUUR per kanaal, geen ander kanaal
    of uitlijning nodig. Frames waarvan de eigen RMS onder `threshold` zit
    worden onderdrukt, ongeacht wat er op een ander kanaal gebeurt -- dit is
    wat een ver-weg-microfoon-lek in live al wegfiltert vóórdat cross-kanaal-
    arbitrage (arbitrate(), stage 2) er ooit aan te pas komt. Zelfde opening/
    closing-smoothing als arbitrate() (isolated blips weg, korte gaten
    overbrugd), niet-causaal dus symmetrisch i.p.v. live's causale hold.

    Retourneert een boolean keep-mask, exact zo lang als audio_f32."""
    envelope, hop_samples = compute_rms_envelope(audio_f32, sample_rate, window_ms, hop_ms)
    if len(envelope) == 0:
        return np.ones(len(audio_f32), dtype=bool)

    suppress_frames = envelope < threshold
    if suppress_frames.any() and not suppress_frames.all():
        ms_per_frame = hop_samples / sample_rate * 1000.0
        min_run_frames = max(1, int(round(min_suppress_run_ms / ms_per_frame)))
        bridge_frames = max(1, int(round(bridge_gap_ms / ms_per_frame)))
        suppress_frames = scipy_ndimage.binary_opening(
            suppress_frames, structure=np.ones(min_run_frames, dtype=bool)
        )
        suppress_frames = scipy_ndimage.binary_closing(
            suppress_frames, structure=np.ones(bridge_frames, dtype=bool)
        )

    keep_frames = ~suppress_frames
    return _expand_frames_to_samples(keep_frames, hop_samples, len(audio_f32))


def _detrend_envelope(env: np.ndarray, window_frames: int) -> np.ndarray:
    """Trek een lokaal voortschrijdend gemiddelde af (reflect-padded) om trage
    niveauveranderingen (bv. beurtwisseling tussen sprekers) te verwijderen,
    zodat cross-correlatie zich richt op de snellere mede-modulatie die een
    akoestisch lek daadwerkelijk verraadt. Alleen gebruikt voor uitlijning +
    confidence -- arbitrate() werkt bewust op de RUWE envelope, want daar is
    "wie is nu luider" precies het signaal dat we willen."""
    if len(env) < 3:
        return env - (env.mean() if len(env) else 0.0)
    window_frames = max(3, window_frames)
    if window_frames % 2 == 0:
        window_frames += 1
    moving_avg = scipy_ndimage.uniform_filter1d(env, size=window_frames, mode="reflect")
    return env - moving_avg


def estimate_alignment(
    ref_env: np.ndarray,
    other_env: np.ndarray,
    hop_samples: int,
    sample_rate: int,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
    detrend_window_s: float = DEFAULT_DETREND_WINDOW_S,
) -> Tuple[int, float]:
    """Cross-correleer twee envelopes en vind de beste lag (in frames).

    Conventie (empirisch geverifieerd met synthetisch-verschoven signalen --
    zie tests/test_cross_channel_gate.py -- verander dit NIET zonder die tests
    opnieuw te bevestigen): met
        corr = scipy.signal.correlate(ref_env, other_env, mode="full")
        lags = arange(-(len(other_env) - 1), len(ref_env))
    geeft lags[argmax(corr)] de lag L waarvoor geldt: other_env[k] hoort thuis
    op de gedeelde tijdlijn-positie (k + L). Met andere woorden: L is direct de
    sample/frame-offset die bij other_env opgeteld moet worden om 'm op ref_env's
    tijdlijn te plaatsen.

    scipy.signal.correlate(..., method="fft") is O(N log N) -- ook voor een
    envelope van een meerdere-uren-sessie (~10^5-10^6 frames) ruim binnen een
    seconde, waar een volledige-resolutie 16kHz-correlatie (~100-1000x meer
    samples) onhaalbaar traag zou zijn.

    Confidence: Pearson-r op het daadwerkelijk overlappende venster bij de
    beste lag (niet de rauwe, ongenormaliseerde correlatiepiek) -- begrensd en
    interpreteerbaar in [-1, 1]. Retourneert expliciet confidence=0.0 (nooit
    NaN) bij een leeg/te klein overlapvenster of een vlak (nul-variantie)
    signaal aan een van beide kanten.

    Het overlapvenster wordt waar mogelijk een halve detrend-vensterbreedte
    ingekort aan weerszijden, gemeten vanaf de ECHTE randen van ref_env EN
    other_env afzonderlijk (niet zomaar de randen van het overlapvenster
    zelf): _detrend_envelope()'s reflect-padding geeft bij een array-rand een
    kunstmatig artefact, en zodra één kanaal het andere pas laat begint te
    overlappen (het normale geval bij een echt tijdsverschil) valt zo'n rand
    van het ene kanaal middenin het overlapvenster, terwijl de overeenkomstige
    positie van het ANDERE kanaal daar helemaal geen rand heeft -- die
    asymmetrie corrumpeert anders de confidence-score ook al is de gevonden
    lag zelf wel exact correct (empirisch gevonden via
    tests/test_cross_channel_gate.py, niet louter theoretisch).
    """
    if len(ref_env) == 0 or len(other_env) == 0:
        return 0, 0.0

    ms_per_frame = hop_samples / sample_rate * 1000.0
    detrend_frames = max(3, int(round(detrend_window_s * 1000.0 / ms_per_frame)))
    ref_dt = _detrend_envelope(ref_env, detrend_frames)
    other_dt = _detrend_envelope(other_env, detrend_frames)

    corr = scipy_signal.correlate(ref_dt, other_dt, mode="full", method="fft")
    lags = np.arange(-(len(other_dt) - 1), len(ref_dt))

    max_lag_frames = max(1, int(round(max_lag_s * sample_rate / hop_samples)))
    valid = np.abs(lags) <= max_lag_frames
    if not valid.any():
        valid = np.ones_like(lags, dtype=bool)

    candidate_lags = lags[valid]
    candidate_corr = corr[valid]
    best_lag = int(candidate_lags[int(np.argmax(candidate_corr))])

    lo = max(0, best_lag)
    hi = min(len(ref_dt), len(other_dt) + best_lag)
    if hi - lo < 2:
        return best_lag, 0.0

    edge_margin = detrend_frames // 2
    # lo2/hi2 blijven >= edge_margin verwijderd van ZOWEL ref_dt[0]/[-1] ALS
    # other_dt[0]/[-1] (other-index = positie - best_lag).
    lo2 = max(lo, edge_margin, best_lag + edge_margin)
    hi2 = min(hi, len(ref_dt) - edge_margin, len(other_dt) + best_lag - edge_margin)
    if hi2 - lo2 >= 2:
        lo, hi = lo2, hi2
    # anders: overlap te kort om ook nog marge te nemen -- beste beschikbare
    # (mogelijk licht rand-vertekende) schatting gebruiken i.p.v. hem weg te gooien.

    ref_slice = ref_dt[lo:hi]
    other_slice = other_dt[lo - best_lag : hi - best_lag]
    if ref_slice.std() == 0.0 or other_slice.std() == 0.0:
        return best_lag, 0.0

    confidence = float(np.corrcoef(ref_slice, other_slice)[0, 1])
    if np.isnan(confidence):
        confidence = 0.0
    return best_lag, confidence


def align_all_channels(
    envelopes: Dict[str, np.ndarray],
    hop_samples: int,
    sample_rate: int,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
    min_confidence: float = DEFAULT_MIN_ALIGN_CONFIDENCE,
    session_id: str = "",
) -> Tuple[str, Dict[str, Optional[int]], Dict[str, float]]:
    """Kies één referentiekanaal (langste audio, ties op channel_id) en lijn
    elk ANDER kanaal daarop uit (O(n), niet O(n^2) paarsgewijs) -- voldoende
    omdat het doel één gedeelde tijdlijn is voor n-weg-arbitrage, niet
    paarsgewijze lek-detectie; als het lek echt is moet het correleren tegen
    elk kanaal dat op hetzelfde moment actief was, inclusief de referentie.

    Retourneert (ref_channel_id, {channel_id: lag_frames_of_None}, {channel_id: confidence}).
    lag_frames is None voor het referentiekanaal zelf (impliciet 0, wel als 0
    in de offsets-dict gezet) en voor elk kanaal met confidence < min_confidence
    (fail-safe: uitgesloten van de gedeelde tijdlijn, gelogd, nooit een gegokte
    offset toegekend).
    """
    channel_ids = list(envelopes.keys())
    ref_channel = sorted(channel_ids, key=lambda ch: (-len(envelopes[ch]), ch))[0]

    offsets: Dict[str, Optional[int]] = {ref_channel: 0}
    confidences: Dict[str, float] = {ref_channel: 1.0}
    ref_env = envelopes[ref_channel]

    for ch in channel_ids:
        if ch == ref_channel:
            continue
        lag, confidence = estimate_alignment(
            ref_env, envelopes[ch], hop_samples, sample_rate, max_lag_s
        )
        confidences[ch] = confidence
        if confidence >= min_confidence:
            offsets[ch] = lag
            lag_ms = lag * hop_samples / sample_rate * 1000.0
            logger.info(
                f"[REFRESH][XGATE][ALIGN] session={session_id} ref_channel={ref_channel} "
                f"channel={ch} lag_ms={lag_ms:.1f} confidence={confidence:.3f} accepted=True"
            )
        else:
            offsets[ch] = None
            # lag_ms hier ook loggen (ook al wordt 'm niet gebruikt): laat zien of de
            # afgewezen piek een plausibele verbindings-jitter-waarde was (paar honderd
            # ms) of een verdachte/willekeurige uitschieter -- nodig om bij een volgende
            # test te kunnen beoordelen of min_confidence simpelweg te streng staat voor
            # een zwak/deels lek, of dat de gevonden piek zelf ruis is.
            lag_ms = lag * hop_samples / sample_rate * 1000.0
            logger.info(
                f"[REFRESH][XGATE][ALIGN][SKIP] session={session_id} channel={ch} "
                f"confidence={confidence:.3f} < min={min_confidence} best_lag_ms={lag_ms:.1f} "
                f"-- alignment onbetrouwbaar, kanaal uitgesloten van cross-channel arbitrage "
                f"(fail-safe: geen onderdrukking)"
            )

    return ref_channel, offsets, confidences


def _count_true_runs(bool_array: np.ndarray) -> int:
    if bool_array.size == 0:
        return 0
    padded = np.concatenate(([False], bool_array, [False]))
    diffs = np.diff(padded.astype(np.int8))
    return int(np.sum(diffs == 1))


def arbitrate(
    envelopes: Dict[str, np.ndarray],
    offsets: Dict[str, int],
    hop_samples: int,
    sample_rate: int,
    arbitration_margin: float = DEFAULT_ARBITRATION_MARGIN,
    min_suppress_run_ms: float = DEFAULT_MIN_SUPPRESS_RUN_MS,
    bridge_gap_ms: float = DEFAULT_BRIDGE_GAP_MS,
    silence_floor: float = DEFAULT_SILENCE_FLOOR,
    session_id: str = "",
) -> Dict[str, np.ndarray]:
    """Arbitreer tussen alle (reeds succesvol uitgelijnde) kanalen op een
    gedeelde frame-tijdlijn, en geef per kanaal een boolean suppress-array terug
    IN DAT KANAAL'S EIGEN LOKALE FRAME-INDEXERING (even lang als
    envelopes[channel_id]) -- de aanroeper hoeft dus niet zelf terug te mappen.

    Regel per gedeeld frame: suspected = eigen_rms < max(peer_rms) * margin,
    waarbij een peer alleen meetelt als die op dat frame aanwezig is EN boven
    silence_floor zit (analoog aan app.js's peer.gateOpen1-check) -- en waarbij
    frames onder silence_floor voor het EIGEN kanaal nooit als verdacht gelden
    (er is dan toch niets zinvols om te onderdrukken).

    Niet-causale smoothing (kan, anders dan de live causale hold, zowel vooruit
    als achteruit kijken): binary_opening wist geïsoleerde verdachte runs
    korter dan min_suppress_run_ms, binary_closing overbrugt open-gaten korter
    dan bridge_gap_ms tussen twee onderdrukte runs.

    Veiligheidsnet (offline verbetering t.o.v. de live versie, die alleen
    zichzelf kan forceren omdat elk kanaal daar alleen zijn eigen verdict ziet):
    als op een gedeeld frame ALLE aanwezige kanalen verdacht zijn, wordt het
    daadwerkelijk luidste kanaal op dat frame geforceerd open gezet.
    """
    channels = list(envelopes.keys())
    if len(channels) <= 1:
        return {ch: np.zeros(len(envelopes[ch]), dtype=bool) for ch in channels}

    min_off = min(offsets[ch] for ch in channels)
    shift = -min_off
    placed_start = {ch: offsets[ch] + shift for ch in channels}
    total_len = max(placed_start[ch] + len(envelopes[ch]) for ch in channels)

    rms_grid = np.zeros((len(channels), total_len), dtype=np.float64)
    present_grid = np.zeros((len(channels), total_len), dtype=bool)
    for i, ch in enumerate(channels):
        s = placed_start[ch]
        e = s + len(envelopes[ch])
        rms_grid[i, s:e] = envelopes[ch]
        present_grid[i, s:e] = True

    eligible_peer = present_grid & (rms_grid >= silence_floor)
    masked_rms = np.where(eligible_peer, rms_grid, 0.0)

    suspected = np.zeros((len(channels), total_len), dtype=bool)
    for i, ch in enumerate(channels):
        others_rms = np.delete(masked_rms, i, axis=0)
        max_peer = others_rms.max(axis=0) if others_rms.shape[0] > 0 else np.zeros(total_len)
        own_rms = rms_grid[i]
        own_present = present_grid[i]
        suspected[i] = (
            own_present
            & (own_rms >= silence_floor)
            & (max_peer > 0)
            & (own_rms < max_peer * arbitration_margin)
        )

    # Met de max-of-peers-regel hierboven is dit wiskundig onbereikbaar: het
    # (globaal) luidste aanwezige kanaal op een frame heeft per definitie
    # max_peer <= eigen_rms, dus eigen_rms < max_peer*margin (margin<1) kan voor
    # dat kanaal nooit waar zijn -- er is dus altijd minstens één niet-verdacht
    # kanaal zolang er >=1 kanaal boven silence_floor zit. Dat is een bewuste,
    # structurele verbetering t.o.v. de live/causale versie (die WEL af en toe
    # alle kanalen tegelijk kon sluiten door verouderde per-kanaal snapshots,
    # zie app.js-commentaar) -- hier is er één consistente, gelijktijdige
    # rms_grid, geen staleness mogelijk. Onderstaande blok blijft toch staan als
    # verdediging-in-diepte: als de arbitrageregel later verandert (bv. naar
    # gemiddelde-van-peers i.p.v. max-van-peers), is deze garantie niet meer
    # automatisch en wordt dit blok wél bereikbaar.
    present_count = present_grid.sum(axis=0)
    suspected_count = (suspected & present_grid).sum(axis=0)
    all_suspected = (present_count > 0) & (suspected_count == present_count)
    n_safety_frames = int(all_suspected.sum())
    if n_safety_frames > 0:
        rms_for_loudest = np.where(present_grid, rms_grid, -1.0)
        loudest_idx = np.argmax(rms_for_loudest, axis=0)
        frame_indices = np.where(all_suspected)[0]
        suspected[loudest_idx[frame_indices], frame_indices] = False
        t0 = frame_indices.min() * hop_samples / sample_rate
        t1 = frame_indices.max() * hop_samples / sample_rate
        logger.info(
            f"[REFRESH][XGATE][SAFETYNET] session={session_id} {n_safety_frames} frame(s) "
            f"tussen {t0:.2f}s-{t1:.2f}s hadden alle aanwezige kanalen verdacht -- "
            f"luidste kanaal per frame geforceerd open"
        )

    ms_per_frame = hop_samples / sample_rate * 1000.0
    min_run_frames = max(1, int(round(min_suppress_run_ms / ms_per_frame)))
    bridge_frames = max(1, int(round(bridge_gap_ms / ms_per_frame)))

    result: Dict[str, np.ndarray] = {}
    for i, ch in enumerate(channels):
        row = suspected[i]
        if row.any():
            row = scipy_ndimage.binary_opening(row, structure=np.ones(min_run_frames, dtype=bool))
            row = scipy_ndimage.binary_closing(row, structure=np.ones(bridge_frames, dtype=bool))
        s = placed_start[ch]
        e = s + len(envelopes[ch])
        result[ch] = row[s:e]
    return result


def compute_cross_channel_gate_masks(
    audio_by_channel: Dict[str, np.ndarray],
    sample_rate: int = 16000,
    *,
    window_ms: float = DEFAULT_WINDOW_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
    min_align_confidence: float = DEFAULT_MIN_ALIGN_CONFIDENCE,
    arbitration_margin: float = DEFAULT_ARBITRATION_MARGIN,
    min_suppress_run_ms: float = DEFAULT_MIN_SUPPRESS_RUN_MS,
    bridge_gap_ms: float = DEFAULT_BRIDGE_GAP_MS,
    silence_floor: float = DEFAULT_SILENCE_FLOOR,
    own_gate_threshold: float = DEFAULT_OWN_GATE_THRESHOLD,
    session_id: str = "",
) -> Dict[str, np.ndarray]:
    """Enige entrypoint die TriviasServer.py aanroept.

    Input: {channel_id: float32 PCM-array}. Arrays mogen verschillende lengtes
    hebben -- nooit gelijke lengte aannemen, kanalen kunnen op verschillende
    momenten starten/stoppen. Wijzigt NOOIT de input-arrays.

    Output: {channel_id: boolean keep-mask}, elk EXACT zo lang als het
    bijbehorende input-array (True = naar ASR voeden, False = onderdrukken).

    Twee lagen, gecombineerd met AND (een sample blijft alleen open als BEIDE
    lagen 'm open laten) -- zelfde tweelaags-ontwerp als de live gate (app.js):
    1. **Eigen-kanaal-ruisdrempel** (compute_own_channel_gate_mask, stage 1):
       per kanaal, geen uitlijning nodig -- maar ALLEEN toegepast op een kanaal
       dat kanaalniveau al structureel stil blijkt (zie _channel_has_active_content()
       en DEFAULT_OWN_GATE_ACTIVITY_PERCENTILE hierboven): vangt een zwak/
       ver-weg-microfoon-lek dat gewoon te stil is om nuttig te zijn, zonder
       een kanaal met aantoonbaar actieve spraak aan flarden te knippen.
    2. **Cross-kanaal-arbitrage** (align_all_channels + arbitrate, stage 2):
       alleen bij 2+ kanalen EN een betrouwbare uitlijning (anders fail-safe:
       geen onderdrukking op deze laag i.p.v. het risico van misalignering en
       het wegvagen van echte spraak).

    len(audio_by_channel) <= 1 past nog wel stage 1 toe (net als live, dat ook
    zonder ander kanaal een eigen ruisdrempel hanteert), slaat alleen stage 2 over.
    """
    channel_ids = list(audio_by_channel.keys())

    # Envelope per kanaal vooraf berekenen (hergebruikt hieronder zowel voor de
    # kanaalniveau-activiteitscheck als -- bij 2+ kanalen -- voor stage 2, i.p.v.
    # 'm twee keer te berekenen).
    envelopes: Dict[str, np.ndarray] = {}
    hop_samples = None
    for ch, audio in audio_by_channel.items():
        envelopes[ch], hop_samples = compute_rms_envelope(audio, sample_rate, window_ms, hop_ms)

    own_gate_masks: Dict[str, np.ndarray] = {}
    for ch, audio in audio_by_channel.items():
        if _channel_has_active_content(envelopes[ch], silence_floor):
            own_gate_masks[ch] = np.ones(len(audio), dtype=bool)
            logger.info(
                f"[REFRESH][XGATE][OWNGATE] session={session_id} channel={ch} "
                f"aantoonbaar actieve inhoud (p{DEFAULT_OWN_GATE_ACTIVITY_PERCENTILE:.0f} "
                f"envelope >= {silence_floor}) -- eigen-ruisdrempel overgeslagen"
            )
            continue
        own_gate_masks[ch] = compute_own_channel_gate_mask(
            audio, sample_rate, own_gate_threshold, window_ms, hop_ms,
            min_suppress_run_ms, bridge_gap_ms,
        )
        n_own_suppressed = int((~own_gate_masks[ch]).sum())
        if n_own_suppressed > 0:
            pct = 100.0 * n_own_suppressed / len(audio)
            logger.info(
                f"[REFRESH][XGATE][OWNGATE] session={session_id} channel={ch} suppressed "
                f"{n_own_suppressed}/{len(audio)} samples ({pct:.1f}%) onder eigen "
                f"ruisdrempel {own_gate_threshold}"
            )

    if len(channel_ids) <= 1:
        logger.info(
            f"[REFRESH][XGATE] session={session_id} {len(channel_ids)} kanaal/kanalen, "
            f"cross-channel arbitrage (stage 2) overgeslagen -- alleen eigen-ruisdrempel toegepast"
        )
        return own_gate_masks

    ref_channel, offsets, confidences = align_all_channels(
        envelopes, hop_samples, sample_rate, max_lag_s, min_align_confidence, session_id
    )

    included_offsets: Dict[str, int] = {ch: off for ch, off in offsets.items() if off is not None}
    excluded = [ch for ch in channel_ids if ch not in included_offsets]

    cross_gate_masks: Dict[str, np.ndarray] = {
        ch: np.ones(len(audio_by_channel[ch]), dtype=bool) for ch in excluded
    }

    if len(included_offsets) > 1:
        included_envelopes = {ch: envelopes[ch] for ch in included_offsets}
        suppress_by_channel = arbitrate(
            included_envelopes,
            included_offsets,
            hop_samples,
            sample_rate,
            arbitration_margin,
            min_suppress_run_ms,
            bridge_gap_ms,
            silence_floor,
            session_id,
        )
        for ch in included_offsets:
            keep_frames = ~suppress_by_channel[ch]
            cross_gate_masks[ch] = _expand_frames_to_samples(
                keep_frames, hop_samples, len(audio_by_channel[ch])
            )
    else:
        # Alleen het referentiekanaal (of niemand anders) kon uitgelijnd worden --
        # geen peer om tegen te arbitreren, dus geen onderdrukking op deze laag.
        for ch in included_offsets:
            cross_gate_masks.setdefault(ch, np.ones(len(audio_by_channel[ch]), dtype=bool))

    final_masks: Dict[str, np.ndarray] = {}
    for ch in channel_ids:
        final_masks[ch] = own_gate_masks[ch] & cross_gate_masks[ch]
        total_len = len(audio_by_channel[ch])
        n_suppressed = int((~final_masks[ch]).sum())
        if n_suppressed > 0:
            n_runs = _count_true_runs(~final_masks[ch])
            pct = 100.0 * n_suppressed / total_len
            logger.info(
                f"[REFRESH][XGATE] session={session_id} channel={ch} totaal onderdrukt "
                f"{n_suppressed}/{total_len} samples ({pct:.1f}%) across {n_runs} run(s) "
                f"(eigen-ruisdrempel + cross-kanaal-arbitrage gecombineerd)"
            )

    return final_masks
