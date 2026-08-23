"""Synthetische-signaal-tests voor whisperlivekit/cross_channel_gate.py.

Geen testsuite/pytest-dependency in dit project (zie CLAUDE.md) -- draai dit
bestand rechtstreeks: `python tests/test_cross_channel_gate.py`. Focus ligt op
het meest risicovolle onderdeel van deze feature (zie plan): een omgekeerd
teken in de lag-conventie faalt niet luid, het misaligneert stilletjes en
onderdrukt dan echte spraak. Dat wordt hier vastgezet vóór productiegebruik.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

# cross_channel_gate.py is verder onafhankelijk van de rest van het pakket (geen
# torch/librosa/etc nodig) -- rechtstreeks via bestandspad laden i.p.v. via
# `whisperlivekit.cross_channel_gate`, want dat laatste triggert
# whisperlivekit/__init__.py, dat de hele (zware) ASR-stack importeert.
_MODULE_PATH = Path(__file__).resolve().parent.parent / "whisperlivekit" / "cross_channel_gate.py"
_spec = importlib.util.spec_from_file_location("cross_channel_gate", _MODULE_PATH)
_ccg = importlib.util.module_from_spec(_spec)
sys.modules["cross_channel_gate"] = _ccg
_spec.loader.exec_module(_ccg)

align_all_channels = _ccg.align_all_channels
arbitrate = _ccg.arbitrate
compute_cross_channel_gate_masks = _ccg.compute_cross_channel_gate_masks
compute_own_channel_gate_mask = _ccg.compute_own_channel_gate_mask
compute_rms_envelope = _ccg.compute_rms_envelope
estimate_alignment = _ccg.estimate_alignment

SAMPLE_RATE = 16000


def _make_tone_burst(rng, n_samples, amplitude=0.2):
    """Witte ruis geschaald naar amplitude -- genoeg envelope-structuur om op
    te correleren, in tegenstelling tot een pure sinus (die op elke integer
    periode-veelvoud even goed correleert en dus een dubbelzinnige lag geeft)."""
    return (rng.standard_normal(n_samples) * amplitude).astype(np.float32)


def test_lag_negative_when_other_is_delayed_copy():
    """other[k] = ref[k - shift] (other bevat ref's inhoud, verschoven naar
    LATERE indices -- other's eigen frame 0..shift is stilte/opvulling).
    Geverifieerde conventie (zie estimate_alignment-docstring): dit moet een
    NEGATIEVE lag opleveren, want other[k] hoort thuis op ref-tijdlijnpositie
    (k + lag) = (k - shift), dus lag = -shift."""
    rng = np.random.default_rng(42)
    ref = _make_tone_burst(rng, SAMPLE_RATE * 20)  # 20s

    ref_env, hop_samples = compute_rms_envelope(ref, SAMPLE_RATE)
    shift_samples = hop_samples * 150  # exact hop-veelvoud -> geen sub-frame-vervaging

    other = np.zeros_like(ref)
    other[shift_samples:] = ref[: len(ref) - shift_samples]
    other_env, _ = compute_rms_envelope(other, SAMPLE_RATE)

    lag_frames, confidence = estimate_alignment(ref_env, other_env, hop_samples, SAMPLE_RATE)

    expected_lag_frames = -(shift_samples // hop_samples)
    assert lag_frames < 0, f"verwacht negatieve lag, kreeg {lag_frames}"
    assert lag_frames == expected_lag_frames, (
        f"lag {lag_frames} frames != verwachte {expected_lag_frames} frames (hop-uitgelijnde shift, exact verwacht)"
    )
    assert confidence > 0.9, f"confidence {confidence:.3f} te laag voor een exacte kopie-met-shift"
    print(f"OK test_lag_negative_when_other_is_delayed_copy (lag={lag_frames} frames, confidence={confidence:.3f})")


def test_lag_positive_when_other_is_truncated_copy():
    """other = ref[shift:] (other mist ref's eerste 'shift' samples -- other's
    frame 0 komt overeen met ref's frame 'shift'). Geverifieerde conventie:
    dit moet een POSITIEVE lag opleveren, other[k] hoort thuis op
    ref-tijdlijnpositie (k + lag) = (k + shift), dus lag = +shift."""
    rng = np.random.default_rng(7)
    ref = _make_tone_burst(rng, SAMPLE_RATE * 20)

    ref_env, hop_samples = compute_rms_envelope(ref, SAMPLE_RATE)
    shift_samples = hop_samples * 100

    other = ref[shift_samples:].copy()
    other_env, _ = compute_rms_envelope(other, SAMPLE_RATE)

    lag_frames, confidence = estimate_alignment(ref_env, other_env, hop_samples, SAMPLE_RATE)

    expected_lag_frames = shift_samples // hop_samples
    assert lag_frames > 0, f"verwacht positieve lag, kreeg {lag_frames}"
    assert lag_frames == expected_lag_frames, (
        f"lag {lag_frames} frames != verwachte {expected_lag_frames} frames (hop-uitgelijnde shift, exact verwacht)"
    )
    assert confidence > 0.9
    print(f"OK test_lag_positive_when_other_is_truncated_copy (lag={lag_frames} frames, confidence={confidence:.3f})")


def test_failsafe_on_unrelated_signals():
    rng = np.random.default_rng(1)
    ch_a = _make_tone_burst(rng, SAMPLE_RATE * 15)
    ch_b = _make_tone_burst(rng, SAMPLE_RATE * 15)  # onafhankelijke ruis, geen enkel verband

    masks = compute_cross_channel_gate_masks({"a": ch_a, "b": ch_b}, SAMPLE_RATE, session_id="test")

    assert masks["a"].all(), "ongerelateerde kanalen mogen nooit onderdrukt worden (fail-safe)"
    assert masks["b"].all(), "ongerelateerde kanalen mogen nooit onderdrukt worden (fail-safe)"
    assert len(masks["a"]) == len(ch_a)
    assert len(masks["b"]) == len(ch_b)
    print("OK test_failsafe_on_unrelated_signals")


def test_silent_channel_suppressed_by_own_gate_not_by_crosstalk_failure():
    """Een volledig stil kanaal krijgt geen betrouwbare cross-kanaal-uitlijning
    (fail-safe, stage 2 doet niets) -- maar wordt WEL onderdrukt door de eigen-
    kanaal-ruisdrempel (stage 1, precies zoals live's gateOpen1 een stil kanaal
    altijd sluit, los van enig ander kanaal). Het luide kanaal blijft intact."""
    rng = np.random.default_rng(2)
    ch_a = _make_tone_burst(rng, SAMPLE_RATE * 10)
    ch_b = np.zeros(SAMPLE_RATE * 10, dtype=np.float32)  # volledig stil

    masks = compute_cross_channel_gate_masks({"a": ch_a, "b": ch_b}, SAMPLE_RATE, session_id="test")

    assert masks["a"].all(), "het luide kanaal mag niet geraakt worden door de stilte van het andere"
    # Niet exact 0 (not .any()): binary_opening/closing's randgedrag laat een
    # handvol frames aan de ware randen van de array soms net niet onderdrukt --
    # cosmetisch, zie ook de vergelijkbare marge in test_arbitrate_suppresses_...
    b_suppressed_frac = 1.0 - masks["b"].mean()
    assert b_suppressed_frac > 0.95, (
        f"een volledig stil kanaal moet vrijwel volledig onderdrukt worden door de "
        f"eigen ruisdrempel, maar slechts {b_suppressed_frac:.1%} was dat"
    )
    print("OK test_silent_channel_suppressed_by_own_gate_not_by_crosstalk_failure")


def test_own_gate_suppresses_quiet_far_mic_leak():
    """Rechtstreekse test van compute_own_channel_gate_mask() (stage 1): een
    kanaal dat overwegend heel zacht is (een ver-weg-microfoon-lek, RMS ruim
    onder de default-drempel van 0.015) moet grotendeels onderdrukt worden --
    dit is exact het scenario dat live al ving en Ververs Transcriptie miste
    vóór deze stage werd toegevoegd."""
    rng = np.random.default_rng(5)
    n = SAMPLE_RATE * 10
    quiet_leak = (rng.standard_normal(n) * 0.003).astype(np.float32)  # ruim < 0.015

    mask = compute_own_channel_gate_mask(quiet_leak, SAMPLE_RATE)
    suppressed_frac = 1.0 - mask.mean()
    assert suppressed_frac > 0.9, (
        f"een doorlopend zacht signaal (RMS 0.003 << drempel 0.015) moet grotendeels "
        f"onderdrukt worden, maar slechts {suppressed_frac:.1%} was dat"
    )
    print(f"OK test_own_gate_suppresses_quiet_far_mic_leak (suppressed={suppressed_frac:.1%})")


def test_own_gate_leaves_loud_speech_untouched():
    rng = np.random.default_rng(6)
    n = SAMPLE_RATE * 10
    real_speech = (rng.standard_normal(n) * 0.2).astype(np.float32)  # ruim > 0.015

    mask = compute_own_channel_gate_mask(real_speech, SAMPLE_RATE)
    assert mask.all(), "duidelijk luide, eigen spraak mag nooit door de ruisdrempel geraakt worden"
    print("OK test_own_gate_leaves_loud_speech_untouched")


def test_single_channel_skips_entirely():
    rng = np.random.default_rng(3)
    ch_a = _make_tone_burst(rng, SAMPLE_RATE * 5)
    masks = compute_cross_channel_gate_masks({"a": ch_a}, SAMPLE_RATE, session_id="test")
    assert masks["a"].all()
    assert len(masks) == 1
    print("OK test_single_channel_skips_entirely")


def test_active_channel_with_natural_pauses_is_not_chopped_up():
    """Reproduceert de echte sessie (2026-08-23) die deze wijziging veroorzaakte:
    een doorlopend gesproken, single-channel opname waarvan Ververs Transcriptie
    65,9% van de audio wegsneed onder de vaste DEFAULT_OWN_GATE_THRESHOLD (0.015)
    -- niet omdat het kanaal een lek was, maar omdat gewone spraakdynamiek
    (pauzes, wegstervende medeklinkers) net zo goed onder een vaste drempel valt.

    Signaal: afwisselend luide (0.2, ruim boven de drempel) en zachte-maar-niet-
    stille (0.008, onder DEFAULT_OWN_GATE_THRESHOLD=0.015 maar boven
    DEFAULT_SILENCE_FLOOR=0.005) stukken -- zoals spraak-met-pauzes, niet een
    structureel stil/lekkend kanaal. compute_own_channel_gate_mask() zou de
    zachte helft nog steeds onderdrukken (rechtstreeks getest); de entrypoint
    moet dat nu NIET meer doen, want het kanaal heeft aantoonbaar actieve
    inhoud (hoog p90)."""
    rng = np.random.default_rng(7)
    segment_n = SAMPLE_RATE * 1  # 1s per segment
    n_segments = 10
    parts = []
    for i in range(n_segments):
        amplitude = 0.2 if i % 2 == 0 else 0.008
        parts.append((rng.standard_normal(segment_n) * amplitude).astype(np.float32))
    speech_with_pauses = np.concatenate(parts)

    # Vóór deze wijziging zou dit ~50% onderdrukken (de zachte helft) -- rechtstreeks
    # getest op de ongewijzigde low-level functie, ter documentatie van het "oude" gedrag:
    direct_mask = compute_own_channel_gate_mask(speech_with_pauses, SAMPLE_RATE)
    direct_suppressed_frac = 1.0 - direct_mask.mean()
    assert direct_suppressed_frac > 0.3, (
        "sanity check: de rauwe per-frame-drempel moet de zachte helft van dit "
        f"signaal nog steeds als 'te stil' zien (was {direct_suppressed_frac:.1%})"
    )

    # De entrypoint (wat Ververs Transcriptie daadwerkelijk aanroept) moet dit
    # kanaal nu als aantoonbaar actief herkennen en helemaal niet onderdrukken:
    masks = compute_cross_channel_gate_masks({"a": speech_with_pauses}, SAMPLE_RATE, session_id="test")
    assert masks["a"].all(), (
        "een kanaal met duidelijke spraakdynamiek (afwisselend luid/zacht) mag niet "
        "aan flarden geknipt worden door de vaste eigen-ruisdrempel"
    )
    print("OK test_active_channel_with_natural_pauses_is_not_chopped_up "
          f"(directe drempel zou {direct_suppressed_frac:.1%} onderdrukt hebben)")


def test_arbitrate_suppresses_the_quieter_side_of_a_swap():
    """Rechtstreekse, deterministische test van arbitrate() (niet via ruwe audio
    + cross-correlatie-uitlijning -- dat introduceert onnodige kansvariatie in
    een test die alleen de arbitrage-REGEL zelf moet controleren, niet de
    uitlijnstap). Twee kanalen, offset 0 (al perfect uitgelijnd), met een
    handmatig geconstrueerde RMS-envelope: A luid/B stil in de eerste helft,
    omgekeerd in de tweede helft -- precies het patroon van een echt lek."""
    hop_samples, sample_rate = 320, SAMPLE_RATE  # 20ms/frame
    n_frames = 200  # 4s per helft

    loud, quiet = 0.2, 0.03  # quiet/loud = 0.15, ruim onder ARBITRATION_MARGIN=0.4
    env_a = np.concatenate([np.full(n_frames, loud), np.full(n_frames, quiet)])
    env_b = np.concatenate([np.full(n_frames, quiet), np.full(n_frames, loud)])

    result = arbitrate(
        {"a": env_a, "b": env_b}, {"a": 0, "b": 0}, hop_samples, sample_rate,
        session_id="test",
    )

    # Marge van enkele frames rond het omslagpunt EN rond de ware randen van de
    # array (binary_opening/closing's border_value-gedrag kan de laatste/eerste
    # paar frames van een randgevende run licht laten eroderen -- cosmetisch,
    # <100ms op een echte sessie, geen reden om de kernbewering te verzwakken).
    m = 5
    assert not result["a"][: n_frames - m].any(), "A mag niet onderdrukt zijn terwijl het zelf luid is"
    assert result["a"][n_frames + m : -m].all(), "A moet onderdrukt zijn in het stille/lekkende deel"
    assert result["b"][m : n_frames - m].all(), "B moet onderdrukt zijn in het stille/lekkende deel"
    assert not result["b"][n_frames + m :].any(), "B mag niet onderdrukt zijn terwijl het zelf luid is"
    print("OK test_arbitrate_suppresses_the_quieter_side_of_a_swap")


def test_arbitrate_never_suppresses_both_equal_channels():
    """Twee kanalen met identiek RMS-niveau -- geen enkele mag onderdrukt worden
    (own_rms < peer_rms * 0.4 kan niet waar zijn als beide gelijk zijn)."""
    hop_samples, sample_rate = 320, SAMPLE_RATE
    env_a = np.full(100, 0.1)
    env_b = np.full(100, 0.1)
    result = arbitrate({"a": env_a, "b": env_b}, {"a": 0, "b": 0}, hop_samples, sample_rate, session_id="test")
    assert not result["a"].any()
    assert not result["b"].any()
    print("OK test_arbitrate_never_suppresses_both_equal_channels")


def test_arbitrate_isolated_blip_is_smoothed_away():
    """Een verdenking van maar 1 frame (ver onder min_suppress_run_ms) moet door
    binary_opening weggepoetst worden -- voorkomt flikkerende onderdrukking."""
    hop_samples, sample_rate = 320, SAMPLE_RATE
    env_a = np.full(100, 0.1)
    env_b = np.full(100, 0.1)
    env_a[50] = 0.01  # één geïsoleerd stil frame temidden van verder gelijke niveaus
    result = arbitrate({"a": env_a, "b": env_b}, {"a": 0, "b": 0}, hop_samples, sample_rate, session_id="test")
    assert not result["a"].any(), "een enkel geïsoleerd frame mag geen onderdrukking triggeren"
    print("OK test_arbitrate_isolated_blip_is_smoothed_away")


def test_different_length_channels():
    """Een kanaal dat pas later begint (korter array) mag geen crash of
    verkeerde-lengte-mask opleveren."""
    rng = np.random.default_rng(11)
    ch_a = _make_tone_burst(rng, SAMPLE_RATE * 30)
    ch_b = _make_tone_burst(rng, SAMPLE_RATE * 12)  # veel korter, ongerelateerd

    masks = compute_cross_channel_gate_masks({"a": ch_a, "b": ch_b}, SAMPLE_RATE, session_id="test")
    assert len(masks["a"]) == len(ch_a)
    assert len(masks["b"]) == len(ch_b)
    print("OK test_different_length_channels")


def test_three_channels_reference_alignment():
    ref_channel, offsets, confidences = align_all_channels(
        {
            "a": np.ones(1000, dtype=np.float64),
            "b": np.ones(700, dtype=np.float64),
            "c": np.ones(500, dtype=np.float64),
        },
        hop_samples=320,
        sample_rate=SAMPLE_RATE,
    )
    assert ref_channel == "a", "langste kanaal moet als referentie gekozen worden"
    assert offsets["a"] == 0
    print(f"OK test_three_channels_reference_alignment (ref={ref_channel})")


if __name__ == "__main__":
    tests = [
        test_lag_negative_when_other_is_delayed_copy,
        test_lag_positive_when_other_is_truncated_copy,
        test_failsafe_on_unrelated_signals,
        test_silent_channel_suppressed_by_own_gate_not_by_crosstalk_failure,
        test_own_gate_suppresses_quiet_far_mic_leak,
        test_own_gate_leaves_loud_speech_untouched,
        test_single_channel_skips_entirely,
        test_active_channel_with_natural_pauses_is_not_chopped_up,
        test_arbitrate_suppresses_the_quieter_side_of_a_swap,
        test_arbitrate_never_suppresses_both_equal_channels,
        test_arbitrate_isolated_blip_is_smoothed_away,
        test_different_length_channels,
        test_three_channels_reference_alignment,
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
