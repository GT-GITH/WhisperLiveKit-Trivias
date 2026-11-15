# E2E-testsuite (laag 1: digitaal)

Stuurt scenario's met bekende tekst door de **echte server**, precies zoals de
browser dat doet (per kanaal een WebSocket, 16 kHz PCM in realtime), en scoort
het resultaat. Bedoeld voor de iteratieve cyclus: testen → interpreteren →
aanpassen → opnieuw testen.

## Op de pod

```bash
cd /workspace/WhisperLiveKit-Trivias && source .venv/bin/activate
pip install datasets soundfile librosa          # eenmalig, alleen voor de builder
python tests/e2e/build_scenarios.py             # eenmalig: scenario's bouwen
python tests/e2e/run_e2e.py                     # alle scenario's draaien
python tests/e2e/run_e2e.py nl_basis            # één scenario
python tests/e2e/run_e2e.py --save-baseline     # huidige uitkomst = basislijn
```

Rapport: `tests/e2e/results/<tijdstempel>/report.md`. Exitcode 1 bij fouten
of regressie t.o.v. `baseline.json`.

## Wat er gemeten wordt

Per scenario, per kanaal, voor twee paden: **incremental** (het tijdens de
sessie opgeslagen transcript) en **refresh** ("Ververs Transcriptie").

| metric | betekenis |
|---|---|
| WER / CER | woord-/tekenfoutpercentage t.o.v. de referentie |
| gemist | referentie-uitingen zonder enig segment |
| spurious | segmenten in stilte: hallucinatie of lek |
| live-fallback | opgeslagen segmenten zonder batch-tekst (niet citeerbaar) |
| ts med/max | afwijking segment-start vs echte start van de uiting |

## Scenario's

Recepten in `recipes/*.json` (deterministisch: vaste dataset-indexen). De
gebouwde audio staat niet in git. Zie `build_scenarios.py` voor het formaat.

- `nl_basis` — één NL-kanaal, korte pauzes
- `nl_lange_pauze` — 45 s stilte halverwege (drift, hallucinatie in stilte)
- `gehoor_tr_3kanaal` — hoorder + vreemdeling (Turks) + tolk
- `ar_ma_routering` — Marokkaans Arabisch via de oddadmix-routering

Scoring-unit-tests (lokaal te draaien): `python tests/e2e/test_scoring.py`.
