# E2E-testsuite (laag 1: digitaal)

Stuurt scenario's met bekende tekst door de **echte server**, precies zoals de
browser dat doet (per kanaal een WebSocket, 16 kHz PCM in realtime), en scoort
het resultaat. Bedoeld voor de iteratieve cyclus: testen → interpreteren →
aanpassen → opnieuw testen.

## Op de pod

```bash
cd /workspace/WhisperLiveKit-Trivias

# Builder: EIGEN venv. Niet in de server-.venv installeren -- `datasets` trekt
# huggingface-hub>=1.0 en numpy>=2 binnen en breekt daarmee transformers/
# tokenizers en de numpy-1.x-stack van de server (gebeurd op 2026-11-15).
python3 -m venv /workspace/e2e-venv
/workspace/e2e-venv/bin/pip install datasets soundfile librosa faster-whisper  # faster-whisper alleen voor de Silero-VAD
/workspace/e2e-venv/bin/python tests/e2e/build_scenarios.py   # scenario's bouwen

# Runner: server-venv is prima (alleen stdlib + websockets)
source .venv/bin/activate
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
- `muziek_tr` — laag 2: 75 s instrumentale muziek, dan Turkse spraak met muziek
  eronder, dan schoon, dan weer muziek (spraak na lange muziek; tekst in muziek)

Achtergrondmuziek komt uit de Free Music Archive (alleen CC-BY e.d., ook
commercieel); de builder kiest zelf instrumentale tracks (frisse VAD < 3% spraak)
en zet titel/artiest/licentie in `reference.json` (naamsvermelding).

Scoring-unit-tests (lokaal te draaien): `python tests/e2e/test_scoring.py`.
