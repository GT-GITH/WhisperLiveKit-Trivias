# Trivias STT – Audio-First Evidence Platform
## Functioneel Ontwerp

---

## 1. Doel van dit document

Dit document legt het **vaste ontwerp- en denkraam** vast voor het Trivias-STT-platform, zodat discussies over live vs batch, transcriptie als bron, en complexiteit **niet telkens opnieuw gevoerd hoeven te worden**.

Het document beschrijft:
- het **waarom** (product & markt)
- het **wat** (scope & capabilities)
- het **hoe** (architectuur & ontwerpprincipes)
- het **wat niet** (bewuste afbakening)

Dit is een **levend ontwerpdocument**, bedoeld voor productbeslissingen, uitleg aan stakeholders/klanten, en consistentie bij doorontwikkeling.

---

## 2. Productvisie (niet-onderhandelbaar)

### 2.1 Kernvisie

Trivias STT is **geen transcriptietool**.

Trivias STT is een:

> **On-prem, evidence-first analyseplatform voor gehoor- en verhoorsituaties**

waarbij:
- **audio** de primaire waarheid is
- **tekst** een afgeleide interpretatielaag is
- **bewijsbaarheid** (tijd + audio) altijd behouden blijft

### 2.2 Doelklanten

Organisaties die:
- formele gesprekken voeren (gehoren, verhoren, interviews)
- met tolken werken
- juridisch moeten kunnen verantwoorden **wat, wanneer, door wie** is gezegd
- **niet naar de cloud mogen**

Voorbeelden: migratie- en asieldiensten, politie/opsporing, rechtspraak, toezichthouders.

---

## 3. Fundamentele ontwerpprincipes

**Principe 1 — Audio is de grondstof**
- De opname zelf is het bewijs
- Transcriptie is metadata over audio
- Geen enkele analyse mag bestaan zonder verwijzing naar audio + tijd

**Principe 2 — Tijdlijn boven tekst**
- Alles is tijd-gebaseerd
- Segmenten/events zijn de ruggengraat
- Tekst kan veranderen; tijd + audio niet

**Principe 3 — Transcriptie is interpretatie, geen waarheid**
- Live transcriptie = snelle, optimistische interpretatie
- Batch transcriptie = langzamere, conservatievere interpretatie
- Beide zijn feilbaar

**Principe 4 — Evidence-first explainability**
- Elke bevinding moet herleidbaar zijn naar: `event_id`, `start/end timestamp`, `audiofragment`

---

## 4. Datamodel (conceptueel)

### 4.1 SpeechEvent (kernentiteit)

Een SpeechEvent vertegenwoordigt een uitspraak in de tijd.

| Veld | Beschrijving |
|------|-------------|
| `event_id` | Stabiel, deterministisch ID |
| `start_ms / end_ms` | Tijdsgrenzen in milliseconden |
| `speaker_id` | Uit diarization of kanaal-mapping |
| `speaker_role` | vreemdeling / tolk / medewerker / onbekend |
| `text_live` | Live decoder output |
| `text_batch` | Batch decoder output |
| `text_final` | Display tekst (batch indien geaccepteerd) |
| `audio_ref` | session.wav + byte offsets |

Afgeleide relaties: `translations`, `inconsistency_flags`, `legal_references`

### 4.2 Invariant

> **Geen SpeechEvent mag bestaan zonder audio-referentie.**

---

## 5. Live transcriptie (situational awareness)

- Streaming / lage latency
- Segmentatie via VAD
- Tekststatus: `LIVE → FINAL`
- **Niet juridisch definitief** — mag fouten bevatten
- Wordt **niet** gebruikt als eindbron voor analyse

---

## 6. Batch transcriptie (kwaliteitslaag)

- Batch windows van **±30 seconden**, gesloten op eerstvolgende stilte (VAD close)
- Hard cap: 45s
- Output geprojecteerd op bestaande SpeechEvents (geen nieuwe segmentstructuur)
- Batch mag kwaliteit verbeteren, maar: **mag geen bewijs verwijderen**, audio blijft leidend

### Parameters (aanbevolen defaults)
```
TARGET_WINDOW_MS = 30_000
MIN_WINDOW_MS    = 15_000
HARD_CAP_MS      = 45_000
CONTEXT_PAD_MS   = 600 (pre/post)
```

---

## 7. Diarization en rollen

### Beslisregel (hard vastgelegd)
```
If number_of_channels >= number_of_speakers → diarization disabled
```

- **Kanaal-gebaseerde toewijzing is het preferred path**: `channel_id → speaker_id → speaker_role`
- Diarization is een **fallback-mechanisme**, geen kernvereiste

---

## 8. Vreemdeling ↔ Tolk inconsistentie-analyse

Automatisch signaleren van: inhoudelijke afwijkingen, weglatingen, toevoegingen, betekenisverschuivingen.

Werkwijze: align op tijdlijn → normaliseer taal → vergelijk semantiek + harde feiten → flag inconsistenties.

Resultaat: **geen oordeel, maar signalen** — altijd met verwijzing naar audio.

---

## 9. Juridische AI-agent

- Koppelt uitspraken aan relevante wetgeving
- Redeneert **alleen** op basis van citeerbare SpeechEvents
- RAG tegen on-prem wetgeving
- Output: relevante wetten + motivatie + verwijzing naar events + audio
- **Geen claims zonder bewijs**

---

## 10. On-prem only (productkeuze)

Geen afhankelijkheid van cloud-SaaS. Gebruik van on-prem modellen en infrastructuur. Modulaire architectuur.

> Dit is geen beperking, maar een **concurrentievoordeel**.

---

## 11. Wat Trivias STT expliciet níet is

- Geen generieke transcriptie-app
- Geen SaaS meeting-tool
- Geen black-box AI
- Geen juridisch beslissingssysteem

> Het systeem **ondersteunt** besluitvorming, het **vervangt** die niet.

---

## 12. Samenvattende ontwerpregels (checklist)

- [ ] Audio is altijd leidend
- [ ] Alles is tijd-gebaseerd
- [ ] Transcriptie is vervangbaar
- [ ] Segmentstructuur is stabiel
- [ ] Batch corrigeert, herschrijft niet blind
- [ ] Analyse is citeerbaar
- [ ] On-prem is verplicht

---

## 13. Technisch ontwerp – Batch transcriptie (Window + Projectie)

### 13.4 Invarianten (niet-onderhandelbaar)

- Audio blijft leidend: batch wijzigt **nooit** audio of timing
- Segmentstructuur is stabiel: batch creëert **geen** nieuwe events in V1
- Batch wijzigt alleen: `text_batch`, optioneel `text_final`
- Elk batch-resultaat is traceerbaar: `batch_job_id`, model/config fingerprint, window start/end

### 13.5 Window-closure regels

Een window sluit wanneer:
1. `(now_ms - w_start_ms) >= TARGET_WINDOW_MS`
2. én een silence-close optreedt (VAD close / segment FINAL event)

Uitzonderingen:
- Geen stilte en `>= HARD_CAP_MS` → force close
- Stilte te vroeg (`< MIN_WINDOW_MS`) → window nog niet sluiten

### 13.6.5 Acceptatie naar text_final

Batch wordt als "verdacht" beschouwd (→ `text_final` blijft live) als:
- `batch token count < 0.8 * live token count`
- of batch empty terwijl live non-empty

### 13.7 WS contract: segment_update

```json
{
  "type": "segment_update",
  "session_id": "...",
  "event_id": "...",
  "start_ms": 1820,
  "end_ms": 4740,
  "text_batch": "...",
  "text_final": "...",
  "batch_job_id": "..."
}
```

UI gedrag: vervang tekst in bestaand UI-blok (zelfde event_id), geen nieuwe blokken, geen scroll jump.

---

## 14. Slot

> Complexiteit in dit platform is geen gevolg van over-engineering, maar van juridische eisen, bewijsbaarheid en productverkoopbaarheid.
>
> **Afwijken van deze principes betekent een ander product bouwen.**
