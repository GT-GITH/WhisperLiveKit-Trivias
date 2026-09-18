# ADR-001 — Audio-first evidence model

- **Status:** Accepted
- **Datum:** 2026-09-18
- **Context van vastlegging:** ontwerpkeuze die al impliciet in de code zat (zie
  "Design Principles" in `CLAUDE.md`) maar nooit als toetsbare invariant was
  opgeschreven. Aanleiding: generatieve AI wordt onderdeel van
  asielbesluitvorming, waarmee de vraag "waar komt deze constatering vandaan"
  een juridische vraag wordt in plaats van een technische.

## Context

Dit platform is geen spraak-naar-tekst-applicatie en geen AI-assistent. Het is
een **audio-first, evidence-based platform** voor besluitvormingsprocessen met
hoge bewijslast (asielgehoren, en verwante overheidsprocessen).

Elke laag boven de audio is een interpretatie:

- een transcript is een interpretatie van wat er gebeurde;
- een vertaling is een interpretatie van dat transcript;
- een gesignaleerde inconsistentie is een interpretatie daarvan;
- een gehoorverslag of beslisadvies staat daar nog een laag vandaan.

De verleiding is om het systeem te optimaliseren voor de bovenste laag ("het
beste transcript", "de slimste signalering"). Dat is een doodlopende weg: het
leidt tot een wedstrijd tussen modellen die per definitie nooit eindigt, en het
maakt het product afhankelijk van de kwaliteit van het model dat er vandaag
toevallig achter hangt.

De keuze is daarom de omgekeerde: het systeem optimaliseert niet voor de
afgeleide laag, maar voor de **terugleidbaarheid** ervan.

## Beslissing

De oorspronkelijke opname is de primaire bron van bewijs. Alles wat daarna
ontstaat is een afgeleide laag en moet terugleidbaar blijven naar die bron.

De bewijsketen is:

```
persoon / rol / kanaal
  -> originele audio
    -> tijdstip / tijdsbereik
      -> transcriptsegment
        -> afgeleide constatering (vertaling, signalering, samenvatting)
          -> menselijke beoordeling
```

Voor elke inhoudelijke uitspraak die het platform doet, moet uiteindelijk
beantwoordbaar zijn: wie zei dit, op welk kanaal en in welke rol, wanneer, wat
is er precies getranscribeerd, welk audiofragment hoort erbij, welke andere
uitspraak of bron leidde tot de constatering, welk model/proces produceerde
haar, en heeft een mens haar geaccepteerd, gewijzigd of verworpen.

"Mogelijke inconsistentie geconstateerd" heeft op zichzelf geen bewijswaarde.
De bruikbare vorm is die constatering *plus* beide onderliggende uitspraken,
elk met spreker, tijdstip, transcriptsegment en afspeelbare audio. De
medewerker doet de beoordeling; het systeem levert het materiaal daarvoor aan.

## Invarianten

Deze zijn bindend voor elke wijziging in deze repository.

1. **De WAV op schijf wordt nooit gewijzigd.** Filtering, gating en
   ruisonderdrukking mogen uitsluitend werken op een in-memory kopie richting
   het ASR-model. (Nu geborgd in `cross_channel_gate.py` en
   `rebuild_channel_transcript()`.)
2. **Geen afgeleid artefact mag losraken van het bewijs dat nodig is om het te
   verifiëren.** Dit is de kernregel; alle andere invarianten volgen eruit.
3. **Kanaalidentiteit is onderdeel van het bewijsmodel, geen presentatie-
   metadata.** Het kanaal bepaalt wie sprak — bij een tolkgehoor is het verschil
   tussen wat de vreemdeling zei, wat de tolk vertaalde en wat de medewerker
   vroeg juridisch relevant.
4. **Tijd is de primaire sleutel.** Elk segment draagt een stroom-relatieve
   tijd in ms. Het audiofragment is daaruit deterministisch af te leiden:
   de opnames zijn 16 kHz mono s16le, dus byte-offset = ms x 32. Die conventie
   is onderdeel van het contract; wijzigt het opnameformaat, dan moet de
   afleiding expliciet meeveranderen.
5. **LIVE, FINAL en AUDIO zijn drie verschillende dingen en mogen nooit
   samenvallen.** Live transcript is een vluchtige UI-weergave, het batch-
   transcript is de autoritatieve tekstuele afgeleide, de audio is het bewijs.
6. **Modellen zijn vervangbare processors en nooit onderdeel van het duurzame
   datamodel.** Whisper-implementaties, LLM's, vertaalmodellen, prompts en
   analyse-engines wisselen; de bewijsketen overleeft dat. Het domeinmodel mag
   niet rond een specifiek model ontworpen worden.
7. **Verschillende modellen per kanaal mogen de herleidbaarheid niet
   verzwakken.** Als een kanaal een eigen model gebruikt, moet vastliggen welk
   model en welke configuratie het resultaat produceerde — inclusief het geval
   waarin op een fallback-model is teruggevallen, en waarom.
8. **AI mag afleiden en voorstellen; de mens beoordeelt.** UI's en API's mogen
   geen vorm aannemen waarin een gegenereerde conclusie losgekoppeld van haar
   herkomst gepresenteerd of geëxporteerd wordt.

## Gevolgen

**Positief.** Het product is niet afhankelijk van de vraag of het model van
vandaag beter hallucineert dan dat van een concurrent; de bewijsketen blijft
gelijk ongeacht welk model erachter hangt. Een discussie achteraf eindigt bij
de audio, niet bij "het model stelde dit vast". De reeds bestaande
meerkanaalsarchitectuur en het batch-model-onafhankelijk maken van de
batchlaag worden hierdoor structureel in plaats van toevallig.

**Kosten.** Afgeleide artefacten mogen niet gratis ontstaan: elk nieuw type
afleiding moet een antwoord hebben op "waar kan ik dit op terugvoeren". Dat
maakt sommige features duurder dan een naïeve implementatie.

**Expliciet niet besloten.** Deze ADR schrijft geen concrete velden of
migraties voor. De bestaande MVP-mechanismen (segment-contract, refresh-pad,
afspeelweergave) blijven ongewijzigd tot een feature ze daadwerkelijk nodig
heeft. Zie "Bekende gaten" hieronder.

## Bekende gaten (stand 2026-09-18)

Vastgelegd om te voorkomen dat ze onbewust doorgroeien; geen van alle vraagt om
een refactor nu.

| Gat | Risico | Wanneer aanpakken |
|---|---|---|
| `event_id` uit `CLAUDE.md`/`FO.md` bestaat niet in code; feitelijk ID is `id`, afgeleid van `start_ms` | ID is niet stabiel onder hertiming | Bij de eerste feature die naar een segment verwijst |
| "Ververs Transcriptie" overschrijft het kanaal-JSON volledig met nieuwe `refresh_*`-ID's | Elke opgeslagen verwijzing naar een segment breekt | Idem — dan `segment_uid` + `supersedes` |
| Gepersisteerde segmenten dragen geen model/modelversie/verwerkingspad | Met modelroutering per kanaal niet meer reproduceerbaar welk model welk segment maakte | **Bij merge van modelroutering per kanaal** — minimaal het gekozen model per kanaal opnemen in de sessie-metadata, waar de per-kanaal taal al staat |
| Geen gedeeld sessienulpunt met sub-seconde-precisie tussen kanalen; offset wordt uit de audio gemeten | Cross-kanaal-citaat heeft geen geverifieerde gemeenschappelijke klok | Bij cross-kanaal-inconsistentiesignalering |
| Geen wall-clock-koppeling per segment | Provability-principe is maar half geïmplementeerd | Bij externe koppeling (INDiGO) of juridische export |
| Afgekeurde/onderdrukte inhoud bestaat alleen in logregels | Niet zichtbaar wat het systeem heeft weggelaten | Bij de eerste feature die dat toont |
| Geen review-status op enig artefact | Menselijke beoordeling niet vastgelegd | Bij de eerste feature met een accepteer/verwerp-actie |

## Zie ook

- `CLAUDE.md`, sectie "Design Principles" en "Audio-first evidence model"
- `docs/FO.md` — functioneel ontwerp
- `features/cross-procedure-consistentie.md`, `features/signalering-dekking.md` —
  de features waarvoor bovenstaande gaten het eerst gaan knellen
