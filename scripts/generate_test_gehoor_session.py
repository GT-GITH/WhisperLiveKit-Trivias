"""Genereert een synthetische, gehoor-achtige sessie (transcript-JSON's +
stille placeholder-WAV's) rechtstreeks in recordings/, zodat je 'm in de
Trivias-UI kan selecteren en direct op "Genereer gehoorverslag" kan klikken
-- zonder eerst een echte opname te hoeven doen.

Puur voor het testen van de gehoorverslag-export-feature (zie
whisperlivekit/gehoorverslag.py). Geen echte audio: de WAV's zijn stilte
(de export leest toch alleen de JSON, zie
TriviasServer._load_merged_transcript()), maar wel geldig en lang genoeg
zodat de sessie-lijst en eventuele terugluister-UI niet stuklopen.

De dialoog is realistische, thematisch geordende interview-content -- geen
classificatie meer om tegen te testen (die feature is losgelaten, zie
features/gehoorverslag-automatisering.md in de projectrepo), puur bruikbaar
als representatieve testinvoer voor de platte export.

Gebruik (op de runpod, vanuit de repo-root):
    python scripts/generate_test_gehoor_session.py
"""

import json
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

RECORDINGS_DIR = Path("recordings")
SAMPLE_RATE = 16000

# (channel_id, tekst) in spreekvolgorde. channel_id bepaalt de rol
# (zie whisperlivekit/gehoorverslag.py ROLE_LABELS):
#   employee    -> Hoormedewerker
#   interpreter -> Tolk
#   foreign_ar  -> Vreemdeling (taal: Arabisch)
DIALOGUE: List[Tuple[str, str]] = [
    # --- 2.1 Correcties en aanvullingen ---
    ("employee", "Voordat we verder gaan, heeft u nog correcties of aanvullingen op het aanmeldgehoor?"),
    ("interpreter", "Voordat we verdergaan, heeft u nog correcties op het eerdere gehoor?"),
    ("foreign_ar", "Nee, dat klopte allemaal."),
    ("interpreter", "Nee, dat klopte allemaal."),

    # --- 2.2 Bespreking onderzoek(en) ---
    ("employee", "We hebben uw documenten laten onderzoeken. De uitslag is dat het paspoort echt lijkt."),
    ("interpreter", "Uw documenten zijn onderzocht, het paspoort lijkt echt te zijn."),
    ("foreign_ar", "Ja, dat is mijn eigen paspoort."),
    ("interpreter", "Ja, dat is mijn eigen paspoort."),
    ("employee", "Heeft u verder nog nieuwe documenten over uw identiteit of reisroute?"),
    ("interpreter", "Heeft u nog andere documenten over uw identiteit of reis?"),
    ("foreign_ar", "Nee, ik heb verder niets meer."),
    ("interpreter", "Nee, ik heb verder niets meer."),

    # --- 4.1 Inleidende vragen ---
    ("employee", "Kunt u kort vertellen om welke reden u uw land heeft verlaten?"),
    ("interpreter", "Kunt u kort vertellen waarom u uw land heeft verlaten?"),
    ("foreign_ar", "Ik moest vluchten vanwege bedreigingen tegen mijn familie."),
    ("interpreter", "Ik moest vluchten vanwege bedreigingen tegen mijn familie."),

    # --- 3 Asielrelaas (open verhaal, in eigen woorden) ---
    ("employee", "Vertelt u eens in uw eigen woorden wat er precies is gebeurd."),
    ("interpreter", "Vertelt u alstublieft in uw eigen woorden wat er is gebeurd."),
    ("foreign_ar", "Ik werd bedreigd door een gewapende groep omdat mijn broer journalist was. Ze kwamen twee keer bij ons huis langs."),
    ("interpreter", "Hij werd bedreigd door een gewapende groep omdat zijn broer journalist was. Ze kwamen twee keer bij hun huis langs."),
    ("foreign_ar", "De tweede keer hebben ze mijn vader geslagen en gezegd dat ik de volgende zou zijn."),
    ("interpreter", "De tweede keer hebben ze zijn vader geslagen en gezegd dat hij de volgende zou zijn."),
    ("foreign_ar", "Toen zijn we diezelfde nacht vertrokken naar de grens."),
    ("interpreter", "Toen zijn ze diezelfde nacht vertrokken naar de grens."),

    # --- 4.2 Verdere vragen (gerichte vervolgvragen) ---
    ("employee", "Kunt u vertellen wanneer die tweede keer precies was?"),
    ("interpreter", "Wanneer was die tweede keer precies?"),
    ("foreign_ar", "Dat was in maart, ik weet de exacte datum niet meer."),
    ("interpreter", "Dat was in maart, de exacte datum weet hij niet meer."),
    ("employee", "U zei dat uw vader geslagen werd. Wat deed dat met u op dat moment?"),
    ("interpreter", "U zei dat uw vader geslagen werd. Wat deed dat met u op dat moment?"),
    ("foreign_ar", "Ik was heel bang, ik dacht dat ze hem zouden vermoorden."),
    ("interpreter", "Hij was heel bang, hij dacht dat ze zijn vader zouden vermoorden."),

    # --- 4.3 Overig asielrelaas (verblijfsmogelijkheden elders) ---
    ("employee", "Zou u ergens anders in het land kunnen wonen, weg van die groep?"),
    ("interpreter", "Zou u ergens anders in het land kunnen wonen, weg van die groep?"),
    ("foreign_ar", "Nee, die groep is actief in het hele land, er is geen veilige plek."),
    ("interpreter", "Nee, die groep is actief in het hele land, er is geen veilige plek."),

    # --- 4.5 Bijzondere individuele omstandigheden ---
    ("employee", "Heeft u verder nog iets dat u kwijt wilt over uw persoonlijke situatie, los van dit verhaal?"),
    ("interpreter", "Heeft u nog iets anders te vertellen over uw persoonlijke situatie?"),
    ("foreign_ar", "Mijn moeder is ernstig ziek en ik maak me grote zorgen om haar."),
    ("interpreter", "Zijn moeder is ernstig ziek en hij maakt zich grote zorgen om haar."),

    # --- 4.6 Getuigenissen van oorlogsmisdrijven ---
    ("employee", "Heeft u zelf ooit oorlogsmisdrijven gezien of meegemaakt die u zou willen melden?"),
    ("interpreter", "Heeft u zelf oorlogsmisdrijven gezien die u zou willen melden?"),
    ("foreign_ar", "Nee, dat is bij mij niet aan de orde geweest."),
    ("interpreter", "Nee, dat is bij hem niet aan de orde geweest."),

    # --- 5.1 Op- en aanmerkingen over de gang van zaken ---
    ("employee", "Heeft u de tolk goed kunnen verstaan en begrijpen tijdens dit gehoor?"),
    ("interpreter", "Heeft u mij goed kunnen verstaan tijdens dit gehoor?"),
    ("foreign_ar", "Ja, dat ging goed, ik heb alles kunnen volgen."),
    ("interpreter", "Ja, dat ging goed, ik heb alles kunnen volgen."),
    ("employee", "Dank u wel, hiermee sluiten we het gehoor af."),
    ("interpreter", "Dank u wel, hiermee sluiten we het gehoor af."),
]


def _layout_timestamps(dialogue: List[Tuple[str, str]]) -> List[Tuple[str, int, int, str]]:
    """Kent start_ms/end_ms toe op basis van woordaantal (~0.45s/woord,
    min. 1.2s) + een korte pauze tussen beurten. Puur voor realistische,
    oplopende tijdstempels -- geen exacte spreeksnelheid nodig voor dit doel."""
    segments = []
    t = 2000  # klein beginmarge
    for channel, text in dialogue:
        n_words = max(1, len(text.split()))
        duration_ms = max(1200, int(n_words * 450))
        start_ms = t
        end_ms = start_ms + duration_ms
        segments.append((channel, start_ms, end_ms, text))
        t = end_ms + 400  # pauze tussen beurten
    return segments


def _write_silent_wav(path: Path, duration_s: float) -> None:
    n_frames = int(duration_s * SAMPLE_RATE)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"\x00\x00" * n_frames)


def main() -> None:
    segments = _layout_timestamps(DIALOGUE)
    session_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    by_channel: dict[str, list[dict]] = {}
    max_end_ms: dict[str, int] = {}
    for channel, start_ms, end_ms, text in segments:
        by_channel.setdefault(channel, []).append({
            "type": "segment_update",
            "id": f"synth_{start_ms}",
            "text_batch": text,
            "text_final": text,
            "state": "FINAL",
            "start_ms": start_ms,
            "end_ms": end_ms,
        })
        max_end_ms[channel] = max(max_end_ms.get(channel, 0), end_ms)

    for channel, entries in by_channel.items():
        stem = f"session_{session_id}_{channel}_{ts}"
        json_path = RECORDINGS_DIR / f"{stem}.json"
        wav_path = RECORDINGS_DIR / f"{stem}.wav"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

        _write_silent_wav(wav_path, duration_s=max_end_ms[channel] / 1000.0 + 1.0)

        print(f"  {channel:12s} {len(entries):2d} segmenten -> {json_path.name}")

    print(f"\nSessie aangemaakt: {session_id}")
    print("Open de Trivias-UI, ga naar Sessies, selecteer deze sessie en klik op 'Genereer gehoorverslag'.")
    print("(De WAV's bevatten alleen stilte -- het gehoorverslag leest toch alleen de JSON-tekst.)")


if __name__ == "__main__":
    main()
