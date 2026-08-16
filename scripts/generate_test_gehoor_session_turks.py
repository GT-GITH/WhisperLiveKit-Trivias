"""Genereert een synthetische gehoor-sessie waarbij de vreemdeling écht
Turks spreekt (channel_id "foreign_tr"), i.p.v. de Nederlandse
placeholder-tekst die generate_test_gehoor_session.py gebruikt.

Doel: testdata voor de (nog te bouwen) vertaalfunctionaliteit van niet-NL
transcript-tekst -- zie het geparkeerde vertaal-toggle-idee in memory
(project_translation_toggle_idea). Om dat te kunnen testen moet er
daadwerkelijk niet-Nederlandse brontekst in het transcript staan; de
bestaande testsessie had dat niet (elk kanaal, ook "foreign_ar", sprak
al Nederlands).

De tolk-regels zijn Nederlandse vertalingen van wat de vreemdeling zegt
(realistisch: een tolk parafraseert, is geen woordelijke vertaling) --
dat is bewust ANDERS dan de toekomstige "vertaal"-knop, die een
onafhankelijke, letterlijke vertaling van de foreign_tr-brontekst zelf
moet leveren. Beide naast elkaar hebben is precies wat nuttig is om te
testen (en is ook voorwerk voor de geparkeerde tolk-inconsistentiecheck).

Zelfde mechanisme als generate_test_gehoor_session.py (stille
placeholder-WAV's, JSON direct in recordings/) -- zie dat script voor
achtergrond over waarom dat volstaat voor dit doel.

Gebruik (op de runpod, vanuit de repo-root):
    python scripts/generate_test_gehoor_session_turks.py
"""

import json
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

RECORDINGS_DIR = Path("recordings")
SAMPLE_RATE = 16000

# (channel_id, tekst) in spreekvolgorde.
#   employee    -> Hoormedewerker (Nederlands)
#   interpreter -> Tolk (Nederlandse weergave/parafrase van het Turks)
#   foreign_tr  -> Vreemdeling (Turks, de brontaal die vertaald moet worden)
DIALOGUE: List[Tuple[str, str]] = [
    ("employee", "Voordat we verder gaan, heeft u nog correcties of aanvullingen op het aanmeldgehoor?"),
    ("interpreter", "Görüşmeye başlamadan önce, önceki görüşmeyle ilgili düzeltmeniz gereken bir şey var mı?"),
    ("foreign_tr", "Hayır, her şey doğruydu, düzeltecek bir şeyim yok."),
    ("interpreter", "Nee, dat klopte allemaal, ik heb niets te corrigeren."),

    ("employee", "We hebben uw documenten laten onderzoeken. De uitslag is dat het paspoort echt lijkt."),
    ("interpreter", "Belgelerinizi inceledik. Sonuca göre pasaportunuz gerçek görünüyor."),
    ("foreign_tr", "Evet, bu benim kendi pasaportum."),
    ("interpreter", "Ja, dat is mijn eigen paspoort."),
    ("employee", "Heeft u verder nog nieuwe documenten over uw identiteit of reisroute?"),
    ("interpreter", "Kimliğiniz veya yolculuğunuzla ilgili başka yeni belgeniz var mı?"),
    ("foreign_tr", "Hayır, başka belgem yok."),
    ("interpreter", "Nee, ik heb geen andere documenten."),

    ("employee", "Kunt u kort vertellen om welke reden u uw land heeft verlaten?"),
    ("interpreter", "Ülkenizi hangi sebeple terk ettiğinizi kısaca anlatabilir misiniz?"),
    ("foreign_tr", "Ailem siyasi baskı yüzünden tehdit edildiği için ülkemi terk etmek zorunda kaldım."),
    ("interpreter", "Ik moest mijn land verlaten omdat mijn familie werd bedreigd vanwege politieke druk."),

    ("employee", "Vertelt u eens in uw eigen woorden wat er precies is gebeurd."),
    ("interpreter", "Lütfen kendi kelimelerinizle tam olarak ne olduğunu anlatın."),
    ("foreign_tr", "Kardeşim gazeteciydi ve hükümeti eleştiren yazılar yazıyordu. Bir gece evimize silahlı kişiler geldi."),
    ("interpreter", "Mijn broer was journalist en schreef stukken die kritiek hadden op de regering. Op een avond kwamen gewapende mannen naar ons huis."),
    ("foreign_tr", "İkinci gelişlerinde babamı dövdüler ve sıradaki kişinin ben olacağımı söylediler."),
    ("interpreter", "De tweede keer dat ze kwamen, hebben ze mijn vader geslagen en gezegd dat ik de volgende zou zijn."),
    ("foreign_tr", "O gece hemen sınıra doğru yola çıktık."),
    ("interpreter", "Diezelfde nacht zijn we meteen naar de grens vertrokken."),

    ("employee", "Kunt u vertellen wanneer die tweede keer precies was?"),
    ("interpreter", "İkinci gelişin tam olarak ne zaman olduğunu söyleyebilir misiniz?"),
    ("foreign_tr", "Bu Mart ayında oldu, tam tarihini hatırlamıyorum."),
    ("interpreter", "Dat was in maart, de exacte datum weet ik niet meer."),
    ("employee", "U zei dat uw vader geslagen werd. Wat deed dat met u op dat moment?"),
    ("interpreter", "Babanızın dövüldüğünü söylediniz. O anda bu sizi nasıl etkiledi?"),
    ("foreign_tr", "Çok korkmuştum, onu öldüreceklerini düşündüm."),
    ("interpreter", "Ik was heel bang, ik dacht dat ze hem zouden vermoorden."),

    ("employee", "Zou u ergens anders in het land kunnen wonen, weg van die groep?"),
    ("interpreter", "Ülkenin başka bir yerinde, o gruptan uzakta yaşayabilir miydiniz?"),
    ("foreign_tr", "Hayır, bu grup ülkenin her yerinde faaliyet gösteriyor, güvenli bir yer yok."),
    ("interpreter", "Nee, die groep is in het hele land actief, er is geen veilige plek."),

    ("employee", "Heeft u verder nog iets dat u kwijt wilt over uw persoonlijke situatie, los van dit verhaal?"),
    ("interpreter", "Bu hikayenin dışında, kişisel durumunuzla ilgili eklemek istediğiniz başka bir şey var mı?"),
    ("foreign_tr", "Annem ciddi şekilde hasta ve onun için çok endişeleniyorum."),
    ("interpreter", "Mijn moeder is ernstig ziek en ik maak me grote zorgen om haar."),

    ("employee", "Heeft u zelf ooit oorlogsmisdrijven gezien of meegemaakt die u zou willen melden?"),
    ("interpreter", "Bildirmek isteyeceğiniz herhangi bir savaş suçuna tanık oldunuz mu ya da yaşadınız mı?"),
    ("foreign_tr", "Hayır, böyle bir şey yaşamadım."),
    ("interpreter", "Nee, dat heb ik niet meegemaakt."),

    ("employee", "Heeft u de tolk goed kunnen verstaan en begrijpen tijdens dit gehoor?"),
    ("interpreter", "Bu görüşme boyunca beni iyi anlayabildiniz mi?"),
    ("foreign_tr", "Evet, iyi gitti, her şeyi takip edebildim."),
    ("interpreter", "Ja, dat ging goed, ik heb alles kunnen volgen."),
    ("employee", "Dank u wel, hiermee sluiten we het gehoor af."),
    ("interpreter", "Teşekkür ederim, görüşmeyi burada sonlandırıyoruz."),
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
    print("Vreemdeling-kanaal (foreign_tr) bevat echte Turkse brontekst -- geschikt")
    print("om de vertaalfunctionaliteit van niet-NL tekst tegen te testen.")
    print("(De WAV's bevatten alleen stilte -- alleen de JSON-tekst is relevant.)")


if __name__ == "__main__":
    main()
