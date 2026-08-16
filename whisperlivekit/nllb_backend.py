"""On-prem vertaling via NLLB-200 (ctranslate2), zie
features/vertaling-niet-nl-tekst.md voor het volledige ontwerp.

Vervangt de eerdere op llm_backend.py/Ollama gebaseerde aanpak voor deze
feature: een algemeen chatmodel (llama3.1:8b) bleek voor morfologisch
complexe brontalen (Turks) onbetrouwbaar te vertalen -- geconstateerd tijdens
testen: een simpele Turkse zin over een zieke moeder ("Annem ciddi sekilde
hasta...") kwam terug met omgekeerde grammaticale persoon ("Ik ben ziek"
i.p.v. "Mijn moeder is ziek"). NLLB-200 is een model dat UITSLUITEND op
vertalen getraind is -- geen chat/instructie-laag die kan "verzinnen" -- en
presteert per parameter aanzienlijk beter op MT-taken.

Draait via ctranslate2 (al een dependency van dit project via faster-whisper,
dezelfde inference-engine) i.p.v. via het bestaande `nllw`-package
(optional-dependency "translation" in pyproject.toml): `nllw` is een
streaming/incrementele wrapper (LocalAgreement-policy voor live
token-per-token vertaling, zie whisperlivekit/core.py's
`--target-language`-scaffolding), geen simpele "vertaal deze losse
tekst"-API -- ongeschikt voor deze per-klik, on-demand usecase.

Puur, fail-safe -- zelfde patroon als llm_backend.py: geen dependencies,
geen geconfigureerd model, of een fout tijdens laden/vertalen -> None, nooit
een exception naar de aanroeper. Vereist de "nllb_translation"
optional-dependency (`pip install whisperlivekit[nllb_translation]`, of
gewoon `pip install transformers`) -- zonder is de feature stil uitgeschakeld.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("whisperlivekit.nllb_backend")

# Splitst op .!? gevolgd door whitespace, behoudt het leesteken bij de
# voorgaande zin. Regex, geen NLP-library (mosestokenizer/wtpsplit staan al
# als ongebruikte optional-dependency "sentence_tokenizer" in pyproject.toml,
# maar zijn zwaarder dan nodig voor de korte, simpele transcript-zinnen hier)
# -- goed genoeg voor dit doel, niet perfect bij afkortingen.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return parts or [text]

try:
    import ctranslate2
except ImportError:
    ctranslate2 = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# ISO 639-1 -> NLLB/FLORES-200-code, alleen de talen die dit project al kent
# (zie LANGUAGES in web_trivias/app.js, volledige lijst: docs/supported_languages.md).
# "ku" (generieke Koerdisch-optie in de UI) is bewust op Noord-Koerdisch
# (Kurmanci, Latijns schrift, "kmr_Latn") gezet -- de meest voorkomende
# variant bij Turkse/Syrische asielgehoren; Sorani-sprekers (Arabisch
# schrift, NLLB-code "ckb_Arab") vallen hier vooralsnog buiten, net als bij
# de bestaande taalselectie in de UI (die ook maar één "ku"-optie heeft).
ISO_TO_NLLB = {
    "nl": "nld_Latn",
    "en": "eng_Latn",
    "ar": "arb_Arab",
    "fa": "pes_Arab",
    "ru": "rus_Cyrl",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "tr": "tur_Latn",
    "so": "som_Latn",
    "ti": "tir_Ethi",
    "ku": "kmr_Latn",
    "sr": "srp_Cyrl",
    "bs": "bos_Latn",
}


class NLLBBackend:
    """Dunne wrapper om een ctranslate2 NLLB-vertaler + de bijbehorende
    HuggingFace-tokenizer. Eén instance per serverproces, hergebruikt
    modelgewichten net als TranscriptionEngine (core.py) -- niet per
    request opnieuw laden."""

    def __init__(self, model_dir: str, device: str = "auto") -> None:
        self._translator = ctranslate2.Translator(model_dir, device=device)
        # GEEN fix_mistral_regex=True (geprobeerd 2026-08-16, teruggedraaid):
        # de waarschuwing bij het laden ("incorrect regex pattern... you should
        # set fix_mistral_regex=True") is kennelijk een generieke, voor déze
        # (NLLB-)tokenizer niet-toepasselijke tekst uit transformers -- de kwarg
        # zette eerder correcte vertalingen (foreign_tr, al eens goed getest) om
        # in onzin. Dus bewust NIET zetten; het originele afkap-probleem (een
        # meerzinnige zin die halverwege stopt) blijft een bekende beperking,
        # zie features/vertaling-niet-nl-tekst.md.
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def translate(self, text: str, source_iso: Optional[str], target_iso: str = "nl") -> Optional[str]:
        """Vertaalt `text` van `source_iso` naar `target_iso` (default Nederlands).

        Retourneert None (nooit een exception) als: de tekst leeg is, de
        brontaal onbekend/niet in ISO_TO_NLLB staat (NLLB heeft -- anders dan
        een chatmodel -- altijd een EXPLICIETE brontaal nodig, kan niet zelf
        "raden"), of de vertaal-call zelf faalt.

        Vertaalt per ZIN, niet de hele `text` in één keer: NLLB-200 is op
        zin-niveau getraind, en bleek bij meerzinnige input (geconstateerd
        2026-08-16, diagnostische logging: n_target_tokens veel lager dan bij
        losse zinnen) een voortijdig stop-token te genereren en zo alles ná de
        eerste zin stilzwijgend te laten vallen. Losse zinnen in ÉÉN
        translate_batch()-call (efficiënt gebatcht, geen N losse calls) en de
        resultaten weer samenvoegen omzeilt dit."""
        text = (text or "").strip()
        if not text:
            return None

        src_code = ISO_TO_NLLB.get((source_iso or "").strip().lower())
        tgt_code = ISO_TO_NLLB.get((target_iso or "nl").strip().lower(), "nld_Latn")
        if not src_code:
            logger.warning(f"[NLLB] onbekende of ontbrekende brontaal-code: {source_iso!r}, vertaling overgeslagen")
            return None

        sentences = _split_sentences(text)
        try:
            self._tokenizer.src_lang = src_code
            source_batch = [
                self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(s)) for s in sentences
            ]
            logger.info(
                f"[NLLB][DIAG] input: src={src_code} tgt={tgt_code} text_len={len(text)} "
                f"n_sentences={len(sentences)} n_source_tokens={[len(t) for t in source_batch]} "
                f"sentences={sentences!r}"
            )
            results = self._translator.translate_batch(source_batch, target_prefix=[[tgt_code]] * len(source_batch))
            translated_sentences = []
            for i, result in enumerate(results):
                target_tokens = result.hypotheses[0][1:]  # eerste token is de target-taalcode zelf
                decoded = self._tokenizer.decode(
                    self._tokenizer.convert_tokens_to_ids(target_tokens),
                    skip_special_tokens=True,
                ).strip()
                logger.info(
                    f"[NLLB][DIAG] sentence {i}: n_target_tokens={len(target_tokens)} decoded={decoded!r}"
                )
                if decoded:
                    translated_sentences.append(decoded)
            translation = " ".join(translated_sentences).strip()
            logger.info(
                f"[NLLB][DIAG] decoded translation (samengevoegd): len={len(translation)} text={translation!r}"
            )
        except Exception as e:
            logger.warning(f"[NLLB] vertalen mislukt: {e}")
            return None

        return translation or None


def build_nllb_backend(args) -> Optional["NLLBBackend"]:
    """Fail-safe factory, zelfde patroon als llm_backend.build_llm_backend().

    Retourneert None (nooit een exception) als de feature niet geconfigureerd
    is (--nllb-model niet gezet), de dependencies ontbreken, of het model niet
    geladen kan worden -- de aanroeper (TriviasServer.py) moet dit dan
    afhandelen als "vertaalfunctie niet beschikbaar" (503), geen harde crash
    bij serverstart."""
    model_ref = getattr(args, "nllb_model", None)
    if not model_ref:
        return None

    if ctranslate2 is None or AutoTokenizer is None:
        logger.warning(
            "[NLLB] --nllb-model is gezet maar de dependency ontbreekt -- "
            "installeer met: pip install transformers (of whisperlivekit[nllb_translation])"
        )
        return None

    model_dir = model_ref
    if not Path(model_ref).is_dir():
        try:
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(model_ref)
        except Exception as e:
            logger.warning(f"[NLLB] kon model '{model_ref}' niet downloaden van HuggingFace Hub: {e}")
            return None

    device = getattr(args, "nllb_device", "auto")
    try:
        backend = NLLBBackend(model_dir, device=device)
    except Exception as e:
        logger.warning(f"[NLLB] laden van model '{model_ref}' mislukt: {e}")
        return None

    logger.info(f"[NLLB] vertaalmodel geladen: {model_ref} (device={device})")
    return backend
