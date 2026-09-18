"""Eenmalig evaluatiescript, GEEN onderdeel van de reguliere pijplijn.

Doel: Sunbird/asr-whisper-51-african-languages testen EXACT zoals de eigen
modelkaart voorschrijft -- plain `transformers` (AutoProcessor +
WhisperForConditionalGeneration, forced_decoder_ids, greedy decode,
num_beams=1), GEEN ct2-transformers-converter, GEEN CTranslate2, GEEN enkele
bestandspatch van onze kant. scripts/eval_somali_wer_fleurs.py forceerde dit
model eerder door onze CTranslate2-pijplijn heen met twee handmatige patches
(tokenizer_config.json's extra_special_tokens, een geleende preprocessor_config.json
van openai/whisper-large-v3) -- terechte kritiek: dat is geen eerlijke test van
het model zoals de auteurs het bedoeld hebben. Dit script laat het model zijn
eigen, onaangepaste bestanden gebruiken en rapporteert eerlijk of dat al dan
niet werkt.

Zelfde FLEURS so_so-testset, dezelfde 10 samples, dezelfde WER/CER-berekening
als eval_somali_wer_fleurs.py, zodat de uitkomst rechtstreeks vergelijkbaar is
met de eerder gemeten cijfers (paza/large-v3/steja/sunbird-via-CTranslate2).

Taal-token-tabel (LANGUAGE_TOKENS_WHISPER/LANGUAGE_NAMES) letterlijk overgenomen
uit Sunbird's eigen modelkaart-voorbeeld -- "som": 50326 staat daar expliciet
onder "Existing languages codes from Whisper" (dus Somalisch gebruikt Whisper's
eigen, ongewijzigde taaltoken, geen van de heringerichte tokenslots voor talen
die Whisper origineel niet kende).

Vereist (eenmalig, niet in pyproject.toml):
    pip install datasets jiwer

Gebruik op de RunPod-pod (venv actief, internet nodig voor FLEURS + het
model, ~6GB):
    python scripts/eval_sunbird_official.py
"""

import io
import os
import re

# Vóór elke import die HF Hub kan aanroken -- zie eval_somali_wer_fleurs.py
# voor de achtergrond (HF_HUB_ENABLE_HF_TRANSFER=1 zonder hf_transfer-pakket
# in deze RunPod-omgeving).
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import soundfile as sf
import torch
import transformers
from datasets import Audio, get_dataset_config_names, load_dataset
from jiwer import cer, wer

MODEL_ID = "Sunbird/asr-whisper-51-african-languages"
N_SAMPLES = 10
SPLIT = "test"
SAMPLE_RATE = 16000

# Letterlijk overgenomen uit Sunbird's eigen modelkaart-voorbeeld.
LANGUAGE_TOKENS_WHISPER = {
    "eng": 50259, "fra": 50265, "swa": 50318, "sna": 50324, "yor": 50325, "som": 50326,
    "afr": 50327, "amh": 50334, "mlg": 50349, "lin": 50353, "hau": 50354,
    "ach": 50357, "aka": 50356, "bam": 50355, "bem": 50352, "ber": 50351,
    "cgg": 50350, "dag": 50348, "dga": 50347, "ewe": 50346, "ful": 50345,
    "ibo": 50344, "kab": 50343, "kau": 50342, "kik": 50341, "kin": 50340,
    "kln": 50339, "koo": 50338, "kpo": 50337, "led": 50336, "lgg": 50335,
    "lth": 50333, "lug": 50332, "luo": 50331, "luy": 50330, "myx": 50329,
    "nbl": 50328, "nya": 50323, "nyn": 50322, "orm": 50321, "pcm": 50320,
    "ruc": 50319, "rwm": 50317, "sot": 50316, "teo": 50315, "tsn": 50314,
    "ttj": 50313, "wol": 50312, "xho": 50311, "xog": 50310, "zul": 50309,
}


def resolve_somali_config() -> str:
    candidate = "so_so"
    try:
        configs = get_dataset_config_names("google/fleurs")
    except Exception as e:
        print(f"Kon configlijst niet ophalen ({e}), probeer '{candidate}' direct.")
        return candidate
    if candidate in configs:
        return candidate
    matches = [c for c in configs if c.startswith("so_")]
    if not matches:
        raise RuntimeError(f"Geen Somalisch-config gevonden in FLEURS. Beschikbare configs: {configs}")
    print(f"'{candidate}' niet gevonden, gebruik in plaats daarvan: {matches[0]}")
    return matches[0]


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    config = resolve_somali_config()
    print(f"FLEURS-config: {config}, split={SPLIT}, n={N_SAMPLES}")
    ds = load_dataset("google/fleurs", config, split=f"{SPLIT}[:{N_SAMPLES}]")
    ds = ds.cast_column("audio", Audio(decode=False))

    print(f"Laad {MODEL_ID} via plain transformers (AutoProcessor + WhisperForConditionalGeneration)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID)
    model = transformers.WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)

    lang_tok = LANGUAGE_TOKENS_WHISPER["som"]
    transcribe_tok = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
    notimestamps_tok = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    forced_decoder_ids = [(1, lang_tok), (2, transcribe_tok), (3, notimestamps_tok)]

    refs, hyps = [], []
    for i, sample in enumerate(ds):
        audio, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"Onverwachte samplerate {sr}, verwacht {SAMPLE_RATE}")
        reference = sample.get("transcription") or sample.get("raw_transcription")
        if not reference:
            print(f"[{i}] geen referentietekst gevonden, velden: {list(sample.keys())}")
            continue

        input_features = processor(
            audio, sampling_rate=SAMPLE_RATE, do_normalize=True, return_tensors="pt"
        ).input_features.to(device)
        predicted_ids = model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids, num_beams=1, do_sample=False,
        )
        hyp = processor.decode(predicted_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

        print(f"\n--- sample {i} ---")
        print("REF   :", reference)
        print("SUNBIRD (officieel):", hyp)

        refs.append(normalize(reference))
        hyps.append(normalize(hyp) or " ")

    w = wer(refs, hyps)
    c = cer(refs, hyps)
    print(f"\n=== Aggregaat: WER={w:.3f}  CER={c:.3f}  (n={len(refs)}) ===")
    print("Ter vergelijking (eval_somali_wer_fleurs.py, via CTranslate2 + onze patches):")
    print("  large-v3 WER=0.924 CER=0.342 | steja WER=0.599 CER=0.202 | sunbird(CT2) WER=0.705 CER=0.368")


if __name__ == "__main__":
    main()
