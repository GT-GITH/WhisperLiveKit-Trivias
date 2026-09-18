"""Eenmalig diagnose-scriptje, GEEN onderdeel van de reguliere pijplijn.

Doel: op EXACT hetzelfde audiofragment (sessie 15f7501c-5bff-45a3-a28a-
9b9653071ab3, job d05651e8, window 0..30230ms -- geldig Somalisch, ditmaal
afgewezen op no_speech_prob=0.857, niet op repetitie of een te korte tekst;
faster-whisper gaf hier 4 zinnen terug, o.a. "Guur kum marhaadu le stedjis...")
vergelijken:
  a) microsoft/paza-whisper-large-v3-turbo (het huidige PoC-kandidaatmodel,
     fine-tuned op 6 Oost-Afrikaanse talen -- 5 Bantoetalen + Somalisch als
     enige Cushitische uitzondering; modelkaart zelf: "not recommended for
     real-world use without further testing")
  b) large-v3 (stock, meertalig, faster-whisper haalt automatisch de
     kant-en-klare CT2-conversie op -- geen eigen conversie nodig)

Als (b) dit fragment wel fatsoenlijk transcribeert terwijl (a) faalt, ligt
het aan de modelkeuze, niet aan onze pijplijn/conversie. Faalt (b) ook, dan
is dit specifieke audiomateriaal het probleem.

Roept faster_whisper.WhisperModel rechtstreeks aan, buiten BatchFasterWhisperASR
en de rest van de pijplijn (geen gate, geen ChannelTranscriptionConfig) om
model, conversie en pijplijn-instellingen als aparte variabelen te scheiden.

Gebruik op de RunPod-pod (venv actief, internet nodig voor de large-v3-download):
    python scripts/diag_somali_batch_isolated.py

Pas WAV_PATH hieronder aan als de sessie-map inmiddels is opgeruimd.
"""

import wave

import numpy as np
from faster_whisper import WhisperModel

WAV_PATH = "recordings/session_15f7501c-5bff-45a3-a28a-9b9653071ab3_foreign_so_20260918T132846Z.wav"
PAZA_MODEL_DIR = "/workspace/models/paza-whisper-large-v3-turbo-ct2"
STOCK_MODEL_NAME = "large-v3"
START_MS = 0
END_MS = 30230
SAMPLE_RATE = 16000


def load_wav_slice_f32(path: str, start_ms: int, end_ms: int) -> np.ndarray:
    with wave.open(path, "rb") as rf:
        assert rf.getframerate() == SAMPLE_RATE, f"onverwachte samplerate: {rf.getframerate()}"
        start_frame = int((start_ms / 1000.0) * SAMPLE_RATE)
        end_frame = int((end_ms / 1000.0) * SAMPLE_RATE)
        rf.setpos(start_frame)
        raw = rf.readframes(end_frame - start_frame)
    audio_i16 = np.frombuffer(raw, dtype=np.int16)
    return audio_i16.astype(np.float32) / 32768.0


def run(label: str, model_ref: str, audio: np.ndarray, **transcribe_kwargs) -> None:
    print(f"\n=== {label} ===")
    print("model:", model_ref, "| transcribe_kwargs:", transcribe_kwargs)
    model = WhisperModel(model_ref, device="cuda", compute_type="float16")
    segments, info = model.transcribe(audio, language="so", **transcribe_kwargs)
    for seg in segments:
        print(
            f"  [{seg.start:6.2f}-{seg.end:6.2f}] avg_logprob={seg.avg_logprob:.3f} "
            f"compression_ratio={seg.compression_ratio:.3f} text={seg.text!r}"
        )


def main() -> None:
    audio = load_wav_slice_f32(WAV_PATH, START_MS, END_MS)
    print(f"Geladen: {len(audio) / SAMPLE_RATE:.2f}s audio uit {WAV_PATH} ({START_MS}..{END_MS}ms)")

    # Test A: Paza (PoC-kandidaat), pijplijn-instellingen -- moet 'baki nda'
    # (of vergelijkbaar sterk verminkt) reproduceren zoals in het serverlog.
    run(
        "A) Paza, huidige pijplijn-instellingen (temperature=[0.0, 0.2])",
        PAZA_MODEL_DIR,
        audio,
        beam_size=7,
        temperature=[0.0, 0.2],
        condition_on_previous_text=False,
        vad_filter=True,
        no_speech_threshold=0.9,
    )

    # Test B: Paza, faster-whisper's eigen volledige temperatuur-ladder.
    run(
        "B) Paza, volledige temperatuur-ladder ([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])",
        PAZA_MODEL_DIR,
        audio,
        beam_size=7,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        condition_on_previous_text=False,
        vad_filter=True,
        no_speech_threshold=0.9,
    )

    # Test C: stock large-v3, dezelfde pijplijn-instellingen als A -- de
    # doorslaggevende vergelijking. Downloadt zichzelf (~3GB) bij eerste run.
    run(
        "C) Stock large-v3, huidige pijplijn-instellingen (temperature=[0.0, 0.2])",
        STOCK_MODEL_NAME,
        audio,
        beam_size=7,
        temperature=[0.0, 0.2],
        condition_on_previous_text=False,
        vad_filter=True,
        no_speech_threshold=0.9,
    )

    # Test D: stock large-v3, pure faster-whisper-defaults als extra referentie.
    run("D) Stock large-v3, pure faster-whisper defaults", STOCK_MODEL_NAME, audio)


if __name__ == "__main__":
    main()
