"""Eenmalig diagnose-scriptje, GEEN onderdeel van de reguliere pijplijn.

Doel: isoleren of de repetitie-hallucinatie op het foreign_so-batchvenster
(zie sessie 82228426-bd69-4bf3-8476-b33188d1792d, job 52460128, ms 30470..66330)
komt door het geconverteerde model zelf, of door dit project's verkorte
temperatuur-fallback-ladder ([0.0, 0.2] i.p.v. faster-whisper's eigen default
[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], zie simul_whisper/config.py:51).

Roept faster_whisper.WhisperModel rechtstreeks aan, buiten BatchFasterWhisperASR
en de rest van de pijplijn (geen gate, geen ChannelTranscriptionConfig) om de
twee variabelen te scheiden.

Gebruik op de RunPod-pod (venv actief):
    python scripts/diag_somali_batch_isolated.py

Pas WAV_PATH hieronder aan als de sessie-map inmiddels is opgeruimd.
"""

import wave

import numpy as np
from faster_whisper import WhisperModel

WAV_PATH = "recordings/session_82228426-bd69-4bf3-8476-b33188d1792d_foreign_so_20260918T124114Z.wav"
MODEL_DIR = "/workspace/models/paza-whisper-large-v3-turbo-ct2"
START_MS = 30470
END_MS = 66330
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


def run(label: str, audio: np.ndarray, **transcribe_kwargs) -> None:
    print(f"\n=== {label} ===")
    print("transcribe_kwargs:", transcribe_kwargs)
    model = WhisperModel(MODEL_DIR, device="cuda", compute_type="float16")
    segments, info = model.transcribe(audio, language="so", **transcribe_kwargs)
    for seg in segments:
        print(
            f"  [{seg.start:6.2f}-{seg.end:6.2f}] avg_logprob={seg.avg_logprob:.3f} "
            f"compression_ratio={seg.compression_ratio:.3f} text={seg.text!r}"
        )


def main() -> None:
    audio = load_wav_slice_f32(WAV_PATH, START_MS, END_MS)
    print(f"Geladen: {len(audio) / SAMPLE_RATE:.2f}s audio uit {WAV_PATH} ({START_MS}..{END_MS}ms)")

    # Test 1: exact zoals BatchFasterWhisperASR dit vandaag aanroept (verkorte
    # temperatuur-ladder [0.0, 0.2]) -- moet dezelfde garbage reproduceren als
    # in het serverlog (job=52460128: 'baki yake yake yake...', compression_ratio=23.9).
    run(
        "1) Huidige pijplijn-instellingen (temperature=[0.0, 0.2])",
        audio,
        beam_size=7,
        temperature=[0.0, 0.2],
        condition_on_previous_text=False,
        vad_filter=True,
        no_speech_threshold=0.9,
    )

    # Test 2: faster-whisper's eigen volledige default temperatuur-ladder,
    # verder identieke instellingen -- isoleert of de langere ladder dit venster redt.
    run(
        "2) Volledige temperatuur-ladder ([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])",
        audio,
        beam_size=7,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        condition_on_previous_text=False,
        vad_filter=True,
        no_speech_threshold=0.9,
    )

    # Test 3: faster-whisper pure defaults (niets van ons overridden) als baseline.
    run("3) faster-whisper pure defaults", audio)


if __name__ == "__main__":
    main()
