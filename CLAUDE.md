# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**WhisperLiveKit-Trivias** is a multi-channel, ultra-low-latency speech-to-text server tailored for courtroom/legal proceedings. It is a fork of WhisperLiveKit that adds session persistence, channel-aware transcription configuration (per-role quality presets), and a "Trivias" web interface with transcript playback.

## Design Principles

These principles are non-negotiable constraints that must be respected in every change to this codebase.

**Audio is authoritative; text is derived.**
The WAV recording is the ground truth. Transcripts are a derivative product. Nothing in the code should prioritize text convenience over audio integrity. If text and audio diverge, the audio wins.

**Everything is time-based.**
Every token, segment, speaker label, and event carries a precise timestamp anchored to the start of the audio stream. Timestamps are never approximate or decorative — they are the primary key that ties all layers (audio, transcript, diarization) together.

**On-premises only.**
No audio, transcript, or session data leaves the machine. There are no external API calls for transcription. The `openai-api` backend is the only exception and must never be the default. Cloud integrations of any kind require explicit opt-in and clear user acknowledgement.

**Provability: event\_id + timestamp + audio fragment.**
Every legally relevant output event must carry a stable `event_id`, a wall-clock + stream-relative timestamp, and a reference to the audio byte range that produced it. These three together make a claim independently verifiable and re-derivable from the recording.

**Live transcription is situational awareness.**
The live (AlignAtt) pass optimizes for latency. Its output is for real-time understanding only — it is provisional and may be revised by the batch pass. Do not design features that treat live output as final or store it as the record of truth.

**Batch transcription is the legally citable source.**
The batch (FasterWhisper) pass is the authoritative transcript. It runs with higher beam width and more context and is the only output that should be quoted in legal proceedings, stored as the official transcript, or surfaced as "final" in the UI. Features that blur this distinction — e.g. mixing live and batch segments in the citation export — are design defects.

## Commands

**Install from source:**
```bash
pip install -e .
```

**Run the server:**
```bash
# Basic
wlk --model base --language en

# Typical Trivias (Dutch, with diarization)
wlk --model small --language nl --diarization --diarization-backend sortformer --host 0.0.0.0 --port 8000

# With translation
wlk --model base --language fr --target-language en

# Docker (GPU)
docker build -t wlk .
docker run --gpus all -p 8000:8000 wlk --model large-v3
```

**Access the UI:** `http://localhost:8000` after starting the server.

**Development scripts:**
- `scripts/determine_alignment_heads.py` — extract alignment heads for custom models
- `scripts/convert_hf_whisper.py` — convert HuggingFace Whisper format to WLK

There is no test suite in this repository.

## Architecture

The pipeline flows: **Browser (WebSocket) → FastAPI Server → AudioProcessor → [VAD + Live ASR + Batch ASR + Diarization] → JSON output over WebSocket + WAV/JSON files on disk.**

### Key files

| File | Role |
|------|------|
| `whisperlivekit/TriviasServer.py` | FastAPI server; WebSocket endpoints (`/asr`, `/ws`), session REST endpoints (`/sessions/*`, `/audio/*`), session metadata tracking |
| `whisperlivekit/audio_processor.py` | Central orchestrator (~1500 lines); buffers incoming audio bytes, coordinates VAD, live ASR, batch ASR, diarization, translation, and writes WAV + JSON recordings |
| `whisperlivekit/core.py` | `TranscriptionEngine` — loads models once globally; initializes ASR and diarization backends |
| `whisperlivekit/simul_whisper/` | **AlignAtt simultaneous streaming backend** (default, SOTA 2025). `simul_whisper.py` = streaming decoder, `backend.py` = `SimulStreamingASR` + `BatchFasterWhisperASR`, `config.py` = per-channel config, `decoder_state.py` = per-session state |
| `whisperlivekit/local_agreement/` | **LocalAgreement WhisperStreaming backend** (older SOTA, fallback) |
| `whisperlivekit/diarization/` | `sortformer_backend.py` (recommended, SOTA 2025), `diart_backend.py` (legacy) |
| `whisperlivekit/silero_vad_iterator.py` | Voice Activity Detection (Silero ONNX) |
| `whisperlivekit/cross_channel_gate.py` | Non-causal, server-side cross-channel acoustic-leak suppression used by `refresh_transcript()` (see "Refresh transcript" below) — envelope cross-correlation for alignment + per-frame RMS arbitration, pure numpy/scipy |
| `whisperlivekit/web_trivias/` | Vanilla JS frontend — `app.js` handles WebSocket, session playback, UI; `recorder_worker.js` = Web Worker for recording; `pcm_worklet.js` = AudioWorklet for raw PCM |

### Dual-pass transcription

Each audio session runs **two parallel ASR passes**:
1. **Live pass** (AlignAtt / SimulStreaming) — low-latency token-by-token output
2. **Batch pass** (FasterWhisper) — periodic high-accuracy refinement over a sliding window

Results are merged before transmission. The tradeoff is tuned by `--frame-threshold` (lower = faster, less accurate) and `--batch-beam-size`.

### Multi-channel / Trivias-specific

`simul_whisper/config.py` defines `ChannelTranscriptionConfig` — per-channel overrides for language, task (transcribe/translate), frame threshold, beam size, etc. Predefined channels include `"default"`, `"interpreter"`, `"lawyer"`, `"employee"`. A channel is identified in the WebSocket handshake and selects its config without reloading models.

### Session persistence

`TriviasServer.py` tracks each connection as a session (UUID). Audio is recorded to `recordings/<session_id>/<channel>.wav` and transcripts to `recordings/<session_id>/transcript.json`. REST endpoints (`/sessions/list`, `/sessions/{id}/transcript`, `/audio/{id}/{channel}`) expose these for playback in the UI.

### Refresh transcript ("Ververs Transcriptie")

`POST /sessions/{id}/refresh_transcript` (`TriviasServer.py`) discards the incrementally-built transcript for every channel of a session and rebuilds it from scratch by feeding the full recorded WAV to `BatchFasterWhisperASR.transcribe_full()` (`simul_whisper/backend.py`) in one call — no live-decoder state (`state.end_buffer`, `cumulative_time_offset`, pause resets) is involved, so this path is immune to the whole class of incremental-pipeline drift bugs. Each faster-whisper segment is gated individually via the shared `evaluate_batch_segment()` (also used by the incremental `_batch_worker()`), then `<wav_stem>.json` is overwritten wholesale. Callable only once a session is stable (Stop, or Pause after sending the `flag=3` WS control frame that flushes the WAV writer). Frontend button: `#refreshButton` (Bediening panel, visible during Pause) and `#refreshPlaybackButton` (session playback view, visible after Stop).

Because this path reads the raw WAVs directly, it bypasses the live client-side cross-channel anti-leak gate (per-chunk RMS arbitration in `web_trivias/app.js`, see below), which only ever suppresses samples in the ASR-facing copy during live streaming and is never persisted. For sessions with 2+ channels, `refresh_transcript()` therefore also runs a non-causal, server-side equivalent (`whisperlivekit/cross_channel_gate.py`, `compute_cross_channel_gate_masks()`): channels are aligned via envelope cross-correlation (no shared sub-second start timestamp exists on disk, so timing offset between independently-started channel WebSockets is measured from the audio itself) and arbitrated per-frame, always failing safe to no suppression when alignment confidence is too low. As with the live gate, only the in-memory copy passed to `transcribe_full()` is ever masked — the WAV on disk is untouched.

### Model sharing

Models are loaded once into `TranscriptionEngine` (singleton per server process). Each WebSocket connection gets its own `DecoderState` (session-scoped beam state, buffers, timing) but shares the underlying model weights.

## Important CLI Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--model` | `base` | tiny / small / medium / large-v3 |
| `--language` | `auto` | ISO 639-1 code or `auto` |
| `--frame-threshold` | `25` | AlignAtt sensitivity; lower = lower latency |
| `--audio-max-len` | `30.0` | Live buffer cap in seconds |
| `--backend-policy` | `simulstreaming` | `simulstreaming` or `local_agreement` |
| `--backend` | `auto` | `faster-whisper` / `mlx-whisper` / `whisper` / `openai-api` |
| `--diarization` | off | Enable speaker identification |
| `--diarization-backend` | `sortformer` | `sortformer` (2025) or `diart` (legacy) |
| `--pcm-input` | off | Raw PCM (s16le) instead of WebM |
| `--no-vad` / `--no-vac` | off | Disable voice detection / controller |
| `--llm-backend-url` | none | Base URL of an on-prem OpenAI-compatible LLM endpoint (e.g. Ollama at `http://localhost:11434/v1`), used for gehoorverslag section classification. Never a cloud endpoint. Unset = feature runs in fail-safe fallback (no classification). |
| `--llm-model` | none | Model name as known to the LLM endpoint (e.g. `llama3.1:8b`). |
| `--llm-api-key` | none | API key for the LLM endpoint, if required (most local runtimes don't need one). |

## Docs

- `docs/FO.md` — **Functioneel Ontwerp** — authoritative functional specification for this project; read this before making feature decisions
- `docs/API.md` — WebSocket protocol and JSON message schema
- `docs/technical_integration.md` — embedding WLK without FastAPI
- `docs/default_and_custom_models.md` — model selection and quantization
- `DEV_NOTES.md` — benchmarks and tuning notes
