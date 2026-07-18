# WhisperLiveKit WebSocket API Documentation

> !! **Note**: The new API structure described in this document is currently under deployment. 
This documentation is intended for devs who want to build custom frontends.

WLK provides real-time speech transcription, speaker diarization, and translation through a WebSocket API. The server sends incremental updates as audio is processed, allowing clients to display live transcription results with minimal latency.

---

## Incoming Audio (Client → Server)

### Default contract (unchanged)

Binary WebSocket messages are raw audio bytes, sent as-is to the server. With `--pcm-input`, this is raw s16le PCM (16kHz, mono, no framing, no headers) — see `docs/technical_integration.md`. This is the contract used by `/asr` and by any `/ws` client that does not opt in to the framing below. **This default is never affected by anything in this section.**

### Opt-in: `gate_framed=1` (cross-channel anti-leak, PCM input only)

Pass `?gate_framed=1` as a query parameter on `/ws` (or `/asr`) to enable per-chunk gate framing. Each binary message becomes:

```
[1 byte: 0x00 = ASR-closed | 0x01 = ASR-open][s16le PCM payload]
```

The flag indicates whether this fragment of audio should be fed into VAD/ASR. **It never affects what gets written to the session WAV file — the server always records 100% of the received audio, regardless of the flag**, per this project's audio-is-authoritative principle (see root `CLAUDE.md`). Gate-closed samples are zeroed out only in the copy used for VAD/live/batch transcription; the WAV recording and all timestamps are computed from the untouched, full-fidelity audio.

This is used by the bundled Trivias frontend (`whisperlivekit/web_trivias/`) to suppress transcription on a channel when it detects (via cross-channel RMS comparison, done client-side since all channels of a session share one browser tab) that its microphone is likely picking up acoustic leakage from another channel's speaker rather than its own. It is not part of the general-purpose WLK protocol and is safe to ignore for any other integration — omitting `gate_framed` (the default) preserves the raw, unframed contract described above.

---

## Legacy API (Current)

### Message Structure

The current API sends complete state snapshots on each update (several time per second)

```typescript
{
  "type": str,
  "status": str,
  "lines": [
    {
      "speaker": int,
      "text": str,
      "start": float,
      "end": float,
      "translation": str | null,
      "detected_language": str
    }
  ],
  "buffer_transcription": str,
  "buffer_diarization": str,
  "remaining_time_transcription": float,
  "remaining_time_diarization": float
}
```

---

## New API (Under Development)

### Philosophy

Principles:

- **Incremental Updates**: Only updates and new segments are sent
- **Ephemeral Buffers**: Temporary, unvalidated data displayed in real-time but overwritten on next update, at speaker level


## Message Format


```typescript
{
  "type": "transcript_update",
  "status": "active_transcription" | "no_audio_detected",
  "segments": [
    {
      "id": number,
      "speaker": number,
      "text": string,
      "start_speaker": float,
      "start": float,
      "end": float,
      "language": string | null,
      "translation": string,
      "words": [
        {
          "text": string,
          "start": float,
          "end": float,
          "validated": {
            "text": boolean,
            "speaker": boolean,
          }
        }
      ],
      "buffer": {
        "transcription": string,
        "diarization": string,
        "translation": string
      }
    }
  ],
  "metadata": {
    "remaining_time_transcription": float,
    "remaining_time_diarization": float
  }
}
```

### Other Message Types

#### Config Message (sent on connection)
```json
{
  "type": "config",
  "useAudioWorklet": true / false
}
```

#### Ready to Stop Message (sent after processing complete)
```json
{
  "type": "ready_to_stop"
}
```

---

## Field Descriptions

### Segment Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `number` | Unique identifier for this segment. Used by clients to update specific segments efficiently. |
| `speaker` | `number` | Speaker ID (1, 2, 3...). Special value `-2` indicates silence. |
| `text` | `string` | Validated transcription text for this update. Should be **appended** to the segment's text on the client side.  |
| `start_speaker` | `float` | Timestamp (seconds) when this speaker segment began. |
| `start` | `float` | Timestamp (seconds) of the first word in this update. |
| `end` | `float` | Timestamp (seconds) of the last word in this update. |
| `language` | `string \| null` | ISO language code (e.g., "en", "fr"). `null` until language is detected. |
| `translation` | `string` | Validated translation text for this update. Should be **appended** to the segment's translation on the client side. |
| `words` | `Array` | Array of word-level objects with timing and validation information. |
| `buffer` | `Object` | Per-segment temporary buffers, see below |

### Word Object

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | The word text. |
| `start` | `number` | Start timestamp (seconds) of this word. |
| `end` | `number` | End timestamp (seconds) of this word. |
| `validated.text` | `boolean` | Whether the transcription text has been validated. if false, word is also in buffer: transcription |
| `validated.speaker` | `boolean` | Whether the speaker assignment has been validated. if false, word is also in buffer: diarization |
| `validated.language` | `boolean` | Whether the language detection has been validated. if false, word is also in buffer: translation |

### Buffer Object (Per-Segment)

Buffers are **ephemeral**. They should be displayed to the user but not stored permanently in the frontend. Each update may contain a completely different buffer value, and previous buffer is likely to be in the next validated text.

| Field | Type | Description |
|-------|------|-------------|
| `transcription` | `string` | Pending transcription text. Displayed immediately but **overwritten** on next update. |
| `diarization` | `string` | Pending diarization text (text waiting for speaker assignment). Displayed immediately but **overwritten** on next update. |
| `translation` | `string` | Pending translation text. Displayed immediately but **overwritten** on next update. |


### Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `remaining_time_transcription` | `float` | Seconds of audio waiting for transcription processing. |
| `remaining_time_diarization` | `float` | Seconds of audio waiting for speaker diarization. |

### Status Values

| Status | Description |
|--------|-------------|
| `active_transcription` | Normal operation, transcription is active. |
| `no_audio_detected` | No audio has been detected yet. |

---

## Update Behavior

### Incremental Updates

The API sends **only changed or new segments**. Clients should:

1. Maintain a local map of segments by ID
2. When receiving an update, merge/update segments by ID
3. Render only the changed segments

### Language Detection

When language is detected for a segment:

```jsonc
// Update 1: No language yet
{
  "segments": [
    {"id": 1, "speaker": 1, "text": "May see", "language": null}
  ]
}

// Update 2: Same segment ID, language now detected
{
  "segments": [
    {"id": 1, "speaker": 1, "text": "Merci", "language": "fr"}
  ]
}
```

**Client behavior**: **Replace** the existing segment with the same ID.

### Buffer Behavior

Buffers are **per-segment** to handle multi-speaker scenarios correctly.

#### Example: Translation with diarization and translation

```jsonc
// Update 1
{
  "segments": [
    {
      "id": 1,
      "speaker": 1,
      "text": "Hello world, how are",
      "translation": "",
      "buffer": {
        "transcription": "",
        "diarization": " you on",
        "translation": "Bonjour le monde"
      }
    }
  ]
}


// ==== Frontend ====
// <SPEAKER>1</SPEAKER>
// <TRANSCRIPTION>Hello world, how are <DIARIZATION BUFFER> you on</DIARIZATION BUFFER></TRANSCRIPTION>
// <TRANSLATION><TRANSLATION BUFFER>Bonjour le monde</TRANSLATION BUFFER></TRANSLATION>


// Update 2
{
  "segments": [
    {
      "id": 1,
      "speaker": 1,
      "text": " you on this",
      "translation": "Bonjour tout le monde",
      "buffer": {
        "transcription": "",
        "diarization": " beautiful day",
        "translation": ",comment"
      }
    },
  ]
}


// ==== Frontend ====
// <SPEAKER>1</SPEAKER>
// <TRANSCRIPTION>Hello world, how are you on this<DIARIZATION BUFFER>  beautiful day</DIARIZATION BUFFER></TRANSCRIPTION>
// <TRANSLATION>Bonjour tout le monde<TRANSLATION BUFFER>, comment</TRANSLATION BUFFER><TRANSLATION>
```

### Silence Segments

Silence is represented with the speaker id = `-2`:

```jsonc
{
  "id": 5,
  "speaker": -2,
  "text": "",
  "start": 10.5,
  "end": 12.3
}
```
