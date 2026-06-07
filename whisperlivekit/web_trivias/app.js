// Trivias STT Simple 1-channel UI with microphone selection

let currentLines = [];
let lineById = new Map();
let lastBufferTranscription = "";
let lastBufferTranslation = "";
let lastStatus = "active_transcription";

let websocket = null;
let websocketUrl = null;

let isRecording = false;
let serverUseAudioWorklet = true;

let audioContext = null;
let microphone = null;
let workletNode = null;
let recorderWorker = null;
let mediaRecorder = null;
let mediaStream = null;

let startTime = null;
let timerInterval = null;
let lastFullTranscript = "";

let configResolve = null;
let waitingForStop = false;
let userClosing = false;

// NEW: microphone selection state
let availableMics = [];
let selectedDeviceId = null;

let pendingSegmentUpdates = new Map(); // id -> last update payload
let currentSessionId = null;
let currentChannelId = "default";

// DOM elements
const recordButton = document.getElementById("recordButton");
const liveTranscriptDiv = document.getElementById("liveTranscript");
if (liveTranscriptDiv) liveTranscriptDiv.style.whiteSpace = "pre-wrap";

const finalTranscriptDiv = document.getElementById("finalTranscript");
const connectionStatusSpan = document.getElementById("connectionStatus");
const micStatusSpan = document.getElementById("micStatus");
const modeStatusSpan = document.getElementById("modeStatus");
const asrStatusSpan = document.getElementById("asrStatus");
const timerSpan = document.getElementById("recordingTimer");
const hintText = document.getElementById("hintText");
const micSelect = document.getElementById("micSelect");
// Zin-segmenten: overschrijven batch groups na front_data render
const sentenceSegmentMap = new Map(); // parentId → [zinnen]

// === Sessie browser ===
const sessionsBtn = document.getElementById("sessionsBtn");
const sessionsModal = document.getElementById("sessionsModal");
const sessionsModalClose = document.getElementById("sessionsModalClose");
const sessionsList = document.getElementById("sessionsList");

async function loadSessionsList() {
  if (!sessionsList) return;
  sessionsList.innerHTML = '<p class="sessions-loading">Laden...</p>';
  try {
    const resp = await fetch("/sessions/list");
    const data = await resp.json();
    if (!data.sessions || data.sessions.length === 0) {
      sessionsList.innerHTML = '<p class="sessions-loading">Geen sessies gevonden.</p>';
      return;
    }
    sessionsList.innerHTML = "";
    for (const s of data.sessions) {
      const date = s.created_at 
        ? s.created_at.replace(/(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z/, "$3-$2-$1 $4:$5")
        : "onbekend";
      const hasTranscript = s.has_transcript;
      const item = document.createElement("div");
      item.className = "session-item";
      item.innerHTML = `
        <div class="session-item-meta">
          <span class="session-item-id">${s.session_id.substring(0, 18)}…</span>
          <span class="session-item-date">📅 ${date} · 🎙 ${s.channels.join(", ")} · ${s.wav_size_mb} MB</span>
        </div>
        <span class="session-item-badge ${hasTranscript ? "" : "no-transcript"}">
          ${hasTranscript ? "transcript ✓" : "geen transcript"}
        </span>`;
      if (hasTranscript) {
        item.addEventListener("click", () => loadSessionTranscript(s.session_id, s.channels[0] || "default"));
      }
      sessionsList.appendChild(item);
    }
  } catch (e) {
    sessionsList.innerHTML = '<p class="sessions-loading">Fout bij laden sessies.</p>';
  }
}

async function loadSessionTranscript(sessionId, channelId) {
  try {
    const resp = await fetch(`/sessions/${encodeURIComponent(sessionId)}/transcript?channel_id=${encodeURIComponent(channelId)}`);
    if (!resp.ok) {
      alert("Transcript niet gevonden.");
      return;
    }
    const data = await resp.json();
    // Sluit modal
    sessionsModal.classList.add("hidden");
    // Laad in UI
    currentSessionId = data.session_id;
    currentChannelId = data.channel_id;
    currentLines = [];
    lineById = new Map();
    sentenceSegmentMap.clear();
    // Verwerk segments als segment_updates
    for (const seg of (data.segments || [])) {
      const id = seg.id;
      if (!id) continue;
      const sentMatch = id.match(/^(.+)_s(\d+)$/);
      if (sentMatch) {
        const parentId = sentMatch[1];
        if (!sentenceSegmentMap.has(parentId)) sentenceSegmentMap.set(parentId, []);
        sentenceSegmentMap.get(parentId).push({
          id, text: seg.text_final || seg.text_batch || "",
          text_batch: seg.text_batch || null,
          state: "FINAL",
          start_ms: seg.start_ms || 0,
          end_ms: seg.end_ms || 0,
          speaker: -1,
        });
      } else {
        const line = {
          id, text: seg.text_final || seg.text_batch || "",
          text_batch: seg.text_batch || null,
          state: "FINAL",
          start_ms: seg.start_ms || 0,
          end_ms: seg.end_ms || 0,
          speaker: -1,
        };
        currentLines.push(line);
        lineById.set(id, line);
      }
    }
    // Sorteer sentenceSegmentMap entries
    for (const [pid, sents] of sentenceSegmentMap.entries()) {
      sents.sort((a, b) => a.start_ms - b.start_ms);
    }
    const renderLines = currentLines.map(l => {
      const sents = sentenceSegmentMap.get(l.id);
      return sents ? sents : [l];
    }).flat();
    if (liveTranscriptDiv) {
      renderTranscript(renderLines, "", "", "active_transcription");
    }
    setAsrStatus(`Sessie geladen: ${sessionId.substring(0, 12)}…`);
  } catch (e) {
    alert("Fout bij laden transcript: " + e.message);
  }
}

if (sessionsBtn) {
  sessionsBtn.addEventListener("click", () => {
    sessionsModal.classList.remove("hidden");
    loadSessionsList();
  });
}

if (sessionsModalClose) {
  sessionsModalClose.addEventListener("click", () => {
    sessionsModal.classList.add("hidden");
  });
}

if (sessionsModal) {
  sessionsModal.addEventListener("click", (e) => {
    if (e.target === sessionsModal) sessionsModal.classList.add("hidden");
  });
}
// Rol mapping: channel_id → leesbare naam
const CHANNEL_ROLE_LABELS = {
  "employee":    "Medewerker",
  "interpreter": "Tolk",
  "lawyer":      "Advocaat",
  "foreign_nl":  "Vreemdeling",
  "foreign_ar":  "Vreemdeling",
  "foreign_fa":  "Vreemdeling",
  "foreign_ru":  "Vreemdeling",
  "foreign_en":  "Vreemdeling",
  "default":     "Spreker",
};

function getRoleLabel(channelId) {
  return CHANNEL_ROLE_LABELS[channelId] || "Spreker";
}

function initWebsocketUrl() {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host || "localhost:8000";
  // We gebruiken nu /ws (server heeft compat endpoint naar /asr)
  websocketUrl = `${proto}//${host}/ws`;
}

initWebsocketUrl();

function formatMs(ms) {
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  return h > 0
    ? `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`
    : `${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`;
}

function setConnectionStatus(connected) {
  if (!connectionStatusSpan) return;
  if (connected) {
    connectionStatusSpan.textContent = "Verbonden";
    connectionStatusSpan.classList.remove("status-disconnected");
    connectionStatusSpan.classList.add("status-connected");
  } else {
    connectionStatusSpan.textContent = "Niet verbonden";
    connectionStatusSpan.classList.remove("status-connected");
    connectionStatusSpan.classList.add("status-disconnected");
  }
}

function rebuildLineIndex(lines) {
  lineById = new Map();
  (lines || []).forEach((l) => {
    if (l && l.id) lineById.set(l.id, l);
  });
}

function setMicStatus(text) {
  if (micStatusSpan) micStatusSpan.textContent = text;
}

function setModeStatus(text) {
  if (modeStatusSpan) modeStatusSpan.textContent = text;
}

function setAsrStatus(text) {
  if (asrStatusSpan) asrStatusSpan.textContent = text;
}

function updateRecordButtonUI() {
  if (!recordButton) return;
  if (isRecording) {
    recordButton.textContent = "Stop";
    recordButton.classList.add("recording");
  } else {
    recordButton.textContent = "­Start";
    recordButton.classList.remove("recording");
  }
}

function updateHint() {
  if (!hintText) return;
  if (!isRecording) {
    hintText.innerHTML =
      "Klik op <strong>Start</strong> om de microfoon te activeren en audio naar de server te sturen.";
  } else {
    hintText.textContent = "Spreek rustig in. De tekst verschijnt hier live.";
  }
}

function formatTime(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const mm = String(Math.floor(s / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

function startTimer() {
  if (!timerSpan) return;
  startTime = Date.now();
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    const elapsed = (Date.now() - startTime) / 1000;
    timerSpan.textContent = formatTime(elapsed);
  }, 1000);
}

function resetTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
  if (timerSpan) {
    timerSpan.textContent = "00:00";
  }
}

function ensureWebSocket() {
  if (websocket && websocket.readyState === WebSocket.OPEN) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    configResolve = resolve;
    try {
      websocket = new WebSocket(websocketUrl);
    } catch (e) {
      console.error("Cannot create WebSocket:", e);
      setConnectionStatus(false);
      setAsrStatus("WebSocket-verbinding mislukt.");
      return reject(e);
    }

    websocket.onopen = () => {
      setConnectionStatus(true);
      setAsrStatus("Verbonden met STT-server, wacht op audio");
    };

    websocket.onerror = (err) => {
      console.error("WebSocket error:", err);
      setConnectionStatus(false);
      setAsrStatus("WebSocket-fout, controleer server.");
      if (configResolve) {
        const r = configResolve;
        configResolve = null;
        r(); // alsnog resolve om niet te blijven hangen
      }
    };

    websocket.onclose = () => {
      setConnectionStatus(false);
      if (isRecording) {
        isRecording = false;
        updateRecordButtonUI();
        updateHint();
      }
      websocket = null;
    };

    websocket.onmessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (e) {
        console.warn("Non-JSON message:", event.data);
        return;
      }

      /* =========================
      * CONFIG (ongewijzigd)
      * ========================= */
      if (data.type === "config") {
        serverUseAudioWorklet = !!data.useAudioWorklet;
        const modeText = serverUseAudioWorklet
          ? "AudioWorklet (PCM)"
          : "MediaRecorder (WebM)";
        setModeStatus(modeText);

        if (configResolve) {
          const r = configResolve;
          configResolve = null;
          r();
        }
        return;
      }

      /* =========================
      * SEGMENT UPDATE (NIEUW)
      * ========================= */
      if (data.type === "segment_update") {
        const id = data.id;
        if (!id) return;

        let line = lineById.get(id);
        if (!line) {
          if (data.text_final || data.text_batch) {
            const sentMatch = id.match(/^(.+)_s(\d+)$/);
            if (sentMatch) {
              const parentId = sentMatch[1];
              if (!sentenceSegmentMap.has(parentId)) {
                sentenceSegmentMap.set(parentId, []);
              }
              sentenceSegmentMap.get(parentId).push({
                id: id,
                text: data.text_final || data.text_batch || "",
                text_batch: data.text_batch || null,
                state: data.state || "FINAL",
                start_ms: data.start_ms || 0,
                end_ms: data.end_ms || 0,
                speaker: -1,
              });
            } else {
              const newLine = {
                id: id,
                text: data.text_final || data.text_batch || "",
                text_batch: data.text_batch || null,
                state: data.state || "FINAL",
                start_ms: data.start_ms || 0,
                end_ms: data.end_ms || 0,
                speaker: -1,
              };
              currentLines.push(newLine);
              lineById.set(id, newLine);
            }
            // Render met zin-segmenten
            let renderLines = currentLines.map(l => {
              const sents = sentenceSegmentMap.get(l.id);
              return sents ? sents : [l];
            }).flat();
            renderTranscript(renderLines, lastBufferTranscription, lastBufferTranslation, lastStatus);

          } else {
            pendingSegmentUpdates.set(id, data);
          }
          return;
        }

        if (data.text_batch !== undefined) {
          line.text_batch = data.text_batch;
        }

        if (data.text_final !== undefined) {
          // Dit is de tekst die we daadwerkelijk tonen
          line.text = data.text_final;
        }

        if (data.state !== undefined) {
          if (data.start_ms !== undefined && data.start_ms !== null) {
            line.start_ms = data.start_ms;
          }
          if (data.end_ms !== undefined && data.end_ms !== null) {
            line.end_ms = data.end_ms;
          }
          line.state = data.state;
        }

        renderTranscript(
          currentLines,
          lastBufferTranscription,
          lastBufferTranslation,
          lastStatus
        );
        return;
      }

      /* =========================
      * FRONT DATA (volledige refresh)
      * ========================= */
      if (data.type === "front_data" || data.lines) {
        const {
          lines = [],
          buffer_transcription = "",
          buffer_translation = "",
          status = "active_transcription",
        } = data;

        currentLines = lines;
        rebuildLineIndex(currentLines);

        if (data.session_id) currentSessionId = data.session_id;
        if (data.channel_id) currentChannelId = data.channel_id || "default";

        // Apply any pending segment updates now that lines exist
        for (const [pid, upd] of pendingSegmentUpdates.entries()) {
          const l = lineById.get(pid);
          if (!l) continue;

          if (upd.text_batch !== undefined) l.text_batch = upd.text_batch;
          if (upd.text_final !== undefined) l.text = upd.text_final;
          if (upd.state !== undefined) l.state = upd.state;

          pendingSegmentUpdates.delete(pid);
        }

        lastBufferTranscription = buffer_transcription;
        lastBufferTranslation = buffer_translation;
        lastStatus = status;

        // Vervang batch groups door zin-segmenten indien beschikbaar
        let renderLines = currentLines.map(l => {
          const sents = sentenceSegmentMap.get(l.id);
          return sents ? sents : [l];
        }).flat();
        
        renderTranscript(renderLines, lastBufferTranscription, lastBufferTranslation, lastStatus);

        return;
      }

      /* =========================
      * FALLBACK / UNKNOWN
      * ========================= */
      console.debug("Unhandled WS message:", data);
    };

  });
}

let liveIndicatorTimeout = null;


function escapeHtml(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function getStartMs(item) {
  if (Number.isFinite(item?.start_ms)) return item.start_ms;
  if (Number.isFinite(item?.start)) return Math.round(item.start * 1000);
  return Number.MAX_SAFE_INTEGER; // live/buffer always last
}

function renderTranscript(lines, bufferTranscription, bufferTranslation, status) {
  if (!liveTranscriptDiv) return;

  const scrollParent = liveTranscriptDiv;
  const isAtBottom = (scrollParent.scrollHeight - scrollParent.scrollTop - scrollParent.clientHeight < 80);

  if (status === "no_audio_detected") {
      liveTranscriptDiv.innerHTML =
      "<em>Geen audio gedetecteerd. Probeer iets dichter bij de microfoon te spreken.</em>";
      return;
  }

  const safeLines = (lines || []).filter((item) => {
    const sp = item?.speaker ?? item?.speaker_id ?? item?.spk;
    // -2 = silence segment → niet tonen
    return sp !== -2;
  });

  const htmlParts = [];

  safeLines.sort((a, b) => getStartMs(a) - getStartMs(b));

  for (const item of safeLines) {

    console.log("[RENDER]", item.id, item.state, "text:", item.text, "text_batch:", item.text_batch);
    const rawTxt = (item?.text_batch || item?.text || "").trim();
    if (!rawTxt) continue;
    // Alleen tonen als batch de tekst heeft goedgekeurd
    if (!item?.text_batch) continue;

    const sp = item?.speaker ?? item?.speaker_id ?? item?.spk;
    const st = (item?.state || "FINAL").toUpperCase();

    let cls = "seg";

    if (st === "LIVE") {
      cls += " seg-live";
    } else if (st === "FINAL") {
      if (item.text_batch && item.text === item.text_batch) {
        cls += " seg-batch";   // echte batch overwrite
      } else {
        cls += " seg-final";   // live-final
      }
    } else {
      cls += " seg-final";
    }

    const prefix =
      sp === undefined || sp === null || sp === "" || sp === -1
        ? ""
        : `[${escapeHtml(sp)}] `;

    const idAttr = item?.id ? ` data-id="${escapeHtml(item.id)}"` : "";

    const startMs = getStartMs(item);
    const endMs = Number.isFinite(item?.end_ms) ? item.end_ms 
                  : Number.isFinite(item?.end) ? Math.round(item.end * 1000) 
                  : 0;
    const sessionId = currentSessionId || "";
    const channelId = currentChannelId || "default";
    
    const audioAttr = startMs >= 0 
      ? ` data-start-ms="${startMs}" data-end-ms="${endMs}" data-session="${escapeHtml(sessionId)}" data-channel="${escapeHtml(channelId)}"` 
      : "";


    const timeLabel = startMs > 0 || item?.start_ms === 0
      ? `<span class="seg-time">[${formatMs(startMs)}]</span> `
      : "";
    const roleLabel = `<span class="seg-role">${escapeHtml(getRoleLabel(currentChannelId))}</span> `;

    htmlParts.push(
      `<div class="${cls} seg-clickable"${idAttr}${audioAttr}>${timeLabel}${roleLabel}${prefix}${escapeHtml(rawTxt)}</div>`
    );
  }

  // Pulserende indicator als er live segmenten zijn zonder batch
  const hasLiveContent = (lines || []).some(
    item => item?.text && !item?.text_batch && item?.speaker !== -2
  );
  if (hasLiveContent || (bufferTranscription && bufferTranscription.trim().length > 0)) {
    htmlParts.push(`<div class="live-indicator"><span class="live-dot"></span> Spreekt...</div>`);
  }

  liveTranscriptDiv.innerHTML =
    htmlParts.join("") || "Nog geen tekst ontvangen";

  if (isAtBottom) {
    scrollParent.scrollTop = scrollParent.scrollHeight;
  }

  setAsrStatus("Live transcriptie actief");
}


// NEW: microfoonlijst ophalen en dropdown vullen (met dedupe)
async function refreshMicrophoneList() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();

    // Alleen audioinput
    const audioInputs = devices.filter((d) => d.kind === "audioinput");

    // Helper om label op te schonen
    const baseLabel = (label) =>
      (label || "")
        .replace(/^Standaard\s*-\s*/i, "")
        .replace(/^Communicatie\s*-\s*/i, "")
        .trim();

    // Dedupe: 1 per groupId / baselabel
    const byKey = new Map();
    for (const dev of audioInputs) {
      const key = dev.groupId || baseLabel(dev.label) || dev.deviceId;
      if (!byKey.has(key)) {
        byKey.set(key, dev);
      }
    }
    availableMics = Array.from(byKey.values());

    if (!micSelect) return;

    const previous = micSelect.value;
    micSelect.innerHTML = "";

    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "Systeemstandaard";
    micSelect.appendChild(defaultOption);

    let idx = 1;
    for (const mic of availableMics) {
      const opt = document.createElement("option");
      opt.value = mic.deviceId;
      opt.textContent = baseLabel(mic.label) || `Microfoon ${idx++}`;
      micSelect.appendChild(opt);
    }

    if (previous && [...micSelect.options].some((o) => o.value === previous)) {
      micSelect.value = previous;
      selectedDeviceId = previous || null;
    } else {
      selectedDeviceId = micSelect.value || null;
    }
  } catch (e) {
    console.warn("Cannot enumerate audio devices:", e);
  }
}


async function startRecording() {
  if (isRecording) return;

  try {
    await ensureWebSocket();
  } catch (e) {
    console.error("Cannot start recording, WebSocket not ready:", e);
    return;
  }

  try {
    const audioConstraints = selectedDeviceId
      ? { deviceId: { exact: selectedDeviceId } }
      : true;

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: audioConstraints,
    });
    mediaStream = stream;
    setMicStatus("Toegang verleend");

    // Na succesvolle toegang: devices verversen (labels worden nu zichtbaar)
    refreshMicrophoneList();

    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    const useWorklet = serverUseAudioWorklet && !!audioContext.audioWorklet;

    if (useWorklet) {
      await audioContext.audioWorklet.addModule("/web/pcm_worklet.js");
      const source = audioContext.createMediaStreamSource(stream);
      workletNode = new AudioWorkletNode(
        audioContext,
        "pcm-worklet-processor",
        {
          numberOfInputs: 1,
          numberOfOutputs: 0,
          channelCount: 1,
        }
      );
      source.connect(workletNode);

      recorderWorker = new Worker("/web/recorder_worker.js");
      recorderWorker.postMessage({
        command: "init",
        config: { sampleRate: audioContext.sampleRate },
      });

      recorderWorker.onmessage = (e) => {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
          websocket.send(e.data.buffer);
        }
      };

      let lastAudioActivityMs = Date.now();
      const AUDIO_WATCHDOG_INTERVAL_MS = 3000;  // check elke 3s
      const AUDIO_SILENCE_WARN_MS = 5000;       // waarschuw na 5s geen audio

      workletNode.port.onmessage = (e) => {
          lastAudioActivityMs = Date.now();
          const data = e.data;
          const ab = data instanceof ArrayBuffer ? data : data.buffer;
          recorderWorker.postMessage(
              { command: "record", buffer: ab },
              [ab]
          );
      };

      // Watchdog: detecteer als worklet stopt met sturen
      const audioWatchdog = setInterval(() => {
          if (!isRecording) {
              clearInterval(audioWatchdog);
              return;
          }
          const silenceDuration = Date.now() - lastAudioActivityMs;
          if (silenceDuration > AUDIO_SILENCE_WARN_MS) {
              console.warn(`[WATCHDOG] Geen audio van worklet sinds ${silenceDuration}ms`);
              setAsrStatus(`⚠️ Geen audio ontvangen sinds ${Math.round(silenceDuration/1000)}s — controleer microfoon`);
          }
      }, AUDIO_WATCHDOG_INTERVAL_MS);

      setModeStatus("AudioWorklet (PCM)");
    } else {
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (e) => {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
          if (e.data && e.data.size > 0) {
            websocket.send(e.data);
          }
        }
      };
      mediaRecorder.start(250);
      setModeStatus("MediaRecorder (WebM)");
    }

    if (liveTranscriptDiv) {
      liveTranscriptDiv.textContent = "Luisteren spreek nu.";
    }
    setAsrStatus("Opname bezig");
    isRecording = true;
    userClosing = false;
    waitingForStop = false;
    updateRecordButtonUI();
    updateHint();
    startTimer();
  } catch (err) {
    console.error("Error starting recording:", err);
    setMicStatus("Toegang geweigerd of fout");
    setAsrStatus("Kon microfoon niet gebruiken. Controleer permissies of apparaat.");
  }
}

function cleanupAudio() {
  if (mediaRecorder) {
    try {
      mediaRecorder.stop();
    } catch (e) {}
    mediaRecorder = null;
  }

  if (recorderWorker) {
    try {
      recorderWorker.terminate();
    } catch (e) {}
    recorderWorker = null;
  }

  if (workletNode) {
    try {
      workletNode.port.onmessage = null;
    } catch (e) {}
    try {
      workletNode.disconnect();
    } catch (e) {}
    workletNode = null;
  }

  if (mediaStream) {
    try {
      mediaStream.getTracks().forEach((t) => t.stop());
    } catch (e) {}
    mediaStream = null;
  }
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  userClosing = true;
  waitingForStop = true;

  cleanupAudio();
  resetTimer();
  updateRecordButtonUI();
  updateHint();

  if (websocket && websocket.readyState === WebSocket.OPEN) {
    const emptyBlob = new Blob([], { type: "audio/webm" });
    websocket.send(emptyBlob);
    setAsrStatus("Opname gestopt. Server is audio aan het afronden");
  } else {
    setAsrStatus("Opname gestopt.");
  }
}

function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}

// Permissions & device handling
async function checkMicPermission() {
  if (!navigator.permissions || !navigator.permissions.query) {
    // Geen fancy permissions-API  toch devices proberen te halen
    refreshMicrophoneList();
    return;
  }
  try {
    const perm = await navigator.permissions.query({ name: "microphone" });
    setMicStatus(perm.state.toUpperCase());

    if (perm.state === "granted") {
      refreshMicrophoneList();
    }

    perm.onchange = () => {
      setMicStatus(perm.state.toUpperCase());
      if (perm.state === "granted") {
        refreshMicrophoneList();
      }
    };
  } catch {
    // Fallback: gewoon proberen
    refreshMicrophoneList();
  }
}

// Event wiring
if (recordButton) {
  recordButton.addEventListener("click", () => {
    toggleRecording();
  });
}

// NEW: change handler voor micSelect
if (micSelect) {
  micSelect.addEventListener("change", () => {
    selectedDeviceId = micSelect.value || null;
    if (isRecording) {
      setAsrStatus(
        "Nieuwe microfoon wordt gebruikt na stoppen en opnieuw starten."
      );
    }
  });
}

// Klik op segment → terugluisteren
liveTranscriptDiv.addEventListener("click", async (e) => {
  const seg = e.target.closest(".seg-clickable");
  if (!seg) return;

  const startMs = parseInt(seg.dataset.startMs || "0", 10);
  const endMs = parseInt(seg.dataset.endMs || "0", 10);
  const session = seg.dataset.session;
  const channel = seg.dataset.channel;

  if (!session) return;

  const url = `/audio/${encodeURIComponent(session)}/${encodeURIComponent(channel)}?start_ms=${startMs}&end_ms=${endMs}`;

  // Verwijder vorige audio player
  const prev = document.getElementById("trivias-audio-player");
  if (prev) prev.remove();

  const audio = document.createElement("audio");
  audio.id = "trivias-audio-player";
  audio.controls = true;
  audio.autoplay = true;
  audio.src = url;
  audio.style.cssText = "position:fixed;bottom:20px;right:20px;z-index:9999;background:#1e293b;border-radius:8px;";
  document.body.appendChild(audio);
});

checkMicPermission();
updateRecordButtonUI();
updateHint();
setConnectionStatus(false);
setAsrStatus("Wachten op opname");
