// Trivias STT – Simple 1-channel UI with microphone selection

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

// DOM elements
const recordButton = document.getElementById("recordButton");
const liveTranscriptDiv = document.getElementById("liveTranscript");
const finalTranscriptDiv = document.getElementById("finalTranscript");
const connectionStatusSpan = document.getElementById("connectionStatus");
const micStatusSpan = document.getElementById("micStatus");
const modeStatusSpan = document.getElementById("modeStatus");
const asrStatusSpan = document.getElementById("asrStatus");
const timerSpan = document.getElementById("recordingTimer");
const hintText = document.getElementById("hintText");
const micSelect = document.getElementById("micSelect");

function initWebsocketUrl() {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host || "localhost:8000";
  // We gebruiken nu /ws (server heeft compat endpoint naar /asr)
  websocketUrl = `${proto}//${host}/ws`;
}

initWebsocketUrl();

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
    recordButton.textContent = "⏹ Stop";
    recordButton.classList.add("recording");
  } else {
    recordButton.textContent = "🎙 Start";
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
      setAsrStatus("Verbonden met STT-server, wacht op audio…");
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

      if (data.type === "ready_to_stop") {
        waitingForStop = false;
        setAsrStatus("Verwerking voltooid. Gereed voor nieuwe opname.");
        if (lastFullTranscript && finalTranscriptDiv) {
          finalTranscriptDiv.textContent = lastFullTranscript;
        }
        return;
      }

      const {
        lines = [],
        buffer_transcription = "",
        buffer_translation = "",
        status = "active_transcription",
      } = data;

      renderTranscript(lines, buffer_transcription, buffer_translation, status);
    };
  });
}

function renderTranscript(lines, bufferTranscription, bufferTranslation, status) {
  if (!liveTranscriptDiv) return;

  if (status === "no_audio_detected") {
    liveTranscriptDiv.innerHTML =
      "<em>Geen audio gedetecteerd. Probeer iets dichter bij de microfoon te spreken.</em>";
    return;
  }

  const base = (lines || [])
    .map((item) => item.text || "")
    .filter((t) => t && t.trim().length > 0)
    .join("\n")
    .trim();

  let liveText = base;
  if (bufferTranscription && bufferTranscription.trim().length > 0) {
    liveText = (liveText ? liveText + " " : "") + bufferTranscription.trim();
  }

  liveTranscriptDiv.textContent = liveText || "Nog geen tekst ontvangen…";
  lastFullTranscript = liveText || lastFullTranscript;

  if (bufferTranslation && bufferTranslation.trim().length > 0 && finalTranscriptDiv) {
    finalTranscriptDiv.textContent = bufferTranslation.trim();
  }
  setAsrStatus("Live transcriptie actief…");
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

      workletNode.port.onmessage = (e) => {
        const data = e.data;
        const ab = data instanceof ArrayBuffer ? data : data.buffer;
        recorderWorker.postMessage(
          {
            command: "record",
            buffer: ab,
          },
          [ab]
        );
      };

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
      liveTranscriptDiv.textContent = "Luisteren… spreek nu.";
    }
    setAsrStatus("Opname bezig…");
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
    setAsrStatus("Opname gestopt. Server is audio aan het afronden…");
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
    // Geen fancy permissions-API → toch devices proberen te halen
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

checkMicPermission();
updateRecordButtonUI();
updateHint();
setConnectionStatus(false);
setAsrStatus("Wachten op opname…");
