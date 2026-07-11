// Trivias STT Multi-channel UI

// === Rol- en taaldefinities ===

const ROLES = [
  { id: "employee",    label: "Medewerker",  color: "#3b82f6" },
  { id: "interpreter", label: "Tolk",        color: "#a855f7", hasLanguage2: true },
  { id: "lawyer",      label: "Advocaat",    color: "#22c55e" },
  { id: "foreign",     label: "Vreemdeling", color: "#f97316" },
  { id: "default",     label: "Spreker",     color: "#9ca3af" },
];

const LANGUAGES = [
  { code: "nl", label: "Nederlands" },
  { code: "en", label: "Engels" },
  { code: "ar", label: "Arabisch" },
  { code: "fa", label: "Farsi / Perzisch" },
  { code: "ru", label: "Russisch" },
  { code: "fr", label: "Frans" },
  { code: "de", label: "Duits" },
  { code: "tr", label: "Turks" },
  { code: "so", label: "Somalisch" },
  { code: "ti", label: "Tigrinya" },
  { code: "ku", label: "Koerdisch" },
  { code: "sr", label: "Servisch" },
  { code: "bs", label: "Bosnisch" },
];

function getRoleById(id) {
  return ROLES.find(r => r.id === id) || ROLES[ROLES.length - 1];
}

function getRoleLabel(roleId) {
  return getRoleById(roleId).label;
}

function getRoleColor(roleId) {
  return getRoleById(roleId).color;
}

// "foreign_ar" → "foreign", "interpreter" → "interpreter"
function channelIdToRoleId(channelId) {
  if (!channelId) return "default";
  if (channelId.startsWith("foreign")) return "foreign";
  return channelId;
}

// channel_id op de server: "foreign" + taal → "foreign_ar", rest ongewijzigd
function getChannelId(cfg) {
  if (cfg.roleId === "foreign") return `foreign_${cfg.language || "nl"}`;
  return cfg.roleId;
}

// === Noise gate presets ===

const GATE_PRESETS = [
  { label: "Uit",    value: 0     },
  { label: "Laag",   value: 0.005 },
  { label: "Normaal",value: 0.015 },
  { label: "Streng", value: 0.03  },
];

// === Channel config management ===

const STORAGE_KEY = "trivias_channel_config";

let channelConfigs = [];
let availableMics = [];

function loadChannelConfigs() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      channelConfigs = JSON.parse(saved);
      return;
    }
  } catch (e) { /* ignore */ }
  channelConfigs = [
    { uid: "ch_default", roleId: "employee", deviceId: "", language: "nl", language2: null, gateThreshold: 0.015 },
  ];
}

function saveChannelConfigs() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(channelConfigs));
}

function addChannelConfig() {
  channelConfigs.push({
    uid: "ch_" + Date.now() + "_" + Math.random().toString(36).slice(2, 7),
    roleId: "default",
    deviceId: "",
    language: "nl",
    language2: null,
    gateThreshold: 0.015,
  });
  saveChannelConfigs();
  renderChannelConfigs();
}

function removeChannelConfig(uid) {
  channelConfigs = channelConfigs.filter(c => c.uid !== uid);
  saveChannelConfigs();
  renderChannelConfigs();
}

// === Tab management ===

function initTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.add("hidden"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`)?.classList.remove("hidden");
    });
  });
}

// === Config tab rendering ===

function cleanMicLabel(label) {
  return (label || "")
    .replace(/^Standaard\s*-\s*/i, "")
    .replace(/^Communicatie\s*-\s*/i, "")
    .trim();
}

function buildLangOptions(selectedCode) {
  return LANGUAGES.map(l =>
    `<option value="${l.code}"${l.code === selectedCode ? " selected" : ""}>${l.label}</option>`
  ).join("");
}

function buildMicOptions(selectedDeviceId) {
  let html = `<option value=""${!selectedDeviceId ? " selected" : ""}>Systeemstandaard</option>`;
  let idx = 1;
  for (const mic of availableMics) {
    const label = cleanMicLabel(mic.label) || `Microfoon ${idx++}`;
    const sel = mic.deviceId === selectedDeviceId ? " selected" : "";
    html += `<option value="${escapeHtml(mic.deviceId)}"${sel}>${escapeHtml(label)}</option>`;
  }
  return html;
}

function renderChannelConfigs() {
  const list = document.getElementById("channelConfigList");
  if (!list) return;
  list.innerHTML = "";

  for (const cfg of channelConfigs) {
    const role = getRoleById(cfg.roleId);
    const isInterpreter = cfg.roleId === "interpreter";

    const div = document.createElement("div");
    div.className = "channel-row";
    div.dataset.uid = cfg.uid;
    div.innerHTML = `
      <div class="channel-color-bar" style="background:${role.color}"></div>
      <div class="channel-row-fields">
        <div class="channel-field">
          <label>Rol</label>
          <select class="channel-select ch-role">
            ${ROLES.map(r => `<option value="${r.id}"${r.id === cfg.roleId ? " selected" : ""}>${r.label}</option>`).join("")}
          </select>
        </div>
        <div class="channel-field">
          <label>Microfoon</label>
          <select class="channel-select ch-mic">
            ${buildMicOptions(cfg.deviceId)}
          </select>
        </div>
        <div class="channel-field">
          <label class="ch-lang-label">${isInterpreter ? "Taal 1" : "Taal"}</label>
          <select class="channel-select ch-lang">
            ${buildLangOptions(cfg.language || "nl")}
          </select>
        </div>
        <div class="channel-field ch-lang2-field${isInterpreter ? "" : " hidden"}">
          <label>Taal 2</label>
          <select class="channel-select ch-lang2">
            ${buildLangOptions(cfg.language2 || "ar")}
          </select>
        </div>
        <div class="channel-field">
          <label>Ruispoort</label>
          <select class="channel-select ch-gate">
            ${GATE_PRESETS.map(p => `<option value="${p.value}"${p.value === (cfg.gateThreshold ?? 0.015) ? " selected" : ""}>${p.label}</option>`).join("")}
          </select>
        </div>
      </div>
      <button class="ch-remove" title="Kanaal verwijderen">✕</button>
    `;

    const colorBar   = div.querySelector(".channel-color-bar");
    const roleSelect = div.querySelector(".ch-role");
    const lang2Field = div.querySelector(".ch-lang2-field");
    const langLabel  = div.querySelector(".ch-lang-label");

    roleSelect.addEventListener("change", () => {
      const c = channelConfigs.find(x => x.uid === cfg.uid);
      if (!c) return;
      c.roleId = roleSelect.value;
      const newRole = getRoleById(c.roleId);
      colorBar.style.background = newRole.color;
      const isInterp = c.roleId === "interpreter";
      lang2Field.classList.toggle("hidden", !isInterp);
      langLabel.textContent = isInterp ? "Taal 1" : "Taal";
      saveChannelConfigs();
    });

    div.querySelector(".ch-mic").addEventListener("change", e => {
      const c = channelConfigs.find(x => x.uid === cfg.uid);
      if (c) { c.deviceId = e.target.value; saveChannelConfigs(); }
    });

    div.querySelector(".ch-lang").addEventListener("change", e => {
      const c = channelConfigs.find(x => x.uid === cfg.uid);
      if (c) { c.language = e.target.value; saveChannelConfigs(); }
    });

    div.querySelector(".ch-lang2").addEventListener("change", e => {
      const c = channelConfigs.find(x => x.uid === cfg.uid);
      if (c) { c.language2 = e.target.value; saveChannelConfigs(); }
    });

    div.querySelector(".ch-gate").addEventListener("change", e => {
      const c = channelConfigs.find(x => x.uid === cfg.uid);
      if (c) { c.gateThreshold = parseFloat(e.target.value); saveChannelConfigs(); }
    });

    div.querySelector(".ch-remove").addEventListener("click", () => removeChannelConfig(cfg.uid));

    list.appendChild(div);
  }
}

// === Multi-channel transcript state ===

// Per kanaal: lines array, id→line map, parentId→sentences map, pending updates
const channelLines        = new Map(); // channelId → line[]
const channelLineById     = new Map(); // channelId → Map(id → line)
const channelSentenceMap  = new Map(); // channelId → Map(parentId → sentence[])
const channelPendingUpd   = new Map(); // channelId → Map(id → update)

// Voor sessie-terugluister (single-channel pad)
let playbackLines = [];
let playbackLineById = new Map();
let playbackSentenceMap = new Map();
let playbackChannelId = "default";
let isPlaybackMode = false;

// === Opname state ===

// uid → { ws, channelId, audioContext, mediaStream, workletNode, recorderWorker, mediaRecorder, watchdog }
const activeConnections = new Map();

let currentSessionId  = null;
let isRecording       = false;
let serverUseAudioWorklet = true;
let startTime         = null;
let timerInterval     = null;
let lastBufferTranscription = "";
let lastBufferTranslation   = "";
let lastStatus              = "active_transcription";

// === DOM refs ===

const recordButton       = document.getElementById("recordButton");
const liveTranscriptDiv  = document.getElementById("liveTranscript");
if (liveTranscriptDiv) liveTranscriptDiv.style.whiteSpace = "pre-wrap";
const connectionStatusSpan = document.getElementById("connectionStatus");
const modeStatusSpan       = document.getElementById("modeStatus");
const asrStatusSpan        = document.getElementById("asrStatus");
const timerSpan            = document.getElementById("recordingTimer");
const hintText             = document.getElementById("hintText");

const sessionsBtn        = document.getElementById("sessionsBtn");
const sessionsModal      = document.getElementById("sessionsModal");
const sessionsModalClose = document.getElementById("sessionsModalClose");
const sessionsList       = document.getElementById("sessionsList");

// === Status UI helpers ===

function setConnectionStatus(connectedCount, totalCount) {
  if (!connectionStatusSpan) return;
  if (connectedCount === 0) {
    connectionStatusSpan.textContent = "Niet verbonden";
    connectionStatusSpan.className = "status-value status-disconnected";
  } else if (connectedCount < totalCount) {
    connectionStatusSpan.textContent = `${connectedCount}/${totalCount} verbonden`;
    connectionStatusSpan.className = "status-value status-recording";
  } else {
    connectionStatusSpan.textContent = totalCount === 1 ? "Verbonden" : `${connectedCount} kanalen verbonden`;
    connectionStatusSpan.className = "status-value status-connected";
  }
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
    recordButton.textContent = "🎙 Start";
    recordButton.classList.remove("recording");
  }
}

function updateHint() {
  if (!hintText) return;
  if (!isRecording) {
    hintText.innerHTML = "Stel kanalen in via <strong>Configuratie</strong>, dan klik <strong>Start</strong>.";
  } else {
    hintText.textContent = "Opname loopt. Spreek in de microfoon(s).";
  }
}

// === Timer ===

function startTimer() {
  if (!timerSpan) return;
  startTime = Date.now();
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    const s = Math.max(0, Math.floor((Date.now() - startTime) / 1000));
    const mm = String(Math.floor(s / 60)).padStart(2, "0");
    const ss = String(s % 60).padStart(2, "0");
    timerSpan.textContent = `${mm}:${ss}`;
  }, 1000);
}

function resetTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
  if (timerSpan) timerSpan.textContent = "00:00";
}

// === Microfoonlijst ===

async function refreshMicrophoneList() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const audioInputs = devices.filter(d => d.kind === "audioinput");
    const byKey = new Map();
    for (const dev of audioInputs) {
      const key = dev.groupId || cleanMicLabel(dev.label) || dev.deviceId;
      if (!byKey.has(key)) byKey.set(key, dev);
    }
    availableMics = Array.from(byKey.values());
    renderChannelConfigs();
  } catch (e) {
    console.warn("Cannot enumerate audio devices:", e);
  }
}

// === Per-kanaal WebSocket + audio ===

function buildWebSocketUrl(sessionId, channelId, cfg) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const params = new URLSearchParams({ session_id: sessionId, channel_id: channelId, lang: cfg.language || "nl" });
  if (cfg.language2) params.set("lang2", cfg.language2);
  return `${proto}//${location.host}/ws?${params}`;
}

async function openAudioStream(ws, cfg, useWorklet) {
  const audioConstraints = cfg.deviceId
    ? { deviceId: { exact: cfg.deviceId } }
    : true;

  const stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });

  const audioContext = new (window.AudioContext || window.webkitAudioContext)();

  let workletNode = null;
  let recorderWorker = null;
  let mediaRecorder = null;
  let watchdog = null;

  if (useWorklet) {
    await audioContext.audioWorklet.addModule("/web/pcm_worklet.js");
    const source = audioContext.createMediaStreamSource(stream);
    workletNode = new AudioWorkletNode(audioContext, "pcm-worklet-processor", {
      numberOfInputs: 1, numberOfOutputs: 0, channelCount: 1,
    });
    source.connect(workletNode);

    recorderWorker = new Worker("/web/recorder_worker.js");
    recorderWorker.postMessage({ command: "init", config: { sampleRate: audioContext.sampleRate } });
    recorderWorker.onmessage = e => {
      if (ws.readyState === WebSocket.OPEN) ws.send(e.data.buffer);
    };

    // Noise gate instellen op de worklet
    workletNode.port.postMessage({ threshold: cfg.gateThreshold ?? 0 });

    let lastActivity = Date.now();
    workletNode.port.onmessage = e => {
      lastActivity = Date.now();
      const ab = e.data instanceof ArrayBuffer ? e.data : e.data.buffer;
      recorderWorker.postMessage({ command: "record", buffer: ab }, [ab.slice(0)]);
    };

    watchdog = setInterval(() => {
      if (isRecording && Date.now() - lastActivity > 5000) {
        setAsrStatus("Geen audio — controleer microfoon");
      }
    }, 3000);

  } else {
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) ws.send(e.data);
    };
    mediaRecorder.start(250);
  }

  return { audioContext, mediaStream: stream, workletNode, recorderWorker, mediaRecorder, watchdog };
}

function cleanupChannelConnection(uid) {
  const conn = activeConnections.get(uid);
  if (!conn) return;

  clearInterval(conn.watchdog);

  if (conn.recorderWorker) {
    try { conn.recorderWorker.terminate(); } catch (e) {}
  }
  if (conn.workletNode) {
    try { conn.workletNode.port.onmessage = null; conn.workletNode.disconnect(); } catch (e) {}
  }
  if (conn.mediaRecorder && conn.mediaRecorder.state !== "inactive") {
    try { conn.mediaRecorder.stop(); } catch (e) {}
  }
  if (conn.mediaStream) {
    try { conn.mediaStream.getTracks().forEach(t => t.stop()); } catch (e) {}
  }
  if (conn.audioContext) {
    try { conn.audioContext.close(); } catch (e) {}
  }

  activeConnections.delete(uid);
  updateConnectionStatus();
}

function updateConnectionStatus() {
  const total = channelConfigs.length;
  const connected = activeConnections.size;
  setConnectionStatus(connected, total);
}

async function startChannelConnection(cfg, sessionId) {
  const channelId = getChannelId(cfg);
  const wsUrl = buildWebSocketUrl(sessionId, channelId, cfg);
  const ws = new WebSocket(wsUrl);

  // Wacht op config-bericht van server
  const configData = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(`WebSocket timeout (${channelId})`)), 8000);
    ws.onerror = () => { clearTimeout(timeout); reject(new Error(`WebSocket fout (${channelId})`)); };
    ws.onclose = () => { clearTimeout(timeout); reject(new Error(`WebSocket gesloten (${channelId})`)); };
    ws.onmessage = e => {
      let data;
      try { data = JSON.parse(e.data); } catch { return; }
      if (data.type === "config") { clearTimeout(timeout); resolve(data); }
    };
  });

  // Sla audio-modus op (van eerste kanaal)
  if (activeConnections.size === 0) {
    serverUseAudioWorklet = !!configData.useAudioWorklet;
    setModeStatus(serverUseAudioWorklet ? "AudioWorklet (PCM)" : "MediaRecorder (WebM)");
  }

  // Transcript-messagehandler instellen
  ws.onmessage = e => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; }
    handleChannelMessage(data, channelId);
  };

  ws.onclose = () => handleChannelClose(cfg.uid, channelId);

  // Audio openen
  const audioResources = await openAudioStream(ws, cfg, serverUseAudioWorklet);

  activeConnections.set(cfg.uid, { ws, channelId, ...audioResources });
  updateConnectionStatus();
}

function handleChannelClose(uid, channelId) {
  cleanupChannelConnection(uid);
  if (isRecording) {
    setAsrStatus(`Kanaal ${channelId} verbroken`);
  }
}

// === Opname start / stop ===

async function startRecording() {
  if (isRecording) return;
  if (channelConfigs.length === 0) {
    alert("Configureer eerst minstens één kanaal via het tabblad Configuratie.");
    return;
  }

  isPlaybackMode = false;
  currentSessionId = crypto.randomUUID();

  // Reset transcript state voor alle kanalen
  channelLines.clear();
  channelLineById.clear();
  channelSentenceMap.clear();
  channelPendingUpd.clear();

  if (liveTranscriptDiv) liveTranscriptDiv.innerHTML = "Verbinding maken…";

  const errors = [];
  await Promise.all(channelConfigs.map(async cfg => {
    try {
      await startChannelConnection(cfg, currentSessionId);
    } catch (e) {
      errors.push(e.message);
      console.error("Kanaal mislukt:", cfg.roleId, e);
    }
  }));

  if (activeConnections.size === 0) {
    alert("Geen enkel kanaal kon verbinden:\n" + errors.join("\n"));
    setConnectionStatus(0, channelConfigs.length);
    return;
  }

  if (errors.length > 0) {
    setAsrStatus(`${errors.length} kanaal/kanalen niet verbonden`);
  }

  isRecording = true;
  updateRecordButtonUI();
  updateHint();
  startTimer();
  setAsrStatus("Live transcriptie actief");
}

async function stopRecording() {
  if (!isRecording) return;
  isRecording = false;

  resetTimer();
  updateRecordButtonUI();
  updateHint();
  setAsrStatus("Opname gestopt. Server rondt af…");

  for (const [uid, conn] of activeConnections.entries()) {
    if (conn.ws?.readyState === WebSocket.OPEN) {
      try {
        const emptyBlob = new Blob([], { type: "audio/webm" });
        conn.ws.send(emptyBlob);
      } catch (e) {}
    }
    cleanupChannelConnection(uid);
  }

  setConnectionStatus(0, channelConfigs.length);
}

function toggleRecording() {
  if (isRecording) stopRecording(); else startRecording();
}

// === Per-kanaal berichtverwerking ===

function handleChannelMessage(data, channelId) {
  if (data.type === "segment_update") {
    handleSegmentUpdate(data, channelId);
  } else if (data.type === "front_data" || data.lines) {
    handleFrontData(data, channelId);
  }
}

function ensureChannelMaps(channelId) {
  if (!channelLines.has(channelId))       channelLines.set(channelId, []);
  if (!channelLineById.has(channelId))    channelLineById.set(channelId, new Map());
  if (!channelSentenceMap.has(channelId)) channelSentenceMap.set(channelId, new Map());
  if (!channelPendingUpd.has(channelId))  channelPendingUpd.set(channelId, new Map());
}

function handleFrontData(data, channelId) {
  const { lines = [], buffer_transcription = "", buffer_translation = "", status = "active_transcription" } = data;

  ensureChannelMaps(channelId);
  channelLines.set(channelId, lines);

  const lbid = new Map();
  for (const l of lines) { if (l?.id) lbid.set(l.id, l); }
  channelLineById.set(channelId, lbid);

  // Verwerk uitgestelde segment_updates
  const pending = channelPendingUpd.get(channelId);
  for (const [pid, upd] of pending.entries()) {
    const l = lbid.get(pid);
    if (!l) continue;
    if (upd.text_batch  !== undefined) l.text_batch = upd.text_batch;
    if (upd.text_final  !== undefined) l.text = upd.text_final;
    if (upd.state       !== undefined) l.state = upd.state;
    pending.delete(pid);
  }

  lastBufferTranscription = buffer_transcription;
  lastBufferTranslation   = buffer_translation;
  lastStatus              = status;

  renderAllChannels();
}

function handleSegmentUpdate(data, channelId) {
  const id = data.id;
  if (!id) return;

  ensureChannelMaps(channelId);
  const lbid    = channelLineById.get(channelId);
  const sentMap = channelSentenceMap.get(channelId);
  const lines   = channelLines.get(channelId);

  let line = lbid.get(id);

  if (!line) {
    if (data.text_final || data.text_batch) {
      const sentMatch = id.match(/^(.+)_s(\d+)$/);
      if (sentMatch) {
        const parentId = sentMatch[1];
        if (!sentMap.has(parentId)) {
          sentMap.set(parentId, []);
          if (!lbid.has(parentId)) {
            const parent = { id: parentId, text: "", text_batch: null, state: "FINAL", start_ms: data.start_ms || 0, end_ms: data.end_ms || 0, speaker: -1, channelId };
            lines.push(parent);
            lbid.set(parentId, parent);
          }
        }
        sentMap.get(parentId).push({ id, text: data.text_final || data.text_batch || "", text_batch: data.text_batch || null, state: data.state || "FINAL", start_ms: data.start_ms || 0, end_ms: data.end_ms || 0, speaker: -1, channelId });
      } else {
        const newLine = { id, text: data.text_final || data.text_batch || "", text_batch: data.text_batch || null, state: data.state || "FINAL", start_ms: data.start_ms || 0, end_ms: data.end_ms || 0, speaker: -1, channelId };
        lines.push(newLine);
        lbid.set(id, newLine);
      }
      renderAllChannels();
    } else {
      channelPendingUpd.get(channelId).set(id, data);
    }
    return;
  }

  if (data.text_batch  !== undefined) line.text_batch = data.text_batch;
  if (data.text_final  !== undefined) line.text = data.text_final;
  if (data.state       !== undefined) {
    if (data.start_ms != null) line.start_ms = data.start_ms;
    if (data.end_ms   != null) line.end_ms   = data.end_ms;
    line.state = data.state;
  }

  renderAllChannels();
}

// === Gecombineerde transcript render ===

function getAllRenderLines() {
  const all = [];
  for (const [channelId, lines] of channelLines.entries()) {
    const sentMap = channelSentenceMap.get(channelId) || new Map();
    const expanded = lines.map(l => {
      const sents = sentMap.get(l.id);
      return sents ? sents : [l];
    }).flat();
    for (const l of expanded) all.push({ ...l, channelId });
  }
  return all;
}

function renderAllChannels() {
  if (isPlaybackMode) return;
  renderTranscript(getAllRenderLines(), lastBufferTranscription, lastBufferTranslation, lastStatus);
}

// === Transcript render (zowel live als terugluister) ===

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
  if (Number.isFinite(item?.start))    return Math.round(item.start * 1000);
  return Number.MAX_SAFE_INTEGER;
}

function formatMs(ms) {
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  return h > 0
    ? `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`
    : `${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`;
}

function renderTranscript(lines, bufferTranscription, bufferTranslation, status) {
  if (!liveTranscriptDiv) return;

  const scrollParent = liveTranscriptDiv;
  const isAtBottom = scrollParent.scrollHeight - scrollParent.scrollTop - scrollParent.clientHeight < 80;

  if (status === "no_audio_detected") {
    liveTranscriptDiv.innerHTML = "<em>Geen audio gedetecteerd. Probeer dichter bij de microfoon.</em>";
    return;
  }

  const safeLines = (lines || []).filter(item => {
    const sp = item?.speaker ?? item?.speaker_id ?? item?.spk;
    return sp !== -2;
  });

  safeLines.sort((a, b) => getStartMs(a) - getStartMs(b));

  const htmlParts = [];

  for (const item of safeLines) {
    const rawTxt = (item?.text_batch || item?.text || "").trim();
    if (!rawTxt) continue;

    const sp = item?.speaker ?? item?.speaker_id ?? item?.spk;
    const st = (item?.state || "FINAL").toUpperCase();

    let cls = "seg";
    if (st === "LIVE") {
      cls += " seg-live";
    } else if (item.text_batch) {
      cls += " seg-batch";
    } else {
      cls += " seg-final";
    }

    const prefix = (sp === undefined || sp === null || sp === "" || sp === -1)
      ? "" : `[${escapeHtml(sp)}] `;

    const idAttr    = item?.id ? ` data-id="${escapeHtml(item.id)}"` : "";
    const startMs   = getStartMs(item);
    const endMs     = Number.isFinite(item?.end_ms) ? item.end_ms : Number.isFinite(item?.end) ? Math.round(item.end * 1000) : 0;

    // channelId: uit line zelf (multi-channel live) of uit playback context
    const channelId = item.channelId || playbackChannelId || "default";
    const sessionId = currentSessionId || "";

    const audioAttr = startMs >= 0
      ? ` data-start-ms="${startMs}" data-end-ms="${endMs}" data-session="${escapeHtml(sessionId)}" data-channel="${escapeHtml(channelId)}"`
      : "";

    const timeLabel = (startMs > 0 || item?.start_ms === 0)
      ? `<span class="seg-time">[${formatMs(startMs)}]</span> ` : "";

    const roleId    = channelIdToRoleId(channelId);
    const roleColor = getRoleColor(roleId);
    const roleLabel = `<span class="seg-role" style="color:${roleColor}">${escapeHtml(getRoleLabel(roleId))}</span> `;

    htmlParts.push(`<div class="${cls} seg-clickable"${idAttr}${audioAttr}>${timeLabel}${roleLabel}${prefix}${escapeHtml(rawTxt)}</div>`);
  }

  const hasLiveContent = (lines || []).some(item => item?.text && !item?.text_batch && item?.speaker !== -2);
  if (hasLiveContent || (bufferTranscription && bufferTranscription.trim())) {
    htmlParts.push(`<div class="live-indicator"><span class="live-dot"></span> Spreekt…</div>`);
  }

  liveTranscriptDiv.innerHTML = htmlParts.join("") || "Nog geen tekst ontvangen";

  if (isAtBottom) scrollParent.scrollTop = scrollParent.scrollHeight;

  setAsrStatus("Live transcriptie actief");
}

// === Sessie browser ===

async function loadSessionsList() {
  if (!sessionsList) return;
  sessionsList.innerHTML = '<p class="sessions-loading">Laden…</p>';
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
      const item = document.createElement("div");
      item.className = "session-item";
      item.innerHTML = `
        <div class="session-item-meta">
          <span class="session-item-id">${s.session_id.substring(0, 18)}…</span>
          <span class="session-item-date">📅 ${date} · 🎙 ${s.channels.join(", ")} · ${s.wav_size_mb} MB</span>
        </div>
        <span class="session-item-badge ${s.has_transcript ? "" : "no-transcript"}">
          ${s.has_transcript ? "transcript ✓" : "geen transcript"}
        </span>`;
      if (s.has_transcript) {
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
    if (!resp.ok) { alert("Transcript niet gevonden."); return; }
    const data = await resp.json();

    sessionsModal.classList.add("hidden");

    currentSessionId   = data.session_id;
    playbackChannelId  = data.channel_id || channelId;
    isPlaybackMode     = true;

    playbackLines       = [];
    playbackLineById    = new Map();
    playbackSentenceMap = new Map();

    for (const seg of (data.segments || [])) {
      const id = seg.id;
      if (!id) continue;
      const sentMatch = id.match(/^(.+)_s(\d+)$/);
      if (sentMatch) {
        const parentId = sentMatch[1];
        if (!playbackSentenceMap.has(parentId)) {
          playbackSentenceMap.set(parentId, []);
          if (!playbackLineById.has(parentId)) {
            const parent = { id: parentId, text: "", text_batch: null, state: "FINAL", start_ms: seg.start_ms || 0, end_ms: seg.end_ms || 0, speaker: -1 };
            playbackLines.push(parent);
            playbackLineById.set(parentId, parent);
          }
        }
        playbackSentenceMap.get(parentId).push({
          id, text: seg.text_final || seg.text_batch || "", text_batch: seg.text_batch || null,
          state: "FINAL", start_ms: seg.start_ms || 0, end_ms: seg.end_ms || 0, speaker: -1,
        });
      } else {
        const line = { id, text: seg.text_final || seg.text_batch || "", text_batch: seg.text_batch || null, state: "FINAL", start_ms: seg.start_ms || 0, end_ms: seg.end_ms || 0, speaker: -1 };
        playbackLines.push(line);
        playbackLineById.set(id, line);
      }
    }

    for (const [pid, sents] of playbackSentenceMap.entries()) {
      sents.sort((a, b) => a.start_ms - b.start_ms);
    }

    const renderLines = playbackLines.map(l => {
      const sents = playbackSentenceMap.get(l.id);
      return sents ? sents : [l];
    }).flat();

    if (liveTranscriptDiv) renderTranscript(renderLines, "", "", "active_transcription");
    setAsrStatus(`Sessie geladen: ${sessionId.substring(0, 12)}…`);
  } catch (e) {
    alert("Fout bij laden transcript: " + e.message);
  }
}

// === Klik op segment → terugluisteren ===

if (liveTranscriptDiv) {
  liveTranscriptDiv.addEventListener("click", async e => {
    const seg = e.target.closest(".seg-clickable");
    if (!seg) return;
    const startMs = parseInt(seg.dataset.startMs || "0", 10);
    const endMs   = parseInt(seg.dataset.endMs   || "0", 10);
    const session = seg.dataset.session;
    const channel = seg.dataset.channel;
    if (!session) return;

    const prev = document.getElementById("trivias-audio-player");
    if (prev) prev.remove();
    const audio = document.createElement("audio");
    audio.id = "trivias-audio-player";
    audio.controls = true;
    audio.autoplay = true;
    audio.src = `/audio/${encodeURIComponent(session)}/${encodeURIComponent(channel)}?start_ms=${startMs}&end_ms=${endMs}`;
    audio.style.cssText = "position:fixed;bottom:20px;right:20px;z-index:9999;background:#1e293b;border-radius:8px;";
    document.body.appendChild(audio);
  });
}

// === Event wiring ===

if (recordButton) recordButton.addEventListener("click", toggleRecording);

if (sessionsBtn) sessionsBtn.addEventListener("click", () => {
  sessionsModal.classList.remove("hidden");
  loadSessionsList();
});
if (sessionsModalClose) sessionsModalClose.addEventListener("click", () => sessionsModal.classList.add("hidden"));
if (sessionsModal) sessionsModal.addEventListener("click", e => { if (e.target === sessionsModal) sessionsModal.classList.add("hidden"); });

const addChannelBtn = document.getElementById("addChannelBtn");
if (addChannelBtn) addChannelBtn.addEventListener("click", addChannelConfig);

// === Init ===

function initWebsocketUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host || "localhost:8000"}/ws`;
}

async function checkMicPermission() {
  if (!navigator.permissions?.query) {
    refreshMicrophoneList();
    return;
  }
  try {
    const perm = await navigator.permissions.query({ name: "microphone" });
    if (perm.state === "granted") refreshMicrophoneList();
    perm.onchange = () => { if (perm.state === "granted") refreshMicrophoneList(); };
  } catch {
    refreshMicrophoneList();
  }
}

loadChannelConfigs();
initTabs();
renderChannelConfigs();
checkMicPermission();
updateRecordButtonUI();
updateHint();
setConnectionStatus(0, channelConfigs.length);
setAsrStatus("Wachten op opname");
