// Trivias STT Multi-channel UI

// === Rol- en taaldefinities ===

// Kleuren wijzen naar de --role-*-variabelen in style.css (:root) i.p.v. hier
// eigen hexwaarden te hardcoden -- vroeger stonden hier fellere, ongerelateerde
// tinten los van die CSS-variabelen (die zelf nergens gebruikt werden). Nu is
// style.css:root de enige plek om het rolpalet ooit nog bij te stellen; elke
// plek die getRoleColor()/role.color gebruikt (rollabels, kanaalbadges,
// live-meter, playback-tijdlijn, filterchips) volgt automatisch mee.
const ROLES = [
  { id: "employee",    label: "Medewerker",  color: "var(--role-employee)" },
  { id: "interpreter", label: "Tolk",        color: "var(--role-interpreter)", hasLanguage2: true },
  { id: "lawyer",      label: "Advocaat",    color: "var(--role-lawyer)" },
  { id: "foreign",     label: "Vreemdeling", color: "var(--role-foreign)" },
  { id: "default",     label: "Spreker",     color: "var(--role-default)" },
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

// Genummerde, gekleurde kanaalbadge -- gedeeld tussen de Configuratie-kaarten
// en de live meter-chips, zodat "kanaal N" overal hetzelfde nummer+kleur
// draagt (i.p.v. alleen een losse kleurbalk zonder nummer).
function channelBadgeHTML(index, role) {
  return `<span class="channel-badge" style="background:${role.color}">${index}</span>`;
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

// === Cross-kanaal anti-lek arbitrage ===
// Onderdrukt ASR op een kanaal wanneer een ander kanaal duidelijk luider is
// (waarschijnlijk akoestisch lek van diens spreker in deze microfoon).
// Raakt nooit de WAV-opname — die krijgt altijd de volledige, ongewijzigde audio.

const CROSS_GATE_STORAGE_KEY = "trivias_cross_gate_enabled";
const ARBITRATION_MARGIN = 0.4;  // eigen RMS moet >= 40% van de luidste andere kanaal zijn
const CLOSE_HOLD_MS      = 100;  // verdenking moet dit lang aanhouden vóór ASR-onderdrukking
const STALE_MS           = 200;  // negeer peers waarvan we >200ms niets meer hoorden

let crossGateEnabled = true;

// uid → { rms, gateOpen1, lastUpdate, suspectSince, combinedGateOpen }
const channelAudioState = new Map();

function loadCrossGateEnabled() {
  try {
    const saved = localStorage.getItem(CROSS_GATE_STORAGE_KEY);
    if (saved !== null) { crossGateEnabled = saved === "1"; return; }
  } catch (e) { /* ignore */ }
  crossGateEnabled = true;
}

function saveCrossGateEnabled() {
  localStorage.setItem(CROSS_GATE_STORAGE_KEY, crossGateEnabled ? "1" : "0");
}

// Combineert de eigen stilte-gate (stage 1) met cross-kanaal arbitrage (stage 2).
// Bij twijfel (geen duidelijk dominant ander kanaal) blijft dit kanaal altijd open.
//
// LET OP: eerder stond hier ook een afkoelperiode vóór heropenen (REOPEN_HOLD_MS),
// bedoeld om flikkeren te voorkomen. Die bleek in de praktijk een echte bug te
// bevatten: de hersteltimer werd bij ELKE korte "verdacht"-meting gereset, dus bij
// RMS-niveaus die rond de arbitragemarge schommelen kon een kanaal permanent dicht
// blijven hangen -- precies het omgekeerde van "bij twijfel altijd open". Dat
// voelde voor de gebruiker aan als een kanaal dat het gewoon niet meer deed.
// Teruggezet naar direct heropenen; alleen CLOSE_HOLD_MS (sneller dicht bij
// verdenking) blijft staan, die heeft dit risico niet.
//
// VEILIGHEIDSNET (2026-07-19): elk kanaal beoordeelt zijn eigen RMS tegen een
// momentopname van de ander (tot STALE_MS oud), onafhankelijk van elkaar. Bij
// snel fluctuerende, dicht bij elkaar liggende volumes (bv. een zin die
// uitdooft) kunnen beide kanalen elkaar -- op net iets andere momenten,
// tegen elkaars stale snapshot -- als "verdacht" bestempelen en allebei
// dichtgaan. Gezien in een echte sessie: 11+ seconden lang waren BEIDE
// kanalen tegelijk onderdrukt, waardoor een echt gesproken zin nergens meer
// binnenkwam. Daarom hieronder een expliciete garantie: als geen enkel ander
// actief kanaal op dit moment open staat voor ASR, mag dit kanaal ook nooit
// dicht -- er blijft altijd minstens één kanaal open.
function computeCombinedGate(uid, state) {
  if (!state.gateOpen1) return false; // eigen stilte-gate heeft altijd voorrang
  if (!crossGateEnabled) return true;

  const now = Date.now();
  let maxPeerRms = 0;
  let anyPeerCombinedOpen = false;
  for (const [otherUid, peer] of channelAudioState.entries()) {
    if (otherUid === uid) continue;
    if (!peer.gateOpen1) continue;
    if (now - peer.lastUpdate > STALE_MS) continue;
    if (peer.rms > maxPeerRms) maxPeerRms = peer.rms;
    if (peer.combinedGateOpen) anyPeerCombinedOpen = true;
  }

  const suspected = maxPeerRms > 0 && state.rms < maxPeerRms * ARBITRATION_MARGIN;

  let open;
  if (!suspected) {
    state.suspectSince = null;
    open = true;
  } else {
    if (state.suspectSince == null) state.suspectSince = now;
    open = (now - state.suspectSince) < CLOSE_HOLD_MS;
  }

  if (!open && !anyPeerCombinedOpen) {
    // Geen enkel ander kanaal staat momenteel open -- dit kanaal alsnog
    // sluiten zou alle kanalen tegelijk laten dichtvallen. Nooit doen.
    open = true;
    state.suspectSince = null;
  }

  state.combinedGateOpen = open;
  return open;
}

// === Channel config management ===

const STORAGE_KEY = "trivias_channel_config";

let channelConfigs = [];
let availableMics = [];

// === Sessie-brede referenties (zaaknummer, cliëntreferentie) ===
// Bewust GEEN localStorage (i.t.t. channelConfigs hierboven, dat legitieme
// apparaatinstellingen bevat): op een gedeeld werkstation mag een zaaknummer
// nooit blijven staan voor de volgende gebruiker. Kale in-memory state,
// expliciet leeggemaakt via resetSessionRefs() bij Stop en bij het starten
// van een nieuwe opname vanaf de startpagina.
let sessionCaseRef = "";
let sessionPersonRef = "";

const sessionCaseRefInput   = document.getElementById("sessionCaseRefInput");
const sessionPersonRefInput = document.getElementById("sessionPersonRefInput");

if (sessionCaseRefInput) {
  sessionCaseRefInput.addEventListener("input", () => {
    sessionCaseRef = sessionCaseRefInput.value.trim();
  });
}
if (sessionPersonRefInput) {
  sessionPersonRefInput.addEventListener("input", () => {
    sessionPersonRef = sessionPersonRefInput.value.trim();
  });
}

function resetSessionRefs() {
  sessionCaseRef = "";
  sessionPersonRef = "";
  if (sessionCaseRefInput) sessionCaseRefInput.value = "";
  if (sessionPersonRefInput) sessionPersonRefInput.value = "";
}

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

  channelConfigs.forEach((cfg, idx) => {
    const role = getRoleById(cfg.roleId);
    const isInterpreter = cfg.roleId === "interpreter";

    const div = document.createElement("div");
    div.className = "channel-row";
    div.dataset.uid = cfg.uid;
    div.innerHTML = `
      ${channelBadgeHTML(idx + 1, role)}
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
      <button class="ch-remove" title="Kanaal verwijderen"><svg class="icon"><use href="#icon-close"></use></svg></button>
    `;

    const badge      = div.querySelector(".channel-badge");
    const roleSelect = div.querySelector(".ch-role");
    const lang2Field = div.querySelector(".ch-lang2-field");
    const langLabel  = div.querySelector(".ch-lang-label");

    roleSelect.addEventListener("change", () => {
      const c = channelConfigs.find(x => x.uid === cfg.uid);
      if (!c) return;
      c.roleId = roleSelect.value;
      const newRole = getRoleById(c.roleId);
      badge.style.background = newRole.color;
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
  });
}

// === Live meter per kanaal (tijdens opname) ===
// Visualiseert channelAudioState (app.js:75-140, al gevuld door de
// AudioWorklet t.b.v. de cross-kanaal anti-lek-arbitrage, maar tot nu toe
// nergens getoond) -- geen decoratieve golfvorm, maar zichtbaar maken welk
// kanaal geluid oppikt en welk kanaal de anti-lek-gate live onderdrukt.

let channelMeterRafId = null;

// rms is sqrt(mean(sample^2)) op genormaliseerde PCM (-1..1, zie
// pcm_worklet.js); normale spraak zit ruwweg in de orde 0.02-0.1, dus x140
// laat dat prettig uitslaan binnen de 14px-balkhoogte. Eerste-orde kalibratie,
// desgewenst bijstellen na testen met echte microfoons.
const METER_RMS_SCALE = 140;

function renderChannelMeters() {
  const el = document.getElementById("channelMeters");
  if (!el) return;
  el.innerHTML = "";
  channelConfigs.forEach((cfg, idx) => {
    const role = getRoleById(cfg.roleId);
    const chip = document.createElement("div");
    chip.className = "channel-meter-chip";
    chip.dataset.uid = cfg.uid;
    chip.innerHTML = `
      ${channelBadgeHTML(idx + 1, role)}
      <span class="channel-meter-bars" style="color:${role.color}">
        <span></span><span></span><span></span><span></span>
      </span>
    `;
    el.appendChild(chip);
  });
}

function updateChannelMeterLevels() {
  const el = document.getElementById("channelMeters");
  if (!el) return;
  for (const cfg of channelConfigs) {
    const chip = el.querySelector(`.channel-meter-chip[data-uid="${cfg.uid}"]`);
    if (!chip) continue;
    const state = channelAudioState.get(cfg.uid);
    const rms = state?.rms || 0;
    const gateOpen = state ? state.combinedGateOpen !== false : true;
    chip.classList.toggle("gate-closed", !gateOpen);
    const bars = chip.querySelectorAll(".channel-meter-bars span");
    bars.forEach((bar, i) => {
      const jitter = 1 - i * 0.12; // balkjes iets aflopend, oogt als eq i.p.v. vlakke blokjes
      const h = Math.min(14, Math.max(2, rms * METER_RMS_SCALE * jitter));
      bar.style.height = h + "px";
    });
  }
}

function startChannelMeterLoop() {
  renderChannelMeters();
  const el = document.getElementById("channelMeters");
  if (el) el.classList.remove("hidden");
  const tick = () => {
    updateChannelMeterLevels();
    channelMeterRafId = requestAnimationFrame(tick);
  };
  if (channelMeterRafId == null) channelMeterRafId = requestAnimationFrame(tick);
}

function stopChannelMeterLoop() {
  if (channelMeterRafId != null) {
    cancelAnimationFrame(channelMeterRafId);
    channelMeterRafId = null;
  }
  const el = document.getElementById("channelMeters");
  if (el) { el.classList.add("hidden"); el.innerHTML = ""; }
}

// === Multi-channel transcript state ===

// Per kanaal: lines array, id→line map, parentId→sentences map, pending updates
const channelLines        = new Map(); // channelId → line[]
const channelLineById     = new Map(); // channelId → Map(id → line)
const channelSentenceMap  = new Map(); // channelId → Map(parentId → sentence[])
const channelPendingUpd   = new Map(); // channelId → Map(id → update)

// Voor sessie-terugluister (gemergd, alle kanalen -- met per-kanaal filter)
let playbackLines = [];
let playbackLineById = new Map();
let playbackSentenceMap = new Map();
let playbackChannelId = "default";
let isPlaybackMode = false;
let playbackChannels = [];        // alle channel_id's aanwezig in deze sessie
let playbackActiveChannels = new Set(); // welke channel_id's momenteel getoond worden

// Voor de playback-tijdlijn: totale sessieduur (van de server, of anders
// afgeleid uit het laatste segment) + welk kanaal/vanaf-welk-tijdstip er nu
// in de ingebouwde <audio>-speler geladen is (nodig om currentTime terug te
// rekenen naar sessie-absolute tijd, zie highlightSegmentAt()).
let playbackDurationMs = 0;
let playbackAudioChannel = null;
let playbackAudioSliceStartMs = 0;
let playbackSessionDate = null; // leesbare datum uit de server, voor de "Sessie terugluisteren"-subtitel

// Voor de read-only sessiesamenvatting (Sessiegegevens/Sprekers/Documenten)
// die tijdens terugluisteren het Bediening/Configuratie-tabbladstelsel
// vervangt, zie updateLiveVsPlaybackUI(). Komen rechtstreeks uit de
// transcript?channel_id=all-respons, geen aparte fetch nodig.
let playbackCaseRef = null;
let playbackPersonRef = null;
let playbackLanguages = {}; // { channel_id: taalcode }
let playbackLanguages2 = {}; // { channel_id: taalcode|null } -- Taal 2, alleen bij taalpaar-kanalen (tolk)
let playbackGehoorverslagGeneratedAt = null;
let playbackCreatedAt = null; // ruwe %Y%m%dT%H%M%SZ-timestamp, voor formatDutchDateTime()

// True terwijl de gebruiker de transportbalk-scrubber vasthoudt -- voorkomt dat
// de timeupdate-listener (renderPlaybackTimeline()) de balk tijdens het slepen
// zelf terugzet naar de actuele afspeelpositie.
let isScrubbingPlayback = false;

// === Opname state ===

// uid → { ws, channelId, audioContext, mediaStream, workletNode, recorderWorker, mediaRecorder, watchdog }
const activeConnections = new Map();

let currentSessionId  = null;
let isRecording       = false;
let isPaused          = false;
let keepAliveTimer    = null;
let serverUseAudioWorklet = true;
let startTime         = null;
let timerInterval     = null;
let lastBufferTranscription = "";
let lastBufferTranslation   = "";
let lastStatus              = "active_transcription";

// Puur voor de keep-alive tijdens pauze: 100ms stilte @ 16kHz, 1-byte gate-vlag (dicht)
// ervoor. Dit is geen echte microfoonaudio -- de mic staat dan al uit (audioContext
// suspended) -- puur bedoeld om de WebSocket/proxy niet te laten timeouten.
const KEEP_ALIVE_INTERVAL_MS = 15000;
const KEEP_ALIVE_SILENT_PCM  = new Uint8Array(1600 * 2); // all-zero = digitale stilte

// === DOM refs ===

const recordButton       = document.getElementById("recordButton");
const pauseButton        = document.getElementById("pauseButton");
const stopButton         = document.getElementById("stopButton");
const refreshButton      = document.getElementById("refreshButton");
const refreshPlaybackButton = document.getElementById("refreshPlaybackButton");
const gehoorverslagButton = document.getElementById("gehoorverslagButton");
const liveTranscriptDiv  = document.getElementById("liveTranscript");
if (liveTranscriptDiv) liveTranscriptDiv.style.whiteSpace = "pre-wrap";
const connectionStatusSpan = document.getElementById("connectionStatus");
const modeStatusSpan       = document.getElementById("modeStatus");
const asrStatusSpan        = document.getElementById("asrStatus");
const connectionStatusDot  = document.getElementById("connectionStatusDot");
const modeStatusDot        = document.getElementById("modeStatusDot");
const asrStatusDot         = document.getElementById("asrStatusDot");
const timerSpan            = document.getElementById("recordingTimer");
const hintText             = document.getElementById("hintText");

const landingPage        = document.getElementById("landingPage");
const workspaceMain      = document.querySelector(".app-main");
const newSessionBtn      = document.getElementById("newSessionBtn");
const landingSearchInput = document.getElementById("landingSearchInput");

// === Status UI helpers ===

function setStatusDot(el, state) {
  if (!el) return;
  el.className = `status-dot dot-${state}`;
}

function setConnectionStatus(connectedCount, totalCount) {
  if (!connectionStatusSpan) return;
  if (connectedCount === 0) {
    connectionStatusSpan.textContent = "Niet verbonden";
    connectionStatusSpan.className = "status-value status-disconnected";
    setStatusDot(connectionStatusDot, "disconnected");
  } else if (connectedCount < totalCount) {
    connectionStatusSpan.textContent = `${connectedCount}/${totalCount} verbonden`;
    connectionStatusSpan.className = "status-value status-recording";
    setStatusDot(connectionStatusDot, "recording");
  } else {
    connectionStatusSpan.textContent = totalCount === 1 ? "Verbonden" : `${connectedCount} kanalen verbonden`;
    connectionStatusSpan.className = "status-value status-connected";
    setStatusDot(connectionStatusDot, "connected");
  }
}

function setModeStatus(text) {
  if (modeStatusSpan) modeStatusSpan.textContent = text;
  setStatusDot(modeStatusDot, text && text !== "–" ? "connected" : "neutral");
}

// Ruwe heuristiek op de tekst zelf i.p.v. elke bestaande aanroepplek van
// setAsrStatus (tientallen, verspreid door dit bestand) te moeten voorzien
// van een expliciete status-parameter -- goed genoeg voor een puur visuele stip.
function setAsrStatus(text) {
  if (asrStatusSpan) asrStatusSpan.textContent = text;
  const t = (text || "").toLowerCase();
  let dot = "neutral";
  if (t.includes("actief")) dot = "connected";
  else if (t.includes("gepauzeerd") || t.includes("ververst")) dot = "recording";
  else if (t.includes("gestopt") || t.includes("mislukt") || t.includes("geen audio") || t.includes("verbroken") || t.includes("fout")) dot = "disconnected";
  setStatusDot(asrStatusDot, dot);
}

function updateRecordButtonUI() {
  const startFromConfigBtn = document.getElementById("startFromConfigBtn");
  if (!recordButton) return;
  if (!isRecording) {
    recordButton.classList.remove("hidden");
    // Bij het bekijken van een opgeslagen sessie is "Start" misleidend (lijkt
    // deze sessie te hervatten) -- de knop doet functioneel nog steeds
    // hetzelfde (startRecording() reset playback-state al correct), alleen
    // het label maakt duidelijk dat dit een NIEUWE opname begint.
    recordButton.innerHTML = isViewingStoredSession()
      ? '<svg class="icon"><use href="#icon-mic"></use></svg> Nieuwe opname starten'
      : '<svg class="icon"><use href="#icon-mic"></use></svg> Start';
    recordButton.classList.remove("recording");
    // In de sessiesamenvatting is dit de secundaire "verlaat het overzicht"-
    // actie, niet de hoofdhandeling van dit scherm (zie .button-secondary,
    // eerder gebruikt voor "Ververs Transcriptie").
    recordButton.classList.toggle("button-secondary", isViewingStoredSession());
    if (pauseButton) pauseButton.classList.add("hidden");
    if (stopButton) stopButton.classList.add("hidden");
    if (refreshButton) refreshButton.classList.add("hidden");
    if (startFromConfigBtn) {
      // Zelfde label-logica als recordButton hierboven, voor consistentie
      // tussen de twee knoppen die allebei startRecording() aanroepen.
      startFromConfigBtn.innerHTML = isViewingStoredSession()
        ? '<svg class="icon"><use href="#icon-mic"></use></svg> Nieuwe opname starten'
        : '<svg class="icon"><use href="#icon-mic"></use></svg> Start opname';
      startFromConfigBtn.classList.remove("hidden");
    }
  } else {
    recordButton.classList.add("hidden");
    if (startFromConfigBtn) startFromConfigBtn.classList.add("hidden");
    if (pauseButton) {
      pauseButton.classList.remove("hidden");
      pauseButton.innerHTML = isPaused
        ? '<svg class="icon"><use href="#icon-play"></use></svg> Hervatten'
        : '<svg class="icon"><use href="#icon-pause"></use></svg> Pauze';
      pauseButton.classList.toggle("paused", isPaused);
    }
    if (stopButton) {
      stopButton.classList.remove("hidden");
      stopButton.classList.add("recording");
    }
    // Alleen klikbaar tijdens Pauze -- tijdens actief opnemen zijn er nog geen
    // stabiele WAV-grenzen om vanaf te herbouwen (zie audio_processor.py flag=3).
    if (refreshButton) refreshButton.classList.toggle("hidden", !isPaused);
  }
}

function updateHint() {
  if (!hintText) return;
  if (isViewingStoredSession()) {
    hintText.innerHTML = "Je bekijkt een opgeslagen sessie. Klik <strong>Nieuwe opname starten</strong> voor een nieuwe opname.";
  } else if (!isRecording) {
    hintText.innerHTML = "Stel kanalen in via <strong>Configuratie</strong>, dan klik <strong>Start</strong>.";
  } else if (isPaused) {
    hintText.textContent = "Gepauzeerd. Klik Hervatten om door te gaan, of Stop om te beëindigen.";
  } else {
    hintText.textContent = "Opname loopt. Spreek in de microfoon(s).";
  }
}

// Onderscheidt "écht een opgeslagen/afgeronde sessie bekijken" van "Ververs
// Transcriptie tijdens Pauze" -- dat laatste zet isPlaybackMode ook tijdelijk
// op true voor een statische snapshot van de HUIDIGE, nog actieve sessie, en
// moet dus gewoon Hervatten/Stop blijven tonen i.p.v. de sessie-infoweergave.
function isViewingStoredSession() {
  return !isRecording && isPlaybackMode && !!currentSessionId;
}

// Centrale schakelaar tussen de live- en playback-weergave: titel/subtitel,
// live-verbindingsstatus (betekenisloos zonder actieve sessie) en het
// sessie-infopaneel i.p.v. de losse Start-knop. Aangeroepen op elk van de 3
// plekken waar isRecording/isPlaybackMode al wijzigen -- geen nieuwe state.
function updateLiveVsPlaybackUI() {
  const viewing = isViewingStoredSession();

  const titleEl    = document.getElementById("mainPanelTitle");
  const subtitleEl = document.getElementById("mainPanelSubtitle");
  if (titleEl) titleEl.textContent = viewing ? "Sessie terugluisteren" : "Live transcriptie";
  if (subtitleEl) {
    const parts = [];
    if (viewing) {
      parts.push(playbackCreatedAt ? formatDutchDateTime(playbackCreatedAt) : (playbackSessionDate || "onbekende datum"));
      // Zaaknummer is de primaire identiteit van een sessie als het er is
      // (zelfde principe als de sessielijst op het Werkoverzicht) -- de
      // sessie-id is dan alleen nog secundaire/technische info, niet meer
      // in deze titelregel.
      if (playbackCaseRef) parts.push(`Zaak ${playbackCaseRef}`);
      else if (currentSessionId) parts.push(`sessie ${currentSessionId.substring(0, 12)}…`);
    }
    subtitleEl.textContent = parts.join(" · ");
    subtitleEl.classList.toggle("hidden", parts.length === 0);
  }

  // Tijdens terugluisteren vervangt de vaste, read-only sessiesamenvatting
  // (hieronder) het hele Bediening/Configuratie-tabbladstelsel -- niet omdat
  // bijconfigureren de bekeken sessie zou beschadigen (dat gebeurt niet),
  // maar omdat zichtbare tabbladen anders suggereren dat je iets van de
  // bekeken sessie aanpast, wat verwarrend is in een juridische context.
  // "Nieuwe opname starten" (in de samenvatting) brengt de normale
  // werkomgeving mét tabbladen weer terug.
  const tabBar = document.getElementById("tabBar");
  if (tabBar) tabBar.classList.toggle("hidden", viewing);
  if (viewing) {
    // Forceer terug naar het Bediening-paneel als Configuratie nog actief
    // stond -- dat paneel bevat de samenvatting hieronder, Configuratie zelf
    // is nu onbereikbaar (tab-bar is immers verborgen).
    document.querySelector('.tab-btn[data-tab="control"]')?.click();
  }

  const connectionRow = document.getElementById("connectionStatusRow");
  const modeRow        = document.getElementById("modeStatusRow");
  const asrRow         = document.getElementById("asrStatusRow");
  if (connectionRow) connectionRow.classList.toggle("hidden", viewing);
  if (modeRow) modeRow.classList.toggle("hidden", viewing);
  if (asrRow) asrRow.classList.toggle("hidden", viewing);

  const infoPanel = document.getElementById("sessionInfoPanel");
  const cardsEl   = document.getElementById("sessionInfoCards");
  if (infoPanel) infoPanel.classList.toggle("hidden", !viewing);
  if (cardsEl && viewing) {
    cardsEl.innerHTML = buildSessionInfoCardsHTML();
  }

  updateRecordButtonUI();
  updateHint();
}

// Bouwt de drie read-only samenvattingskaarten (Sessiegegevens/Sprekers/
// Documenten) die tijdens terugluisteren het tabbladstelsel vervangen, zie
// updateLiveVsPlaybackUI() hierboven. "Microfoon" per spreker is bewust
// weggelaten -- dat is pure client-side apparaatkeuze, nooit naar de server
// gestuurd of opgeslagen, en dus niet betrouwbaar bekend voor een sessie die
// (mogelijk na een herstart) van disk geladen is.
function buildSessionInfoCardsHTML() {
  const dateText = playbackCreatedAt ? formatDutchDateTime(playbackCreatedAt) : (playbackSessionDate || "onbekend");

  // Volgorde: zaaknummer (indien aanwezig, primaire identiteit) -> datum ->
  // duur -> status -> sessie-id als kleinere, secundaire technische regel
  // (zie .session-info-id hieronder in style.css), niet meer gelijkwaardig
  // aan de andere rijen.
  const sessieRows = [];
  if (playbackCaseRef) sessieRows.push(["Zaaknummer", playbackCaseRef]);
  sessieRows.push(["Datum", dateText]);
  sessieRows.push(["Duur", formatMs(getPlaybackDurationMs())]);
  // "Opgeslagen" is geen verzonnen gegeven -- deze kaart wordt uitsluitend
  // getoond wanneer isViewingStoredSession() al waar is, dus dit beschrijft
  // simpelweg de huidige, altijd-ware schermstaat.
  sessieRows.push(["Status", "Opgeslagen"]);

  const sessieCard = `<div class="session-info-card">
    <p class="session-info-title">Sessiegegevens</p>
    <dl class="session-info-list">${sessieRows.map(([label, value]) =>
      `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`
    ).join("")}</dl>
    <p class="session-info-id">Sessie-id: ${currentSessionId ? escapeHtml(`${currentSessionId.substring(0, 12)}…`) : "-"}</p>
  </div>`;

  const speakerCards = playbackChannels.map(ch => {
    const roleId = channelIdToRoleId(ch);
    const color = getRoleColor(roleId);
    const label = getRoleLabel(roleId);
    const lang2 = playbackLanguages2[ch];
    // Taal 2 is alleen bekend als dit kanaal ooit met een lang2-queryparam
    // verbonden heeft (typisch de tolk) -- zie get_channel_language2() op de
    // backend. Zonder die waarde tonen we gewoon de ene bekende taal, geen
    // verzonnen tweede regel.
    const langText = lang2
      ? `Taal 1: ${(playbackLanguages[ch] || "-").toUpperCase()} · Taal 2: ${lang2.toUpperCase()}`
      : (playbackLanguages[ch] || "-").toUpperCase();
    return `<div class="speaker-card">
      <span class="channel-filter-dot" style="background:${color}"></span>
      <span class="speaker-card-role">${escapeHtml(label)}</span>
      <span class="speaker-card-lang">${escapeHtml(langText)}</span>
    </div>`;
  }).join("") || `<p class="session-info-empty">Geen kanalen bekend</p>`;
  const sprekersCard = `<div class="session-info-card">
    <p class="session-info-title">Sprekers</p>
    ${speakerCards}
  </div>`;

  const hasReport = !!playbackGehoorverslagGeneratedAt;
  const gehoorverslagStatus = hasReport
    ? `Laatst gegenereerd op ${new Date(playbackGehoorverslagGeneratedAt).toLocaleString("nl-NL")}`
    : "Nog niet gegenereerd";
  const documentenCard = `<div class="session-info-card">
    <p class="session-info-title">Documenten</p>
    <div class="doc-status-row">
      <svg class="icon"><use href="#icon-document"></use></svg>
      <span>Gehoorverslag</span>
      <span class="doc-status-value${hasReport ? "" : " doc-status-pending"}">${escapeHtml(gehoorverslagStatus)}</span>
    </div>
  </div>`;

  return sessieCard + sprekersCard + documentenCard;
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
  const params = new URLSearchParams({ session_id: sessionId, channel_id: channelId, lang: cfg.language || "nl", gate_framed: "1" });
  if (cfg.language2) params.set("lang2", cfg.language2);
  // Elk kanaal opent zijn eigen WS-verbinding maar deelt dezelfde session_id;
  // de backend merget case_ref/person_ref per sessie (SessionManager.create_or_update),
  // dus het is onschadelijk dat elk kanaal dezelfde waarde meestuurt.
  if (sessionCaseRef) params.set("case_ref", sessionCaseRef);
  if (sessionPersonRef) params.set("person_ref", sessionPersonRef);
  return `${proto}//${location.host}/ws?${params}`;
}

async function openAudioStream(ws, cfg, useWorklet) {
  // AGC/echo-cancellation/noise-suppression expliciet uit: deze normaliseren volume
  // en vervormen het signaal, wat zowel de cross-kanaal RMS-arbitrage ondermijnt
  // (lek wordt opgepompt richting normaal niveau) als de ruwe audio-integriteit aantast.
  const audioConstraints = {
    autoGainControl: false,
    echoCancellation: false,
    noiseSuppression: false,
    ...(cfg.deviceId ? { deviceId: { exact: cfg.deviceId } } : {}),
  };

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
      if (ws.readyState !== WebSocket.OPEN) return;
      // Frame: [1 byte gate-vlag][s16le PCM]. Audio wordt altijd verstuurd —
      // de vlag bepaalt alleen of de server dit fragment naar ASR doorlaat,
      // nooit of het wordt opgenomen (WAV blijft altijd compleet).
      const pcm = new Uint8Array(e.data.buffer);
      const framed = new Uint8Array(pcm.length + 1);
      framed[0] = e.data.gateOpen ? 1 : 0;
      framed.set(pcm, 1);
      ws.send(framed.buffer);
    };

    // Noise gate instellen op de worklet
    workletNode.port.postMessage({ threshold: cfg.gateThreshold ?? 0 });

    channelAudioState.set(cfg.uid, { rms: 0, gateOpen1: false, lastUpdate: 0, suspectSince: null, combinedGateOpen: true });

    let lastActivity = Date.now();
    workletNode.port.onmessage = e => {
      lastActivity = Date.now();
      const { buffer: pcmData, rms, gateOpen } = e.data; // pcmData: Float32Array

      const state = channelAudioState.get(cfg.uid);
      if (state) {
        state.rms = rms;
        state.gateOpen1 = gateOpen;
        state.lastUpdate = Date.now();
      }
      const combinedGate = state ? computeCombinedGate(cfg.uid, state) : gateOpen;

      const ab = pcmData.buffer;
      recorderWorker.postMessage({ command: "record", buffer: ab, gateOpen: combinedGate }, [ab.slice(0)]);
    };

    watchdog = setInterval(() => {
      if (isRecording && !isPaused && Date.now() - lastActivity > 5000) {
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
  channelAudioState.delete(uid);
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

  ws.onclose = (event) => {
    // [DIAG] "verdwenen helft opname"-onderzoek (2026-07-19): serverlog bewees een
    // stille, onverklaarde 4,5-minuten-stilstand gevolgd door een gloednieuwe sessie
    // (nieuwe session_id, WAV/vensters herbeginnen bij 0) middenin een pauzeloze
    // opname -- zonder enige exception. Dit logt code/reason/wasClean van de
    // onderliggende WS-close, om de volgende keer vast te stellen OF en WAAROM de
    // verbinding zelf brak (i.p.v. alleen te zien dat de server niets meer ontving).
    console.log(
      "[DIAG][WS_CLOSE]", channelId,
      "code=" + event.code, "reason=" + JSON.stringify(event.reason),
      "wasClean=" + event.wasClean, "isRecording=" + isRecording, "isPaused=" + isPaused
    );
    handleChannelClose(cfg.uid, channelId);
  };
  ws.onerror = () => {
    console.log("[DIAG][WS_ERROR]", channelId, "isRecording=" + isRecording, "isPaused=" + isPaused);
  };

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

  // [DIAG] "verdwenen helft opname"-onderzoek (2026-07-19): als hier nog
  // achtergebleven verbindingen van een vorige sessie openstaan, betekent dat een
  // nieuwe sessie start bovenop een oude i.p.v. schoon te beginnen -- precies het
  // patroon dat het serverlog liet zien (twee sessie-ids, oude nooit netjes
  // afgesloten). Dit legt vast OF dat hier ook gebeurt, en met welke oude sessionId.
  if (activeConnections.size > 0) {
    console.log(
      "[DIAG][START_RECORDING] nieuwe sessie start met nog", activeConnections.size,
      "openstaande verbinding(en) van vorige sessie", currentSessionId
    );
  }

  isPlaybackMode = false;
  hidePlaybackChannelFilter();
  hidePlaybackTimeline();
  updateRefreshPlaybackButtonUI();
  currentSessionId = crypto.randomUUID();
  console.log("[DIAG][START_RECORDING] nieuwe sessie", currentSessionId);

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
  updateLiveVsPlaybackUI();
  startTimer();
  startChannelMeterLoop();
  setAsrStatus("Live transcriptie actief");
}

async function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  isPaused = false;
  stopKeepAlive();

  resetTimer();
  updateLiveVsPlaybackUI();
  stopChannelMeterLoop();
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
  resetSessionRefs();
}

async function confirmAndStop() {
  if (!isRecording) return;
  const ok = confirm(
    "Weet je zeker dat je de opname wilt beëindigen?\n\n" +
    "Dit sluit de sessie definitief af en kan niet ongedaan worden gemaakt. " +
    "Gebruik Pauze als je later wilt doorgaan."
  );
  if (!ok) return;
  await stopRecording();
}

// === Pauze / hervatten ===
// Pauze stopt de daadwerkelijke microfooncapture (audioContext.suspend()) -- er wordt
// dan helemaal niets meer opgenomen, niet alleen "niet naar ASR gestuurd". De WebSocket-
// verbinding en de sessie op de server blijven gewoon leven; bij hervatten gaat dezelfde
// sessie/WAV/transcript gewoon door, er wordt geen nieuwe sessie gestart.

function sendKeepAliveFrames() {
  for (const [, conn] of activeConnections.entries()) {
    if (conn.ws?.readyState !== WebSocket.OPEN) continue;
    // Alleen relevant voor de PCM-worklet-modus; de WebM/MediaRecorder-fallback
    // gebruikt mediaRecorder.pause()/resume(), die geen los keep-alive-signaal nodig heeft.
    if (!serverUseAudioWorklet) continue;
    const framed = new Uint8Array(KEEP_ALIVE_SILENT_PCM.length + 1);
    framed[0] = 0; // gate dicht: puur keep-alive, nooit naar ASR
    framed.set(KEEP_ALIVE_SILENT_PCM, 1);
    try { conn.ws.send(framed.buffer); } catch (e) {}
  }
}

function startKeepAlive() {
  stopKeepAlive();
  keepAliveTimer = setInterval(sendKeepAliveFrames, KEEP_ALIVE_INTERVAL_MS);
}

function stopKeepAlive() {
  if (keepAliveTimer) {
    clearInterval(keepAliveTimer);
    keepAliveTimer = null;
  }
}

async function pauseAllConnections() {
  for (const [, conn] of activeConnections.entries()) {
    try {
      // Audio weggooien op de bron (worklet), AudioContext blijft gewoon draaien --
      // suspend()/resume() bleek onbetrouwbaar bij hervatten, zie pcm_worklet.js.
      if (conn.workletNode) {
        conn.workletNode.port.postMessage({ appPaused: true });
      }
      if (conn.mediaRecorder && conn.mediaRecorder.state === "recording") {
        conn.mediaRecorder.pause();
      }
      // Vraag de server om het huidige live-segment/batch-venster netjes af te
      // sluiten (zoals Stop dat doet), zonder de sessie/WAV te beëindigen.
      if (conn.ws?.readyState === WebSocket.OPEN && serverUseAudioWorklet) {
        conn.ws.send(new Uint8Array([2]).buffer); // vlag=2: pauze-flush, geen payload
      }
    } catch (e) { console.warn("Pauzeren mislukt voor kanaal:", conn.channelId, e); }
  }
}

async function resumeAllConnections() {
  for (const [, conn] of activeConnections.entries()) {
    try {
      if (conn.workletNode) {
        conn.workletNode.port.postMessage({ appPaused: false });
      }
      if (conn.mediaRecorder && conn.mediaRecorder.state === "paused") {
        conn.mediaRecorder.resume();
      }
    } catch (e) { console.warn("Hervatten mislukt voor kanaal:", conn.channelId, e); }
  }
}

async function pauseRecording() {
  if (!isRecording || isPaused) return;
  isPaused = true;
  await pauseAllConnections();
  startKeepAlive();
  updateRecordButtonUI();
  updateHint();
  setAsrStatus("Gepauzeerd");
}

async function resumeRecording() {
  if (!isRecording || !isPaused) return;
  isPaused = false;
  stopKeepAlive();
  // Als "Ververs Transcriptie" tijdens deze pauze de weergave in playback-mode
  // zette (een statische snapshot, zie loadSessionTranscript()), moet Hervatten
  // altijd terug naar de live-weergave -- anders blijft renderAllChannels() nieuwe
  // front_data stilzwijgend overslaan (isPlaybackMode-check) en lijkt de opname
  // "vast" te zitten na hervatten.
  isPlaybackMode = false;
  hidePlaybackChannelFilter();
  hidePlaybackTimeline();
  updateRefreshPlaybackButtonUI();
  await resumeAllConnections();
  updateLiveVsPlaybackUI();
  setAsrStatus("Live transcriptie actief");
}

function togglePause() {
  if (!isRecording) return;
  if (isPaused) resumeRecording(); else pauseRecording();
}

// === Ververs Transcriptie ===
// Vervangt het transcript van de huidige sessie volledig door een verse batch-
// herbouw vanaf de tot-nu-toe opgenomen WAV, los van de incrementele live-decoder-
// boekhouding (zie POST /sessions/{id}/refresh_transcript). Klikbaar vanuit twee
// plekken -- de Bediening-knop (tijdens Pauze) en de knop boven het transcript
// (bij het bekijken van een gestopte/eerdere sessie) -- met identiek gedrag: is de
// sessie nog live (activeConnections niet leeg), stuur eerst een flush-signaal
// zodat de WAV op schijf actueel is; is de sessie al gestopt, dan is dat een no-op
// (activeConnections is dan al leeg) en is de WAV toch al gesloten/compleet.
async function refreshTranscript() {
  if (!currentSessionId) return;
  const ok = confirm(
    "Dit vervangt het huidige transcript door een verse batchtranscriptie van de " +
    "opname tot nu toe. Dit kan enkele minuten duren bij een lange opname.\n\n" +
    "Doorgaan?"
  );
  if (!ok) return;

  const wasBusy = refreshButton?.disabled || refreshPlaybackButton?.disabled;
  if (wasBusy) return;
  if (refreshButton) refreshButton.disabled = true;
  if (refreshPlaybackButton) refreshPlaybackButton.disabled = true;
  setAsrStatus("Transcript wordt ververst… dit kan even duren.");

  try {
    if (activeConnections.size > 0) {
      for (const [, conn] of activeConnections.entries()) {
        if (conn.ws?.readyState === WebSocket.OPEN && serverUseAudioWorklet) {
          try { conn.ws.send(new Uint8Array([3]).buffer); } catch (e) {}
        }
      }
      // Geen echte ack-ronde (zie audio_processor.py flag=3) -- een korte vaste
      // wachttijd is voldoende, want _flush_wav() zelf is vrijwel instant.
      await new Promise(r => setTimeout(r, 500));
    }

    const resp = await fetch(`/sessions/${encodeURIComponent(currentSessionId)}/refresh_transcript`, {
      method: "POST",
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      alert("Verversen mislukt: " + (err.error || resp.statusText));
      setAsrStatus(isPaused ? "Gepauzeerd" : "Live transcriptie actief");
      return;
    }

    await loadSessionTranscript(currentSessionId);
    setAsrStatus("Transcript ververst.");
  } catch (e) {
    alert("Verversen mislukt: " + e.message);
    setAsrStatus(isPaused ? "Gepauzeerd" : "Live transcriptie actief");
  } finally {
    if (refreshButton) refreshButton.disabled = false;
    if (refreshPlaybackButton) refreshPlaybackButton.disabled = false;
  }
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

  // [DIAG] client-side "spooklijn"-onderzoek (2026-07-19): server-log bewijst dat de
  // server herhaaldelijk een correcte, ingetrokken-hallucinatie-vrije snapshot stuurt --
  // dit logt wat de client daadwerkelijk ONTVANGT en toepast, om te zien of/waar het
  // daarna alsnog misgaat (bv. isPlaybackMode die renderAllChannels() overslaat, of een
  // segment_update die een regel terugzet nadat front_data hem al had laten vallen).
  console.log("[DIAG][FRONTDATA_RECV]", channelId, "n_lines=" + lines.length,
    "ids=" + JSON.stringify(lines.map(l => l && l.id)), "isPlaybackMode=" + isPlaybackMode);

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

  // [DIAG] zie [DIAG][FRONTDATA_RECV] hierboven -- dit is het ANDERE pad dat een regel
  // kan toevoegen/bijwerken buiten een volledige front_data-snapshot om. Van belang: of
  // dit pad na een correctie alsnog een reeds-ingetrokken regel terugzet.
  console.log("[DIAG][SEGMENT_UPDATE]", channelId, "id=" + id,
    "text_final=" + JSON.stringify(data.text_final), "text_batch=" + JSON.stringify(data.text_batch),
    "state=" + data.state, "isPlaybackMode=" + isPlaybackMode);

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
  if (isPlaybackMode) {
    console.log("[DIAG][RENDER_SKIPPED] isPlaybackMode=true, render overgeslagen");
    return;
  }
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

  // Dossierweergave (vaste kolommen, gedempte iconen, ...) alleen tijdens
  // terugluisteren -- alle bijbehorende CSS is geschoold onder .doc-view, dus
  // deze toggle is de enige plek die bepaalt of het live-scherm meeverandert
  // (dat mag het niet: zie CLAUDE.md "live vs. batch" en de expliciete scope
  // van deze ronde, alleen het terugluister-scherm).
  liveTranscriptDiv.classList.toggle("doc-view", isPlaybackMode);

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
    const rawTxt = (item?.text_batch || item?.text || item?.text_live || "").trim();
    if (!rawTxt) continue;

    const sp = item?.speaker ?? item?.speaker_id ?? item?.spk;
    const st = (item?.state || "FINAL").toUpperCase();

    let cls = "seg";
    const isBatchConfirmed = st !== "LIVE" && !!item.text_batch;
    if (st === "LIVE") {
      cls += " seg-live";
    } else if (isBatchConfirmed) {
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
    // .seg-role-dot is alleen zichtbaar onder .doc-view (terugluisteren) --
    // in het live-scherm blijft de bestaande gekleurde .seg-role-tekst
    // ongewijzigd, zie style.css.
    const roleDot   = `<span class="seg-role-dot" style="background:${roleColor}"></span>`;
    const roleLabel = `<span class="seg-role" style="color:${roleColor}">${roleDot}${escapeHtml(getRoleLabel(roleId))}</span> `;

    // Wit vs. groen oogde voor klanten als "werkt niet goed" -- FINAL en batch-bevestigde
    // tekst zien er nu identiek uit; alleen een klein vinkje geeft aan dat de batch-pass
    // dit segment heeft bevestigd. Het onderscheid blijft intact in de data (text_batch),
    // alleen de live-weergave is verzacht.
    const confirmedBadge = isBatchConfirmed ? `<span class="seg-confirmed" title="Bevestigd door batch-pass"><svg class="icon"><use href="#icon-check"></use></svg></span>` : "";

    // Vertaalicoontje: bij bevestigde (batch) tekst van elk kanaal behalve medewerker/
    // advocaat (die zijn per ontwerp altijd Nederlands) -- ook de tolk kan soms iets in
    // de brontaal van de vreemdeling zeggen, zie features/vertaling-niet-nl-tekst.md.
    // data-raw-text ipv de al-geëscapete regeltekst, zodat de klik-handler de
    // ongewijzigde brontekst naar /translate stuurt.
    const translateIcon = (isBatchConfirmed && roleId !== "employee" && roleId !== "lawyer")
      ? `<span class="seg-translate" title="Vertaal naar het Nederlands" data-raw-text="${escapeHtml(rawTxt)}"><svg class="icon"><use href="#icon-globe"></use></svg></span>`
      : "";

    // .seg-text/.seg-actions zijn no-op wrappers (display:inline, geen eigen
    // CSS) buiten .doc-view -- alleen daar krijgen ze grid-kolommen, zie
    // style.css. Live-scherm-markup/opmaak blijft zo ongewijzigd.
    const textSpan    = `<span class="seg-text">${prefix}${escapeHtml(rawTxt)}</span>`;
    const actionsSpan = (confirmedBadge || translateIcon)
      ? `<span class="seg-actions">${confirmedBadge}${translateIcon}</span>` : "";

    htmlParts.push(`<div class="${cls} seg-clickable"${idAttr}${audioAttr}>${timeLabel}${roleLabel}${textSpan}${actionsSpan}</div>`);
  }

  const hasLiveContent = (lines || []).some(item => (item?.text || item?.text_live) && !item?.text_batch && item?.speaker !== -2);
  if (hasLiveContent || (bufferTranscription && bufferTranscription.trim())) {
    htmlParts.push(`<div class="live-indicator"><span class="live-dot"></span> Spreekt…</div>`);
  }

  liveTranscriptDiv.innerHTML = htmlParts.join("") || "Nog geen tekst ontvangen";

  if (isAtBottom) scrollParent.scrollTop = scrollParent.scrollHeight;

  // Alleen overschrijven tijdens een actieve, niet-gepauzeerde opname -- anders vecht
  // dit met de "Gepauzeerd"/waakhond-status bij elk binnenkomend serverbericht.
  if (isRecording && !isPaused) {
    setAsrStatus("Live transcriptie actief");
  }
}

// === Startpagina ===
// Georganiseerd rondom zaken/gesprekken, niet rondom bestanden: één lijst
// waarin elke sessie precies één keer voorkomt (i.p.v. drie parallelle
// kolommen met dezelfde sessies), gefilterd via knoppen + een zoekveld.
// Zaaknummer is de primaire identiteit van een rij, niet de technische
// sessie-id. Vervangt ook de vorige sessie-browser-modal.

let landingSessionsCache = [];
let landingActiveFilter = "all";
let landingSortMode = "newest";
let landingPageIndex = 0;
let landingCurrentView = "overview"; // "overview" | "sessions"
const SESSIONS_PER_PAGE = 15;
const OVERVIEW_RECENT_COUNT = 5;

// Bewust geen naam in de begroeting ("Goedemorgen, {naam}") -- er is nog
// geen echt account-systeem (zie CLAUDE.md/de login-discussie), dus een
// naam tonen zou een niet-bestaande login voorwenden. Het tijdstip komt
// gewoon van de systeemklok, geen gebruikersgegeven.
function landingGreetingText() {
  const hour = new Date().getHours();
  if (hour < 12) return "Goedemorgen";
  if (hour < 18) return "Goedemiddag";
  return "Goedenavond";
}

async function loadLandingData() {
  const greetingEl = document.getElementById("landingGreeting");
  if (greetingEl) greetingEl.textContent = landingGreetingText();
  try {
    const resp = await fetch("/sessions/list");
    const data = await resp.json();
    landingSessionsCache = data.sessions || [];
  } catch (e) {
    landingSessionsCache = [];
  }
  landingPageIndex = 0;
  renderLandingStats();
  renderLandingAttention();
  renderOverviewRecent();
  renderLandingList();
}

// Klikbare navigatie tussen Werkoverzicht en Sessies -- beide leunen op
// dezelfde landingSessionsCache/renderfuncties, dus geen nieuwe fetch nodig
// om te wisselen.
function setLandingView(view) {
  landingCurrentView = view;
  document.getElementById("overviewView")?.classList.toggle("hidden", view !== "overview");
  document.getElementById("sessionsView")?.classList.toggle("hidden", view !== "sessions");
  document.querySelectorAll(".app-nav-item[data-view]").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.view === view);
  });
}

// Een héél korte opname (test/per ongeluk gestarte sessie) telt niet mee als
// "moet nog een gehoorverslag krijgen" -- anders ontstaat een permanente
// lijst met schijnproblemen zodra er testfragmenten tussen de echte sessies
// staan. Er is geen "dit was een afgerond gehoor"-vlag, dus duur is de enige
// eerlijke, al beschikbare proxy hiervoor. Onbekende duur (oudere sessies
// zonder wav-header-uitlezing) telt voorzichtigheidshalve WEL mee.
const REPORT_MIN_DURATION_MS = 2 * 60 * 1000;
function isReportWorthy(s) {
  return s.has_transcript && !s.gehoorverslag_generated_at
    && (!Number.isFinite(s.duration_ms) || s.duration_ms >= REPORT_MIN_DURATION_MS);
}

// Bescheiden oriëntatie, geen dashboard -- platte tekst, geen grote
// getal-kaarten.
function renderLandingStats() {
  const statsEl = document.getElementById("landingStats");
  if (!statsEl) return;
  const total = landingSessionsCache.length;
  if (!total) { statsEl.innerHTML = ""; return; }
  const needsReport = landingSessionsCache.filter(isReportWorthy).length;
  const noTranscript = landingSessionsCache.filter(s => !s.has_transcript).length;
  const parts = [`<span><strong>${total}</strong> sessies</span>`, `<span><strong>${needsReport}</strong> zonder verslag</span>`];
  // "Zonder transcriptie" i.p.v. "wordt verwerkt" -- we weten uit deze data
  // niet of er nu daadwerkelijk een achtergrondtaak loopt, alleen dat er
  // (nog) geen transcript-bestand is. "Wordt verwerkt" suggereert actieve
  // voortgang die er misschien allang niet meer is (bv. oude testdata).
  if (noTranscript) parts.push(`<span><strong>${noTranscript}</strong> zonder transcriptie</span>`);
  statsEl.innerHTML = parts.join("");
}

// Zijkolom "Aandacht nodig": alleen categorieën die daadwerkelijk actie
// vragen -- geen "niet aan zaak gekoppeld"-rij: een zaaknummer is bewust
// optioneel (zie Configuratie-tab), dus veel/alle sessies zonder zaaknummer
// is normaal, geen probleem om te melden. Elke rij is klikbaar en filtert
// de hoofdlijst (zelfde definitie als hier, zie matchesFilter in
// renderLandingList()).
function renderLandingAttention() {
  const el = document.getElementById("landingAttentionList");
  if (!el) return;

  const rows = [
    { count: landingSessionsCache.filter(s => !s.has_transcript).length, label: "sessies zonder transcriptie", filter: "no_transcript" },
    { count: landingSessionsCache.filter(isReportWorthy).length, label: "zonder gehoorverslag", filter: "needs_report" },
  ].filter(r => r.count > 0);

  if (rows.length === 0) {
    el.innerHTML = '<li class="landing-attention-empty">Niets dat aandacht vraagt.</li>';
    return;
  }
  el.innerHTML = rows.map(r =>
    `<li><button type="button" class="landing-attention-item" data-filter="${r.filter}"><strong>${r.count}</strong> ${escapeHtml(r.label)}</button></li>`
  ).join("");
  el.querySelectorAll(".landing-attention-item").forEach(btn => {
    btn.addEventListener("click", () => applyLandingFilter(btn.dataset.filter));
  });
}

// Gedeeld tussen de filterknoppen in de toolbar en de klikbare rijen in
// "Aandacht nodig" (die nu op Werkoverzicht staan) -- filteren betekent dus
// ook naar de Sessies-pagina navigeren, want daar leeft de gefilterde lijst.
function applyLandingFilter(filter) {
  landingActiveFilter = filter;
  landingPageIndex = 0;
  document.querySelectorAll(".landing-filter").forEach(b => b.classList.toggle("active", b.dataset.filter === filter));
  setLandingView("sessions");
  renderLandingList();
}

function sortLandingSessions(list) {
  const arr = [...list];
  if (landingSortMode === "oldest") {
    arr.sort((a, b) => (a.created_at || "").localeCompare(b.created_at || ""));
  } else if (landingSortMode === "case_ref") {
    arr.sort((a, b) => {
      if (!a.case_ref && !b.case_ref) return 0;
      if (!a.case_ref) return 1;
      if (!b.case_ref) return -1;
      return a.case_ref.localeCompare(b.case_ref);
    });
  } else {
    arr.sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
  }
  return arr;
}

// Werkoverzicht: enkele recente gesprekken, altijd nieuwste-eerst ongeacht de
// sorteerkeuze op de Sessies-pagina (onafhankelijke context) -- geen
// zoeken/filteren hier, dat hoort bij het volledige archief.
function renderOverviewRecent() {
  const el = document.getElementById("overviewRecentList");
  if (!el) return;
  const recent = [...landingSessionsCache]
    .sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""))
    .slice(0, OVERVIEW_RECENT_COUNT);
  el.innerHTML = "";
  if (recent.length === 0) {
    el.innerHTML = '<p class="sessions-loading">Nog geen sessies.</p>';
    return;
  }
  for (const s of recent) el.appendChild(createSessionItemEl(s));
}

// Sessies-pagina: het volledige, doorzoekbare/filterbare/sorteerbare archief
// met echte paginering (i.p.v. de vorige "alles tonen"-toggle -- dit IS nu
// het volledige-lijst-scherm, geen samenvatting meer).
function renderLandingList() {
  const listEl = document.getElementById("landingSessionsList");
  const paginationEl = document.getElementById("landingPagination");
  if (!listEl) return;

  const query = (landingSearchInput?.value || "").trim().toLowerCase();
  const matchesQuery = s => !query
    || s.session_id.toLowerCase().includes(query)
    || (s.case_ref || "").toLowerCase().includes(query);
  const matchesFilter = s => {
    if (landingActiveFilter === "no_transcript") return !s.has_transcript;
    if (landingActiveFilter === "needs_report") return isReportWorthy(s);
    if (landingActiveFilter === "has_report") return !!s.gehoorverslag_generated_at;
    return true;
  };

  const filtered = sortLandingSessions(landingSessionsCache.filter(s => matchesQuery(s) && matchesFilter(s)));
  const totalPages = Math.max(1, Math.ceil(filtered.length / SESSIONS_PER_PAGE));
  if (landingPageIndex >= totalPages) landingPageIndex = totalPages - 1;
  if (landingPageIndex < 0) landingPageIndex = 0;
  const start = landingPageIndex * SESSIONS_PER_PAGE;
  const visible = filtered.slice(start, start + SESSIONS_PER_PAGE);

  listEl.innerHTML = "";
  if (filtered.length === 0) {
    listEl.innerHTML = '<p class="sessions-loading">Geen sessies gevonden.</p>';
  } else {
    for (const s of visible) listEl.appendChild(createSessionItemEl(s));
  }

  if (paginationEl) {
    if (filtered.length <= SESSIONS_PER_PAGE) {
      paginationEl.innerHTML = "";
    } else {
      paginationEl.innerHTML = `
        <button type="button" id="landingPrevBtn"${landingPageIndex === 0 ? " disabled" : ""}>← Vorige</button>
        <span>Pagina ${landingPageIndex + 1} van ${totalPages}</span>
        <button type="button" id="landingNextBtn"${landingPageIndex >= totalPages - 1 ? " disabled" : ""}>Volgende →</button>`;
      document.getElementById("landingPrevBtn")?.addEventListener("click", () => { landingPageIndex--; renderLandingList(); });
      document.getElementById("landingNextBtn")?.addEventListener("click", () => { landingPageIndex++; renderLandingList(); });
    }
  }
}

// "employee"/"foreign_tr"/... zijn interne kanaal-id's -- getRoleLabel()/
// channelIdToRoleId() (al bestaand, ook gebruikt door de live transcriptie
// zelf) vertalen dat naar wat een gebruiker daar ook al ziet staan
// ("Medewerker", "Vreemdeling", ...), hier gededupliceerd.
function humanRoleLabels(channels) {
  const labels = (channels || []).map(ch => getRoleLabel(channelIdToRoleId(ch)));
  return [...new Set(labels)].join(", ");
}

const DUTCH_MONTHS = ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober", "november", "december"];

function parseCreatedAt(createdAt) {
  const m = createdAt && createdAt.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/);
  if (!m) return null;
  const month = DUTCH_MONTHS[parseInt(m[2], 10) - 1];
  if (!month) return null;
  return { day: parseInt(m[3], 10), month, year: m[1], hour: m[4], minute: m[5] };
}

// "22 augustus 2026, 07:18" -- volledige datum/tijd voor de detailregel.
function formatDutchDateTime(createdAt) {
  const p = parseCreatedAt(createdAt);
  return p ? `${p.day} ${p.month} ${p.year}, ${p.hour}:${p.minute}` : "onbekende datum";
}

// "22 augustus, 07:18" -- kortere variant + tijd (i.p.v. alleen de datum) voor
// de titel van een sessie zonder zaaknummer: de tijd is nodig om meerdere
// gesprekken op dezelfde dag uit elkaar te houden (zie createSessionItemEl()).
function formatDutchDateTimeShort(createdAt) {
  const p = parseCreatedAt(createdAt);
  return p ? `${p.day} ${p.month}, ${p.hour}:${p.minute}` : null;
}

function createSessionItemEl(s) {
  // Zaaknummer is het hoofdonderwerp van de rij als het er is. Zonder
  // zaaknummer is dat bewust GEEN waarschuwing -- een zaaknummer is
  // optioneel, dus dit is een normale, geen foutieve staat. "Gesprek ·
  // {datum, tijd}" i.p.v. steeds dezelfde generieke tekst (met tijd, anders
  // onderscheiden meerdere gesprekken op één dag zich niet); "Geen
  // zaaknummer" blijft alleen als bescheiden aanduiding staan, niet meer
  // naast de titel maar bij de overige sessiedetails.
  const hasCase = !!s.case_ref;
  const title = hasCase
    ? `Zaak ${escapeHtml(s.case_ref)}`
    : `Gesprek · ${escapeHtml(formatDutchDateTimeShort(s.created_at) || "onbekende datum")}`;
  const shortId = `${s.session_id.substring(0, 8)}…`;
  const duration = Number.isFinite(s.duration_ms) ? formatMs(s.duration_ms) : null;

  // Eén doorlopende, met "·" gescheiden regel i.p.v. losse blokjes die tegen
  // elkaar aan stonden. Datum/tijd alleen hier tonen als de titel die nog
  // niet al bevat (bij "Gesprek · {datum, tijd}" zou dat dubbelop zijn).
  const detailBits = hasCase ? [formatDutchDateTime(s.created_at)] : [];
  if (duration) detailBits.push(duration);
  const roles = humanRoleLabels(s.channels);
  if (roles) detailBits.push(roles);

  // Secundaire regel: technische sessie-id (klein/gedempt, alleen relevant
  // voor support/verwijzing) + eventueel "Geen zaaknummer".
  const metaBits = [`sessie ${shortId}`];
  if (!hasCase) metaBits.push("Geen zaaknummer");
  const metaLine = metaBits.map(escapeHtml).join(" · ");

  // Drie eerlijke statussen, geen 4e "klaar voor controle" -- daar is geen
  // echt bijgehouden gegeven voor (zie mark_gehoorverslag_generated() in
  // TriviasServer.py). De feitelijke tijdstempel wordt aan detailBits
  // toegevoegd (niet in de badge zelf) -- vandaar dat detailLine pas hierna
  // wordt opgebouwd.
  let badge;
  let rightSlot;
  const item = document.createElement("div");
  item.className = "session-item";
  if (!s.has_transcript) {
    // "Geen transcriptie" i.p.v. "wordt verwerkt" -- we weten hier niet of
    // er nu echt een achtergrondtaak loopt, alleen dat het bestand ontbreekt.
    badge = `<span class="session-item-badge session-item-badge-pending">Geen transcriptie</span>`;
    // Geen "Open"-aanduiding voor een kaart die niet te openen is -- i.p.v.
    // gewoon leeg, expliciet uitleggen waarom.
    rightSlot = `<span class="session-item-action-note">Nog niet te openen</span>`;
    item.classList.add("session-item-disabled");
  } else {
    rightSlot = `<span class="session-item-open">Open →</span>`;
    if (s.gehoorverslag_generated_at) {
      badge = `<span class="session-item-badge session-item-badge-done">Verslag gegenereerd</span>`;
      detailBits.push(`gegenereerd op ${new Date(s.gehoorverslag_generated_at).toLocaleString("nl-NL")}`);
    } else {
      badge = `<span class="session-item-badge session-item-badge-missing">Verslag nog niet gegenereerd</span>`;
    }
  }

  const detailLine = `<svg class="icon"><use href="#icon-calendar"></use></svg> ${detailBits.map(escapeHtml).join(" · ")}`;

  item.innerHTML = `
    <div class="session-item-main">
      <span class="session-item-title">${title}</span>
      ${rightSlot}
    </div>
    <div class="session-item-detail">${detailLine}</div>
    <div class="session-item-meta-line">${metaLine}</div>
    ${badge}`;

  if (s.has_transcript) {
    item.addEventListener("click", () => {
      if (!confirmLeaveActiveRecording()) return;
      showWorkspace();
      loadSessionTranscript(s.session_id);
    });
  }
  return item;
}

function showLandingPage() {
  if (landingPage) landingPage.classList.remove("hidden");
  if (workspaceMain) workspaceMain.classList.add("hidden");
  // Model/taal zijn relevant in de werkomgeving, niet op de startpagina.
  document.getElementById("appHeaderTags")?.classList.add("hidden");
  setLandingView("overview");
  loadLandingData();
}

function showWorkspace() {
  if (landingPage) landingPage.classList.add("hidden");
  if (workspaceMain) workspaceMain.classList.remove("hidden");
  document.getElementById("appHeaderTags")?.classList.remove("hidden");
  // Geen navitem blijft "actief" ogen als je hier via een sessiekaart bent
  // beland (i.p.v. via "Nieuwe opname") -- startNewSessionFlow() zet die
  // markering zelf terug aan wanneer dat wél de aanleiding was.
  document.querySelectorAll(".app-nav-item").forEach(b => b.classList.remove("active"));
}

// Haalt altijd het gemergde transcript van ALLE kanalen van een sessie op
// (chronologisch door elkaar, zoals de live-weergave al deed) -- per-kanaal
// bekijken kan achteraf via de filter-chips, i.p.v. vooraf te moeten kiezen
// en de andere kanalen helemaal niet te zien.
async function loadSessionTranscript(sessionId) {
  try {
    const resp = await fetch(`/sessions/${encodeURIComponent(sessionId)}/transcript?channel_id=all`);
    if (!resp.ok) { alert("Transcript niet gevonden."); return; }
    const data = await resp.json();

    currentSessionId   = data.session_id;
    isPlaybackMode     = true;
    playbackChannels   = data.channels || [];
    playbackActiveChannels = new Set(playbackChannels);
    playbackDurationMs = data.duration_ms || 0;
    playbackSessionDate = data.date || null;
    playbackCaseRef = data.case_ref || null;
    playbackPersonRef = data.person_ref || null;
    playbackLanguages = data.languages || {};
    playbackLanguages2 = data.languages2 || {};
    playbackGehoorverslagGeneratedAt = data.gehoorverslag_generated_at || null;
    playbackCreatedAt = data.created_at || null;

    playbackLines       = [];
    playbackLineById    = new Map();
    playbackSentenceMap = new Map();

    for (const seg of (data.segments || [])) {
      const id = seg.id;
      if (!id) continue;
      const channelId = seg.channel_id || "default";
      const sentMatch = id.match(/^(.+)_s(\d+)$/);
      if (sentMatch) {
        const parentId = sentMatch[1];
        if (!playbackSentenceMap.has(parentId)) {
          playbackSentenceMap.set(parentId, []);
          if (!playbackLineById.has(parentId)) {
            const parent = { id: parentId, text: "", text_batch: null, state: "FINAL", start_ms: seg.start_ms || 0, end_ms: seg.end_ms || 0, speaker: -1, channelId };
            playbackLines.push(parent);
            playbackLineById.set(parentId, parent);
          }
        }
        playbackSentenceMap.get(parentId).push({
          id, text: seg.text_final || seg.text_batch || "", text_batch: seg.text_batch || null,
          state: "FINAL", start_ms: seg.start_ms || 0, end_ms: seg.end_ms || 0, speaker: -1, channelId,
        });
      } else {
        const line = { id, text: seg.text_final || seg.text_batch || "", text_batch: seg.text_batch || null, state: "FINAL", start_ms: seg.start_ms || 0, end_ms: seg.end_ms || 0, speaker: -1, channelId };
        playbackLines.push(line);
        playbackLineById.set(id, line);
      }
    }

    for (const [pid, sents] of playbackSentenceMap.entries()) {
      sents.sort((a, b) => a.start_ms - b.start_ms);
    }

    renderPlaybackChannelFilter();
    renderPlaybackFiltered();
    renderPlaybackTimeline();
    updateRefreshPlaybackButtonUI();
    updateLiveVsPlaybackUI();
    setAsrStatus(`Sessie geladen: ${sessionId.substring(0, 12)}…`);
  } catch (e) {
    alert("Fout bij laden transcript: " + e.message);
  }
}

// Bouwt de per-kanaal filter-chips boven het gemergde sessie-transcript.
// Bij één kanaal (geen echte multi-channel sessie) toont het geen filter --
// niets om te filteren.
function renderPlaybackChannelFilter() {
  const el = document.getElementById("playbackChannelFilter");
  if (!el) return;
  if (!isPlaybackMode || playbackChannels.length <= 1) {
    el.classList.add("hidden");
    el.innerHTML = "";
    return;
  }

  const chips = playbackChannels.map(ch => {
    const roleId = channelIdToRoleId(ch);
    const label = getRoleLabel(roleId);
    const color = getRoleColor(roleId);
    const checked = playbackActiveChannels.has(ch) ? "checked" : "";
    return `<label class="channel-filter-chip">
      <input type="checkbox" data-channel="${escapeHtml(ch)}" ${checked} />
      <span class="channel-filter-dot" style="background:${color}"></span>
      <span>${escapeHtml(label)}</span>
    </label>`;
  }).join("");

  el.innerHTML = `<span class="channel-filter-label">Toon:</span>${chips}`;
  el.classList.remove("hidden");

  el.querySelectorAll("input[type=checkbox]").forEach(cb => {
    cb.addEventListener("change", () => {
      const ch = cb.dataset.channel;
      if (cb.checked) playbackActiveChannels.add(ch);
      else playbackActiveChannels.delete(ch);
      renderPlaybackFiltered();
    });
  });
}

function hidePlaybackChannelFilter() {
  const el = document.getElementById("playbackChannelFilter");
  if (el) { el.classList.add("hidden"); el.innerHTML = ""; }
}

// === Playback-tijdlijn (terugluisteren van een sessie) ===
// Kleurbanden per kanaal op basis van de al-bekende segment-timestamps --
// bewust géén canvas/echte waveform-peaks, zie CLAUDE.md "live vs. batch" en
// de mockup-discussie in Log/AUDIOTIMELINE.png: functioneel (transcriptiesegment
// ↔ tijdstip ↔ audiofragment koppelen), niet decoratief.

// Zelfde flattening als renderPlaybackFiltered(), maar zonder het
// kanaalfilter -- de tijdlijn toont altijd alle kanalen voor oriëntatie,
// los van welke kanalen momenteel als tekst zichtbaar zijn.
function getAllPlaybackSegmentsFlat() {
  return playbackLines
    .map(l => {
      const sents = playbackSentenceMap.get(l.id);
      return sents ? sents : [l];
    })
    .flat();
}

function getPlaybackDurationMs() {
  if (playbackDurationMs > 0) return playbackDurationMs;
  const segments = getAllPlaybackSegmentsFlat();
  return Math.max(1000, ...segments.map(s => s.end_ms || 0), 1000);
}

function renderPlaybackTimeline() {
  const wrap   = document.getElementById("playbackTimeline");
  const labels = document.getElementById("playbackTimelineLabels");
  const list   = document.getElementById("playbackTimelineLanesList");
  const player = document.getElementById("playbackAudioPlayer");
  if (!wrap || !list || !labels) return;

  if (!isPlaybackMode || playbackChannels.length === 0) {
    hidePlaybackTimeline();
    return;
  }

  const segments   = getAllPlaybackSegmentsFlat();
  const durationMs = getPlaybackDurationMs();

  const scrubber = document.getElementById("playbackScrubber");
  if (scrubber && !isScrubbingPlayback) scrubber.max = String(durationMs);

  labels.innerHTML = "";
  list.innerHTML = "";
  for (const channelId of playbackChannels) {
    const roleId = channelIdToRoleId(channelId);
    const color  = getRoleColor(roleId);

    const labelEl = document.createElement("div");
    labelEl.className = "playback-timeline-label";
    labelEl.style.color = color;
    labelEl.textContent = getRoleLabel(roleId);
    labels.appendChild(labelEl);

    const lane = document.createElement("div");
    lane.className = "playback-timeline-lane";
    lane.dataset.channel = channelId;

    for (const seg of segments) {
      if (seg.channelId !== channelId) continue;
      const startMs = seg.start_ms || 0;
      const endMs   = Math.max(seg.end_ms || 0, startMs + 200); // minimale zichtbare breedte
      const left  = Math.min(100, (startMs / durationMs) * 100);
      const width = Math.max(0.3, ((endMs - startMs) / durationMs) * 100);
      const block = document.createElement("div");
      block.className = "playback-timeline-segment";
      block.dataset.segId   = seg.id || "";
      block.dataset.startMs = String(startMs);
      block.dataset.endMs   = String(endMs);
      block.style.left  = left + "%";
      block.style.width = width + "%";
      block.style.background = color;
      block.title = `[${formatMs(startMs)}] ${getRoleLabel(roleId)}`;
      lane.appendChild(block);
    }
    list.appendChild(lane);
  }

  wrap.classList.remove("hidden");

  const timeLabelEl = document.getElementById("playbackTimeLabel");
  if (timeLabelEl && !isScrubbingPlayback) {
    const posMs = playbackAudioChannel ? (playbackAudioSliceStartMs + (player?.currentTime || 0) * 1000) : 0;
    timeLabelEl.textContent = `${formatMs(posMs)} / ${formatMs(durationMs)}`;
  }

  if (player && player.dataset.wired !== "1") {
    player.dataset.wired = "1";
    player.addEventListener("timeupdate", () => {
      if (!isPlaybackMode || !playbackAudioChannel) return;
      const absMs = playbackAudioSliceStartMs + player.currentTime * 1000;
      movePlayheadTo(absMs);
      highlightSegmentAt(playbackAudioChannel, absMs);
      if (timeLabelEl && !isScrubbingPlayback) {
        timeLabelEl.textContent = `${formatMs(absMs)} / ${formatMs(getPlaybackDurationMs())}`;
      }
      if (scrubber && !isScrubbingPlayback) scrubber.value = String(absMs);
    });

    const playPauseBtn = document.getElementById("playbackPlayPause");
    if (playPauseBtn) {
      const setIcon = paused => {
        playPauseBtn.innerHTML = `<svg class="icon"><use href="#${paused ? "icon-play" : "icon-pause"}"></use></svg>`;
      };
      playPauseBtn.addEventListener("click", () => {
        if (!isPlaybackMode) return;
        if (!playbackAudioChannel) {
          // Nog geen kanaal geladen (nog niet op een regel/baan geklikt) --
          // begin gewoon vanaf het begin van het eerste zichtbare kanaal,
          // zelfde als klikken op tijdstip 0 van die baan.
          if (playbackChannels.length > 0) playFromChannelAt(playbackChannels[0], 0);
          return;
        }
        if (player.paused) player.play().catch(() => {});
        else player.pause();
      });
      player.addEventListener("play",  () => setIcon(false));
      player.addEventListener("pause", () => setIcon(true));
      player.addEventListener("ended", () => setIcon(true));
    }

    if (scrubber) {
      scrubber.addEventListener("input", () => {
        isScrubbingPlayback = true;
        if (timeLabelEl) timeLabelEl.textContent = `${formatMs(Number(scrubber.value))} / ${formatMs(getPlaybackDurationMs())}`;
      });
      scrubber.addEventListener("change", () => {
        const targetMs = Number(scrubber.value);
        const channel = playbackAudioChannel || playbackChannels[0];
        isScrubbingPlayback = false;
        if (channel) playFromChannelAt(channel, targetMs);
      });
    }

    const volumeInput = document.getElementById("playbackVolume");
    if (volumeInput) {
      volumeInput.addEventListener("input", () => {
        player.volume = Number(volumeInput.value);
      });
    }
  }
}

function hidePlaybackTimeline() {
  const wrap   = document.getElementById("playbackTimeline");
  const labels = document.getElementById("playbackTimelineLabels");
  const list   = document.getElementById("playbackTimelineLanesList");
  const player = document.getElementById("playbackAudioPlayer");
  if (player) player.pause();
  if (wrap) wrap.classList.add("hidden");
  if (labels) labels.innerHTML = "";
  if (list) list.innerHTML = "";
  playbackAudioChannel = null;
  isScrubbingPlayback = false;
  const playPauseBtn = document.getElementById("playbackPlayPause");
  if (playPauseBtn) playPauseBtn.innerHTML = '<svg class="icon"><use href="#icon-play"></use></svg>';
  const timeLabelEl = document.getElementById("playbackTimeLabel");
  if (timeLabelEl) timeLabelEl.textContent = "00:00 / 00:00";
  const scrubber = document.getElementById("playbackScrubber");
  if (scrubber) scrubber.value = "0";
  movePlayheadTo(-1);
}

// Laadt (een deel van) de WAV van `channelId` vanaf `atMs` in de ingebouwde
// speler en start afspelen -- geen sleepbaar scrubben over de hele sessie
// (zie plan: /audio ondersteunt geen HTTP Range, dus vooraf de volledige WAV
// van elk kanaal laden zou bij lange zittingen te veel data zijn). Klikken
// vraagt telkens een nieuwe slice op vanaf het gekozen moment tot het einde.
function playFromChannelAt(channelId, atMs) {
  if (!currentSessionId || !channelId) return;
  const player = document.getElementById("playbackAudioPlayer");
  if (!player) return;
  const ms = Math.max(0, Math.round(atMs));
  playbackAudioChannel = channelId;
  playbackAudioSliceStartMs = ms;
  player.src = `/audio/${encodeURIComponent(currentSessionId)}/${encodeURIComponent(channelId)}?start_ms=${ms}`;
  player.play().catch(() => {});
  movePlayheadTo(ms);

  // Directe feedback i.p.v. te wachten op de eerste timeupdate-tick van de
  // net herladen speler (die duurt soms een fractie na een nieuwe src).
  const scrubber = document.getElementById("playbackScrubber");
  if (scrubber) scrubber.value = String(ms);
  const timeLabelEl = document.getElementById("playbackTimeLabel");
  if (timeLabelEl) timeLabelEl.textContent = `${formatMs(ms)} / ${formatMs(getPlaybackDurationMs())}`;
}

function movePlayheadTo(atMs) {
  const playhead = document.getElementById("playbackTimelinePlayhead");
  if (!playhead) return;
  if (atMs < 0) { playhead.style.display = "none"; return; }
  const durationMs = getPlaybackDurationMs();
  const pct = Math.min(100, Math.max(0, (atMs / durationMs) * 100));
  playhead.style.left = pct + "%";
  playhead.style.display = "block";
}

// Highlight het transcriptiesegment (en het bijbehorende blokje op de
// tijdlijn) dat overeenkomt met de huidige afspeelpositie -- gebruikt
// dezelfde data-start-ms/data-end-ms/data-channel attributen die
// renderTranscript() al op elk .seg-clickable-element zet, en de identieke
// start/end-tijden die renderPlaybackTimeline() op elk tijdlijn-blokje zet.
function highlightSegmentAt(channelId, absMs) {
  if (liveTranscriptDiv) {
    const prev = liveTranscriptDiv.querySelector(".seg-now-playing");
    if (prev) prev.classList.remove("seg-now-playing");
    for (const el of liveTranscriptDiv.querySelectorAll(".seg-clickable")) {
      if (el.dataset.channel !== channelId) continue;
      const s  = parseInt(el.dataset.startMs || "0", 10);
      const en = parseInt(el.dataset.endMs || "0", 10) || (s + 1);
      if (absMs >= s && absMs < en) {
        el.classList.add("seg-now-playing");
        el.scrollIntoView({ block: "nearest" });
        break;
      }
    }
  }

  const list = document.getElementById("playbackTimelineLanesList");
  if (list) {
    const prevBlock = list.querySelector(".seg-playing");
    if (prevBlock) prevBlock.classList.remove("seg-playing");
    const lane = list.querySelector(`.playback-timeline-lane[data-channel="${CSS.escape(channelId)}"]`);
    if (lane) {
      for (const block of lane.querySelectorAll(".playback-timeline-segment")) {
        const s  = parseInt(block.dataset.startMs || "0", 10);
        const en = parseInt(block.dataset.endMs || "0", 10) || (s + 1);
        if (absMs >= s && absMs < en) {
          block.classList.add("seg-playing");
          break;
        }
      }
    }
  }
}

// Toont de "Ververs Transcriptie"-knop boven het transcript alleen wanneer we
// daadwerkelijk een specifieke (net gestopte of eerder opgenomen) sessie bekijken --
// currentSessionId moet dan bekend zijn.
function updateRefreshPlaybackButtonUI() {
  if (!refreshPlaybackButton) return;
  refreshPlaybackButton.classList.toggle("hidden", !(isPlaybackMode && currentSessionId));
  if (gehoorverslagButton) {
    gehoorverslagButton.classList.toggle("hidden", !(isPlaybackMode && currentSessionId));
  }
}

// Downloadt het (v1, letterlijke) gehoorverslag als .docx voor de huidig
// bekeken sessie. Geen fetch/blob nodig -- de browser handelt de
// Content-Disposition: attachment-header van GET /sessions/{id}/gehoorverslag
// vanzelf af.
function downloadGehoorverslag() {
  if (!currentSessionId) return;
  window.location.href = `/sessions/${encodeURIComponent(currentSessionId)}/gehoorverslag`;
}

// Rendert playbackLines, gefilterd op de momenteel aangevinkte kanalen.
function renderPlaybackFiltered() {
  const renderLines = playbackLines
    .filter(l => playbackActiveChannels.has(l.channelId))
    .map(l => {
      const sents = playbackSentenceMap.get(l.id);
      return sents ? sents : [l];
    })
    .flat();

  if (liveTranscriptDiv) renderTranscript(renderLines, "", "", "active_transcription");
}

// === Klik op segment → terugluisteren ===

if (liveTranscriptDiv) {
  liveTranscriptDiv.addEventListener("click", async e => {
    const translateIcon = e.target.closest(".seg-translate");
    if (translateIcon) {
      e.stopPropagation();
      await handleTranslateClick(translateIcon);
      return;
    }

    const seg = e.target.closest(".seg-clickable");
    if (!seg) return;
    const startMs = parseInt(seg.dataset.startMs || "0", 10);
    const endMs   = parseInt(seg.dataset.endMs   || "0", 10);
    const session = seg.dataset.session;
    const channel = seg.dataset.channel;
    if (!session) return;

    // In playback-mode hergebruiken we de ingebouwde speler + tijdlijn-playhead
    // uit #playbackTimeline (zie renderPlaybackTimeline()) i.p.v. steeds een
    // nieuwe losse popup te spawnen -- zelfde klik-to-listen, maar nu zichtbaar
    // gekoppeld aan de kanaalbanden en met transcript-highlight tijdens afspelen.
    if (isPlaybackMode) {
      playFromChannelAt(channel, startMs);
      return;
    }

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

// Klik op de tijdlijn zelf (niet op een transcriptsegment) -- springt naar
// dat moment op het aangeklikte kanaal.
const playbackTimelineLanesList = document.getElementById("playbackTimelineLanesList");
if (playbackTimelineLanesList) {
  playbackTimelineLanesList.addEventListener("click", e => {
    const lane = e.target.closest(".playback-timeline-lane");
    if (!lane) return;
    const rect = lane.getBoundingClientRect();
    const frac = rect.width > 0 ? Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)) : 0;
    const targetMs = Math.round(frac * getPlaybackDurationMs());
    playFromChannelAt(lane.dataset.channel, targetMs);
  });
}

// === Vertaling niet-NL tekst (on-demand, zie features/vertaling-niet-nl-tekst.md) ===

// Client-side cache binnen dit paginabezoek -- niet persistent, geen sessionStorage/
// localStorage: een vertaling is een AI-afleiding, geen onderdeel van de autoritatieve
// transcriptie (CLAUDE.md "audio is authoritative"). Alleen bedoeld om een dubbele klik
// op dezelfde regel geen tweede LLM-call te laten kosten.
const translationCache = new Map();

async function handleTranslateClick(iconEl) {
  const seg = iconEl.closest(".seg-clickable");
  if (!seg) return;

  const rawText = iconEl.dataset.rawText || "";
  const channel = seg.dataset.channel || "default";
  const session = seg.dataset.session || currentSessionId || "";
  if (!rawText.trim()) return;

  const existing = seg.querySelector(".seg-translation, .seg-translation-error");
  if (existing) {
    existing.remove();
    return;
  }

  const cacheKey = `${channel}::${rawText}`;
  if (translationCache.has(cacheKey)) {
    insertTranslationLine(seg, translationCache.get(cacheKey));
    return;
  }

  // Laadstatus via een class (spin-animatie op het bestaande icoontje, zie
  // style.css) i.p.v. het icoontje te vervangen -- iconEl bevat nu een
  // <svg>, dus textContent overschrijven zou die weer vervangen door platte
  // tekst.
  iconEl.classList.add("loading");
  try {
    const resp = await fetch("/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: rawText, channel_id: channel, session_id: session }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || !data.translation) {
      insertTranslationLine(seg, null, data.error || "Vertalen mislukt");
      return;
    }
    translationCache.set(cacheKey, data.translation);
    insertTranslationLine(seg, data.translation);
  } catch (err) {
    insertTranslationLine(seg, null, "Vertalen mislukt (verbindingsfout)");
  } finally {
    iconEl.classList.remove("loading");
  }
}

function insertTranslationLine(seg, translation, errorMessage) {
  const existing = seg.querySelector(".seg-translation, .seg-translation-error");
  if (existing) existing.remove();

  // Terugluisteren krijgt een neutrale "Vertaling"-aanduiding i.p.v. het
  // gekleurde globe-icoon + cursief -- zodat het leest als een rustige
  // vervolgregel van dezelfde spreker, niet als een nieuwe/opvallende
  // gebeurtenis. Het live-scherm behoudt de bestaande opmaak ongewijzigd.
  const line = document.createElement("div");
  if (translation) {
    line.className = isPlaybackMode ? "seg-translation doc-translation" : "seg-translation";
    line.innerHTML = isPlaybackMode
      ? `<span class="doc-translation-tag">Vertaling</span>${escapeHtml(translation)}`
      : `<svg class="icon"><use href="#icon-globe"></use></svg> ${escapeHtml(translation)}`;
  } else {
    line.className = "seg-translation-error";
    line.innerHTML = `<svg class="icon"><use href="#icon-warning"></use></svg> ${escapeHtml(errorMessage || "Vertalen mislukt")}`;
  }
  seg.appendChild(line);
}

// === Event wiring ===

// #recordButton is hetzelfde element in twee contexten: "Start" in het
// Bediening-tabblad (start meteen op met de al ingestelde kanaalconfiguratie
// -- correct in die context) vs. "Nieuwe opname starten" in de read-only
// sessiesamenvatting tijdens terugluisteren (zie updateRecordButtonUI()).
// Dat laatste moet NIET blind opnemen met een willekeurige oude configuratie
// -- de gebruiker heeft nog niets voor déze nieuwe opname ingesteld. Zelfde
// bestemming als de "Nieuwe opname"-hoofdnavigatie: naar Configuratie, niet
// direct starten.
if (recordButton) {
  recordButton.addEventListener("click", () => {
    if (isViewingStoredSession()) startNewSessionFlow();
    else startRecording();
  });
}

const startFromConfigBtn = document.getElementById("startFromConfigBtn");
if (startFromConfigBtn) {
  startFromConfigBtn.addEventListener("click", async () => {
    await startRecording();
    // Springt na het starten zelf naar Bediening, zodat de live-status en
    // Pauze/Stop meteen zichtbaar zijn i.p.v. dat je handmatig moet wisselen.
    document.querySelector('.tab-btn[data-tab="control"]')?.click();
  });
}
if (pauseButton) pauseButton.addEventListener("click", togglePause);
if (stopButton) stopButton.addEventListener("click", confirmAndStop);
if (refreshButton) refreshButton.addEventListener("click", refreshTranscript);
if (refreshPlaybackButton) refreshPlaybackButton.addEventListener("click", refreshTranscript);
if (gehoorverslagButton) gehoorverslagButton.addEventListener("click", downloadGehoorverslag);

// Een lopende opname mag niet ongemerkt buiten beeld raken: de WebSocket-
// verbindingen zijn onafhankelijk van welke view zichtbaar is, dus wegnavigeren
// stopt de opname NIET -- maar Pauze/Stop zijn dan niet meer bereikbaar zonder
// terug te navigeren. Risicovol tijdens een gehoor, dus expliciete bevestiging
// i.p.v. stilzwijgend wegklikken (zelfde confirm()-patroon als confirmAndStop()).
function confirmLeaveActiveRecording() {
  if (!isRecording) return true;
  return confirm(
    "Er loopt nog een actieve opname.\n\n" +
    "Wegnavigeren stopt de opname niet, maar Pauze/Stop zijn dan niet meer " +
    "in beeld totdat je terugkeert naar de werkomgeving.\n\nToch doorgaan?"
  );
}

// Gedeeld door de hero-knop op Werkoverzicht en de "Nieuwe opname"-navitem.
function startNewSessionFlow() {
  if (!confirmLeaveActiveRecording()) return;
  resetSessionRefs();
  showWorkspace();
  document.querySelectorAll('.app-nav-item[data-view="new"]').forEach(b => b.classList.add("active"));
  document.querySelector('.tab-btn[data-tab="config"]')?.click();
}

if (newSessionBtn) newSessionBtn.addEventListener("click", startNewSessionFlow);

// Hoofdnavigatie: Werkoverzicht/Sessies wisselen binnen de startpagina (geen
// nieuwe fetch nodig, zie setLandingView()); "Nieuwe opname" gaat naar de
// werkomgeving. Vervangt de vorige losse "◄ Overzicht"-knop -- die had geen
// functie meer nu de navigatie er is.
document.querySelectorAll(".app-nav-item[data-view]").forEach(btn => {
  btn.addEventListener("click", () => {
    const view = btn.dataset.view;
    if (view === "new") { startNewSessionFlow(); return; }
    if (!confirmLeaveActiveRecording()) return;
    showLandingPage();
    setLandingView(view);
  });
});

const viewAllSessionsBtn = document.getElementById("viewAllSessionsBtn");
if (viewAllSessionsBtn) {
  viewAllSessionsBtn.addEventListener("click", () => {
    document.querySelector('.app-nav-item[data-view="sessions"]')?.click();
  });
}

if (landingSearchInput) {
  landingSearchInput.addEventListener("input", () => {
    landingPageIndex = 0;
    renderLandingList();
  });
}

document.querySelectorAll(".landing-filter").forEach(btn => {
  btn.addEventListener("click", () => applyLandingFilter(btn.dataset.filter));
});

const landingSortSelect = document.getElementById("landingSortSelect");
if (landingSortSelect) {
  landingSortSelect.addEventListener("change", () => {
    landingSortMode = landingSortSelect.value;
    landingPageIndex = 0;
    renderLandingList();
  });
}

const addChannelBtn = document.getElementById("addChannelBtn");
if (addChannelBtn) addChannelBtn.addEventListener("click", addChannelConfig);

const crossGateToggle = document.getElementById("crossGateToggle");
if (crossGateToggle) {
  crossGateToggle.addEventListener("change", () => {
    crossGateEnabled = crossGateToggle.checked;
    saveCrossGateEnabled();
  });
}

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
loadCrossGateEnabled();
if (crossGateToggle) crossGateToggle.checked = crossGateEnabled;
initTabs();
renderChannelConfigs();
checkMicPermission();
updateRecordButtonUI();
updateHint();
setConnectionStatus(0, channelConfigs.length);
setAsrStatus("Wachten op opname");

// Startpagina is de landingsplek i.p.v. direct het live-scherm (index.html
// toont #landingPage al zichtbaar / .app-main al hidden by default -- dit
// vult 'm alleen met data, wisselt geen zichtbaarheid).
loadLandingData();
