// VoxFlow — browser side.
// It records the microphone, sends audio or text to the Python backend, and shows the results.
// All speech recognition and correction happens in Python; this file only displays things.

// ---------------------------------------------------------------- saved data (browser storage)

const LANGUAGES = [
  { code: "en-US", name: "English" },
  { code: "hi-IN", name: "Hindi" },
  { code: "mr-IN", name: "Marathi" },
  { code: "bn-IN", name: "Bengali" },
  { code: "pa-IN", name: "Punjabi" },
];
const MAX_SESSIONS = 50;

function load(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch (e) {
    return fallback;
  }
}
function save(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* storage full or blocked */ }
}

const state = {
  sessions: load("voxflow-sessions", load("linguaflow-sessions", [])),          // newest first
  currentId: null,
  settings: Object.assign({ language: "en-US", autoSpeakBack: true, intensity: 50 },
                          load("voxflow-settings", load("linguaflow-settings", {}))),
  metrics: null,                                       // benchmark results from the backend
};
if (state.settings.language === "auto") state.settings.language = "en-US";   // Auto-detect was removed
state.currentId = state.sessions.length ? state.sessions[0].id : null;

function currentSession() {
  return state.sessions.find((s) => s.id === state.currentId) || null;
}
function saveSessions() { save("voxflow-sessions", state.sessions); }
function saveSettings() { save("voxflow-settings", state.settings); }

// ---------------------------------------------------------------- small helpers

const $ = (id) => document.getElementById(id);

function toast(message) {
  const box = $("toast");
  box.textContent = message;
  box.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (box.hidden = true), 3500);
}

function languageName(code) {
  const lang = LANGUAGES.find((l) => l.code === code);
  return lang ? lang.name : code;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Same rule as the backend: no space before closing punctuation like "." or "।".
const CLOSING = new Set([".", ",", "!", "?", ";", ":", "।", "॥", ")", "]", "}", '"', "'", "”", "’"]);
const OPENING = new Set(["(", "[", "{", "“", "‘"]);
function joinTokens(tokens) {
  let out = "";
  for (const tok of tokens) {
    if (!out) out = tok;
    else if (CLOSING.has(tok) || OPENING.has(out[out.length - 1])) out += tok;
    else out += " " + tok;
  }
  return out;
}

// The corrected text, built from the user's accept/reject choices.
// A rejected correction puts the original words back.
function finalText(session) {
  const tokens = [];
  for (const segment of session.segments) {
    if (segment.kind === "same") {
      tokens.push(...segment.tokens);
    } else {
      const c = session.corrections.find((x) => x.id === segment.id);
      tokens.push(...(c.accepted === false ? c.original : c.corrected));
    }
  }
  return joinTokens(tokens);
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return readResponse(response);
}

async function readResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return data;
}

// ---------------------------------------------------------------- page switching

function showPage(name) {
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("active", p.id === "page-" + name));
  document.querySelectorAll(".nav-link").forEach((b) => b.classList.toggle("active", b.dataset.page === name));
  if (name === "dashboard") renderDashboard();
  if (name === "output") renderOutput();
  if (name === "speak") renderSpeak();
  if (name === "settings") renderSettings();
}

document.querySelectorAll(".nav-link").forEach((button) => {
  button.addEventListener("click", () => showPage(button.dataset.page));
});

// ---------------------------------------------------------------- handling a backend result

function addSession(data) {
  const session = {
    id: String(Date.now()),
    time: Date.now(),
    input: data.input,                        // "audio" or "text"
    language: data.language,
    rawText: data.recognized_text,
    stage1Text: data.stage1_text,
    segments: data.segments,
    corrections: data.corrections.map((c) => Object.assign({ accepted: null }, c)),
    stage2: data.stage2,
    asrConfidence: data.asr_confidence,       // real value from Google, null for typed text
    preprocessed: data.preprocessed,
    audioSeconds: data.audio_seconds ?? null,
    asrEngine: data.asr_engine || "",
    uncertain: data.asr_uncertain || [],
    latency: data.latency_ms,
  };
  if (recordings.next) {
    recordings[session.id] = recordings.next;
    delete recordings.next;
  }
  state.sessions.unshift(session);
  state.sessions = state.sessions.slice(0, MAX_SESSIONS);
  state.currentId = session.id;
  saveSessions();
  renderInsights();
  showPage(state.settings.autoSpeakBack ? "speak" : "output");
}

// ---------------------------------------------------------------- Speech Input: recording

const recorder = { context: null, stream: null, node: null, analyser: null, chunks: [],
                   sampleRate: 48000, startedAt: 0, timer: null, status: "idle" };

function setRecordStatus(status) {
  recorder.status = status;
  const button = $("record-button");
  button.classList.toggle("recording", status === "recording");
  button.textContent = status === "recording" ? "■" : status === "idle" ? "●" : "…";
  button.disabled = status === "processing" || status === "starting";
  $("record-status").textContent = { idle: "Idle", starting: "Starting mic… wait", recording: "Recording… speak now",
                                     processing: "Processing…" }[status];
  $("clear-recording").disabled = status !== "recording";
  $("text-submit").disabled = status !== "idle";
}

async function startRecording() {
  setRecordStatus("starting");          // the button says "Starting mic…" until audio really flows
  try {
    // Raw microphone audio: the browser's noise suppression treats hiss-like sounds ("sh", "s", "f")
    // as noise and cuts them, which made "She" come out as "He". The recognisers handle noise themselves.
    recorder.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, noiseSuppression: false, echoCancellation: false, autoGainControl: false },
    });
  } catch (e) {
    setRecordStatus("idle");
    toast("Microphone access was denied.");
    return;
  }
  recorder.context = new AudioContext();
  await recorder.context.resume();       // Safari creates audio contexts paused
  recorder.sampleRate = recorder.context.sampleRate;
  await recorder.context.audioWorklet.addModule("/static/recorder-worklet.js");
  const source = recorder.context.createMediaStreamSource(recorder.stream);
  recorder.node = new AudioWorkletNode(recorder.context, "recorder");
  recorder.analyser = recorder.context.createAnalyser();
  recorder.analyser.fftSize = 512;
  recorder.chunks = [];
  // Safari only runs the recorder steadily if it is connected to the output, so connect it
  // through a muted gain node (you hear nothing).
  const mute = recorder.context.createGain();
  mute.gain.value = 0;
  source.connect(recorder.analyser);
  source.connect(recorder.node);
  recorder.node.connect(mute);
  mute.connect(recorder.context.destination);

  // Only say "Recording" once the first block of microphone audio has actually arrived.
  recorder.node.port.onmessage = (event) => {
    recorder.chunks.push(event.data);
    if (recorder.status === "starting") {
      recorder.startedAt = Date.now();
      recorder.timer = setInterval(updateTimer, 250);
      setRecordStatus("recording");
      drawWaveform();
    }
  };
}

function stopMicrophone() {
  clearInterval(recorder.timer);
  if (recorder.stream) recorder.stream.getTracks().forEach((t) => t.stop());
  if (recorder.context) recorder.context.close();
  recorder.stream = recorder.context = recorder.node = recorder.analyser = null;
}

async function stopRecording() {
  if (recorder.status !== "recording") return;
  setRecordStatus("processing");
  await new Promise((resolve) => setTimeout(resolve, 250));   // let the last audio blocks arrive
  stopMicrophone();
  const chunks = recorder.chunks;
  recorder.chunks = [];
  if (!chunks.length) {
    setRecordStatus("idle");
    toast("No audio was captured. Try again.");
    return;
  }
  setRecordStatus("processing");
  const wav = encodeWav(chunks, recorder.sampleRate);
  const form = new FormData();
  form.append("file", wav, "recording.wav");
  form.append("language", state.settings.language);
  try {
    // 1) speech -> text, shown at once; 2) the correction, which takes longer (the LLM)
    const heard = await readResponse(await fetch("/api/transcribe", { method: "POST", body: form }));
    $("record-status").textContent = `${heard.latency_ms.asr} ms: ${heard.stage1_text}` +
      (heard.llm_step ? " · checking the grammar…" : "");
    const data = await postJson("/api/text", { text: heard.recognized_text, language: heard.language,
                                               intensity: state.settings.intensity, heard });
    recordings.next = URL.createObjectURL(wav);
    addSession(data);
    toast("Done.");
  } catch (e) {
    toast(e.message);
  }
  setRecordStatus("idle");
}

function clearRecording() {
  stopMicrophone();
  recorder.chunks = [];
  $("timer").textContent = "00:00";
  setRecordStatus("idle");
  toast("Recording discarded.");
}

function updateTimer() {
  const seconds = Math.floor((Date.now() - recorder.startedAt) / 1000);
  $("timer").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  if (seconds >= 60) stopRecording();          // the backend accepts up to 60 seconds
}

// Live waveform from the real microphone signal.
function drawWaveform() {
  const canvas = $("waveform");
  const ctx = canvas.getContext("2d");
  const color = getComputedStyle(document.body).getPropertyValue("--accent");
  const data = new Uint8Array(256);
  const frame = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!recorder.analyser) return;
    recorder.analyser.getByteTimeDomainData(data);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = (i / (data.length - 1)) * canvas.width;
      const y = (v / 255) * canvas.height;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    requestAnimationFrame(frame);
  };
  frame();
}

// Pack the recorded samples into a 16-bit mono WAV file (the format the backend reads).
function encodeWav(chunks, sampleRate) {
  const length = chunks.reduce((n, c) => n + c.length, 0);
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  const text = (offset, s) => [...s].forEach((ch, i) => view.setUint8(offset + i, ch.charCodeAt(0)));
  text(0, "RIFF"); view.setUint32(4, 36 + length * 2, true); text(8, "WAVE");
  text(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, "data"); view.setUint32(40, length * 2, true);
  let offset = 44;
  for (const chunk of chunks) {
    for (const sample of chunk) {
      const s = Math.max(-1, Math.min(1, sample));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([view], { type: "audio/wav" });
}

$("record-button").addEventListener("click", () => {
  if (recorder.status === "idle") startRecording();
  else if (recorder.status === "recording") stopRecording();
});
$("clear-recording").addEventListener("click", clearRecording);

// "R" starts/stops recording, except while typing in a box.
document.addEventListener("keydown", (event) => {
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName);
  if (event.key.toLowerCase() !== "r" || typing || event.metaKey || event.ctrlKey) return;
  if (!$("page-record").classList.contains("active")) return;
  $("record-button").click();
});

// ---------------------------------------------------------------- Speech Input: typed text

$("text-submit").addEventListener("click", async () => {
  const text = $("text-input").value.trim();
  if (!text) return;
  setRecordStatus("processing");
  try {
    const data = await postJson("/api/text", { text, language: state.settings.language,
                                               intensity: state.settings.intensity });
    $("text-input").value = "";
    addSession(data);
  } catch (e) {
    toast(e.message);
  }
  setRecordStatus("idle");
});

// ---------------------------------------------------------------- language pickers

function fillLanguageSelect(select) {
  select.innerHTML = LANGUAGES.map((l) => `<option value="${l.code}">${l.name} (${l.code})</option>`).join("");
  select.value = state.settings.language;
  select.addEventListener("change", () => {
    state.settings.language = select.value;
    saveSettings();
    document.querySelectorAll(".language-select").forEach((s) => (s.value = select.value));
  });
}
document.querySelectorAll(".language-select").forEach(fillLanguageSelect);

// ---------------------------------------------------------------- Text Output

function renderOutput() {
  const session = currentSession();
  $("output-empty").hidden = !!session;
  $("output-body").hidden = !session;
  if (!session) return;

  $("detected-language").textContent = `Language: ${languageName(session.language)}`;
  $("raw-text").textContent = session.rawText;
  $("final-text").textContent = finalText(session);
  $("stage2-note").textContent = stage2Description(session.stage2);
  const rec = recordings[session.id];
  $("recording-box").hidden = !rec;
  if (rec) {
    $("recording-player").src = rec;
    $("recording-download").href = rec;
  }

  const list = $("corrections");
  if (!session.corrections.length) {
    list.innerHTML = '<p class="muted">No corrections were needed.</p>';
    return;
  }
  list.innerHTML = session.corrections.map((c) => {
    const cls = c.accepted === true ? "accepted" : c.accepted === false ? "rejected" : "";
    const before = c.original.length ? `<span class="old">${escapeHtml(joinTokens(c.original))}</span>` : "";
    const after = c.corrected.length ? `<span class="new">${escapeHtml(joinTokens(c.corrected))}</span>` : "";
    return `
      <div class="correction ${cls}">
        <div>
          <span class="tag ${c.type}">${c.type}</span>
          <span class="muted small">${c.source === "llm" ? "Stage 2 · LLM" : "Stage 1 · rule"}</span>
          <div>${before}${after}</div>
        </div>
        <div class="buttons">
          <button class="secondary" data-id="${c.id}" data-choice="accept" ${c.accepted === true ? "disabled" : ""}>Accept</button>
          <button class="secondary" data-id="${c.id}" data-choice="reject" ${c.accepted === false ? "disabled" : ""}>Reject</button>
        </div>
      </div>`;
  }).join("");
}

$("corrections").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-id]");
  if (!button) return;
  const session = currentSession();
  const correction = session.corrections.find((c) => c.id === button.dataset.id);
  correction.accepted = button.dataset.choice === "accept";
  saveSessions();
  renderOutput();
  renderInsights();
});

function stage2Description(stage2) {
  if (!stage2) return "";
  const model = stage2.model ? ` (${stage2.model})` : "";
  return {
    applied: `Stage 2 LLM${model} made contextual corrections.` + (stage2.reason ? ` ${stage2.reason}.` : ""),
    unchanged: stage2.reason ? `Stage 2 LLM${model}: ${stage2.reason}, so the text was kept.`
                             : `Stage 2 LLM${model} found nothing more to fix.`,
    rejected: `Stage 2 LLM${model} answer was not used: ${stage2.reason}.`,
    skipped: `Stage 2 skipped: ${stage2.reason}.`,
    unavailable: `Stage 2 LLM is not available, so only Stage 1 rules were applied. Is Ollama running?`,
  }[stage2.status] || "";
}

$("copy-text").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(finalText(currentSession()));
    toast("Copied.");
  } catch (e) {
    toast("Copy failed.");
  }
});

$("export-json").addEventListener("click", () => {
  const s = currentSession();
  const snapshot = {
    language: s.language, input: s.input, rawText: s.rawText, stage1Text: s.stage1Text,
    correctedText: finalText(s), asrConfidence: s.asrConfidence, stage2: s.stage2,
    latencyMs: s.latency,
    corrections: s.corrections.map((c) => ({
      type: c.type, source: c.source, original: joinTokens(c.original),
      corrected: joinTokens(c.corrected), accepted: c.accepted,
    })),
  };
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([JSON.stringify(snapshot, null, 2)], { type: "application/json" }));
  link.download = `voxflow-${s.id}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});

// ---------------------------------------------------------------- Speak Back

const speech = { key: null, url: null };
const recordings = {};   // session id -> URL of the recorded WAV (kept in memory for this visit)   // the last audio made, so it isn't fetched twice

async function renderSpeak() {
  const session = currentSession();
  $("speak-empty").hidden = !!session;
  $("speak-body").hidden = !session;
  if (!session) return;

  const text = finalText(session);
  $("speak-text").textContent = text;
  const key = session.language + "|" + text;   // re-fetch if accept/reject changed the text
  if (speech.key === key) return;

  const player = $("player");
  player.removeAttribute("src");
  $("speak-status").textContent = "Generating speech…";
  try {
    const response = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, language: session.language }),
    });
    if (!response.ok) await readResponse(response);
    if (speech.url) URL.revokeObjectURL(speech.url);
    speech.url = URL.createObjectURL(await response.blob());
    speech.key = key;
    player.src = speech.url;
    player.playbackRate = Number(document.querySelector(".speed.active").dataset.speed);
    $("speak-status").textContent = `${languageName(session.language)} voice · ${response.headers.get("X-Voice") || ""}`;
  } catch (e) {
    $("speak-status").textContent = "Speech generation failed: " + e.message;
  }
}

document.querySelectorAll(".speed").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".speed").forEach((b) => b.classList.toggle("active", b === button));
    $("player").playbackRate = Number(button.dataset.speed);
  });
});

// ---------------------------------------------------------------- Settings

function intensityHelp(value) {
  if (value < 34) return "Low: Stage 1 rules only, the LLM is not used.";
  if (value < 80) return "Normal: rules + LLM. Fixes grammar and misspelt words; words you said correctly are never swapped.";
  return "High: as Normal, and the LLM may also replace words the speech recogniser was unsure it heard.";
}

function renderSettings() {
  $("auto-speak").checked = state.settings.autoSpeakBack;
  $("intensity").value = state.settings.intensity;
  $("intensity-value").textContent = state.settings.intensity + "%";
  $("intensity-help").textContent = intensityHelp(state.settings.intensity);
  const n = state.sessions.length;
  $("session-count").textContent = `${n} session${n === 1 ? "" : "s"} saved in this browser.`;
  $("clear-sessions").disabled = n === 0;
}

$("auto-speak").addEventListener("change", (e) => {
  state.settings.autoSpeakBack = e.target.checked;
  saveSettings();
});
$("intensity").addEventListener("input", (e) => {
  state.settings.intensity = Number(e.target.value);
  saveSettings();
  renderSettings();
});
$("clear-sessions").addEventListener("click", () => {
  if (!confirm("Delete all saved sessions?")) return;
  state.sessions = [];
  state.currentId = null;
  saveSessions();
  renderSettings();
  renderInsights();
  toast("All sessions cleared.");
});

// ---------------------------------------------------------------- Insights panel

function renderInsights() {
  const s = currentSession();
  if (!s) {
    $("insights").innerHTML = '<p class="muted">Record a session to see insights.</p>';
    return;
  }
  const count = (type) => s.corrections.filter((c) => c.type === type && c.accepted !== false).length;
  const confidence = s.asrConfidence == null ? null : Math.round(s.asrConfidence * 100);
  const bench = state.metrics && state.metrics.per_language && state.metrics.per_language[s.language];
  const lat = s.latency || {};

  $("insights").innerHTML = `
    <div class="stat"><div class="label">Language</div>
      <div class="value">${languageName(s.language)}</div>
      <div class="muted small">${s.input} input</div></div>
    <div class="stat"><div class="label">ASR confidence${s.asrEngine ? " (" + escapeHtml(s.asrEngine) + ")" : ""}</div>
      <div class="value">${confidence == null ? "—" : confidence + "%"}</div>
      ${confidence == null ? `<div class="muted small">${s.input === "text" ? "typed text, no recognition" : "the recogniser gave no confidence score for this clip"}</div>`
                           : `<div class="bar"><div style="width:${confidence}%"></div></div>`}
      ${s.preprocessed ? '<div class="muted small">recognised after noise reduction + VAD</div>' : ""}
      ${(s.uncertain || []).length ? `<div class="muted small">unsure about: ${escapeHtml(s.uncertain.join(", "))}</div>` : ""}</div>
    <div class="stat"><div class="label">Word error rate (benchmark, ${languageName(s.language)})</div>
      ${bench ? `<div class="value">${pct(bench.wer.full)}</div>
                 <div class="muted small">raw ASR ${pct(bench.wer.asr)} → full pipeline, FLEURS, n=${bench.n}</div>`
              : '<div class="value">—</div><div class="muted small">run the evaluation to measure it</div>'}</div>
    ${s.audioSeconds != null ? `<div class="stat"><div class="label">Audio received</div>
      <div class="value">${s.audioSeconds.toFixed(1)} s</div></div>` : ""}
    <div class="stat"><div class="label">Processing latency</div>
      <div class="value">${lat.total ?? "—"} ms</div>
      <div class="muted small">ASR ${lat.asr ?? 0} · Stage 1 ${lat.stage1 ?? 0} · Stage 2 ${lat.stage2 ?? 0} ms</div></div>
    <div class="stat"><div class="label">Corrections (not rejected)</div>
      <div class="small"><span class="ok">+ Insertions</span> ${count("insertion")}</div>
      <div class="small"><span class="bad">− Deletions</span> ${count("deletion")}</div>
      <div class="small"><span class="warn">~ Modifications</span> ${count("modification")}</div></div>`;
}

function asrLabel(asr) {
  if (!asr) return "Google";
  if (typeof asr === "string") return asr;
  return Object.entries(asr).map(([code, engine]) => `${languageName(code)} ${engine}`).join(", ");
}

function pct(x) { return (x * 100).toFixed(1) + "%"; }

// ---------------------------------------------------------------- Dashboard

async function renderDashboard() {
  $("language-list").innerHTML = LANGUAGES.map((l) => `<span class="chip">${l.name} · ${l.code}</span>`).join("");

  const recent = state.sessions.slice(0, 5);
  $("recent-sessions").innerHTML = recent.length ? recent.map((s) => `
      <button class="secondary card" style="display:block;width:100%;text-align:left" data-session="${s.id}">
        ${escapeHtml(s.rawText.slice(0, 70))}${s.rawText.length > 70 ? "…" : ""}
        <div class="muted small">${languageName(s.language)} · ${s.corrections.length} corrections ·
          ${new Date(s.time).toLocaleString()}</div>
      </button>`).join("")
    : '<p class="muted">No sessions yet. Start recording to create your first session.</p>';

  try {
    const status = await (await fetch("/api/status")).json();
    const llm = status.stage2;
    const llmState = !llm.online ? ['bad', 'Offline', llm.chain ? 'Check the API keys in .env, or start Ollama'
                                                               : 'Start Ollama to enable Stage 2']
      : !llm.model_available ? ['warn', 'Model missing', `Run: ollama pull ${llm.model}`]
      : ['ok', 'Ready', llm.model];
    $("status-cards").innerHTML = `
      ${statusCard("Speech recognition", "ok", "Configured", status.asr.engine)}
      ${statusCard("Stage 1 rules", "ok", "Ready", status.stage1.languagetool ? "rules + LanguageTool" : "rule-based")}
      ${statusCard("Stage 2 grammar model", ...llmState)}
      ${statusCard("Speak Back", "ok", "Configured", status.tts.engine)}`;
  } catch (e) {
    $("status-cards").innerHTML = '<p class="bad">Backend not reachable.</p>';
  }
  renderBenchmark();
}

function statusCard(title, cls, label, detail) {
  return `<div class="card"><div class="muted small">${title}</div>
          <div class="${cls}" style="font-weight:600">${label}</div>
          <div class="muted small">${escapeHtml(detail)}</div></div>`;
}

function renderBenchmark() {
  const m = state.metrics;
  if (!m || !m.available) {
    $("benchmark").innerHTML = '<p class="muted">Not measured yet. Run <code>python -m eval.run_eval</code>.</p>';
    return;
  }
  const rows = Object.entries(m.per_language).map(([code, r]) => `
    <tr><td>${languageName(code)}</td><td>${r.n}</td><td>${pct(r.wer.asr)}</td><td>${pct(r.wer.stage1)}</td>
        <td>${pct(r.wer.stage2)}</td><td>${pct(r.wer.full)}</td></tr>`).join("");
  $("benchmark").innerHTML = `
    <div class="card"><table>
      <tr><th>Language</th><th>Clips</th><th>Raw ASR</th><th>+ Stage 1</th><th>Stage 2 only</th><th>Full pipeline</th></tr>
      ${rows}</table>
      <p class="muted small">Word error rate, lower is better. ${escapeHtml(m.dataset)}; speech recognition: ${escapeHtml(asrLabel(m.asr))}; Stage 2 model ${escapeHtml(m.model)}; measured ${escapeHtml(m.date)}.</p>
    </div>`;
}

$("recent-sessions").addEventListener("click", (event) => {
  const button = event.target.closest("[data-session]");
  if (!button) return;
  state.currentId = button.dataset.session;
  renderInsights();
  showPage("output");
});

// ---------------------------------------------------------------- start

fetch("/api/metrics").then((r) => r.json()).then((m) => {
  state.metrics = m;
  renderInsights();
  renderBenchmark();
}).catch(() => {});
setRecordStatus("idle");
renderInsights();
showPage("dashboard");
