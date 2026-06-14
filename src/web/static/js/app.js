const state = {
  songs: [],
  models: [],
};

function byId(id) {
  return document.getElementById(id);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  return response.json();
}

function selectedSong() {
  return byId("songSelect").value || "2001";
}

function selectedModel() {
  const index = byId("voiceSelect").selectedIndex;
  return state.models[index] || {};
}

function requestPayload() {
  const model = selectedModel();
  const maxPhrases = byId("maxPhrases").value.trim();
  return {
    song_id: selectedSong(),
    start_phrase: Number(byId("startPhrase").value || 1),
    max_phrases: maxPhrases ? Number(maxPhrases) : null,
    assembly_mode: byId("assemblyMode").value,
    normalize: byId("normalize").checked,
    infer_class: byId("inferClass").value || "cascade",
    f0_naturalize: byId("f0Naturalize").checked,
    voice_model: model.model_name || "",
    index_file: model.index_path || "",
    f0_method: byId("f0Method").value,
    f0_up_key: Number(byId("f0UpKey").value || 0),
    index_rate: Number(byId("indexRate").value || 0.5),
    protect: Number(byId("protect").value || 0.33),
    filter_radius: Number(byId("filterRadius").value || 3),
    resample_sr: Number(byId("resampleSr").value || 0),
  };
}

function setStatus(text, className = "") {
  const node = byId("systemStatus");
  node.textContent = text;
  node.className = `status-pill ${className}`;
}

function renderSongs() {
  const select = byId("songSelect");
  select.innerHTML = state.songs.map((song) => `<option value="${song.song_id}">${song.label}</option>`).join("");
  if (state.songs.length) {
    const defaultSong = state.songs.find((song) => song.song_id === "2001") || state.songs[0];
    select.value = defaultSong.song_id;
  }
  renderSongMeta();
}

function renderSongMeta() {
  const song = state.songs.find((item) => item.song_id === selectedSong());
  byId("songMeta").innerHTML = song
    ? [
        ["midi", song.midi_exists, song.midi_path],
        ["wav", song.wav_exists, song.wav_path],
        ["TextGrid", song.textgrid_exists, song.textgrid_path],
      ].map(([name, exists, path]) => `<div>${name}: <strong class="${exists ? "status-success" : "status-failed"}">${exists ? "存在" : "缺失"}</strong><br>${path}</div>`).join("")
    : "<div>未找到歌曲</div>";
}

function renderModels() {
  const select = byId("voiceSelect");
  select.innerHTML = state.models.map((model) => `<option value="${model.model_name}">${model.name}</option>`).join("");
  renderVoiceMeta();
}

function renderVoiceMeta() {
  const model = selectedModel();
  byId("voiceMeta").innerHTML = model.name
    ? [
        `<div>pth: ${model.pth_path}</div>`,
        `<div>index: <strong class="${model.has_index ? "status-success" : "status-failed"}">${model.has_index ? "存在" : "缺失"}</strong><br>${model.index_path || ""}</div>`,
        `<div>疑似 F0 模型: <strong class="${model.maybe_f0_model ? "status-success" : "warning"}">${model.maybe_f0_model ? "是" : "否"}</strong></div>`,
        model.warning ? `<div class="warning">${model.warning}</div>` : "",
      ].join("")
    : "<div>未找到 RVC 模型</div>";
}

async function loadChoices() {
  const [songs, models] = await Promise.all([api("/api/songs"), api("/api/voice-models")]);
  state.songs = songs.data.songs || [];
  state.models = models.data.models || [];
  renderSongs();
  renderModels();
}

async function runTask(path) {
  setStatus("Submitting", "status-running");
  const result = await api(path, { method: "POST", body: JSON.stringify(requestPayload()) });
  if (result.warnings && result.warnings.length) {
    setStatus(result.warnings[0], "warning");
  } else if (result.ok) {
    setStatus("Job started", "status-running");
  } else {
    setStatus(result.errors.join("; "), "status-failed");
  }
  await refreshAll();
}

function renderJobs(jobs) {
  byId("jobs").innerHTML = jobs.length
    ? jobs.map((job) => `
      <div class="job">
        <strong>${job.name}</strong>
        <span class="status-${job.status}">${job.status}</span>
        <div>${job.job_id}</div>
        <div>${job.command}</div>
        <div>${job.start_time || ""} ${job.end_time || ""}</div>
        <div>${job.log_path}</div>
      </div>
    `).join("")
    : "<div class='job'>暂无任务</div>";
}

function renderAudio(items) {
  byId("audioList").innerHTML = items.map((item) => `
    <div class="audio-item">
      <strong>${item.label}</strong>
      <div>${item.path}</div>
      ${item.exists && item.url ? `<audio controls src="${item.url}?t=${Date.now()}"></audio>` : "<div class='warning'>文件不存在</div>"}
    </div>
  `).join("");
}

function valueText(value) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function renderReports(data) {
  const reports = data.reports || {};
  byId("reportCards").innerHTML = Object.entries(reports).map(([name, report]) => {
    const summary = report.summary || {};
    const fields = ["status", "checkpoint_ready", "dataset", "song_id", "phrase_count", "selected_input_types", "infer_class", "assembly_mode", "duration_sec", "output_audio", "f0_source", "f0_naturalized", "audio_generated"];
    return `
      <div class="report-card">
        <h2>${name}</h2>
        <dl>${fields.map((field) => `<dt>${field}</dt><dd>${valueText(summary[field])}</dd>`).join("")}</dl>
      </div>
    `;
  }).join("");
  renderPhrases(data.segments || []);
}

function renderPhrases(segments) {
  const maxEnd = Math.max(1, ...segments.map((item) => Number(item.end || 0)));
  byId("phraseTimeline").innerHTML = segments.slice(0, 80).map((item) => {
    const start = Number(item.start || 0);
    const end = Number(item.end || start);
    const width = Math.max(1, ((end - start) / maxEnd) * 100);
    return `<div class="phrase-bar" title="${item.item_name || ""} ${start}-${end}"><div class="phrase-fill" style="width:${width}%">${item.phrase_id || ""} ${start.toFixed(2)}-${end.toFixed(2)}</div></div>`;
  }).join("");
  byId("phraseRows").innerHTML = segments.map((item) => {
    const stats = item.f0_stats || {};
    const mel = item.mel_stats || {};
    const start = Number(item.start || 0);
    const end = Number(item.end || start);
    return `<tr>
      <td>${valueText(item.phrase_id)}</td>
      <td>${valueText(item.item_name)}</td>
      <td>${valueText(item.start)}</td>
      <td>${valueText(item.end)}</td>
      <td>${(end - start).toFixed(3)}</td>
      <td>${valueText(item.f0_source)}</td>
      <td>${valueText(stats.min)}</td>
      <td>${valueText(stats.max)}</td>
      <td>${valueText(stats.mean)}</td>
      <td>${valueText(mel.mean)}</td>
      <td>${valueText(item.output_wav)}</td>
    </tr>`;
  }).join("");
}

async function refreshAll() {
  const [status, jobs, audio, reports, logs] = await Promise.all([
    api("/api/status"),
    api("/api/jobs"),
    api(`/api/audio/list?song_id=${encodeURIComponent(selectedSong())}`),
    api("/api/reports/latest"),
    api("/api/logs/latest"),
  ]);
  const active = status.data.active_jobs || [];
  setStatus(active.length ? `${active.length} job running` : "Ready", active.length ? "status-running" : "status-success");
  renderJobs(jobs.data.jobs || []);
  renderAudio(audio.data.audio || []);
  renderReports(reports.data || {});
  byId("logPath").textContent = logs.data.path || "";
  byId("logContent").textContent = logs.data.content || "";
}

function bindEvents() {
  byId("songSelect").addEventListener("change", () => {
    renderSongMeta();
    refreshAll();
  });
  byId("voiceSelect").addEventListener("change", renderVoiceMeta);
  byId("runScore").addEventListener("click", () => runTask("/api/run/score"));
  byId("runSvs").addEventListener("click", () => runTask("/api/run/svs"));
  byId("runSvc").addEventListener("click", () => runTask("/api/run/svc"));
  byId("runFull").addEventListener("click", () => runTask("/api/run/full"));
  byId("refreshAll").addEventListener("click", refreshAll);
  byId("refreshReports").addEventListener("click", refreshAll);
  byId("openOutput").addEventListener("click", async () => {
    const result = await api("/api/open-output-directory", { method: "POST", body: "{}" });
    setStatus(result.data.opened ? "Output directory opened" : result.data.path, result.data.opened ? "status-success" : "warning");
  });
}

async function boot() {
  bindEvents();
  await loadChoices();
  await refreshAll();
  setInterval(refreshAll, 3000);
}

boot().catch((error) => {
  setStatus(String(error), "status-failed");
});
