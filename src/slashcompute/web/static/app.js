const FILES = {
  contribute: "CONTRIBUTE.sc",
  take: "TAKE.sc",
  ledger: "LEDGER.sc",
};

const state = {
  file: "contribute",
  finish: "carbon",
  settings: {
    mode: "host", url: "", gpu_percent: 50, contribute: true, finish: "carbon",
    session_token: "", grant_split: 0,
  },
  status: {},
  nodes: [],
  jobs: [],
  ledger: [],
  selectedJob: null,
  error: "",
  notice: "",
  datasetName: "",
  datasetFile: null,
  take: {
    model: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    steps: 10,
    min_stages: 2,
  },
};

const $ = (sel) => document.querySelector(sel);

async function api(path, opts = {}) {
  const next = { ...opts, headers: { ...(opts.headers || {}) } };
  if (next.body && !(next.body instanceof FormData) && !next.headers["content-type"]) {
    next.headers["content-type"] = "application/json";
  }
  const r = await fetch(path, next);
  const text = await r.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!r.ok) {
    const msg = (data && data.detail)
      ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))
      : (text || r.statusText);
    throw new Error(msg);
  }
  return data;
}

function kw(s) { return `<span class="kw">${esc(s)}</span>`; }
function str(s) { return `<span class="str">${esc(s)}</span>`; }
function cm(s) { return `<span class="cm">${esc(s)}</span>`; }
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function line(n, html) {
  return `<div class="line"><span class="ln">${n}</span><div>${html}</div></div>`;
}
function lines(rows) {
  return rows.map((html, i) => line(i + 1, html || "&nbsp;")).join("");
}
function btn(id, label, extra = "") {
  return `<button type="button" class="btn ${extra}" data-act="${id}">${label}</button>`;
}

// Text and number fields reach `state` only on `change`, so until then what is
// being typed lives in the field alone and a redraw of the pane would lose it.
function draftField() {
  const el = document.activeElement;
  if (!el || !/^(INPUT|TEXTAREA)$/.test(el.tagName)) return null;
  if (/^(range|checkbox|file)$/.test(el.type)) return null;
  return $("#code").contains(el) && el.value !== el.defaultValue ? el : null;
}

function render() {
  const active = document.activeElement;
  const keepId = active && active.id && /^(INPUT|TEXTAREA)$/.test(active.tagName) ? active.id : null;
  const keepPos = keepId && typeof active.selectionStart === "number" ? active.selectionStart : null;

  const file = state.file;
  $("#filename").textContent = FILES[file] || file;
  document.querySelectorAll(".badge").forEach((b) => {
    const on = b.dataset.file === file;
    b.classList.toggle("is-on", on);
    b.disabled = false;
    b.setAttribute("aria-selected", on ? "true" : "false");
  });
  $("#index").textContent = String(state.status.nodes ?? 0);
  const pane = $("#code");
  // The poll redraws every 2 s: leave the pane alone while a field in it holds
  // uncommitted text; it catches up on the first redraw after the edit ends.
  if (pane.dataset.file === file && draftField()) return;
  pane.dataset.file = file;
  if (file === "contribute") pane.innerHTML = lines(contributeLines());
  else if (file === "take") pane.innerHTML = lines(takeLines());
  else pane.innerHTML = lines(ledgerLines());
  bindPane();

  if (keepId) {
    const el = document.getElementById(keepId);
    if (el) {
      el.focus();
      if (keepPos != null && el.setSelectionRange) {
        try { el.setSelectionRange(keepPos, keepPos); } catch { /* number */ }
      }
    }
  }
}

function poolLines() {
  const s = state.settings;
  const st = state.status;
  const host = s.mode === "host";
  const rows = [
    `${kw("mode")}    = ${str(s.mode.toUpperCase())}`,
    `<span class="row">${btn("mode-host", "HOST")}${btn("mode-join", "JOIN", "ghost")}</span>`,
  ];
  if (host) {
    rows.push(`${kw("address")} = ${str(st.lan_ip || "—")}`);
    rows.push(`<span class="row">${btn("copy-ip", "COPY")}</span>`);
    rows.push(`<label class="chk"><input type="checkbox" id="contribute" ${s.contribute ? "checked" : ""}> also contribute this Mac</label>`);
  } else {
    rows.push(`${kw("coordinator")} =`);
    rows.push(`<input type="text" id="url" value="${esc(s.url)}" placeholder="192.168.1.10:8765" spellcheck="false">`);
    rows.push(`<span class="row">${btn("discover", "FIND_ON_LAN")}</span>`);
  }
  rows.push(`${kw("coordinator")} ${st.coordinator_up ? str("UP") : kw("OFFLINE")}  ${cm(`nodes=${st.nodes ?? 0}`)}`);
  rows.push(`${kw("agent")}       ${st.agent_running ? str(st.agent_status || "RUNNING") : kw("STOPPED")}`);
  return rows;
}

function contributeLines() {
  const s = state.settings;
  const st = state.status;
  const rows = [
    `${kw("kind")} = ${str("contribute")}`,
    ...poolLines(),
    `${kw("gpu")}   = ${str(s.gpu_percent + "%")}`,
    `<input type="range" id="gpu" min="1" max="100" value="${s.gpu_percent}">`,
  ];
  if (state.error) rows.push(`<span class="err">${esc(state.error)}</span>`);
  else if (st.last_error && !st.coordinator_up) rows.push(`<span class="err">${esc(st.last_error)}</span>`);
  rows.push(`<span class="row">${btn("start", "START")}${btn("stop", "STOP", "ghost")}</span>`);
  rows.push(`${cm("# allow TCP 8765 on the host, 9700 on each agent")}`);
  return rows;
}

function takeLines() {
  const t = state.take;
  const rows = [
    `${kw("kind")}       = ${str("lora_finetune")}`,
    `${kw("model")}      =`,
    `<input type="text" id="model" value="${esc(t.model)}" spellcheck="false">`,
    `${kw("steps")}      = <input type="number" id="steps" min="1" value="${t.steps}" style="max-width:88px">`,
    `${kw("min_stages")} = <input type="number" id="min_stages" min="1" value="${t.min_stages}" style="max-width:88px">`,
    `${kw("dataset")}    = ${str(state.datasetName || "none")}`,
    `<span class="row"><label class="btn">${state.datasetName ? "REPLACE.JSONL" : "DATASET.JSONL"}<input type="file" id="dataset" accept=".jsonl,application/jsonl,text/plain" hidden></label></span>`,
    `<span class="row">${btn("submit", "SUBMIT")}</span>`,
  ];
  if (state.notice) rows.push(`${cm("# " + state.notice)}`);
  if (state.error) rows.push(`<span class="err">${esc(state.error)}</span>`);
  rows.push("");
  rows.push(`${cm("# queue  (FIFO — later jobs wait if the head cannot fit)")}`);
  if (!state.status.coordinator_up) rows.push(`${cm("# coordinator offline")}`);
  else if (!state.jobs.length) rows.push(`<span class="empty">// queue empty</span>`);
  else rows.push(`<div class="jobs">${state.jobs.map(jobRow).join("")}</div>`);
  return rows;
}

function jobRow(j) {
  const on = state.selectedJob === j.id ? " is-on" : "";
  const wait = j.wait_reason ? `  ${j.wait_reason}` : "";
  return `<div class="job${on}" data-job="${esc(j.id)}"><b>${esc(j.id)}</b><span>${esc(j.status)}${esc(wait)}</span><em>${esc(j.progress_step ?? 0)}/${esc(j.steps)}</em></div>`;
}

function ledgerLines() {
  const rows = [
    `${kw("kind")} = ${str("meter")}`,
    `${cm("# raw FLOPs per node")}`,
  ];
  if (!state.ledger.length) rows.push(`<span class="empty">// no steps yet</span>`);
  for (const r of state.ledger) {
    rows.push(`${str((r.node_id || "").slice(0, 12))}  ${kw(r.kind)}  ${str(fmt(r.flops))}`);
  }
  return rows;
}

function fmt(n) {
  const x = Number(n);
  if (!Number.isFinite(x)) return "0";
  if (Math.abs(x) >= 1e6 || (x !== 0 && Math.abs(x) < 1e-2)) return x.toExponential(2);
  return String(Math.round(x * 100) / 100);
}

function bindPane() {
  const map = {
    url: (el) => { state.settings.url = el.value.trim(); saveSettings(); },
    gpu: (el) => { state.settings.gpu_percent = Number(el.value); saveSettings(); render(); },
    contribute: (el) => { state.settings.contribute = el.checked; saveSettings(); },
    model: (el) => { state.take.model = el.value.trim(); },
    steps: (el) => { state.take.steps = Number(el.value) || 10; },
    min_stages: (el) => { state.take.min_stages = Number(el.value) || 2; },
  };
  for (const [id, fn] of Object.entries(map)) {
    const el = document.getElementById(id);
    if (!el) continue;
    const live = el.type === "range" || el.type === "checkbox";
    el.addEventListener(live ? "input" : "change", () => {
      fn(el);
      // committed: no longer a draft, so redraws may resume while it keeps focus
      if (!live) el.defaultValue = el.value;
    });
  }
  const file = $("#dataset");
  if (file) file.addEventListener("change", () => {
    state.datasetFile = file.files[0] || null;
    state.datasetName = state.datasetFile ? state.datasetFile.name : "";
    render();
  });
  document.querySelectorAll("[data-act]").forEach((b) => {
    b.addEventListener("click", () => act(b.dataset.act));
  });
  document.querySelectorAll("[data-job]").forEach((el) => {
    el.addEventListener("click", () => { state.selectedJob = el.dataset.job; render(); });
  });
}

async function saveSettings() {
  try {
    await api("/api/settings", { method: "POST", body: JSON.stringify(state.settings) });
  } catch (e) {
    state.error = e.message;
  }
}

async function act(name) {
  state.error = "";
  state.notice = "";
  try {
    if (name === "mode-host") { state.settings.mode = "host"; await saveSettings(); }
    else if (name === "mode-join") { state.settings.mode = "join"; await saveSettings(); }
    else if (name === "copy-ip") await navigator.clipboard.writeText(state.status.lan_ip || "");
    else if (name === "discover") {
      const r = await api("/api/discover", { method: "POST" });
      state.settings.url = r.url;
    } else if (name === "start") {
      Object.assign(state.status, await api("/api/start", {
        method: "POST", body: JSON.stringify(state.settings),
      }));
    } else if (name === "stop") {
      Object.assign(state.status, await api("/api/stop", { method: "POST" }));
    } else if (name === "submit") await submitJob();
  } catch (e) {
    state.error = e.message;
  }
  await refresh();
}

async function submitJob() {
  if (!state.datasetFile) throw new Error("Choose a JSONL dataset on this Mac.");
  const body = new FormData();
  body.append("dataset", state.datasetFile);
  body.append("model", state.take.model);
  body.append("steps", String(state.take.steps));
  body.append("min_stages", String(state.take.min_stages));
  const job = await api("/api/coord/jobs/upload", { method: "POST", body });
  state.selectedJob = job.id;
  state.notice = "Queued. FIFO — it waits if the pool cannot fit it yet.";
}

async function refresh() {
  try {
    const snap = await api("/api/status");
    state.status = snap;
    const focus = document.activeElement && document.activeElement.id;
    state.settings.mode = snap.mode;
    if (focus !== "url") state.settings.url = snap.url;
    if (focus !== "gpu") state.settings.gpu_percent = snap.gpu_percent;
    state.settings.contribute = snap.contribute;
    state.settings.session_token = snap.session_token || "";
    state.settings.grant_split = snap.grant_split ?? 0;
    if (snap.finish && snap.finish !== state.finish) applyFinish(snap.finish, false);
    if (snap.coordinator_up) {
      try {
        state.nodes = await api("/api/coord/nodes");
        state.jobs = await api("/api/coord/jobs");
        state.ledger = await api("/api/coord/ledger");
      } catch (e) {
        state.error = state.error || e.message;
      }
    }
  } catch (e) {
    state.error = e.message;
  }
  render();
}

function applyFinish(name, persist = true) {
  state.finish = name;
  state.settings.finish = name;
  document.documentElement.dataset.ink = name;
  document.querySelectorAll(".seg").forEach((s) => s.classList.toggle("is-on", s.dataset.ink === name));
  const wipe = $("#wipe");
  wipe.hidden = false;
  wipe.classList.remove("is-on");
  void wipe.offsetWidth;
  wipe.classList.add("is-on");
  window.setTimeout(() => { wipe.classList.remove("is-on"); wipe.hidden = true; }, 760);
  if (persist) saveSettings();
}

document.querySelectorAll(".badge").forEach((b) => {
  b.addEventListener("click", () => {
    state.file = b.dataset.file;
    state.error = "";
    state.notice = "";
    render();
  });
});
document.querySelectorAll(".seg").forEach((s) => {
  s.addEventListener("click", () => applyFinish(s.dataset.ink));
});

refresh();
window.setInterval(refresh, 2000);
