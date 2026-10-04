// /compute dashboard. Polls /api/overview and updates the widgets in place;
// forms are static markup, so polling never clobbers what you are typing.

const T = 1e12;
const SETTING_KEYS = [
  "mode", "url", "gpu_percent", "contribute", "finish", "session_token", "grant_split",
  "training", "memory_gb", "inference", "inference_memory_gb", "inference_head", "models_dir", "transport",
];
const GIB = 1024 ** 3;
const GRANTS_REFRESH_MS = 20000;
const OUTDATED_COORDINATOR = "This pool's coordinator has no LLM inference: it runs an older /compute. "
  + "Ask whoever hosts it to update and restart it, or host a pool on this Mac.";
const STATUS_TONE = {
  running: "ok", completed: "line", starting: "line", queued: "warn", recovering: "warn",
  failed: "hot", cancelled: "",
};

const state = {
  tab: "contributions",
  ov: null,
  settings: null,
  saving: 0,
  rates: [],
  last: null,
  grants: null,
  grantsUp: null,     // coordinator_up when the board was last loaded
  grantsAt: 0,
  grantsSeq: 0,
  sort: "top",
  admin: false,
  fundOpen: null,
  fundMsg: "",
  busy: new Set(),
  dataset: null,
  modelsShown: "",
  keys: {},
  user: null,
  credits: null,
  authMode: "login",
  terms: "",
  llm: {
    net: null,          // /inference/status
    models: [],         // /v1/models: servable right now
    model: "",
    messages: [],       // {role, content, meta?, error?}
    streaming: false,
    abort: null,
    upload: null,       // {name, pct} while a GGUF is on its way
  },
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

async function api(path, opts = {}) {
  const next = { ...opts, credentials: "include", headers: { ...(opts.headers || {}) } };
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

const post = (path, body) => api(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
const patch = (path, body) => api(path, { method: "PATCH", body: JSON.stringify(body) });
const signedIn = () => !!(state.user && state.user.id);

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// ------------------------------------------------------------ formatting

function fmtFlops(x) {
  let v = Number(x) || 0;
  let unit = "";
  for (const u of ["K", "M", "G", "T", "P", "E"]) {
    if (Math.abs(v) < 1000) break;
    v /= 1000;
    unit = u;
  }
  const text = v === 0 ? "0" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2);
  return unit ? `${text} ${unit}` : text;
}

function withUnit(x, unit = "FLOPs") {
  const s = fmtFlops(x);
  return /[A-Z]$/.test(s) ? s + unit : `${s} ${unit}`;
}

function fmtBytes(n) {
  let v = Number(n) || 0;
  let unit = "B";
  for (const u of ["KB", "MB", "GB", "TB"]) {
    if (Math.abs(v) < 1024) break;
    v /= 1024;
    unit = u;
  }
  return v === 0 ? "0" : `${v >= 100 ? v.toFixed(0) : v.toFixed(1)} ${unit}`;
}

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const shortModel = (m) => String(m || "").split("/").pop();
const modelLabel = (m) => shortModel(m).replace(/-Instruct-4bit$/, "").replace(/-/g, " ");
const setText = (sel, text) => { const el = $(sel); if (el && el.textContent !== text) el.textContent = text; };

function setTag(sel, text, tone) {
  const el = $(sel);
  el.textContent = text;
  el.className = `tag ${tone || ""}`.trim();
}

function setDot(sel, tone, live = false) {
  const el = $(sel);
  el.className = `sq ${tone || ""} ${live ? "live" : ""}`.trim();
}

// Re-render a list only when its data changed, so hover and focus survive polls.
function renderOnce(key, data, el, html) {
  const sig = JSON.stringify(data);
  if (state.keys[key] === sig) return;
  state.keys[key] = sig;
  el.innerHTML = html();
}

function rangeFill(el) {
  const pct = ((el.value - el.min) / (el.max - el.min)) * 100;
  el.style.setProperty("--pct", `${pct}%`);
}

// ------------------------------------------------------------ memory sliders

// Both sliders stop at the GPU working set (75% of RAM); 0 is "Auto".
function memoryLimits() {
  const st = status();
  const total = (Number(st.memory_total_bytes) || 16 * GIB) / GIB;
  return { total, cap: Math.max(1, Math.round(total * 0.75)), free: (Number(st.memory_available_bytes) || 0) / GIB };
}

const trainingAutoGb = () => Math.min(memoryLimits().free, memoryLimits().cap);   // what is free at start
const inferenceAutoGb = () => Math.max(2, Math.round(memoryLimits().total * 0.75 - 4));
const memLabel = (gb, autoGb) => (gb ? `${gb} GB` : `Auto · ≈${Math.round(autoGb)} GB`);

function syncMemSlider(el, out, saved, autoGb) {
  el.max = String(Math.max(memoryLimits().cap, Number(saved) || 0));
  if (document.activeElement !== el && saved != null) el.value = saved;
  rangeFill(el);
  out.textContent = memLabel(Number(el.value), autoGb);
}

function toast(text, tone = "ok") {
  const el = $("#toast");
  el.querySelector("span").textContent = text;
  el.className = `toast ${tone === "bad" ? "bad" : ""}`.trim();
  el.hidden = false;
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => { el.hidden = true; }, tone === "bad" ? 6000 : 3500);
}

// ------------------------------------------------------------ derived values

const status = () => (state.ov && state.ov.status) || {};
const pool = () => (state.ov && state.ov.pool) || { online: false, nodes: [], jobs: [], capacity: {} };
const me = () => (state.ov && state.ov.me) || { flops: 0 };
// The share the running agent was started with; a slider move only saves the next one.
const liveGpuPercent = () => (me().node || {}).gpu_percent ?? (state.settings || {}).gpu_percent ?? 0;

function credits() {
  const split = Number((state.user && state.user.grant_split)
    ?? (state.settings || {}).grant_split) || 0;
  if (state.credits) {
    const earned = Math.max(0, Number(state.credits.lifetime_earned) || 0);
    const toGrants = earned * split / 100;
    return {
      earned, toGrants, kept: earned - toGrants, split,
      balance: Math.max(0, Number(state.credits.balance) || 0),
    };
  }
  const earned = Math.max(0, Number(me().flops) || 0);
  const toGrants = earned * split / 100;
  return { earned, toGrants, kept: earned - toGrants, split, balance: 0 };
}

function grantBalance() {
  if (state.credits) return Math.max(0, Number(state.credits.balance) || 0);
  if (state.grants && state.grants.available != null) return Math.max(0, Number(state.grants.available) || 0);
  return 0;
}

function fmtWait(s) {
  if (s == null || !Number.isFinite(Number(s))) return "unknown wait";
  const n = Number(s);
  if (n < 60) return `${Math.max(1, Math.round(n))}s`;
  if (n < 3600) return `${Math.max(1, Math.round(n / 60))} min`;
  return `${(n / 3600).toFixed(1)} h`;
}

// ------------------------------------------------------------ settings

function pickSettings(src) {
  return Object.fromEntries(SETTING_KEYS.map((k) => [k, src[k]]));
}

async function saveSettings(changes) {
  state.settings = { ...state.settings, ...changes };
  state.saving += 1;
  try {
    state.settings = pickSettings(await post("/api/settings", state.settings));
  } catch (e) {
    toast(e.message, "bad");
  } finally {
    state.saving -= 1;
  }
}

// ------------------------------------------------------------ polling

async function poll() {
  try {
    const ov = await api("/api/overview");
    trackRate(ov);
    state.ov = ov;
    if (!state.settings || state.saving === 0) state.settings = pickSettings(ov.status);
    if (ov.status && ov.status.coordinator_up) {
      await loadAuth();
      if (state.tab === "llm") await loadLlm();
    }
    else { state.user = null; state.credits = null; }
  } catch (e) {
    state.ov = null;
    state.user = null;
    state.credits = null;
  }
  // The board comes from the coordinator: reload it when the pool comes or goes,
  // and now and then while the tab is open, so it never reads Live while offline.
  const up = !!status().coordinator_up;
  const stale = state.tab === "grants" && Date.now() - state.grantsAt > GRANTS_REFRESH_MS;
  if (up !== state.grantsUp || stale) loadGrants(true);
  render();
}

async function loadAuth() {
  try {
    const me = await api("/api/coord/auth/me");
    state.user = (me && me.user) || null;
    state.credits = (me && me.credits) || null;
    if (state.user && !state.user.accepted_terms && !state.terms) {
      const t = await api("/api/coord/auth/terms");
      state.terms = (t && t.text) || "";
    }
  } catch {
    state.user = null;
    state.credits = null;
  }
}

function trackRate(ov) {
  if (!ov.pool.online) return;
  const now = Date.now() / 1000;
  const flops = Number(ov.me.flops) || 0;
  if (state.last && now > state.last[0]) {
    state.rates.push(Math.max(0, (flops - state.last[1]) / (now - state.last[0])));
    if (state.rates.length > 48) state.rates.shift();
  }
  state.last = [now, flops];
}

function render() {
  renderSidebar();
  renderAuth();
  renderContributions();
  renderUsage();
  renderGrantsLive();
  renderPool();
  renderLlm();
}

// ------------------------------------------------------------ sidebar + tabs

function showTab(name) {
  state.tab = name;
  $$(".nav-item").forEach((b) => {
    const on = b.dataset.tab === name;
    b.classList.toggle("is-on", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
  });
  $$(".view").forEach((v) => v.classList.toggle("is-on", v.id === `view-${name}`));
  $("#main").scrollTop = 0;
  if (name === "grants") loadGrants();
  if (name === "llm" && status().coordinator_up) loadLlm().then(renderLlm);
}

// The pool on screen is ours only in host mode, not one left running from before joining another.
const hostingHere = (st) => (state.settings || {}).mode === "host" && st.coordinator_pid != null;

function renderSidebar() {
  const st = status();
  if (!state.ov) {
    setDot("#side-dot", "hot");
    setText("#side-title", "No connection");
    setText("#side-sub", "The local /compute service is not answering.");
  } else if (st.coordinator_up) {
    setDot("#side-dot", "ok");
    setText("#side-title", hostingHere(st) ? "Hosting" : "Joined");
    setText("#side-sub", `${plural(st.nodes || 0, "Mac")} · ${plural(st.jobs || 0, "job")}`);
  } else {
    setDot("#side-dot", "");
    setText("#side-title", "Offline");
    setText("#side-sub", "No pool running");
  }
  const agent = $("#side-agent");
  agent.hidden = !st.agent_running;
  agent.textContent = `Contributing · ${liveGpuPercent()}%`;
}

function renderAuth() {
  const gate = $("#auth-gate");
  const form = $("#auth-form");
  const terms = $("#auth-terms");
  const userBox = $("#auth-user");
  const online = !!(status().coordinator_up);
  $("#auth-submit").disabled = !online;
  if (signedIn() && !state.user.accepted_terms) {
    gate.hidden = true;
    form.hidden = true;
    terms.hidden = false;
    userBox.hidden = true;
    if (state.terms) $("#terms-text").textContent = state.terms;
    return;
  }
  if (signedIn()) {
    gate.hidden = true;
    form.hidden = true;
    terms.hidden = true;
    userBox.hidden = false;
    setText("#auth-who", state.user.name || state.user.email);
    setText("#auth-sub", state.user.admin ? "Admin on this pool" : state.user.email);
    return;
  }
  userBox.hidden = true;
  terms.hidden = true;
  if (!form.hidden) {
    gate.hidden = true;
    $("#auth-submit").textContent = state.authMode === "register" ? "Create account" : "Sign in";
    $("#auth-name").hidden = state.authMode !== "register";
  } else {
    gate.hidden = false;
  }
}

// ------------------------------------------------------------ contributions

function sparkline(svg, values) {
  const w = 200;
  const h = 40;
  const vals = values.slice(-48);
  const top = Math.max(0, ...vals);
  if (vals.length < 2 || top <= 0) {
    svg.innerHTML = `<line x1="0" y1="${h - 1}" x2="${w}" y2="${h - 1}" stroke="#3a3a3a" stroke-width="2" stroke-dasharray="4 5"/>`;
    return;
  }
  const step = w / (vals.length - 1);
  const pts = vals.map((v, i) => `${(i * step).toFixed(1)},${(h - 2 - (v / top) * (h - 6)).toFixed(1)}`);
  svg.innerHTML = `<polygon points="0,${h} ${pts.join(" ")} ${w},${h}" fill="rgba(214,255,0,0.12)"/>`
    + `<polyline points="${pts.join(" ")}" fill="none" stroke="#d6ff00" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
}

function renderContributions() {
  const st = status();
  const m = me();
  const c = credits();
  const rate = state.rates.length ? state.rates[state.rates.length - 1] : 0;

  setText("#c-flops", fmtFlops(c.earned));
  setText("#c-flops-sub", c.earned
    ? `FLOPs total · ${rate > 0 ? withUnit(rate, "FLOP/s") + " now" : "idle right now"}`
    : "Nothing yet. Start contributing to earn.");
  sparkline($("#c-spark"), state.rates);
  setText("#c-earned", fmtFlops(c.earned));
  setText("#c-earned-sub", signedIn()
    ? `On this account · you keep ${withUnit(c.kept)}`
    : `Estimated · you keep ${withUnit(c.kept)}`);
  setText("#c-grants", fmtFlops(c.toGrants));
  setText("#c-grants-sub", signedIn()
    ? `${c.split}% of lifetime earnings`
    : `Estimated · ${c.split}% of what you earn`);
  setText("#c-rank", m.rank ? `#${m.rank}` : "—");
  setText("#c-rank-sub", m.rank ? `of ${plural(m.of, "Mac")} in the pool` : "Not ranked yet");

  const running = !!st.agent_running;
  const s = state.settings || {};
  setTag("#c-pill", running ? `Contributing · ${liveGpuPercent()}%` : "Not contributing", running ? "ok" : "");
  setDot("#c-dot", running ? "ok" : "", running);
  if (running) {
    const job = st.agent_job_id ? ` · job ${String(st.agent_job_id).slice(0, 8)}` : "";
    setText("#c-state", `Contributing${job}`);
    setText("#c-detail", `Agent is ${st.agent_status || "running"}. Earning whenever the pool has work.`);
  } else {
    const where = s.mode === "host" ? "hosted on this Mac"
      : s.mode === "public" ? (s.url || status().public_url || "public pool")
      : (s.url || "no address set yet");
    setText("#c-state", "Not contributing");
    setText("#c-detail", `Start to lend this Mac to the pool (${where}). It runs in the background.`);
  }
  const toggle = $("#c-toggle");
  if (!state.busy.has("contribute")) {
    toggle.textContent = running ? "Stop contributing" : "Start contributing";
    toggle.className = `btn block ${running ? "ghost" : "primary"}`;
    toggle.disabled = !state.ov;
  }

  const gpu = $("#gpu");
  if (document.activeElement !== gpu && s.gpu_percent != null) gpu.value = s.gpu_percent;
  rangeFill(gpu);
  setText("#gpu-out", `${gpu.value}%`);
  const live = m.node && m.node.gpu_percent;
  setText("#gpu-hint", running && live && live !== Number(gpu.value)
    ? `Running at ${live}%. The new share applies next time you start.`
    : "Higher shares earn credits faster but leave less GPU for you.");

  syncMemSlider($("#mem"), $("#mem-out"), s.memory_gb, trainingAutoGb());
  const lent = m.node && m.node.memory_contrib_bytes;
  setText("#mem-hint", running && lent
    ? `Lending ${fmtBytes(lent)} now. A new amount applies next time you start.`
    : `Auto lends what is free when you start. Up to ${memoryLimits().cap} GB; more than is free makes macOS squeeze other apps.`);

  const split = $("#split");
  if (document.activeElement !== split) split.value = c.split;
  rangeFill(split);
  setText("#split-grants", `${split.value}%`);
  setText("#split-keep", `${100 - split.value}%`);
  setText("#split-grants-amt", `≈ ${withUnit(c.toGrants)}`);
  setText("#split-keep-amt", `≈ ${withUnit(c.kept)}`);
  setText("#split-hint", signedIn()
    ? "Credits are 1:1 with FLOPs. This split is saved on your account."
    : "Credits are 1:1 with FLOPs. Sign in so this split is saved on your account.");

  const node = m.node;
  renderOnce("mac", [node, m.node_id], $("#c-mac"), () => node ? `
    <dl class="kv">
      <dt>Name</dt><dd>${esc(node.name)}</dd>
      <dt>Chip</dt><dd>${esc(node.chip || "—")}</dd>
      <dt>Memory lent</dt><dd>${fmtBytes(node.memory_contrib_bytes)}</dd>
      <dt>Matmul speed</dt><dd>${Number(node.matmul_tflops || 0).toFixed(1)} TFLOPS</dd>
      <dt>GPU share in use</dt><dd>${esc(node.gpu_percent)}%</dd>
      <dt>Data address</dt><dd>${esc(node.data_addr || "—")}</dd>
      <dt>Node id</dt><dd class="mono-sm">${esc(String(m.node_id || "").slice(0, 12))}</dd>
    </dl>` : `<p class="empty">Start contributing to register this Mac with the pool.</p>`);
}

// ------------------------------------------------------------ usage

function renderUsage() {
  const p = pool();
  const cap = p.capacity || {};
  setText("#u-macs", String(cap.macs || 0));
  setText("#u-macs-sub", p.online ? "contributing right now" : "pool offline");
  setText("#u-tflops", Number(cap.tflops || 0).toFixed(1));
  setText("#u-mem", fmtBytes(cap.memory_bytes || 0));
  setText("#u-jobs", String(cap.running || 0));
  setText("#u-jobs-sub", `running · ${cap.waiting || 0} waiting`);
  setText("#u-pill", signedIn()
    ? `${withUnit(credits().balance)} available`
    : `≈ ${withUnit(credits().kept)} to spend`);
  $("#flop-budget").hidden = !signedIn();
  const minMacs = $("#min_stages");
  if ((state.settings || {}).mode === "public") {
    minMacs.value = "1";
    minMacs.disabled = true;
  } else {
    minMacs.disabled = false;
  }

  const models = status().models || [];
  if (models.length && state.modelsShown !== models.join()) {
    const sel = $("#model");
    const keep = sel.value;
    sel.innerHTML = models.map((m, i) => `<option value="${esc(m)}">${esc(modelLabel(m))}${i === 0 ? " · quick test" : ""}</option>`).join("");
    if (models.includes(keep)) sel.value = keep;
    state.modelsShown = models.join();
  }

  const submit = $("#job-submit");
  if (!state.busy.has("submit")) submit.disabled = !p.online;
  $("#u-open-pool").hidden = p.online;
  const msg = $("#job-msg");
  if (!p.online && !msg.dataset.sticky) setMsg("#job-msg", "Start or join a pool to submit jobs.", "");
  else if (p.online && msg.dataset.offline) setMsg("#job-msg", "", "");
  msg.dataset.offline = p.online ? "" : "1";

  setText("#u-count", String(p.jobs.length));
  renderOnce("jobs", p.jobs, $("#jobs"), () => p.jobs.length ? p.jobs.map(jobCard).join("")
    : `<p class="empty">No jobs yet. Submit one and it will appear here.</p>`);
}

function jobCard(j) {
  const status = String(j.status || "");
  const cls = ["running", "starting"].includes(status) ? "is-active"
    : ["queued", "recovering"].includes(status) ? "is-waiting" : "";
  const meta = [`Step ${j.progress_step ?? 0} of ${j.steps ?? 0}`];
  if (j.last_loss != null) meta.push(`loss ${Number(j.last_loss).toFixed(3)}`);
  if (j.stages && j.stages.length) meta.push(plural(j.stages.length, "Mac"));
  meta.push(String(j.id).slice(0, 8));
  let note = "";
  if (j.error) note = `<p class="note">${esc(j.error)}</p>`;
  else if (["queued", "recovering"].includes(status) && (j.queue_position != null || j.wait_s != null || j.wait_reason)) {
    const bits = [];
    if (j.queue_position != null) bits.push(`queue #${j.queue_position}`);
    bits.push(j.wait_s == null ? "unknown wait" : `~${fmtWait(j.wait_s)}`);
    if (j.wait_reason) bits.push(j.wait_reason);
    note = `<p class="note">Waitlist ${esc(bits.join(" · "))}</p>`;
  } else if (j.wait_reason) note = `<p class="note">Waiting: ${esc(j.wait_reason)}</p>`;
  else if (j.adapter_dir) note = `<p class="note ok">Adapter ready at ${esc(j.adapter_dir)}</p>`;
  const cancel = j.can_cancel
    ? `<button type="button" class="btn ghost sm" data-cancel="${esc(j.id)}">Cancel</button>` : "";
  return `<div class="job ${cls}">
    <div class="job-top"><b>${esc(modelLabel(j.model))}</b><span class="tag ${STATUS_TONE[status] || ""}">${esc(status)}</span>${cancel}</div>
    <div class="bar ${status === "completed" ? "done" : ""}"><i style="width:${(Number(j.progress) * 100).toFixed(1)}%"></i></div>
    <p class="meta">${esc(meta.join(" · "))}</p>${note}
  </div>`;
}

function setMsg(sel, text, tone) {
  const el = $(sel);
  el.textContent = text;
  el.className = `msg ${tone || ""}`.trim();
}

// ------------------------------------------------------------ grants

async function loadGrants(quiet = false) {
  const seq = ++state.grantsSeq;
  state.grantsUp = !!status().coordinator_up;
  state.grantsAt = Date.now();
  let board = null;
  try {
    board = await api(`/api/grants?sort=${encodeURIComponent(state.sort)}`);
  } catch (e) {
    if (!quiet) toast(e.message, "bad");
  }
  if (seq !== state.grantsSeq) return;
  state.grants = board;
  renderGrants();
  renderGrantsLive();
}

function renderGrantsLive() {
  const g = state.grants;
  const bal = grantBalance();
  setText("#g-avail", g || signedIn() ? fmtFlops(bal) : "—");
  setText("#g-avail-sub", signedIn()
    ? `${withUnit(credits().balance)} personal balance`
    : "Sign in to fund grants with your credits.");
  setText("#g-pledged", g ? fmtFlops(g.pledged) : "—");
  setText("#g-open", g ? String(g.grants.length) : "—");
  setText("#g-open-sub", g ? (g.sample ? "demo grants" : `${g.pending.length} waiting for review`) : "");
  setTag("#g-pill", g && g.sample ? "Demo" : g && g.online ? "Live" : "Offline",
    g && g.sample ? "line" : g && g.online ? "ok" : "");

  const board = (g && g.leaders && g.leaders.length)
    ? g.leaders
    : ((state.ov && state.ov.leaderboard) || []);
  const online = pool().online;
  renderOnce("leaders", [board, online], $("#leaders"), () => {
    if (!board.length) {
      return `<li class="empty">${online ? "No contributions yet." : "Start or join a pool to see who is giving the most."}</li>`;
    }
    return board.slice(0, 8).map((r) => `<li class="${r.is_me ? "is-me" : ""}">
      <span class="rank">${r.rank}</span>
      <span class="name">${esc(r.name)}${r.is_me ? ` <span class="tag ok">You</span>` : ""}</span>
      <span class="flops">${esc(withUnit(r.flops))}</span>
    </li>`).join("");
  });
}

function renderGrants() {
  const g = state.grants;
  $$("#g-sort button").forEach((b) => b.classList.toggle("is-on", b.dataset.sort === state.sort));
  const admin = !!(state.user && state.user.admin);
  const online = !!(g && g.online);
  const grants = g ? g.grants : [];
  const pending = g ? g.pending : [];

  renderOnce("grants", [grants, online, !!(g && g.sample), state.fundOpen, state.fundMsg], $("#g-list"), () => grants.length
    ? grants.map(grantCard).join("")
    : `<article class="card"><p class="empty">${online ? "No public grants yet." : "Start or join a pool to see live grants."}</p></article>`);

  $("#g-review").hidden = !admin || !!(g && g.sample);
  setTag("#g-review-count", String(pending.length), pending.length ? "hot" : "");
  renderOnce("pending", pending, $("#g-pending"), () => pending.length ? pending.map((p) => `<div class="pending">
      <div class="top"><b>${esc(p.title)}</b><span class="muted">${esc(withUnit(p.goal))} goal</span></div>
      <p class="hint">by ${esc(p.author)}</p>
      <p class="summary">${esc(p.summary)}</p>
      <div class="btn-row">
        <button type="button" class="btn primary sm" data-review="${esc(p.id)}" data-approve="1">Approve</button>
        <button type="button" class="btn danger sm" data-review="${esc(p.id)}" data-approve="">Decline</button>
      </div>
    </div>`).join("") : `<p class="empty">Nothing waiting for review.</p>`);
}

function grantCard(g) {
  const funded = g.remaining <= 0;
  const open = state.fundOpen === g.id;
  const demo = !!(state.grants && state.grants.sample);
  const actions = demo ? `<p class="hint">Demo grant — not spendable.</p>`
    : open ? `
      <input type="number" min="1" step="1" id="fund-${esc(g.id)}" value="${Math.max(1, Math.min(25, Math.floor(g.remaining / T)))}" aria-label="TFLOPs to pledge">
      <span class="unit">TFLOPs</span>
      <button type="button" class="btn primary sm" data-fund-go="${esc(g.id)}">Pledge</button>
      <button type="button" class="btn ghost sm" data-fund-close>Cancel</button>`
    : `<button type="button" class="btn ${funded ? "ghost" : "primary"} sm" data-fund-open="${esc(g.id)}" ${funded ? "disabled" : ""}>${funded ? "Funded" : "Fund this grant"}</button>`;
  return `<article class="card grant">
    <div class="top"><span class="tag line">${esc(g.tag)}</span><span class="hint">${plural(g.backers, "backer")}</span></div>
    <h3>${esc(g.title)}</h3>
    <p class="by">by ${esc(g.author)}</p>
    <p class="summary">${esc(g.summary)}</p>
    <div class="bar ${funded ? "done" : ""}"><i style="width:${(g.progress * 100).toFixed(1)}%"></i></div>
    <div class="nums"><span>${esc(fmtFlops(g.raised))} of ${esc(withUnit(g.goal))}</span><span>${Math.round(g.progress * 100)}%</span></div>
    <div class="actions">${actions}</div>
    ${open && state.fundMsg ? `<p class="msg bad">${esc(state.fundMsg)}</p>` : ""}
  </article>`;
}

async function grantAction(path, body, okText) {
  state.grantsSeq += 1;   // a board load already in flight is older than this answer
  try {
    state.grants = await post(path, { ...body, sort: state.sort });
    if (okText) toast(okText);
    return true;
  } catch (e) {
    return e.message;
  } finally {
    renderGrants();
    renderGrantsLive();
  }
}

// ------------------------------------------------------------ pool

function renderPool() {
  const st = status();
  const s = state.settings || {};
  const mode = s.mode === "join" || s.mode === "public" ? s.mode : "host";
  $$("#p-mode button").forEach((b) => b.classList.toggle("is-on", b.dataset.mode === mode));
  $("#p-host").hidden = mode !== "host";
  $("#p-join").hidden = mode !== "join";
  $("#p-public").hidden = mode !== "public";
  setText("#p-ip", st.lan_ip || "—");
  const url = $("#url");
  if (document.activeElement !== url && url.value !== (s.url || "")) url.value = s.url || "";
  const pub = $("#public-url");
  const pubVal = s.url || st.public_url || "";
  if (pub && document.activeElement !== pub && pub.value !== pubVal) pub.value = pubVal;

  const banner = $("#p-banner");
  banner.hidden = !st.last_error;
  banner.textContent = st.last_error || "";

  if (!state.busy.has("pool")) {
    // A coordinator that failed (port taken, crashed) must leave Start hosting free for a retry.
    const hosting = st.coordinator_pid != null && !(st.last_error && !st.coordinator_up);
    $("#p-start").textContent = hosting ? "Hosting" : "Start hosting";
    $("#p-start").disabled = hosting || !state.ov;
    $("#p-stop").disabled = !(st.coordinator_up || st.agent_running);
  }

  const job = st.agent_job_id ? ` · job ${String(st.agent_job_id).slice(0, 8)}` : "";
  const rows = [
    ["Coordinator", st.coordinator_up, st.coordinator_up ? (hostingHere(st) ? "hosting here" : "reachable") : "offline"],
    ["This Mac's agent", st.agent_running, st.agent_running ? `${st.agent_status || "running"}${job}` : "not contributing"],
    ["Macs in pool", (st.nodes || 0) > 0, st.coordinator_up ? String(st.nodes || 0) : "—"],
    ["Jobs", (st.jobs || 0) > 0, st.coordinator_up ? String(st.jobs || 0) : "—"],
    ["LLM node", !!st.inference_running, st.inference_running ? llmNodeState(st) : "not serving"],
    ["Address", st.coordinator_up, st.coordinator_url || "—"],
  ];
  // Left the Host tab with our pool still up: it keeps serving whoever joined it until Connect stops it.
  if (mode !== "host" && st.coordinator_pid != null) rows.push(["Pool hosted here", true, "still running · Connect stops it"]);
  renderOnce("status", rows, $("#p-status"), () => rows.map(([name, ok, value]) =>
    `<dt><i class="sq ${ok ? "ok" : ""}"></i>${esc(name)}</dt><dd>${esc(value)}</dd>`).join(""));

  const nodes = pool().nodes;
  setText("#p-count", String(nodes.length));
  renderOnce("macs", nodes, $("#p-macs"), () => nodes.length ? `<table class="macs">
      <thead><tr><th>Mac</th><th>Chip</th><th>Memory lent</th><th>GPU</th><th>FLOPs given</th><th>Status</th></tr></thead>
      <tbody>${nodes.map((n) => {
        const ok = n.canary_passed !== false && !n.draining;
        return `<tr>
          <td><span class="who"><i class="sq ${ok ? "ok" : "hot"}"></i>${esc(n.name || String(n.node_id).slice(0, 8))}${n.is_me ? ` <span class="tag ok">You</span>` : ""}</span></td>
          <td>${esc(n.chip || "—")}</td>
          <td class="num">${fmtBytes(n.memory_contrib_bytes)}</td>
          <td class="num">${esc(n.gpu_percent)}%</td>
          <td class="num">${esc(withUnit(n.flops))}</td>
          <td>${esc(n.draining ? "draining" : (n.status || "—"))}</td>
        </tr>`;
      }).join("")}</tbody></table>`
    : `<p class="empty">No Macs connected yet. Start contributing here, or have others join.</p>`);

  setText("#p-hint", mode === "host"
    ? "Firewall: allow incoming TCP 8765 on this Mac and 9700 on each contributing Mac."
    : mode === "public"
      ? "Each Take runs on one signed-in Mac. Nobody opens ports at home."
      : "Firewall: allow incoming TCP 9700 on this Mac so pipeline peers can reach it.");
}

// ------------------------------------------------------------ LLMs

function llmNodeState(st) {
  const n = st.inference_status || {};
  if (n.available === false) return n.reason || "paused";
  const busy = (n.pipelines || []).length ? "serving" : "ready";
  return `${busy} · ${String(n.transport || "").toLowerCase() || "—"}`;
}

async function loadLlm() {
  const l = state.llm;
  try {
    l.net = await api("/api/coord/inference/status");
    l.models = ((await api("/api/coord/v1/models")) || {}).data || [];
    if (!l.models.some((m) => m.id === l.model)) l.model = l.models.length ? l.models[0].id : "";
  } catch {
    l.net = null;
    l.models = [];
  }
}

function renderLlm() {
  const st = status();
  const s = state.settings || {};
  const l = state.llm;
  const net = l.net;
  const up = !!st.coordinator_up;
  const serving = net ? net.nodes.filter((n) => n.online && n.available) : [];
  const pipe = net && net.pipelines.find((p) => p.model === l.model && !["stopped", "broken"].includes(p.state));
  const live = st.inference_transport || (net && net.transport) || "";

  const unsupported = up && st.inference_supported === false;
  setTag("#l-pill", !up ? "Offline" : unsupported ? "Pool has no LLMs"
    : l.models.length ? `${plural(l.models.length, "model")} ready` : "No models yet",
  !up || unsupported ? "hot" : l.models.length ? "ok" : "warn");
  setText("#l-models", String(l.models.length));
  setText("#l-models-sub", net ? `${plural(net.models.length, "known model")}, ${l.models.length} ready` : "ready to chat");
  setText("#l-nodes", String(serving.length));
  setText("#l-nodes-sub", net ? `${plural(net.nodes.length, "Mac")} registered` : "running llama.cpp");
  setText("#l-speed", pipe && (pipe.live_tok_s || pipe.est_tok_s) ? (pipe.live_tok_s || pipe.est_tok_s).toFixed(1) : "—");
  setText("#l-speed-sub", pipe ? (pipe.live_tok_s ? "tokens/s, last reply" : "tokens/s, estimated") : "tokens per second");
  setText("#l-transport", live ? live.toUpperCase() : "—");
  setText("#l-transport-sub", live === "relay" ? "RPC through the coordinator" : live ? "Macs talk directly on the LAN" : "");
  setTag("#l-chat-tag", l.streaming ? "Replying" : pipe ? pipe.state : "Idle", l.streaming ? "ok" : "");

  const sel = $("#l-model");
  renderOnce("llm-models", [l.models.map((m) => m.id), l.model], sel, () => l.models.length
    ? l.models.map((m) => `<option value="${esc(m.id)}"${m.id === l.model ? " selected" : ""}>${esc(m.id)} · ${esc(m.size_gb)} GB</option>`).join("")
    : `<option value="">No model a head can serve yet</option>`);
  $("#l-send").disabled = l.streaming || !l.model || !up;
  $("#l-stop").disabled = !l.streaming;

  const catalog = net ? net.models : [];
  setText("#l-count", String(catalog.length));
  renderOnce("llm-catalog", [catalog, up, unsupported], $("#l-catalog"), () => catalog.length ? catalog.map((m) => {
    const dl = Object.entries(m.downloading || {}).map(([n, f]) => `${n} ${Math.round(f * 100)}%`).join(", ");
    const tone = m.status === "rejected" ? "is-waiting" : m.servable ? "is-active" : "";
    const where = m.status === "rejected" ? (m.status_reason || "rejected")
      : dl ? `downloading on ${dl}` : m.heads.length ? `on ${m.heads.join(", ")}` : "no head has it yet";
    const del = m.uploaded ? `<button type="button" class="btn ghost sm" data-act="llm-delete" data-model="${esc(m.id)}">Remove</button>` : "";
    return `<div class="job ${tone}"><div class="job-top"><b>${esc(m.id)}</b>${del}</div>
      <p class="meta">${esc(m.size_gb)} GB${m.arch ? ` · ${esc(m.arch)}` : ""} · ${esc(where)}</p></div>`;
  }).join("") : `<p class="empty">${unsupported ? esc(OUTDATED_COORDINATOR)
    : up ? "No models yet. Upload a GGUF, or put one in a head's models folder." : "Start or join a pool first."}</p>`);

  const up_ = l.upload;
  $("#l-upbar").hidden = !up_;
  if (up_) $("#l-upbar i").style.width = `${Math.round(up_.pct * 100)}%`;
  // Why uploading is unavailable right now; a click explains it instead of doing nothing.
  const pickWhy = up_ ? "An upload is already running." : unsupported ? OUTDATED_COORDINATOR
    : !up ? "Start or join a pool first." : "";
  $("#l-pick").classList.toggle("is-disabled", !!pickWhy);
  $("#l-pick").dataset.why = pickWhy;

  setTag("#l-pipe-tag", pipe ? pipe.state : "None", pipe && pipe.state === "active" ? "ok" : "");
  renderOnce("llm-pipe", [pipe || null, !!l.model], $("#l-pipe"), () => pipe ? `
    <table class="macs">
      <thead><tr><th>Mac</th><th>Role</th><th>Layers</th><th>Share</th><th>Memory</th></tr></thead>
      <tbody>${pipe.members.map((m) => `<tr>
        <td><span class="who"><i class="sq ok"></i>${esc(m.node)}</span></td><td>${esc(m.role)}</td>
        <td class="num">${m.layer_start}–${m.layer_end - 1}</td><td class="num">${Math.round(m.share * 100)}%</td>
        <td class="num">${esc(m.memory_gb)} GB</td></tr>`).join("")}</tbody></table>
    <p class="hint">${esc(pipe.explanation || "")}</p>
    <button type="button" class="btn ghost sm" data-act="llm-unload" data-pipeline="${esc(pipe.id)}">Unload</button>`
    : `<p class="empty">${l.model ? "Not loaded. Your first message plans a split across the pool and loads it." : "Pick a model to see how it's split."}</p>`);

  const running = !!st.inference_running;
  const n = st.inference_status || {};
  setDot("#l-dot", running ? (n.available === false ? "hot" : "ok") : "", running && (n.pipelines || []).length > 0);
  setText("#l-state", running ? `Serving · ${llmNodeState(st)}` : "Not serving");
  setText("#l-detail", running
    ? `${plural((n.models || []).length, "model")} on disk${Object.keys(n.downloads || {}).length ? ", downloading" : ""}. ${n.last_error || ""}`
    : "Lend memory to run LLM layers. Uploaded models are pushed here if this Mac may head.");
  syncMemSlider($("#l-mem"), $("#l-mem-out"), s.inference_memory_gb, inferenceAutoGb());
  setText("#l-mem-hint", running
    ? "A new amount applies next time you start serving."
    : `Auto lends 75% of RAM minus 4 GB. Up to ${memoryLimits().cap} GB.`);
  const dir = $("#l-dir");
  if (document.activeElement !== dir && s.models_dir) dir.value = s.models_dir;
  $$("#l-role button").forEach((b) => b.classList.toggle("is-on", (b.dataset.llmHead === "1") === (s.inference_head !== false)));
  $("#l-transport-field").hidden = s.mode === "join";
  $$("#l-transport-seg button").forEach((b) => b.classList.toggle("is-on", b.dataset.transport === (s.transport || "direct")));
  const toggle = $("#l-toggle");
  if (!state.busy.has("llm")) {
    toggle.textContent = running ? "Stop serving" : "Start serving";
    toggle.className = `btn block ${running ? "ghost" : "primary"}`;
    toggle.disabled = !state.ov;
  }
  renderChat();
}

function chatFooter(net) {
  if (!net) return "";
  const who = (net.members || []).map((m) => `${m.node} ${m.layers.length ? `L${m.layers[0]}–${m.layers[1]}` : ""}`.trim()).join(", ");
  const tok = net.tok_s ? ` · ${net.tok_s.toFixed(1)} tok/s` : "";
  return `≈ ${withUnit(net.flops)} · w=${(net.gen_weight || 1).toFixed(1)}${tok} · ${who}`;
}

// Fold one SSE event from /api/chat into the reply being streamed.
function applyChatEvent(reply, ev) {
  if (ev.error) reply.error = ev.error.message;
  if (ev.network) reply.meta = chatFooter(ev.network);
  const choice = ev.choices && ev.choices[0];
  if (!choice) return;
  const delta = choice.delta || {};
  // thinking models (Qwen3 etc.) stream their reasoning separately, before the answer
  if (delta.reasoning_content) reply.reasoning = (reply.reasoning || "") + delta.reasoning_content;
  if (delta.content) reply.content += delta.content;
  if (choice.finish_reason) reply.finish = choice.finish_reason;
}

// A thinking model's reasoning: open while it streams in, folded away once the answer starts
// (unless the user opened or closed it themselves).
function thinkBlock(m, i) {
  if (!m.reasoning) return "";
  const thinking = m.live && !m.content;
  const open = m.thinkOpen ?? thinking;
  return `<details class="think" data-think="${i}"${open ? ` open=""` : ""}>
      <summary>${thinking ? "Thinking…" : "Thoughts"}</summary><p>${esc(m.reasoning)}</p></details>`;
}

function replyText(m) {
  if (m.error) return `<p><span class="hot">${esc(m.error)}</span></p>`;
  if (m.content || m.role === "user") return `<p>${esc(m.content)}</p>`;
  if (!m.reasoning) return `<p>…</p>`;
  return !m.live && m.finish === "length" ? `<p class="empty">Ran out of tokens while thinking, before answering.</p>` : "";
}

function renderChat() {
  const log = $("#l-log");
  const msgs = state.llm.messages;
  const stick = log.scrollHeight - log.scrollTop - log.clientHeight < 32;
  const html = msgs.length ? msgs.map((m, i) => `<div class="say ${m.role === "user" ? "you" : "bot"}">
      <label>${m.role === "user" ? "You" : "Pool"}</label>
      ${thinkBlock(m, i)}${replyText(m)}
      ${m.meta ? `<small>${esc(m.meta)}</small>` : ""}</div>`).join("")
    : `<p class="empty">Replies stream in here, token by token.</p>`;
  if (log.innerHTML !== html) log.innerHTML = html;
  if (stick) log.scrollTop = log.scrollHeight;
}

async function sendChat() {
  const l = state.llm;
  const draft = $("#l-draft");
  const text = draft.value.trim();
  if (!text || l.streaming) return;
  if (!l.model) return setMsg("#l-msg", "No model to talk to yet. Upload a GGUF first.", "bad");
  setMsg("#l-msg", "", "");
  l.messages.push({ role: "user", content: text });
  const reply = { role: "assistant", content: "", live: true };
  l.messages.push(reply);
  draft.value = "";
  l.streaming = true;
  l.abort = new AbortController();
  renderLlm();
  const history = l.messages.filter((m) => m !== reply && !m.error).map((m) => ({ role: m.role, content: m.content }));
  try {
    const r = await fetch("/api/chat", {
      method: "POST", credentials: "include", signal: l.abort.signal,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: l.model, messages: history, max_tokens: 512 }),
    });
    if (!r.ok) {
      const t = await r.text();
      let msg = t;
      try { const d = JSON.parse(t); msg = (d.error && d.error.message) || d.detail || t; } catch { /* plain text */ }
      throw new Error(msg || r.statusText);
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const event = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        for (const ln of event.split("\n")) {
          const payload = ln.startsWith("data:") ? ln.slice(5).trim() : "";
          if (!payload || payload === "[DONE]") continue;
          applyChatEvent(reply, JSON.parse(payload));
        }
        renderChat();
      }
    }
  } catch (e) {
    if (e.name === "AbortError") reply.meta = "Stopped.";
    else reply.error = e.message;
  } finally {
    reply.live = false;
    l.streaming = false;
    l.abort = null;
    await loadLlm();
    renderLlm();
  }
}

function uploadModel(file) {
  const l = state.llm;
  if (!/\.gguf$/i.test(file.name)) return setMsg("#l-upmsg", "Pick a .gguf model file.", "bad");
  if (status().inference_supported === false) return setMsg("#l-upmsg", OUTDATED_COORDINATOR, "bad");
  const pooled = ((l.net && l.net.nodes) || []).filter((n) => n.online).reduce((sum, n) => sum + (Number(n.committed_gb) || 0), 0);
  const needs = file.size / GIB;
  const tooBig = l.net && needs > pooled
    ? `Heads-up: its weights alone need ≈${needs.toFixed(1)} GB but the pool's Macs lend ${pooled.toFixed(1)} GB in total. `
      + "It will upload, but won't load until more memory joins." : "";
  l.upload = { name: file.name, pct: 0 };
  setText("#l-file-name", file.name);
  setMsg("#l-upmsg", tooBig, tooBig ? "warn" : "");
  const xhr = new XMLHttpRequest();   // fetch() has no upload progress
  xhr.open("POST", "/api/models/upload");
  xhr.withCredentials = true;
  xhr.setRequestHeader("x-filename", file.name);
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) { l.upload.pct = e.loaded / e.total; renderLlm(); } };
  xhr.onload = async () => {
    let data = null;
    try { data = JSON.parse(xhr.responseText); } catch { /* plain text */ }
    l.upload = null;
    if (xhr.status >= 200 && xhr.status < 300) {
      setMsg("#l-upmsg", `${data.name}: ${data.layers} layers, sent to ${plural(data.pushed, "head")}. ${tooBig}`.trim(),
        tooBig ? "warn" : "ok");
    } else {
      setMsg("#l-upmsg", (data && (data.detail || (data.error && data.error.message))) || xhr.statusText || "Upload failed.", "bad");
    }
    await loadLlm();
    renderLlm();
  };
  xhr.onerror = () => { l.upload = null; setMsg("#l-upmsg", "Upload failed: the pool is not reachable.", "bad"); renderLlm(); };
  xhr.send(file);
  renderLlm();
}

// ------------------------------------------------------------ actions

async function withBusy(key, button, busyText, fn) {
  if (state.busy.has(key)) return;
  state.busy.add(key);
  const idle = button ? button.textContent : "";
  if (button) { button.disabled = true; button.textContent = busyText; }
  try {
    await fn();
  } catch (e) {
    toast(e.message, "bad");
  } finally {
    state.busy.delete(key);
    if (button) { button.disabled = false; button.textContent = idle; }
    await poll();
  }
}

// Joining another pool stops the coordinator hosted here once that pool answers: ask first.
function leaveHostedPool() {
  return status().coordinator_pid == null || window.confirm(
    "Connecting stops the pool hosted on this Mac. Macs that joined it lose their coordinator. Continue?");
}

const actions = {
  "toggle-llm": (btn) => {
    const st = status();
    const running = !!st.inference_running;
    return withBusy("llm", btn, running ? "Stopping…" : "Starting…", async () => {
      const keep = { contribute: true, training: !!st.agent_running };   // leave fine-tune contribution as it is
      const snap = await post("/api/start", { ...state.settings, ...keep, inference: !running });
      if (snap.last_error) toast(snap.last_error, "bad");
      else toast(running ? "Stopped serving. The current reply finishes first."
        : "Serving. This Mac hosts LLM layers whenever a chat needs them.");
    });
  },
  "llm-stop": () => { if (state.llm.abort) state.llm.abort.abort(); },
  "llm-clear": () => { state.llm.messages = []; renderChat(); },
  "llm-unload": (btn) => withBusy("llm-unload", btn, "Unloading…", async () => {
    await post(`/api/coord/inference/pipelines/${encodeURIComponent(btn.dataset.pipeline)}/stop`);
    toast("Unloading after the current reply.");
    await loadLlm();
  }),
  "llm-delete": (btn) => withBusy(`llm-del-${btn.dataset.model}`, btn, "Removing…", async () => {
    await api(`/api/coord/inference/models/${encodeURIComponent(btn.dataset.model)}`, { method: "DELETE" });
    await loadLlm();
  }),
  "toggle-contribute": (btn) => {
    const running = !!status().agent_running;
    return withBusy("contribute", btn, running ? "Stopping…" : "Starting…", async () => {
      if (running) {
        await post("/api/stop-agent");
        toast("Stopped contributing. The current step finishes first.");
        return;
      }
      const snap = await post("/api/start", { ...state.settings, contribute: true, training: true });
      if (snap.last_error) toast(snap.last_error, "bad");
      else toast("Contributing. This Mac picks up work whenever the pool has some.");
    });
  },

  host: (btn) => {
    const st = status();
    // Hosting only switches the mode: whatever this Mac already lends keeps running, now to its own pool.
    const keep = {
      contribute: !!(st.agent_running || st.inference_running),
      training: !!st.agent_running,
      inference: !!st.inference_running,
    };
    return withBusy("pool", btn, "Starting…", async () => {
      const snap = await post("/api/start", { ...state.settings, ...keep, mode: "host" });
      if (snap.last_error) toast(snap.last_error, "bad");
      else toast("Pool is up. Share this Mac's address with the others.");
    });
  },

  "stop-pool": (btn) => withBusy("pool", btn, "Stopping…", async () => {
    await post("/api/stop");
    toast("Pool stopped.");
  }),

  connect: (btn) => {
    const st = status();
    // Joining moves whatever this Mac lends to the new pool (and stops a pool hosted here).
    const keep = {
      contribute: !!(st.agent_running || st.inference_running),
      training: !!st.agent_running,
      inference: !!st.inference_running,
    };
    return withBusy("connect", btn, "Connecting…", async () => {
      const url = $("#url").value.trim();
      if (!url) throw new Error("Enter the host Mac's address, or press Find on LAN.");
      if (!leaveHostedPool()) return;
      const snap = await post("/api/start", { ...state.settings, ...keep, mode: "join", url });
      if (snap.last_error) throw new Error(snap.last_error);
      const ov = await api("/api/overview");
      if (!ov.status.coordinator_up) throw new Error(`No coordinator answering at ${ov.status.coordinator_url}.`);
      toast(`Connected to ${ov.status.coordinator_url}`);
    });
  },

  "connect-public": (btn) => withBusy("connect", btn, "Connecting…", async () => {
    if (!signedIn()) throw new Error("Sign in first.");
    const url = $("#public-url").value.trim() || status().public_url || "";
    if (!url) throw new Error("Enter the public coordinator URL.");
    if (!leaveHostedPool()) return;
    const snap = await post("/api/start", { ...state.settings, mode: "public", url });
    if (snap.last_error) throw new Error(snap.last_error);
    const ov = await api("/api/overview");
    if (!ov.status.coordinator_up) throw new Error(`No coordinator answering at ${ov.status.coordinator_url}.`);
    toast(`Connected to ${ov.status.coordinator_url}`);
  }),

  discover: (btn) => withBusy("discover", btn, "Searching…", async () => {
    const r = await post("/api/discover");
    state.settings = { ...state.settings, url: r.url };
    $("#url").value = r.url;
    toast(`Found a pool at ${r.url}`);
  }),

  "copy-ip": async (btn) => {
    const ip = status().lan_ip || "";
    try {
      await navigator.clipboard.writeText(ip);
    } catch {
      const t = Object.assign(document.createElement("textarea"), { value: ip });
      document.body.append(t);
      t.select();
      document.execCommand("copy");
      t.remove();
    }
    btn.textContent = "Copied";
    window.setTimeout(() => { btn.textContent = "Copy"; }, 1400);
  },

  "show-auth": () => {
    $("#auth-gate").hidden = true;
    $("#auth-form").hidden = false;
    $("#auth-email").focus();
    renderAuth();
  },

  "auth-mode": () => {
    state.authMode = state.authMode === "register" ? "login" : "register";
    const btn = document.querySelector("[data-act='auth-mode']");
    if (btn) btn.textContent = state.authMode === "register" ? "Have an account" : "Create account";
    renderAuth();
  },

  "accept-terms": (btn) => withBusy("terms", btn, "Saving…", async () => {
    await post("/api/coord/auth/accept-terms", {});
    await loadAuth();
    toast("Terms accepted.");
  }),

  logout: (btn) => withBusy("auth", btn, "Signing out…", async () => {
    await post("/api/coord/auth/logout", {});
    await saveSettings({ session_token: "" });
    state.user = null;
    state.credits = null;
    // Back to the sign-in form, even if this session started with a registration.
    state.authMode = "login";
    const mode = document.querySelector("[data-act='auth-mode']");
    if (mode) mode.textContent = "Create account";
    // The grant board is per user (admin review queue, pledges): rebuild it
    // so nothing from the old session stays on screen.
    $("#g-request").hidden = true;
    await loadGrants();
    toast("Signed out.");
  }),

  "toggle-request": () => {
    if (!signedIn()) return toast("Sign in to request a grant.", "bad");
    const card = $("#g-request");
    card.hidden = !card.hidden;
    setMsg("#g-msg", "", "");
    if (!card.hidden) $("#g-title").focus();
  },
};

document.addEventListener("click", async (e) => {
  const el = e.target.closest("button, [data-tab]");
  if (!el || el.disabled) return;
  const d = el.dataset;
  if (d.tab) return showTab(d.tab);
  if (d.act && actions[d.act]) return actions[d.act](el);
  if (d.mode) {
    await saveSettings({ mode: d.mode });
    return poll();
  }
  if (d.llmHead) {
    await saveSettings({ inference_head: d.llmHead === "1" });
    return renderLlm();
  }
  if (d.transport) {
    await saveSettings({ transport: d.transport });
    toast(status().inference_running || status().coordinator_pid
      ? "Saved. Start serving again to restart the pool on the new transport."
      : "Saved. Used next time this Mac hosts.");
    return renderLlm();
  }
  if (d.sort) {
    state.sort = d.sort;
    return loadGrants();
  }
  if (d.cancel) {
    return withBusy(`cancel-${d.cancel}`, el, "Cancelling…", () => post(`/api/coord/jobs/${encodeURIComponent(d.cancel)}/cancel`));
  }
  if (d.fundOpen) {
    state.fundOpen = d.fundOpen;
    state.fundMsg = "";
    renderGrants();
    const input = $(`#fund-${CSS.escape(d.fundOpen)}`);
    if (input) input.focus();
    return;
  }
  if ("fundClose" in d) {
    state.fundOpen = null;
    return renderGrants();
  }
  if ((d.fundGo || d.review) && state.grants && state.grants.sample) return;
  if (d.fundGo) {
    const tflops = Number($(`#fund-${CSS.escape(d.fundGo)}`).value);
    const grant = state.grants.grants.find((g) => g.id === d.fundGo);
    const res = await grantAction(`/api/grants/${encodeURIComponent(d.fundGo)}/fund`,
      { amount: tflops * T }, `Pledged ${withUnit(tflops * T)} to ${grant ? grant.title : "the grant"}`);
    state.fundMsg = res === true ? "" : res;
    if (res === true) state.fundOpen = null;
    return renderGrants();
  }
  if (d.review) {
    const approve = !!d.approve;
    const res = await grantAction(`/api/grants/${encodeURIComponent(d.review)}/review`, { approve },
      approve ? "Approved. It's public now." : "Declined.");
    if (res !== true) toast(res, "bad");
  }
});

$("#gpu").addEventListener("input", (e) => {
  rangeFill(e.target);
  setText("#gpu-out", `${e.target.value}%`);
});
$("#gpu").addEventListener("change", async (e) => {
  await saveSettings({ gpu_percent: Number(e.target.value) });
  render();
});

$("#mem").addEventListener("input", (e) => {
  rangeFill(e.target);
  setText("#mem-out", memLabel(Number(e.target.value), trainingAutoGb()));
});
$("#mem").addEventListener("change", async (e) => {
  await saveSettings({ memory_gb: Number(e.target.value) });
  render();
});

$("#split").addEventListener("input", (e) => {
  state.settings = { ...state.settings, grant_split: Number(e.target.value) };
  renderContributions();
  renderGrantsLive();
});
$("#split").addEventListener("change", async (e) => {
  const grant_split = Number(e.target.value);
  await saveSettings({ grant_split });
  if (signedIn()) {
    try {
      const r = await patch("/api/coord/auth/me", { grant_split });
      if (r && r.user) state.user = r.user;
    } catch (err) {
      toast(err.message, "bad");
    }
  }
});

$("#url").addEventListener("keydown", (e) => {
  if (e.key === "Enter") actions.connect($("#p-connect"));
});

$("#public-url").addEventListener("keydown", (e) => {
  if (e.key === "Enter") actions["connect-public"]($("#p-connect-public"));
});

$("#dataset").addEventListener("change", (e) => {
  state.dataset = e.target.files[0] || null;
  const name = $("#dataset-name");
  name.textContent = state.dataset ? state.dataset.name : "No file chosen";
  name.classList.toggle("sig", !!state.dataset);
});

$("#job-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const msg = $("#job-msg");
  if (!state.dataset) return setMsg("#job-msg", "Choose a JSONL dataset first.", "bad");
  const steps = Number($("#steps").value);
  const publicPool = (state.settings || {}).mode === "public";
  const minStages = publicPool ? 1 : Number($("#min_stages").value);
  if (!Number.isInteger(steps) || steps < 1) return setMsg("#job-msg", "Steps must be a whole number above zero.", "bad");
  if (!Number.isInteger(minStages) || minStages < 1) return setMsg("#job-msg", "Min Macs must be at least 1.", "bad");
  const body = new FormData();
  body.append("dataset", state.dataset);
  body.append("model", $("#model").value);
  body.append("steps", String(steps));
  body.append("min_stages", String(minStages));
  if (signedIn()) {
    const tflops = Number($("#max_flops").value);
    if (!Number.isFinite(tflops) || tflops <= 0) {
      return setMsg("#job-msg", "Set a FLOP budget above zero.", "bad");
    }
    body.append("max_flops", String(tflops * T));
  }
  setMsg("#job-msg", `Uploading ${state.dataset.name}…`, "");
  msg.dataset.sticky = "1";
  return withBusy("submit", $("#job-submit"), "Uploading…", async () => {
    try {
      const job = await api("/api/coord/jobs/upload", { method: "POST", body });
      setMsg("#job-msg", `Submitted job ${String(job.id).slice(0, 8)}. It starts when enough Macs are free.`, "ok");
      toast("Job submitted");
    } catch (err) {
      setMsg("#job-msg", err.message, "bad");
    } finally {
      delete msg.dataset.sticky;
    }
  });
});

$("#grant-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const res = await grantAction("/api/grants", {
    title: $("#g-title").value,
    summary: $("#g-summary").value,
    goal: Number($("#g-goal").value) * T,
  }, "Request sent. It goes public once an admin approves it.");
  if (res !== true) return setMsg("#g-msg", res, "bad");
  $("#g-title").value = "";
  $("#g-summary").value = "";
  $("#g-request").hidden = true;
});

$("#auth-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const email = $("#auth-email").value.trim();
  const password = $("#auth-password").value;
  const name = $("#auth-name").value.trim();
  if (!email || !password) return setMsg("#auth-msg", "Email and password are required.", "bad");
  if (state.authMode === "register" && !name) return setMsg("#auth-msg", "Give the account a name.", "bad");
  return withBusy("auth", $("#auth-submit"), "Working…", async () => {
    const path = state.authMode === "register" ? "/api/coord/auth/register" : "/api/coord/auth/login";
    const body = { email, password };
    if (state.authMode === "register") body.name = name;
    const r = await post(path, body);
    if (r.token) await saveSettings({ session_token: r.token });
    state.user = r.user || null;
    await loadAuth();
    if (state.user && !state.user.accepted_terms) {
      try {
        const t = await api("/api/coord/auth/terms");
        state.terms = (t && t.text) || "";
      } catch { state.terms = ""; }
    }
    setMsg("#auth-msg", "", "");
    toast(state.authMode === "register" ? "Account created." : "Signed in.");
    loadGrants();
  });
});

$("#l-model").addEventListener("change", (e) => { state.llm.model = e.target.value; renderLlm(); });
$("#l-mem").addEventListener("input", (e) => {
  rangeFill(e.target);
  setText("#l-mem-out", memLabel(Number(e.target.value), inferenceAutoGb()));
});
$("#l-mem").addEventListener("change", async (e) => {
  await saveSettings({ inference_memory_gb: Number(e.target.value) });
  render();
});
$("#l-dir").addEventListener("change", (e) => saveSettings({ models_dir: e.target.value.trim() || "~/models" }));
$("#l-pick").addEventListener("click", (e) => {
  const why = e.currentTarget.dataset.why;
  if (!why) return;
  e.preventDefault();   // keep the file picker closed and say why
  setMsg("#l-upmsg", why, "bad");
});
$("#l-file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (f) uploadModel(f);
});
$("#l-draft").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
});
$("#chat-form").addEventListener("submit", (e) => { e.preventDefault(); sendChat(); });
// Remember a Thinking block the user opened or closed, so the next streamed token doesn't undo it.
$("#l-log").addEventListener("click", (e) => {
  const block = e.target.closest("summary") && e.target.closest("[data-think]");
  if (block) state.llm.messages[block.dataset.think].thinkOpen = !block.open;
});

poll();
window.setInterval(poll, 2000);
