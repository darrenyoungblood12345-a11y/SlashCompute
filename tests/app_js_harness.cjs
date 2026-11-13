// Runs web/static/app.js against a stub DOM so pytest can drive the 2 s poll.
// Usage: node app_js_harness.cjs <path to app.js> <scenario>
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const [appJs, scenario] = process.argv.slice(2);

const unesc = (s) => s.replace(/&(amp|lt|gt|quot|#39);/g, (_, c) => (
  { amp: "&", lt: "<", gt: ">", quot: '"', "#39": "'" }[c]));

class El {
  constructor(tag, attrs = {}) {
    this.tagName = tag.toUpperCase();
    this.id = attrs.id || "";
    this.type = attrs.type || (tag === "input" ? "text" : "");
    this.dataset = {};
    for (const [k, v] of Object.entries(attrs)) {
      if (k.startsWith("data-")) this.dataset[k.slice(5)] = v;
    }
    this.defaultValue = attrs.value ?? "";
    this.value = this.defaultValue;
    this.checked = "checked" in attrs;
    this.hidden = "hidden" in attrs;
    this.disabled = false;
    this.files = [];
    this.textContent = "";
    this.offsetWidth = 0;
    this.children = [];
    this.listeners = {};
    this.classList = { toggle() {}, add() {}, remove() {} };
    // like a browser: only text-like fields have a caret
    this.selectionStart = this.selectionEnd = this.type === "text" ? this.value.length : null;
    this.html = "";
  }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  dispatch(type) { for (const fn of this.listeners[type] || []) fn({ target: this }); }
  setAttribute() {}
  focus() { document.activeElement = this; }
  setSelectionRange(a, b) { this.selectionStart = a; this.selectionEnd = b; }
  contains(el) { return this.children.includes(el); }
  get innerHTML() { return this.html; }
  set innerHTML(html) {
    // replacing the children drops the focused one, as in a browser
    if (this.children.includes(document.activeElement)) document.activeElement = body;
    this.html = html;
    this.children = [];
    for (const [, tag, raw] of html.matchAll(/<(input|button|div)\b([^>]*)>/g)) {
      const attrs = {};
      for (const [, k, v] of raw.matchAll(/([\w-]+)(?:="([^"]*)")?/g)) attrs[k] = unesc(v ?? "");
      this.children.push(new El(tag, attrs));
    }
  }
}

const body = new El("body");
const pane = new El("div");
const fixed = { "#filename": new El("p"), "#index": new El("div"), "#code": pane, "#wipe": new El("div") };
const badges = ["contribute", "take", "ledger"].map((f) => new El("button", { "data-file": f }));
const document = {
  activeElement: body,
  documentElement: { dataset: {} },
  getElementById: (id) => pane.children.find((e) => e.id === id) || null,
  querySelector: (sel) => fixed[sel] || document.getElementById(sel.slice(1)),
  querySelectorAll(sel) {
    if (sel === ".badge") return badges;
    if (sel === "[data-act]") return pane.children.filter((e) => "act" in e.dataset);
    if (sel === "[data-job]") return pane.children.filter((e) => "job" in e.dataset);
    return [];
  },
};

// the shell API, as far as app.js uses it
const server = {
  status: {
    mode: "join", url: "http://10.0.0.1:8765", gpu_percent: 50, contribute: true,
    finish: "carbon", session_token: "", grant_split: 0,
    coordinator_up: false, nodes: 0, agent_running: false, lan_ip: "192.168.1.20",
  },
  posts: [],
};
async function fetch(path, opts = {}) {
  let data = [];
  if (path === "/api/status") data = server.status;
  else if (opts.method === "POST") {
    const sent = typeof opts.body === "string" ? JSON.parse(opts.body) : null;
    server.posts.push({ path, body: sent });
    if (path === "/api/settings" || path === "/api/start") Object.assign(server.status, sent);
    data = server.status;
  }
  return { ok: true, statusText: "OK", text: async () => JSON.stringify(data) };
}

let poll = null;
const window = { setInterval: (fn) => { poll = fn; }, setTimeout() {} };
const settle = () => new Promise((r) => setImmediate(r));
const field = (id) => {
  const el = document.getElementById(id);
  assert.ok(el, `no #${id} in the pane`);
  return el;
};
// what a user does: click into a field and type, with no `change` until they leave it
function type(id, text, caret = text.length) {
  const el = field(id);
  el.focus();
  el.value = text;
  if (el.selectionStart !== null) el.setSelectionRange(caret, caret);
  return el;
}
const leave = (el) => { el.dispatch("change"); document.activeElement = body; };
async function click(el) { el.dispatch("click"); await settle(); }

const scenarios = {
  async poll_keeps_text_being_typed() {
    type("url", "192.168.1.50", 5);
    await poll();
    const el = field("url");
    assert.equal(el.value, "192.168.1.50");
    assert.equal(document.activeElement, el);
    assert.equal(el.selectionStart, 5);
    assert.deepEqual(server.posts, []);  // still only a draft
  },

  async poll_keeps_text_being_typed_in_every_field() {
    await click(badges[1]);  // TAKE.sc
    for (const [id, text] of [["model", "mlx-community/other"], ["steps", "25"], ["min_stages", "3"]]) {
      type(id, text);
      await poll();
      assert.equal(field(id).value, text, id);
      assert.equal(document.activeElement, field(id), id);
      leave(field(id));
    }
    await poll();
    // each draft was committed on leaving its field and survives later redraws
    assert.equal(field("model").value, "mlx-community/other");
    assert.equal(field("steps").value, "25");
    assert.equal(field("min_stages").value, "3");
  },

  async typed_text_survives_a_poll_and_is_used_by_start() {
    type("url", "192.168.1.50");
    await poll();
    leave(field("url"));  // clicking START blurs the field first
    await click(document.querySelectorAll("[data-act]").find((b) => b.dataset.act === "start"));
    const start = server.posts.find((p) => p.path === "/api/start");
    assert.equal(start.body.url, "192.168.1.50");
    assert.equal(field("url").value, "192.168.1.50");
  },

  async pane_still_follows_the_poll_when_nothing_is_being_typed() {
    // not focused: the saved value wins
    server.status.url = "http://10.0.0.7:8765";
    await poll();
    assert.equal(field("url").value, "http://10.0.0.7:8765");
    // focused but untouched: status lines keep updating, focus stays
    field("url").focus();
    server.status.nodes = 3;
    await poll();
    assert.match(pane.innerHTML, /nodes=3/);
    assert.equal(document.activeElement, field("url"));
    // committed with Enter (focus stays): no longer a draft, so redraws resume
    const el = type("url", "192.168.1.50");
    el.dispatch("change");
    await settle();
    assert.equal(server.posts.at(-1).body.url, "192.168.1.50");
    server.status.nodes = 5;
    await poll();
    assert.match(pane.innerHTML, /nodes=5/);
    assert.equal(field("url").value, "192.168.1.50");
    assert.equal(document.activeElement, field("url"));
  },
};

(async () => {
  assert.ok(scenarios[scenario], `unknown scenario ${scenario}`);
  const context = vm.createContext({ document, window, fetch, navigator: {}, FormData: class {} });
  vm.runInContext(fs.readFileSync(appJs, "utf8"), context, { filename: appJs });
  await settle();  // the first refresh() that app.js starts on load
  assert.ok(poll, "app.js did not schedule its poll");
  await scenarios[scenario]();
  console.log("ok");
})().catch((e) => {
  console.error(e && e.stack ? e.stack : e);
  process.exit(1);
});
