import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from slashcompute.common.config import EngineConfig
from slashcompute.coordinator.app import create_app
from slashcompute.launcher.controller import Launcher, LauncherSettings, stateless_http
from slashcompute.web.server import create_shell

SHELL = "http://127.0.0.1:8766"   # the shell refuses any Host but loopback


class FakeProc:
    def __init__(self, pid: int, argv: list[str]) -> None:
        self.pid = pid
        self.argv = argv

    def poll(self):
        return None


class FakeHTTP:
    def __init__(self, health=None) -> None:
        self.health = health

    def get(self, url: str, timeout: float = 1.0, params=None, headers=None):
        if self.health is None:
            raise ConnectionError("down")
        class R:
            status_code = 200
            content = b'{"ok":true,"nodes":1,"jobs":0}'
            headers = {"content-type": "application/json"}
            def json(self_inner):
                return {"ok": True, "nodes": 1, "jobs": 0}
        return R()

    def post(self, url: str, content=None, headers=None, timeout: float = 1.0, **_):
        class R:
            status_code = 200
            content = b'{"user":{"id":"u1","email":"ada@lan.test"},"token":"sess"}'
            headers = {"content-type": "application/json",
                       "set-cookie": "slashcompute_session=sess; HttpOnly; SameSite=lax"}
            def json(self_inner):
                return {"user": {"id": "u1", "email": "ada@lan.test"}, "token": "sess"}
        return R()


def _shell(tmp_path, **kw):
    spawned = []
    n = {"p": 5000}

    def popen(argv, **_):
        n["p"] += 1
        proc = FakeProc(n["p"], argv)
        spawned.append(proc)
        return proc

    launcher = Launcher(
        home=tmp_path,
        python="/opt/venv/bin/python",
        popen=popen,
        http=kw.pop("http", FakeHTTP()),
        discover_fn=kw.pop("discover_fn", lambda timeout=5.0: "http://10.0.0.9:8765"),
        lan_ip_fn=lambda: "192.168.1.20",
        port_free_fn=lambda host, port: True,
    )
    app = create_shell(launcher)
    return app, launcher, spawned


def test_unavailable_upload_button_explains_itself(tmp_path):
    # A disabled "Upload GGUF" used to swallow clicks (pointer-events: none); it must say why.
    app, _, _ = _shell(tmp_path)
    with TestClient(app, base_url=SHELL) as c:
        css = c.get("/static/app.css").text
        js = c.get("/static/app.js").text
    assert "#l-pick.is-disabled { opacity: 0.4; cursor: not-allowed; }" in css
    assert "pointer-events: none" not in css.split("#l-pick.is-disabled", 1)[1].split("}", 1)[0]
    assert '$("#l-pick").dataset.why = pickWhy;' in js
    assert 'setMsg("#l-upmsg", why, "bad");' in js


def test_index_and_css(tmp_path):
    app, _, _ = _shell(tmp_path)
    with TestClient(app, base_url=SHELL) as c:
        r = c.get("/")
        assert r.status_code == 200
        assert b"COMPUTE" in r.content
        for tab in (b"contributions", b"usage", b"grants", b"pool"):
            assert b'data-tab="' + tab + b'"' in r.content
        assert b'data-ink="signal"' in r.content
        assert b"/static/logo.png" in r.content
        assert c.get("/static/logo.png").status_code == 200
        assert b'data-file="community"' not in r.content
        assert b">COM<" not in r.content
        assert b'data-mode="public"' in r.content
        assert b"Public pool" in r.content
        js = c.get("/static/app.js")
        assert js.status_code == 200
        assert b"AUTH.sc" not in js.content
        assert b"/api/overview" in js.content
        assert b"GOOGLE" not in js.content
        assert b"connect-public" in js.content
        assert b'mode: "host", contribute: false' not in js.content   # hosting must not stop contributing
        css = c.get("/static/app.css")
        assert css.status_code == 200
        assert b"IBM Plex Sans" in css.content
        assert b"archivo-black" not in css.content
        assert c.get("/static/fonts/ibm-plex-sans-regular.woff2").status_code == 200
        assert c.get("/static/fonts/ibm-plex-mono-regular.woff2").status_code == 200


def test_logout_rebuilds_grant_board(tmp_path):
    # The admin review queue is only toggled by renderGrants(); signing out has
    # to reload the board or Approve/Decline stays on screen for the next user.
    app, _, _ = _shell(tmp_path)
    with TestClient(app, base_url=SHELL) as c:
        js = c.get("/static/app.js").text
    start = js.index("  logout: (btn) =>")
    body = js[start:js.index("\n  }),", start)]
    assert "state.user = null" in body
    assert "await loadGrants()" in body
    assert body.index("state.user = null") < body.index("await loadGrants()")


def test_settings_and_status(tmp_path):
    app, launcher, _ = _shell(tmp_path)
    with TestClient(app, base_url=SHELL) as c:
        r = c.post("/api/settings", json={"mode": "join", "url": "10.1.2.3",
                                          "gpu_percent": 40, "finish": "signal"})
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "join" and body["finish"] == "signal"
        assert body["url"] == "10.1.2.3"
        assert body["has_session"] is False
        assert body["grant_split"] == 0
        got = c.get("/api/settings").json()
        assert got["gpu_percent"] == 40
        st = c.get("/api/status").json()
        assert st["lan_ip"] == "192.168.1.20"
        assert "carbon" in st["finishes"]
    assert launcher.load_settings().finish == "signal"


def test_shell_refuses_foreign_host_and_cross_origin_writes(tmp_path):
    app, launcher, _ = _shell(tmp_path)
    launcher.save_settings(LauncherSettings(session_token="secret-sess"))
    stopped = []
    real_stop = launcher.stop_agent
    launcher.stop_agent = lambda: stopped.append(1) or real_stop()
    with TestClient(app, base_url=SHELL) as c:
        # DNS rebinding: evil.example resolves to 127.0.0.1 but the Host header gives it away.
        for path in ("/api/settings", "/api/status", "/"):
            assert c.get(path, headers={"host": "evil.example:8766"}).status_code == 403
        # CSRF: a "simple" cross-site POST (no preflight) must not act.
        for origin in ("https://evil.example", "null", "http://127.0.0.1.evil.example"):
            r = c.post("/api/stop-agent", headers={"origin": origin, "content-type": "text/plain"})
            assert r.status_code == 403, origin
        for path in ("/api/stop", "/api/discover", "/api/coord/auth/logout"):
            assert c.post(path, headers={"origin": "https://evil.example"}).status_code == 403
        # Another local port is another origin (a dev server, a page from some other local app), even
        # though it is same-site and so still gets the SameSite=Lax session cookie.
        for origin in ("http://localhost:3000", "http://127.0.0.1:9810", "http://[::1]", "https://127.0.0.1"):
            r = c.post("/api/stop-agent", headers={"origin": origin, "content-type": "text/plain"})
            assert r.status_code == 403, origin
        assert not stopped
        assert launcher.load_settings().session_token == "secret-sess"
        # The UI itself: loopback Host with any port and loopback name; Origin on the shell's own port.
        for host in ("127.0.0.1:8766", "localhost:9810", "[::1]:8766", "localhost"):
            assert c.get("/api/shell", headers={"host": host}).status_code == 200, host
        for origin in ("http://127.0.0.1:8766", "http://localhost:8766", "http://[::1]:8766"):
            r = c.post("/api/stop-agent", headers={"origin": origin})
            assert r.status_code == 200, origin
        assert c.post("/api/stop-agent").status_code == 200   # no Origin: not a browser
        assert c.get("/api/settings", headers={"origin": "https://evil.example"}).status_code == 200
    assert len(stopped) == 4


def test_shell_allows_configured_lan_bind_host(tmp_path, monkeypatch):
    monkeypatch.setattr("slashcompute.web.server.SHELL_HOST", "192.168.1.20")
    app, _, _ = _shell(tmp_path)
    with TestClient(app, base_url="http://192.168.1.20:8766") as c:
        assert c.get("/api/shell").status_code == 200
        assert c.post("/api/settings", json={}, headers={"origin": "http://192.168.1.20:8766"}).status_code == 200
        assert c.post("/api/settings", json={}, headers={"origin": "http://192.168.1.20:3000"}).status_code == 403
        assert c.get("/api/shell", headers={"host": "evil.example"}).status_code == 403


def test_settings_never_expose_session_token(tmp_path):
    app, launcher, _ = _shell(tmp_path)
    with TestClient(app, base_url=SHELL) as c:
        assert c.post("/api/settings", json={"session_token": "secret-sess"}).json()["has_session"] is True
        for path in ("/api/settings", "/api/status", "/api/overview"):
            assert b"secret-sess" not in c.get(path).content, path
        assert c.get("/api/settings").json()["has_session"] is True
        # The UI round-trips settings without the token; that must not sign it out.
        assert c.post("/api/settings", json={"gpu_percent": 30}).json()["has_session"] is True
        assert launcher.load_settings().session_token == "secret-sess"
        assert c.post("/api/settings", json={"session_token": ""}).json()["has_session"] is False
    assert launcher.load_settings().session_token == ""


def test_sign_in_rebinds_running_processes_to_the_pool_that_issued_it(tmp_path):
    app, launcher, _ = _shell(tmp_path)
    rebinds = []
    launcher.rebind_session = lambda: rebinds.append(launcher.session_for(launcher.load_settings()))
    pool_a, pool_b = {"mode": "join", "url": "10.0.0.1"}, {"mode": "join", "url": "10.0.0.2"}
    with TestClient(app, base_url=SHELL) as c:
        assert c.post("/api/settings", json={**pool_a, "session_token": "sess"}).json()["has_session"] is True
        assert rebinds == ["sess"]                    # the running agent and LLM node now earn for it
        c.post("/api/settings", json={**pool_a, "gpu_percent": 30})
        assert rebinds == ["sess"]
        # Another pool never sees this session; coming back to the first one finds it again.
        assert c.post("/api/settings", json=pool_b).json()["has_session"] is False
        assert c.post("/api/settings", json=pool_a).json()["has_session"] is True


def test_start_stop_and_discover(tmp_path, monkeypatch):
    http = FakeHTTP()
    app, launcher, spawned = _shell(tmp_path, http=http)
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive",
                        lambda pid: any(p.pid == pid for p in spawned))

    def health_after(url):
        if spawned:
            http.health = {"ok": True, "nodes": 0, "jobs": 0}
            return {"ok": True, "nodes": 0, "jobs": 0}
        return None

    launcher.poll_health = health_after
    with TestClient(app, base_url=SHELL) as c:
        r = c.post("/api/start", json={"mode": "host", "gpu_percent": 50, "contribute": True})
        assert r.status_code == 200, r.text
        assert len(spawned) == 2
        found = c.post("/api/discover").json()
        assert found["url"] == "http://10.0.0.9:8765"
        c.post("/api/stop")


def test_join_start_requires_url(tmp_path):
    app, _, spawned = _shell(tmp_path)
    with TestClient(app, base_url=SHELL) as c:
        r = c.post("/api/start", json={"mode": "join", "url": ""})
        assert r.status_code == 400
        assert spawned == []


def test_proxy_allows_health_and_blocks_other(tmp_path):
    app, launcher, _ = _shell(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app, base_url=SHELL) as c:
        assert c.get("/api/coord/health").status_code == 200
        assert c.get("/api/coord/auth/me").status_code == 200
        assert c.get("/api/coord/verify/secret").status_code == 404
        assert c.get("/api/coord/jobs/../verify/secret").status_code == 404
        assert c.get("/api/coord/VERIFY/secret").status_code == 404
        assert c.get("/api/shell").json()["generation"] >= 6


class RoutedHTTP:
    """Answers GETs by path. ``routes`` maps path -> JSON payload."""

    def __init__(self, routes: dict) -> None:
        self.routes = routes

    def get(self, url: str, timeout: float = 1.0, params=None, headers=None):
        path = "/" + url.split("://", 1)[-1].split("/", 1)[-1]
        if path not in self.routes:
            raise ConnectionError("down")
        payload = self.routes[path]

        class R:
            status_code = 200

            def json(self_inner):
                return payload
        return R()


POOL = {
    "/health": {"ok": True, "nodes": 2, "jobs": 2},
    "/nodes": [
        {"node_id": "me", "name": "Air", "matmul_tflops": 2.0, "memory_contrib_bytes": 8 << 30,
         "gpu_percent": 50},
        {"node_id": "b", "name": "Studio", "matmul_tflops": 20.0,
         "memory_contrib_bytes": 64 << 30, "gpu_percent": 80},
    ],
    "/jobs": [
        {"id": "old", "status": "completed", "steps": 10, "progress_step": 10, "submitted_at": 1},
        {"id": "new", "status": "running", "steps": 10, "progress_step": 4, "submitted_at": 2},
    ],
    "/ledger": [
        {"node_id": "me", "kind": "train", "flops": 4e12, "disputed_flops": 0.0},
        {"node_id": "b", "kind": "train", "flops": 9e12, "disputed_flops": 0.0},
    ],
}


def test_overview_offline(tmp_path):
    app, _, _ = _shell(tmp_path, http=RoutedHTTP({}))
    with TestClient(app, base_url=SHELL) as c:
        ov = c.get("/api/overview").json()
    assert ov["pool"]["online"] is False
    assert ov["pool"]["jobs"] == [] and ov["leaderboard"] == []
    assert ov["me"]["flops"] == 0 and ov["me"]["rank"] is None
    assert ov["status"]["lan_ip"] == "192.168.1.20"
    assert ov["status"]["models"][0].endswith("0.5B-Instruct-4bit")
    assert "public_url" in ov["status"]


def test_settings_accept_public_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("SLASHCOMPUTE_PUBLIC_URL", "https://pool.example.com")
    app, launcher, _ = _shell(tmp_path)
    with TestClient(app, base_url=SHELL) as c:
        r = c.post("/api/settings", json={"mode": "public", "url": "", "gpu_percent": 40})
        assert r.status_code == 200
        assert r.json()["mode"] == "public"
        ov = c.get("/api/overview").json()
        assert ov["status"]["mode"] == "public"
        assert ov["status"]["public_url"] == "https://pool.example.com"
        assert ov["status"]["coordinator_url"] == "https://pool.example.com"


def test_overview_online_ranks_this_mac(tmp_path):
    app, launcher, _ = _shell(tmp_path, http=RoutedHTTP(POOL))
    launcher.save_settings(LauncherSettings(mode="host", grant_split=25))
    launcher.paths.node_id_file.write_text("me\n")
    with TestClient(app, base_url=SHELL) as c:
        ov = c.get("/api/overview").json()
    assert ov["pool"]["online"] is True
    assert [j["id"] for j in ov["pool"]["jobs"]] == ["new", "old"]
    assert ov["pool"]["jobs"][0]["progress"] == 0.4 and ov["pool"]["jobs"][0]["can_cancel"]
    assert ov["pool"]["capacity"]["macs"] == 2 and ov["pool"]["capacity"]["running"] == 1
    assert ov["me"]["flops"] == 4e12
    assert ov["me"]["credits"] == {"earned": 4e12, "kept": 3e12, "to_grants": 1e12}
    assert (ov["me"]["rank"], ov["me"]["of"]) == (2, 2)
    assert [n["is_me"] for n in ov["pool"]["nodes"]] == [True, False]


def test_stop_agent_endpoint_leaves_coordinator(tmp_path, monkeypatch):
    kills = []
    monkeypatch.setattr("os.kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr("slashcompute.agent.daemon._alive", lambda pid: True)
    app, launcher, _ = _shell(tmp_path)
    (tmp_path / "coordinator.pid").write_text("111\n")
    launcher.paths.pid_file.write_text("222\n")
    with TestClient(app, base_url=SHELL) as c:
        assert c.post("/api/stop-agent").status_code == 200
    assert (222, 15) in kills and (111, 15) not in kills


class GrantHTTP:
    """In-memory coordinator stand-in for the live grants adapter."""

    def __init__(self) -> None:
        self.grants = [
            {"id": "g1", "title": "Parser", "author": "Ada",
             "body": "Need FLOPs for a parser thesis project.",
             "goal_flops": 50e12, "received_flops": 10e12, "status": "approved", "progress": 0.2},
            {"id": "g2", "title": "Waiting", "author": "Bea",
             "body": "Waiting for review of this grant request.",
             "goal_flops": 20e12, "received_flops": 0.0, "status": "pending", "progress": 0.0},
        ]
        self.donated = 0.0

    def _path(self, url: str) -> str:
        return "/" + url.split("://", 1)[-1].split("/", 1)[-1].split("?")[0]

    def _resp(self, payload, status=200):
        raw = json.dumps(payload).encode()

        class R:
            status_code = status
            content = raw
            headers = {"content-type": "application/json"}

            def json(self_inner):
                return payload
        return R()

    def get(self, url: str, timeout: float = 1.0, params=None, headers=None):
        path = self._path(url)
        if path == "/grants":
            return self._resp(self.grants)
        if path == "/auth/me":
            return self._resp({
                "user": {"id": "u1", "name": "Ada", "admin": True, "grant_split": 10,
                         "accepted_terms": True},
                "credits": {"balance": 80e12, "lifetime_earned": 100e12},
            })
        if path == "/credits/transactions":
            return self._resp({"items": [
                {"kind": "donate", "amount": -self.donated} if self.donated else {"kind": "earn", "amount": 1},
            ]})
        if path == "/community/leaderboard":
            return self._resp([{"user_id": "u1", "name": "Ada", "lifetime_earned": 100e12}])
        raise ConnectionError("down")

    def post(self, url: str, content=None, headers=None, timeout: float = 1.0, **_):
        path = self._path(url)
        body = json.loads(content or b"{}")
        if path == "/grants":
            title = str(body.get("title", "")).strip()
            text = str(body.get("body", "")).strip()
            if len(title) < 4 or len(text) < 20:
                return self._resp({"detail": "Describe the need: a title and at least a short paragraph."}, 400)
            row = {"id": "g3", "title": title, "author": "You", "body": text,
                   "goal_flops": float(body.get("goal_flops") or 0), "received_flops": 0.0,
                   "status": "pending", "progress": 0.0}
            self.grants.append(row)
            return self._resp(row)
        if path.endswith("/donate"):
            gid = path.split("/")[2]
            g = next((x for x in self.grants if x["id"] == gid), None)
            if g is None or g["status"] != "approved":
                return self._resp({"detail": "Only approved grants can receive FLOPs."}, 400)
            flops = float(body.get("flops") or 0)
            if flops <= 0:
                return self._resp({"detail": "Enter a number."}, 400)
            g["received_flops"] += flops
            g["progress"] = g["received_flops"] / g["goal_flops"]
            self.donated += flops
            return self._resp(g)
        if "/admin/grants/" in path and path.endswith("/review"):
            gid = path.split("/")[3]
            g = next((x for x in self.grants if x["id"] == gid), None)
            if g is None or g["status"] != "pending":
                return self._resp({"detail": "This grant was already reviewed."}, 400)
            g["status"] = "approved" if body.get("approve") else "declined"
            return self._resp(g)
        return self._resp({"detail": "not found"}, 404)


def test_proxy_forwards_set_cookie(tmp_path):
    app, launcher, _ = _shell(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app, base_url=SHELL) as c:
        r = c.post("/api/coord/auth/login", json={"email": "ada@lan.test", "password": "password1"})
        assert r.status_code == 200
        assert "slashcompute_session=sess" in r.headers.get("set-cookie", "")


def test_proxy_never_reuses_another_browsers_session(tmp_path, monkeypatch):
    monkeypatch.setattr("slashcompute.community.auth.ITERATIONS", 1)
    coord_app = create_app(EngineConfig(home=tmp_path / "coord", scheduler_tick_s=0.05))
    with TestClient(coord_app) as coord:
        app, launcher, _ = _shell(tmp_path / "shell", http=stateless_http(transport=coord._transport))
        launcher.save_settings(LauncherSettings(mode="join", url="http://testserver"))
        with TestClient(app, base_url=SHELL) as c:
            r = c.post("/api/coord/auth/register", json={
                "email": "ada@lan.test", "password": "password1", "name": "Ada"})
            assert r.status_code == 200, r.text
            assert "slashcompute_session=" in r.headers.get("set-cookie", "")
            assert c.get("/api/coord/auth/me").json()["user"]["email"] == "ada@lan.test"
            c.cookies.clear()
            assert c.get("/api/coord/auth/me").json()["user"] is None
        assert not launcher._http.cookies


def test_live_grants_empty_when_coordinator_down(tmp_path):
    app, _, _ = _shell(tmp_path, http=RoutedHTTP({}))
    with TestClient(app, base_url=SHELL) as c:
        board = c.get("/api/grants?sort=least").json()
    assert board["sample"] is True
    assert board["online"] is False
    assert any("Irish-language" in g["title"] for g in board["grants"])
    assert board["pending"]


def test_sample_grants_when_coordinator_has_none(tmp_path):
    app, launcher, _ = _shell(tmp_path, http=RoutedHTTP({**POOL, "/grants": []}))
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app, base_url=SHELL) as c:
        board = c.get("/api/grants?sort=top").json()
    assert board["sample"] is True
    assert board["online"] is True
    assert any("Irish-language" in g["title"] for g in board["grants"])
    assert all(g["id"].startswith("g") for g in board["grants"])


def test_live_grants_are_not_mixed_with_samples(tmp_path):
    grant = {"id": "live-1", "title": "Parser", "author": "Ada",
             "body": "Need FLOPs for a parser.", "goal_flops": 50e12,
             "received_flops": 10e12, "status": "approved"}
    app, launcher, _ = _shell(tmp_path, http=RoutedHTTP({**POOL, "/grants": [grant]}))
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app, base_url=SHELL) as c:
        board = c.get("/api/grants?sort=top").json()
    assert board["sample"] is False
    assert [g["id"] for g in board["grants"]] == ["live-1"]
    assert not any("Irish-language" in g["title"] for g in board["grants"])


# Runs app.js under Node with a stub DOM; fetch answers from the `routes` the
# test swaps in, so a poll can see the coordinator come and go.


APP_JS = Path(__file__).resolve().parents[1] / "src/slashcompute/web/static/app.js"


DOM_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const [appJs, phasesFile] = process.argv.slice(1);
const phases = JSON.parse(fs.readFileSync(phasesFile, "utf8"));
const stub = () => new Proxy(function () {}, {
  get: (_, k) => (k === Symbol.toPrimitive ? () => "" : k === "then" ? undefined : stub()),
  apply: () => stub(),
});
const els = new Map();
const element = (sel) => {
  if (!els.has(sel)) {
    const props = { textContent: "", innerHTML: "", className: "", hidden: false, value: "",
      dataset: {}, style: { setProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } };
    els.set(sel, new Proxy(props, { get: (o, k) => (k in o ? o[k] : stub()) }));
  }
  return els.get(sel);
};
let routes = {};
const ctx = {
  document: { querySelector: element, querySelectorAll: () => [], addEventListener() {},
    createElement: element, body: stub(), activeElement: null },
  window: { setInterval() {}, setTimeout() {}, clearTimeout() {}, confirm: () => true },
  CSS: { escape: (s) => s }, navigator: stub(), XMLHttpRequest: function () {},
  FormData: function () {}, console, calls: [],
  fetch: async (path, opts = {}) => {
    ctx.calls.push({ path, method: opts.method || "GET", body: opts.body ? JSON.parse(opts.body) : null });
    const body = routes[path.split("?")[0]];
    const ok = body !== undefined;
    return { ok, status: ok ? 200 : 404, statusText: "x",
      text: async () => JSON.stringify(ok ? body : { detail: "nope" }) };
  },
};
vm.createContext(ctx);
const settle = () => new Promise((r) => setTimeout(r, 30));
(async () => {
  const seen = [];
  for (const [i, phase] of phases.entries()) {
    routes = phase.routes;
    if (i === 0) {
      vm.runInContext(fs.readFileSync(appJs, "utf8"), ctx);
      vm.runInContext("showTab('grants')", ctx);
    } else {
      vm.runInContext(phase.run, ctx);
    }
    await settle();
    seen.push({ pill: element("#g-pill").textContent, open: element("#g-open").textContent,
      list: element("#g-list").innerHTML, probe: phase.probe && vm.runInContext(phase.probe, ctx) });
  }
  console.log(JSON.stringify(seen));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_grants_tab_follows_coordinator_on_poll(tmp_path):
    grant = {"id": "g1", "title": "Parser", "author": "Ada", "body": "Need FLOPs for a parser.",
             "goal_flops": 50e12, "received_flops": 10e12, "status": "approved"}
    app, launcher, _ = _shell(tmp_path / "up", http=RoutedHTTP({**POOL, "/grants": [grant]}))
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app, base_url=SHELL) as c:
        live = {"/api/overview": c.get("/api/overview").json(), "/api/grants": c.get("/api/grants").json()}
    assert live["/api/overview"]["status"]["coordinator_up"] and live["/api/grants"]["online"]
    app, _, _ = _shell(tmp_path / "down", http=RoutedHTTP({}))
    with TestClient(app, base_url=SHELL) as c:
        down = {"/api/overview": c.get("/api/overview").json(), "/api/grants": c.get("/api/grants").json()}
    assert not down["/api/overview"]["status"]["coordinator_up"]
    board = live["/api/grants"]
    more = {**live, "/api/grants": {**board, "grants": board["grants"] + [{**board["grants"][0], "id": "g2"}]}}
    phases = [
        {"routes": live},
        {"routes": down, "run": "poll()"},
        {"routes": live, "run": "poll()"},
        # Same coordinator, new grant: only the slow refresh picks it up.
        {"routes": more, "run": "poll()"},
        {"routes": more, "run": "state.grantsAt -= 60000; poll()"},
    ]
    (tmp_path / "phases.json").write_text(json.dumps(phases))
    out = subprocess.run(["node", "-e", DOM_HARNESS, str(APP_JS), str(tmp_path / "phases.json")],
                         capture_output=True, text=True, timeout=30, check=True)
    seen = json.loads(out.stdout.strip().splitlines()[-1])
    assert (seen[0]["pill"], seen[0]["open"]) == ("Live", "1")
    assert "Fund this grant" in seen[0]["list"]
    assert (seen[1]["pill"], seen[1]["open"]) == ("Demo", "6")
    assert "Fund this grant" not in seen[1]["list"]
    assert "Irish-language" in seen[1]["list"]
    assert "Demo grant" in seen[1]["list"]
    assert (seen[2]["pill"], seen[2]["open"]) == ("Live", "1")
    assert seen[3]["open"] == "1"
    assert seen[4]["open"] == "2"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_chat_shows_a_thinking_models_reasoning(tmp_path):
    """reasoning_content deltas render as a dim Thinking… block before the answer, so a thinking
    model's reply is never blank, even when it runs out of tokens before answering."""
    log = 'renderChat(); $("#l-log").innerHTML'
    phases = [
        {"routes": {}},
        {"routes": {}, "probe": log, "run": """
            state.llm.messages = [{ role: "user", content: "Is 1001 a prime number?" },
                                  { role: "assistant", content: "", live: true }];
            applyChatEvent(state.llm.messages[1], { choices: [{ delta: { role: "assistant" } }] });
            applyChatEvent(state.llm.messages[1], { choices: [{ delta: { reasoning_content: "1001 = 7 * 11" } }] });
            applyChatEvent(state.llm.messages[1], { choices: [{ delta: { reasoning_content: " * 13 <so no>" } }] });"""},
        {"routes": {}, "probe": log,
         "run": 'applyChatEvent(state.llm.messages[1], { choices: [{ delta: { content: "No." } }] })'},
        # the user opened it: the next render keeps it open
        {"routes": {}, "probe": log, "run": "state.llm.messages[1].thinkOpen = true"},
        {"routes": {}, "probe": log, "run": """
            const r = { role: "assistant", content: "", live: true };
            state.llm.messages.push(r);
            applyChatEvent(r, { choices: [{ delta: { reasoning_content: "Hmm" }, finish_reason: "length" }] });
            r.live = false;"""},
    ]
    (tmp_path / "phases.json").write_text(json.dumps(phases))
    out = subprocess.run(["node", "-e", DOM_HARNESS, str(APP_JS), str(tmp_path / "phases.json")],
                         capture_output=True, text=True, timeout=30, check=True)
    thinking, answered, reopened, ran_out = [s["probe"] for s in json.loads(out.stdout.strip().splitlines()[-1])[1:]]
    assert "<summary>Thinking…</summary>" in thinking and 'open=""' in thinking
    assert "1001 = 7 * 11 * 13 &lt;so no&gt;" in thinking and "<p>…</p>" not in thinking
    assert "<summary>Thoughts</summary>" in answered and 'open=""' not in answered
    assert answered.index("1001 = 7") < answered.index("<p>No.</p>")
    assert 'open=""' in reopened
    assert "Ran out of tokens while thinking" in ran_out.split("Hmm")[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_start_hosting_stays_clickable_after_the_coordinator_fails(tmp_path):
    """A coordinator that died (its port was taken) must not leave Start hosting disabled."""
    def probe(st):
        return (f'state.settings = {{ mode: "host" }}; state.ov = {{ status: {json.dumps(st)} }}; '
                'renderPool(); [$("#p-start").disabled, $("#p-start").textContent]')
    port = "Port 8765 is already in use — quit the other /compute or coordinator, then Start hosting."
    cases = [
        {"coordinator_pid": 4001, "coordinator_up": True, "last_error": ""},
        {"coordinator_pid": None, "coordinator_up": False, "last_error": port},
        # a status from before the launcher noticed the exit, or a pid it cannot vouch for
        {"coordinator_pid": 4001, "coordinator_up": False, "last_error": port},
    ]
    phases = [{"routes": {}}] + [{"routes": {}, "run": "0", "probe": probe(st)} for st in cases]
    (tmp_path / "phases.json").write_text(json.dumps(phases))
    out = subprocess.run(["node", "-e", DOM_HARNESS, str(APP_JS), str(tmp_path / "phases.json")],
                         capture_output=True, text=True, timeout=30, check=True)
    hosting, failed, stale = [s["probe"] for s in json.loads(out.stdout.strip().splitlines()[-1])[1:]]
    assert hosting == [True, "Hosting"]
    assert failed == [False, "Start hosting"]
    assert stale == [False, "Start hosting"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_contributing_pill_shows_the_running_share(tmp_path):
    """Moving the slider while contributing saves the next share; the pills keep the live one."""
    def probe(node):
        ov = {"status": {"agent_running": True}, "me": {"flops": 0, **({"node": node} if node else {})}}
        return (f'state.settings = {{ gpu_percent: 30 }}; state.ov = {json.dumps(ov)}; '
                'renderContributions(); renderSidebar(); [$("#c-pill").textContent, $("#side-agent").textContent]')
    phases = [{"routes": {}}] + [{"routes": {}, "run": "0", "probe": probe(n)}
                                 for n in ({"gpu_percent": 50}, None)]
    (tmp_path / "phases.json").write_text(json.dumps(phases))
    out = subprocess.run(["node", "-e", DOM_HARNESS, str(APP_JS), str(tmp_path / "phases.json")],
                         capture_output=True, text=True, timeout=30, check=True)
    live, unlisted = [s["probe"] for s in json.loads(out.stdout.strip().splitlines()[-1])[1:]]
    assert live == ["Contributing · 50%", "Contributing · 50%"]
    # not in the pool's node list yet: fall back to the saved share
    assert unlisted == ["Contributing · 30%", "Contributing · 30%"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_sign_out_after_registering_shows_sign_in(tmp_path):
    routes = {"/api/coord/auth/logout": {}, "/api/settings": {"session_token": ""}}
    phases = [
        {"routes": {}},
        {"routes": routes, "run": """
            state.user = { id: "u1", email: "ada@example.com", accepted_terms: true };
            actions["auth-mode"]();
            actions.logout(null);""",
         "probe": """[state.authMode, $("#auth-submit").textContent, $("[data-act='auth-mode']").textContent]"""},
    ]
    (tmp_path / "phases.json").write_text(json.dumps(phases))
    out = subprocess.run(["node", "-e", DOM_HARNESS, str(APP_JS), str(tmp_path / "phases.json")],
                         capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1])[1]["probe"] == ["login", "Sign in", "Create account"]


def test_live_grants_flow(tmp_path):
    T = 1e12
    app, launcher, _ = _shell(tmp_path, http=GrantHTTP())
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app, base_url=SHELL) as c:
        board = c.get("/api/grants?sort=top").json()
        assert board["sample"] is False and board["online"] is True
        assert board["grants"][0]["summary"].startswith("Need FLOPs")
        assert board["grants"][0]["goal"] == 50e12
        assert board["pending"][0]["title"] == "Waiting"
        assert board["leaders"][0]["name"] == "Ada"
        assert board["available"] == 80e12

        funded = c.post("/api/grants/g1/fund", json={"amount": 10 * T})
        assert funded.status_code == 200, funded.text
        assert funded.json()["pledged"] == 10 * T
        assert funded.json()["grants"][0]["raised"] == 20e12

        made = c.post("/api/grants", json={
            "title": "Lecture notes",
            "summary": "Fine-tune a helper on my course notes for first years.",
            "goal": 50 * T,
        })
        assert made.status_code == 200, made.text
        assert any(g["title"] == "Lecture notes" for g in made.json()["pending"])
        assert c.post("/api/grants", json={"title": "", "summary": "x", "goal": 1}).status_code == 400
        assert c.post("/api/grants", json={"title": "t", "summary": "x",
                                           "goal": "1e309"}).status_code == 400
        assert c.post("/api/grants/g1/fund", json={"amount": "nan"}).status_code == 400

        approved = c.post("/api/grants/g2/review", json={"approve": True}).json()
        assert any(g["id"] == "g2" for g in approved["grants"])
        again = c.post("/api/grants/g2/review", json={"approve": True})
        assert again.status_code == 400
        # bool("false") is True: the shell used to forward this as an approval.
        assert c.post("/api/grants/g3/review", json={"approve": "yes"}).status_code == 400
        declined = c.post("/api/grants/g3/review", json={"approve": "false"})
        assert declined.status_code == 200, declined.text
        assert all(g["id"] != "g3" for g in declined.json()["grants"])


def test_grant_amounts_must_be_positive_and_finite(tmp_path):
    http = GrantHTTP()
    app, launcher, _ = _shell(tmp_path, http=http)
    launcher.save_settings(LauncherSettings(mode="host"))
    summary = "Fine-tune a helper on my course notes for first years."
    with TestClient(app, base_url=SHELL) as c:
        # Raw bodies: the JSON spec has no NaN/Infinity, but Python's parser (and float("inf")) accepts them.
        for bad in ("NaN", "Infinity", "-Infinity", "0", "-5", '"nan"', '"inf"', '"1e999"'):
            made = c.post("/api/grants", content=f'{{"title": "Lecture notes", "summary": "{summary}", '
                                                 f'"goal": {bad}}}',
                          headers={"content-type": "application/json"})
            assert made.status_code == 400, (bad, made.text)
            assert "finite" in made.json()["detail"]
            funded = c.post("/api/grants/g1/fund", content=f'{{"amount": {bad}}}',
                            headers={"content-type": "application/json"})
            assert funded.status_code == 400, (bad, funded.text)
        assert c.post("/api/grants/g1/fund", json={"amount": "lots"}).status_code == 400
        assert c.get("/api/grants").status_code == 200
    assert len(http.grants) == 2 and http.donated == 0.0


def test_non_finite_settings_are_clamped_not_500(tmp_path):
    app, _, _ = _shell(tmp_path)
    with TestClient(app, base_url=SHELL) as c:
        r = c.post("/api/settings", content='{"gpu_percent": Infinity, "grant_split": -Infinity, '
                                            '"memory_gb": NaN, "inference_memory_gb": 1e999}',
                   headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["gpu_percent"], body["grant_split"], body["memory_gb"], body["inference_memory_gb"]) == \
        (50, 0, 0, 0)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_connect_moves_what_this_mac_lends_and_labels_follow_the_mode(tmp_path):
    """Connect restarts the running agent and LLM node on the new pool (it used to only save the
    address), and a coordinator still running here reads as Hosting only in host mode."""
    up = {"agent_running": True, "inference_running": True, "coordinator_pid": 4001, "coordinator_up": True}
    routes = {"/api/start": {"last_error": ""},
              "/api/overview": {"status": {"coordinator_up": True, "coordinator_url": "http://10.0.0.8:8765"}}}

    def labels(mode):
        return (f'state.settings = {{ mode: "{mode}" }}; state.ov = {{ status: {json.dumps(up)} }}; '
                'renderSidebar(); renderPool(); [$("#side-title").textContent, $("#p-status").innerHTML]')

    def connect(confirmed):
        return (f'calls.length = 0; window.confirm = () => {json.dumps(confirmed)}; '
                f'state.settings = {{ mode: "join", url: "", gpu_percent: 50 }}; state.ov = {{ status: {json.dumps(up)} }}; '
                '$("#url").value = "10.0.0.8"; actions.connect({})')

    starts = 'calls.filter((c) => c.path === "/api/start").map((c) => c.body)'
    phases = [{"routes": {}},
              {"routes": {}, "run": "0", "probe": labels("host")},
              {"routes": {}, "run": "0", "probe": labels("join")},
              {"routes": routes, "run": connect(False), "probe": starts},
              {"routes": routes, "run": connect(True), "probe": starts}]
    (tmp_path / "phases.json").write_text(json.dumps(phases))
    out = subprocess.run(["node", "-e", DOM_HARNESS, str(APP_JS), str(tmp_path / "phases.json")],
                         capture_output=True, text=True, timeout=30, check=True)
    host, join, declined, started = [s["probe"] for s in json.loads(out.stdout.strip().splitlines()[-1])[1:]]
    assert host[0] == "Hosting" and "hosting here" in host[1] and "Pool hosted here" not in host[1]
    assert join[0] == "Joined" and "reachable" in join[1] and "Pool hosted here" in join[1]
    assert declined == []   # keeping the pool hosted here: nothing changes
    assert len(started) == 1
    assert {k: started[0][k] for k in ("mode", "url", "training", "inference")} == {
        "mode": "join", "url": "10.0.0.8", "training": True, "inference": True}
