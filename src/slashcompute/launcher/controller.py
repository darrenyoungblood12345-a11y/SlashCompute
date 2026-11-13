"""Start/stop the coordinator and agent; persist launcher settings."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from slashcompute.agent.daemon import request_stop
from slashcompute.agent.paths import AgentPaths
from slashcompute.common.config import EngineConfig
from slashcompute.common.discovery import discover, lan_ip


class LauncherError(Exception):
    """User-facing start/stop failure."""


FINISHES = ("carbon", "poster", "signal", "thermal", "void")


@dataclass
class LauncherSettings:
    mode: str = "host"
    url: str = ""
    gpu_percent: int = 50
    contribute: bool = True
    finish: str = "carbon"
    session_token: str = ""
    grant_split: int = 0

    def clamp(self) -> "LauncherSettings":
        mode = self.mode if self.mode in ("host", "join") else "host"
        finish = self.finish if self.finish in FINISHES else "carbon"
        try:
            gpu = max(1, min(100, int(self.gpu_percent)))
        except (TypeError, ValueError):
            gpu = 50
        try:
            split = max(0, min(100, int(self.grant_split)))
        except (TypeError, ValueError):
            split = 0
        return LauncherSettings(
            mode=mode, url=str(self.url or ""), gpu_percent=gpu,
            contribute=bool(self.contribute), finish=finish,
            session_token=str(self.session_token or ""), grant_split=split,
        )


@dataclass
class StatusSnapshot:
    coordinator_up: bool = False
    nodes: int = 0
    jobs: int = 0
    agent_running: bool = False
    agent_status: str = ""
    agent_job_id: Optional[str] = None
    coordinator_pid: Optional[int] = None
    agent_pid: Optional[int] = None
    lan_ip: str = ""
    last_error: str = ""


PopenFn = Callable[..., Any]


def normalize_url(url: str, port: int = 8765) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if "://" not in u:
        if ":" not in u.split("/")[0]:
            u = f"{u}:{port}"
        u = f"http://{u}"
    return u.rstrip("/")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class Launcher:
    def __init__(
        self,
        home: Optional[Path] = None,
        python: Optional[str] = None,
        popen: PopenFn = subprocess.Popen,
        http: Optional[httpx.Client] = None,
        discover_fn: Callable[[float], Optional[str]] = discover,
        lan_ip_fn: Callable[[], str] = lan_ip,
    ) -> None:
        self.cfg = EngineConfig.from_env(home=home)
        if home is not None:
            self.cfg.home = Path(home)
        self.home = Path(self.cfg.home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.python = python or sys.executable
        self._popen = popen
        self._http = http or httpx.Client()
        self._discover = discover_fn
        self._lan_ip = lan_ip_fn
        self.last_error = ""
        self.paths = AgentPaths(self.home)

    @property
    def settings_path(self) -> Path:
        return self.home / "launcher.json"

    @property
    def coordinator_pid_path(self) -> Path:
        return self.home / "coordinator.pid"

    @property
    def log_dir(self) -> Path:
        d = self.home / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load_settings(self) -> LauncherSettings:
        if not self.settings_path.exists():
            return LauncherSettings()
        try:
            raw = json.loads(self.settings_path.read_text())
        except (OSError, json.JSONDecodeError):
            return LauncherSettings()
        if not isinstance(raw, dict):
            return LauncherSettings()
        return LauncherSettings(
            mode=raw.get("mode", "host"),
            url=raw.get("url", ""),
            gpu_percent=raw.get("gpu_percent", 50),
            contribute=raw.get("contribute", True),
            finish=raw.get("finish", "carbon"),
            session_token=raw.get("session_token", ""),
            grant_split=raw.get("grant_split", 0),
        ).clamp()

    def save_settings(self, settings: LauncherSettings) -> None:
        s = settings.clamp()
        self.settings_path.write_text(json.dumps(asdict(s), indent=2) + "\n")

    def coordinator_url(self, settings: LauncherSettings) -> str:
        if settings.mode == "host":
            return f"http://{self._lan_ip()}:{self.cfg.coordinator_port}"
        return normalize_url(settings.url, self.cfg.coordinator_port)

    def proxy_url(self, settings: Optional[LauncherSettings] = None) -> str:
        """Coordinator URL the local shell should dial (loopback when hosting)."""
        s = settings.clamp() if settings is not None else self.load_settings()
        if s.mode == "host":
            return f"http://127.0.0.1:{self.cfg.coordinator_port}"
        return self.coordinator_url(s)

    def coordinator_argv(self) -> list[str]:
        return [self.python, "-m", "slashcompute.coordinator.main", "serve",
                "--home", str(self.home)]

    def agent_argv(self, url: str, gpu_percent: int, session_token: str = "") -> list[str]:
        argv = [
            self.python, "-m", "slashcompute.agent.main", "start",
            "--url", url, "--gpu-percent", str(int(gpu_percent)),
            "--no-sandbox", "--home", str(self.home),
        ]
        if session_token:
            argv.extend(["--session-token", session_token])
        return argv

    def read_coordinator_pid(self) -> Optional[int]:
        if not self.coordinator_pid_path.exists():
            return None
        try:
            pid = int(self.coordinator_pid_path.read_text().strip())
        except ValueError:
            return None
        if process_alive(pid):
            return pid
        self.coordinator_pid_path.unlink(missing_ok=True)
        return None

    def write_coordinator_pid(self, pid: int) -> None:
        self.coordinator_pid_path.write_text(str(pid) + "\n")

    def poll_health(self, url: str) -> Optional[dict]:
        if not url:
            return None
        try:
            r = self._http.get(f"{url.rstrip('/')}/health", timeout=1.0)
            if r.status_code == 200:
                data = r.json()
                return data if isinstance(data, dict) else {"ok": True}
        except Exception:
            return None
        return None

    def wait_health(self, url: str, timeout: float = 8.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.poll_health(url):
                return True
            time.sleep(0.2)
        return False

    def find_on_lan(self, timeout: float = 5.0) -> Optional[str]:
        found = self._discover(timeout)
        return normalize_url(found, self.cfg.coordinator_port) if found else None

    def start(self, settings: LauncherSettings) -> StatusSnapshot:
        s = settings.clamp()
        self.save_settings(s)
        self.last_error = ""
        url = self.coordinator_url(s)
        if s.mode == "join" and not url:
            self.last_error = "Enter a coordinator URL, or find one on the LAN."
            raise LauncherError(self.last_error)

        want_coord = s.mode == "host"
        want_agent = s.mode == "join" or s.contribute

        if want_coord and not self.poll_health(url):
            self._spawn(self.coordinator_argv(), self.log_dir / "coordinator.log",
                        pid_writer=self.write_coordinator_pid)
            check = self.proxy_url(s) if s.mode == "host" else url
            if not (self.wait_health(check) or self.poll_health(url)):
                self.last_error = (
                    f"Coordinator started but is not answering {url}/health yet. "
                    f"Watch {self.log_dir / 'coordinator.log'}."
                )

        agent_url = self.proxy_url(s) if s.mode == "host" else url
        if not want_agent and self._agent_running():
            request_stop(self.paths)
        if want_agent:
            if s.mode == "join" and not self.poll_health(url):
                self.last_error = f"No coordinator at {url}."
                raise LauncherError(self.last_error)
            # A running agent keeps the URL, GPU limit and session it was started
            # with, so restart it whenever any of those no longer match.
            argv = self.agent_argv(agent_url, s.gpu_percent, s.session_token)
            if self._agent_running() and self._read_agent_args() != argv:
                request_stop(self.paths)
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and self._agent_running():
                    time.sleep(0.05)
                if self._agent_running():
                    self.last_error = (
                        "Training agent is still stopping. Settings have not been applied; "
                        "start again after its current work finishes."
                    )
            if not self._agent_running():
                self._spawn(argv, self.log_dir / "agent.log")
                self._write_agent_args(argv)

        snap = self.snapshot(s)
        if snap.coordinator_up and "not answering" in (self.last_error or ""):
            self.last_error = ""
            snap.last_error = ""
        return snap

    def stop(self) -> StatusSnapshot:
        self.last_error = ""
        request_stop(self.paths)
        pid = self.read_coordinator_pid()
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                self.coordinator_pid_path.unlink(missing_ok=True)
        return self.snapshot()

    def snapshot(self, settings: Optional[LauncherSettings] = None) -> StatusSnapshot:
        s = settings.clamp() if settings is not None else self.load_settings()
        url = self.coordinator_url(s)
        health = self.poll_health(url) or {}
        if not health and s.mode == "host":
            health = self.poll_health(self.proxy_url(s)) or {}
        agent = self.paths.read_status()
        agent_pid = self.paths.read_pid()
        agent_running = bool(agent_pid and process_alive(agent_pid))
        if agent_pid and not agent_running:
            self.paths.clear_pid()
            agent_pid = None
        return StatusSnapshot(
            coordinator_up=bool(health),
            nodes=int(health.get("nodes", 0) or 0),
            jobs=int(health.get("jobs", 0) or 0),
            agent_running=agent_running,
            agent_status=str(agent.get("status", "") or ""),
            agent_job_id=agent.get("job_id"),
            coordinator_pid=self.read_coordinator_pid(),
            agent_pid=agent_pid if agent_running else None,
            lan_ip=self._lan_ip(),
            last_error=self.last_error,
        )

    def _agent_args_path(self) -> Path:
        return self.home / "agent.args"

    def _read_agent_args(self) -> list[str]:
        try:
            return json.loads(self._agent_args_path().read_text())
        except (OSError, ValueError):
            return []

    def _write_agent_args(self, argv: list[str]) -> None:
        path = self._agent_args_path()
        # argv can carry the session token: keep it private to the user
        path.touch(mode=0o600)
        path.chmod(0o600)
        path.write_text(json.dumps(argv))

    def _agent_running(self) -> bool:
        pid = self.paths.read_pid()
        return bool(pid and process_alive(pid))

    def _spawn(self, argv: list[str], log_path: Path,
               pid_writer: Optional[Callable[[int], None]] = None) -> Any:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "ab")
        try:
            proc = self._popen(
                argv, stdout=fh, stderr=subprocess.STDOUT,
                start_new_session=True, env=os.environ.copy(),
            )
        except OSError as e:
            fh.close()
            self.last_error = f"Could not start process: {e}"
            raise LauncherError(self.last_error) from e
        if pid_writer is not None:
            pid_writer(int(proc.pid))
        return proc
