"""A build that runs beside another /compute on the same Mac (see scripts/macos/build_dmg.sh)
takes its ports and Bonjour service from the environment. Without those variables nothing changes."""

import os
import subprocess
import sys

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from slashcompute.agent import main as agent_main
from slashcompute.launcher.controller import Launcher
from slashcompute.web.server import create_shell


def _launcher(tmp_path, found="10.0.0.9"):
    return Launcher(home=tmp_path, python="/opt/venv/bin/python", popen=lambda *a, **k: None,
                    discover_fn=lambda timeout=5.0: found, lan_ip_fn=lambda: "192.168.1.20")


def test_status_reports_the_usual_ports(tmp_path):
    with TestClient(create_shell(_launcher(tmp_path))) as c:
        st = c.get("/api/status").json()
    assert (st["coordinator_port"], st["data_port"]) == (8765, 9700)


def test_ports_from_the_environment_reach_the_ui_the_lan_search_and_the_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("SLASHCOMPUTE_COORDINATOR_PORT", "8775")
    monkeypatch.setenv("SLASHCOMPUTE_DATA_PORT", "9710")
    launcher = _launcher(tmp_path)
    with TestClient(create_shell(launcher)) as c:
        st = c.get("/api/status").json()
    assert (st["coordinator_port"], st["data_port"]) == (8775, 9710)
    # an address found without a port gets this build's port, not the usual one
    assert launcher.find_on_lan() == "http://10.0.0.9:8775"

    seen = {}

    class Stop(Exception):
        pass

    def options(**kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(agent_main, "AgentOptions", options)
    result = CliRunner().invoke(agent_main.app, ["start", "--url", "http://10.0.0.9:8775",
                                                "--home", str(tmp_path)])
    assert isinstance(result.exception, Stop)
    assert seen["data_port"] == 9710
    # an explicit --data-port still wins
    CliRunner().invoke(agent_main.app, ["start", "--data-port", "9800", "--home", str(tmp_path)])
    assert seen["data_port"] == 9800


def test_bonjour_service_from_the_environment():
    # read in a fresh interpreter: the service type is fixed when the module is first imported
    def service(**env):
        code = "from slashcompute.common.discovery import MDNS_SERVICE_TYPE as s; print(s)"
        return subprocess.run([sys.executable, "-c", code], env={**os.environ, **env},
                              capture_output=True, text=True, check=True).stdout.strip()

    assert service(SLASHCOMPUTE_MDNS_SERVICE="") == "_slashcompute._tcp.local."
    assert (service(SLASHCOMPUTE_MDNS_SERVICE="_slashcompute-classic._tcp.local.")
            == "_slashcompute-classic._tcp.local.")
