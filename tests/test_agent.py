import asyncio

import pytest

from slashcompute.agent.paths import resolve_data_host
from slashcompute.agent.throttle import sleep_s
from slashcompute.transport import Frame, LinkServer, TcpLink


def test_sleep_s_duty_cycle():
    assert sleep_s(2.0, 100) == 0.0
    assert sleep_s(2.0, 50) == pytest.approx(2.0)
    assert sleep_s(1.0, 25) == pytest.approx(3.0)
    assert sleep_s(1.0, 0) == 3600.0


def test_resolve_data_host_localhost():
    assert resolve_data_host(True) == "127.0.0.1"
    assert resolve_data_host(False) != ""


def test_benchmark_profile_shape():
    from slashcompute.agent.benchmark import benchmark
    from slashcompute.common.protocol import DeviceProfile

    d = benchmark(max_memory_bytes=512 * 1024 * 1024)
    assert isinstance(d, DeviceProfile)
    assert d.memory_contrib_bytes <= 512 * 1024 * 1024
    assert d.matmul_tflops > 0 and d.mem_bandwidth_gbps > 0
    assert d.chip


async def test_hello_reject_does_not_leak_expect():
    server = await LinkServer("127.0.0.1", 0, {"job_id": "secret", "epoch": 9}).start()
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    link = TcpLink(reader, writer)
    await link.send(Frame("hello", {"job_id": "nope", "epoch": 0}))
    ack = await link.recv(5)
    assert ack.kind == "hello_reject"
    assert "expect" not in ack.meta and "secret" not in str(ack.meta)
    await link.close()
    await server.close()


def test_agent_options_localhost_host(tmp_path):
    from slashcompute.agent.daemon import AgentOptions

    opt = AgentOptions(url="http://127.0.0.1:8765", home=tmp_path, localhost=True)
    assert opt.data_host == "127.0.0.1"
    assert opt.node_id == AgentOptions(url="http://127.0.0.1:8765", home=tmp_path).node_id


# --------------------------------------------------------------------------- node token


def _capture(monkeypatch):
    """Route CoordHTTP's requests into a list instead of the network."""
    import httpx

    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, content=b"blob")

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(httpx, "stream", client.stream)
    monkeypatch.setattr(httpx, "post", client.post)
    return requests


@pytest.mark.parametrize("token", [None, "node-secret"])
@pytest.mark.parametrize("path", ["/jobs/j/dataset", "http://pool.test:8765/jobs/j/dataset"])
def test_agent_presents_its_node_token_to_the_coordinator(monkeypatch, tmp_path, token, path):
    from slashcompute.agent.http import CoordHTTP

    requests = _capture(monkeypatch)
    coord = CoordHTTP("http://pool.test:8765", node_token=token)
    assert coord.get_file(path, tmp_path / "dataset").read_bytes() == b"blob"
    coord.put_bytes("/jobs/j/checkpoints/1", b"checkpoint", params={"epoch": 1, "stage": 0})
    assert [r.method for r in requests] == ["GET", "POST"]
    assert requests[1].content == b"checkpoint" and requests[1].url.params["stage"] == "0"
    assert [r.headers.get("X-Node-Token") for r in requests] == [token, token]


@pytest.mark.parametrize("url", [
    "http://elsewhere.test:8765/dataset",   # another host
    "https://pool.test:8765/dataset",       # another scheme
    "http://pool.test:9999/dataset",        # another port
])
def test_agent_never_sends_its_node_token_to_another_origin(monkeypatch, tmp_path, url):
    from slashcompute.agent.http import CoordHTTP

    requests = _capture(monkeypatch)
    coord = CoordHTTP("http://pool.test:8765", node_token="node-secret")
    coord.get_file(url, tmp_path / "dataset")
    coord.put_bytes(url, b"data")
    assert len(requests) == 2
    assert all("X-Node-Token" not in r.headers for r in requests)


def test_node_token_reaches_the_sandboxed_worker(monkeypatch, tmp_path):
    """The sandboxed worker uploads checkpoints itself, so the token issued to
    the daemon has to travel in its assignment file (owner-readable only)."""
    import json
    import stat
    import sys
    from types import SimpleNamespace

    from slashcompute.agent.daemon import AgentOptions, Daemon
    from slashcompute.agent.worker import _stdio_app
    from slashcompute.common.protocol import LoraFinetuneSpec, StageAssignment

    opt = AgentOptions(url="http://pool.test:8765", home=tmp_path, localhost=True)
    assert opt.http.node_token is None
    opt.http.node_token = "node-secret"  # what the daemon stores from Welcome
    daemon = Daemon(opt)
    assignment = StageAssignment(
        job_id="j", epoch=1, stage_idx=0, num_stages=1, layer_start=0, layer_end=8,
        num_layers=8, spec=LoraFinetuneSpec(dataset_path="dataset.jsonl", steps=2),
        checkpoint_every=25, verify_ring_size=8,
    )
    job_dir = opt.paths.job_dir("j", 1)

    async def spawn(*args, **kwargs):
        return SimpleNamespace(stdout=None, stdin=None, returncode=0)

    monkeypatch.setattr("slashcompute.agent.daemon.wrap_command", lambda cmd, *_: cmd)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    asyncio.run(daemon._start_sandboxed(assignment, job_dir))
    spec_path = job_dir / "assignment.json"
    assert json.loads(spec_path.read_text())["node_token"] == "node-secret"
    assert stat.S_IMODE(spec_path.stat().st_mode) == 0o600

    contexts = []

    async def run_stage(ctx, emit):
        contexts.append(ctx)

    monkeypatch.setattr("slashcompute.agent.worker.run_stage", run_stage)
    monkeypatch.setattr("slashcompute.common.logging.setup_logging", lambda *_: None)
    monkeypatch.setattr(sys, "argv", ["worker", "--assignment", str(spec_path)])
    _stdio_app()
    assert contexts[0].http.node_token == "node-secret"
    assert contexts[0].http.base == "http://pool.test:8765"
    assert contexts[0].assignment == assignment

