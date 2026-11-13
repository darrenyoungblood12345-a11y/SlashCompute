"""The daemon and its sandboxed worker process, over the worker's stdin and stdout.

Which process each test runs:

* ``_started(..., script=...)``: the real stdio CLI (``worker._stdio_app``) around a fake
  stage, as a plain subprocess (no sandbox-exec), so each test controls what the stage does.
* ``real_sandbox=True``: the same under the real ``/usr/bin/sandbox-exec`` and the packaged
  profile (skipped where sandbox-exec does not exist).
* ``test_real_sandboxed_stage_*``: the unmodified worker (``python -m slashcompute.agent.worker``)
  training the tiny model under the real sandbox, against a local stand-in for the
  coordinator's blob store on an ephemeral port.
"""

import asyncio
import json
import os
import shutil
import signal
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import slashcompute
from slashcompute.agent import sandbox
from slashcompute.agent.daemon import AgentOptions, Daemon
from slashcompute.common.protocol import Drain, LoraFinetuneSpec, StageAssignment

SRC = str(Path(slashcompute.__file__).resolve().parents[1])

# The real stdio CLI around a stage that runs until it is asked to drain.
_DRAINABLE_WORKER = """
import asyncio
from types import SimpleNamespace
from slashcompute.agent import worker
from slashcompute.common.protocol import StageFinished, StageReady

async def run_stage(ctx, emit):
    asg = ctx.assignment
    ctx.session.runner = SimpleNamespace(drain=asyncio.Event())
    await emit(StageReady(job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx))
    await ctx.session.runner.drain.wait()
    await emit(StageFinished(job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx,
                             reason="drained", last_step=7))

worker.run_stage = run_stage
worker._stdio_app()
"""

# The same CLI around a stage that never reaches a step boundary.
_STUCK_WORKER = """
import asyncio
from slashcompute.agent import worker
from slashcompute.common.protocol import StageReady

async def run_stage(ctx, emit):
    asg = ctx.assignment
    await emit(StageReady(job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx))
    await asyncio.Event().wait()

worker.run_stage = run_stage
worker._stdio_app()
"""


def _assignment(model="unused", steps=100, **kw):
    spec = LoraFinetuneSpec(model=str(model), dataset_path="dataset.jsonl", steps=steps,
                            batch_size=2, microbatches=1, lora_rank=4, max_seq_len=32)
    fields = dict(job_id="j", epoch=1, stage_idx=0, num_stages=1, layer_start=0, layer_end=6,
                  num_layers=6, spec=spec, checkpoint_every=25, verify_ring_size=8)
    return StageAssignment(**(fields | kw))


class FakeWS:
    """The coordinator's end of the agent's websocket: records what the agent sends."""

    def __init__(self):
        self.sent, self.closed = [], False

    async def send(self, raw):
        assert not self.closed, "sent after the websocket closed"
        self.sent.append(json.loads(raw))

    async def close(self):
        self.closed = True

    def of(self, kind):
        return [m for m in self.sent if m["type"] == kind]


def _daemon(home, monkeypatch, script=None, real_sandbox=False, url="http://127.0.0.1:9920"):
    """A daemon that runs its stages in a worker process: ``script`` instead of the worker
    module when given, under the real sandbox-exec when ``real_sandbox``."""
    monkeypatch.delenv("SLASHCOMPUTE_SANDBOX", raising=False)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")])))

    def wrap(cmd, job_dir, agent_dir):
        if script is not None:
            cmd = [sys.executable, "-c", script, *cmd[3:]]
        if real_sandbox:
            cmd = sandbox.wrap_command(cmd, job_dir, agent_dir)
        daemon.launched.append(cmd)
        return cmd

    monkeypatch.setattr("slashcompute.agent.daemon.wrap_command", wrap)
    opt = AgentOptions(url=url, home=home, localhost=True, sandbox=True, gpu_percent=100)
    daemon = Daemon(opt)
    daemon._ws = FakeWS()
    daemon.launched = []                               # the command line of each worker
    return daemon, daemon._ws


async def _until(cond, timeout, what="condition"):
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        assert asyncio.get_running_loop().time() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.05)


async def _started(daemon, ws, asg, timeout=120):
    """Start ``asg`` and wait until its worker reported StageReady. Returns the process."""
    await daemon._handle(asg)
    proc = daemon._proc
    assert proc is not None
    await _until(lambda: ws.of("stage_ready") or proc.returncode is not None, timeout, "stage_ready")
    assert ws.of("stage_ready"), "the worker exited before it was ready"
    return proc


async def _reap(proc):
    if proc is not None and proc.returncode is None:
        proc.kill()
        await proc.wait()


def _outside_temp_home():
    """An agent home outside every temp directory the sandbox profile lets a worker write,
    so the profile's JOBDIR / AGENTDIR rules are what allows the worker's files."""
    if shutil.which("sandbox-exec") is None:
        pytest.skip("sandbox-exec not available")
    here = Path(__file__).resolve().parent
    if any(here.is_relative_to(d) for d in ("/tmp", "/private/tmp", "/private/var/folders")):
        pytest.skip("this checkout is inside a directory the profile lets the worker write")
    return Path(tempfile.mkdtemp(prefix=".sbx-home-", dir=here))


# ------------------------------------------------------------ drain


async def test_coordinator_drain_reaches_a_sandboxed_worker(tmp_path, monkeypatch):
    """The daemon wrote the drain line to the worker's stdin, but the worker never read it."""
    daemon, ws = _daemon(tmp_path, monkeypatch, _DRAINABLE_WORKER)
    proc = await _started(daemon, ws, _assignment())
    try:
        assert daemon.status == "running"
        await daemon._handle(Drain(job_id="j", epoch=1))
        await _until(lambda: ws.of("stage_finished"), 30, "stage_finished")
        (finished,) = ws.of("stage_finished")
        assert finished["reason"] == "drained" and finished["last_step"] == 7
        await asyncio.wait_for(proc.wait(), 30)
        await _until(lambda: daemon._proc is None, 10, "the pump to finish")
        assert proc.returncode == 0 and daemon.status == "idle"
    finally:
        await _reap(proc)


async def test_stop_drains_a_sandboxed_worker_before_disconnecting(tmp_path, monkeypatch):
    """`stop` used to drop the websocket under a sandboxed worker, so the job rolled back."""
    daemon, ws = _daemon(tmp_path, monkeypatch, _DRAINABLE_WORKER)
    daemon.opt.cfg.grace_period_s = 60
    proc = await _started(daemon, ws, _assignment())
    try:
        await asyncio.wait_for(daemon.shutdown(), 45)
        (finished,) = ws.of("stage_finished")          # FakeWS refuses anything after close()
        assert finished["reason"] == "drained" and finished["last_step"] == 7
        assert ws.of("drain_notice") and ws.closed and proc.returncode == 0
    finally:
        await _reap(proc)


async def test_stop_gives_up_on_a_worker_that_never_drains(tmp_path, monkeypatch):
    """The wait for the worker's checkpoint is bounded by the grace period."""
    daemon, ws = _daemon(tmp_path, monkeypatch, _STUCK_WORKER)
    daemon.opt.cfg.grace_period_s = 0.5
    proc = await _started(daemon, ws, _assignment())
    try:
        await asyncio.wait_for(daemon.shutdown(), 30)
        assert ws.closed and not ws.of("stage_finished")
        assert proc.returncode is not None and daemon._proc is None
    finally:
        await _reap(proc)


async def test_worker_ignores_ctrl_c_and_still_drains(tmp_path, monkeypatch):
    """Ctrl-C in the agent's terminal goes to the whole process group. The worker must leave
    it to the daemon, which drains it, instead of dying mid-step."""
    daemon, ws = _daemon(tmp_path, monkeypatch, _DRAINABLE_WORKER)
    daemon.opt.cfg.grace_period_s = 60
    proc = await _started(daemon, ws, _assignment())
    try:
        proc.send_signal(signal.SIGINT)
        await asyncio.sleep(0.5)
        assert proc.returncode is None and not ws.of("stage_finished")
        await asyncio.wait_for(daemon.shutdown(), 45)
        assert [m["reason"] for m in ws.of("stage_finished")] == ["drained"] and proc.returncode == 0
    finally:
        await _reap(proc)


async def test_a_finished_worker_does_not_take_the_next_one_with_it(tmp_path, monkeypatch):
    """The old worker's stdout pump used to clear the daemon's handle on whatever worker was
    current when it ended; a stop then no longer reached (or waited for) the new one."""
    daemon, ws = _daemon(tmp_path, monkeypatch, _DRAINABLE_WORKER)
    daemon.opt.cfg.grace_period_s = 60
    first = await _started(daemon, ws, _assignment())
    old_pump = daemon._pump
    second = None
    try:
        await daemon._handle(_assignment(epoch=2))     # cancels epoch 1, starts epoch 2
        second = daemon._proc
        await asyncio.wait_for(old_pump, 30)
        assert first.returncode is not None and daemon._proc is second
        await _until(lambda: len(ws.of("stage_ready")) == 2 or second.returncode is not None, 120)
        await asyncio.wait_for(daemon.shutdown(), 45)
        assert [(m["epoch"], m["reason"]) for m in ws.of("stage_finished")] == [(2, "drained")]
        assert second.returncode == 0
    finally:
        await _reap(first)
        await _reap(second)


async def test_stop_drains_a_worker_under_the_real_sandbox(tmp_path, monkeypatch):
    home = _outside_temp_home()
    daemon, ws = _daemon(home, monkeypatch, _DRAINABLE_WORKER, real_sandbox=True)
    daemon.opt.cfg.grace_period_s = 120
    proc = None
    try:
        proc = await _started(daemon, ws, _assignment(), timeout=180)
        assert Path(daemon.launched[0][0]).name == "sandbox-exec"
        await asyncio.wait_for(daemon.shutdown(), 150)
        assert [m["reason"] for m in ws.of("stage_finished")] == ["drained"]
        assert ws.closed and proc.returncode == 0
    finally:
        await _reap(proc)
        shutil.rmtree(home, ignore_errors=True)


# ------------------------------------------------------------ the unmodified worker, sandboxed


class _BlobStore:
    """A local stand-in for the coordinator's HTTP blob routes: serves the dataset, keeps
    what the worker and the daemon upload."""

    def __init__(self, dataset: Path):
        store = self
        self.uploads = {}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                data = dataset.read_bytes()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                store.uploads[self.path.split("?")[0]] = body
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def blobs(tiny_dataset):
    store = _BlobStore(tiny_dataset)
    yield store
    store.close()


@pytest.mark.integration
async def test_real_sandboxed_stage_checkpoints_before_the_agent_stops(tmp_path, monkeypatch, tiny_model, blobs):
    """End to end on the default path: a real stage training in a sandbox-exec worker is
    stopped; it must finish its step, upload a checkpoint at that step and report "drained"
    before the agent disconnects (it used to be cut off, losing work since the last one)."""
    home = _outside_temp_home()
    daemon, ws = _daemon(home, monkeypatch, real_sandbox=True, url=blobs.url)
    daemon.opt.gpu_percent = 10                        # slow steps: the stop lands mid-run
    daemon.opt.cfg.grace_period_s = 120
    asg = _assignment(tiny_model, steps=100000, dataset_url="/jobs/j/dataset", checkpoint_every=100000)
    proc = None
    try:
        proc = await _started(daemon, ws, asg, timeout=180)
        assert Path(daemon.launched[0][0]).name == "sandbox-exec"
        assert daemon.launched[0][-5:-1] == [sys.executable, "-m", "slashcompute.agent.worker", "--assignment"]
        await _until(lambda: ws.of("step_metrics") or proc.returncode is not None, 120, "a step")
        assert ws.of("step_metrics") and not blobs.uploads
        await asyncio.wait_for(daemon.shutdown(), 150)
        (finished,) = ws.of("stage_finished")
        assert finished["reason"] == "drained" and finished["last_step"] >= 1
        assert finished["last_step"] == ws.of("step_metrics")[-1]["step"]
        assert list(blobs.uploads) == [f"/jobs/j/checkpoints/{finished['last_step']}"]
        assert ws.closed and proc.returncode == 0
    finally:
        await _reap(proc)
        shutil.rmtree(home, ignore_errors=True)


# ------------------------------------------------------------ sampled steps (verify bundles)

# The real stdio CLI around a stage that holds one sampled step (3) until it is killed.
_BUNDLE_WORKER = """
import asyncio
from slashcompute.agent import worker
from slashcompute.common.protocol import StageReady


class Ring:
    def save_bundle(self, step, path):
        if step != 3:
            return False
        path.write_bytes(b"bundle-3")
        return True


async def run_stage(ctx, emit):
    asg = ctx.assignment
    ctx.session.compute = Ring()
    await emit(StageReady(job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx))
    await asyncio.Event().wait()

worker.run_stage = run_stage
worker._stdio_app()
"""

# A worker deep in a compute call: it reads nothing and says nothing.
_SILENT_WORKER = "import time; time.sleep(600)"


def _bundle_daemon(home, monkeypatch, script=_BUNDLE_WORKER, **kw):
    """A daemon whose worker is ``script``, with the coordinator's blob store replaced by
    the returned ``uploads`` dict."""
    daemon, ws = _daemon(home, monkeypatch, script, **kw)
    uploads = {}
    daemon.opt.http.put_bytes = lambda path, data, params=None: uploads.__setitem__(path, data)
    return daemon, ws, uploads


def _fetch(verify_id, step=3, job_id="j", epoch=1):
    from slashcompute.common.protocol import VerifyFetch

    return VerifyFetch(verify_id=verify_id, job_id=job_id, epoch=epoch, stage_idx=0, step=step)


def _bundle_replies(ws):
    return {m["verify_id"]: m for m in ws.of("verify_bundle_ready")}


async def test_sampled_step_is_fetched_from_a_sandboxed_worker(tmp_path, monkeypatch):
    """Only an in-process stage could hand over a sampled step: for the default, sandboxed
    worker the daemon answered "bundle no longer held" while the worker still held it, so
    sandboxed work was never replayed."""
    daemon, ws, uploads = _bundle_daemon(tmp_path, monkeypatch)
    proc = await _started(daemon, ws, _assignment())
    try:
        await daemon._handle(_fetch("v1"))
        await _until(lambda: "v1" in _bundle_replies(ws), 30, "the bundle reply")
        reply = _bundle_replies(ws)["v1"]
        assert reply["error"] is None and (reply["job_id"], reply["stage_idx"], reply["step"]) == ("j", 0, 3)
        assert Path(reply["path"]) == daemon.opt.paths.job_dir("j", 1) / "bundle_3.safetensors"
        assert uploads == {"/verify/v1/bundle": b"bundle-3"}

        await daemon._handle(_fetch("v2", step=9))             # a step the worker doesn't hold
        await daemon._handle(_fetch("v3", epoch=2))            # another epoch's ring
        await daemon._handle(_fetch("v4", job_id="other"))
        await _until(lambda: len(_bundle_replies(ws)) == 4, 30, "the refusals")
        replies = _bundle_replies(ws)
        assert {replies[v]["error"] for v in ("v2", "v3", "v4")} == {"bundle no longer held"}
        assert all(replies[v]["path"] is None for v in ("v2", "v3", "v4"))
        assert set(uploads) == {"/verify/v1/bundle"} and proc.returncode is None
    finally:
        await daemon._cancel_stage()
        await _reap(proc)


async def test_fetch_from_a_busy_worker_does_not_hold_up_the_message_loop(tmp_path, monkeypatch):
    """A worker deep in a compute call answers late (this one never does): the daemon must
    go on reading messages, and still answer the fetch once the worker is gone."""
    from slashcompute.agent import daemon as daemon_mod
    from slashcompute.common.protocol import CancelStage

    monkeypatch.setattr(daemon_mod, "BUNDLE_STOP_GRACE_S", 0.2)
    daemon, ws, uploads = _bundle_daemon(tmp_path, monkeypatch, _SILENT_WORKER)
    await daemon._handle(_assignment())
    proc = daemon._proc
    try:
        await asyncio.wait_for(daemon._handle(_fetch("v1")), 2)
        assert not _bundle_replies(ws)                         # waiting on the worker, off the loop
        await asyncio.wait_for(daemon._handle(CancelStage(job_id="j", epoch=1)), 10)
        assert proc.returncode is not None
        await _until(lambda: _bundle_replies(ws), 5, "the reply")   # not left to the answer timeout
        assert _bundle_replies(ws)["v1"]["error"] == "bundle no longer held" and not uploads
    finally:
        await _reap(proc)


async def test_fetch_is_answered_when_the_worker_stays_silent(tmp_path, monkeypatch):
    from slashcompute.agent import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "BUNDLE_ANSWER_TIMEOUT_S", 0.3)
    daemon, ws, uploads = _bundle_daemon(tmp_path, monkeypatch, _SILENT_WORKER)
    await daemon._handle(_assignment())
    proc = daemon._proc
    try:
        await daemon._handle(_fetch("v1"))
        await _until(lambda: _bundle_replies(ws), 10, "the reply")
        assert _bundle_replies(ws)["v1"]["error"] == "bundle no longer held"
        assert proc.returncode is None and not uploads         # the stage itself is left alone
    finally:
        await daemon._cancel_stage()
        await _reap(proc)


async def test_fetch_just_before_a_cancel_still_gets_the_bundle(tmp_path, monkeypatch):
    """A fetch and a cancel sent back to back: the worker must get to answer before it is
    stopped."""
    from slashcompute.common.protocol import CancelStage

    daemon, ws, uploads = _bundle_daemon(tmp_path, monkeypatch)
    proc = await _started(daemon, ws, _assignment())
    try:
        await daemon._handle(_fetch("v1"))
        await daemon._handle(CancelStage(job_id="j", epoch=1))
        await _until(lambda: _bundle_replies(ws), 30, "the bundle reply")
        assert _bundle_replies(ws)["v1"]["error"] is None
        assert uploads == {"/verify/v1/bundle": b"bundle-3"}
        assert proc.returncode is not None and daemon._proc is None
    finally:
        await _reap(proc)


async def test_failed_bundle_upload_is_reported_not_raised(tmp_path, monkeypatch):
    daemon, ws, _ = _bundle_daemon(tmp_path, monkeypatch)

    def refused(path, data, params=None):
        raise ConnectionError("coordinator unreachable")

    daemon.opt.http.put_bytes = refused
    proc = await _started(daemon, ws, _assignment())
    try:
        await daemon._handle(_fetch("v1"))
        await _until(lambda: _bundle_replies(ws), 30, "the reply")
        assert "coordinator unreachable" in _bundle_replies(ws)["v1"]["error"]
    finally:
        await daemon._cancel_stage()
        await _reap(proc)


async def test_sampled_step_is_fetched_through_the_real_sandbox(tmp_path, monkeypatch):
    """Under sandbox-exec the worker may only write where the profile allows; the bundle goes
    to its job directory, which the profile names (the agent home here is outside every temp
    directory the profile also allows)."""
    home = _outside_temp_home()
    daemon, ws, uploads = _bundle_daemon(home, monkeypatch, real_sandbox=True)
    proc = None
    try:
        proc = await _started(daemon, ws, _assignment(), timeout=180)
        assert Path(daemon.launched[0][0]).name == "sandbox-exec"
        await daemon._handle(_fetch("v1"))
        await _until(lambda: _bundle_replies(ws), 60, "the bundle reply")
        assert _bundle_replies(ws)["v1"]["error"] is None
        assert uploads == {"/verify/v1/bundle": b"bundle-3"}
        assert (daemon.opt.paths.job_dir("j", 1) / "bundle_3.safetensors").read_bytes() == b"bundle-3"
    finally:
        await daemon._cancel_stage()
        await _reap(proc)
        shutil.rmtree(home, ignore_errors=True)


def _worker_ctx(tmp_path, held=(3,)):
    """A sandboxed worker's side of the pipe: its context, and the steps it was asked to save."""
    from slashcompute.agent.worker import StageSession, WorkerContext

    saved = []

    class Ring:
        def save_bundle(self, step, path):
            saved.append((step, path))
            if step not in held:
                return False
            path.write_bytes(b"x")
            return True

    session = StageSession()
    session.compute = Ring()
    ctx = WorkerContext(assignment=_assignment(job_id="j", epoch=4), http=None, job_dir=tmp_path / "job",
                        data_bind="127.0.0.1", data_port=0, gpu_percent=100, node_id="n",
                        session=session)
    return ctx, saved


def test_worker_exports_only_its_own_stage_and_only_into_its_job_dir(tmp_path):
    from slashcompute.agent.worker import _export_bundle

    ctx, saved = _worker_ctx(tmp_path)
    ask = {"type": "bundle", "job_id": "j", "epoch": 4, "step": 3}
    assert _export_bundle(ctx, ask) == {**ask, "ok": True}
    assert saved == [(3, tmp_path / "job" / "bundle_3.safetensors")]
    assert _export_bundle(ctx, {**ask, "step": 5}) == {**ask, "step": 5, "ok": False}   # not held

    del saved[:]
    stray = tmp_path / "elsewhere.safetensors"
    for bad in ({**ask, "job_id": "other"}, {**ask, "epoch": 3}, {**ask, "epoch": "4"},
                {**ask, "epoch": 4.0}, {**ask, "step": "3"}, {**ask, "step": True},
                {**ask, "step": 3.0}, {**ask, "step": None}, {"type": "bundle"},
                {**ask, "step": "../../elsewhere"}, {**ask, "path": str(stray), "step": [3]}):
        reply = _export_bundle(ctx, bad)
        assert reply["ok"] is False and reply["type"] == "bundle", bad
    assert not saved and not stray.exists()
    assert sorted(p.name for p in (tmp_path / "job").iterdir()) == ["bundle_3.safetensors"]


def test_worker_answers_a_failed_export_instead_of_dying(tmp_path):
    from slashcompute.agent.worker import _export_bundle

    ctx, _ = _worker_ctx(tmp_path)
    ctx.session.compute.save_bundle = lambda step, path: 1 / 0
    assert _export_bundle(ctx, {"type": "bundle", "job_id": "j", "epoch": 4, "step": 3})["ok"] is False
    ctx.session.compute = None                                  # the stage has not built its ring yet
    assert _export_bundle(ctx, {"type": "bundle", "job_id": "j", "epoch": 4, "step": 3})["ok"] is False


async def test_only_the_answer_to_a_pending_request_counts():
    """The worker's stdout is the less trusted end: an answer for another job, epoch or step,
    or one nobody asked for, resolves nothing (and is not forwarded to the coordinator)."""
    from slashcompute.agent.daemon import _bundle_answer

    asg = _assignment(job_id="j", epoch=4)
    waiter = asyncio.get_running_loop().create_future()
    asked = {3: waiter}
    ok = {"type": "bundle", "job_id": "j", "epoch": 4, "step": 3, "ok": True}
    for other in ({**ok, "job_id": "x"}, {**ok, "epoch": 5}, {**ok, "epoch": True}, {**ok, "step": 4},
                  {**ok, "step": "3"}, {**ok, "step": 3.0}, {"type": "bundle"}):
        assert _bundle_answer(asg, asked, json.dumps(other)) is True     # recognised, and dropped
        assert not waiter.done()
    for not_ours in ("not json", "[1]", json.dumps({"type": "stage_ready"}), json.dumps({**ok, "type": "x"})):
        assert _bundle_answer(asg, asked, not_ours) is False
    assert _bundle_answer(asg, asked, json.dumps({**ok, "ok": "yes"})) is True
    assert waiter.result() is False                                      # only a literal true is a yes
    waiter = asked[3] = asyncio.get_running_loop().create_future()
    assert _bundle_answer(asg, asked, json.dumps(ok)) is True and waiter.result() is True
    assert _bundle_answer(asg, asked, json.dumps(ok)) is True            # a repeat: nothing left to wake


@pytest.mark.integration
async def test_real_sandboxed_stage_hands_over_a_sampled_step(tmp_path, monkeypatch, tiny_model, blobs):
    """End to end on the default path: the unmodified worker trains under sandbox-exec, the
    daemon is asked for a step it ran, and the bundle the worker saved is uploaded: the very
    input and output whose digests the stage reported for that step."""
    import mlx.core as mx

    from slashcompute.transport import digest

    home = _outside_temp_home()
    daemon, ws = _daemon(home, monkeypatch, real_sandbox=True, url=blobs.url)
    daemon.opt.gpu_percent = 10                        # slow steps: step 1 is still in the ring
    asg = _assignment(tiny_model, steps=100000, dataset_url="/jobs/j/dataset", checkpoint_every=100000)
    proc = None
    try:
        proc = await _started(daemon, ws, asg, timeout=180)
        assert Path(daemon.launched[0][0]).name == "sandbox-exec"
        await _until(lambda: ws.of("step_metrics") or proc.returncode is not None, 120, "a step")
        metrics = ws.of("step_metrics")[-1]
        await daemon._handle(_fetch("v1", step=metrics["step"]))
        await _until(lambda: _bundle_replies(ws), 120, "the bundle reply")
        assert _bundle_replies(ws)["v1"]["error"] is None
        bundle = tmp_path / "uploaded.safetensors"
        bundle.write_bytes(blobs.uploads["/verify/v1/bundle"])
        tensors = mx.load(str(bundle))
        assert digest(tensors["x_in"]) == metrics["in_digest"]
        assert digest(tensors["out"]) == metrics["out_digest"]
        assert any(k.startswith("adapter/") for k in tensors)

        await daemon._handle(_fetch("v2", step=metrics["step"], epoch=2))   # not this stage's epoch
        await _until(lambda: "v2" in _bundle_replies(ws), 60, "the refusal")
        assert _bundle_replies(ws)["v2"]["error"] == "bundle no longer held"
        assert proc.returncode is None
    finally:
        await daemon._cancel_stage()
        await _reap(proc)
        shutil.rmtree(home, ignore_errors=True)
