"""agent/worker.py: one stage's setup, peer handshake and error reporting (run in-process
here, the way a --no-sandbox agent runs it)."""

import asyncio
import json
import socket
import threading
import time

import pytest

from slashcompute.agent import worker
from slashcompute.agent.worker import WorkerContext, _peer_links, run_stage
from slashcompute.common.protocol import PeerAddr, StageAssignment, StageFinished, StepMetrics
from slashcompute.jobs import LoraFinetuneSpec
from slashcompute.transport import LinkClosed, LinkServer, connect

HELLO = {"job_id": "j", "epoch": 1}


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _assignment(model="unused", stage_idx=0, next_port=None, epoch=1, **kw):
    spec = LoraFinetuneSpec(model=str(model), dataset_path="unused.jsonl", steps=2, batch_size=2,
                            microbatches=1, lora_rank=4, max_seq_len=32)
    nxt = PeerAddr(node_id="next", host="127.0.0.1", port=next_port) if next_port else None
    layers = {0: (0, 3), 1: (3, 6)}[stage_idx]
    return StageAssignment(job_id="j", epoch=epoch, stage_idx=stage_idx, num_stages=2,
                           layer_start=layers[0], layer_end=layers[1], num_layers=6, spec=spec,
                           next_peer=nxt, checkpoint_every=10, verify_ring_size=2, **kw)


def _ctx(tmp_path, asg, port=0, http=None):
    return WorkerContext(assignment=asg, http=http, job_dir=tmp_path / f"job-{asg.epoch}",
                         data_bind="127.0.0.1", data_port=port, gpu_percent=100, node_id="n")


def _peer_ctx(tmp_path, port, next_port=None):
    """A downstream stage (it listens on ``port``), dialing ``next_port`` when it is a middle one."""
    return _ctx(tmp_path, _assignment(stage_idx=1, next_port=next_port), port)


async def _bind_again(port):
    """What the agent's next stage does with its data port; EADDRINUSE while a listener leaks."""
    server = await LinkServer("127.0.0.1", port, {"job_id": "next", "epoch": 1}).start()
    await server.close()


async def _listening(port):
    try:
        _, writer = await asyncio.open_connection("127.0.0.1", port)
    except OSError:
        return False
    writer.close()
    return True


def _collector():
    sent = []

    async def emit(msg):
        sent.append(msg)

    return sent, emit


# ------------------------------------------------------------ peer handshake cleanup


async def test_cancelled_peer_handshake_frees_the_data_port(tmp_path):
    """A stage cancelled while it waited for its neighbours kept its listener (run_stage only
    gets the links once the handshake returns), so on a --no-sandbox agent no later stage
    could bind the data port until the process restarted."""
    port = _free_port()
    for i in range(120):                       # a cancel landing anywhere from start() to accept()
        task = asyncio.create_task(_peer_links(_peer_ctx(tmp_path, port)))
        for _ in range(i % 12):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _bind_again(port)


async def test_cancelled_peer_handshake_closes_the_link_it_already_dialed(tmp_path):
    port = _free_port()
    for i in range(24):
        downstream = await LinkServer("127.0.0.1", 0, HELLO).start()
        task = asyncio.create_task(_peer_links(_peer_ctx(tmp_path, port, downstream.port)))
        dialed = None
        try:
            dialed = await downstream.accept(timeout=10)
            if i % 2:                                      # odd: the dial has returned, the stage
                await asyncio.sleep(0.02)                  # waits for its upstream; even: mid-hello
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await _bind_again(port)
            with pytest.raises(LinkClosed):                # not left holding a dead connection
                await dialed.recv(5)
        finally:
            task.cancel()
            if dialed is not None:                         # or downstream.close() waits for it
                await dialed.close()
            await downstream.close()


async def test_refused_dial_frees_the_data_port(tmp_path, monkeypatch):
    """The next stage never comes up: the listener this stage already started must go."""
    monkeypatch.setattr(worker, "PEER_DIAL_TIMEOUT_S", 0.05)
    port = _free_port()
    for _ in range(6):
        with pytest.raises(OSError):
            await _peer_links(_peer_ctx(tmp_path, port, _free_port()))
        await _bind_again(port)


async def test_rejected_dial_frees_the_data_port(tmp_path):
    """The next stage is up but on another epoch, and rejects the hello."""
    port = _free_port()
    downstream = await LinkServer("127.0.0.1", 0, {"job_id": "j", "epoch": 99}).start()
    try:
        for _ in range(6):
            with pytest.raises(LinkClosed):
                await _peer_links(_peer_ctx(tmp_path, port, downstream.port))
            await _bind_again(port)
    finally:
        await downstream.close()


async def test_upstream_that_never_connects_frees_the_data_port(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "PEER_ACCEPT_TIMEOUT_S", 0.05)
    port = _free_port()
    downstream = await LinkServer("127.0.0.1", 0, HELLO).start()
    try:
        for _ in range(3):
            with pytest.raises(asyncio.TimeoutError):
                await _peer_links(_peer_ctx(tmp_path, port, downstream.port))
            await _bind_again(port)
            dialed = await downstream.accept(timeout=5)    # the link it had dialed is closed too
            with pytest.raises(LinkClosed):
                await dialed.recv(5)
            await dialed.close()
            await downstream.close()
            downstream = await LinkServer("127.0.0.1", downstream.port, HELLO).start()
    finally:
        await downstream.close()


async def test_closing_a_listener_drops_links_nobody_claimed():
    """An upstream that connected after the stage gave up on it, and one still in its hello:
    closing the listener must not wait for them (asyncio waits for every connection)."""
    server = await LinkServer("127.0.0.1", 0, HELLO).start()
    accepted = await connect("127.0.0.1", server.port, HELLO)          # never accept()ed
    _, silent = await asyncio.open_connection("127.0.0.1", server.port)  # sends no hello
    await asyncio.sleep(0.05)
    await asyncio.wait_for(server.close(), 5)
    with pytest.raises(LinkClosed):
        await accepted.recv(5)
    await accepted.close()
    silent.close()


async def test_cancelled_dial_drops_the_connection():
    """A dial cancelled between the TCP connect and the hello reply owns a connection that no
    caller ever sees; the peer must not be left holding it."""
    held = []

    async def on_conn(reader, writer):                 # accepts, never answers the hello
        held.append((reader, writer))

    silent = await asyncio.start_server(on_conn, "127.0.0.1", 0)
    port = silent.sockets[0].getsockname()[1]
    try:
        task = asyncio.create_task(connect("127.0.0.1", port, HELLO))
        for _ in range(100):
            if held:
                break
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        reader, writer = held[0]
        await asyncio.wait_for(reader.read(), 5)       # EOF: the dialer let go of it
        writer.close()
    finally:
        silent.close()


# ------------------------------------------------------------ run_stage setup failures


async def test_stage_cancelled_during_the_handshake_lets_the_next_stage_listen(tmp_path, tiny_model):
    """The same through run_stage, twice on one port: what an in-process agent does when the
    coordinator cancels an epoch that is still connecting and assigns the next one."""
    port = _free_port()
    for epoch in (1, 2):
        ctx = _ctx(tmp_path, _assignment(tiny_model, stage_idx=1, epoch=epoch), port)
        sent, emit = _collector()
        task = asyncio.create_task(run_stage(ctx, emit))
        for _ in range(600):                   # loading the tiny model comes first
            if task.done() or await _listening(port):
                break
            await asyncio.sleep(0.05)
        assert not task.done(), sent           # epoch 2 failed here with "address already in use"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert [type(m) for m in sent] == [StageFinished] and sent[0].reason == "cancelled"
    await _bind_again(port)


async def test_failed_peer_setup_is_reported_and_frees_the_data_port(tmp_path, tiny_model, monkeypatch):
    """A dial that fails used to leave the listener bound and send nothing: the coordinator
    waited for a stage that never started, and the agent could not listen again."""
    monkeypatch.setattr(worker, "PEER_DIAL_TIMEOUT_S", 0.05)
    port = _free_port()
    for epoch in (1, 2):
        asg = _assignment(tiny_model, stage_idx=1, epoch=epoch, next_port=_free_port(), resume_step=0)
        sent, emit = _collector()
        with pytest.raises(OSError):
            await asyncio.wait_for(run_stage(_ctx(tmp_path, asg, port), emit), 60)
        assert [type(m) for m in sent] == [StageFinished]
        assert sent[0].reason == "error" and sent[0].epoch == epoch and sent[0].detail
        await _bind_again(port)


async def test_setup_error_reports_a_failed_stage(tmp_path, tiny_model):
    """Setup ran before run_stage's try block, so a failed download or a bad dataset raised
    without a StageFinished; the daemon kept the stage's session and the coordinator waited."""

    class Http:
        def get_file(self, url, dest):
            dest.write_text('{"nope": 1}\n')
            return dest

    asg = _assignment(tiny_model, dataset_url="/jobs/j/dataset")
    sent, emit = _collector()
    with pytest.raises(ValueError):
        await run_stage(_ctx(tmp_path, asg, http=Http()), emit)
    assert [type(m) for m in sent] == [StageFinished]
    assert sent[0].reason == "error" and "unrecognised dataset row keys" in sent[0].detail


def _in_process_daemon(tmp_path, monkeypatch, send_fails=False):
    """An agent that runs stages in-process and cannot download its dataset; ``sent`` is what
    it told the coordinator."""
    from slashcompute.agent.daemon import AgentOptions, Daemon

    monkeypatch.delenv("SLASHCOMPUTE_SANDBOX", raising=False)
    opt = AgentOptions(url="http://127.0.0.1:9641", home=tmp_path, localhost=True, sandbox=False,
                       data_port=_free_port())
    sent = []

    class FakeWS:
        async def send(self, raw):
            if send_fails:
                raise ConnectionError("websocket gone")
            sent.append(json.loads(raw))

    def refused(url, dest):
        raise ConnectionError("coordinator unreachable")

    opt.http.get_file = refused
    daemon = Daemon(opt)
    daemon._ws = FakeWS()
    return daemon, sent


async def test_in_process_daemon_frees_its_session_when_setup_fails(tmp_path, monkeypatch):
    """The daemon only let go of an in-process stage when it saw its StageFinished, so a
    stage whose setup failed blocked the agent as "loading" until it was restarted."""
    daemon, sent = _in_process_daemon(tmp_path, monkeypatch)
    await daemon._handle(_assignment(dataset_url="/jobs/j/dataset"))
    session = daemon._session
    assert session is not None and daemon.status == "loading"
    await asyncio.wait_for(session.task, 30)
    assert daemon._session is None and daemon.status == "idle" and daemon.job_id is None
    (finished,) = [m for m in sent if m["type"] == "stage_finished"]
    assert finished["reason"] == "error" and "coordinator unreachable" in finished["detail"]


async def test_in_process_daemon_frees_its_session_when_the_report_cannot_be_sent(tmp_path, monkeypatch):
    daemon, _ = _in_process_daemon(tmp_path, monkeypatch, send_fails=True)
    await daemon._handle(_assignment(dataset_url="/jobs/j/dataset"))
    await asyncio.wait_for(daemon._session.task, 30)
    assert daemon._session is None and daemon.status == "idle" and daemon.job_id is None


# ------------------------------------------------------------ setup off the event loop


class _Coordinator:
    """Stands in for the coordinator's blob store (``CoordHTTP``): serves one dataset and one
    resume checkpoint, keeps what is uploaded, and can hold a download up."""

    def __init__(self, dataset=None, checkpoint=None, delay=0.0, gate=None):
        self.dataset, self.checkpoint, self.delay, self.gate = dataset, checkpoint, delay, gate
        self.started = threading.Event()
        self.uploads = {}
        self.threads = set()                           # the threads its calls blocked

    def get_file(self, url, dest):
        self.threads.add(threading.get_ident())
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(60), "the test never let the download finish"
        time.sleep(self.delay)
        src = self.checkpoint if "checkpoint" in url else self.dataset
        dest.write_bytes(src.read_bytes())
        return dest

    def put_bytes(self, path, data, params=None):
        self.threads.add(threading.get_ident())
        time.sleep(self.delay)
        self.uploads[path] = data


async def _ticks(counter):
    while True:
        counter.append(time.monotonic())
        await asyncio.sleep(0.01)


async def test_blocking_setup_does_not_block_the_event_loop(tmp_path, tiny_model):
    """Setup (downloads, model load, dataset parsing) ran on the event loop, which in a
    --no-sandbox agent also sends the heartbeats: a slow start got a healthy agent evicted."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"nope": 1}\n')                    # ends the stage right after the download
    http = _Coordinator(dataset=bad, delay=0.6)
    ticks = []
    ticker = asyncio.create_task(_ticks(ticks))
    sent, emit = _collector()
    try:
        with pytest.raises(ValueError):
            await run_stage(_ctx(tmp_path, _assignment(tiny_model, dataset_url="/jobs/j/dataset"),
                                 http=http), emit)
    finally:
        ticker.cancel()
    assert len(ticks) >= 20, f"the loop ran {len(ticks)} times during a 0.6 s download"
    assert threading.get_ident() not in http.threads
    assert sent[0].reason == "error"


async def test_heartbeats_keep_flowing_while_a_stage_sets_up(tmp_path, tiny_model, monkeypatch):
    """The same seen from the coordinator: an in-process agent whose stage is still
    downloading must keep sending heartbeats (its timeout is 20 s, a model download minutes)."""
    daemon, sent = _in_process_daemon(tmp_path, monkeypatch)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"nope": 1}\n')
    daemon.opt.http.get_file = _Coordinator(dataset=bad, delay=1.0).get_file
    beats = asyncio.create_task(daemon._heartbeats(0.05))
    try:
        await daemon._handle(_assignment(tiny_model, dataset_url="/jobs/j/dataset"))
        await asyncio.wait_for(daemon._session.task, 60)
    finally:
        beats.cancel()
    types = [m["type"] for m in sent]
    loading = [m for m in sent[:types.index("stage_finished")]
               if m["type"] == "heartbeat" and m["status"] == "loading"]
    assert len(loading) >= 5, f"{len(loading)} heartbeats during a 1 s download"


async def test_stage_cancelled_during_setup_ends_at_once_and_stops_loading(tmp_path, tiny_model, tiny_dataset,
                                                                         monkeypatch):
    """A cancel must not wait for the download the setup thread is in, and the abandoned
    setup must not go on to load the model (the next stage needs that memory)."""
    built = []
    monkeypatch.setattr(worker, "build_compute", lambda *a, **kw: built.append(a))
    gate = threading.Event()
    http = _Coordinator(dataset=tiny_dataset, gate=gate)
    sent, emit = _collector()
    task = asyncio.create_task(run_stage(
        _ctx(tmp_path, _assignment(tiny_model, dataset_url="/jobs/j/dataset"), http=http), emit))
    try:
        assert await asyncio.to_thread(http.started.wait, 30)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)            # not held up by the blocked download
        assert [type(m) for m in sent] == [StageFinished] and sent[0].reason == "cancelled"
    finally:
        gate.set()
    # The next stage's setup runs after the abandoned one, on the same thread.
    sent, emit = _collector()
    with pytest.raises(FileNotFoundError):
        await asyncio.wait_for(run_stage(
            _ctx(tmp_path, _assignment(tiny_model, epoch=2, dataset_url="/jobs/j/dataset"),
                 http=_Coordinator(dataset=tmp_path / "missing.jsonl")), emit), 30)
    assert not built and sent[0].reason == "error"


async def test_stage_trains_and_resumes_after_setup_on_another_thread(tmp_path, tiny_model, tiny_dataset):
    """MLX only evaluates a lazy array on the thread that created it, so what setup builds on
    its thread (weights, adapters, a resumed optimizer state) must be usable on the event
    loop, where the steps run. Also: the checkpoint upload must not run on the loop."""
    from slashcompute.agent.worker import StageSession

    http = _Coordinator(dataset=tiny_dataset)
    asg = _assignment(tiny_model, dataset_url="/jobs/j/dataset").model_copy(update={
        "num_stages": 1, "layer_end": 6, "checkpoint_every": 2})
    asg.spec.batch_size, asg.spec.microbatches = 4, 2
    ctx = _ctx(tmp_path, asg, http=http)
    ctx.session = StageSession()
    sent, emit = _collector()
    result = await asyncio.wait_for(run_stage(ctx, emit), 120)
    assert (result.reason, result.last_step) == ("done", 2)
    assert [m.step for m in sent if isinstance(m, StepMetrics)] == [1, 2]
    assert list(http.uploads) == ["/jobs/j/checkpoints/2"]
    assert threading.get_ident() not in http.threads   # neither the download nor the upload
    assert ctx.session.save_bundle(2, tmp_path / "bundle.safetensors")

    # Epoch 2 resumes from that checkpoint for two more steps.
    ckpt = tmp_path / "resume.safetensors"
    ckpt.write_bytes(http.uploads["/jobs/j/checkpoints/2"])
    asg2 = asg.model_copy(update={"epoch": 2, "resume_step": 2, "checkpoint_url": "/jobs/j/checkpoint"})
    asg2.spec.steps = 4
    http2 = _Coordinator(dataset=tiny_dataset, checkpoint=ckpt)
    sent, emit = _collector()
    result = await asyncio.wait_for(run_stage(_ctx(tmp_path, asg2, http=http2), emit), 120)
    assert (result.reason, result.last_step) == ("done", 4)
    assert [m.step for m in sent if isinstance(m, StepMetrics)] == [3, 4]
    assert sent[-1].reason == "done"
