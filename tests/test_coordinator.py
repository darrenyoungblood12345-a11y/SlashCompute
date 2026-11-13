"""Coordinator behaviour against fake agents over the real WebSocket API."""

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from slashcompute.common import protocol as P
from slashcompute.common.canary import compare_stats, expected_stats, run_canary_mlx
from slashcompute.common.config import EngineConfig
from slashcompute.coordinator.app import create_app
from slashcompute.jobs import LoraFinetuneSpec
from slashcompute.pipeline.local import build_compute

GB = 1024**3
REMOTE = ("192.168.1.50", 50000)


class FakeAgent:
    def __init__(self, client: TestClient, node_id: str, mem: int = 8 * GB, port: int = 9000):
        self.node_id = node_id
        self.ws = client.websocket_connect("/ws/agent").__enter__()
        self.ws.send_text(P.dump(P.Register(
            node_id=node_id, name=node_id, data_host="127.0.0.1", data_port=port, gpu_percent=100,
            device=P.DeviceProfile(chip="test", memory_total_bytes=mem, memory_available_bytes=mem,
                                   working_set_bytes=mem, memory_contrib_bytes=mem,
                                   matmul_tflops=1.0, mem_bandwidth_gbps=100.0))))
        welcome = self.recv()
        assert isinstance(welcome, P.Welcome)
        # Issued per registration; proves on HTTP that a request is this node's.
        self.token = welcome.node_token

    def recv(self):
        return P.parse_coordinator_message(self.ws.receive_text())

    def send(self, msg):
        self.ws.send_text(P.dump(msg))

    def pass_canary(self):
        req = self.recv()
        assert isinstance(req, P.VerifyRequest) and req.kind == "canary"
        self.send(P.VerifyResult(verify_id=req.verify_id, kind="canary",
                                 stats=run_canary_mlx(req.seed, req.size)))

    def close(self):
        self.ws.__exit__(None, None, None)


def _usage():
    return P.UsageSample(flops=1e9, tokens=10, peak_mem_bytes=1, resident_mem_bytes=1,
                         mem_byte_seconds=1.0, wall_s=1.0, busy_s=0.5)


def _wait(cond, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met")


@pytest.fixture
def env(tmp_path, tiny_model, tiny_dataset):
    cfg = EngineConfig(home=tmp_path / "home", scheduler_tick_s=0.02, verify_rate=0.0,
                       canary_size=64, stage_overhead_bytes=0, heartbeat_timeout_s=60)
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client, app.state.core, tiny_model, tiny_dataset, tmp_path


def _spec(tiny_model, tiny_dataset, **kw):
    base = dict(model=str(tiny_model), dataset_path=str(tiny_dataset), steps=2,
                batch_size=2, microbatches=1, lora_rank=4, min_stages=2)
    return LoraFinetuneSpec(**(base | kw))


def _ckpt_bytes(spec, a: P.StageAssignment, step, tmp_path) -> bytes:
    comp = build_compute(spec, a.layer_start, a.layer_end, a.num_layers)
    path = tmp_path / f"s{a.stage_idx}_e{a.epoch}_{step}.safetensors"
    comp.save_checkpoint(path, step)
    return path.read_bytes()


def _post_ckpt(client, a: P.StageAssignment, step, data, token, **override):
    """Upload as an agent does. `override` lets a test lie about job/epoch/stage."""
    q = {"job_id": a.job_id, "epoch": a.epoch, "stage": a.stage_idx} | override
    return client.post(f"/jobs/{q['job_id']}/checkpoints/{step}?epoch={q['epoch']}&stage={q['stage']}",
                       content=data, headers={"X-Node-Token": token} if token else {})


def _upload_ckpt(client, spec, a: P.StageAssignment, step, tmp_path, agent):
    r = _post_ckpt(client, a, step, _ckpt_bytes(spec, a, step, tmp_path), agent.token)
    assert r.status_code == 200, r.text


def _running_job(client, spec, a, b):
    """Submit `spec` to two idle agents and bring the job to 'running'.
    Returns (job_id, {stage_idx: (assignment, agent)})."""
    r = client.post("/jobs", json=json.loads(spec.model_dump_json()))
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    by_stage = {}
    for x in (a, b):
        m = x.recv()
        assert isinstance(m, P.StageAssignment)
        by_stage[m.stage_idx] = (m, x)
    for idx, (m, ag) in by_stage.items():
        ag.send(P.StageReady(job_id=job_id, epoch=m.epoch, stage_idx=idx))
    _wait(lambda: client.get(f"/jobs/{job_id}").json()["status"] == "running")
    return job_id, by_stage


def test_full_job_lifecycle(env):
    client, core, tiny_model, tiny_dataset, tmp_path = env
    a, b = FakeAgent(client, "node-a"), FakeAgent(client, "node-b", port=9001)
    a.pass_canary(); b.pass_canary()
    _wait(lambda: all(n.canary_passed for n in core.registry.nodes.values()))

    spec = _spec(tiny_model, tiny_dataset)
    r = client.post("/jobs", json=json.loads(spec.model_dump_json()))
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    asg = {x.node_id: x.recv() for x in (a, b)}
    for msg in asg.values():
        assert isinstance(msg, P.StageAssignment) and msg.num_stages == 2 and msg.epoch == 1
    s0 = next(m for m in asg.values() if m.stage_idx == 0)
    s1 = next(m for m in asg.values() if m.stage_idx == 1)
    assert (s0.layer_start, s0.layer_end, s1.layer_end) == (0, s1.layer_start, 6)
    assert s0.next_peer.port == (9001 if s1 is asg["node-b"] else 9000)
    assert s0.dataset_url and s1.dataset_url is None and s0.resume_step == 0
    assert client.get(s0.dataset_url).status_code == 200

    by_stage = {0: (s0, a if asg["node-a"] is s0 else b), 1: (s1, a if asg["node-a"] is s1 else b)}
    for idx, (m, ag) in by_stage.items():
        ag.send(P.StageReady(job_id=job_id, epoch=1, stage_idx=idx))
    _wait(lambda: client.get(f"/jobs/{job_id}").json()["status"] == "running")

    for step in (1, 2):
        for idx, (m, ag) in by_stage.items():
            ag.send(P.StepMetrics(job_id=job_id, epoch=1, stage_idx=idx, step=step,
                                  loss=1.0 / step if idx == 1 else None,
                                  in_digest=f"in{idx}-{step}", out_digest=f"in{idx + 1}-{step}",
                                  usage=_usage()))
    for idx, (m, ag) in by_stage.items():
        _upload_ckpt(client, spec, m, 2, tmp_path, ag)
        ag.send(P.StageFinished(job_id=job_id, epoch=1, stage_idx=idx, reason="done", last_step=2))
    _wait(lambda: client.get(f"/jobs/{job_id}").json()["status"] == "completed")

    job = client.get(f"/jobs/{job_id}").json()
    assert job["last_checkpoint_step"] == 2 and job["last_loss"] == 0.5
    cfg = client.get(f"/jobs/{job_id}/adapter/adapter_config.json").json()
    assert cfg["num_layers"] == 6 and cfg["lora_parameters"]["rank"] == 4
    assert client.get(f"/jobs/{job_id}/adapter/adapters.safetensors").status_code == 200

    ledger = {(r["node_id"], r["kind"]): r for r in client.get("/ledger").json()}
    assert ledger[("node-a", "train")]["flops"] == 2e9 and ledger[("node-a", "train")]["disputed_flops"] == 0
    assert not [v for v in client.get("/verifications").json() if v["kind"] == "chain"]
    a.close(); b.close()


def test_failed_adapter_export_fails_the_job(env):
    """Both stages report done but the final checkpoint never arrived, so there
    is nothing to export: the job must not be advertised as completed."""
    client, core, tiny_model, tiny_dataset, tmp_path = env
    a, b = FakeAgent(client, "node-a"), FakeAgent(client, "node-b", port=9001)
    try:
        a.pass_canary(); b.pass_canary()
        _wait(lambda: all(n.canary_passed for n in core.registry.nodes.values()))
        job_id, by_stage = _running_job(client, _spec(tiny_model, tiny_dataset), a, b)
        for idx, (m, ag) in by_stage.items():
            ag.send(P.StageFinished(job_id=job_id, epoch=1, stage_idx=idx, reason="done", last_step=2))
        _wait(lambda: client.get(f"/jobs/{job_id}").json()["finished_at"] is not None)
        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "failed", job
        assert "adapter export failed" in job["error"]
        assert job["adapter_dir"] is None
        for name in ("adapters.safetensors", "adapter_config.json"):
            assert client.get(f"/jobs/{job_id}/adapter/{name}").status_code == 404
    finally:
        a.close(); b.close()


def test_adapter_export_leaves_nothing_behind_when_it_fails(tmp_path, tiny_model, tiny_dataset, monkeypatch):
    from slashcompute.coordinator import checkpoints as C

    spec = _spec(tiny_model, tiny_dataset, min_stages=1)
    store = C.CheckpointStore(tmp_path / "jobs")
    comp = build_compute(spec, 0, 6, 6)
    comp.save_checkpoint(tmp_path / "stage0.safetensors", 2)
    assert store.store_stage("j", 1, 2, 0, (tmp_path / "stage0.safetensors").read_bytes(), 1)
    adapter = store.job_dir("j") / "adapter"

    def boom(*a, **kw):
        raise OSError("disk full")

    # The weights are written first; failing on the config used to leave an
    # adapter directory that the download endpoint served as if complete.
    with monkeypatch.context() as m:
        m.setattr(C.json, "dumps", boom)
        with pytest.raises(OSError):
            store.export_adapter("j", 2, spec, 6)
    assert not adapter.exists()
    assert [p.name for p in store.job_dir("j").iterdir() if "adapter" in p.name] == []

    assert store.export_adapter("j", 2, spec, 6) == adapter
    assert sorted(p.name for p in adapter.iterdir()) == ["adapter_config.json", "adapters.safetensors"]
    # A re-export that fails keeps the previous complete adapter.
    with monkeypatch.context() as m:
        m.setattr(C.json, "dumps", boom)
        with pytest.raises(OSError):
            store.export_adapter("j", 2, spec, 6)
    assert sorted(p.name for p in adapter.iterdir()) == ["adapter_config.json", "adapters.safetensors"]


def _flush(core, agent):
    """A node's messages are handled in order: once this heartbeat has landed,
    so has everything the node sent before it."""
    agent.send(P.Heartbeat(node_id=agent.node_id, status="draining"))
    _wait(lambda: core.registry.get(agent.node_id).status == "draining")


@pytest.fixture
def victim_job(env):
    """A two-stage job on two honest nodes, plus a connected node that was
    never assigned to it (too small to be scheduled)."""
    client, core, tiny_model, tiny_dataset, _ = env
    agents = [FakeAgent(client, "node-a"), FakeAgent(client, "node-b", port=9001)]
    try:
        for x in agents:
            x.pass_canary()
        _wait(lambda: all(n.canary_passed for n in core.registry.nodes.values()))
        spec = _spec(tiny_model, tiny_dataset, steps=50)
        r = client.post("/jobs", json=json.loads(spec.model_dump_json()))
        assert r.status_code == 200, r.text
        job_id = r.json()["id"]
        by_stage = {}
        for x in agents:
            m = x.recv()
            assert isinstance(m, P.StageAssignment)
            by_stage[m.stage_idx] = (m, x)
        agents.append(FakeAgent(client, "intruder", mem=1 << 20, port=9002))
        yield client, core, job_id, by_stage, agents[-1]
    finally:
        for x in agents:
            try:
                x.close()
            except Exception:
                pass


def test_unassigned_node_cannot_ready_another_jobs_stage(victim_job):
    client, core, job_id, by_stage, intruder = victim_job
    for idx in by_stage:
        intruder.send(P.StageReady(job_id=job_id, epoch=1, stage_idx=idx))
    _flush(core, intruder)
    assert core.jobs[job_id].current.ready == set()
    assert client.get(f"/jobs/{job_id}").json()["status"] == "starting"

    # A stage's own node may not vouch for the other stage either.
    _, stage0_node = by_stage[0]
    stage0_node.send(P.StageReady(job_id=job_id, epoch=1, stage_idx=1))
    stage0_node.send(P.StageReady(job_id=job_id, epoch=1, stage_idx=0))
    _flush(core, stage0_node)
    assert core.jobs[job_id].current.ready == {0}
    assert client.get(f"/jobs/{job_id}").json()["status"] == "starting"


def test_unassigned_node_cannot_report_steps_for_another_job(victim_job):
    client, core, job_id, by_stage, intruder = victim_job
    intruder.send(P.StepMetrics(job_id=job_id, epoch=1, stage_idx=1, step=7, loss=0.01,
                                in_digest="x", out_digest="y", usage=_usage()))
    _flush(core, intruder)
    job = client.get(f"/jobs/{job_id}").json()
    assert job["progress_step"] == 0 and job["last_loss"] is None
    assert client.get(f"/jobs/{job_id}/usage").json() == []


@pytest.mark.parametrize("reason", ["error", "cancelled", "done", "drained"])
def test_unassigned_node_cannot_finish_another_jobs_stage(victim_job, reason):
    client, core, job_id, by_stage, intruder = victim_job
    for idx in by_stage:
        intruder.send(P.StageFinished(job_id=job_id, epoch=1, stage_idx=idx, reason=reason,
                                      last_step=0, detail="pwned"))
    _flush(core, intruder)
    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "starting" and job["recoveries"] == 0 and job["error"] is None
    assert core.jobs[job_id].current.finished == {}
    assert not core.jobs[job_id].current.closed
    for _, node in by_stage.values():
        assert core.registry.get(node.node_id).assignment is not None

    # The assigned node's own report still counts.
    m, node = by_stage[0]
    node.send(P.StageFinished(job_id=job_id, epoch=1, stage_idx=0, reason="error", last_step=0))
    _wait(lambda: client.get(f"/jobs/{job_id}").json()["recoveries"] == 1)


def _stored(core, job_id):
    """Every checkpoint file of a job, with its content."""
    root = core.checkpoints.job_dir(job_id) / "ckpt"
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*.safetensors"))}


def test_checkpoint_upload_requires_the_assigned_node(victim_job, tmp_path):
    """Checkpoint uploads were open to anyone who knew a job id: a stranger
    could replace the data a job recovers from."""
    client, core, job_id, by_stage, intruder = victim_job
    spec = core.jobs[job_id].spec
    (m0, n0), (m1, n1) = by_stage[0], by_stage[1]
    for m, n in by_stage.values():
        _upload_ckpt(client, spec, m, 5, tmp_path, n)
    _wait(lambda: client.get(f"/jobs/{job_id}").json()["last_checkpoint_step"] == 5)
    good = _stored(core, job_id)
    assert set(good) == {"epoch1/step000005/stage0.safetensors", "epoch1/step000005/stage1.safetensors",
                         "step_000005.safetensors"}

    remote = _remote(client)
    evil = _ckpt_bytes(spec.model_copy(update={"seed": 99, "lora_scale": 1e6}), m0, 5, tmp_path)
    attempts = [
        (remote, None, {}, 401),                      # no credential
        (remote, "not-a-token", {}, 401),             # made-up credential
        (remote, "", {}, 401),
        (remote, n0.node_id, {}, 401),                # node ids are public, not credentials
        (remote, intruder.token, {}, 403),            # a real node, but not on this job
        (remote, n1.token, {}, 403),                  # the other stage's node
        (client, intruder.token, {}, 403),            # being on the coordinator machine is not enough
        (remote, n0.token, {"stage": 1}, 403),        # right node, somebody else's stage
        (remote, n0.token, {"stage": 7}, 403),        # a stage that does not exist
    ]
    for who, token, lie, want in attempts:
        for step in (5, 6):  # overwrite an existing checkpoint, or plant the next one
            r = _post_ckpt(who, m0, step, evil, token, **lie)
            assert r.status_code == want, (token, lie, step, r.text)
    # Stale epochs are refused as before.
    assert _post_ckpt(remote, m0, 5, evil, n0.token, epoch=0).status_code in (403, 409)
    assert _post_ckpt(remote, m0, 5, evil, n0.token, job_id="nope").status_code in (403, 409)
    assert _stored(core, job_id) == good
    assert client.get(f"/jobs/{job_id}").json()["last_checkpoint_step"] == 5

    # The assigned node can still write (and rewrite) its own stage.
    assert _post_ckpt(remote, m0, 5, good["epoch1/step000005/stage0.safetensors"], n0.token).status_code == 200


def test_assigned_node_cannot_publish_an_invalid_checkpoint(victim_job, tmp_path, monkeypatch):
    """Even the right node must send a checkpoint of its own stage and step,
    and a rejected or interrupted upload must not disturb what is stored."""
    client, core, job_id, by_stage, _ = victim_job
    spec = core.jobs[job_id].spec
    (m0, n0), (m1, n1) = by_stage[0], by_stage[1]
    for m, n in by_stage.values():
        _upload_ckpt(client, spec, m, 5, tmp_path, n)
    good = _stored(core, job_id)
    assert "step_000005.safetensors" in good

    other_step = _ckpt_bytes(spec, m0, 4, tmp_path)
    other_stage = _ckpt_bytes(spec, m1, 5, tmp_path)   # stage 1's layers, sent as stage 0
    for data in (b"", b"not a safetensors file", other_step, other_stage):
        r = _post_ckpt(client, m0, 5, data, n0.token)
        assert r.status_code == 400, r.text
    for step in (0, -3, spec.steps + 1):
        r = _post_ckpt(client, m0, step, _ckpt_bytes(spec, m0, step, tmp_path), n0.token)
        assert r.status_code in (400, 422), (step, r.text)
    assert _stored(core, job_id) == good

    # A merge that dies halfway keeps the previous merged checkpoint whole.
    from slashcompute.coordinator import checkpoints as C

    def boom(parts, out):
        Path(out).write_bytes(b"half a file")
        raise OSError("disk full")

    monkeypatch.setattr(C, "merge_checkpoints", boom)
    with pytest.raises(OSError):
        _post_ckpt(client, m0, 5, good["epoch1/step000005/stage0.safetensors"], n0.token)
    assert _stored(core, job_id) == good
    assert client.get(f"/jobs/{job_id}").json()["last_checkpoint_step"] == 5


def test_checkpoint_upload_cannot_reach_another_job(env, tmp_path):
    """A node assigned to one job must not write into a different job."""
    client, core, tiny_model, tiny_dataset, _ = env
    agents = [FakeAgent(client, f"node-{x}", port=9000 + i) for i, x in enumerate("abcd")]
    try:
        for x in agents:
            x.pass_canary()
        _wait(lambda: all(n.canary_passed for n in core.registry.nodes.values()))
        spec = _spec(tiny_model, tiny_dataset, max_stages=2, steps=10)
        jobs = {}
        for _ in range(2):
            job_id = client.post("/jobs", json=json.loads(spec.model_dump_json())).json()["id"]
            _wait(lambda: core.jobs[job_id].current is not None)
            jobs[job_id] = {}
        for x in agents:
            m = x.recv()
            assert isinstance(m, P.StageAssignment)
            jobs[m.job_id][m.stage_idx] = (m, x)
        (job_a, a), (job_b, b) = jobs.items()
        assert len(a) == len(b) == 2
        for idx in (0, 1):
            m_b, node_b = b[idx]
            data = _ckpt_bytes(spec, m_b, 5, tmp_path)
            # job B's node, B's own epoch/stage numbers, but aimed at job A
            assert _post_ckpt(client, m_b, 5, data, node_b.token, job_id=job_a).status_code == 403
        assert _stored(core, job_a) == {}
        for idx in (0, 1):
            _upload_ckpt(client, spec, a[idx][0], 5, tmp_path, a[idx][1])
        assert "step_000005.safetensors" in _stored(core, job_a)
        assert _stored(core, job_b) == {}
    finally:
        for x in agents:
            try:
                x.close()
            except Exception:
                pass


def test_node_token_is_per_registration_and_never_listed(victim_job, tmp_path):
    client, core, job_id, by_stage, intruder = victim_job
    spec = core.jobs[job_id].spec
    m0, n0 = by_stage[0]
    tokens = {n.token for _, n in by_stage.values()} | {intruder.token}
    assert len(tokens) == 3 and all(t and len(t) >= 32 for t in tokens)
    listing = client.get("/nodes").text + client.get(f"/jobs/{job_id}").text + client.get("/health").text
    assert not any(t in listing for t in tokens)

    # Re-registering under a victim's (public) node id does not inherit its stage:
    # the old token dies with the old session and the new one holds no assignment.
    old = n0.token
    thief = FakeAgent(client, n0.node_id, port=9003)
    try:
        assert thief.token != old
        data = _ckpt_bytes(spec, m0, 5, tmp_path)
        assert _post_ckpt(client, m0, 5, data, old).status_code == 401
        assert _post_ckpt(client, m0, 5, data, thief.token).status_code in (403, 409)
        assert _stored(core, job_id) == {}
    finally:
        thief.close()


def test_chain_mismatch_disputes(env):
    client, core, tiny_model, tiny_dataset, _ = env
    a, b = FakeAgent(client, "node-a"), FakeAgent(client, "node-b", port=9001)
    a.pass_canary(); b.pass_canary()
    _wait(lambda: all(n.canary_passed for n in core.registry.nodes.values()))
    job_id = client.post("/jobs", json=json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())).json()["id"]
    asg = {x.node_id: x.recv() for x in (a, b)}
    ag = {m.stage_idx: (a if k == "node-a" else b) for k, m in asg.items()}
    ag[0].send(P.StepMetrics(job_id=job_id, epoch=1, stage_idx=0, step=1, in_digest="t",
                             out_digest="X", usage=_usage()))
    ag[1].send(P.StepMetrics(job_id=job_id, epoch=1, stage_idx=1, step=1, loss=1.0, in_digest="Y",
                             out_digest="z", usage=_usage()))
    _wait(lambda: len([v for v in client.get("/verifications").json() if v["kind"] == "chain"]) == 2)
    ledger = client.get("/ledger").json()
    assert all(r["disputed_flops"] == 1e9 for r in ledger if r["kind"] == "train")
    a.close(); b.close()


def test_failed_canary_excludes_node(env):
    client, core, *_ = env
    a = FakeAgent(client, "liar")
    req = a.recv()
    a.send(P.VerifyResult(verify_id=req.verify_id, kind="canary",
                          stats={k: 0.0 for k in ("sum", "abs_sum", "fro", "trace", "c00", "c_last", "row0_dot")}))
    _wait(lambda: core.registry.get("liar").canary_passed is False)
    assert core.registry.schedulable() == []
    a.close()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_canary_rejects_non_finite_stats(bad):
    exp = expected_stats(1, 64)
    assert compare_stats(dict(exp), exp, 0.5) == (True, 0.0)
    assert compare_stats({k: bad for k in exp}, exp, 0.5) == (False, float("inf"))
    assert compare_stats({**exp, "trace": bad}, exp, 0.5) == (False, float("inf"))


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_canary_result_fails_the_node(env, bad):
    """End to end: P.dump() would turn NaN into null, so send the raw JSON a
    hostile node would."""
    client, core, *_ = env
    a = FakeAgent(client, "nan-node")
    try:
        req = a.recv()
        stats = ", ".join(f'"{k}": {bad}' for k in
                          ("sum", "abs_sum", "fro", "trace", "c00", "c_last", "row0_dot"))
        a.ws.send_text('{"type": "verify_result", "verify_id": "%s", "kind": "canary", '
                       '"stats": {%s}}' % (req.verify_id, stats))
        _wait(lambda: core.registry.get("nan-node").canary_passed is not None)
        assert core.registry.get("nan-node").canary_passed is False
        assert core.registry.schedulable() == []
        # The failed row must still be listable: inf/NaN are not valid JSON.
        v = next(v for v in client.get("/verifications").json() if v["id"] == req.verify_id)
        assert v["status"] == "failed" and v["rel_error"] is None
    finally:
        a.close()


def test_node_loss_triggers_recovery_and_reschedule(env):
    client, core, tiny_model, tiny_dataset, tmp_path = env
    agents = [FakeAgent(client, f"node-{x}", port=9000 + i) for i, x in enumerate("abc")]
    try:
        for x in agents:
            x.pass_canary()
        _wait(lambda: all(n.canary_passed for n in core.registry.nodes.values()))
        spec = _spec(tiny_model, tiny_dataset, max_stages=2, steps=10)
        job_id = client.post("/jobs", json=json.loads(spec.model_dump_json())).json()["id"]
        _wait(lambda: sum(1 for n in core.registry.nodes.values() if n.assignment) == 2)
        assigned = [x for x in agents if core.registry.get(x.node_id) and core.registry.get(x.node_id).assignment]
        spare = next(x for x in agents if x not in assigned)
        msgs = {x.node_id: x.recv() for x in assigned}

        for x in assigned:
            _upload_ckpt(client, spec, msgs[x.node_id], 5, tmp_path, x)
        _wait(lambda: client.get(f"/jobs/{job_id}").json()["last_checkpoint_step"] == 5)
        lost, survivor = assigned
        lost.close()

        cancel = survivor.recv()
        assert isinstance(cancel, P.CancelStage) and cancel.epoch == 1
        # The torn-down epoch no longer accepts checkpoints, even from its own node.
        late = _ckpt_bytes(spec, msgs[survivor.node_id], 6, tmp_path)
        assert _post_ckpt(client, msgs[survivor.node_id], 6, late, survivor.token).status_code in (403, 409)
        survivor.send(P.StageFinished(job_id=job_id, epoch=1, stage_idx=msgs[survivor.node_id].stage_idx,
                                      reason="cancelled", last_step=5))
        new = {x.node_id: x.recv() for x in (survivor, spare)}
        for m in new.values():
            assert isinstance(m, P.StageAssignment) and m.epoch == 2 and m.resume_step == 5
            assert m.checkpoint_url == f"/jobs/{job_id}/checkpoints/5"
        assert client.get(f"/jobs/{job_id}/checkpoints/5").status_code == 200
        job = client.get(f"/jobs/{job_id}").json()
        assert job["recoveries"] == 1 and job["status"] == "starting"
    finally:
        for x in agents:
            try:
                x.close()
            except Exception:
                pass


def test_drain_sends_drain_to_stage0_and_resumes(env):
    client, core, tiny_model, tiny_dataset, tmp_path = env
    agents = [FakeAgent(client, f"node-{x}", port=9000 + i) for i, x in enumerate("abc")]
    try:
        for x in agents:
            x.pass_canary()
        _wait(lambda: all(n.canary_passed for n in core.registry.nodes.values()))
        spec = _spec(tiny_model, tiny_dataset, max_stages=2, steps=10)
        job_id = client.post("/jobs", json=json.loads(spec.model_dump_json())).json()["id"]
        _wait(lambda: sum(1 for n in core.registry.nodes.values() if n.assignment) == 2)
        assigned = [x for x in agents if core.registry.get(x.node_id) and core.registry.get(x.node_id).assignment]
        spare = next(x for x in agents if x not in assigned)
        msgs = {x.node_id: x.recv() for x in assigned}
        stage0 = next(x for x in assigned if msgs[x.node_id].stage_idx == 0)
        leaver = next(x for x in assigned if x is not stage0)

        leaver.send(P.DrainNotice(node_id=leaver.node_id))
        drain = stage0.recv()
        assert isinstance(drain, P.Drain) and drain.epoch == 1
        for x in assigned:
            m = msgs[x.node_id]
            _upload_ckpt(client, spec, m, 3, tmp_path, x)
            x.send(P.StageFinished(job_id=job_id, epoch=1, stage_idx=m.stage_idx, reason="drained", last_step=3))
        new = {x.node_id: x.recv() for x in (stage0, spare)}
        assert all(m.epoch == 2 and m.resume_step == 3 for m in new.values())
        assert client.get(f"/jobs/{job_id}").json()["recoveries"] == 0
    finally:
        for x in agents:
            try:
                x.close()
            except Exception:
                pass


def test_job_upload_accepts_dataset_file(env):
    client, core, tiny_model, tiny_dataset, _ = env
    r = client.post(
        "/jobs/upload",
        files={"dataset": ("train.jsonl", tiny_dataset.read_bytes(), "application/jsonl")},
        data={"model": str(tiny_model), "steps": 2, "min_stages": 1,
              "batch_size": 2, "microbatches": 1},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "lora_finetune"
    assert body["id"] in core.jobs


def _remote(client):
    """The same coordinator as seen from another machine on the LAN."""
    return TestClient(client.app, client=REMOTE)

