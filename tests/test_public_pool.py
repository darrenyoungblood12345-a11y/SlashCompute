"""Public coordinator: login, one stage, no WAN pipeline."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from slashcompute.agent.daemon import Daemon, AgentOptions, peered_assignment, public_transport
from slashcompute.common import protocol as P
from slashcompute.common.canary import run_canary_mlx
from slashcompute.common.config import EngineConfig
from slashcompute.coordinator.app import create_app
from slashcompute.jobs import LoraFinetuneSpec


@pytest.fixture
def env(tmp_path, tiny_model, tiny_dataset):
    cfg = EngineConfig(home=tmp_path / "home", scheduler_tick_s=0.05, verify_rate=0.0,
                       public_pool=True)
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client, app.state.core, tiny_model, tiny_dataset


def _device(mem=8 << 30):
    return P.DeviceProfile(
        chip="test", memory_total_bytes=mem, memory_available_bytes=mem,
        working_set_bytes=mem, memory_contrib_bytes=mem,
        matmul_tflops=1.0, mem_bandwidth_gbps=100.0,
    )


def _spec(tiny_model, tiny_dataset, **kw):
    base = dict(model=str(tiny_model), dataset_path=str(tiny_dataset), steps=2,
                batch_size=2, microbatches=1, lora_rank=4, min_stages=2, max_stages=4)
    return LoraFinetuneSpec(**(base | kw))


def _hdr(token):
    return {"Authorization": f"Bearer {token}"}


def _account(client, email="ada@lan.test"):
    token = client.post("/auth/register", json={
        "email": email, "password": "password1", "name": "Ada",
    }).json()["token"]
    client.post("/auth/accept-terms", headers=_hdr(token))
    return token


def test_public_anon_submit_rejected(env):
    client, core, tiny_model, tiny_dataset = env
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    r = client.post("/jobs", json=spec)
    assert r.status_code == 401
    assert core.jobs == {}


def test_public_upload_clamps_to_one_stage(env):
    client, core, tiny_model, tiny_dataset = env
    token = _account(client)
    core.credits.contribute(core.auth.session_user(token).id, 1e12, 0)
    r = client.post(
        "/jobs/upload",
        headers=_hdr(token),
        files={"dataset": ("train.jsonl", tiny_dataset.read_bytes(), "application/jsonl")},
        data={"model": str(tiny_model), "steps": 2, "min_stages": 4,
              "batch_size": 2, "microbatches": 1, "max_flops": 1e9},
    )
    assert r.status_code == 200, r.text
    job = core.jobs[r.json()["id"]]
    assert job.spec.min_stages == 1
    assert job.spec.max_stages == 1


def test_public_register_requires_session(env):
    client, *_ = env
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect("/ws/agent") as ws:
            ws.send_text(P.dump(P.Register(
                node_id="anon", name="mac", device=_device(),
                data_host="127.0.0.1", data_port=9700, gpu_percent=50,
            )))
            ws.receive_text()
    assert e.value.code == 4003


def test_public_register_requires_terms(env):
    client, *_ = env
    token = client.post("/auth/register", json={
        "email": "bob@lan.test", "password": "password1", "name": "Bob",
    }).json()["token"]
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect("/ws/agent") as ws:
            ws.send_text(P.dump(P.Register(
                node_id="bob", name="mac", device=_device(),
                data_host="127.0.0.1", data_port=9700, gpu_percent=50,
                session_token=token,
            )))
            ws.receive_text()
    assert e.value.code == 4003


def test_public_register_welcome_after_terms(env):
    client, core, *_ = env
    token = _account(client)
    with client.websocket_connect("/ws/agent") as ws:
        ws.send_text(P.dump(P.Register(
            node_id="ok", name="mac", device=_device(),
            data_host="127.0.0.1", data_port=9700, gpu_percent=50,
            session_token=token,
        )))
        welcome = P.parse_coordinator_message(ws.receive_text())
        assert isinstance(welcome, P.Welcome)
        assert core.registry.get("ok").user_id is not None


def _register(ws, node_id, token):
    ws.send_text(P.dump(P.Register(
        node_id=node_id, name="mac", device=_device(),
        data_host="127.0.0.1", data_port=9700, gpu_percent=50,
        session_token=token,
    )))
    return P.parse_coordinator_message(ws.receive_text())


def test_public_register_refuses_another_users_node_id(env):
    client, core, *_ = env
    victim = _account(client, "victim@lan.test")
    attacker = _account(client, "mallory@lan.test")
    victim_id = core.auth.session_user(victim).id
    with client.websocket_connect("/ws/agent") as honest:
        assert isinstance(_register(honest, "honest-node", victim), P.Welcome)
        honest_state = core.registry.get("honest-node")
        with pytest.raises(WebSocketDisconnect) as e:
            with client.websocket_connect("/ws/agent") as ws:
                _register(ws, "honest-node", attacker)
        assert e.value.code == 4003
        assert core.registry.get("honest-node") is honest_state
        assert honest_state.user_id == victim_id
        assert core.credits.owner_of("honest-node") == victim_id
    # The owner reconnecting with its own node id (agent restart) still works.
    with client.websocket_connect("/ws/agent") as ws:
        assert isinstance(_register(ws, "honest-node", victim), P.Welcome)
        assert core.registry.get("honest-node").user_id == victim_id
    with pytest.raises(PermissionError):
        core.credits.bind_node("honest-node", core.auth.session_user(attacker).id)
    assert core.credits.owner_of("honest-node") == victim_id


def test_public_account_switch_while_offline_rebinds_node(env):
    client, core, *_ = env
    first = _account(client, "first@lan.test")
    second = _account(client, "second@lan.test")
    first_id, second_id = core.auth.session_user(first).id, core.auth.session_user(second).id
    with client.websocket_connect("/ws/agent") as ws:
        assert isinstance(_register(ws, "same-mac", first), P.Welcome)
    core.credits.contribute(first_id, 100.0, 0, node_id="same-mac")
    earned = core.credits.summary(first_id)["balance"]
    # Signed out, signed in as someone else: the persistent node id comes back under the new account.
    with client.websocket_connect("/ws/agent") as ws:
        assert isinstance(_register(ws, "same-mac", second), P.Welcome)
        assert core.registry.get("same-mac").user_id == second_id
        assert core.credits.owner_of("same-mac") == second_id
    assert core.credits.summary(first_id)["balance"] == earned
    # An inference node id (never a training node) can't be claimed through the agent socket.
    core.credits.bind_node("n-inference", first_id)
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect("/ws/agent") as ws:
            _register(ws, "n-inference", second)
    assert e.value.code == 4003
    assert core.credits.owner_of("n-inference") == first_id


def test_inference_account_switch_rebinds_node(env):
    from slashcompute.coordinator.inference_accounting import CoreAccounting
    client, core, *_ = env
    first = _account(client, "first@lan.test")
    second = _account(client, "second@lan.test")
    acct = CoreAccounting(core)
    acct.bind_node("n-mac", first)
    acct.bind_node("n-mac", second)       # the node token already proved it's the same machine
    assert core.credits.owner_of("n-mac") == core.auth.session_user(second).id
    # ...but a live training node with that id still answers only to its owner.
    with client.websocket_connect("/ws/agent") as ws:
        assert isinstance(_register(ws, "live-mac", first), P.Welcome)
        acct.bind_node("live-mac", second)
        assert core.credits.owner_of("live-mac") == core.auth.session_user(first).id


def test_lan_anonymous_cannot_displace_live_owned_node(tmp_path):
    app = create_app(EngineConfig(home=tmp_path / "home", scheduler_tick_s=0.05, verify_rate=0.0))
    with TestClient(app) as client:
        core = app.state.core
        owner = _account(client)
        owner_id = core.auth.session_user(owner).id
        with client.websocket_connect("/ws/agent") as honest:
            assert isinstance(_register(honest, "owned-mac", owner), P.Welcome)
            state = core.registry.get("owned-mac")
            with pytest.raises(WebSocketDisconnect) as e:
                with client.websocket_connect("/ws/agent") as ws:
                    _register(ws, "owned-mac", None)
            assert e.value.code == 4003
            assert core.registry.get("owned-mac") is state
            assert state.user_id == owner_id
        # Offline, a LAN Mac whose session lapsed reconnects anonymously and still pays its owner.
        with client.websocket_connect("/ws/agent") as ws:
            assert isinstance(_register(ws, "owned-mac", None), P.Welcome)
            assert core.credits.owner_of("owned-mac") == owner_id
        for _ in range(2):
            with client.websocket_connect("/ws/agent") as ws:
                assert isinstance(_register(ws, "lan-mac", None), P.Welcome)


def test_public_transport_and_peered_assignment():
    spec = LoraFinetuneSpec(dataset_path="/tmp/d.jsonl", steps=2)
    peered = P.StageAssignment(
        job_id="j", epoch=1, stage_idx=0, num_stages=2, layer_start=0, layer_end=4,
        num_layers=8, spec=spec, next_peer=P.PeerAddr(node_id="b", host="h", port=1),
        checkpoint_every=25, verify_ring_size=8,
    )
    solo = P.StageAssignment(
        job_id="j", epoch=1, stage_idx=0, num_stages=1, layer_start=0, layer_end=8,
        num_layers=8, spec=spec, checkpoint_every=25, verify_ring_size=8,
    )
    assert public_transport("https://pool.example.com") is True
    assert public_transport("http://10.0.0.1:8765") is False
    assert public_transport("http://10.0.0.1:8765", public_pool=True) is True
    assert peered_assignment(peered) is True
    assert peered_assignment(solo) is False


def test_daemon_refuses_peered_assignment_on_https(tmp_path):
    opt = AgentOptions(url="https://pool.example.com", home=tmp_path, localhost=True)
    daemon = Daemon(opt)
    asg = P.StageAssignment(
        job_id="j", epoch=1, stage_idx=0, num_stages=2, layer_start=0, layer_end=4,
        num_layers=8, spec=LoraFinetuneSpec(dataset_path="/tmp/d.jsonl", steps=2),
        next_peer=P.PeerAddr(node_id="b", host="h", port=1),
        checkpoint_every=25, verify_ring_size=8,
    )
    asyncio.run(daemon._start_stage(asg))
    assert daemon._session is None
    assert daemon.status == "idle"


def _node(client, node_id, token, mem=8 << 30):
    ws = client.websocket_connect("/ws/agent").__enter__()
    ws.send_text(P.dump(P.Register(
        node_id=node_id, name=node_id, device=_device(mem),
        data_host="127.0.0.1", data_port=9700, gpu_percent=100, session_token=token,
    )))
    assert isinstance(P.parse_coordinator_message(ws.receive_text()), P.Welcome)
    return ws


def _pass_canary(ws):
    req = P.parse_coordinator_message(ws.receive_text())
    assert isinstance(req, P.VerifyRequest) and req.kind == "canary"
    ws.send_text(P.dump(P.VerifyResult(verify_id=req.verify_id, kind="canary",
                                       stats=run_canary_mlx(req.seed, req.size))))


def _flush(core, ws, node_id):
    """A node's messages are handled in order: once its heartbeat lands, so has
    everything it sent before."""
    ws.send_text(P.dump(P.Heartbeat(node_id=node_id, status="draining")))
    deadline = time.time() + 10
    while core.registry.get(node_id).status != "draining":
        assert time.time() < deadline, "heartbeat never handled"
        time.sleep(0.02)


def _usage(flops):
    return P.UsageSample(flops=flops, tokens=1, peak_mem_bytes=0, resident_mem_bytes=0,
                         mem_byte_seconds=0, wall_s=0, busy_s=0)


@pytest.fixture
def victim_job(env):
    """A victim's job assigned to an honest node, plus a signed-in attacker node."""
    client, core, tiny_model, tiny_dataset = env
    victim, attacker, host = (_account(client, f"{n}@lan.test") for n in ("victim", "attacker", "host"))
    core.credits.contribute(core.auth.session_user(victim).id, 1e15, 0)
    honest = _node(client, "honest", host)
    _pass_canary(honest)
    r = client.post(
        "/jobs/upload", headers=_hdr(victim),
        files={"dataset": ("train.jsonl", tiny_dataset.read_bytes(), "application/jsonl")},
        data={"model": str(tiny_model), "steps": 50, "batch_size": 2, "microbatches": 1,
              "max_flops": 1e15},
    )
    assert r.status_code == 200, r.text
    asg = P.parse_coordinator_message(honest.receive_text())
    assert isinstance(asg, P.StageAssignment)
    bad = _node(client, "attacker", attacker, mem=1 << 20)
    yield client, core, asg, honest, bad, core.auth.session_user(attacker).id
    honest.__exit__(None, None, None)
    bad.__exit__(None, None, None)


def test_unassigned_node_cannot_bill_another_users_job(victim_job):
    client, core, asg, honest, bad, attacker_id = victim_job
    before = core.credits.balance(attacker_id)
    bad.send_text(P.dump(P.StepMetrics(
        job_id=asg.job_id, epoch=asg.epoch, stage_idx=0, step=1, loss=0.1,
        in_digest="x", out_digest="y", usage=_usage(1e15))))
    _flush(core, bad, "attacker")
    job = client.get(f"/jobs/{asg.job_id}").json()
    assert core.credits.balance(attacker_id) == before
    assert job["spent_flops"] == 0 and job["status"] not in ("cancelled", "failed")
    assert not [r for r in core.ledger.job_records(asg.job_id) if r.node_id == "attacker"]

    honest.send_text(P.dump(P.StepMetrics(
        job_id=asg.job_id, epoch=asg.epoch, stage_idx=0, step=1, loss=0.1,
        in_digest="x", out_digest="y", usage=_usage(1e9))))
    _flush(core, honest, "honest")
    assert client.get(f"/jobs/{asg.job_id}").json()["spent_flops"] == 1e9


def test_nan_step_usage_keeps_the_node_connected(victim_job):
    """A NaN FLOP count used to hit a SQLite NOT NULL error and drop the stage holder."""
    client, core, asg, honest, *_ = victim_job
    nan_step = P.StepMetrics(
        job_id=asg.job_id, epoch=asg.epoch, stage_idx=0, step=1, loss=0.1,
        in_digest="x", out_digest="y", usage=_usage(0.0)).model_dump_json()
    honest.send_text(nan_step.replace('"flops":0.0', '"flops":NaN'))
    honest.send_text(P.dump(P.StepMetrics(
        job_id=asg.job_id, epoch=asg.epoch, stage_idx=0, step=1, loss=0.1,
        in_digest="x", out_digest="y", usage=_usage(1e9))))
    _flush(core, honest, "honest")
    assert client.get(f"/jobs/{asg.job_id}").json()["spent_flops"] == 1e9


@pytest.mark.parametrize("reason", ["error", "done"])
def test_unassigned_node_cannot_end_another_users_job(victim_job, reason):
    client, core, asg, honest, bad, _ = victim_job
    bad.send_text(P.dump(P.StageFinished(job_id=asg.job_id, epoch=asg.epoch, stage_idx=0,
                                         reason=reason, last_step=0, detail="pwned")))
    _flush(core, bad, "attacker")
    job = client.get(f"/jobs/{asg.job_id}").json()
    assert job["status"] in ("starting", "running") and job["recoveries"] == 0
    assert core.jobs[asg.job_id].current.finished == {}
    assert core.registry.get("honest").assignment is not None
