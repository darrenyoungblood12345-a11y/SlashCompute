"""Two real agent processes over TCP against a live coordinator."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

from slashcompute.pipeline.tiny import make_tiny_dataset, make_tiny_model

pytestmark = pytest.mark.integration


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait(cond, timeout=90.0, interval=0.2):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            if cond():
                return
        except Exception as e:
            last = e
        time.sleep(interval)
    raise AssertionError(f"condition not met ({last})")


def _popen(args, env, log_file):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = log_file.open("w")
    proc = subprocess.Popen(args, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
    proc._log_file = fh  # type: ignore[attr-defined]
    return proc


@pytest.fixture
def cluster(tmp_path):
    model = make_tiny_model(tmp_path / "model")
    data = make_tiny_dataset(tmp_path / "train.jsonl")
    port = _free_port()
    d0, d1 = _free_port(), _free_port()
    url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update({
        "SLASHCOMPUTE_VERIFY_RATE": "0",
        "SLASHCOMPUTE_SCHEDULER_TICK_S": "0.15",
        "SLASHCOMPUTE_CANARY_SIZE": "32",
        "SLASHCOMPUTE_STAGE_OVERHEAD_BYTES": "0",
        "SLASHCOMPUTE_SANDBOX": "0",
        "SLASHCOMPUTE_LOG": "INFO",
    })
    procs = []
    try:
        procs.append(_popen(
            [sys.executable, "-m", "slashcompute.coordinator.main", "serve",
             "--host", "127.0.0.1", "--port", str(port),
             "--home", str(tmp_path / "coord"), "--no-mdns"],
            env, tmp_path / "coord.log",
        ))
        _wait(lambda: httpx.get(f"{url}/health", timeout=1).json()["ok"])
        for i, dp in enumerate((d0, d1)):
            procs.append(_popen(
                [sys.executable, "-m", "slashcompute.agent.main", "start",
                 "--url", url, "--localhost", "--no-sandbox",
                 "--data-port", str(dp), "--gpu-percent", "100",
                 "--max-memory-gb", "4", "--home", str(tmp_path / f"agent{i}"),
                 "--name", f"it-{i}"],
                env, tmp_path / f"agent{i}.log",
            ))
        _wait(lambda: len(httpx.get(f"{url}/nodes", timeout=2).json()) == 2)
        _wait(lambda: all(n.get("canary_passed") for n in httpx.get(f"{url}/nodes").json()))
        yield url, model, data, procs, tmp_path
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()
            fh = getattr(p, "_log_file", None)
            if fh:
                fh.close()


def test_two_agents_complete_lora_job(cluster):
    url, model, data, _, tmp_path = cluster
    spec = {
        "kind": "lora_finetune",
        "model": str(model),
        "dataset_path": str(data),
        "steps": 3,
        "batch_size": 2,
        "microbatches": 1,
        "lora_rank": 4,
        "max_seq_len": 32,
        "min_stages": 2,
        "learning_rate": 1e-2,
        "seed": 7,
        "checkpoint_every": 3,
    }
    # Submitting by path is allowed because this caller is on the coordinator
    # machine (loopback); a remote one has to upload the dataset.
    r = httpx.post(f"{url}/jobs", json=spec, timeout=30)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    def done():
        j = httpx.get(f"{url}/jobs/{job_id}", timeout=5).json()
        return j["status"] in ("completed", "failed", "cancelled")

    try:
        _wait(done, timeout=120)
    except AssertionError:
        job = httpx.get(f"{url}/jobs/{job_id}").json()
        nodes = httpx.get(f"{url}/nodes").json()
        logs = []
        for name in ("coord.log", "agent0.log", "agent1.log"):
            p = tmp_path / name
            logs.append(f"===== {name} =====\n{p.read_text()[-4000:] if p.exists() else 'missing'}")
        raise AssertionError(f"job={job}\nnodes={nodes}\n" + "\n".join(logs))
    job = httpx.get(f"{url}/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    assert job["progress_step"] == 3
    assert job["last_loss"] is not None

    rows = httpx.get(f"{url}/jobs/{job_id}/usage").json()
    train = [u for u in rows if u["kind"] == "train"]
    by_step = {}
    for u in train:
        by_step.setdefault(u["step"], {})[u["stage_idx"]] = u
    assert set(by_step) >= {1, 2, 3}
    for step, stages in by_step.items():
        if 0 in stages and 1 in stages:
            assert stages[0]["out_digest"] == stages[1]["in_digest"], (step, stages)
