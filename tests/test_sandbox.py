"""The worker sandbox: its profile ships with the package, and a requested sandbox that is
unavailable refuses to run the worker rather than running it unsandboxed."""

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

from slashcompute.agent import sandbox
from slashcompute.common.protocol import LoraFinetuneSpec, StageAssignment

REPO = Path(__file__).resolve().parents[1]


def test_sandbox_profile_ships_inside_the_package():
    import slashcompute.agent as agent_pkg

    profile = sandbox.profile_path()
    assert profile.is_file()
    assert profile.parent == Path(agent_pkg.__file__).resolve().parent
    assert "(deny default)" in profile.read_text()


def test_wrap_command_fails_closed_without_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox.shutil, "which", lambda _: "/usr/bin/sandbox-exec")
    monkeypatch.setattr(sandbox, "profile_path", lambda: tmp_path / "missing.sb")
    with pytest.raises(RuntimeError, match="unsandboxed"):
        sandbox.wrap_command(["python", "-m", "worker"], tmp_path, tmp_path)


def test_wrap_command_fails_closed_without_sandbox_exec(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="unsandboxed"):
        sandbox.wrap_command(["python", "-m", "worker"], tmp_path, tmp_path)


def test_wrap_command_prefixes_sandbox_exec(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox.shutil, "which", lambda _: "/usr/bin/sandbox-exec")
    cmd = sandbox.wrap_command(["python", "-m", "worker"], tmp_path / "job", tmp_path)
    assert cmd[:3] == ["/usr/bin/sandbox-exec", "-f", str(sandbox.profile_path())]
    assert cmd[-4:] == ["--", "python", "-m", "worker"]


@pytest.mark.integration
def test_built_wheel_contains_the_sandbox_profile(tmp_path):
    """A wheel install has no repo checkout around it: the profile must be in the wheel."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not available to build the wheel")
    out = tmp_path / "dist"
    built = subprocess.run([uv, "build", "--wheel", "--out-dir", str(out), str(REPO)],
                           capture_output=True, text=True, timeout=300)
    assert built.returncode == 0, built.stderr
    (wheel,) = out.glob("slashcompute-*.whl")
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        assert "slashcompute/agent/sandbox.py" in names
        assert "slashcompute/agent/worker.sb" in names
        assert "(deny default)" in zf.read("slashcompute/agent/worker.sb").decode()


def test_packaged_profile_is_enforced_by_the_real_sandbox(tmp_path):
    """The real /usr/bin/sandbox-exec with the packaged profile: a write inside the job
    directory succeeds, one outside every allowed directory is denied."""
    if shutil.which("sandbox-exec") is None:
        pytest.skip("sandbox-exec not available")
    # The profile allows the temp directories, so "outside" has to be somewhere else.
    here = Path(__file__).resolve().parent
    if any(here.is_relative_to(d) for d in ("/tmp", "/private/tmp", "/private/var/folders")):
        pytest.skip("this checkout is inside a directory the profile lets the worker write")
    outside = Path(tempfile.mkdtemp(prefix=".sbx-outside-", dir=here))
    job_dir, agent_dir = tmp_path / "job", tmp_path / "agent"
    job_dir.mkdir()
    agent_dir.mkdir()
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1], 'inside').write_text('ok')\n"
        "try:\n"
        "    Path(sys.argv[2], 'escaped').write_text('no')\n"
        "except PermissionError:\n"
        "    print('denied')\n"
    )
    try:
        cmd = sandbox.wrap_command([sys.executable, "-c", script, str(job_dir), str(outside)],
                                   job_dir, agent_dir)
        assert Path(cmd[0]).name == "sandbox-exec"
        ran = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout.strip() == "denied"
        assert (job_dir / "inside").read_text() == "ok"
        assert not (outside / "escaped").exists()
    finally:
        shutil.rmtree(outside, ignore_errors=True)


# ------------------------------------------------------------ the daemon, sandbox unavailable


def _assignment():
    return StageAssignment(
        job_id="j", epoch=1, stage_idx=0, num_stages=1, layer_start=0, layer_end=8,
        num_layers=8, spec=LoraFinetuneSpec(dataset_path="dataset.jsonl", steps=100),
        checkpoint_every=25, verify_ring_size=8, resume_step=25,
    )


def _sandboxed_daemon(tmp_path, monkeypatch):
    from slashcompute.agent.daemon import AgentOptions, Daemon

    monkeypatch.delenv("SLASHCOMPUTE_SANDBOX", raising=False)
    monkeypatch.delenv("SLASHCOMPUTE_HOME", raising=False)
    opt = AgentOptions(url="http://127.0.0.1:9640", home=tmp_path, localhost=True, sandbox=True)
    assert opt.sandbox and opt.paths.root == tmp_path / "agent"
    return Daemon(opt)


async def test_agent_refuses_to_start_when_the_sandbox_is_unavailable(tmp_path, monkeypatch):
    """Not after joining the pool: its work would be refused one stage at a time."""
    daemon = _sandboxed_daemon(tmp_path, monkeypatch)
    monkeypatch.setattr("slashcompute.agent.daemon.benchmark",
                        lambda *_: pytest.fail("must refuse before doing anything else"))
    monkeypatch.setattr(sandbox.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit, match="no-sandbox"):
        await asyncio.wait_for(daemon.run(), 5)
    assert daemon.opt.paths.read_pid() is None


async def test_missing_sandbox_fails_the_stage_not_the_daemon(tmp_path, monkeypatch):
    """A sandbox that vanishes later must not run the stage unsandboxed, nor take the agent
    down with an exception out of the message loop: the stage is reported as failed."""
    daemon = _sandboxed_daemon(tmp_path, monkeypatch)
    sent = []

    class FakeWS:
        async def send(self, raw):
            sent.append(json.loads(raw))

    daemon._ws = FakeWS()
    monkeypatch.setattr(sandbox, "profile_path", lambda: tmp_path / "missing.sb")
    spawned = []

    async def no_spawn(*cmd, **kw):
        spawned.append(cmd)
        raise AssertionError("the worker must not be started")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_spawn)
    await daemon._handle(_assignment())
    assert not spawned and daemon._proc is None
    (finished,) = [m for m in sent if m["type"] == "stage_finished"]
    assert finished["reason"] == "error" and finished["last_step"] == 25
    assert "unsandboxed" in finished["detail"]
    assert daemon.status == "idle" and daemon.job_id is None
