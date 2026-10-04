"""Optional macOS sandbox-exec wrapper around the worker subprocess."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def profile_path() -> Path:
    # Shipped inside the package so wheel installs (and compute.app) have it.
    return Path(__file__).resolve().with_name("worker.sb")


def sandbox_enabled(flag: bool | None, cfg_default: bool) -> bool:
    env = os.environ.get("SLASHCOMPUTE_SANDBOX")
    if env is not None:
        return env.lower() in ("1", "true", "yes", "on")
    if flag is not None:
        return flag
    return bool(cfg_default) and sys.platform == "darwin"


def unavailable_reason() -> str:
    """Why the sandbox cannot run here, else ""."""
    exe = shutil.which("sandbox-exec")
    profile = profile_path()
    if exe is None or not profile.is_file():
        return (
            f"sandbox requested but unavailable (sandbox-exec={exe}, profile={profile}); "
            "refusing to run the worker unsandboxed (pass --no-sandbox to opt out)"
        )
    return ""


def wrap_command(cmd: list[str], job_dir: Path, agent_dir: Path) -> list[str]:
    """Prefix ``cmd`` with sandbox-exec. Fails closed: never runs the worker unsandboxed."""
    why = unavailable_reason()
    if why:
        raise RuntimeError(why)
    exe = shutil.which("sandbox-exec")
    profile = profile_path()
    tmp = Path(os.environ.get("TMPDIR") or "/tmp")
    return [
        exe, "-f", str(profile),
        "-D", f"JOBDIR={job_dir}",
        "-D", f"AGENTDIR={agent_dir}",
        "-D", f"TMPDIR={tmp}",
        "--", *cmd,
    ]
