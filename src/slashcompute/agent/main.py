"""``slashcompute-agent`` CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from slashcompute.agent.daemon import AgentOptions, request_stop, start_daemon
from slashcompute.agent.paths import AgentPaths
from slashcompute.common.config import EngineConfig
from slashcompute.common.logging import setup_logging

app = typer.Typer(no_args_is_help=True, help="/compute contributor agent")


def _home(home: Optional[Path]) -> Path:
    return Path(home) if home else EngineConfig.from_env().home


@app.command()
def start(
    url: Optional[str] = typer.Option(None, help="Coordinator URL (default: LAN mDNS)"),
    gpu_percent: int = typer.Option(50, min=1, max=100, help="Fraction of time spent computing"),
    data_port: Optional[int] = typer.Option(
        None, help="TCP port peers dial for activations/gradients (default 9700)"),
    max_memory_gb: Optional[float] = typer.Option(None, help="Cap contributed unified memory"),
    localhost: bool = typer.Option(False, help="Advertise 127.0.0.1 (same-machine cluster)"),
    sandbox: Optional[bool] = typer.Option(None, help="Run the worker under sandbox-exec"),
    no_sandbox: bool = typer.Option(False, help="Disable the macOS sandbox"),
    name: Optional[str] = typer.Option(None, help="Display name (default: hostname)"),
    home: Optional[Path] = typer.Option(None, help="State directory (default ~/.slashcompute)"),
    session_token: Optional[str] = typer.Option(None, help="Account session so this node earns credits"),
):
    """Join the pool and wait for stage assignments. Runs in the foreground."""
    setup_logging("agent")
    use_sandbox = False if no_sandbox else sandbox
    opt = AgentOptions(
        url=url, home=home, gpu_percent=gpu_percent,
        data_port=data_port or EngineConfig.from_env().data_port,
        max_memory_gb=max_memory_gb, localhost=localhost, sandbox=use_sandbox, name=name,
        session_token=session_token,
    )
    paths = opt.paths
    existing = paths.read_pid()
    if existing is not None:
        import os

        try:
            os.kill(existing, 0)
        except OSError:
            paths.clear_pid()
        else:
            typer.echo(f"agent already running (pid {existing})", err=True)
            raise typer.Exit(1)
    import asyncio

    asyncio.run(start_daemon(opt))


@app.command()
def stop(home: Optional[Path] = typer.Option(None)):
    """Ask a running agent to drain and exit."""
    paths = AgentPaths(_home(home))
    if request_stop(paths):
        typer.echo(f"sent SIGTERM to pid {paths.read_pid()}")
    else:
        typer.echo("no running agent", err=True)
        raise typer.Exit(1)


@app.command()
def status(home: Optional[Path] = typer.Option(None)):
    """Show the last-known agent status."""
    paths = AgentPaths(_home(home))
    st = paths.read_status()
    pid = paths.read_pid()
    if pid is not None:
        import os

        try:
            os.kill(pid, 0)
            st["running"] = True
            st["pid"] = pid
        except OSError:
            st["running"] = False
    else:
        st["running"] = False
    typer.echo(json.dumps(st, indent=2) if st else "{}")


if __name__ == "__main__":
    app()
