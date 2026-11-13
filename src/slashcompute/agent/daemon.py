"""Coordinator WebSocket session: register, heartbeat, run stages."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import socket
import sys
from pathlib import Path
from typing import Optional

import websockets
from pydantic import BaseModel, ValidationError

from slashcompute.agent.benchmark import benchmark
from slashcompute.agent.http import CoordHTTP
from slashcompute.agent.paths import AgentPaths, resolve_data_host
from slashcompute.agent.sandbox import sandbox_enabled, unavailable_reason, wrap_command
from slashcompute.agent.verify import run_canary, run_replay
from slashcompute.agent.worker import StageSession, WorkerContext, run_stage
from slashcompute.common.config import EngineConfig
from slashcompute.common.discovery import discover
from slashcompute.common.protocol import (
    CancelStage, Drain, DrainNotice, Heartbeat, Register, StageAssignment, StageFinished,
    StageReady, VerifyBundleReady, VerifyFetch, VerifyRequest, VerifyResult, Welcome, dump,
    parse_coordinator_message,
)

log = logging.getLogger(__name__)


def resolve_coordinator(url: Optional[str]) -> str:
    if url:
        return url.rstrip("/")
    found = discover()
    if not found:
        raise SystemExit("No coordinator found on the LAN. Pass --url or start one.")
    return found.rstrip("/")


def _ws_url(http_url: str) -> str:
    base = http_url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
    return base + "/ws/agent"


def _alive(pid: int) -> bool:
    try:
        os_kill = __import__("os").kill
        os_kill(pid, 0)
        return True
    except OSError:
        return False


class AgentOptions:
    def __init__(
        self,
        url: Optional[str] = None,
        home: Optional[Path] = None,
        gpu_percent: int = 50,
        data_port: int = 9700,
        max_memory_gb: Optional[float] = None,
        localhost: bool = False,
        sandbox: Optional[bool] = None,
        name: Optional[str] = None,
        session_token: Optional[str] = None,
    ) -> None:
        self.cfg = EngineConfig.from_env(home=home)
        if home is not None:
            self.cfg.home = Path(home)
        self.paths = AgentPaths(self.cfg.home)
        self.gpu_percent = max(1, min(100, int(gpu_percent)))
        self.data_port = int(data_port)
        self.max_memory_bytes = int(max_memory_gb * 1024**3) if max_memory_gb else None
        self.localhost = localhost
        self.sandbox = sandbox_enabled(sandbox, self.cfg.sandbox)
        self.name = name or socket.gethostname().split(".")[0]
        self.coordinator = resolve_coordinator(url)
        self.http = CoordHTTP(self.coordinator)
        self.node_id = self.paths.node_id()
        self.data_host = resolve_data_host(localhost)
        self.data_bind = "0.0.0.0"
        self.session_token = session_token or __import__("os").environ.get("SLASHCOMPUTE_SESSION")


class Daemon:
    def __init__(self, opt: AgentOptions) -> None:
        self.opt = opt
        self.status = "idle"
        self.job_id: Optional[str] = None
        self.epoch: Optional[int] = None
        self._ws = None
        self._send_lock = asyncio.Lock()
        self._session: Optional[StageSession] = None
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._pump: Optional[asyncio.Task] = None    # forwards the sandboxed worker's stdout
        self._stop = asyncio.Event()
        self._draining = False

    def _write_status(self) -> None:
        self.opt.paths.write_status(
            pid=__import__("os").getpid(), node_id=self.opt.node_id, name=self.opt.name,
            status=self.status, draining=self._draining, job_id=self.job_id, epoch=self.epoch,
            coordinator=self.opt.coordinator, data_addr=f"{self.opt.data_host}:{self.opt.data_port}",
            gpu_percent=self.opt.gpu_percent,
        )

    async def send(self, msg: BaseModel) -> None:
        if self._ws is None:
            return
        async with self._send_lock:
            await self._ws.send(dump(msg))

    async def run(self) -> None:
        opt = self.opt
        if opt.sandbox and (why := unavailable_reason()):
            raise SystemExit(why)        # don't join a pool whose work we would refuse to run
        opt.paths.write_pid()
        self._write_status()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:
                signal.signal(sig, lambda *_: asyncio.create_task(self.shutdown()))

        try:
            log.info("benchmarking device…")
            device = benchmark(opt.max_memory_bytes)
            log.info("chip=%s lending %.1f GB  %.1f TFLOPS  gpu %d%%",
                     device.chip, device.memory_contrib_bytes / 1e9, device.matmul_tflops,
                     opt.gpu_percent)

            ws_url = _ws_url(opt.coordinator)
            log.info("connecting to %s as %s (%s)", ws_url, opt.node_id[:8], opt.name)
            async with websockets.connect(ws_url, max_size=64 * 1024 * 1024, ping_interval=20) as ws:
                self._ws = ws
                await self.send(Register(
                    node_id=opt.node_id, name=opt.name, device=device,
                    data_host=opt.data_host, data_port=opt.data_port, gpu_percent=opt.gpu_percent,
                    session_token=opt.session_token,
                ))
                welcome = parse_coordinator_message(await ws.recv())
                if not isinstance(welcome, Welcome):
                    raise SystemExit(f"expected welcome, got {type(welcome).__name__}")
                opt.http.node_token = welcome.node_token
                interval = welcome.heartbeat_interval_s
                hb = asyncio.create_task(self._heartbeats(interval))
                try:
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            msg = parse_coordinator_message(raw)
                        except ValidationError as e:
                            log.warning("bad coordinator message: %s", e)
                            continue
                        await self._handle(msg)
                finally:
                    hb.cancel()
                    self._ws = None
        finally:
            opt.paths.clear_pid()
            self.status = "stopped"
            self._write_status()

    async def _heartbeats(self, interval: float) -> None:
        try:
            while not self._stop.is_set():
                await self.send(Heartbeat(
                    node_id=self.opt.node_id, status=self.status if not self._draining else "draining",
                    job_id=self.job_id, epoch=self.epoch,
                ))
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return

    async def _handle(self, msg) -> None:
        if isinstance(msg, StageAssignment):
            await self._start_stage(msg)
        elif isinstance(msg, Drain):
            log.info("drain requested for job %s", msg.job_id)
            await self._request_drain()
        elif isinstance(msg, CancelStage):
            log.info("cancel stage job %s epoch %d", msg.job_id, msg.epoch)
            await self._cancel_stage()
        elif isinstance(msg, VerifyFetch):
            await self._on_fetch(msg)
        elif isinstance(msg, VerifyRequest):
            await self._on_verify(msg)
        else:
            log.warning("unhandled coordinator message %s", type(msg).__name__)

    async def _start_stage(self, asg: StageAssignment) -> None:
        if self._session or self._proc:
            log.warning("assignment while a stage is running; cancelling the old one")
            await self._cancel_stage()
        self.status, self.job_id, self.epoch = "loading", asg.job_id, asg.epoch
        self._write_status()
        job_dir = self.opt.paths.job_dir(asg.job_id, asg.epoch)
        if self.opt.sandbox:
            try:
                await self._start_sandboxed(asg, job_dir)
            except (RuntimeError, OSError) as e:     # sandbox gone, or the worker would not spawn
                log.error("could not start the sandboxed worker: %s", e)
                self.status, self.job_id, self.epoch = "idle", None, None
                self._write_status()
                await self.send(StageFinished(
                    job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx,
                    reason="error", last_step=asg.resume_step, detail=str(e),
                ))
            return
        session = StageSession()
        session.assignment = asg
        ctx = WorkerContext(
            assignment=asg, http=self.opt.http, job_dir=job_dir,
            data_bind=self.opt.data_bind, data_port=self.opt.data_port,
            gpu_percent=self.opt.gpu_percent, node_id=self.opt.node_id, session=session,
        )

        async def emit(m) -> None:
            if isinstance(m, StageReady):
                self.status = "running"
                self._write_status()
            await self.send(m)
            if isinstance(m, StageFinished):
                self.status, self.job_id, self.epoch = "idle", None, None
                self._session = None
                self._write_status()

        async def _run() -> None:
            try:
                await run_stage(ctx, emit)
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("worker crashed")
            finally:
                # emit() lets go of the stage on its StageFinished; this is for a stage that
                # ended without one getting through (the send itself failed, say), which
                # would otherwise hold this agent as busy for good.
                if self._session is session:
                    self._session = None
                    self.status, self.job_id, self.epoch = "idle", None, None
                    self._write_status()

        session.task = asyncio.create_task(_run())
        self._session = session

    async def _start_sandboxed(self, asg: StageAssignment, job_dir: Path) -> None:
        spec_path = job_dir / "assignment.json"
        spec_path.write_text(json.dumps({
            "assignment": json.loads(asg.model_dump_json()),
            "coordinator_url": self.opt.coordinator,
            "job_dir": str(job_dir),
            "data_bind": self.opt.data_bind,
            "data_port": self.opt.data_port,
            "gpu_percent": self.opt.gpu_percent,
            "node_id": self.opt.node_id,
            "node_token": self.opt.http.node_token,
        }))
        spec_path.chmod(0o600)  # holds the node token
        cmd = wrap_command(
            [sys.executable, "-m", "slashcompute.agent.worker", "--assignment", str(spec_path)],
            job_dir, self.opt.paths.root,
        )
        log.info("starting sandboxed worker: %s", " ".join(cmd))
        self._proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.PIPE,
        )
        self._pump = asyncio.create_task(self._pump_worker_stdout())

    async def _pump_worker_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                raw = line.decode().strip()
                if not raw:
                    continue
                try:
                    from slashcompute.common.protocol import parse_agent_message

                    msg = parse_agent_message(raw)
                except ValidationError:
                    log.warning("worker stdout: %s", raw[:200])
                    continue
                if isinstance(msg, StageReady):
                    self.status = "running"
                    self._write_status()
                await self.send(msg)
                if isinstance(msg, StageFinished):
                    self.status, self.job_id, self.epoch = "idle", None, None
                    self._write_status()
        finally:
            if proc.returncode is None:
                await proc.wait()
            # Only for its own worker: by now a newer stage's worker may be the current one,
            # and stop / drain / cancel must still be able to reach that one.
            if self._proc is proc:
                self._proc = None
                if self.status == "running":
                    self.status, self.job_id, self.epoch = "idle", None, None
                    self._write_status()

    async def _request_drain(self) -> None:
        if self._session:
            self._session.request_drain()
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(b'{"type":"drain"}\n')
                await self._proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass                             # the worker already exited

    async def _cancel_stage(self) -> None:
        if self._session:
            await self._session.cancel()
            self._session = None
        if self._proc:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
            self._proc = None
        self.status, self.job_id, self.epoch = "idle", None, None
        self._write_status()

    async def _on_fetch(self, msg: VerifyFetch) -> None:
        dest = self.opt.paths.job_dir(msg.job_id, msg.epoch) / f"bundle_{msg.step}.safetensors"
        ok = False
        if self._session:
            ok = self._session.save_bundle(msg.step, dest)
        if not ok:
            await self.send(VerifyBundleReady(
                verify_id=msg.verify_id, job_id=msg.job_id, stage_idx=msg.stage_idx,
                step=msg.step, error="bundle no longer held",
            ))
            return
        self.opt.http.put_bytes(f"/verify/{msg.verify_id}/bundle", dest.read_bytes())
        await self.send(VerifyBundleReady(
            verify_id=msg.verify_id, job_id=msg.job_id, stage_idx=msg.stage_idx,
            step=msg.step, path=str(dest),
        ))

    async def _on_verify(self, msg: VerifyRequest) -> None:
        try:
            if msg.kind == "canary":
                stats = run_canary(msg.seed or 0, msg.size or self.opt.cfg.canary_size)
                await self.send(VerifyResult(verify_id=msg.verify_id, kind="canary", stats=stats))
                return
            work = self.opt.paths.root / "verify" / msg.verify_id
            work.mkdir(parents=True, exist_ok=True)
            bundle = self.opt.http.get_file(msg.bundle_url, work / "bundle.safetensors")
            dest = work / "replay.safetensors"
            stats = run_replay(msg, bundle, dest)
            self.opt.http.put_bytes(f"/verify/{msg.verify_id}/result", dest.read_bytes())
            await self.send(VerifyResult(
                verify_id=msg.verify_id, kind="replay", stats=stats, output_path=str(dest),
            ))
        except Exception as e:
            log.exception("verify %s failed", msg.kind)
            await self.send(VerifyResult(verify_id=msg.verify_id, kind=msg.kind, error=str(e)))

    async def shutdown(self) -> None:
        if self._stop.is_set():
            return
        log.info("shutting down")
        self._draining = True
        self._write_status()
        try:
            await self.send(DrainNotice(node_id=self.opt.node_id))
        except Exception:
            pass
        if self._session:
            self._session.request_drain()
            try:
                await asyncio.wait_for(self._session.task, timeout=self.opt.cfg.grace_period_s)
            except (asyncio.TimeoutError, asyncio.CancelledError, TypeError):
                await self._cancel_stage()
        elif self._proc:
            # Sandboxed: the worker drains on a stdin line; the pump forwards its StageFinished.
            await self._request_drain()
            try:
                await asyncio.wait_for(asyncio.shield(self._pump), timeout=self.opt.cfg.grace_period_s)
            except (asyncio.TimeoutError, TypeError):
                await self._cancel_stage()
        self._stop.set()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass


async def start_daemon(opt: AgentOptions) -> None:
    await Daemon(opt).run()


def request_stop(paths: AgentPaths) -> bool:
    pid = paths.read_pid()
    if pid is None or not _alive(pid):
        paths.clear_pid()
        return False
    __import__("os").kill(pid, signal.SIGTERM)
    return True
