"""Run one pipeline stage: load shard, connect peers, train, report."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import stat
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

import mlx.core as mx

from slashcompute.agent.http import CoordHTTP
from slashcompute.agent.throttle import make_pace
from slashcompute.common.protocol import StageAssignment, StageFinished, StageReady, StepMetrics
from slashcompute.metering.recorder import UsageRecorder
from slashcompute.pipeline.data import load_examples
from slashcompute.pipeline.local import build_compute
from slashcompute.pipeline.model_profile import profile_model
from slashcompute.pipeline.schedule import StageResult, StageRunner, StepStats
from slashcompute.transport import Link, LinkServer, connect

log = logging.getLogger(__name__)

OnMessage = Callable[[object], Awaitable[None]]


def bundle_path(job_dir: Path, step: int) -> Path:
    """Where a stage saves what a replay of ``step`` needs (see ``StageSession.save_bundle``)."""
    return job_dir / f"bundle_{step}.safetensors"


class StageSession:
    """In-process stage so the daemon can drain / save verify bundles."""

    def __init__(self) -> None:
        self.runner: Optional[StageRunner] = None
        self.compute = None
        self.task: Optional[asyncio.Task] = None
        self.assignment: Optional[StageAssignment] = None

    def request_drain(self) -> None:
        if self.runner is not None:
            self.runner.drain.set()

    def save_bundle(self, step: int, dest: Path) -> bool:
        if self.compute is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        return self.compute.save_bundle(step, dest)

    async def cancel(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass


# How long a stage keeps dialing the next stage, and waits for the previous one to dial it.
PEER_DIAL_TIMEOUT_S = 60.0
PEER_ACCEPT_TIMEOUT_S = 180.0


@dataclass
class WorkerContext:
    assignment: StageAssignment
    http: CoordHTTP
    job_dir: Path
    data_bind: str
    data_port: int
    gpu_percent: int
    node_id: str
    session: Optional[StageSession] = None


async def _peer_links(ctx: WorkerContext) -> tuple[Optional[Link], Optional[Link], Optional[LinkServer]]:
    asg = ctx.assignment
    hello = {"job_id": asg.job_id, "epoch": asg.epoch}
    server = None
    prev = nxt = None
    # Downstream listens; upstream dials (see transport.peer).
    try:
        if asg.stage_idx > 0:
            server = await LinkServer(ctx.data_bind, ctx.data_port, hello).start()
            log.info("listening for upstream on %s:%s", ctx.data_bind, server.port)
        if asg.next_peer is not None:
            log.info("dialing next stage %s:%s", asg.next_peer.host, asg.next_peer.port)
            nxt = await connect(asg.next_peer.host, asg.next_peer.port, hello,
                                timeout=PEER_DIAL_TIMEOUT_S)
        if server is not None:
            prev = await server.accept(timeout=PEER_ACCEPT_TIMEOUT_S)
            log.info("upstream connected from %s", getattr(prev, "peername", "?"))
    except BaseException:
        # The caller only gets these once the handshake returns, so a cancel or a failure in
        # the middle of it must let go of them here: a listener left bound keeps this agent's
        # data port, and no later stage in this process (--no-sandbox) could listen on it.
        # The listener first: closing it frees the port before anything below can wait.
        if server is not None:
            await server.close()
        if nxt is not None:
            await nxt.close()
        raise
    return prev, nxt, server


# Stage setup blocks for as long as its downloads and the model load take (minutes on first
# use), so it runs on this thread, not on the event loop: in an agent that runs stages
# in-process that loop also sends the heartbeats, and the coordinator evicts a node that
# misses them. One thread for the whole process: a setup whose stage was cancelled cannot be
# interrupted, and the next stage's model load must not overlap with it in memory.
_SETUP = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage-setup")


def _setup_stage(ctx: WorkerContext, abandoned: threading.Event):
    """The blocking part of stage setup, run on the setup thread. Returns None once the
    stage is ``abandoned`` (cancelled while this was running)."""
    asg = ctx.assignment
    spec = asg.spec
    ckdir = ctx.job_dir / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)

    dataset_path = None
    if asg.dataset_url:
        dataset_path = ctx.http.get_file(asg.dataset_url, ctx.job_dir / "dataset.jsonl")
    resume_from = None
    if asg.resume_step > 0 and asg.checkpoint_url:
        resume_from = ctx.http.get_file(asg.checkpoint_url, ctx.job_dir / "resume.safetensors")
    if abandoned.is_set():
        return None                              # don't load a model nobody is waiting for

    compute = build_compute(spec, asg.layer_start, asg.layer_end, asg.num_layers,
                            ring_size=asg.verify_ring_size)
    if resume_from is not None:
        compute.load_checkpoint(resume_from)
    # MLX evaluates a lazy array only on the thread that created it ("There is no
    # Stream(gpu, N) in current thread"), and the steps run on the event loop's thread: so
    # nothing built here may leave this thread unevaluated.
    mx.eval(compute.shard.parameters(), compute.optimizer.state)

    profile = profile_model(spec.model)
    recorder = UsageRecorder(profile, asg.layer_start, asg.layer_end)
    examples = load_examples(dataset_path, spec.model, spec.max_seq_len) if dataset_path else None
    return ckdir, compute, recorder, examples


def _upload_checkpoint(ctx: WorkerContext, step: int, path: Path) -> None:
    asg = ctx.assignment
    ctx.http.put_bytes(
        f"/jobs/{asg.job_id}/checkpoints/{step}",
        path.read_bytes(),
        params={"epoch": asg.epoch, "stage": asg.stage_idx},
    )


async def run_stage(ctx: WorkerContext, emit: OnMessage) -> StageResult:
    asg = ctx.assignment
    spec = asg.spec
    prev = nxt = server = None
    abandoned = threading.Event()
    # Setup is inside the try so a failure still reports StageFinished(error)
    # rather than leaving the coordinator waiting on a stage that never starts.
    try:
        try:
            ckdir, compute, recorder, examples = await asyncio.get_running_loop().run_in_executor(
                _SETUP, _setup_stage, ctx, abandoned)
        except asyncio.CancelledError:
            abandoned.set()                      # the thread goes on; it stops at its next check
            raise

        prev, nxt, server = await _peer_links(ctx)
        await emit(StageReady(job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx))

        async def on_step(s: StepStats) -> None:
            await emit(StepMetrics(
                job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx, step=s.step,
                loss=s.loss, in_digest=s.in_digest, out_digest=s.out_digest,
                usage=recorder.sample(s),
            ))

        async def on_checkpoint(step: int, path: Path) -> None:
            # Off the loop like setup: a slow upload must not hold up heartbeats.
            await asyncio.to_thread(_upload_checkpoint, ctx, step, path)

        runner = StageRunner(
            compute=compute, total_steps=spec.steps, microbatches=spec.microbatches,
            microbatch_size=spec.microbatch_size, checkpoint_every=asg.checkpoint_every,
            checkpoint_dir=ckdir, prev=prev, next=nxt,
            examples=examples, batch_size=spec.batch_size, seed=spec.seed,
            start_step=asg.resume_step, on_step=on_step, on_checkpoint=on_checkpoint,
            pace=make_pace(ctx.gpu_percent),
        )
        ctx_session = getattr(ctx, "session", None)
        if ctx_session is not None:
            ctx_session.runner = runner
            ctx_session.compute = compute
        result = await runner.run()
        await emit(StageFinished(
            job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx,
            reason=result.reason, last_step=result.last_step,
        ))
        return result
    except asyncio.CancelledError:
        await emit(StageFinished(
            job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx,
            reason="cancelled", last_step=asg.resume_step,
        ))
        raise
    except Exception as e:
        log.exception("stage failed")
        await emit(StageFinished(
            job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx,
            reason="error", last_step=asg.resume_step, detail=str(e),
        ))
        raise
    finally:
        for link in (prev, nxt):
            if link is not None:
                try:
                    await link.close()
                except Exception:
                    pass
        if server is not None:
            await server.close()


# --------------------------------------------------------------------------- subprocess CLI


def _export_bundle(ctx: WorkerContext, msg: dict) -> dict:
    """Save what this stage holds for a sampled step, for the daemon to upload; the answer
    echoes the request with ``ok``. Only this stage's own job and epoch and a plain step
    number are served, and the file goes to this stage's job directory (the one place the
    sandbox lets it write that the daemon reads), never to a path taken from the request."""
    asg = ctx.assignment
    job_id, epoch, step = msg.get("job_id"), msg.get("epoch"), msg.get("step")
    ok = False
    if job_id == asg.job_id and type(epoch) is int and epoch == asg.epoch and type(step) is int:
        try:
            ok = bool(ctx.session.save_bundle(step, bundle_path(ctx.job_dir, step)))
        except Exception:  # answer anyway: the daemon is waiting, and the stage must go on
            log.exception("could not save the bundle of step %d", step)
    return {"type": "bundle", "job_id": job_id, "epoch": epoch, "step": step, "ok": ok}


async def _serve_stdin(ctx: WorkerContext, reader: asyncio.StreamReader,
                       answer: Callable[[dict], None]) -> None:
    """What the daemon asks of a sandboxed worker, one JSON line each on stdin:
    ``{"type":"drain"}``, and ``{"type":"bundle","job_id":…,"epoch":…,"step":…}`` for a
    sampled step, which is answered on stdout (see ``_export_bundle``)."""
    while line := await reader.readline():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if not isinstance(msg, dict):
            continue
        if msg.get("type") == "drain":
            ctx.session.request_drain()
        elif msg.get("type") == "bundle":
            answer(_export_bundle(ctx, msg))


def _stdin_is_pipe() -> bool:
    try:
        mode = os.fstat(sys.stdin.fileno()).st_mode
    except (OSError, ValueError):
        return False
    return stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)


def _stdio_app() -> None:
    """Read assignment JSON from argv and emit protocol messages as JSON lines."""
    import argparse

    from slashcompute.agent.http import CoordHTTP
    from slashcompute.common.logging import setup_logging
    from slashcompute.common.protocol import dump, parse_coordinator_message

    p = argparse.ArgumentParser(description="/compute sandboxed worker")
    p.add_argument("--assignment", type=Path, required=True)
    args = p.parse_args()
    setup_logging("worker")
    blob = json.loads(args.assignment.read_text())
    asg = parse_coordinator_message(blob["assignment"])
    assert isinstance(asg, StageAssignment)

    ctx = WorkerContext(
        assignment=asg, http=CoordHTTP(blob["coordinator_url"], node_token=blob.get("node_token")),
        job_dir=Path(blob["job_dir"]), data_bind=blob["data_bind"],
        data_port=int(blob["data_port"]), gpu_percent=int(blob["gpu_percent"]),
        node_id=blob["node_id"], session=StageSession(),
    )

    async def emit(msg) -> None:
        print(dump(msg), flush=True)

    # Ctrl-C in the agent's terminal reaches the whole process group. What this worker does
    # is the daemon's call (a drain line on stdin, or SIGTERM), so it must not die of it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def answer(reply: dict) -> None:
        print(json.dumps(reply), flush=True)

    async def main() -> None:
        requests = None
        if _stdin_is_pipe():
            reader = asyncio.StreamReader()
            await asyncio.get_running_loop().connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
            requests = asyncio.create_task(_serve_stdin(ctx, reader, answer))
        else:
            log.warning("stdin is not a pipe; this worker cannot be drained or asked for "
                        "sampled steps")
        try:
            await run_stage(ctx, emit)
        finally:
            if requests is not None:
                requests.cancel()

    asyncio.run(main())


if __name__ == "__main__":
    _stdio_app()
