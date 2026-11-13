"""Run one pipeline stage: load shard, connect peers, train, report."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

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
    if asg.stage_idx > 0:
        server = await LinkServer(ctx.data_bind, ctx.data_port, hello).start()
        log.info("listening for upstream on %s:%s", ctx.data_bind, server.port)
    if asg.next_peer is not None:
        log.info("dialing next stage %s:%s", asg.next_peer.host, asg.next_peer.port)
        nxt = await connect(asg.next_peer.host, asg.next_peer.port, hello)
    if server is not None:
        prev = await server.accept(timeout=180.0)
        log.info("upstream connected from %s", getattr(prev, "peername", "?"))
    return prev, nxt, server


async def run_stage(ctx: WorkerContext, emit: OnMessage) -> StageResult:
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

    compute = build_compute(spec, asg.layer_start, asg.layer_end, asg.num_layers,
                            ring_size=asg.verify_ring_size)
    if resume_from is not None:
        compute.load_checkpoint(resume_from)

    profile = profile_model(spec.model)
    recorder = UsageRecorder(profile, asg.layer_start, asg.layer_end)
    examples = load_examples(dataset_path, spec.model, spec.max_seq_len) if dataset_path else None

    prev, nxt, server = await _peer_links(ctx)
    try:
        await emit(StageReady(job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx))

        async def on_step(s: StepStats) -> None:
            await emit(StepMetrics(
                job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx, step=s.step,
                loss=s.loss, in_digest=s.in_digest, out_digest=s.out_digest,
                usage=recorder.sample(s),
            ))

        async def on_checkpoint(step: int, path: Path) -> None:
            ctx.http.put_bytes(
                f"/jobs/{asg.job_id}/checkpoints/{step}",
                path.read_bytes(),
                params={"epoch": asg.epoch, "stage": asg.stage_idx},
            )

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
        node_id=blob["node_id"],
    )

    async def emit(msg) -> None:
        print(dump(msg), flush=True)

    async def main() -> None:
        await run_stage(ctx, emit)

    asyncio.run(main())


if __name__ == "__main__":
    _stdio_app()
