"""FIFO job scheduling and per-job runtime state.

Strict FIFO: only the oldest waiting job is considered. If the pool can't fit
it yet, later jobs wait too (no starvation of large jobs).

Each (re)start of a job is an *epoch*. Messages carry the epoch so anything
from a torn-down epoch is ignored.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from slashcompute.common.protocol import StageAssignment
from slashcompute.coordinator.db import Job, StageRun, now
from slashcompute.coordinator.partitioner import NodeCapacity, Overhead, PartitionError, StagePlan, partition
from slashcompute.coordinator.registry import Assignment
from slashcompute.jobs import LoraFinetuneSpec
from slashcompute.pipeline.model_profile import ModelProfile, profile_model

if TYPE_CHECKING:
    from slashcompute.coordinator.core import Coordinator

log = logging.getLogger(__name__)

WAITING = ("queued", "recovering")
ACTIVE = ("starting", "running")
TERMINAL = ("completed", "failed", "cancelled")


@dataclass
class EpochState:
    epoch: int
    plans: list[StagePlan]
    started: float = field(default_factory=time.monotonic)
    ready: set[int] = field(default_factory=set)
    finished: dict[int, str] = field(default_factory=dict)
    # stage_idx -> last step whose metrics were accepted (each step bills once)
    metered: dict[int, int] = field(default_factory=dict)
    drain_requested: bool = False
    closed: bool = False

    def node_for(self, stage_idx: int) -> str:
        return self.plans[stage_idx].node_id

    def stage_of(self, node_id: str) -> Optional[StagePlan]:
        return next((p for p in self.plans if p.node_id == node_id), None)


@dataclass
class JobRuntime:
    row: Job
    spec: LoraFinetuneSpec
    profile: Optional[ModelProfile] = None
    current: Optional[EpochState] = None
    # (epoch, step) -> {stage_idx: (in_digest, out_digest, node_id)}
    digests: dict[tuple[int, int], dict[int, tuple[str, str, str]]] = field(default_factory=dict)
    wait_reason: Optional[str] = None

    @property
    def id(self) -> str:
        return self.row.id

    @property
    def num_stages(self) -> int:
        return len(self.current.plans) if self.current else 0


class Scheduler:
    def __init__(self, core: "Coordinator") -> None:
        self.core = core

    def overhead(self) -> Overhead:
        cfg = self.core.cfg
        return Overhead(frac=cfg.stage_overhead_frac, fixed_bytes=cfg.stage_overhead_bytes)

    def next_waiting(self) -> Optional[JobRuntime]:
        waiting = [j for j in self.core.jobs.values()
                   if j.row.status in WAITING and (j.current is None or j.current.closed)]
        return min(waiting, key=lambda j: j.row.submitted_at, default=None)

    async def tick(self) -> None:
        job = self.next_waiting()
        if job is not None:
            await self.try_start(job)
        await self._check_start_timeouts()

    async def try_start(self, job: JobRuntime) -> bool:
        core = self.core
        if job.profile is None:
            try:
                job.profile = profile_model(job.spec.model)
            except Exception as e:
                await core.fail_job(job, f"could not read model {job.spec.model!r}: {e}")
                return False
        nodes = core.registry.schedulable()
        caps = [NodeCapacity(n.node_id, n.device.memory_contrib_bytes) for n in nodes]
        try:
            plans = partition(job.profile, caps, min_stages=job.spec.min_stages,
                              max_stages=job.spec.max_stages, overhead=self.overhead())
        except PartitionError as e:
            if job.wait_reason != str(e):
                log.info("job %s waiting: %s", job.id, e)
            job.wait_reason = str(e)
            return False
        job.wait_reason = None

        row = job.row
        row.epoch += 1
        resume = row.last_checkpoint_step
        if row.started_at is None:
            row.started_at = now()
        row.status = "starting"
        job.current = EpochState(epoch=row.epoch, plans=plans)
        core.db.save(row)
        log.info("job %s epoch %d: %d stage(s) %s resume_step=%d", job.id, row.epoch, len(plans),
                 [(p.node_id[:8], p.layer_start, p.layer_end) for p in plans], resume)

        for p in plans:
            node = core.registry.get(p.node_id)
            node.assignment = Assignment(job.id, row.epoch, p.stage_idx)
            core.db.add(StageRun(job_id=job.id, epoch=row.epoch, stage_idx=p.stage_idx,
                                 node_id=p.node_id, layer_start=p.layer_start, layer_end=p.layer_end))
        for p in plans:
            prev_node = core.registry.get(plans[p.stage_idx - 1].node_id) if p.stage_idx > 0 else None
            next_node = core.registry.get(plans[p.stage_idx + 1].node_id) if p.stage_idx < len(plans) - 1 else None
            msg = StageAssignment(
                job_id=job.id, epoch=row.epoch, stage_idx=p.stage_idx, num_stages=len(plans),
                layer_start=p.layer_start, layer_end=p.layer_end, num_layers=job.profile.num_layers,
                spec=job.spec, prev_peer=prev_node.peer if prev_node else None,
                next_peer=next_node.peer if next_node else None, resume_step=resume,
                checkpoint_url=f"/jobs/{job.id}/checkpoints/{resume}" if resume > 0 else None,
                dataset_url=f"/jobs/{job.id}/dataset" if p.stage_idx == 0 else None,
                checkpoint_every=job.spec.checkpoint_every or core.cfg.checkpoint_every,
                verify_ring_size=core.cfg.verify_ring_size,
            )
            await core.send(p.node_id, msg)
        return True

    async def on_stage_ready(self, job: JobRuntime, epoch: int, stage_idx: int) -> None:
        cur = job.current
        if cur is None or cur.epoch != epoch or cur.closed:
            return
        cur.ready.add(stage_idx)
        if len(cur.ready) == len(cur.plans) and job.row.status == "starting":
            job.row.status = "running"
            self.core.db.save(job.row)
            log.info("job %s epoch %d running", job.id, epoch)

    async def _check_start_timeouts(self) -> None:
        limit = self.core.cfg.stage_start_timeout_s
        for job in list(self.core.jobs.values()):
            cur = job.current
            if (job.row.status == "starting" and cur and not cur.closed
                    and time.monotonic() - cur.started > limit):
                await self.core.recovery.abort_epoch(job, f"stages not ready within {limit:.0f}s")
