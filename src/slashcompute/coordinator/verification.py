"""Checking that contributed compute is real and correct.

Three mechanisms:

* Chain check (every step, free): stage k's committed output digest must
  equal stage k+1's committed input digest.
* Sampled replay (~verify_rate of steps): the stage uploads the microbatch-0
  input, output and adapters it used. The upload must match the digests it
  committed at step time, then a *different* idle node recomputes the output
  and the two are compared within a tolerance. If no other node is idle the
  replay waits (for example until the job finishes and frees its nodes).
* Canary: on join (and periodically when idle) a node runs a seeded matmul
  whose statistics the coordinator computes independently. Nodes that fail
  are excluded from scheduling.

Failures mark the affected usage records ``disputed``.
"""

from __future__ import annotations

import logging
import math
import random
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import mlx.core as mx
import numpy as np
from sqlmodel import select

from slashcompute.common.canary import compare_stats, expected_stats
from slashcompute.common.protocol import (
    StepMetrics, UsageSample, VerifyBundleReady, VerifyFetch, VerifyRequest, VerifyResult,
)
from slashcompute.coordinator.db import Node, StageRun, Verification, now
from slashcompute.metering.flops import replay_flops
from slashcompute.transport.serialization import digest

if TYPE_CHECKING:
    from slashcompute.coordinator.core import Coordinator
    from slashcompute.coordinator.scheduler import JobRuntime

log = logging.getLogger(__name__)


def relative_error(claimed: np.ndarray, replay: np.ndarray) -> float:
    a, b = claimed.astype(np.float64), replay.astype(np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


class VerificationManager:
    def __init__(self, core: "Coordinator") -> None:
        self.core = core
        self.rng = random.Random()
        self._canary_expected: dict[str, tuple[int, int]] = {}

    def dir(self, vid: str) -> Path:
        d = self.core.cfg.coordinator_dir / "verify" / vid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def bundle_path(self, vid: str) -> Path:
        return self.dir(vid) / "bundle.safetensors"

    def result_path(self, vid: str) -> Path:
        return self.dir(vid) / "replay.safetensors"

    def _get(self, vid: str) -> Optional[Verification]:
        return self.core.db.get(Verification, vid)

    def _finish(self, v: Verification, status: str, detail: str | None = None,
                rel_error: float | None = None) -> None:
        # inf/NaN are not valid JSON and would break GET /verifications.
        if rel_error is not None and not math.isfinite(rel_error):
            rel_error = None
        v.status, v.detail, v.rel_error, v.finished_at = status, detail, rel_error, now()
        self.core.db.save(v)
        if status == "failed" and v.job_id is not None and v.step is not None:
            self.core.ledger.dispute(v.job_id, v.epoch, v.stage_idx, v.step)
        log.log(logging.WARNING if status == "failed" else logging.INFO,
                "verification %s (%s job=%s step=%s stage=%s node=%s): %s %s", v.id[:8], v.kind,
                v.job_id, v.step, v.stage_idx, v.target_node_id[:8], status, detail or "")

    # ------------------------------------------------------------ per step

    async def on_step(self, job: "JobRuntime", node_id: str, msg: StepMetrics) -> None:
        self._chain_check(job, node_id, msg)
        if self.rng.random() < self.core.cfg.verify_rate:
            await self.request_replay(job, node_id, msg.epoch, msg.stage_idx, msg.step)

    def _chain_check(self, job: "JobRuntime", node_id: str, msg: StepMetrics) -> None:
        key = (msg.epoch, msg.step)
        per_stage = job.digests.setdefault(key, {})
        per_stage[msg.stage_idx] = (msg.in_digest, msg.out_digest, node_id)
        for a, b in ((msg.stage_idx - 1, msg.stage_idx), (msg.stage_idx, msg.stage_idx + 1)):
            if a in per_stage and b in per_stage and per_stage[a][1] != per_stage[b][0]:
                for idx in (a, b):
                    v = Verification(id=uuid.uuid4().hex, kind="chain", job_id=job.id,
                                     epoch=msg.epoch, step=msg.step, stage_idx=idx,
                                     target_node_id=per_stage[idx][2])
                    self._finish(v, "failed", f"stage {a} output != stage {b} input")
        # keep memory bounded
        for old in [k for k in job.digests if k[0] == msg.epoch and k[1] < msg.step - 50]:
            job.digests.pop(old, None)

    async def request_replay(self, job: "JobRuntime", node_id: str, epoch: int, stage_idx: int,
                             step: int) -> str:
        v = Verification(id=uuid.uuid4().hex, kind="replay", job_id=job.id, epoch=epoch, step=step,
                         stage_idx=stage_idx, target_node_id=node_id, status="fetching")
        self.core.db.add(v)
        await self.core.send(node_id, VerifyFetch(verify_id=v.id, job_id=job.id, epoch=epoch,
                                                  stage_idx=stage_idx, step=step))
        return v.id

    # ------------------------------------------------------------ bundle

    def store_bundle(self, vid: str, data: bytes) -> None:
        v = self._get(vid)
        if v is None or v.kind != "replay" or v.status != "fetching":
            raise KeyError(vid)
        self.bundle_path(vid).write_bytes(data)

    async def on_bundle_ready(self, node_id: str, msg: VerifyBundleReady) -> None:
        v = self._get(msg.verify_id)
        if v is None or v.status != "fetching":
            return
        if msg.error or not self.bundle_path(v.id).exists():
            self._finish(v, "error", msg.error or "bundle missing")
            return
        record = self.core.ledger.find_step(v.job_id, v.epoch, v.stage_idx, v.step)
        tensors = mx.load(str(self.bundle_path(v.id)))
        if record is None or digest(tensors["x_in"]) != record.in_digest \
                or digest(tensors["out"]) != record.out_digest:
            self._finish(v, "failed", "uploaded bundle does not match digests committed at step time")
            return
        v.status = "queued"
        self.core.db.save(v)

    # ------------------------------------------------------------ dispatch

    async def tick(self) -> None:
        await self._dispatch_replays()
        await self._dispatch_canaries()

    async def _dispatch_replays(self) -> None:
        core = self.core
        with core.db.session() as s:
            queued = list(s.exec(select(Verification).where(
                Verification.kind == "replay", Verification.status == "queued",
            ).order_by(Verification.created_at)).all())
        for v in queued:
            idle = [n for n in core.registry.schedulable()
                    if n.node_id != v.target_node_id and n.canary_passed]
            if not idle:
                continue
            verifier = idle[0]
            job = core.jobs.get(v.job_id)
            with core.db.session() as s:
                run = s.exec(select(StageRun).where(
                    StageRun.job_id == v.job_id, StageRun.epoch == v.epoch,
                    StageRun.stage_idx == v.stage_idx)).first()
            if job is None or run is None or job.profile is None:
                self._finish(v, "error", "job or stage metadata missing")
                continue
            verifier.verifying = v.id
            v.status, v.verifier_node_id = "running", verifier.node_id
            core.db.save(v)
            spec = job.spec
            await core.send(verifier.node_id, VerifyRequest(
                verify_id=v.id, kind="replay", model=spec.model, layer_start=run.layer_start,
                layer_end=run.layer_end, num_layers=job.profile.num_layers,
                lora_rank=spec.lora_rank, lora_scale=spec.lora_scale, lora_targets=spec.lora_targets,
                bundle_url=f"/verify/{v.id}/bundle",
            ))

    async def _dispatch_canaries(self) -> None:
        core = self.core
        for node in list(core.registry.nodes.values()):
            due = node.canary_passed is None or (
                node.canary_passed and time.monotonic() - node.last_canary > core.cfg.canary_interval_s)
            if not due or node.draining or node.assignment or node.verifying:
                continue
            seed, size = self.rng.randrange(2**31), core.cfg.canary_size
            v = Verification(id=uuid.uuid4().hex, kind="canary", target_node_id=node.node_id,
                             verifier_node_id=node.node_id, status="running")
            core.db.add(v)
            self._canary_expected[v.id] = (seed, size)
            node.verifying = v.id
            node.last_canary = time.monotonic()
            await core.send(node.node_id, VerifyRequest(verify_id=v.id, kind="canary",
                                                        seed=seed, size=size))

    # ------------------------------------------------------------ results

    def store_result(self, vid: str, data: bytes) -> None:
        v = self._get(vid)
        if v is None or v.kind != "replay" or v.status != "running":
            raise KeyError(vid)
        self.result_path(vid).write_bytes(data)

    async def on_result(self, node_id: str, msg: VerifyResult) -> None:
        core = self.core
        node = core.registry.get(node_id)
        if node is not None and node.verifying == msg.verify_id:
            node.verifying = None
        v = self._get(msg.verify_id)
        if v is None or v.status != "running" or v.verifier_node_id != node_id:
            return
        if msg.kind == "canary":
            seed, size = self._canary_expected.pop(v.id)
            if msg.error:
                passed, worst, detail = False, None, msg.error
            else:
                passed, worst = compare_stats(msg.stats, expected_stats(seed, size),
                                              core.cfg.canary_rel_tolerance)
                detail = None if math.isfinite(worst) else "missing or non-finite statistic"
            self._finish(v, "passed" if passed else "failed", detail, worst)
            if node is not None:
                node.canary_passed = passed
            row = core.db.get(Node, v.target_node_id)
            if row is not None:
                row.canary_passed = passed
                core.db.save(row)
            return

        if msg.error or not self.result_path(v.id).exists():
            self._finish(v, "error", msg.error or "replay output missing")
            return
        claimed = mx.load(str(self.bundle_path(v.id)))
        replay = mx.load(str(self.result_path(v.id)))["out"]
        err = relative_error(np.asarray(claimed["out"].astype(mx.float32)),
                             np.asarray(replay.astype(mx.float32)))
        ok = err <= core.cfg.verify_rel_tolerance
        self._finish(v, "passed" if ok else "failed", None if ok else "replay mismatch", err)
        job = core.jobs.get(v.job_id)
        if job is not None and job.profile is not None:
            with core.db.session() as s:
                run = s.exec(select(StageRun).where(
                    StageRun.job_id == v.job_id, StageRun.epoch == v.epoch,
                    StageRun.stage_idx == v.stage_idx)).first()
            x = claimed["x_in"]
            flops = replay_flops(job.profile, run.layer_start, run.layer_end, x.shape[0] * x.shape[1],
                                 x.shape[1], run.layer_end == job.profile.num_layers)
            core.ledger.record_verify(node_id, v.job_id, UsageSample(
                flops=flops, tokens=x.shape[0] * x.shape[1], peak_mem_bytes=0,
                resident_mem_bytes=0, mem_byte_seconds=0.0,
                wall_s=msg.stats.get("wall_s", 0.0), busy_s=msg.stats.get("busy_s", 0.0)))

    async def on_node_lost(self, node_id: str) -> None:
        core = self.core
        with core.db.session() as s:
            open_vs = list(s.exec(select(Verification).where(
                Verification.status.in_(["fetching", "running"]))).all())
        for v in open_vs:
            if v.status == "fetching" and v.target_node_id == node_id:
                self._finish(v, "error", "target node left before uploading bundle")
            elif v.status == "running" and v.verifier_node_id == node_id:
                if v.kind == "canary":
                    self._finish(v, "error", "node left during canary")
                else:  # hand it to another verifier
                    v.status, v.verifier_node_id = "queued", None
                    core.db.save(v)
