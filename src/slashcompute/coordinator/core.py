"""Coordinator state and message routing. All mutation happens on the event
loop thread, so no locking is needed."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from sqlmodel import select

from slashcompute.common.config import EngineConfig
from slashcompute.common.protocol import (
    CancelStage, DrainNotice, Heartbeat, Register, StageFinished, StageReady, StepMetrics,
    UsageSample, VerifyBundleReady, VerifyResult, Welcome,
)
from slashcompute.community.auth import Auth
from slashcompute.community.credits import Credits
from slashcompute.community.grants import Grants
from slashcompute.coordinator.checkpoints import CheckpointStore
from slashcompute.coordinator.db import Checkpoint, Database, Job, Node, now
from slashcompute.coordinator.ledger import Ledger
from slashcompute.coordinator.recovery import Recovery
from slashcompute.coordinator.registry import Registry, SendFn
from slashcompute.coordinator.scheduler import ACTIVE, TERMINAL, WAITING, JobRuntime, Scheduler
from slashcompute.coordinator.verification import VerificationManager
from slashcompute.common.config import allowed_model
from slashcompute.jobs import LoraFinetuneSpec, parse_spec

log = logging.getLogger(__name__)

MAX_DATASET_BYTES = 32 * 1024 * 1024


def _sane_usage(u: UsageSample) -> bool:
    return all(math.isfinite(v) and v >= 0 for v in u.model_dump().values())


def safe_dataset_source(raw: str, *, max_bytes: int = MAX_DATASET_BYTES) -> Path:
    """Host path the coordinator may copy into a job. Rejects traversal, links, and non-JSONL."""
    if not (raw or "").strip():
        raise ValueError("dataset path is required.")
    if ".." in raw.replace("\\", "/"):
        raise ValueError("dataset path is not allowed.")
    src = Path(raw).expanduser()
    if src.is_symlink():
        raise ValueError("dataset path must not be a symlink.")
    if not src.is_file():
        raise FileNotFoundError(f"dataset not found on coordinator: {src}")
    resolved = src.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("dataset path is not allowed.")
    if resolved.suffix.lower() != ".jsonl":
        raise ValueError("dataset must be a .jsonl file.")
    if resolved.stat().st_size > max_bytes:
        raise ValueError("dataset is too large.")
    return resolved


class Coordinator:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        cfg.coordinator_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(cfg.coordinator_dir / "coordinator.db")
        self.registry = Registry()
        self.ledger = Ledger(self.db)
        self.auth = Auth(self.db)
        self.credits = Credits(self.db)
        self.grants = Grants(self.db, self.credits)
        self.checkpoints = CheckpointStore(cfg.coordinator_dir / "jobs")
        self.scheduler = Scheduler(self)
        self.recovery = Recovery(self)
        self.verification = VerificationManager(self)
        self.jobs: dict[str, JobRuntime] = {}
        self._task: Optional[asyncio.Task] = None
        self._load_jobs()

    def _load_jobs(self) -> None:
        """Unfinished jobs survive a coordinator restart and resume from their
        last checkpoint once nodes reconnect."""
        with self.db.session() as s:
            rows = s.exec(select(Job)).all()
        for row in rows:
            spec = parse_spec(json.loads(row.spec_json))
            if row.status in ACTIVE:
                row.status = "recovering"
                self.db.save(row)
            self.jobs[row.id] = JobRuntime(row=row, spec=spec)

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:
                log.exception("coordinator tick failed")
            await asyncio.sleep(self.cfg.scheduler_tick_s)

    async def tick(self) -> None:
        await self.recovery.tick()
        await self.verification.tick()
        await self.scheduler.tick()

    async def send(self, node_id: str, msg: BaseModel) -> None:
        node = self.registry.get(node_id)
        if node is None:
            return
        try:
            await node.send(msg)
        except Exception as e:
            log.warning("send to %s failed: %s", node_id[:8], e)

    # ------------------------------------------------------------ jobs

    def submit(self, spec: LoraFinetuneSpec) -> JobRuntime:
        if self.cfg.public_pool:
            spec = spec.model_copy(update={"min_stages": 1, "max_stages": 1})
        if not allowed_model(spec.model):
            raise ValueError(f"model {spec.model!r} is not allowed.")
        src = safe_dataset_source(spec.dataset_path)
        job_id = uuid.uuid4().hex[:12]
        dest = self.checkpoints.job_dir(job_id) / "dataset.jsonl"
        shutil.copyfile(src, dest)
        row = Job(id=job_id, kind=spec.kind, spec_json=spec.model_dump_json())
        self.db.add(row)
        job = JobRuntime(row=row, spec=spec)
        self.jobs[job_id] = job
        log.info("job %s submitted: %s steps=%d", job_id, spec.model, spec.steps)
        return job

    def abandon_job(self, job: JobRuntime, reason: str) -> None:
        """Drop a job that never reserved credits (failed Take). It was never
        accepted, so nothing of it may survive a restart."""
        log.info("job %s abandoned: %s", job.id, reason)
        self.jobs.pop(job.id, None)
        with self.db.session() as s:
            row = s.get(Job, job.id)
            if row is not None:
                s.delete(row)
                s.commit()
        shutil.rmtree(self.checkpoints.root / job.id, ignore_errors=True)

    def dataset_path(self, job_id: str) -> Path:
        return self.checkpoints.job_dir(job_id) / "dataset.jsonl"

    async def cancel_job(self, job: JobRuntime) -> None:
        if job.row.status in TERMINAL:
            return
        cur = job.current
        if cur and not cur.closed:
            cur.closed = True
            for p in cur.plans:
                node = self.registry.get(p.node_id)
                if node and node.assignment and node.assignment.job_id == job.id:
                    node.assignment = None
                await self.send(p.node_id, CancelStage(job_id=job.id, epoch=cur.epoch))
        job.row.status, job.row.finished_at = "cancelled", now()
        self.db.save(job.row)
        self.credits.settle_job(job.id)

    async def fail_job(self, job: JobRuntime, reason: str) -> None:
        log.error("job %s failed: %s", job.id, reason)
        job.row.status, job.row.error, job.row.finished_at = "failed", reason, now()
        self.db.save(job.row)
        self.credits.settle_job(job.id)

    async def complete_job(self, job: JobRuntime) -> None:
        row = job.row
        try:
            self.checkpoints.export_adapter(job.id, job.spec.steps, job.spec, job.profile.num_layers)
        except Exception as e:
            log.exception("adapter export failed")
            row.error = f"adapter export failed: {e}"
        else:
            row.error = None  # an earlier epoch's abort reason no longer applies
        row.status, row.finished_at = "completed", now()
        self.db.save(row)
        self.credits.settle_job(job.id)
        log.info("job %s completed (%d steps, last loss %s)", job.id, row.progress_step, row.last_loss)

    def on_checkpoint_upload(self, job_id: str, epoch: int, stage_idx: int, step: int,
                             data: bytes) -> None:
        job = self.jobs.get(job_id)
        if job is None or job.current is None or job.current.epoch != epoch:
            raise ValueError("unknown or stale job epoch")
        merged = self.checkpoints.store_stage(job_id, epoch, step, stage_idx, data, job.num_stages)
        if merged is not None and step > job.row.last_checkpoint_step:
            job.row.last_checkpoint_step = step
            self.db.save(job.row)
            self.db.add(Checkpoint(job_id=job_id, step=step, path=str(merged)))

    # ------------------------------------------------------------ agent sessions

    async def on_register(self, msg: Register, send: SendFn) -> None:
        user = self.auth.session_user(msg.session_token) if msg.session_token else None
        if self.cfg.public_pool and user is None:
            raise PermissionError("sign in")
        if user is not None and user.banned:
            raise PermissionError("banned")
        if user is not None and user.accepted_terms_at is None:
            if self.cfg.public_pool:
                raise PermissionError("accept terms")
            log.warning("node %s session has not accepted terms — compute only", msg.node_id[:8])
            user = None
        elif msg.session_token and user is None:
            log.warning("node %s presented a bad session token", msg.node_id[:8])
        self._check_claim(msg.node_id, user.id if user else None)
        if self.registry.get(msg.node_id) is not None:
            await self.recovery.on_node_lost(msg.node_id, "re-registered")
        state = self.registry.register(msg, send)
        if user is not None:
            state.user_id = user.id
            self.credits.bind_node(msg.node_id, user.id, take_over=True)
        d = msg.device
        row = self.db.get(Node, msg.node_id) or Node(
            id=msg.node_id, name=msg.name, chip=d.chip, memory_contrib_bytes=d.memory_contrib_bytes,
            matmul_tflops=d.matmul_tflops, mem_bandwidth_gbps=d.mem_bandwidth_gbps)
        row.name, row.chip, row.memory_contrib_bytes = msg.name, d.chip, d.memory_contrib_bytes
        row.matmul_tflops, row.mem_bandwidth_gbps = d.matmul_tflops, d.mem_bandwidth_gbps
        row.online, row.last_seen = True, now()
        self.db.save(row)
        log.info("node %s registered: %s (%s, lends %.1f GB, %.1f TFLOPS, gpu %d%%)",
                 msg.node_id[:8], msg.name, d.chip, d.memory_contrib_bytes / 1e9, d.matmul_tflops,
                 msg.gpu_percent)
        await send(Welcome(node_id=msg.node_id, heartbeat_interval_s=self.cfg.heartbeat_interval_s))

    def _check_claim(self, node_id: str, claimant: Optional[str]) -> None:
        """Node ids are public (GET /nodes), so a stranger must not evict a live node or rebind
        its earnings: a live node answers only to its owner. An offline training node may be taken
        over by any signed-in account (a Mac that switched accounts; the old owner keeps what it
        earned), but not anonymously. Inference nodes share the owner table yet never appear in
        the training node table, so they stay out of reach here."""
        live = self.registry.get(node_id)
        owner = (live.user_id if live else None) or self.credits.owner_of(node_id)
        if owner is None or owner == claimant:
            return
        if claimant is None and live is None and not self.cfg.public_pool:
            # LAN: a Mac whose session lapsed (or that carries another pool's token) keeps
            # contributing; owner_of still pays its owner, so nothing is taken.
            return
        if claimant is None:
            raise PermissionError("node id belongs to an account: sign in and accept the terms")
        if live is not None or self.db.get(Node, node_id) is None:
            raise PermissionError("node id belongs to another account")

    async def on_disconnect(self, node_id: str) -> None:
        await self.recovery.on_node_lost(node_id, "disconnected")

    async def handle(self, node_id: str, msg: BaseModel) -> None:
        # One message that trips a bug must not drop the node's whole connection.
        try:
            await self._dispatch(node_id, msg)
        except Exception:
            log.exception("failed handling %s from %s", type(msg).__name__, node_id[:8])

    async def _dispatch(self, node_id: str, msg: BaseModel) -> None:
        if isinstance(msg, Heartbeat):
            self.registry.heartbeat(node_id, msg.status)
        elif isinstance(msg, DrainNotice):
            await self.recovery.on_drain(node_id)
        elif isinstance(msg, (StageReady, StepMetrics, StageFinished)) \
                and not self._holds_stage(node_id, msg.job_id, msg.epoch, msg.stage_idx):
            # Only the node planned for a stage may report on it; otherwise any
            # node could bill a stranger's budget or end their job.
            log.warning("dropping %s from %s: not assigned to job %s epoch %d stage %d",
                        type(msg).__name__, node_id[:8], msg.job_id, msg.epoch, msg.stage_idx)
        elif isinstance(msg, StageReady):
            await self.scheduler.on_stage_ready(self.jobs[msg.job_id], msg.epoch, msg.stage_idx)
        elif isinstance(msg, StepMetrics):
            await self._on_step(node_id, msg)
        elif isinstance(msg, StageFinished):
            await self.recovery.on_stage_finished(node_id, msg)
        elif isinstance(msg, VerifyBundleReady):
            await self.verification.on_bundle_ready(node_id, msg)
        elif isinstance(msg, VerifyResult):
            await self.verification.on_result(node_id, msg)
        else:
            log.warning("unexpected message from %s: %s", node_id[:8], type(msg).__name__)

    def _holds_stage(self, node_id: str, job_id: str, epoch: int, stage_idx: int) -> bool:
        job = self.jobs.get(job_id)
        cur = job.current if job else None
        return cur is not None and cur.epoch == epoch and any(
            p.stage_idx == stage_idx and p.node_id == node_id for p in cur.plans)

    async def _on_step(self, node_id: str, msg: StepMetrics) -> None:
        job = self.jobs.get(msg.job_id)
        if job is None or job.current is None or job.current.epoch != msg.epoch:
            return
        if not _sane_usage(msg.usage):
            # NaN can't be stored, and inf would bill the job's whole reserve to this node.
            log.warning("ignoring step %d of job %s from %s: bad usage %s",
                        msg.step, msg.job_id, node_id[:8], msg.usage)
            return
        self.ledger.record_step(node_id, msg)
        flops = float(msg.usage.flops)
        job.last_step_flops = flops
        acct = self.credits.job_account(job.id)
        take = self.credits.consume_job(job.id, flops) if acct is not None else 0.0
        node = self.registry.get(node_id)
        owner_id = (node.user_id if node and node.user_id else self.credits.owner_of(node_id))
        self.credits.credit_host(owner_id, take, node_id=node_id, job_id=msg.job_id)
        if acct is not None and self.credits.job_exhausted(job.id):
            await self.cancel_job(job)
            job.row.error = "FLOP budget spent"
            self.db.save(job.row)
            return
        if msg.loss is not None:
            job.row.progress_step = msg.step
            job.row.last_loss = msg.loss
            self.db.save(job.row)
            if msg.step % 10 == 0 or msg.step == job.spec.steps:
                log.info("job %s step %d/%d loss %.4f", job.id, msg.step, job.spec.steps, msg.loss)
        await self.verification.on_step(job, node_id, msg)

    # ------------------------------------------------------------ waitlist

    def waiting_jobs(self) -> list[JobRuntime]:
        return sorted(
            (j for j in self.jobs.values() if j.row.status in WAITING),
            key=lambda j: j.row.submitted_at,
        )

    def pool_tflops(self) -> float:
        return float(sum(n.device.matmul_tflops for n in self.registry.nodes.values()))

    def last_step_flops(self, job: JobRuntime) -> Optional[float]:
        if job.last_step_flops is not None:
            return float(job.last_step_flops)
        recs = [r for r in self.ledger.job_records(job.id) if r.kind == "train"]
        if not recs:
            return None
        job.last_step_flops = float(recs[-1].flops)
        return job.last_step_flops

    def remaining_flops(self, job: JobRuntime) -> Optional[float]:
        acct = self.credits.job_account(job.id)
        if acct is not None:
            return max(0.0, acct.reserved_flops - acct.spent_flops)
        last = self.last_step_flops(job)
        if last is None:
            return None
        leftover = max(0, int(job.spec.steps) - int(job.row.progress_step))
        return leftover * last

    def queue_position(self, job: JobRuntime) -> Optional[int]:
        if job.row.status not in WAITING:
            return None
        for i, other in enumerate(self.waiting_jobs(), 1):
            if other.id == job.id:
                return i
        return None

    def wait_seconds(self, job: JobRuntime) -> Optional[float]:
        if job.row.status not in WAITING:
            return None
        ahead = [
            other for other in self.jobs.values()
            if other.id != job.id and (
                other.row.status in ACTIVE
                or (other.row.status in WAITING and other.row.submitted_at < job.row.submitted_at)
            )
        ]
        remaining = 0.0
        for other in ahead:
            rem = self.remaining_flops(other)
            if rem is None:
                return None
            remaining += rem
        tflops = self.pool_tflops()
        if tflops <= 0:
            return None
        return remaining / (tflops * 1e12)

    def waitlist(self) -> list[dict]:
        return [self.job_view(j) for j in self.waiting_jobs()]

    # ------------------------------------------------------------ views

    def job_view(self, job: JobRuntime) -> dict:
        row = job.row
        cur = job.current
        return {
            "id": row.id, "kind": row.kind, "status": row.status, "model": job.spec.model,
            "steps": job.spec.steps, "progress_step": row.progress_step, "last_loss": row.last_loss,
            "epoch": row.epoch, "recoveries": row.recoveries,
            "last_checkpoint_step": row.last_checkpoint_step, "error": row.error,
            "wait_reason": job.wait_reason if row.status in WAITING else None,
            "queue_position": self.queue_position(job),
            "wait_s": self.wait_seconds(job),
            "submitted_at": row.submitted_at, "started_at": row.started_at,
            "finished_at": row.finished_at,
            "stages": [
                {"stage_idx": p.stage_idx, "node_id": p.node_id, "layers": [p.layer_start, p.layer_end],
                 "est_bytes": p.est_bytes, "ready": p.stage_idx in cur.ready,
                 "finished": cur.finished.get(p.stage_idx)}
                for p in cur.plans
            ] if cur and not cur.closed else [],
            "adapter_dir": str(self.checkpoints.job_dir(row.id) / "adapter")
            if row.status == "completed" else None,
            **self._job_account_view(job.id),
        }

    def _job_account_view(self, job_id: str) -> dict:
        acct = self.credits.job_account(job_id)
        if acct is None:
            return {"user_id": None, "reserved_flops": None, "spent_flops": None}
        return {"user_id": acct.user_id, "reserved_flops": acct.reserved_flops,
                "spent_flops": acct.spent_flops}

    def node_view(self) -> list[dict]:
        out = []
        for n in self.registry.nodes.values():
            out.append({
                "node_id": n.node_id, "name": n.name, "chip": n.device.chip,
                "memory_contrib_bytes": n.device.memory_contrib_bytes,
                "matmul_tflops": n.device.matmul_tflops, "gpu_percent": n.gpu_percent,
                "status": n.status, "draining": n.draining, "canary_passed": n.canary_passed,
                "assignment": n.assignment.__dict__ if n.assignment else None,
                "data_addr": f"{n.data_host}:{n.data_port}",
                "user_id": n.user_id,
            })
        return out
