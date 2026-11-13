"""Control-plane messages exchanged over the agent <-> coordinator WebSocket
and the daemon <-> worker stdio pipe. Every message is a JSON object with a
``type`` discriminator."""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter

from slashcompute.jobs.lora_finetune import LoraFinetuneSpec


class DeviceProfile(BaseModel):
    chip: str
    memory_total_bytes: int
    memory_available_bytes: int
    working_set_bytes: int
    # What the contributor is willing to lend (after any --max-memory-gb cap).
    memory_contrib_bytes: int
    matmul_tflops: float
    mem_bandwidth_gbps: float


class UsageSample(BaseModel):
    """Raw per-step measurements for one stage. Credits post from these when a job is reserved."""

    flops: float
    tokens: int
    peak_mem_bytes: int
    resident_mem_bytes: int
    mem_byte_seconds: float
    wall_s: float
    busy_s: float


class PeerAddr(BaseModel):
    node_id: str
    host: str
    port: int


# ---------------------------------------------------------------- agent -> coordinator


class Register(BaseModel):
    type: Literal["register"] = "register"
    node_id: str
    name: str
    device: DeviceProfile
    data_host: str
    data_port: int
    gpu_percent: int
    session_token: Optional[str] = None


class Heartbeat(BaseModel):
    type: Literal["heartbeat"] = "heartbeat"
    node_id: str
    status: Literal["idle", "loading", "running", "draining"]
    job_id: Optional[str] = None
    epoch: Optional[int] = None


class DrainNotice(BaseModel):
    """Contributor asked to stop; the node leaves after its current step."""

    type: Literal["drain_notice"] = "drain_notice"
    node_id: str


class StageReady(BaseModel):
    type: Literal["stage_ready"] = "stage_ready"
    job_id: str
    epoch: int
    stage_idx: int


class StepMetrics(BaseModel):
    type: Literal["step_metrics"] = "step_metrics"
    job_id: str
    epoch: int
    stage_idx: int
    step: int
    loss: Optional[float] = None
    # sha256 of the first microbatch's stage input/output bytes, committed at
    # step time so later verification uploads cannot be swapped.
    in_digest: str
    out_digest: str
    usage: UsageSample


class CheckpointReady(BaseModel):
    type: Literal["checkpoint_ready"] = "checkpoint_ready"
    job_id: str
    epoch: int
    stage_idx: int
    step: int
    layer_start: int
    layer_end: int
    path: str  # local to the worker; the daemon uploads it


class StageFinished(BaseModel):
    type: Literal["stage_finished"] = "stage_finished"
    job_id: str
    epoch: int
    stage_idx: int
    reason: Literal["done", "drained", "cancelled", "error"]
    last_step: int
    detail: Optional[str] = None


class VerifyBundleReady(BaseModel):
    type: Literal["verify_bundle_ready"] = "verify_bundle_ready"
    verify_id: str
    job_id: str
    stage_idx: int
    step: int
    path: Optional[str] = None  # local to the worker; None if no longer held
    error: Optional[str] = None


class VerifyResult(BaseModel):
    type: Literal["verify_result"] = "verify_result"
    verify_id: str
    kind: Literal["replay", "canary"]
    stats: dict[str, float] = {}
    output_path: Optional[str] = None  # replay output, local to the verifier
    error: Optional[str] = None


# ---------------------------------------------------------------- coordinator -> agent


class Welcome(BaseModel):
    type: Literal["welcome"] = "welcome"
    node_id: str
    heartbeat_interval_s: float
    # Secret for this registration. The agent presents it on HTTP requests
    # (X-Node-Token) so the coordinator knows which node is calling.
    node_token: Optional[str] = None


class StageAssignment(BaseModel):
    type: Literal["stage_assignment"] = "stage_assignment"
    job_id: str
    epoch: int
    stage_idx: int
    num_stages: int
    layer_start: int
    layer_end: int
    num_layers: int
    spec: LoraFinetuneSpec
    prev_peer: Optional[PeerAddr] = None
    next_peer: Optional[PeerAddr] = None
    resume_step: int = 0  # 0 = fresh start; otherwise a complete checkpoint step
    checkpoint_url: Optional[str] = None
    dataset_url: Optional[str] = None  # only for stage 0
    checkpoint_every: int
    verify_ring_size: int


class Drain(BaseModel):
    """Sent to stage 0: finish the current step, checkpoint, propagate STOP."""

    type: Literal["drain"] = "drain"
    job_id: str
    epoch: int


class CancelStage(BaseModel):
    type: Literal["cancel_stage"] = "cancel_stage"
    job_id: str
    epoch: int


class VerifyFetch(BaseModel):
    """Ask a stage to upload the input/output/adapters it used at ``step``."""

    type: Literal["verify_fetch"] = "verify_fetch"
    verify_id: str
    job_id: str
    epoch: int
    stage_idx: int
    step: int


class VerifyRequest(BaseModel):
    type: Literal["verify_request"] = "verify_request"
    verify_id: str
    kind: Literal["replay", "canary"]
    # replay
    model: Optional[str] = None
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None
    num_layers: Optional[int] = None
    lora_rank: Optional[int] = None
    lora_scale: Optional[float] = None
    lora_targets: Optional[list[str]] = None
    bundle_url: Optional[str] = None
    # canary
    seed: Optional[int] = None
    size: Optional[int] = None


AgentMessage = Annotated[
    Union[
        Register, Heartbeat, DrainNotice, StageReady, StepMetrics, CheckpointReady,
        StageFinished, VerifyBundleReady, VerifyResult,
    ],
    Field(discriminator="type"),
]

CoordinatorMessage = Annotated[
    Union[Welcome, StageAssignment, Drain, CancelStage, VerifyFetch, VerifyRequest],
    Field(discriminator="type"),
]

_agent_adapter = TypeAdapter(AgentMessage)
_coord_adapter = TypeAdapter(CoordinatorMessage)


def parse_agent_message(raw: str | bytes | dict):
    if isinstance(raw, dict):
        return _agent_adapter.validate_python(raw)
    return _agent_adapter.validate_json(raw)


def parse_coordinator_message(raw: str | bytes | dict):
    if isinstance(raw, dict):
        return _coord_adapter.validate_python(raw)
    return _coord_adapter.validate_json(raw)


def dump(msg: BaseModel) -> str:
    return msg.model_dump_json()
