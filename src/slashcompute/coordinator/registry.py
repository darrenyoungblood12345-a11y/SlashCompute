"""Live view of connected nodes."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from pydantic import BaseModel

from slashcompute.common.protocol import DeviceProfile, PeerAddr, Register

SendFn = Callable[[BaseModel], Awaitable[None]]


@dataclass
class Assignment:
    job_id: str
    epoch: int
    stage_idx: int


@dataclass
class NodeState:
    node_id: str
    name: str
    device: DeviceProfile
    data_host: str
    data_port: int
    gpu_percent: int
    send: SendFn
    last_heartbeat: float = field(default_factory=time.monotonic)
    status: str = "idle"
    draining: bool = False
    drain_deadline: Optional[float] = None
    assignment: Optional[Assignment] = None
    verifying: Optional[str] = None  # verification id
    canary_passed: Optional[bool] = None
    last_canary: float = 0.0
    user_id: Optional[str] = None
    # Issued in Welcome and presented on the node's HTTP requests. New for every
    # registration, so taking over a node id does not take over its uploads.
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)

    @property
    def peer(self) -> PeerAddr:
        return PeerAddr(node_id=self.node_id, host=self.data_host, port=self.data_port)

    @property
    def schedulable(self) -> bool:
        return (not self.draining and self.assignment is None and self.verifying is None
                and self.canary_passed is not False)


class Registry:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeState] = {}

    def register(self, msg: Register, send: SendFn) -> NodeState:
        state = NodeState(node_id=msg.node_id, name=msg.name, device=msg.device,
                          data_host=msg.data_host, data_port=msg.data_port,
                          gpu_percent=msg.gpu_percent, send=send)
        self.nodes[msg.node_id] = state
        return state

    def get(self, node_id: str) -> Optional[NodeState]:
        return self.nodes.get(node_id)

    def by_token(self, token: Optional[str]) -> Optional[NodeState]:
        if not token:
            return None
        return next((n for n in self.nodes.values()
                     if secrets.compare_digest(n.token.encode(), token.encode())), None)

    def remove(self, node_id: str) -> Optional[NodeState]:
        return self.nodes.pop(node_id, None)

    def heartbeat(self, node_id: str, status: str) -> None:
        n = self.nodes.get(node_id)
        if n is not None:
            n.last_heartbeat = time.monotonic()
            n.status = status

    def expired(self, timeout_s: float) -> list[NodeState]:
        cutoff = time.monotonic() - timeout_s
        return [n for n in self.nodes.values() if n.last_heartbeat < cutoff]

    def schedulable(self) -> list[NodeState]:
        return [n for n in self.nodes.values() if n.schedulable]
