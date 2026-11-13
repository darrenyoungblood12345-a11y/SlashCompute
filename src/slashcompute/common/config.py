"""Engine-wide defaults. Every value can be overridden with an environment
variable named ``SLASHCOMPUTE_<FIELD>`` (upper-case)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

DEV_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# Ordered smallest to largest. The planner picks the largest one that fits the
# pool but not any single node.
DEMO_MODEL_CANDIDATES = [
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "mlx-community/Qwen2.5-32B-Instruct-4bit",
]

# A build that runs beside another /compute on one Mac advertises under its own service, so the
# two never list or join each other's pools (set by the app bundle; see scripts/macos/build_dmg.sh).
MDNS_SERVICE_TYPE = os.environ.get("SLASHCOMPUTE_MDNS_SERVICE") or "_slashcompute._tcp.local."


@dataclass
class EngineConfig:
    home: Path = field(default_factory=lambda: Path.home() / ".slashcompute")
    coordinator_host: str = "0.0.0.0"
    coordinator_port: int = 8765
    data_port: int = 9700

    checkpoint_every: int = 25
    grace_period_s: float = 60.0
    heartbeat_interval_s: float = 5.0
    heartbeat_timeout_s: float = 20.0
    scheduler_tick_s: float = 1.0
    stage_start_timeout_s: float = 900.0  # includes model download on first use
    max_recoveries: int = 20
    canary_size: int = 512
    canary_interval_s: float = 1800.0

    verify_rate: float = 0.05
    verify_rel_tolerance: float = 2e-2
    # GPU fp32 vs NumPy reference; zeros/random still fail (~1.0).
    canary_rel_tolerance: float = 0.5
    verify_ring_size: int = 8

    # Fraction of each stage's weight bytes reserved for activations, grads and
    # optimizer state, plus a fixed headroom per stage.
    stage_overhead_frac: float = 0.25
    stage_overhead_bytes: int = 512 * 1024**2

    sandbox: bool = True

    @classmethod
    def from_env(cls, **overrides) -> "EngineConfig":
        cfg = cls()
        for f in fields(cls):
            env = os.environ.get(f"SLASHCOMPUTE_{f.name.upper()}")
            if env is None:
                continue
            cur = getattr(cfg, f.name)
            if isinstance(cur, bool):
                val = env.lower() in ("1", "true", "yes", "on")
            elif isinstance(cur, Path):
                val = Path(env).expanduser()
            else:
                val = type(cur)(env)
            setattr(cfg, f.name, val)
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    @property
    def coordinator_dir(self) -> Path:
        return self.home / "coordinator"

    @property
    def agent_dir(self) -> Path:
        return self.home / "agent"
