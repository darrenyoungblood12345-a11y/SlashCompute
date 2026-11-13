"""Checkpoint storage. Stages upload per-stage files; once every stage of an
epoch has uploaded a step, they are merged into one file keyed per layer, so
a resumed job can be re-partitioned freely."""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Optional

import mlx.core as mx

from slashcompute.jobs import LoraFinetuneSpec
from slashcompute.pipeline.stage import merge_checkpoints

log = logging.getLogger(__name__)

_ADAPTER_KEY = re.compile(r"^adapter/layers\.L(\d+)\.(.+)$")


class CheckpointStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def job_dir(self, job_id: str) -> Path:
        d = self.root / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def stage_path(self, job_id: str, epoch: int, step: int, stage_idx: int) -> Path:
        d = self.job_dir(job_id) / "ckpt" / f"epoch{epoch}" / f"step{step:06d}"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"stage{stage_idx}.safetensors"

    def merged_path(self, job_id: str, step: int) -> Path:
        return self.job_dir(job_id) / "ckpt" / f"step_{step:06d}.safetensors"

    def store_stage(self, job_id: str, epoch: int, step: int, stage_idx: int, data: bytes,
                    num_stages: int) -> Optional[Path]:
        """Save one stage's upload. Returns the merged path once the step is
        complete for this epoch, else None."""
        self.stage_path(job_id, epoch, step, stage_idx).write_bytes(data)
        parts = [self.stage_path(job_id, epoch, step, i) for i in range(num_stages)]
        if not all(p.exists() and p.stat().st_size > 0 for p in parts):
            return None
        out = self.merged_path(job_id, step)
        merge_checkpoints(parts, out)
        log.info("checkpoint complete job=%s step=%d", job_id, step)
        return out

    def export_adapter(self, job_id: str, step: int, spec: LoraFinetuneSpec, num_layers: int) -> Path:
        """Write an mlx-lm compatible adapter directory (usable with
        ``mlx_lm.generate --adapter-path``)."""
        tensors = mx.load(str(self.merged_path(job_id, step)))
        out = {}
        for k, v in tensors.items():
            m = _ADAPTER_KEY.match(k)
            if m:
                out[f"model.layers.{m.group(1)}.{m.group(2)}"] = v
        d = self.job_dir(job_id) / "adapter"
        # Build next to the final directory and rename it into place, so a
        # failed export never leaves a half-written adapter to be served.
        tmp = d.with_name("adapter.partial")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        try:
            mx.save_safetensors(str(tmp / "adapters.safetensors"), out)
            (tmp / "adapter_config.json").write_text(json.dumps({
                "fine_tune_type": "lora",
                "model": spec.model,
                "num_layers": num_layers,
                "lora_parameters": {"rank": spec.lora_rank, "scale": spec.lora_scale,
                                    "dropout": 0.0, "keys": spec.lora_targets},
            }, indent=2))
            shutil.rmtree(d, ignore_errors=True)
            tmp.rename(d)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return d
