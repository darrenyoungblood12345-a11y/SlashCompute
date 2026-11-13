"""Per-layer size and parameter counts for a model, without loading weights.

Reads ``config.json`` plus safetensors headers (local files, or HTTP range
requests for an uncached Hugging Face repo). Used by the partitioner to size
stages and by metering to estimate FLOPs.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8,
                "I8": 1, "U8": 1, "BOOL": 1, "I16": 2, "U16": 2}
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class ModelProfile:
    model: str
    num_layers: int
    hidden_size: int
    vocab_size: int
    tie_word_embeddings: bool
    layer_bytes: tuple[int, ...]
    embed_bytes: int
    head_bytes: int  # includes the final norm; equals embed when tied
    layer_params: int  # logical (unquantized) parameter count per layer
    head_params: int

    def stage_weight_bytes(self, start: int, end: int, num_layers: int | None = None) -> int:
        n = num_layers or self.num_layers
        total = sum(self.layer_bytes[start:end])
        if start == 0:
            total += self.embed_bytes
        if end == n:
            total += self.head_bytes
            if start == 0 and self.tie_word_embeddings:
                # one stage owns both ends and holds the tied embedding once,
                # but embed_bytes and head_bytes each include it
                total -= self.embed_bytes
        return total

    @property
    def total_weight_bytes(self) -> int:
        return self.stage_weight_bytes(0, self.num_layers)


def resolve_model_path(model: str, download: bool = True) -> Path:
    """Local directory for ``model`` (a path or an HF repo id)."""
    p = Path(model).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.safetensors", "*.py", "tokenizer.model", "*.tiktoken", "*.txt",
                "*.jinja"]
    if download:
        return Path(snapshot_download(model, allow_patterns=patterns))
    return Path(snapshot_download(model, allow_patterns=patterns, local_files_only=True))


def _local_headers(path: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(path.glob("model*.safetensors")):
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        header.pop("__metadata__", None)
        out.update(header)
    return out


def _remote(model: str) -> tuple[dict, dict[str, dict]]:
    from huggingface_hub import get_safetensors_metadata, hf_hub_download

    config = json.loads(Path(hf_hub_download(model, "config.json")).read_text())
    meta = get_safetensors_metadata(model)
    headers = {}
    for fm in meta.files_metadata.values():
        for name, info in fm.tensors.items():
            headers[name] = {"dtype": info.dtype, "shape": list(info.shape),
                             "data_offsets": [0, info.data_offsets[1] - info.data_offsets[0]]}
    return config, headers


def _logical_layer_params(cfg: dict) -> int:
    h = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    head_dim = cfg.get("head_dim") or h // n_heads
    inter = cfg["intermediate_size"]
    attn = h * n_heads * head_dim * 2 + h * n_kv * head_dim * 2
    mlp = 3 * h * inter
    return attn + mlp


def profile_from(model: str, config: dict, headers: dict[str, dict]) -> ModelProfile:
    cfg = config.get("text_config", config) | {k: v for k, v in config.items() if k != "text_config"}
    n = cfg["num_hidden_layers"]
    layer_bytes = [0] * n
    embed = head = 0
    for name, info in headers.items():
        size = info["data_offsets"][1] - info["data_offsets"][0]
        m = _LAYER_RE.search(name)
        if m:
            layer_bytes[int(m.group(1))] += size
        elif "embed_tokens" in name:
            embed += size
        else:  # final norm, lm_head
            head += size
    tied = bool(cfg.get("tie_word_embeddings", False))
    if tied:
        head += embed
    return ModelProfile(
        model=model, num_layers=n, hidden_size=cfg["hidden_size"], vocab_size=cfg["vocab_size"],
        tie_word_embeddings=tied, layer_bytes=tuple(layer_bytes), embed_bytes=embed,
        head_bytes=head, layer_params=_logical_layer_params(cfg),
        head_params=cfg["hidden_size"] * cfg["vocab_size"],
    )


@lru_cache(maxsize=32)
def profile_model(model: str) -> ModelProfile:
    p = Path(model).expanduser()
    if not p.is_dir():
        try:
            p = resolve_model_path(model, download=False)
        except Exception:
            config, headers = _remote(model)
            return profile_from(model, config, headers)
    config = json.loads((p / "config.json").read_text())
    return profile_from(model, config, _local_headers(p))
