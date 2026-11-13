"""Known-answer canary task: a seeded fp32 matmul. The coordinator computes
the expected statistics on CPU in float64; the node computes them on its GPU.
A node that skips the work or returns made-up numbers fails."""

from __future__ import annotations

import math

import numpy as np


def canary_inputs(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n), dtype=np.float32)
    b = rng.standard_normal((n, n), dtype=np.float32)
    return a, b


def canary_stats(c: np.ndarray) -> dict[str, float]:
    c64 = c.astype(np.float64)
    return {
        "sum": float(c64.sum()),
        "abs_sum": float(np.abs(c64).sum()),
        "fro": float(np.linalg.norm(c64)),
        "trace": float(np.trace(c64)),
        "c00": float(c64[0, 0]),
        "c_last": float(c64[-1, -1]),
        "row0_dot": float(c64[0] @ np.arange(c64.shape[1], dtype=np.float64)),
    }


def expected_stats(seed: int, n: int) -> dict[str, float]:
    # Match the agent's fp32 matmul (Metal/MLX). float64 reference drifts
    # enough on large n to fail the default 1e-3 relative check.
    a, b = canary_inputs(seed, n)
    return canary_stats(a @ b)


def run_canary_mlx(seed: int, n: int) -> dict[str, float]:
    import mlx.core as mx

    a, b = canary_inputs(seed, n)
    c = mx.matmul(mx.array(a), mx.array(b))
    mx.eval(c)
    return canary_stats(np.asarray(c))


def compare_stats(got: dict[str, float], expected: dict[str, float], rel_tol: float) -> tuple[bool, float]:
    """Errors are measured relative to the Frobenius norm, which is stable
    even when a statistic (like the sum) is close to zero."""
    scale = max(abs(expected["fro"]), 1e-12)
    worst = 0.0
    for k, v in expected.items():
        # max() silently drops NaN, so a non-finite value must fail outright.
        if k not in got or not math.isfinite(got[k]):
            return False, float("inf")
        worst = max(worst, abs(got[k] - v) / scale)
    return worst <= rel_tol, worst
