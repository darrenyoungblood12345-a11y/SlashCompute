"""One-shot device profile for registration."""

from __future__ import annotations

import platform
import subprocess
import time

import psutil

from slashcompute.common.protocol import DeviceProfile

MIN_CONTRIB_BYTES = 64 * 1024 * 1024


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.check_output(["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL)
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def chip_name() -> str:
    return _sysctl("machdep.cpu.brand_string") or _sysctl("hw.model") or platform.processor() \
        or platform.machine() or "unknown"


def _memory() -> tuple[int, int]:
    vm = psutil.virtual_memory()
    return int(vm.total), int(vm.available)


def _matmul_tflops(n: int = 1024, repeats: int = 3) -> float:
    import mlx.core as mx

    a = mx.random.normal((n, n))
    b = mx.random.normal((n, n))
    mx.eval(a, b, a @ b)  # warmup + compile
    t0 = time.perf_counter()
    for _ in range(repeats):
        mx.eval(a @ b)
    dt = time.perf_counter() - t0
    return (repeats * 2.0 * n * n * n) / max(dt, 1e-9) / 1e12


def _mem_bandwidth_gbps(nbytes: int = 64 * 1024 * 1024) -> float:
    import mlx.core as mx

    n = nbytes // 4
    x = mx.random.normal((n,))
    mx.eval(x, x + 1)
    t0 = time.perf_counter()
    mx.eval(x + 1)
    dt = time.perf_counter() - t0
    # read x + write result
    return (2.0 * nbytes) / max(dt, 1e-9) / 1e9


def contrib_bytes(available: int, max_memory_bytes: int | None = None) -> int:
    """Memory to lend: what is available (at least the floor), within the user's cap."""
    contrib = max(available, MIN_CONTRIB_BYTES)
    if max_memory_bytes is not None:
        # the cap is the upper bound: apply it last so the floor cannot exceed it
        contrib = min(contrib, max(int(max_memory_bytes), 0))
    return contrib


def benchmark(max_memory_bytes: int | None = None) -> DeviceProfile:
    total, available = _memory()
    working = max(int(total * 0.75), available)
    contrib = contrib_bytes(available, max_memory_bytes)
    return DeviceProfile(
        chip=chip_name(),
        memory_total_bytes=total,
        memory_available_bytes=available,
        working_set_bytes=working,
        memory_contrib_bytes=contrib,
        matmul_tflops=_matmul_tflops(),
        mem_bandwidth_gbps=_mem_bandwidth_gbps(),
    )
