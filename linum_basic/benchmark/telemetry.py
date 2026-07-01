"""CUDA timing and memory telemetry for the A/B benchmark harness."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "MemoryStats",
    "TelemetryRecord",
    "TimingPhase",
    "collect_memory_stats",
    "collect_multi_gpu_memory_stats",
    "collect_precision_metadata",
    "peak_single_gpu_vram_bytes",
    "run_with_phases",
    "timed_call",
]


class TimingPhase(StrEnum):
    """Labels for benchmark timing phases."""

    WARMUP = "warmup"
    COLD_CACHE = "cold_cache"
    WARM_CACHE = "warm_cache"
    STEADY_STATE = "steady_state"


@dataclass(frozen=True, slots=True, init=False)
class MemoryStats:
    """Peak GPU memory counters for one timed region.

    Attributes
    ----------
    max_memory_allocated_bytes : int
        Peak bytes allocated by the CUDA caching allocator.
    max_memory_reserved_bytes : int
        Peak bytes reserved by the CUDA caching allocator.
    """

    max_memory_allocated_bytes: int
    max_memory_reserved_bytes: int

    def __init__(self, max_memory_allocated_bytes: int, max_memory_reserved_bytes: int) -> None:
        """Initialise peak GPU memory counters.

        Parameters
        ----------
        max_memory_allocated_bytes : int
            Peak bytes allocated by the CUDA caching allocator.
        max_memory_reserved_bytes : int
            Peak bytes reserved by the CUDA caching allocator.
        """
        object.__setattr__(self, "max_memory_allocated_bytes", max_memory_allocated_bytes)
        object.__setattr__(self, "max_memory_reserved_bytes", max_memory_reserved_bytes)


@dataclass(frozen=True, slots=True, init=False)
class TelemetryRecord:
    """Timing, memory, and compile metadata for one benchmark run.

    Attributes
    ----------
    warmup_ms : float or None
        Total warmup phase duration in milliseconds.
    cold_cache_ms : float or None
        First measured repeat after warmup (cold cache).
    warm_cache_ms : float or None
        Mean of later measured repeats (warm cache).
    steady_state_ms : float or None
        Median of measured repeats (steady state).
    max_memory_allocated_bytes : int or None
        Peak allocated GPU memory when CUDA telemetry is available.
    max_memory_reserved_bytes : int or None
        Peak reserved GPU memory when CUDA telemetry is available.
    chunk_size : int or None
        Batched z-chunk size when applicable.
    compile_status : str or None
        ``torch.compile`` / Inductor status label.
    inductor_cache_path : str or None
        On-disk Inductor cache directory.
    device : str or None
        Device string used for the timed call.
    backend : str or None
        Array backend label (``numpy`` or ``torch``).
    """

    warmup_ms: float | None
    cold_cache_ms: float | None
    warm_cache_ms: float | None
    steady_state_ms: float | None
    max_memory_allocated_bytes: int | None
    max_memory_reserved_bytes: int | None
    chunk_size: int | None
    compile_status: str | None
    inductor_cache_path: str | None
    device: str | None
    backend: str | None

    def __init__(
        self,
        *,
        warmup_ms: float | None,
        cold_cache_ms: float | None,
        warm_cache_ms: float | None,
        steady_state_ms: float | None,
        max_memory_allocated_bytes: int | None,
        max_memory_reserved_bytes: int | None,
        chunk_size: int | None,
        compile_status: str | None,
        inductor_cache_path: str | None,
        device: str | None,
        backend: str | None,
    ) -> None:
        """Initialise a telemetry record."""
        object.__setattr__(self, "warmup_ms", warmup_ms)
        object.__setattr__(self, "cold_cache_ms", cold_cache_ms)
        object.__setattr__(self, "warm_cache_ms", warm_cache_ms)
        object.__setattr__(self, "steady_state_ms", steady_state_ms)
        object.__setattr__(self, "max_memory_allocated_bytes", max_memory_allocated_bytes)
        object.__setattr__(self, "max_memory_reserved_bytes", max_memory_reserved_bytes)
        object.__setattr__(self, "chunk_size", chunk_size)
        object.__setattr__(self, "compile_status", compile_status)
        object.__setattr__(self, "inductor_cache_path", inductor_cache_path)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "backend", backend)


def _use_cuda_timing(device: str | None, backend: str | None) -> bool:
    if not device or not device.lower().startswith("cuda"):
        return False
    if backend != "torch":
        return False
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _cuda_peak_stats_device(device: str | None) -> str | object | None:
    """Prepare the active CUDA device and return a peak-stats handle.

    PyTorch requires an initialized CUDA context before
    ``reset_peak_memory_stats`` accepts indexed devices such as ``cuda:0``.
    """
    if not device or not device.lower().startswith("cuda"):
        return device
    try:
        import torch
    except ImportError:
        return device
    if not torch.cuda.is_available():
        return device

    set_device = getattr(torch.cuda, "set_device", None)
    if set_device is not None:
        if device == "cuda":
            set_device(0)
        elif device.startswith("cuda:"):
            set_device(int(device.split(":", 1)[1]))

    return device


def collect_memory_stats(device: str | None) -> MemoryStats | None:
    """Return peak allocated/reserved GPU memory counters when available.

    Parameters
    ----------
    device : str or None
        PyTorch device string (for example ``"cuda:0"``).

    Returns
    -------
    MemoryStats or None
        Peak counters when CUDA is available; otherwise ``None``.
    """
    if not device or not device.lower().startswith("cuda"):
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return MemoryStats(
        max_memory_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        max_memory_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
    )


def collect_multi_gpu_memory_stats(devices: Sequence[str]) -> dict[str, MemoryStats]:
    """Return peak memory counters for each CUDA device string.

    Parameters
    ----------
    devices : sequence of str
        PyTorch device strings (for example ``["cuda:0", "cuda:1"]``).

    Returns
    -------
    dict[str, MemoryStats]
        Per-device peak counters when CUDA is available; otherwise ``{}``.
    """
    if not devices:
        return {}
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}

    stats: dict[str, MemoryStats] = {}
    for device in devices:
        if not device.lower().startswith("cuda"):
            continue
        if device == "cuda":
            index = 0
        elif device.startswith("cuda:"):
            index = int(device.split(":", 1)[1])
        else:
            continue
        stats[device] = MemoryStats(
            max_memory_allocated_bytes=int(torch.cuda.max_memory_allocated(index)),
            max_memory_reserved_bytes=int(torch.cuda.max_memory_reserved(index)),
        )
    return stats


def peak_single_gpu_vram_bytes(stats_by_device: dict[str, MemoryStats]) -> int:
    """Return the highest peak allocated VRAM across visible devices (D-16 tie-break).

    Parameters
    ----------
    stats_by_device : dict[str, MemoryStats]
        Output of :func:`collect_multi_gpu_memory_stats`.

    Returns
    -------
    int
        Maximum ``max_memory_allocated_bytes`` across devices, or ``0`` when empty.
    """
    if not stats_by_device:
        return 0
    return max(entry.max_memory_allocated_bytes for entry in stats_by_device.values())


def collect_precision_metadata(*, compile_fallback_reason: str | None = None) -> dict[str, Any]:
    """Capture TF32 and torch.compile precision settings for artifact metadata.

    Parameters
    ----------
    compile_fallback_reason : str or None
        Optional reason when ``torch.compile`` fell back to eager mode.

    Returns
    -------
    dict
        Keys ``float32_matmul_precision``, ``allow_tf32_matmul``,
        ``allow_tf32_cudnn``, ``compile_requested``, and
        ``compile_fallback_reason``.  When PyTorch is unavailable, TF32 fields
        are ``None`` and ``compile_requested`` is ``False``.
    """
    unavailable: dict[str, Any] = {
        "float32_matmul_precision": None,
        "allow_tf32_matmul": None,
        "allow_tf32_cudnn": None,
        "compile_requested": False,
        "compile_fallback_reason": compile_fallback_reason,
    }
    try:
        import torch
    except ImportError:
        return unavailable

    float32_matmul_precision: str | None = None
    get_precision = getattr(torch, "get_float32_matmul_precision", None)
    if callable(get_precision):
        float32_matmul_precision = str(get_precision())

    allow_tf32_matmul: bool | None = None
    allow_tf32_cudnn: bool | None = None
    cuda_backends = getattr(getattr(torch, "backends", None), "cuda", None)
    if cuda_backends is not None:
        matmul = getattr(cuda_backends, "matmul", None)
        if matmul is not None:
            allow_tf32_matmul = bool(getattr(matmul, "allow_tf32", None))
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        allow_tf32_cudnn = bool(getattr(cudnn, "allow_tf32", None))

    compile_requested = hasattr(torch, "compile")
    return {
        "float32_matmul_precision": float32_matmul_precision,
        "allow_tf32_matmul": allow_tf32_matmul,
        "allow_tf32_cudnn": allow_tf32_cudnn,
        "compile_requested": compile_requested,
        "compile_fallback_reason": compile_fallback_reason,
    }


def timed_call[T](
    fn: Callable[[], T],
    *,
    device: str | None,
    backend: str | None,
) -> tuple[float, T]:
    """Run *fn* under CUDA events or wall-clock timing and return elapsed ms.

    Parameters
    ----------
    fn : callable
        Zero-argument callable to time.
    device : str or None
        Device string; CUDA events are used when this starts with ``"cuda"``.
    backend : str or None
        Array backend label; CUDA events require ``"torch"``.

    Returns
    -------
    elapsed_ms : float
        Elapsed time in milliseconds.
    result : T
        Return value of *fn*.
    """
    if _use_cuda_timing(device, backend):
        import torch

        torch.cuda.reset_peak_memory_stats(_cuda_peak_stats_device(device))
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize(device)
        elapsed_ms = float(start.elapsed_time(end))
        return elapsed_ms, result

    t0 = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms, result


def run_with_phases[T](
    fn: Callable[[], T],
    *,
    device: str | None,
    backend: str | None,
    repeats: int = 3,
    warmup: int = 1,
    chunk_size: int | None = None,
    compile_status: str | None = None,
    inductor_cache_path: str | None = None,
) -> TelemetryRecord:
    """Execute warmup and measured repeats with explicit phase labels.

    Parameters
    ----------
    fn : callable
        Zero-argument callable to benchmark.
    device : str or None
        Device string forwarded to :func:`timed_call`.
    backend : str or None
        Backend label forwarded to :func:`timed_call`.
    repeats : int
        Number of measured repeats after warmup (default 3).
    warmup : int
        Number of untimed warmup iterations (default 1).
    chunk_size : int or None
        Optional batched z-chunk size recorded on the result.
    compile_status : str or None
        Optional ``torch.compile`` status label.
    inductor_cache_path : str or None
        Optional Inductor cache directory path.

    Returns
    -------
    TelemetryRecord
        Phase timings, memory peaks, and compile metadata.
    """
    if repeats < 1:
        msg = f"repeats must be >= 1, got {repeats}"
        raise ValueError(msg)
    if warmup < 0:
        msg = f"warmup must be >= 0, got {warmup}"
        raise ValueError(msg)

    if _use_cuda_timing(device, backend):
        import torch

        torch.cuda.reset_peak_memory_stats(_cuda_peak_stats_device(device))
        torch.cuda.synchronize(device)

    warmup_ms: float | None = None
    if warmup > 0:
        warmup_total = 0.0
        for _ in range(warmup):
            elapsed, _ = timed_call(fn, device=device, backend=backend)
            warmup_total += elapsed
        warmup_ms = warmup_total

    measured_ms: list[float] = []
    for _ in range(repeats):
        elapsed, _ = timed_call(fn, device=device, backend=backend)
        measured_ms.append(elapsed)

    cold_cache_ms = measured_ms[0]
    warm_cache_ms = float(statistics.mean(measured_ms[1:])) if len(measured_ms) > 1 else None

    steady_state_ms = float(statistics.median(measured_ms))

    memory = collect_memory_stats(device)
    return TelemetryRecord(
        warmup_ms=warmup_ms,
        cold_cache_ms=cold_cache_ms,
        warm_cache_ms=warm_cache_ms,
        steady_state_ms=steady_state_ms,
        max_memory_allocated_bytes=memory.max_memory_allocated_bytes if memory else None,
        max_memory_reserved_bytes=memory.max_memory_reserved_bytes if memory else None,
        chunk_size=chunk_size,
        compile_status=compile_status,
        inductor_cache_path=inductor_cache_path,
        device=device,
        backend=backend,
    )
