"""Reproducibility metadata collection for benchmark runs."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from linum_basic._parallel import list_cuda_devices
from linum_basic.benchmark.telemetry import collect_precision_metadata

__all__ = [
    "RunMetadata",
    "collect_compile_cache_info",
    "collect_concurrency_metadata",
    "collect_git_commit",
    "collect_host_info",
    "collect_precision_metadata",
    "collect_run_metadata",
    "collect_torch_cuda_info",
]


@dataclass(frozen=True, slots=True, init=False)
class RunMetadata:
    """Full provenance metadata for one benchmark run.

    Attributes
    ----------
    git_commit : str
        Short git commit hash or ``"unknown"``.
    host_node : str
        Hostname from :func:`platform.node`.
    platform : str
        Platform string from :func:`platform.platform`.
    python_version : str
        Interpreter version string.
    numpy_version : str
        NumPy version string.
    torch_version : str
        PyTorch version or ``"unavailable"``.
    cuda_version : str or None
        CUDA toolkit version when torch is importable.
    cuda_available : bool
        Whether CUDA is available to PyTorch.
    cuda_devices : list of str
        Visible CUDA device strings.
    strategy_params : dict
        Resolved fit strategy parameters.
    z_indices : list of int
        Selected z-planes for the run.
    cache_mode : str
        Inductor cache mode label.
    inductor_cache_path : str or None
        On-disk Inductor cache directory when set.
    fx_graph_cache_enabled : bool or None
        Whether ``TORCHINDUCTOR_FX_GRAPH_CACHE`` is enabled.
    timestamp : str
        UTC ISO-8601 run timestamp.
    """

    git_commit: str
    host_node: str
    platform: str
    python_version: str
    numpy_version: str
    torch_version: str
    cuda_version: str | None
    cuda_available: bool
    cuda_devices: list[str]
    strategy_params: dict[str, Any]
    z_indices: list[int]
    cache_mode: str
    inductor_cache_path: str | None
    fx_graph_cache_enabled: bool | None
    timestamp: str

    def __init__(
        self,
        *,
        git_commit: str,
        host_node: str,
        platform: str,
        python_version: str,
        numpy_version: str,
        torch_version: str,
        cuda_version: str | None,
        cuda_available: bool,
        cuda_devices: list[str],
        strategy_params: dict[str, Any],
        z_indices: list[int],
        cache_mode: str,
        inductor_cache_path: str | None,
        fx_graph_cache_enabled: bool | None,
        timestamp: str,
    ) -> None:
        """Initialise run provenance metadata."""
        object.__setattr__(self, "git_commit", git_commit)
        object.__setattr__(self, "host_node", host_node)
        object.__setattr__(self, "platform", platform)
        object.__setattr__(self, "python_version", python_version)
        object.__setattr__(self, "numpy_version", numpy_version)
        object.__setattr__(self, "torch_version", torch_version)
        object.__setattr__(self, "cuda_version", cuda_version)
        object.__setattr__(self, "cuda_available", cuda_available)
        object.__setattr__(self, "cuda_devices", cuda_devices)
        object.__setattr__(self, "strategy_params", strategy_params)
        object.__setattr__(self, "z_indices", z_indices)
        object.__setattr__(self, "cache_mode", cache_mode)
        object.__setattr__(self, "inductor_cache_path", inductor_cache_path)
        object.__setattr__(self, "fx_graph_cache_enabled", fx_graph_cache_enabled)
        object.__setattr__(self, "timestamp", timestamp)


def collect_git_commit() -> str:
    r"""Return the short git commit hash, or ``"unknown"`` when unavailable.

    Returns
    -------
    str
        Short commit hash from ``git rev-parse --short HEAD``.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def collect_host_info() -> dict[str, str]:
    """Collect host platform and interpreter versions.

    Returns
    -------
    dict of str
        Keys ``host_node``, ``platform``, ``python_version``, ``numpy_version``.
    """
    return {
        "host_node": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    }


def collect_torch_cuda_info() -> dict[str, Any]:
    """Collect PyTorch and CUDA availability without requiring a GPU.

    Returns
    -------
    dict
        Keys ``torch_version``, ``cuda_version``, ``available``, ``devices``.
    """
    try:
        import torch
    except ImportError, RuntimeError:
        return {
            "torch_version": "unavailable",
            "cuda_version": None,
            "available": False,
            "devices": [],
        }

    cuda_available = bool(torch.cuda.is_available())
    devices = list_cuda_devices("cuda") if cuda_available else []
    cuda_version = getattr(torch.version, "cuda", None)
    return {
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "available": cuda_available,
        "devices": devices,
    }


def collect_compile_cache_info(*, configure: bool = False) -> dict[str, Any]:
    """Report Inductor cache path and FX graph cache status.

    Parameters
    ----------
    configure : bool
        When ``True``, call :func:`linum_basic._torch_cache.configure_torch_inductor_cache`
        before reading environment variables.

    Returns
    -------
    dict
        Keys ``inductor_cache_path`` and ``fx_graph_cache_enabled``.
    """
    if configure:
        from linum_basic._torch_cache import configure_torch_inductor_cache, enable_fx_graph_cache

        cache_path = str(configure_torch_inductor_cache())
        enable_fx_graph_cache()
    else:
        cache_path = os.environ.get("TORCHINDUCTOR_CACHE_DIR")

    fx_env = os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE")
    fx_enabled = None if fx_env is None else fx_env == "1"

    return {
        "inductor_cache_path": cache_path,
        "fx_graph_cache_enabled": fx_enabled,
    }


_STRATEGY_FORK_MODELS: dict[str, str] = {
    "multi": "maxForks_2_scalar_per_gpu",
    "batched": "maxForks_1_batched_multi_gpu",
}


def collect_concurrency_metadata(
    *,
    strategy: str,
    fork_model: str | None,
    n_gpus: int,
    gpu_map: dict[str, str],
    inductor_cache_path: str | None,
    compile_status: str | None = None,
    quality_verdict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the D-15 concurrency metadata block for a benchmark candidate.

    Parameters
    ----------
    strategy : str
        Concurrency strategy label (``multi`` or ``batched``).
    fork_model : str or None
        Explicit fork-model label; derived from ``strategy`` when omitted.
    n_gpus : int
        Number of GPUs used for the run.
    gpu_map : dict[str, str]
        Mapping from device strings to worker or allocation labels.
    inductor_cache_path : str or None
        Shared Inductor cache directory; resolved from the environment when omitted.
    compile_status : str or None, optional
        ``torch.compile`` / Inductor status label.
    quality_verdict : dict or None, optional
        Quality gate verdict payload for the mode.

    Returns
    -------
    dict
        JSON-serializable D-15 concurrency metadata block.
    """
    resolved_fork_model = fork_model if fork_model is not None else _STRATEGY_FORK_MODELS.get(strategy, strategy)
    resolved_cache_path = inductor_cache_path
    if resolved_cache_path is None:
        resolved_cache_path = collect_compile_cache_info(configure=False)["inductor_cache_path"]

    return {
        "strategy": strategy,
        "fork_model": resolved_fork_model,
        "n_gpus": n_gpus,
        "gpu_map": dict(gpu_map),
        "inductor_cache_path": resolved_cache_path,
        "compile_status": compile_status,
        "quality_verdict": quality_verdict,
    }


def collect_run_metadata(
    *,
    strategy_params: dict[str, Any],
    z_indices: list[int],
    cache_mode: str,
    configure_cache: bool = False,
) -> RunMetadata:
    """Assemble full reproducibility metadata for a benchmark run.

    Parameters
    ----------
    strategy_params : dict
        Resolved fit strategy parameters for the run.
    z_indices : list of int
        Selected z-planes included in the benchmark.
    cache_mode : str
        Inductor cache mode label (for example ``"persistent"``).
    configure_cache : bool
        When ``True``, configure the Inductor cache before reading paths.

    Returns
    -------
    RunMetadata
        Full provenance record for artifact serialization.
    """
    host = collect_host_info()
    torch_info = collect_torch_cuda_info()
    cache_info = collect_compile_cache_info(configure=configure_cache)
    timestamp = datetime.now(UTC).isoformat()

    return RunMetadata(
        git_commit=collect_git_commit(),
        host_node=host["host_node"],
        platform=host["platform"],
        python_version=host["python_version"],
        numpy_version=host["numpy_version"],
        torch_version=str(torch_info["torch_version"]),
        cuda_version=torch_info["cuda_version"],
        cuda_available=bool(torch_info["available"]),
        cuda_devices=list(torch_info["devices"]),
        strategy_params=dict(strategy_params),
        z_indices=list(z_indices),
        cache_mode=cache_mode,
        inductor_cache_path=cache_info["inductor_cache_path"],
        fx_graph_cache_enabled=cache_info["fx_graph_cache_enabled"],
        timestamp=timestamp,
    )
