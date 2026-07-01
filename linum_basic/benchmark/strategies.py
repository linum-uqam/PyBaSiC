"""Strategy resolution for the A/B benchmark harness and auto fit policy.

Static policy constants cite Phase 2/3 UAT evidence (``evidence_policy`` /
``baseline_reference_id``) without runtime artifact reads — see
``EVIDENCE_POLICY_ID`` and ``BASELINE_REFERENCE_ID``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from linum_basic._parallel import list_cuda_devices

BUILTIN_STRATEGIES: tuple[str, ...] = ("baseline", "sequential", "multi", "batched")

# Phase 2/3 signed UAT policy (D-11, D-12) — static strings, no artifact I/O.
EVIDENCE_POLICY_ID = "phase2-uat-20260630"
BASELINE_REFERENCE_ID = "baseline-20260630T163351-e7c47a4-sub-22"
DEFAULT_PRODUCTION_WORKING_SIZE = 128
BATCHED_CUDA_WS_GUARD = 128

OVERRIDE_SOURCES: tuple[str, ...] = (
    "auto_resolver",
    "user_kwargs",
    "env",
    "strategy_lock",
)

# Machine-readable reason codes (D-20).
REASON_AUTO_WS128_SINGLE_GPU_SEQUENTIAL = "AUTO_WS128_SINGLE_GPU_SEQUENTIAL"
REASON_AUTO_WS128_MULTI_GPU_FANOUT = "AUTO_WS128_MULTI_GPU_FANOUT"
REASON_POLICY_PHASE2_WS128_MANDATORY = "POLICY_PHASE2_WS128_MANDATORY"
REASON_AUTO_WS64_SINGLE_Z_SEQUENTIAL = "AUTO_WS64_SINGLE_Z_SEQUENTIAL"
REASON_AUTO_WS64_BATCHED_CUDA = "AUTO_WS64_BATCHED_CUDA"
REASON_AUTO_WS64_BATCHED_CUDA_MULTI_GPU = "AUTO_WS64_BATCHED_CUDA_MULTI_GPU"
REASON_CUDA_UNAVAILABLE = "CUDA_UNAVAILABLE"
REASON_OVERRIDE_USER_KWARGS = "OVERRIDE_USER_KWARGS"
REASON_OVERRIDE_ENV = "OVERRIDE_ENV"
REASON_GUARD_WS128_BLOCKED_BATCHED = "GUARD_WS128_BLOCKED_BATCHED"

ALLOWED_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "working_size",
        "estimate_darkfield",
        "max_reweighting_iterations",
        "l_s",
        "l_d",
        "epsilon",
        "batched_z_chunk_size",
        "warm_start_reweighting",
        "device",
        "backend",
        "force_batched_cuda",
        "reweighting_tolerance",
        "convergence_check_every",
        "tile_subsample_ratio",
    }
)

__all__ = [
    "ALLOWED_OVERRIDE_KEYS",
    "BASELINE_REFERENCE_ID",
    "BATCHED_CUDA_WS_GUARD",
    "BUILTIN_STRATEGIES",
    "DEFAULT_PRODUCTION_WORKING_SIZE",
    "EVIDENCE_POLICY_ID",
    "OVERRIDE_SOURCES",
    "REASON_AUTO_WS64_BATCHED_CUDA",
    "REASON_AUTO_WS64_BATCHED_CUDA_MULTI_GPU",
    "REASON_AUTO_WS64_SINGLE_Z_SEQUENTIAL",
    "REASON_AUTO_WS128_MULTI_GPU_FANOUT",
    "REASON_AUTO_WS128_SINGLE_GPU_SEQUENTIAL",
    "REASON_CUDA_UNAVAILABLE",
    "REASON_GUARD_WS128_BLOCKED_BATCHED",
    "REASON_OVERRIDE_ENV",
    "REASON_OVERRIDE_USER_KWARGS",
    "REASON_POLICY_PHASE2_WS128_MANDATORY",
    "AutoStrategyResult",
    "StrategyResult",
    "WorkloadContext",
    "build_workload_context",
    "estimate_strategy_vram_bytes",
    "load_overrides",
    "resolve_auto_strategy",
    "resolve_strategy",
]


@dataclass(frozen=True, slots=True, init=False)
class StrategyResult:
    """Resolved fit strategy for a benchmark run.

    Attributes
    ----------
    name : str
        Strategy name (one of ``BUILTIN_STRATEGIES``).
    basic_kwargs : dict
        Keyword arguments passed to :func:`linum_basic.fit.fit_mosaic`.
    batched_z_chunk_size : int or None
        Optional z chunk size for batched CUDA mode.
    force_batched : bool
        Whether to force the batched CUDA path.
    release_gate : bool
        ``True`` for real-subject runs eligible for release decisions.
    label : str
        Human-readable strategy label.
    """

    name: str
    basic_kwargs: dict[str, Any]
    batched_z_chunk_size: int | None
    force_batched: bool
    release_gate: bool
    label: str

    def __init__(
        self,
        name: str,
        basic_kwargs: dict[str, Any],
        batched_z_chunk_size: int | None,
        force_batched: bool,
        release_gate: bool,
        label: str,
    ) -> None:
        """Initialise a resolved strategy result.

        Parameters
        ----------
        name : str
            Strategy name (one of ``BUILTIN_STRATEGIES``).
        basic_kwargs : dict
            Keyword arguments passed to :func:`linum_basic.fit.fit_mosaic`.
        batched_z_chunk_size : int or None
            Optional z chunk size for batched CUDA mode.
        force_batched : bool
            Whether to force the batched CUDA path.
        release_gate : bool
            ``True`` for real-subject runs eligible for release decisions.
        label : str
            Human-readable strategy label.
        """
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "basic_kwargs", basic_kwargs)
        object.__setattr__(self, "batched_z_chunk_size", batched_z_chunk_size)
        object.__setattr__(self, "force_batched", force_batched)
        object.__setattr__(self, "release_gate", release_gate)
        object.__setattr__(self, "label", label)


@dataclass(frozen=True, slots=True)
class WorkloadContext:
    """Measured mosaic shape and hardware context for auto strategy resolution.

    Attributes
    ----------
    n_z : int
        Number of z-planes to fit.
    n_tiles : int
        Tile count in the mosaic grid.
    field_mode : str
        ``"per-z"`` or ``"global"``.
    working_size : int
        BaSiC working resolution.
    n_gpus : int
        Visible CUDA device count from :func:`list_cuda_devices`.
    cuda_available : bool
        Whether CUDA execution is available.
    memory_estimate_bytes : int or None
        Heuristic VRAM estimate for explainability (metadata only).
    """

    n_z: int
    n_tiles: int
    field_mode: str
    working_size: int
    n_gpus: int
    cuda_available: bool
    memory_estimate_bytes: int | None


@dataclass(frozen=True, slots=True)
class AutoStrategyResult:
    """Resolved auto strategy with fit kwargs and explainability metadata.

    Attributes
    ----------
    name : str
        Always ``"auto"`` for the auto resolver.
    basic_kwargs : dict
        Keyword arguments merged into :func:`linum_basic.fit.fit_mosaic` params.
    strategy_metadata : dict
        Nested ``_strategy`` dict with D-19 explainability keys.
    execution_path : str
        Resolved execution path label.
    override_source : str
        One of :data:`OVERRIDE_SOURCES`.
    chunk_size : int or None
        Batched z-chunk size when applicable; ``None`` until fit finalises.
    """

    name: str
    basic_kwargs: dict[str, Any]
    strategy_metadata: dict[str, Any]
    execution_path: str
    override_source: str
    chunk_size: int | None


def estimate_strategy_vram_bytes(
    n_z_chunk: int,
    n_tiles: int,
    working_size: int,
) -> int:
    """Conservative VRAM estimate for batched sort stacks (metadata only).

    Returns ``n_z_chunk * n_tiles * working_size ** 2 * 4 * 4`` — float32
    elements with a 4x activation multiplier.  Does **not** gate resolver
    decisions (STRAT-02 explainability only).

    Parameters
    ----------
    n_z_chunk : int
        Z-planes per batched chunk.
    n_tiles : int
        Mosaic tile count.
    working_size : int
        BaSiC working resolution.

    Returns
    -------
    int
        Estimated bytes.
    """
    bytes_per_elem = 4
    sort_stack = n_z_chunk * n_tiles * working_size * working_size * bytes_per_elem
    return sort_stack * 4


def build_workload_context(
    *,
    n_z: int,
    n_tiles: int,
    field_mode: str,
    working_size: int,
    device: str | None = None,
) -> WorkloadContext:
    """Build a :class:`WorkloadContext` from shape inputs and GPU discovery.

    ``n_gpus`` and ``cuda_available`` are derived from
    :func:`linum_basic._parallel.list_cuda_devices`, which respects
    ``CUDA_VISIBLE_DEVICES``.  ``memory_estimate_bytes`` uses a provisional
    chunk of ``min(n_z, 8)`` for STRAT-02 metadata only.

    Parameters
    ----------
    n_z : int
        Number of z-planes to fit.
    n_tiles : int
        Mosaic tile count.
    field_mode : str
        ``"per-z"`` or ``"global"``.
    working_size : int
        BaSiC working resolution.
    device : str or None, optional
        Provisional device string for GPU enumeration; defaults to ``"cuda"``.

    Returns
    -------
    WorkloadContext
        Frozen context for :func:`resolve_auto_strategy`.
    """
    devices = list_cuda_devices(device or "cuda")
    cuda_available = bool(devices)
    n_gpus = len(devices)
    chunk = min(n_z, 8)
    memory = estimate_strategy_vram_bytes(chunk, n_tiles, working_size) if cuda_available else None
    return WorkloadContext(
        n_z=n_z,
        n_tiles=n_tiles,
        field_mode=field_mode,
        working_size=working_size,
        n_gpus=n_gpus,
        cuda_available=cuda_available,
        memory_estimate_bytes=memory,
    )


def resolve_auto_strategy(
    context: WorkloadContext,
    *,
    user_kwargs: dict[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> AutoStrategyResult:
    """Resolve ``strategy="auto"`` from workload shape, hardware, and overrides.

    Precedence chain (highest wins): explicit allowlisted ``user_kwargs`` →
    environment variables (``LINUM_BASIC_FORCE_BATCHED_CUDA``,
    ``LINUM_BASIC_BATCHED_Z_CHUNK_SIZE``) → ``strategy_lock`` (named
    strategies, handled outside this function) → auto heuristics.

    At ``working_size >= BATCHED_CUDA_WS_GUARD`` (128), auto **never** selects
    batched CUDA — Phase 2/3 UAT evidence (``EVIDENCE_POLICY_ID``) mandates
    sequential scalar on one GPU or multi-GPU per-z fan-out on two or more
    visible devices.  Batched at ws≥128 requires explicit override.

    Parameters
    ----------
    context : WorkloadContext
        Mosaic shape and hardware discovery inputs.
    user_kwargs : dict or None, optional
        Explicit BaSiC kwargs; allowlisted keys override auto output (D-08).
    env : mapping or None, optional
        Environment mapping for override detection; defaults to ``os.environ``.

    Returns
    -------
    AutoStrategyResult
        Resolved fit kwargs, execution path, and nested ``_strategy`` metadata.
    """
    env_map = env if env is not None else os.environ
    user = dict(user_kwargs or {})

    execution_path, auto_kwargs, reason_codes, reason_summary = _resolve_auto_base(
        context,
    )
    basic_kwargs = dict(auto_kwargs)
    env_applied = False

    if env_map.get("LINUM_BASIC_FORCE_BATCHED_CUDA", "0") == "1" and "force_batched_cuda" not in user:
        basic_kwargs["force_batched_cuda"] = True
        env_applied = True

    env_chunk = env_map.get("LINUM_BASIC_BATCHED_Z_CHUNK_SIZE")
    if env_chunk is not None and "batched_z_chunk_size" not in user:
        basic_kwargs["batched_z_chunk_size"] = int(env_chunk)
        env_applied = True

    user_override = False
    for key in ALLOWED_OVERRIDE_KEYS:
        if key in user and basic_kwargs.get(key) != user[key]:
            user_override = True
            break

    if user:
        basic_kwargs.update(user)

    if user_override:
        override_source = "user_kwargs"
        if REASON_OVERRIDE_USER_KWARGS not in reason_codes:
            reason_codes.append(REASON_OVERRIDE_USER_KWARGS)
    elif env_applied:
        override_source = "env"
        if REASON_OVERRIDE_ENV not in reason_codes:
            reason_codes.append(REASON_OVERRIDE_ENV)
    else:
        override_source = "auto_resolver"

    device = basic_kwargs.get("device")
    backend = basic_kwargs.get("backend", "torch")
    memory_bytes = context.memory_estimate_bytes
    if memory_bytes is None and context.cuda_available:
        chunk = min(context.n_z, 8)
        memory_bytes = estimate_strategy_vram_bytes(
            chunk,
            context.n_tiles,
            context.working_size,
        )

    strategy_metadata: dict[str, Any] = {
        "name": "auto",
        "execution_path": execution_path,
        "chunk_size": None,
        "device": device,
        "backend": backend,
        "working_size": context.working_size,
        "n_z": context.n_z,
        "n_tiles": context.n_tiles,
        "n_gpus": context.n_gpus,
        "memory_estimate_bytes": memory_bytes,
        "batched_cuda_guard": None,
        "attempted_execution_path": None,
        "reason_codes": list(reason_codes),
        "reason_summary": reason_summary,
        "override_source": override_source,
        "evidence_policy": EVIDENCE_POLICY_ID,
        "baseline_reference_id": BASELINE_REFERENCE_ID,
    }

    return AutoStrategyResult(
        name="auto",
        basic_kwargs=basic_kwargs,
        strategy_metadata=strategy_metadata,
        execution_path=execution_path,
        override_source=override_source,
        chunk_size=None,
    )


def _resolve_auto_base(
    context: WorkloadContext,
) -> tuple[str, dict[str, Any], list[str], str]:
    """Return auto-resolved path, kwargs, reason codes, and summary."""
    ws = context.working_size

    if not context.cuda_available:
        return (
            "sequential_scalar",
            {"backend": "numpy"},
            [REASON_CUDA_UNAVAILABLE],
            "CUDA unavailable; falling back to NumPy sequential scalar.",
        )

    basic_kwargs: dict[str, Any] = {
        "backend": "torch",
        "working_size": ws,
    }
    reason_codes: list[str] = []

    if ws >= BATCHED_CUDA_WS_GUARD:
        reason_codes.append(REASON_POLICY_PHASE2_WS128_MANDATORY)
        if context.n_gpus >= 2:
            basic_kwargs["device"] = "cuda"
            reason_codes.append(REASON_AUTO_WS128_MULTI_GPU_FANOUT)
            summary = f"{context.n_gpus} GPUs visible at ws={ws}; multi-GPU per-z fan-out per Phase 2 evidence."
            return "multi_gpu_fanout", basic_kwargs, reason_codes, summary

        basic_kwargs["device"] = "cuda:0"
        reason_codes.append(REASON_AUTO_WS128_SINGLE_GPU_SEQUENTIAL)
        summary = f"Single GPU at ws={ws}; sequential scalar on cuda:0."
        return "sequential_scalar", basic_kwargs, reason_codes, summary

    if context.n_z <= 1:
        basic_kwargs["device"] = "cuda:0"
        reason_codes.append(REASON_AUTO_WS64_SINGLE_Z_SEQUENTIAL)
        return (
            "sequential_scalar",
            basic_kwargs,
            reason_codes,
            f"Single z-plane at ws={ws}; sequential scalar.",
        )

    basic_kwargs["force_batched_cuda"] = True
    if context.n_gpus >= 2:
        basic_kwargs["device"] = "cuda"
        reason_codes.append(REASON_AUTO_WS64_BATCHED_CUDA_MULTI_GPU)
        return (
            "batched_cuda_multi_gpu",
            basic_kwargs,
            reason_codes,
            f"ws={ws} with n_z={context.n_z} and {context.n_gpus} GPUs; batched CUDA multi-GPU fan-out.",
        )

    basic_kwargs["device"] = "cuda:0"
    reason_codes.append(REASON_AUTO_WS64_BATCHED_CUDA)
    return (
        "batched_cuda",
        basic_kwargs,
        reason_codes,
        f"ws={ws} with n_z={context.n_z}; batched CUDA on single GPU.",
    )


def resolve_strategy(
    name: str,
    *,
    working_size: int,
    estimate_darkfield: bool,
    max_reweighting_iterations: int,
    batched_z_chunk_size: int | None = None,
    overrides: dict[str, Any] | None = None,
    is_synthetic: bool = False,
) -> StrategyResult:
    """Resolve a built-in strategy name to validated fit kwargs.

    Parameters
    ----------
    name : str
        One of ``BUILTIN_STRATEGIES``.
    working_size : int
        BaSiC working resolution.
    estimate_darkfield : bool
        Whether to estimate a dark-field.
    max_reweighting_iterations : int
        ALM reweighting iteration cap.
    batched_z_chunk_size : int or None, optional
        Optional z chunk size for batched mode.
    overrides : dict or None, optional
        Allowlisted override keys merged onto base kwargs.
    is_synthetic : bool, optional
        When ``True``, marks the run as smoke-only (``release_gate=False``).

    Returns
    -------
    StrategyResult
        Resolved strategy with fit kwargs and release-gate flag.

    Raises
    ------
    ValueError
        If *name* is not a built-in strategy.
    """
    if name not in BUILTIN_STRATEGIES:
        msg = f"Unknown strategy: {name}; choose from {BUILTIN_STRATEGIES}"
        raise ValueError(msg)

    basic_kwargs: dict[str, Any] = {
        "backend": "torch",
        "working_size": working_size,
        "estimate_darkfield": estimate_darkfield,
        "max_reweighting_iterations": max_reweighting_iterations,
    }

    force_batched = False
    chunk_size = batched_z_chunk_size

    if name in ("baseline", "sequential"):
        basic_kwargs["device"] = "cuda:0"
        basic_kwargs["warm_start_reweighting"] = False
    elif name == "multi":
        basic_kwargs["device"] = "cuda"
        basic_kwargs["warm_start_reweighting"] = True
    elif name == "batched":
        basic_kwargs["device"] = "cuda"
        basic_kwargs["warm_start_reweighting"] = True
        basic_kwargs["force_batched_cuda"] = True
        force_batched = True
        if chunk_size is not None:
            basic_kwargs["batched_z_chunk_size"] = chunk_size

    if overrides:
        basic_kwargs.update(overrides)
        if "batched_z_chunk_size" in overrides:
            chunk_size = overrides["batched_z_chunk_size"]

    release_gate = not is_synthetic
    return StrategyResult(
        name=name,
        basic_kwargs=basic_kwargs,
        batched_z_chunk_size=chunk_size,
        force_batched=force_batched,
        release_gate=release_gate,
        label=name,
    )


def load_overrides(path: str | Path) -> dict[str, Any]:
    """Load and validate JSON or YAML override file.

    Parameters
    ----------
    path : str or Path
        Path to a ``.json``, ``.yaml``, or ``.yml`` file.

    Returns
    -------
    dict
        Allowlisted override key-value pairs.

    Raises
    ------
    ValueError
        If the file format is unsupported, content is not a mapping,
        unknown keys are present, or PyYAML is missing for YAML files.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            msg = "PyYAML is required to load YAML override files"
            raise ValueError(msg) from exc
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    else:
        msg = f"Unsupported override file extension: {suffix}"
        raise ValueError(msg)

    if not isinstance(raw, dict):
        msg = "Override file must contain a mapping at the top level"
        raise ValueError(msg)

    unknown = set(raw) - ALLOWED_OVERRIDE_KEYS
    if unknown:
        msg = f"Unknown override keys: {sorted(unknown)}"
        raise ValueError(msg)

    return dict(raw)
