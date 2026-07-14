"""Adaptive ``working_size`` resolution for mosaic fitting.

This module resolves the ``working_size="auto"`` sentinel to a concrete
integer drawn from the validated Optuna grid, *before* the value reaches
:class:`~linum_basic.core.BaSiC`. It implements the design in
``docs/adaptive_working_size.md`` (milestone M006, slice S02).

The resolver is a **pure function** of a frozen context dataclass. It never
runs a BaSiC / ALM solve at any candidate size (the *cost bound*). Selection
is built from two deliberately asymmetric branches:

* **Memory-ceiling shrink** (always safe) — fires when ``128`` does not fit
  the memory budget, returning the largest feasible size ``<= 128``.
* **Quality-floor raise** (opt-in, gated) — fires when ``128`` fits, the
  preview DCT signal indicates unrepresentable high-frequency structure, and
  memory permits enlarging.

Whenever a required signal is unavailable or ambiguous the resolver **fails
safe to 128** and records a ``fallback_reason``. The resolved value and the
reasoning behind it are returned in a :class:`WorkingSizeResolution` for
reproducibility, mirroring the ``params["_strategy"]`` explainability
pattern (D-19) in :mod:`linum_basic.fit`.

Notes
-----
The grid ``{64, 96, 128, 160, 192}`` mirrors
``linum_basic.tuning._DEFAULT_SEARCH_SPACE["working_size"]``. The
importance of keeping the two in sync is enforced by a drift-check test in
``tests/test_working_size_resolver.py`` (the constant is duplicated here to
avoid an import cycle: :mod:`linum_basic.tuning` imports :mod:`linum_basic.fit`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from linum_basic.benchmark.strategies import estimate_strategy_vram_bytes

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "GATE_STATUS_OPT_IN",
    "PREVIEW_RESOLUTION",
    "PREVIEW_Z_SAMPLE",
    "SAFE_DEFAULT",
    "WORKING_SIZE_GRID",
    "WORKING_SIZE_QUALITY_RAISE_THRESHOLD",
    "WorkingSizeContext",
    "WorkingSizeResolution",
    "build_working_size_context",
    "compute_preview_quality",
    "peak_memory_estimate",
    "resolve_working_size",
]

#: The validated candidate grid. Mirrors
#: ``linum_basic.tuning._DEFAULT_SEARCH_SPACE["working_size"]``. A test asserts
#: the two never drift.
WORKING_SIZE_GRID: tuple[int, ...] = (64, 96, 128, 160, 192)

#: The production default and the safe fail-safe value (D002 / MEM002).
SAFE_DEFAULT: int = 128

#: Promotion state of the raise branch (D-19 / R058). S03 evaluated the raise
#: branch on real subjects and it FAILED the K01 seam gate at both 160 and 192
#: (see ``scripts/experiments/s03_artifacts/S03-DECISION.md``), so ``"auto"``
#: is opt-in only and the raise branch is not promoted to a default.
GATE_STATUS_OPT_IN: str = "opt-in (raise branch not promoted: K01 failed at 160/192 on real subjects)"

#: Fixed preview resolution for the DCT quality signal. Computing the signal at
#: a single resolution makes it comparable across datasets and independent of
#: the candidate ``working_size`` (see design doc).
PREVIEW_RESOLUTION: int = 256

#: Maximum z-planes sampled to form the preview mean image, matching the
#: ``min(n_z, 8)`` chunk convention already in ``build_workload_context``.
PREVIEW_Z_SAMPLE: int = 8

#: Conservative default for the quality-floor (raise) threshold. S03 (M006)
#: found this value (0.15) is low enough to fire on real per-z-mode data: on
#: sub-22 it drove ``auto -> 192``, and both raise targets (160, 192) then
#: FAILED the K01 real-subject seam gate (see
#: ``scripts/experiments/s03_artifacts/S03-DECISION.md``). The value is kept
#: at 0.15 — no evidence-based replacement is available and the raise branch
#: must remain opt-in until a future milestone finds a quality-safe raise
#: target or recalibrates upward with persisted preview signals. The branch is
#: additionally gated by the opt-in ``"auto"`` sentinel and by the memory
#: budget permitting a size ``>= 160``.
WORKING_SIZE_QUALITY_RAISE_THRESHOLD: float = 0.15


@dataclass(frozen=True, slots=True)
class WorkingSizeContext:
    """Cheap signal inputs consumed by :func:`resolve_working_size`.

    Analogous to :class:`linum_basic.benchmark.strategies.WorkloadContext`.
    Every field is an O(1) shape/memory query or a single bounded DCT result;
    none requires running BaSiC or the ALM solver.

    Attributes
    ----------
    n_z : int
        Number of z-planes in the mosaic workload.
    n_tiles : int
        Tile count in the mosaic grid.
    field_mode : str
        ``"per-z"`` or ``"global"``.
    tile_shape : tuple of int or None
        Representative tile shape ``(H, W)``.  ``None`` for non-mosaic inputs
        (triggers fail-safe).
    memory_budget_bytes : int or None
        Hard memory ceiling — the only legitimate reason to select a size
        below 128. ``None`` when it cannot be determined (fail-safe to 128).
    preview_quality : float or None
        High-frequency energy fraction ``hf_fraction`` in ``[0, 1]`` from
        :func:`compute_preview_quality`, or ``None`` when the preview mean
        could not be formed. Values outside ``[0, 1]`` are treated as
        ambiguous (fail-safe to 128).
    requested : str or int
        What the caller asked for (``"auto"`` or an explicit integer).
    """

    n_z: int
    n_tiles: int
    field_mode: str
    tile_shape: tuple[int, int] | None
    memory_budget_bytes: int | None
    preview_quality: float | None
    requested: str | int = "auto"


@dataclass(frozen=True, slots=True)
class WorkingSizeResolution:
    """Resolved ``working_size`` plus explainability metadata.

    Attributes
    ----------
    resolved_working_size : int
        The concrete integer chosen (always a member of
        :data:`WORKING_SIZE_GRID`).
    requested : str or int
        What the caller asked for (``"auto"`` or an explicit integer).
    candidate_grid : tuple of int
        The grid the resolver selected from.
    rule_path : str
        One of ``"baseline-default"``, ``"memory-ceiling-shrink"``,
        ``"quality-floor-raise"``, or ``"fallback-safe-default"``.
    fallback_reason : str or None
        Why 128 was chosen when a required signal was unavailable/ambiguous.
    gate_status : str
        Promotion state of the raise branch. After S03's real-subject K01
        evaluation rejected the raise branch (160 and 192 both fail the seam
        gate on sub-22), this is always :data:`GATE_STATUS_OPT_IN`; ``"auto"``
        is opt-in only and the raise branch is not promoted.
    signals : dict
        Snapshot of the input signals used for the decision.
    peak_memory_estimate_bytes : dict
        Per-candidate peak-memory estimate keyed by candidate size (string).
    """

    resolved_working_size: int
    requested: str | int
    candidate_grid: tuple[int, ...]
    rule_path: str
    fallback_reason: str | None
    gate_status: str
    signals: dict[str, Any]
    peak_memory_estimate_bytes: dict[str, int]

    def metadata(self) -> dict[str, Any]:
        """Return the observability dict for ``params["_working_size_selector"]``.

        The shape mirrors the existing ``params["_strategy"]`` explainability
        pattern (D-19). Purely additive metadata — it never affects fit
        numerics.

        Returns
        -------
        dict
            The explainability metadata dict, with keys ``resolved_working_size``,
            ``requested``, ``candidate_grid``, ``rule_path``, ``fallback_reason``,
            ``gate_status``, ``signals``, and ``peak_memory_estimate_bytes``.
        """
        return {
            "resolved_working_size": self.resolved_working_size,
            "requested": self.requested,
            "candidate_grid": list(self.candidate_grid),
            "rule_path": self.rule_path,
            "fallback_reason": self.fallback_reason,
            "gate_status": self.gate_status,
            "signals": dict(self.signals),
            "peak_memory_estimate_bytes": dict(self.peak_memory_estimate_bytes),
        }


def peak_memory_estimate(working_size: int, ctx: WorkingSizeContext) -> int:
    """Conservative per-solve peak-memory estimate for *working_size*.

    Reuses :func:`linum_basic.benchmark.strategies.estimate_strategy_vram_bytes`
    with a per-solve chunk of one z-plane (``n_z_chunk=1``): for ``per-z``
    mode the solver sees one z-plane at a time, and ``global`` mode fits one
    model per z-level then averages, so the per-solve peak is identical.

    Parameters
    ----------
    working_size : int
        Candidate working resolution.
    ctx : WorkingSizeContext
        The workload context (supplies ``n_tiles``).

    Returns
    -------
    int
        Estimated peak bytes.
    """
    return estimate_strategy_vram_bytes(1, ctx.n_tiles, working_size)


def _preview_quality_is_valid(value: float | None) -> bool:
    """Return whether *value* is a usable hf_fraction in ``[0, 1]``."""
    if value is None:
        return False
    try:
        v = float(value)
    except TypeError, ValueError:
        return False
    return 0.0 <= v <= 1.0


def resolve_working_size(ctx: WorkingSizeContext) -> WorkingSizeResolution:
    """Resolve a concrete ``working_size`` from *ctx*.

    Pure function: it performs only O(1) arithmetic over pre-gathered
    signals. It never instantiates :class:`~linum_basic.core.BaSiC`, never
    calls the ALM solver, and never runs a probe fit at any candidate size.

    Parameters
    ----------
    ctx : WorkingSizeContext
        Frozen signal bundle (see :class:`WorkingSizeContext`).

    Returns
    -------
    WorkingSizeResolution
        The chosen integer plus full explainability metadata.

    Notes
    -----
    The rule has two branches:

    1. **Memory-ceiling shrink** (always safe). When the budget is known and
       128 does not fit, return the largest feasible size ``<= 128``. If even
       64 does not fit, fail safe to 128 (better to let the caller OOM
       predictably on the documented default than to silently pick an
       off-grid value).
    2. **Quality-floor raise** (opt-in, gated). When 128 fits, the preview
       signal exceeds :data:`WORKING_SIZE_QUALITY_RAISE_THRESHOLD`, and memory
       permits a size ``>= 160``, return the largest feasible size ``> 128``.

    Whenever the budget is unknown or no feasible set exists, the resolver
    returns 128 with a populated ``fallback_reason``.
    """
    peak_per_candidate: dict[str, int] = {str(s): peak_memory_estimate(s, ctx) for s in WORKING_SIZE_GRID}

    candidate = SAFE_DEFAULT
    rule_path = "baseline-default"
    fallback_reason: str | None = None

    budget = ctx.memory_budget_bytes

    # --- (2) MEMORY CEILING (always-safe shrink) ---------------------------
    feasible: list[int] | None = None
    if budget is not None:
        try:
            budget_int = int(budget)
        except TypeError, ValueError:
            budget_int = None
        if budget_int is not None:
            feasible = [s for s in WORKING_SIZE_GRID if peak_per_candidate[str(s)] <= budget_int]

    if feasible is not None:
        at_or_below = [s for s in feasible if s <= SAFE_DEFAULT]
        candidate = max(at_or_below) if at_or_below else SAFE_DEFAULT
        if candidate < SAFE_DEFAULT:
            rule_path = "memory-ceiling-shrink"
        if not feasible:
            # Nothing fits at all (even 64): fail safe to 128.
            candidate = SAFE_DEFAULT
            rule_path = "fallback-safe-default"
            fallback_reason = "memory_budget_bytes too small for any candidate size; falling back to the safe default 128"
    else:
        # Budget unknown / unparseable: fail safe to 128.
        candidate = SAFE_DEFAULT
        rule_path = "fallback-safe-default"
        fallback_reason = "memory_budget_bytes unavailable; falling back to the safe default 128"

    # --- (3) QUALITY FLOOR RAISE (opt-in, gated) ---------------------------
    if candidate == SAFE_DEFAULT and rule_path != "fallback-safe-default":
        pq = ctx.preview_quality
        if pq is not None and _preview_quality_is_valid(pq) and pq > WORKING_SIZE_QUALITY_RAISE_THRESHOLD:
            # Memory must permit enlarging: require 160 to be feasible.
            enlarge_feasible: list[int] = [s for s in feasible if s > SAFE_DEFAULT] if feasible is not None else []
            if 160 in enlarge_feasible:
                candidate = max(enlarge_feasible)
                rule_path = "quality-floor-raise"

    return WorkingSizeResolution(
        resolved_working_size=candidate,
        requested=ctx.requested,
        candidate_grid=WORKING_SIZE_GRID,
        rule_path=rule_path,
        fallback_reason=fallback_reason,
        gate_status=GATE_STATUS_OPT_IN,
        signals={
            "n_z": ctx.n_z,
            "n_tiles": ctx.n_tiles,
            "tile_shape": list(ctx.tile_shape) if ctx.tile_shape is not None else None,
            "field_mode": ctx.field_mode,
            "memory_budget_bytes": ctx.memory_budget_bytes,
            "preview_quality": ctx.preview_quality,
        },
        peak_memory_estimate_bytes=peak_per_candidate,
    )


def compute_preview_quality(mean_image: NDArray[np.floating[Any]] | np.ndarray) -> float | None:
    """High-frequency DCT energy fraction of a mean image at fixed resolution.

    The mean image is resized to :data:`PREVIEW_RESOLUTION` (one
    :func:`cv2.resize`) and the 2-D DCT is computed. ``hf_fraction`` is the
    share of total ``|DCT|`` energy lying in spatial frequencies **above** the
    Nyquist limit of a ``128 x 128`` grid -- i.e. outside the top-left
    ``128 x 128`` low-frequency block of the ``PREVIEW_RESOLUTION^2`` DCT.

    A value near ``0`` means the illumination field is smooth and ``128``
    resolves it fully; a large value means the ``128^2`` grid throws away real
    structure. Returns ``None`` when the input cannot be processed (non-finite,
    wrong shape, or non-positive mean), which feeds the fail-safe path.

    Parameters
    ----------
    mean_image : numpy.ndarray
        2-D per-pixel mean image (e.g. across tiles and a bounded z-subsample).

    Returns
    -------
    float or None
        ``hf_fraction`` in ``[0, 1]``, or ``None`` if the signal is unusable.
    """
    if mean_image is None:
        return None
    try:
        arr = np.asarray(mean_image, dtype=np.float64)
    except TypeError, ValueError:
        return None
    if arr.ndim != 2 or arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    mean_val = float(arr.mean())
    if mean_val <= 0:
        return None

    import cv2

    resized = cv2.resize(arr, (PREVIEW_RESOLUTION, PREVIEW_RESOLUTION), interpolation=cv2.INTER_AREA)
    normalised = resized / (mean_val + 1e-9)

    from scipy.fft import dctn

    coeffs = np.abs(dctn(normalised, norm="ortho"))
    total = float(coeffs.sum())
    if total <= 0:
        return None

    # Low-frequency block: top-left SAFE_DEFAULT x SAFE_DEFAULT coefficients.
    low = coeffs[:SAFE_DEFAULT, :SAFE_DEFAULT]
    hf_energy = total - float(low.sum())
    fraction = hf_energy / total
    # Guard against tiny float overshoot from rounding.
    return float(min(1.0, max(0.0, fraction)))


def _query_memory_budget(device: str | None) -> int | None:
    """Best-effort detection of the available memory budget in bytes.

    Resolution order (per the design doc):

    1. CUDA device memory via ``torch.cuda.mem_get_info`` when *device* targets
       CUDA and torch is importable.
    2. Host virtual memory (Linux ``/proc/meminfo``; otherwise unavailable).

    Returns ``None`` when the budget cannot be determined, which triggers the
    fail-safe path in :func:`resolve_working_size`. No third-party dependency
    (e.g. psutil) is introduced.
    """
    dev = (device or "").lower()
    if dev.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info(device)
                return int(free)
        except ImportError, RuntimeError, ValueError:
            return None

    # Linux host memory: MemAvailable (kB).
    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except OSError, ValueError, IndexError:
        pass

    # macOS / other: no stdlib RAM query without psutil -> None (fail-safe).
    return None


def build_working_size_context(
    *,
    n_z: int,
    n_tiles: int,
    field_mode: str,
    tile_shape: tuple[int, int] | None,
    mean_image: np.ndarray | None = None,
    memory_budget_bytes: int | None = None,
    device: str | None = None,
    requested: str | int = "auto",
) -> WorkingSizeContext:
    """Gather cheap signals and build a :class:`WorkingSizeContext`.

    This is the wiring helper used by :func:`linum_basic.fit.fit_mosaic` and
    :func:`linum_basic.tuning.tune` to turn mosaic metadata into the resolver's
    frozen input. It performs at most: one memory query and (optionally) one
    bounded DCT via :func:`compute_preview_quality`. It never runs a solve.

    The preview quality signal is derived from *mean_image* when provided
    (the caller-formed mean across tiles and a bounded z-subsample); otherwise
    it is left ``None`` (fail-safe).

    Parameters
    ----------
    n_z : int
        Number of z-planes in the mosaic workload.
    n_tiles : int
        Number of tiles in the mosaic grid.
    field_mode : str
        ``"per-z"`` or ``"global"``.
    tile_shape : tuple of int or None
        Representative tile shape ``(H, W)``, or ``None`` for non-mosaic inputs.
    mean_image : numpy.ndarray or None
        Pre-formed per-pixel mean image for the quality signal.
    memory_budget_bytes : int or None
        Explicit memory budget. When ``None`` the budget is auto-detected via
        :func:`_query_memory_budget`.
    device : str or None
        Device string used for CUDA budget detection.
    requested : str or int
        What the caller asked for.

    Returns
    -------
    WorkingSizeContext
        Frozen signal bundle for :func:`resolve_working_size`.
    """
    budget = memory_budget_bytes
    if budget is None:
        budget = _query_memory_budget(device)

    preview = compute_preview_quality(mean_image) if mean_image is not None else None

    return WorkingSizeContext(
        n_z=n_z,
        n_tiles=n_tiles,
        field_mode=field_mode,
        tile_shape=tile_shape,
        memory_budget_bytes=budget,
        preview_quality=preview,
        requested=requested,
    )
