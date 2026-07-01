"""Phase 8.1 algorithmic foundation audit (ALGO-01 through ALGO-04).

Provides frozen report-row dataclasses, a paper-to-code traceability matrix,
synthetic micro-benchmark helpers, and ranked architectural finding builders
consumed by Phase 9 forensics and Phase 11 optimisation work.
"""

from __future__ import annotations

import cProfile
import dataclasses
import importlib
import importlib.util
import io
import json
import pstats
import random
import re
import time
import tracemalloc
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from pstats import SortKey
from typing import Any
from unittest.mock import patch

import numpy as np

from linum_basic.benchmark.metadata import collect_git_commit
from linum_basic.core import BaSiC


def _sbh_simulator_available() -> bool:
    try:
        return importlib.util.find_spec("sbh_simulator.simulator") is not None
    except ModuleNotFoundError:
        return False


_SBH_SIMULATOR_AVAILABLE = _sbh_simulator_available()

MIN_N = 5
MAX_N = 64
MIN_WS = 32
MAX_WS = 128

AUDIT_MAX_REWEIGHTING_ITERATIONS = 5
AUDIT_INNER_ALM_MAX_ITER = 100
RECOVERY_CORRELATION_THRESHOLD = 0.85

TRACEABILITY_STATUSES = ("MATCH", "INTENTIONAL", "GAP", "PARTIAL")
FIX_CLASSES = ("algorithm", "data_layout", "backend", "allocation")

__all__ = [
    "AUDIT_INNER_ALM_MAX_ITER",
    "AUDIT_MAX_REWEIGHTING_ITERATIONS",
    "FIX_CLASSES",
    "MAX_N",
    "MAX_WS",
    "MIN_N",
    "MIN_WS",
    "RECOVERY_CORRELATION_THRESHOLD",
    "TRACEABILITY_STATUSES",
    "_SBH_SIMULATOR_AVAILABLE",
    "AuditFinding",
    "HotspotRecord",
    "MicroBenchmarkResult",
    "TraceabilityRow",
    "audit_alm_cap",
    "build_audit_report",
    "build_traceability_matrix",
    "make_handrolled_stack",
    "make_sbh_stack",
    "profile_call",
    "profile_prepare_run",
    "run_micro_benchmark",
    "write_audit_artifacts",
]


@dataclass(frozen=True, slots=True)
class TraceabilityRow:
    """One paper-to-code traceability matrix row (ALGO-01).

    Attributes
    ----------
    paper_step : str
        BaSiC algorithm step from Peng et al. 2017.
    paper_ref : str
        Paper equation or section reference.
    code_refs : tuple[str, ...]
        ``path:line`` anchors into linum-basic source.
    status : str
        Closed vocabulary: MATCH, INTENTIONAL, GAP, or PARTIAL.
    intentional : bool
        When ``True``, deviation is documented and must not be ranked as a fix.
    notes : str
        Concise audit notes citing paper concepts.
    """

    paper_step: str
    paper_ref: str
    code_refs: tuple[str, ...]
    status: str
    intentional: bool
    notes: str

    def __post_init__(self) -> None:  # noqa: D105
        if self.status not in TRACEABILITY_STATUSES:
            msg = f"status must be one of {TRACEABILITY_STATUSES}, got {self.status!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class AuditFinding:
    """One ranked architectural audit finding (ALGO-04).

    Field names align with :class:`~linum_basic.benchmark.profile.RankedLever`
    so ``finding_id`` and ``code_refs[0]`` map to ``lever_id`` and
    ``target_file`` for Phase 9/11 schema continuity.

    Attributes
    ----------
    finding_id : str
        Stable finding identifier.
    rank : int
        Priority rank; lower values are higher priority.
    description : str
        Human-readable finding summary.
    fix_class : str
        Closed vocabulary: algorithm, data_layout, backend, or allocation.
    measured_cost_ms : float
        Measured synthetic wall time in milliseconds.
    measured_cost_pct : float
        Share of total micro-benchmark wall time (0-100).
    paper_ref : str
        Paper equation or section reference.
    code_refs : tuple[str, ...]
        ``path:line`` anchors into linum-basic source.
    phase9_change_class : str
        FORE-02 change class for forensics handoff.
    phase11_priority : str
        Qualitative Phase 11 priority label.
    suggested_action : str
        Recommended next step for operators.
    """

    finding_id: str
    rank: int
    description: str
    fix_class: str
    measured_cost_ms: float
    measured_cost_pct: float
    paper_ref: str
    code_refs: tuple[str, ...]
    phase9_change_class: str
    phase11_priority: str
    suggested_action: str

    def __post_init__(self) -> None:  # noqa: D105
        if self.fix_class not in FIX_CLASSES:
            msg = f"fix_class must be one of {FIX_CLASSES}, got {self.fix_class!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class MicroBenchmarkResult:
    """Timing result from one synthetic micro-benchmark run (ALGO-02).

    Attributes
    ----------
    name : str
        Benchmark scenario label.
    working_size : int
        BaSiC ``working_size`` used during the run.
    n_images : int
        Number of images in the synthetic stack.
    backend : str
        Array backend (``numpy`` or ``torch``).
    max_reweighting_iterations : int
        Reweighting cap applied for the audit run.
    prepare_ms : float
        Wall time for :meth:`~linum_basic.core.BaSiC.prepare` in ms.
    run_ms : float
        Wall time for :meth:`~linum_basic.core.BaSiC.run` in ms.
    normalize_ms : float
        Wall time for a single :meth:`~linum_basic.core.BaSiC.normalize` call in ms.
    peak_bytes : int
        Peak traced memory in bytes (``tracemalloc`` high-water mark).
    """

    name: str
    working_size: int
    n_images: int
    backend: str
    max_reweighting_iterations: int
    prepare_ms: float
    run_ms: float
    normalize_ms: float
    peak_bytes: int


@dataclass(frozen=True, slots=True)
class HotspotRecord:
    """One cProfile hotspot row with an audit fix_class (ALGO-03).

    Attributes
    ----------
    function : str
        Fully qualified function name from cProfile stats.
    cumtime_s : float
        Cumulative time in seconds (hotspot ranking only).
    fix_class : str
        Closed vocabulary: algorithm, data_layout, backend, or allocation.
    """

    function: str
    cumtime_s: float
    fix_class: str

    def __post_init__(self) -> None:  # noqa: D105
        if self.fix_class not in FIX_CLASSES:
            msg = f"fix_class must be one of {FIX_CLASSES}, got {self.fix_class!r}"
            raise ValueError(msg)


def profile_call(
    fn: Callable[..., Any],
    *args: Any,
    top: int = 20,
    **kwargs: Any,
) -> dict[str, Any]:
    """Profile a callable with wall time, peak memory, and cProfile hotspots (ALGO-03).

    Wall time comes from :func:`time.perf_counter` and is the authoritative
    duration for rankings. cProfile cumulative totals are for hotspot ordering
    only — the profiler adds overhead and must not be used as ground-truth
    benchmarking (Python profiling docs, Pitfall 4 in Phase 8.1 research).

    Parameters
    ----------
    fn : callable
        Function to profile.
    *args
        Positional arguments passed to ``fn``.
    top : int
        Number of cumulative-time rows to include in ``pstats_top``.
    **kwargs
        Keyword arguments passed to ``fn``.

    Returns
    -------
    dict
        Keys: ``wall_ms``, ``peak_bytes``, ``pstats_top``, ``result``.
    """
    tracemalloc.start()
    try:
        t0 = time.perf_counter()
        with cProfile.Profile() as prof:
            result = fn(*args, **kwargs)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats(SortKey.CUMULATIVE).print_stats(top)
    return {
        "wall_ms": wall_ms,
        "peak_bytes": peak_bytes,
        "pstats_top": buf.getvalue(),
        "result": result,
    }


_PSTATS_ROW_RE = re.compile(
    r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+)$",
)


def _assign_fix_class(function: str) -> str:
    """Map a cProfile function name to an audit fix_class (ALGO-03)."""
    name = function.lower()
    if any(token in name for token in ("resize", "cv2", "transpose", "upsample")):
        return "data_layout"
    if "sort" in name:
        return "allocation"
    if any(token in name for token in ("svd", "backend", "torch", "cuda")):
        return "backend"
    if any(
        token in name
        for token in (
            "dct",
            "idct",
            "_alm_core_step",
            "shrink",
            "inexact_alm",
            "update",
            "prepare",
            "run",
            "normalize",
        )
    ):
        return "algorithm"
    return "algorithm"


def _parse_pstats_hotspots(pstats_top: str, *, top: int = 20) -> list[HotspotRecord]:
    """Parse cProfile text output into structured hotspot records."""
    records: list[HotspotRecord] = []
    for line in pstats_top.splitlines():
        match = _PSTATS_ROW_RE.match(line)
        if not match:
            continue
        cumtime_s = float(match.group(4))
        function = match.group(6).strip()
        records.append(
            HotspotRecord(
                function=function,
                cumtime_s=cumtime_s,
                fix_class=_assign_fix_class(function),
            )
        )
        if len(records) >= top:
            break
    return records


def profile_prepare_run(
    stack: np.ndarray,
    *,
    working_size: int = 64,
    estimate_darkfield: bool = True,
) -> dict[str, Any]:
    """Profile BaSiC prepare/run/normalize phases with hotspot fix_classes (ALGO-03).

    Uses the NumPy backend and audit caps from Plan 02
    (``max_reweighting_iterations=5``, inner ALM ``max_iter=100``). Each phase
    is profiled independently via :func:`profile_call`.

    Parameters
    ----------
    stack : numpy.ndarray
        Synthetic image stack of shape ``(n, h, w)`` with ``5 <= n <= 64``.
    working_size : int
        BaSiC resize target (``32 <= working_size <= 128``); audit defaults to 64.
    estimate_darkfield : bool
        Whether to estimate a dark-field during the fit.

    Returns
    -------
    dict
        Keys ``prepare``, ``run``, ``normalize`` mapping to phase dicts with
        ``wall_ms``, ``peak_bytes``, and ``hotspots`` (list of HotspotRecord
        dicts).
    """
    _validate_n(int(stack.shape[0]))
    _validate_working_size(working_size)

    with audit_alm_cap(AUDIT_INNER_ALM_MAX_ITER):
        model = BaSiC(stack, estimate_darkfield=estimate_darkfield, backend="numpy", verbose=False)
        model.working_size = working_size
        model.max_reweighting_iterations = AUDIT_MAX_REWEIGHTING_ITERATIONS

        prepare_out = profile_call(model.prepare, stack)
        run_out = profile_call(model.run)
        normalize_out = profile_call(model.normalize, stack[0])

    def _phase_payload(out: dict[str, Any]) -> dict[str, Any]:
        hotspots = _parse_pstats_hotspots(str(out["pstats_top"]))
        return {
            "wall_ms": out["wall_ms"],
            "peak_bytes": out["peak_bytes"],
            "hotspots": [
                {
                    "function": h.function,
                    "cumtime_s": h.cumtime_s,
                    "fix_class": h.fix_class,
                }
                for h in hotspots
            ],
        }

    return {
        "prepare": _phase_payload(prepare_out),
        "run": _phase_payload(run_out),
        "normalize": _phase_payload(normalize_out),
    }


def _validate_n(n: int) -> None:
    if n < MIN_N or n > MAX_N:
        msg = f"n must be in [{MIN_N}, {MAX_N}], got {n}"
        raise ValueError(msg)


def _validate_working_size(ws: int) -> None:
    if ws < MIN_WS or ws > MAX_WS:
        msg = f"working_size must be in [{MIN_WS}, {MAX_WS}], got {ws}"
        raise ValueError(msg)


def make_handrolled_stack(
    n: int = 12,
    ws: int = 64,
    *,
    seed: int = 0,
    estimate_darkfield: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Build a deterministic synthetic stack with known flat-field (ALGO-02).

    Parameters
    ----------
    n : int
        Number of images in the stack (``5 <= n <= 64``).
    ws : int
        Square side length in pixels (``32 <= ws <= 128``).
    seed : int
        RNG seed for sparse additive noise.
    estimate_darkfield : bool
        When ``True``, include a smooth zero-mean dark-field ground truth.

    Returns
    -------
    tuple of numpy.ndarray
        ``(stack, flatfield, darkfield)`` with ``stack`` of shape
        ``(n, ws, ws)`` and dtype ``float32``. ``darkfield`` is ``None``
        unless ``estimate_darkfield`` is ``True``.
    """
    _validate_n(n)
    _validate_working_size(ws)

    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(
        np.linspace(0.8, 1.2, ws),
        np.linspace(0.9, 1.1, ws),
        indexing="xy",
    )
    flatfield = (xs * ys).astype(np.float32)
    flatfield = flatfield / flatfield.mean()

    noise = rng.normal(0, 0.05, (n, ws, ws)).astype(np.float32)
    mask = rng.random((n, ws, ws)) < 0.05
    noise *= mask

    darkfield: np.ndarray | None = None
    if estimate_darkfield:
        df_xs, df_ys = np.meshgrid(
            np.linspace(-0.5, 0.5, ws),
            np.linspace(-0.5, 0.5, ws),
            indexing="xy",
        )
        darkfield = (0.02 * (df_xs**2 + df_ys**2)).astype(np.float32)
        stack = (flatfield[None, :, :] + darkfield[None, :, :] + noise).astype(np.float32)
    else:
        stack = (flatfield[None, :, :] + noise).astype(np.float32)

    return stack, flatfield, darkfield


def make_sbh_stack(
    n: int = 12,
    ws: int = 64,
    *,
    seed: int = 42,
    kind: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic stack via sbh-simulator when the extra is installed.

    Parameters
    ----------
    n : int
        Number of images (``5 <= n <= 64``).
    ws : int
        Square side length (``32 <= ws <= 128``).
    seed : int
        RNG seed passed to sbh-simulator.
    kind : str
        Vignette model: ``"gaussian"`` (default).

    Returns
    -------
    tuple of numpy.ndarray
        ``(stack, flatfield, darkfield)`` all ``float32`` with spatial shape
        ``(ws, ws)`` for the ground-truth fields.

    Raises
    ------
    ImportError
        When ``sbh-simulator`` is not installed.
    """
    if not _SBH_SIMULATOR_AVAILABLE:
        msg = "sbh-simulator not installed; install with validation extra"
        raise ImportError(msg)

    _validate_n(n)
    _validate_working_size(ws)

    simulator = importlib.import_module("sbh_simulator.simulator")
    rng = random.Random(seed)
    if kind == "gaussian":
        flatfield = simulator.generate_gaussian_vignette(width=ws, height=ws, sigma=0.7, rng=rng).astype(np.float32)
        darkfield = simulator.generate_gaussian_darkfield(
            width=ws,
            height=ws,
            sigma=0.7,
            max_offset=0.05,
            rng=rng,
        ).astype(np.float32)
    elif kind == "zernike":
        flatfield = simulator.generate_zernike_vignette(width=ws, height=ws, order=4, rng=rng).astype(np.float32)
        darkfield = simulator.generate_zernike_darkfield(
            width=ws,
            height=ws,
            order=4,
            max_offset=0.05,
            rng=rng,
        ).astype(np.float32)
    else:
        msg = f"kind must be 'gaussian' or 'zernike', got {kind!r}"
        raise ValueError(msg)

    np_rng = np.random.default_rng(seed)
    noise = np_rng.normal(0, 0.02, (n, ws, ws)).astype(np.float32)
    stack = (flatfield[None, :, :] + darkfield[None, :, :] + noise).astype(np.float32)
    return stack, flatfield, darkfield


@contextmanager
def audit_alm_cap(max_iter: int = AUDIT_INNER_ALM_MAX_ITER) -> Iterator[None]:
    """Cap inner ALM iterations during audit micro-benchmarks only (ALGO-02).

    Patches :func:`~linum_basic.core.inexact_alm_l1` for the duration of the
    context so production defaults (500 inner iterations) are not used during
    CPU audit runs. Documented in the Phase 8.1 audit report footnotes.

    Parameters
    ----------
    max_iter : int, optional
        Inner ALM iteration cap for audit workloads. Default is
        ``AUDIT_INNER_ALM_MAX_ITER`` (100).

    Yields
    ------
    None
        Restores the original solver after the context exits.
    """
    import linum_basic.core as core_module

    original = core_module.inexact_alm_l1

    def _capped(*args: Any, **kwargs: Any) -> Any:
        kwargs["max_iter"] = max_iter
        return original(*args, **kwargs)

    with patch.object(core_module, "inexact_alm_l1", _capped):
        yield


def _fit_with_audit_caps(
    stack: np.ndarray,
    *,
    working_size: int = 64,
    estimate_darkfield: bool = True,
) -> BaSiC:
    """Fit BaSiC with audit reweighting and inner-ALM caps (ALGO-02)."""
    _validate_n(int(stack.shape[0]))
    _validate_working_size(working_size)
    with audit_alm_cap(AUDIT_INNER_ALM_MAX_ITER):
        model = BaSiC(stack, estimate_darkfield=estimate_darkfield, backend="numpy", verbose=False)
        model.working_size = working_size
        model.max_reweighting_iterations = AUDIT_MAX_REWEIGHTING_ITERATIONS
        model.prepare()
        model.run()
    return model


def run_micro_benchmark(
    stack: np.ndarray,
    *,
    working_size: int = 64,
    estimate_darkfield: bool = True,
    name: str = "handrolled",
) -> MicroBenchmarkResult:
    """Run a CPU micro-benchmark with per-phase timings (ALGO-02).

    Parameters
    ----------
    stack : numpy.ndarray
        Synthetic image stack of shape ``(n, h, w)`` with ``5 <= n <= 64``.
    working_size : int
        BaSiC resize target (``32 <= working_size <= 128``).
    estimate_darkfield : bool
        Whether to estimate a dark-field during the fit.
    name : str
        Scenario label stored on the result.

    Returns
    -------
    MicroBenchmarkResult
        Per-phase wall times, peak traced memory, and audit cap metadata.
    """
    _validate_n(int(stack.shape[0]))
    _validate_working_size(working_size)

    tracemalloc.start()
    try:
        with audit_alm_cap(AUDIT_INNER_ALM_MAX_ITER):
            model = BaSiC(stack, estimate_darkfield=estimate_darkfield, backend="numpy", verbose=False)
            model.working_size = working_size
            model.max_reweighting_iterations = AUDIT_MAX_REWEIGHTING_ITERATIONS

            t0 = time.perf_counter()
            model.prepare()
            prepare_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            model.run()
            run_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            _ = model.normalize(stack[0])
            normalize_ms = (time.perf_counter() - t0) * 1000.0

        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return MicroBenchmarkResult(
        name=name,
        working_size=working_size,
        n_images=int(stack.shape[0]),
        backend="numpy",
        max_reweighting_iterations=AUDIT_MAX_REWEIGHTING_ITERATIONS,
        prepare_ms=prepare_ms,
        run_ms=run_ms,
        normalize_ms=normalize_ms,
        peak_bytes=peak_bytes,
    )


def build_traceability_matrix() -> list[TraceabilityRow]:
    """Return the static paper-to-code traceability catalog (ALGO-01).

    Returns
    -------
    list[TraceabilityRow]
        Rows covering CONTEXT paper steps, AGENTS.md invariants, and
        confirmed algorithm gaps (Eq. 8 two-step B, Eq. 9 correction).
    """
    return [
        TraceabilityRow(
            paper_step="Stack assembly / resize",
            paper_ref="Methods",
            code_refs=("linum_basic/core.py:291",),
            status="MATCH",
            intentional=False,
            notes="ThreadPoolExecutor resize; OpenCV double-transpose documented separately.",
        ),
        TraceabilityRow(
            paper_step="OpenCV double-transpose resize",
            paper_ref="Methods",
            code_refs=("linum_basic/core.py:291",),
            status="INTENTIONAL",
            intentional=True,
            notes="cv2.resize(img.T, ...).T matches OpenCV column-major convention (AGENTS.md).",
        ),
        TraceabilityRow(
            paper_step="Measurement matrix - intensity sort",
            paper_ref="Eq. 4-5, Step I",
            code_refs=("linum_basic/core.py:348",),
            status="MATCH",
            intentional=False,
            notes="Full (N, ws, ws) copy via np.sort; unsorted stack never passed to ALM.",
        ),
        TraceabilityRow(
            paper_step="Auto lambda_s, lambda_d from DCT sum",
            paper_ref="Supp. Note 4",
            code_refs=("linum_basic/core.py:342", "linum_basic/core.py:344", "linum_basic/core.py:346"),
            status="MATCH",
            intentional=False,
            notes="dct_energy(mean)/800 and /2000 shared with tuning and batched fit paths.",
        ),
        TraceabilityRow(
            paper_step="Low-rank + sparse split (I_B, I_R init)",
            paper_ref="Step II-III",
            code_refs=("linum_basic/_alm.py:272", "linum_basic/core.py:363"),
            status="MATCH",
            intentional=False,
            notes="ALM state initialised in inexact_alm_l1; W/Ib/Ir zeroed in prepare().",
        ),
        TraceabilityRow(
            paper_step="Eq. 6 ALM optimisation",
            paper_ref="Eq. 6",
            code_refs=("linum_basic/_alm.py:272", "linum_basic/_alm.py:186"),
            status="MATCH",
            intentional=False,
            notes="inexact_alm_l1 and _alm_core_step implement reweighted ALM.",
        ),
        TraceabilityRow(
            paper_step="W⊙I_R L1 (weighted residual soft-threshold)",
            paper_ref="Eq. 6",
            code_refs=("linum_basic/_alm.py:186",),
            status="MATCH",
            intentional=False,
            notes="Weighted spatial shrink on residual Ir.",
        ),
        TraceabilityRow(
            paper_step="lambda_s‖F(S)‖₁ flat-field DCT sparsity",
            paper_ref="Eq. 6",
            code_refs=("linum_basic/_alm.py:190", "linum_basic/_alm.py:192"),
            status="MATCH",
            intentional=False,
            notes="Mean over images, 2-D DCT, shrink, inverse DCT for S.",
        ),
        TraceabilityRow(
            paper_step="Dark-field dual shrink (D_R DCT + spatial)",
            paper_ref="Eq. 6",
            code_refs=("linum_basic/_alm.py:238", "linum_basic/_alm.py:241"),
            status="INTENTIONAL",
            intentional=True,
            notes="Dual shrink on dark-field D_R; regression-sensitive (AGENTS.md).",
        ),
        TraceabilityRow(
            paper_step="Per-image baseline B inside sorted ALM",
            paper_ref="Eq. 6 (sorted)",
            code_refs=("linum_basic/_alm.py:197", "linum_basic/_alm.py:199"),
            status="PARTIAL",
            intentional=False,
            notes="B updated on sorted matrix each ALM iteration; shape (N, 1).",
        ),
        TraceabilityRow(
            paper_step="Two-step B estimation on unsorted matrix",
            paper_ref="Eq. 8-9",
            code_refs=("linum_basic/core.py:404", "linum_basic/core.py:405"),
            status="GAP",
            intentional=False,
            notes="img_stack_resized unused after sort; all update() passes use img_sort only.",
        ),
        TraceabilityRow(
            paper_step="B1 monotonicity guard",
            paper_ref="MATLAB legacy",
            code_refs=("linum_basic/_alm.py:227",),
            status="INTENTIONAL",
            intentional=True,
            notes="Keep prior B1 when clamped candidate <= 0; do not replace with max(0, ...) (AGENTS.md).",
        ),
        TraceabilityRow(
            paper_step="Reweighted L1 outer loop",
            paper_ref="Methods / Candès 2008",
            code_refs=("linum_basic/core.py:446", "linum_basic/core.py:371", "linum_basic/core.py:424"),
            status="MATCH",
            intentional=False,
            notes="run() → update() → update_weights(); class default 10 iters, harness may use 500.",
        ),
        TraceabilityRow(
            paper_step="Flat-field extraction from Ib",
            paper_ref="Methods",
            code_refs=("linum_basic/core.py:426", "linum_basic/core.py:427"),
            status="MATCH",
            intentional=False,
            notes="Ib.mean(axis=0) - D, mean-normalised to single S.",
        ),
        TraceabilityRow(
            paper_step="Dark-field zero init + mad convergence guard",
            paper_ref="Methods",
            code_refs=("linum_basic/core.py:360", "linum_basic/core.py:435"),
            status="INTENTIONAL",
            intentional=True,
            notes="Zero dark-field init; mad_dark=1 when prior D sum ≈ 0 (AGENTS.md).",
        ),
        TraceabilityRow(
            paper_step="Correction (normalize)",
            paper_ref="Eq. 7 / Eq. 9",
            code_refs=("linum_basic/core.py:529", "linum_basic/core.py:552"),
            status="PARTIAL",
            intentional=False,
            notes="Applies (I - D) / (S + ε); Eq. 9 per-frame B_i subtraction not exposed.",
        ),
        TraceabilityRow(
            paper_step="Backend DCT / power-iteration SVD",
            paper_ref="Methods",
            code_refs=("linum_basic/backend.py:477", "linum_basic/backend.py:524"),
            status="INTENTIONAL",
            intentional=True,
            notes="Torch path uses batched power iteration for σ₁; no full GPU SVD (AGENTS.md).",
        ),
    ]


AUDIT_REPORT_SCHEMA_VERSION = "8.1.0"

_INVARIANT_FINDING_IDS = frozenset(
    {
        "b1-monotonicity-guard",
        "dark-field-dual-shrink",
        "opencv-double-transpose",
        "power-iteration-svd",
    }
)

_PHASE9_HANDOFF_FORENSICS_HYPOTHESES: tuple[str, ...] = (
    "Commits touching core.prepare sort/copy logic",
    "Introduction of batched fit path vs sequential BaSiC",
    "Harness max_reweighting_iterations raised to 500",
    "torch.compile / inductor integration in _alm.py",
)

_PHASE9_FORE02_CHANGE_CLASSES: tuple[str, ...] = (
    "algorithm",
    "harness_defaults",
    "working_size_strategy",
    "torch_compile_inductor",
)

_PHASE9_DEFERRED: tuple[str, ...] = (
    "BaSiCPy ceiling",
    "steady_state_ms production protocol",
    "ws=128 spot-check",
)

_INTENTIONAL_INVARIANTS: tuple[tuple[str, str], ...] = (
    ("B1 monotonicity guard", "Keep prior B1 when clamped candidate <= 0 (_alm.py); do not replace with max(0, ...)."),
    ("Dark-field dual shrink", "Dual shrink on D_R per Eq. 6; regression-sensitive dark-field path."),
    ("OpenCV double-transpose resize", "cv2.resize(img.T, ...).T matches OpenCV column-major convention."),
    ("Power-iteration SVD (Torch)", "Torch backend uses batched power iteration for σ₁; no full GPU SVD."),
)


def _traceability_row_to_dict(row: TraceabilityRow) -> dict[str, Any]:
    return {
        "paper_step": row.paper_step,
        "paper_ref": row.paper_ref,
        "code_refs": list(row.code_refs),
        "status": row.status,
        "intentional": row.intentional,
        "notes": row.notes,
    }


def _micro_benchmark_to_dict(result: MicroBenchmarkResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "working_size": result.working_size,
        "n_images": result.n_images,
        "backend": result.backend,
        "max_reweighting_iterations": result.max_reweighting_iterations,
        "prepare_ms": result.prepare_ms,
        "run_ms": result.run_ms,
        "normalize_ms": result.normalize_ms,
        "peak_bytes": result.peak_bytes,
    }


def _audit_finding_to_dict(finding: AuditFinding) -> dict[str, Any]:
    return {
        "finding_id": finding.finding_id,
        "rank": finding.rank,
        "description": finding.description,
        "fix_class": finding.fix_class,
        "measured_cost_ms": finding.measured_cost_ms,
        "measured_cost_pct": finding.measured_cost_pct,
        "paper_ref": finding.paper_ref,
        "code_refs": list(finding.code_refs),
        "phase9_change_class": finding.phase9_change_class,
        "phase11_priority": finding.phase11_priority,
        "suggested_action": finding.suggested_action,
    }


def _flatten_profile_hotspots(profile_hotspots: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, payload in profile_hotspots.items():
        if not isinstance(payload, dict):
            continue
        hotspots = payload.get("hotspots", ())
        if not isinstance(hotspots, list):
            continue
        for hotspot in hotspots:
            if not isinstance(hotspot, dict):
                continue
            rows.append(
                {
                    "phase": str(phase),
                    "function": str(hotspot.get("function", "")),
                    "cumtime_s": float(hotspot.get("cumtime_s", 0.0)),
                    "fix_class": str(hotspot.get("fix_class", "algorithm")),
                }
            )
    return rows


def _total_micro_benchmark_ms(results: Sequence[MicroBenchmarkResult]) -> float:
    total = 0.0
    for result in results:
        total += result.prepare_ms + result.run_ms + result.normalize_ms
    return total


def _default_catalog_findings(
    traceability: Sequence[TraceabilityRow],
    micro_benchmarks: Sequence[MicroBenchmarkResult],
) -> list[AuditFinding]:
    total_ms = _total_micro_benchmark_ms(micro_benchmarks)
    run_ms = sum(r.run_ms for r in micro_benchmarks)
    run_pct = (run_ms / total_ms * 100.0) if total_ms > 0 else 0.0

    findings: list[AuditFinding] = [
        AuditFinding(
            finding_id="missing-eq8-two-step-b",
            rank=1,
            description="No unsorted-matrix pass for per-frame B after S*,D*",
            fix_class="algorithm",
            measured_cost_ms=0.0,
            measured_cost_pct=0.0,
            paper_ref="Eq. 8-9",
            code_refs=("linum_basic/core.py:404", "linum_basic/core.py:405"),
            phase9_change_class="algorithm",
            phase11_priority="high",
            suggested_action="Search git for partial Eq. 8 port; evaluate mosaic vs time-lapse need",
        ),
        AuditFinding(
            finding_id="reweighting-iter-production-500",
            rank=2,
            description="Harness uses max_reweighting_iterations=500 vs BaSiC class default 10",
            fix_class="algorithm",
            measured_cost_ms=run_ms,
            measured_cost_pct=run_pct,
            paper_ref="Methods",
            code_refs=("linum_basic/core.py:204", "scripts/benchmark_speedup.py:447"),
            phase9_change_class="harness_defaults",
            phase11_priority="medium",
            suggested_action="Phase 9 forensics: when did 500 become production default",
        ),
    ]

    next_rank = 3
    for row in traceability:
        if row.intentional or row.status not in {"GAP", "PARTIAL"}:
            continue
        if "Two-step B" in row.paper_step:
            continue
        finding_id = row.paper_step.lower().replace(" ", "-").replace("/", "-")
        findings.append(
            AuditFinding(
                finding_id=finding_id,
                rank=next_rank,
                description=row.notes,
                fix_class="algorithm",
                measured_cost_ms=0.0,
                measured_cost_pct=0.0,
                paper_ref=row.paper_ref,
                code_refs=row.code_refs,
                phase9_change_class="algorithm",
                phase11_priority="medium",
                suggested_action=f"Review traceability gap for {row.paper_step}",
            )
        )
        next_rank += 1

    findings.sort(key=lambda item: item.rank)
    for index, finding in enumerate(findings, start=1):
        if finding.rank != index:
            findings[index - 1] = dataclasses.replace(finding, rank=index)
    return findings


def _build_phase9_handoff() -> dict[str, Any]:
    return {
        "forensics_hypotheses": list(_PHASE9_HANDOFF_FORENSICS_HYPOTHESES),
        "fore02_change_classes": list(_PHASE9_FORE02_CHANGE_CLASSES),
        "deferred_to_phase9": list(_PHASE9_DEFERRED),
    }


def _project_root() -> Path:
    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


def _resolve_confined_output_dir(
    output_dir: Path | str,
    *,
    allowed_base: Path | str | None = None,
) -> Path:
    base = Path(allowed_base).resolve() if allowed_base is not None else _project_root()
    resolved = Path(output_dir).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        msg = f"Output directory {resolved} escapes allowed base {base}"
        raise ValueError(msg) from exc
    return resolved


def build_audit_report(
    traceability: Sequence[TraceabilityRow],
    micro_benchmarks: Sequence[MicroBenchmarkResult],
    profile_hotspots: Mapping[str, Any],
    *,
    ranked_findings: Sequence[AuditFinding] | None = None,
) -> dict[str, Any]:
    """Assemble the Phase 8.1 audit machine contract (ALGO-04).

    Parameters
    ----------
    traceability : sequence of TraceabilityRow
        Paper-to-code matrix rows from :func:`build_traceability_matrix`.
    micro_benchmarks : sequence of MicroBenchmarkResult
        Synthetic micro-benchmark timing rows.
    profile_hotspots : mapping
        Nested prepare/run/normalize profile payload from
        :func:`profile_prepare_run`.
    ranked_findings : sequence of AuditFinding or None, optional
        Explicit ranked findings; when omitted, a default catalog is derived
        from traceability GAP/PARTIAL rows and harness amplification notes.

    Returns
    -------
    dict
        Top-level audit report with ``schema_version`` ``8.1.0``.
    """
    traceability_rows = [_traceability_row_to_dict(row) for row in traceability]
    micro_rows = [_micro_benchmark_to_dict(result) for result in micro_benchmarks]
    profile_rows = _flatten_profile_hotspots(dict(profile_hotspots))

    if ranked_findings is None:
        resolved_findings = _default_catalog_findings(traceability, micro_benchmarks)
    else:
        resolved_findings = list(ranked_findings)

    filtered_findings = [
        finding
        for finding in resolved_findings
        if finding.finding_id not in _INVARIANT_FINDING_IDS
        and not any(
            row.intentional and row.paper_step.lower() in finding.description.lower()
            for row in traceability
            if row.intentional
        )
    ]
    filtered_findings.sort(key=lambda item: item.rank)
    finding_rows = [_audit_finding_to_dict(finding) for finding in filtered_findings]

    return {
        "schema_version": AUDIT_REPORT_SCHEMA_VERSION,
        "git_commit": collect_git_commit(),
        "traceability": [dict(row) for row in traceability_rows],
        "micro_benchmarks": [dict(row) for row in micro_rows],
        "profile_hotspots": [dict(row) for row in profile_rows],
        "ranked_findings": [dict(row) for row in finding_rows],
        "phase9_handoff": dict(_build_phase9_handoff()),
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._\n"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(row[h]) for h in headers) + " |" for row in rows),
    ]
    return "\n".join(lines) + "\n"


def _render_audit_markdown(report: Mapping[str, Any]) -> str:
    findings = list(report.get("ranked_findings", ()))
    traceability = list(report.get("traceability", ()))
    micro = list(report.get("micro_benchmarks", ()))
    hotspots = list(report.get("profile_hotspots", ()))
    handoff = dict(report.get("phase9_handoff", {}))

    architecture_count = sum(1 for row in traceability if row.get("status") in {"GAP", "PARTIAL"})
    top_finding = findings[0]["description"] if findings else "No ranked findings"
    summary = (
        f"Phase 8.1 audit on commit `{report.get('git_commit', 'unknown')}`. "
        f"Primary hypothesis: slowness is rooted in computational architecture "
        f"({architecture_count} traceability gaps/partials) rather than parameter tuning alone. "
        f"Top finding: {top_finding}."
    )

    handoff_lines = "\n".join(f"- {item}" for item in handoff.get("forensics_hypotheses", ()))
    deferred_lines = "\n".join(f"- {item}" for item in handoff.get("deferred_to_phase9", ()))
    fore02 = ", ".join(handoff.get("fore02_change_classes", ()))

    invariant_lines = "\n".join(
        f"- **{title}:** {detail} Do not change in Phase 11 without a full regression plan."
        for title, detail in _INTENTIONAL_INVARIANTS
    )

    sections = [
        "# BaSiC Algorithmic Foundation Audit Report\n",
        "## Executive summary\n",
        summary + "\n",
        "## Traceability matrix\n",
        _markdown_table(traceability),
        "## Micro-benchmark results\n",
        _markdown_table(micro),
        "## Profiling hot paths\n",
        _markdown_table(hotspots[:10]),
        "## Ranked findings\n",
        _markdown_table(findings),
        "## Phase 9 handoff\n",
        "### Forensics hypotheses\n",
        handoff_lines + "\n",
        f"### FORE-02 change classes\n\n{fore02}\n",
        "### Deferred to Phase 9\n",
        deferred_lines + "\n",
        "## Intentional invariants appendix\n",
        invariant_lines + "\n",
    ]
    return "\n".join(sections)


def write_audit_artifacts(
    output_dir: Path | str,
    report: Mapping[str, Any],
    *,
    allowed_base: Path | str | None = None,
) -> None:
    """Write ``audit-report.json`` and ``audit-report.md`` under *output_dir* (ALGO-04).

    Parameters
    ----------
    output_dir : Path or str
        Destination directory for both artifacts.
    report : mapping
        Report dict from :func:`build_audit_report`.
    allowed_base : Path or str or None, optional
        Directory that *output_dir* must stay within after
        :meth:`pathlib.Path.resolve`. Defaults to the project root containing
        ``pyproject.toml``.

    Raises
    ------
    ValueError
        When the resolved output path escapes *allowed_base*.
    """
    resolved_dir = _resolve_confined_output_dir(output_dir, allowed_base=allowed_base)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    json_path = resolved_dir / "audit-report.json"
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    json_path.write_text(json_text, encoding="utf-8")

    md_path = resolved_dir / "audit-report.md"
    md_path.write_text(_render_audit_markdown(report), encoding="utf-8")
