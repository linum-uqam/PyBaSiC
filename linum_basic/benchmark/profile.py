"""Profiler bottleneck taxonomy and ranked optimization lever reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from linum_basic.benchmark.artifacts import BaselineBundle, CandidateArtifact
from linum_basic.benchmark.sweep import SPEED_RATIO_THRESHOLD

BottleneckClass = Literal[
    "memory-bandwidth",
    "compute",
    "compile-shape",
    "synchronization",
    "chunking",
]

BOTTLENECK_MEMORY_BANDWIDTH: BottleneckClass = "memory-bandwidth"
BOTTLENECK_COMPUTE: BottleneckClass = "compute"
BOTTLENECK_COMPILE_SHAPE: BottleneckClass = "compile-shape"
BOTTLENECK_SYNCHRONIZATION: BottleneckClass = "synchronization"
BOTTLENECK_CHUNKING: BottleneckClass = "chunking"

__all__ = [
    "BOTTLENECK_CHUNKING",
    "BOTTLENECK_COMPILE_SHAPE",
    "BOTTLENECK_COMPUTE",
    "BOTTLENECK_MEMORY_BANDWIDTH",
    "BOTTLENECK_SYNCHRONIZATION",
    "BottleneckClass",
    "BottleneckHotspot",
    "BottleneckReport",
    "HistoricalBaseline",
    "LeverAttemptRow",
    "LeverAttemptTable",
    "RankedLever",
    "build_bottleneck_report",
    "build_forensics_bottleneck_report",
    "build_forensics_change_attribution",
    "build_forensics_recovery_levers",
    "build_forensics_report",
    "build_lever_attempt_table",
    "build_phase3_handoff_config",
    "build_phase5_backlog",
    "build_phase5_fast_path",
    "build_phase5_optimization_report",
    "build_phase6_concurrency_verdict",
    "build_phase7_integration_summary",
    "build_regression_triage_result",
    "compute_stack_speed_ratio",
    "diagnose_regression_triage",
    "is_fast_era",
    "load_historical_baselines_from_harness_candidate",
    "load_historical_baselines_from_iteration_ab",
    "warn_git_commit_drift",
    "write_forensics_report_bundle",
]

_LEVER_DEFINITIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "sync-cadence",
        "linum_basic/_alm.py",
        "medium",
        "Tune convergence_check_every sync amortization; full gate stack required.",
    ),
    (
        "compile-surfacing",
        "linum_basic/_alm.py",
        "low",
        "Surface torch.compile fallback warnings and precision metadata (D-14, D-15).",
    ),
    (
        "reweighting-tolerance",
        "linum_basic/core.py",
        "medium",
        "Early-exit reweighting when outer loop plateaus (D-08).",
    ),
    (
        "dct-kernel-tuning",
        "linum_basic/backend.py",
        "medium",
        "Optimize DCT matmul path inside compiled ALM step.",
    ),
    (
        "compile-shape-stability",
        "linum_basic/_alm.py",
        "medium",
        "Stabilize torch.compile guards and mu tensor shape to avoid recompile storms.",
    ),
    (
        "inductor-cache-warm-policy",
        "linum_basic/_torch_cache.py",
        "low",
        "Extra untimed warm fits before measured benchmark repeats for steady-state timing.",
    ),
    (
        "tile-subsampling",
        "linum_basic/tuning.py",
        "high",
        "Last-resort tile subsampling when prepare/update dominates (D-07).",
    ),
)

_PRIMARY_LEVER_ORDER: dict[BottleneckClass, tuple[str, ...]] = {
    BOTTLENECK_COMPUTE: (
        "dct-kernel-tuning",
        "inductor-cache-warm-policy",
        "sync-cadence",
        "compile-surfacing",
        "compile-shape-stability",
        "reweighting-tolerance",
        "tile-subsampling",
    ),
    BOTTLENECK_SYNCHRONIZATION: (
        "sync-cadence",
        "reweighting-tolerance",
        "dct-kernel-tuning",
        "inductor-cache-warm-policy",
        "compile-surfacing",
        "compile-shape-stability",
        "tile-subsampling",
    ),
    BOTTLENECK_MEMORY_BANDWIDTH: (
        "tile-subsampling",
        "sync-cadence",
        "dct-kernel-tuning",
        "inductor-cache-warm-policy",
        "compile-surfacing",
        "compile-shape-stability",
        "reweighting-tolerance",
    ),
    BOTTLENECK_COMPILE_SHAPE: (
        "compile-surfacing",
        "compile-shape-stability",
        "inductor-cache-warm-policy",
        "sync-cadence",
        "dct-kernel-tuning",
        "reweighting-tolerance",
        "tile-subsampling",
    ),
    BOTTLENECK_CHUNKING: (
        "tile-subsampling",
        "sync-cadence",
        "dct-kernel-tuning",
        "inductor-cache-warm-policy",
        "compile-surfacing",
        "compile-shape-stability",
        "reweighting-tolerance",
    ),
}


@dataclass(frozen=True, slots=True, init=False)
class BottleneckHotspot:
    """One ranked profiler hotspot with taxonomy classification.

    Attributes
    ----------
    name : str
        Profiler op or kernel name.
    self_cuda_time_ms : float
        Self CUDA time in milliseconds from profiler aggregates.
    bottleneck_class : str
        Taxonomy class for this hotspot.
    evidence : str
        Short human-readable classification rationale.
    """

    name: str
    self_cuda_time_ms: float
    bottleneck_class: str
    evidence: str

    def __init__(
        self,
        name: str,
        self_cuda_time_ms: float,
        bottleneck_class: str,
        evidence: str,
    ) -> None:
        """Initialise a bottleneck hotspot.

        Parameters
        ----------
        name : str
            Profiler op or kernel name.
        self_cuda_time_ms : float
            Self CUDA time in milliseconds from profiler aggregates.
        bottleneck_class : str
            Taxonomy class for this hotspot.
        evidence : str
            Short human-readable classification rationale.
        """
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "self_cuda_time_ms", self_cuda_time_ms)
        object.__setattr__(self, "bottleneck_class", bottleneck_class)
        object.__setattr__(self, "evidence", evidence)


@dataclass(frozen=True, slots=True, init=False)
class RankedLever:
    """One optimization lever ranked for Phase 3 attempts.

    Attributes
    ----------
    lever_id : str
        Stable lever identifier for harness overrides.
    target_file : str
        Primary source file for the lever change.
    priority : int
        Attempt order; lower values are higher priority.
    expected_risk : str
        Qualitative regression risk label.
    gate_notes : str
        Gate or scope notes for operators.
    """

    lever_id: str
    target_file: str
    priority: int
    expected_risk: str
    gate_notes: str

    def __init__(
        self,
        lever_id: str,
        target_file: str,
        priority: int,
        expected_risk: str,
        gate_notes: str,
    ) -> None:
        """Initialise a ranked optimization lever.

        Parameters
        ----------
        lever_id : str
            Stable lever identifier for harness overrides.
        target_file : str
            Primary source file for the lever change.
        priority : int
            Attempt order; lower values are higher priority.
        expected_risk : str
            Qualitative regression risk label.
        gate_notes : str
            Gate or scope notes for operators.
        """
        object.__setattr__(self, "lever_id", lever_id)
        object.__setattr__(self, "target_file", target_file)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "expected_risk", expected_risk)
        object.__setattr__(self, "gate_notes", gate_notes)


@dataclass(frozen=True, slots=True, init=False)
class BottleneckReport:
    """Sequential ws=128 profiler bottleneck report (OPT-01).

    Attributes
    ----------
    schema_version : str
        Report schema version tag.
    working_size : int
        BaSiC working_size used during profiling.
    primary_limit : str
        Dominant bottleneck taxonomy class.
    hotspots : tuple of BottleneckHotspot
        Ranked profiler hotspots.
    ranked_levers : tuple of RankedLever
        Optimization levers sorted by ascending priority.
    profiler_summary : dict
        Raw profiler aggregate metadata preserved for audit.
    """

    schema_version: str
    working_size: int
    primary_limit: str
    hotspots: tuple[BottleneckHotspot, ...]
    ranked_levers: tuple[RankedLever, ...]
    profiler_summary: dict[str, Any]

    def __init__(
        self,
        schema_version: str,
        working_size: int,
        primary_limit: str,
        hotspots: tuple[BottleneckHotspot, ...],
        ranked_levers: tuple[RankedLever, ...],
        profiler_summary: dict[str, Any],
    ) -> None:
        """Initialise a bottleneck report.

        Parameters
        ----------
        schema_version : str
            Report schema version tag.
        working_size : int
            BaSiC working_size used during profiling.
        primary_limit : str
            Dominant bottleneck taxonomy class.
        hotspots : tuple of BottleneckHotspot
            Ranked profiler hotspots.
        ranked_levers : tuple of RankedLever
            Optimization levers sorted by ascending priority.
        profiler_summary : dict
            Raw profiler aggregate metadata preserved for audit.
        """
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "working_size", working_size)
        object.__setattr__(self, "primary_limit", primary_limit)
        object.__setattr__(self, "hotspots", hotspots)
        object.__setattr__(self, "ranked_levers", ranked_levers)
        object.__setattr__(self, "profiler_summary", profiler_summary)


def _require_key_averages(profiler_events: dict[str, Any]) -> list[dict[str, Any]]:
    raw = profiler_events.get("key_averages")
    if not isinstance(raw, list):
        msg = "profiler_events missing key_averages list"
        raise KeyError(msg)
    rows: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        time_ms = float(entry.get("self_cuda_time_total", 0.0))
        if time_ms <= 0.0:
            continue
        rows.append(entry)
    return rows


def _classify_op(name: str, *, category: str | None = None) -> BottleneckClass:
    lowered = name.lower()
    if category is not None and category.lower() == "chunking":
        return BOTTLENECK_CHUNKING
    if "chunk" in lowered and ("cat" in lowered or "batch" in lowered):
        return BOTTLENECK_CHUNKING
    if any(token in lowered for token in ("inductor", "compile", "triton_compile")):
        return BOTTLENECK_COMPILE_SHAPE
    if any(token in lowered for token in ("cudadevicesynchronize", "synchronize", "item")):
        return BOTTLENECK_SYNCHRONIZATION
    if lowered.endswith("::norm") or lowered == "aten::norm":
        return BOTTLENECK_SYNCHRONIZATION
    if any(token in lowered for token in ("memcpy", "copy_", "cat", "clone", "contiguous")):
        return BOTTLENECK_MEMORY_BANDWIDTH
    if any(token in lowered for token in ("::mm", "::bmm", "matmul", "addmm")):
        return BOTTLENECK_COMPUTE
    return BOTTLENECK_COMPUTE


def _classification_evidence(name: str, bottleneck_class: BottleneckClass) -> str:
    return f"{name} classified as {bottleneck_class} from op-name heuristics"


def _rank_levers(primary_limit: BottleneckClass) -> tuple[RankedLever, ...]:
    order = _PRIMARY_LEVER_ORDER.get(primary_limit, _PRIMARY_LEVER_ORDER[BOTTLENECK_COMPUTE])
    catalog = {lever_id: (target, risk, notes) for lever_id, target, risk, notes in _LEVER_DEFINITIONS}
    levers: list[RankedLever] = []
    for priority, lever_id in enumerate(order, start=1):
        target, risk, notes = catalog[lever_id]
        levers.append(
            RankedLever(
                lever_id=lever_id,
                target_file=target,
                priority=priority,
                expected_risk=risk,
                gate_notes=notes,
            )
        )
    return tuple(levers)


def build_bottleneck_report(
    profiler_events: dict[str, Any],
    *,
    working_size: int = 128,
) -> BottleneckReport:
    """Build an OPT-01 bottleneck report from profiler key_averages export.

    Parameters
    ----------
    profiler_events : dict
        Fixture or export dict with a ``key_averages`` list of op rows.
    working_size : int, optional
        BaSiC working_size used during profiling (default 128).

    Returns
    -------
    BottleneckReport
        Taxonomy-classified hotspots and ranked optimization levers.
    """
    rows = _require_key_averages(profiler_events)
    class_totals: dict[BottleneckClass, float] = dict.fromkeys(_PRIMARY_LEVER_ORDER, 0.0)
    hotspots: list[BottleneckHotspot] = []

    for entry in rows:
        name = str(entry["name"])
        time_ms = float(entry["self_cuda_time_total"])
        category = entry.get("category")
        category_str = str(category) if category is not None else None
        bottleneck_class = _classify_op(name, category=category_str)
        class_totals[bottleneck_class] += time_ms
        hotspots.append(
            BottleneckHotspot(
                name=name,
                self_cuda_time_ms=time_ms,
                bottleneck_class=bottleneck_class,
                evidence=_classification_evidence(name, bottleneck_class),
            )
        )

    hotspots.sort(key=lambda hotspot: hotspot.self_cuda_time_ms, reverse=True)

    if class_totals and any(total > 0.0 for total in class_totals.values()):
        primary_limit = max(class_totals.items(), key=lambda item: item[1])[0]
    else:
        primary_limit = BOTTLENECK_COMPUTE

    summary = profiler_events.get("summary")
    profiler_summary: dict[str, Any] = dict(summary) if isinstance(summary, dict) else {}
    profiler_summary["class_time_ms"] = dict(class_totals)

    return BottleneckReport(
        schema_version="1",
        working_size=working_size,
        primary_limit=primary_limit,
        hotspots=tuple(hotspots),
        ranked_levers=_rank_levers(primary_limit),
        profiler_summary=profiler_summary,
    )


@dataclass(frozen=True, slots=True, init=False)
class LeverAttemptRow:
    """One ranked optimization lever attempt row (OPT-03 partial).

    Attributes
    ----------
    lever_id : str
        Stable lever identifier from the bottleneck ranked table.
    run_label : str
        Operator run label for the candidate attempt.
    artifact_id : str
        Saved candidate artifact identifier.
    overall : str
        Harness overall verdict (``promote`` or ``reject``).
    quality_passed : bool
        Whether the Phase 1 quality gate passed.
    speed_passed : bool
        Whether steady-state timing meets the adoption threshold.
    steady_state_ms : float
        Median steady-state fit time in milliseconds.
    speed_ratio : float
        Baseline steady-state ms divided by candidate steady-state ms.
    overrides_applied : dict
        Allowlisted overrides applied for this lever attempt.
    quality_failures : tuple of str
        First-class quality metrics that failed the gate for this attempt.
    """

    lever_id: str
    run_label: str
    artifact_id: str
    overall: str
    quality_passed: bool
    speed_passed: bool
    steady_state_ms: float
    speed_ratio: float
    overrides_applied: dict[str, Any]
    quality_failures: tuple[str, ...]

    def __init__(
        self,
        lever_id: str,
        run_label: str,
        artifact_id: str,
        overall: str,
        quality_passed: bool,
        speed_passed: bool,
        steady_state_ms: float,
        speed_ratio: float,
        overrides_applied: dict[str, Any],
        quality_failures: tuple[str, ...] = (),
    ) -> None:
        """Initialise one lever attempt row."""
        object.__setattr__(self, "lever_id", lever_id)
        object.__setattr__(self, "run_label", run_label)
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "overall", overall)
        object.__setattr__(self, "quality_passed", quality_passed)
        object.__setattr__(self, "speed_passed", speed_passed)
        object.__setattr__(self, "steady_state_ms", steady_state_ms)
        object.__setattr__(self, "speed_ratio", speed_ratio)
        object.__setattr__(self, "overrides_applied", dict(overrides_applied))
        object.__setattr__(self, "quality_failures", quality_failures)


@dataclass(frozen=True, slots=True, init=False)
class LeverAttemptTable:
    """Aggregated promote/reject table for ranked optimization levers.

    Attributes
    ----------
    schema_version : str
        Table schema version tag.
    baseline_id : str
        Referenced baseline bundle id.
    rows : tuple of LeverAttemptRow
        Attempt rows ordered by ``ranked_levers`` priority.
    stacked_overrides : dict
        Cumulative overrides from promoted attempts (D-11 prep).
    stack_speed_ratio : float or None
        Cumulative stack speed ratio vs baseline when a stack candidate is supplied (D-04).
    """

    schema_version: str
    baseline_id: str
    rows: tuple[LeverAttemptRow, ...]
    stacked_overrides: dict[str, Any]
    stack_speed_ratio: float | None

    def __init__(
        self,
        schema_version: str,
        baseline_id: str,
        rows: tuple[LeverAttemptRow, ...],
        stacked_overrides: dict[str, Any],
        stack_speed_ratio: float | None = None,
    ) -> None:
        """Initialise a lever attempt table."""
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "baseline_id", baseline_id)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "stacked_overrides", dict(stacked_overrides))
        object.__setattr__(self, "stack_speed_ratio", stack_speed_ratio)


def _lever_attempt_steady_state_ms(metadata: dict[str, Any], *, artifact_label: str) -> float:
    telemetry = metadata.get("telemetry")
    if not isinstance(telemetry, dict):
        msg = f"{artifact_label} metadata missing telemetry block"
        raise KeyError(msg)
    if "steady_state_ms" not in telemetry:
        msg = f"{artifact_label} telemetry missing steady_state_ms"
        raise KeyError(msg)
    return float(telemetry["steady_state_ms"])


def _lever_attempt_speed_ratio(*, baseline_ms: float, candidate_ms: float) -> float:
    if candidate_ms <= 0:
        return float("inf") if baseline_ms > 0 else 1.0
    return baseline_ms / candidate_ms


def compute_stack_speed_ratio(*, baseline_ms: float, candidate_ms: float) -> float:
    """Compute cumulative stack speed ratio (baseline_ms / stack_candidate_ms).

    Uses the same convention as per-lever ``speed_ratio`` in ``LeverAttemptRow``.

    Parameters
    ----------
    baseline_ms : float
        Steady-state baseline wall time in milliseconds.
    candidate_ms : float
        Steady-state stack-candidate wall time in milliseconds.

    Returns
    -------
    float
        Ratio ``baseline_ms / candidate_ms`` (values above 1.0 mean faster).
    """
    return _lever_attempt_speed_ratio(baseline_ms=baseline_ms, candidate_ms=candidate_ms)


def compute_stack_speed_ratio_for_artifacts(
    baseline: BaselineBundle,
    stack_candidate: CandidateArtifact,
) -> float:
    """Compute stack speed ratio for a stack candidate against a baseline bundle.

    Parameters
    ----------
    baseline : BaselineBundle
        Reference baseline artifact bundle.
    stack_candidate : CandidateArtifact
        Stack candidate whose ``baseline_id`` must match *baseline*.

    Returns
    -------
    float
        Cumulative stack speed ratio from steady-state telemetry metadata.
    """
    if stack_candidate.baseline_id != baseline.baseline_id:
        msg = f"stack candidate baseline_id {stack_candidate.baseline_id!r} does not match baseline {baseline.baseline_id!r}"
        raise ValueError(msg)
    baseline_ms = _lever_attempt_steady_state_ms(baseline.metadata, artifact_label=baseline.baseline_id)
    stack_ms = _lever_attempt_steady_state_ms(
        stack_candidate.metadata,
        artifact_label=stack_candidate.candidate_id,
    )
    return compute_stack_speed_ratio(baseline_ms=baseline_ms, candidate_ms=stack_ms)


def _resolve_overrides_applied(
    baseline: BaselineBundle,
    candidate: CandidateArtifact,
) -> dict[str, Any]:
    metadata = candidate.metadata
    raw = metadata.get("overrides_applied")
    if isinstance(raw, dict):
        return dict(raw)

    baseline_params = baseline.strategy_params
    candidate_params = candidate.strategy_params
    return {
        key: candidate_params[key]
        for key in candidate_params
        if key not in baseline_params or candidate_params[key] != baseline_params[key]
    }


def build_lever_attempt_table(
    baseline: BaselineBundle,
    candidates: Sequence[CandidateArtifact],
    *,
    ranked_levers: Sequence[RankedLever],
    speed_ratio_threshold: float = SPEED_RATIO_THRESHOLD,
    stack_candidate: CandidateArtifact | None = None,
) -> LeverAttemptTable:
    """Build a ranked lever attempt table from saved candidate artifacts.

    Parameters
    ----------
    baseline : BaselineBundle
        Saved ws=128 baseline bundle used for speed comparisons.
    candidates : sequence of CandidateArtifact
        Lever attempt candidate artifacts in attempt order.
    ranked_levers : sequence of RankedLever
        Ranked levers from the bottleneck report; paired by index with *candidates*.
    speed_ratio_threshold : float, optional
        Minimum speed ratio for ``speed_passed`` (default 1.30).
    stack_candidate : CandidateArtifact or None, optional
        Combined promoted-override stack candidate for cumulative speed ratio (D-04).

    Returns
    -------
    LeverAttemptTable
        Promote/reject rows, cumulative stacked overrides, and optional stack speed ratio.
    """
    if len(candidates) != len(ranked_levers):
        msg = f"candidate count ({len(candidates)}) must match ranked_levers count ({len(ranked_levers)})"
        raise ValueError(msg)

    baseline_ms = _lever_attempt_steady_state_ms(baseline.metadata, artifact_label=baseline.baseline_id)
    rows: list[LeverAttemptRow] = []
    stacked_overrides: dict[str, Any] = {}

    for candidate, lever in zip(candidates, ranked_levers, strict=True):
        quality_verdict = candidate.metadata.get("quality_verdict")
        if not isinstance(quality_verdict, dict) or "passed" not in quality_verdict:
            msg = f"{candidate.candidate_id} metadata missing quality_verdict.passed"
            raise KeyError(msg)
        overall = candidate.metadata.get("overall")
        if overall is None:
            msg = f"{candidate.candidate_id} metadata missing overall verdict"
            raise KeyError(msg)

        steady_state_ms = _lever_attempt_steady_state_ms(
            candidate.metadata,
            artifact_label=candidate.candidate_id,
        )
        speed_ratio = _lever_attempt_speed_ratio(baseline_ms=baseline_ms, candidate_ms=steady_state_ms)
        overrides_applied = _resolve_overrides_applied(baseline, candidate)
        raw_failures = quality_verdict.get("failures", ())
        quality_failures = tuple(str(metric) for metric in raw_failures) if isinstance(raw_failures, list) else ()

        rows.append(
            LeverAttemptRow(
                lever_id=lever.lever_id,
                run_label=candidate.run_label,
                artifact_id=candidate.candidate_id,
                overall=str(overall),
                quality_passed=bool(quality_verdict["passed"]),
                speed_passed=speed_ratio >= speed_ratio_threshold,
                steady_state_ms=steady_state_ms,
                speed_ratio=speed_ratio,
                overrides_applied=overrides_applied,
                quality_failures=quality_failures,
            )
        )
        if str(overall) == "promote":
            stacked_overrides.update(overrides_applied)

    stack_speed_ratio: float | None = None
    if stack_candidate is not None:
        stack_speed_ratio = compute_stack_speed_ratio_for_artifacts(baseline, stack_candidate)

    return LeverAttemptTable(
        schema_version="1",
        baseline_id=baseline.baseline_id,
        rows=tuple(rows),
        stacked_overrides=stacked_overrides,
        stack_speed_ratio=stack_speed_ratio,
    )


PHASE5_DEFERRED_LEVER_IDS: frozenset[str] = frozenset()

PHASE5_DEFERRAL_REASONS: dict[str, str] = {}


def _backlog_rationale_for_row(row: LeverAttemptRow) -> str:
    parts: list[str] = [f"Harness overall verdict: {row.overall}."]
    if not row.quality_passed:
        parts.append("Quality gate failed.")
    if not row.speed_passed:
        parts.append(f"Speed ratio {row.speed_ratio:.3f} below adoption threshold.")
    return " ".join(parts)


def build_phase5_backlog(
    ranked_levers: Sequence[RankedLever],
    attempt_table: LeverAttemptTable,
) -> dict[str, Any]:
    """Build Phase 5 backlog from rejected, blocked, and deferred levers (D-10).

    Levers never attempted in Phase 3 (for example ``inductor-cache-warm-policy``)
    emit explicit deferral rows for Phase 5 ALGO-01 continuity.

    Parameters
    ----------
    ranked_levers : sequence of RankedLever
        Ranked levers from the bottleneck report.
    attempt_table : LeverAttemptTable
        Aggregated promote/reject table from GPU lever attempts.

    Returns
    -------
    dict
        ``phase5-backlog.json`` payload with ``entries`` listing rationale strings.
    """
    attempted_ids = {row.lever_id for row in attempt_table.rows}
    entries: list[dict[str, Any]] = []

    for row in attempt_table.rows:
        if row.overall == "promote":
            continue
        entry: dict[str, Any] = {
            "lever_id": row.lever_id,
            "status": "rejected",
            "rationale": _backlog_rationale_for_row(row),
            "artifact_id": row.artifact_id,
        }
        entries.append(entry)

    for lever in ranked_levers:
        if lever.lever_id in attempted_ids:
            continue
        entries.append(
            {
                "lever_id": lever.lever_id,
                "status": "blocked",
                "rationale": f"Not attempted in Phase 3 (ranked priority {lever.priority}).",
                "deferral_reason": (f"Ranked lever {lever.lever_id} was not attempted before Phase 3 exit."),
            }
        )

    for lever_id in sorted(PHASE5_DEFERRED_LEVER_IDS):
        if lever_id in attempted_ids:
            continue
        entries.append(
            {
                "lever_id": lever_id,
                "status": "deferred",
                "rationale": PHASE5_DEFERRAL_REASONS[lever_id],
                "deferral_reason": PHASE5_DEFERRAL_REASONS[lever_id],
            }
        )

    return {
        "schema_version": "1",
        "baseline_id": attempt_table.baseline_id,
        "entries": entries,
    }


ALGO05_ENTRY_CLASSES: dict[str, tuple[str, ...]] = {
    "convergence-policy": ("reweighting-tolerance",),
    "sync-cadence": ("sync-cadence",),
    "tile-subsampling": ("tile-subsampling",),
    "precision-tf32": ("compile-surfacing",),
    "chunking": ("tile-subsampling",),
    "compile-shape": (
        "dct-kernel-tuning",
        "compile-shape-stability",
        "compile-surfacing",
        "inductor-cache-warm-policy",
    ),
}


def _lever_target_file(lever_id: str) -> str:
    catalog = {entry[0]: entry[1] for entry in _LEVER_DEFINITIONS}
    return catalog[lever_id]


def _metric_gate_passed(failures: tuple[str, ...], metric: str) -> bool:
    return metric not in failures


def _bottleneck_evidence_for_lever(lever_id: str, report: BottleneckReport) -> str | None:
    for lever in report.ranked_levers:
        if lever.lever_id == lever_id:
            return f"primary_limit={report.primary_limit}; ranked priority {lever.priority}; {lever.gate_notes}"
    return None


def _algo05_entry_class_coverage(attempted_ids: set[str]) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for entry_class, lever_ids in ALGO05_ENTRY_CLASSES.items():
        attempted = [lever_id for lever_id in lever_ids if lever_id in attempted_ids]
        coverage[entry_class] = {
            "lever_ids": list(lever_ids),
            "attempted": attempted,
            "carried_from_backlog": [lever_id for lever_id in lever_ids if lever_id not in attempted_ids],
        }
    return coverage


def build_phase5_optimization_report(
    attempt_table: LeverAttemptTable,
    *,
    bottleneck_report: BottleneckReport | None = None,
    stack_speed_ratio: float | None = None,
) -> dict[str, Any]:
    """Build ALGO-01 ranked optimization report from a lever attempt table.

    Parameters
    ----------
    attempt_table : LeverAttemptTable
        Promote/reject rows from GPU lever attempts.
    bottleneck_report : BottleneckReport or None, optional
        Refreshed profiler bottleneck report for promoted-row evidence (D-16).
    stack_speed_ratio : float or None, optional
        Cumulative stack ratio override; defaults to ``attempt_table.stack_speed_ratio``.

    Returns
    -------
    dict
        Ranked per-lever optimization evidence with quality and speed metrics.
    """
    resolved_stack_ratio = stack_speed_ratio
    if resolved_stack_ratio is None:
        resolved_stack_ratio = attempt_table.stack_speed_ratio

    attempted_ids = {row.lever_id for row in attempt_table.rows}
    entries: list[dict[str, Any]] = []

    for rank, row in enumerate(attempt_table.rows, start=1):
        overall = row.overall
        if overall == "promote" and not row.quality_passed:
            overall = "reject"

        entry: dict[str, Any] = {
            "rank": rank,
            "lever_id": row.lever_id,
            "target_file": _lever_target_file(row.lever_id),
            "artifact_id": row.artifact_id,
            "speed_ratio": row.speed_ratio,
            "steady_state_ms": row.steady_state_ms,
            "quality_passed": row.quality_passed,
            "seam_l1_passed": _metric_gate_passed(row.quality_failures, "seam_l1"),
            "seam_curvature_passed": _metric_gate_passed(row.quality_failures, "seam_curvature"),
            "speed_passed": row.speed_passed,
            "overall": overall,
            "overrides_applied": dict(row.overrides_applied),
        }
        if overall == "promote" and bottleneck_report is not None:
            entry["bottleneck_evidence"] = _bottleneck_evidence_for_lever(row.lever_id, bottleneck_report)
        if overall != "promote":
            entry["reject_rationale"] = _backlog_rationale_for_row(row)
        entries.append(entry)

    return {
        "schema_version": "1",
        "baseline_id": attempt_table.baseline_id,
        "stack_speed_ratio": resolved_stack_ratio,
        "stacked_overrides": dict(attempt_table.stacked_overrides),
        "entry_class_coverage": _algo05_entry_class_coverage(attempted_ids),
        "entries": entries,
    }


def build_phase3_handoff_config(
    promoted_overrides: dict[str, Any],
    *,
    baseline_id: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build stacked Phase 3 handoff config for Phase 4/5 (D-11).

    Parameters
    ----------
    promoted_overrides : dict
        Cumulative allowlisted overrides from promoted lever attempts.
    baseline_id : str
        Phase 1 ws=128 baseline bundle identifier (D-12).
    timestamp : str or None, optional
        ISO-8601 timestamp; defaults to current UTC when omitted.

    Returns
    -------
    dict
        ``phase3-handoff-config.json`` payload with ``working_size=128``.
    """
    from datetime import UTC, datetime

    return {
        "schema_version": "1",
        "working_size": 128,
        "baseline_id": baseline_id,
        "timestamp": timestamp or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "stacked_overrides": dict(promoted_overrides),
    }


DEFAULT_CODE_PATH_FLAGS: dict[str, Any] = {
    "dct_kernel": "default",
    "compile_mode": "default",
    "inductor_warm_passes": 0,
}


def build_phase5_fast_path(
    promoted_overrides: dict[str, Any],
    *,
    baseline_id: str,
    lever_stack: Sequence[str],
    code_path_flags: dict[str, Any] | None = None,
    evidence_artifact_ids: Sequence[str],
    git_commit: str | None = None,
    stack_speed_ratio: float | None = None,
    end_to_end_ms: float | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build frozen Phase 5 fast-path manifest for Phase 6/7 deploy (ALGO-04).

    Parameters
    ----------
    promoted_overrides : dict
        Cumulative allowlisted overrides from promoted lever attempts.
    baseline_id : str
        Phase 1 ws=128 baseline bundle identifier (D-12).
    lever_stack : sequence of str
        Ordered promoted lever ids from the winning stack (D-13).
    code_path_flags : dict or None, optional
        Active code-path lever settings; production defaults when omitted.
    evidence_artifact_ids : sequence of str
        Baseline and candidate artifact ids for audit traceability (D-14).
    git_commit : str or None, optional
        Git commit hash captured at manifest freeze time.
    stack_speed_ratio : float or None, optional
        Cumulative stack speed ratio vs baseline when measured.
    end_to_end_ms : float or None, optional
        End-to-end steady-state timing for the winning stack candidate.
    timestamp : str or None, optional
        ISO-8601 timestamp; defaults to current UTC when omitted.

    Returns
    -------
    dict
        ``phase5-fast-path.json`` payload with params, code-path flags, and
        ``no_optimization=true`` when no levers promoted.
    """
    from datetime import UTC, datetime

    no_optimization = not promoted_overrides and not lever_stack
    resolved_flags = dict(DEFAULT_CODE_PATH_FLAGS)
    if code_path_flags is not None:
        resolved_flags.update(code_path_flags)

    manifest: dict[str, Any] = {
        "schema_version": "1",
        "working_size": 128,
        "baseline_id": baseline_id,
        "timestamp": timestamp or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "stacked_overrides": dict(promoted_overrides),
        "lever_stack": list(lever_stack),
        "code_path_flags": resolved_flags,
        "evidence_artifact_ids": list(evidence_artifact_ids),
        "no_optimization": no_optimization,
    }
    if git_commit is not None:
        manifest["git_commit"] = git_commit
    if stack_speed_ratio is not None:
        manifest["stack_speed_ratio"] = stack_speed_ratio
    if end_to_end_ms is not None:
        manifest["end_to_end_ms"] = end_to_end_ms
    return manifest


_END_TO_END_MS_TOLERANCE = 1e-9

_STRATEGY_FORK_MODELS: dict[str, str] = {
    "multi": "maxForks_2_scalar_per_gpu",
    "batched": "maxForks_1_batched_multi_gpu",
}

_STRATEGY_MAX_FORKS: dict[str, int] = {
    "multi": 2,
    "batched": 1,
}


def _quality_passes(quality_verdict: Any) -> bool:
    if quality_verdict is None:
        return False
    if isinstance(quality_verdict, dict):
        return bool(quality_verdict.get("passed"))
    passed = getattr(quality_verdict, "passed", None)
    if passed is not None:
        return bool(passed)
    return bool(quality_verdict)


def _normalize_concurrency_mode(mode: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(mode, dict):
        raw = mode
    else:
        raw = {
            "strategy": getattr(mode, "strategy", None),
            "fork_model": getattr(mode, "fork_model", None),
            "end_to_end_ms": getattr(mode, "end_to_end_ms", None),
            "per_z_ms": getattr(mode, "per_z_ms", None),
            "steady_state_ms": getattr(mode, "steady_state_ms", None),
            "quality_verdict": getattr(mode, "quality_verdict", None),
            "artifact_id": getattr(mode, "artifact_id", None),
            "peak_vram_bytes": getattr(mode, "peak_vram_bytes", None),
            "gpu_map": getattr(mode, "gpu_map", None),
        }

    strategy = str(raw["strategy"])
    fork_model = raw.get("fork_model") or _STRATEGY_FORK_MODELS.get(strategy, strategy)
    peak_vram = raw.get("peak_vram_bytes")
    if peak_vram is None:
        peak_vram = raw.get("peak_vram")
    raw_gpu_map = raw.get("gpu_map")
    gpu_map: dict[str, str] | None = (
        {str(key): str(value) for key, value in raw_gpu_map.items()} if isinstance(raw_gpu_map, dict) else None
    )

    end_to_end_ms = raw["end_to_end_ms"]
    per_z_ms = raw["per_z_ms"]
    steady_state_ms = raw["steady_state_ms"]
    artifact_id = raw["artifact_id"]
    if end_to_end_ms is None or per_z_ms is None or steady_state_ms is None or artifact_id is None:
        msg = "concurrency mode requires end_to_end_ms, per_z_ms, steady_state_ms, and artifact_id"
        raise ValueError(msg)

    return {
        "strategy": strategy,
        "fork_model": str(fork_model),
        "end_to_end_ms": float(end_to_end_ms),
        "per_z_ms": float(per_z_ms),
        "steady_state_ms": float(steady_state_ms),
        "quality_verdict": raw["quality_verdict"],
        "artifact_id": str(artifact_id),
        "peak_vram_bytes": int(peak_vram or 0),
        "_gpu_map": gpu_map,
    }


def _select_concurrency_winner(
    modes: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    passing = [mode for mode in modes if _quality_passes(mode["quality_verdict"])]
    if not passing:
        return None, "all modes failed quality gate"

    def sort_key(mode: dict[str, Any]) -> tuple[float, int]:
        return (mode["end_to_end_ms"], mode["peak_vram_bytes"])

    winner = min(passing, key=sort_key)
    rationale = "lowest end_to_end_ms among quality-passing modes"

    tied = [mode for mode in passing if abs(mode["end_to_end_ms"] - winner["end_to_end_ms"]) <= _END_TO_END_MS_TOLERANCE]
    if len(tied) > 1:
        vram_values = {mode["peak_vram_bytes"] for mode in tied}
        if len(vram_values) > 1:
            rationale = "tied end_to_end_ms; selected lowest peak_vram_bytes for single-GPU VRAM headroom (D-16)"

    return (
        {
            "strategy": winner["strategy"],
            "fork_model": winner["fork_model"],
            "rationale": rationale,
        },
        None,
    )


def build_phase6_concurrency_verdict(
    modes: Sequence[dict[str, Any] | Any],
    *,
    baseline_id: str,
    phase5_fast_path_ref: str,
    evidence_artifact_ids: Sequence[str],
    git_commit: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build frozen Phase 6 concurrency verdict manifest for Phase 7 deploy (PERF-04).

    Returns
    -------
    dict[str, Any]
        Verdict manifest with ``recommended_max_forks``, ``gpu_allocation_map``,
        ``selection_rationale``, and per-mode evidence rows.
    """
    from datetime import UTC, datetime

    normalized_modes = [_normalize_concurrency_mode(mode) for mode in modes]
    winner, no_winner_rationale = _select_concurrency_winner(normalized_modes)

    recommended_max_forks: int | None
    gpu_allocation_map: dict[str, str] | None
    selection_rationale: str | None = no_winner_rationale

    if winner is None:
        recommended_max_forks = None
        gpu_allocation_map = None
    else:
        recommended_max_forks = _STRATEGY_MAX_FORKS.get(winner["strategy"])
        winner_mode = next(mode for mode in normalized_modes if mode["strategy"] == winner["strategy"])
        gpu_allocation_map = winner_mode.get("_gpu_map")

    public_modes = [{k: v for k, v in mode.items() if not k.startswith("_")} for mode in normalized_modes]

    manifest: dict[str, Any] = {
        "schema_version": "1",
        "baseline_id": baseline_id,
        "timestamp": timestamp or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "modes": public_modes,
        "winner": winner,
        "recommended_max_forks": recommended_max_forks,
        "gpu_allocation_map": gpu_allocation_map,
        "phase5_fast_path_ref": phase5_fast_path_ref,
        "evidence_artifact_ids": list(evidence_artifact_ids),
    }
    if git_commit is not None:
        manifest["git_commit"] = git_commit
    if selection_rationale is not None:
        manifest["selection_rationale"] = selection_rationale
    return manifest


_FORENSICS_SCHEMA_VERSION = "1"
_FAST_ERA_RATIO_THRESHOLD = 1.20

_DEFAULT_FORENSICS_WORKLOAD: dict[str, Any] = {
    "working_size": 128,
    "z_indices": [0, 13, 27, 40, 54],
    "estimate_darkfield": True,
    "max_reweighting_iterations": 500,
}


def is_fast_era(current_steady_state_ms: float, baseline_steady_state_ms: float) -> bool:
    """Return whether *baseline* qualifies as a fast era vs *current* timing (FORE-03, D-23).

    A baseline is a fast era when ``current_steady_state_ms / baseline_steady_state_ms``
    is at or above the locked 20% threshold (ratio >= 1.20).
    """
    if baseline_steady_state_ms <= 0:
        return False
    return current_steady_state_ms / baseline_steady_state_ms >= _FAST_ERA_RATIO_THRESHOLD


@dataclass(frozen=True, slots=True)
class HistoricalBaseline:
    """One historical timing era for D-23 regression triage (FORE-03).

    Attributes
    ----------
    git_commit : str
        Git commit hash for the era.
    steady_state_ms : float
        Median steady-state per-z fit time in milliseconds.
    end_to_end_ms : float or None
        Optional end-to-end wall time in milliseconds.
    artifact_id : str
        Source artifact identifier (for example ``iteration-ab-z27``).
    change_class : str or None
        FORE-02 change class when known (for example ``torch_compile_inductor``).
    is_fast_era : bool
        Precomputed fast-era flag for serialization; callers may recompute via
        :func:`is_fast_era`.
    """

    git_commit: str
    steady_state_ms: float
    end_to_end_ms: float | None
    artifact_id: str
    change_class: str | None
    is_fast_era: bool


def build_regression_triage_result(
    *,
    harness_overall: str,
    wallclock_regressed: bool,
    current_steady_state_ms: float,
    historical_baselines: Sequence[HistoricalBaseline],
) -> dict[str, Any]:
    """Extend D-23 regression triage with ``historical_baselines[]`` evidence (FORE-03).

    Returns the unchanged :func:`diagnose_regression_triage` enum plus historical
    baseline rows, fast-era commit hashes, and the best (fastest) fast-era baseline.
    """
    regression_triage = diagnose_regression_triage(
        harness_overall=harness_overall,
        wallclock_regressed=wallclock_regressed,
    )

    serialized_baselines: list[dict[str, Any]] = []
    fast_era_commits: list[str] = []
    best_fast_era: HistoricalBaseline | None = None

    for baseline in historical_baselines:
        fast = is_fast_era(current_steady_state_ms, baseline.steady_state_ms)
        row = HistoricalBaseline(
            git_commit=baseline.git_commit,
            steady_state_ms=baseline.steady_state_ms,
            end_to_end_ms=baseline.end_to_end_ms,
            artifact_id=baseline.artifact_id,
            change_class=baseline.change_class,
            is_fast_era=fast,
        )
        serialized_baselines.append(asdict(row))
        if fast:
            fast_era_commits.append(baseline.git_commit)
            if best_fast_era is None or baseline.steady_state_ms < best_fast_era.steady_state_ms:
                best_fast_era = row

    return {
        "regression_triage": regression_triage,
        "historical_baselines": serialized_baselines,
        "fast_era_commits": fast_era_commits,
        "historical_regression_detected": bool(fast_era_commits),
        "best_fast_era": asdict(best_fast_era) if best_fast_era is not None else None,
    }


def _parse_iteration_ab_era(
    era_key: str,
    era_payload: Any,
    *,
    artifact_id: str,
    current_steady_state_ms: float,
) -> HistoricalBaseline:
    if not isinstance(era_payload, dict):
        msg = f"{era_key} must be an object"
        raise ValueError(msg)
    if "git_commit" not in era_payload:
        msg = f"{era_key} missing required key git_commit"
        raise ValueError(msg)
    if "steady_state_ms" not in era_payload:
        msg = f"{era_key} missing required key steady_state_ms"
        raise ValueError(msg)

    steady_state_ms = float(era_payload["steady_state_ms"])
    if steady_state_ms <= 0:
        msg = f"{era_key}.steady_state_ms must be positive"
        raise ValueError(msg)

    end_to_end_raw = era_payload.get("end_to_end_ms")
    end_to_end_ms = float(end_to_end_raw) if end_to_end_raw is not None else None

    change_class = era_payload.get("change_class")
    if change_class is not None:
        change_class = str(change_class)

    git_commit = str(era_payload["git_commit"])
    return HistoricalBaseline(
        git_commit=git_commit,
        steady_state_ms=steady_state_ms,
        end_to_end_ms=end_to_end_ms,
        artifact_id=artifact_id,
        change_class=change_class,
        is_fast_era=is_fast_era(current_steady_state_ms, steady_state_ms),
    )


def load_historical_baselines_from_iteration_ab(
    path: Path | str,
    *,
    current_steady_state_ms: float,
) -> tuple[HistoricalBaseline, ...]:
    """Ingest ad-hoc server A/B JSON into :class:`HistoricalBaseline` rows (FORE-03, D-23).

    Reads existing ``iteration-ab-z*.json`` artifacts without running new GPU benchmarks.
    Unknown extra keys are ignored for forward compatibility with ad-hoc scripts.
    """
    import json

    resolved = Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{resolved} must contain a JSON object"
        raise ValueError(msg)

    artifact_id = resolved.stem
    rows: list[HistoricalBaseline] = []
    for era_key in ("pre_fix", "post_fix"):
        if era_key not in payload:
            msg = f"{resolved} missing required key {era_key}"
            raise ValueError(msg)
        try:
            rows.append(
                _parse_iteration_ab_era(
                    era_key,
                    payload[era_key],
                    artifact_id=artifact_id,
                    current_steady_state_ms=current_steady_state_ms,
                )
            )
        except ValueError as exc:
            msg = f"{resolved}: {exc}"
            raise ValueError(msg) from exc

    return tuple(rows)


def load_historical_baselines_from_harness_candidate(
    candidate_path: Path | str,
    compare_path: Path | str,
    *,
    current_steady_state_ms: float,
) -> tuple[HistoricalBaseline, ...]:
    """Ingest Phase 11 harness candidate/compare JSON into D-23 historical rows (D-12).

    Reads ``candidate-artifact.json`` telemetry and git metadata plus a companion
    ``compare-summary.json`` for forward-compatible validation. Speed verdict fields
    are not stored on the returned row.
    """
    import json

    candidate_file = Path(candidate_path)
    compare_file = Path(compare_path)

    payload = json.loads(candidate_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{candidate_file} must contain a JSON object"
        raise ValueError(msg)

    compare_payload = json.loads(compare_file.read_text(encoding="utf-8"))
    if not isinstance(compare_payload, dict):
        msg = f"{compare_file} must contain a JSON object"
        raise ValueError(msg)

    if "candidate_id" not in payload:
        msg = f"{candidate_file} missing required key candidate_id"
        raise ValueError(msg)

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        msg = f"{candidate_file} metadata missing required telemetry block"
        raise ValueError(msg)

    if "git_commit" not in metadata:
        msg = f"{candidate_file} metadata missing required key git_commit"
        raise ValueError(msg)

    telemetry = metadata.get("telemetry")
    if not isinstance(telemetry, dict):
        msg = f"{candidate_file} metadata missing required telemetry block"
        raise ValueError(msg)

    if "steady_state_ms" not in telemetry:
        msg = f"{candidate_file} telemetry missing required key steady_state_ms"
        raise ValueError(msg)

    steady_state_ms = float(telemetry["steady_state_ms"])
    if steady_state_ms <= 0:
        msg = f"{candidate_file} telemetry.steady_state_ms must be positive"
        raise ValueError(msg)

    end_to_end_raw = telemetry.get("end_to_end_ms")
    end_to_end_ms = float(end_to_end_raw) if end_to_end_raw is not None else None

    _ = compare_payload.get("speed_verdict")

    artifact_id = str(payload["candidate_id"])
    git_commit = str(metadata["git_commit"])

    row = HistoricalBaseline(
        git_commit=git_commit,
        steady_state_ms=steady_state_ms,
        end_to_end_ms=end_to_end_ms,
        artifact_id=artifact_id,
        change_class="torch_compile_inductor",
        is_fast_era=is_fast_era(current_steady_state_ms, steady_state_ms),
    )
    return (row,)


def build_forensics_bottleneck_report() -> dict[str, Any]:
    """Build a single-lever bottleneck report for Phase 11 worker-compile-off (D-01, PERF-01).

    Unlike :func:`build_bottleneck_report`, this does not require Phase 5 profiler output.
    Only the priority-1 ``worker-compile-off`` lever is included; ``auto-l-s-config`` is
    excluded because it is a companion config promotion, not a harness attempt row (D-03).
    """
    worker_lever = next(lever for lever in build_forensics_recovery_levers() if lever.lever_id == "worker-compile-off")
    return {
        "schema_version": "1",
        "working_size": 128,
        "primary_limit": BOTTLENECK_COMPILE_SHAPE,
        "hotspots": [],
        "ranked_levers": [asdict(worker_lever)],
    }


def build_forensics_recovery_levers() -> tuple[RankedLever, ...]:
    """Return ranked recovery levers from ad-hoc A6000 forensics (FORE-04).

    Priority 1: worker compile-off (``511c88c``); priority 2: auto ``l_s`` config.
    """
    return (
        RankedLever(
            lever_id="worker-compile-off",
            target_file="linum_basic/_parallel.py",
            priority=1,
            expected_risk="low",
            gate_notes=(
                "Set LINUM_BASIC_ALM_COMPILE_MODE=off in CUDA joblib workers (511c88c); "
                "validate seam_l1 quality gates before Phase 11 promotion."
            ),
        ),
        RankedLever(
            lever_id="auto-l-s-config",
            target_file="linumpy subject config",
            priority=2,
            expected_risk="low",
            gate_notes=(
                "Set fix_illum_smoothness_flatfield=null on subject configs for auto l_s; "
                "confirm seam_l1 non-regression vs fixed l_s=0.05."
            ),
        ),
    )


def build_forensics_change_attribution(
    historical_baselines: Sequence[HistoricalBaseline],
) -> list[dict[str, Any]]:
    """Map historical baselines to FORE-02 change-class attribution rows."""
    entries: list[dict[str, Any]] = []
    for baseline in historical_baselines:
        if baseline.change_class is None:
            continue
        summary = f"steady_state_ms={baseline.steady_state_ms:.1f} at {baseline.git_commit} ({baseline.artifact_id})"
        if baseline.end_to_end_ms is not None:
            summary += f"; end_to_end_ms={baseline.end_to_end_ms:.1f}"
        entries.append(
            {
                "change_class": baseline.change_class,
                "git_commit": baseline.git_commit,
                "steady_state_ms": baseline.steady_state_ms,
                "evidence_summary": summary,
            }
        )
    return entries


def build_forensics_report(
    *,
    slice_id: int,
    historical_baselines: Sequence[HistoricalBaseline],
    current_steady_state_ms: float,
    current_git_commit: str,
    harness_compare_overall: str = "promote",
    wallclock_regressed: bool = False,
    evidence_artifact_ids: Sequence[str],
    workload: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Assemble slim forensics-report.json manifest from ingested baselines (FORE-01)."""
    from datetime import UTC, datetime

    triage = build_regression_triage_result(
        harness_overall=harness_compare_overall,
        wallclock_regressed=wallclock_regressed,
        current_steady_state_ms=current_steady_state_ms,
        historical_baselines=historical_baselines,
    )
    resolved_workload = dict(_DEFAULT_FORENSICS_WORKLOAD)
    if workload is not None:
        resolved_workload.update(workload)

    return {
        "schema_version": _FORENSICS_SCHEMA_VERSION,
        "slice_id": slice_id,
        "workload": resolved_workload,
        "current_git_commit": current_git_commit,
        "historical_baselines": triage["historical_baselines"],
        "regression_triage": triage["regression_triage"],
        "historical_regression_detected": triage["historical_regression_detected"],
        "fast_era_commits": triage["fast_era_commits"],
        "best_fast_era": triage["best_fast_era"],
        "change_attribution": build_forensics_change_attribution(historical_baselines),
        "ranked_recovery_levers": [asdict(lever) for lever in build_forensics_recovery_levers()],
        "evidence_artifact_ids": list(evidence_artifact_ids),
        "timestamp": timestamp or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def write_forensics_report_bundle(output_dir: Path | str, report: dict[str, Any]) -> None:
    """Write ``forensics-report.json`` and ``forensics-report.md`` under *output_dir*.

    Default operator path on the A6000 server:
    ``/scratch/workspace/sub-22/runs/forensics/{slice_id}/``.
    """
    import json

    from linum_basic.benchmark.artifacts import write_summary_table

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "forensics-report.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = [
        {
            "era": "historical",
            "git_commit": baseline.get("git_commit", ""),
            "steady_state_ms": baseline.get("steady_state_ms", ""),
            "change_class": baseline.get("change_class", ""),
            "lever_id": "",
            "priority": "",
        }
        for baseline in report.get("historical_baselines", [])
    ]
    rows.extend(
        {
            "era": "recovery_lever",
            "git_commit": "",
            "steady_state_ms": "",
            "change_class": "",
            "lever_id": lever.get("lever_id", ""),
            "priority": lever.get("priority", ""),
        }
        for lever in report.get("ranked_recovery_levers", [])
    )

    md_path = out / "forensics-report.md"
    preamble = (
        "# Forensics report\n\n"
        "Root cause: CUDA joblib worker torch.compile default plus fixed l_s=0.05 in "
        "linumpy configs caused ~229 s/z regression; worker compile-off (511c88c) and "
        "auto l_s restore ~3-6 s/z with improved seam_l1.\n\n"
    )
    if rows:
        table_path = out / "_forensics-table.md"
        write_summary_table(table_path, rows, fmt="markdown")
        md_path.write_text(preamble + table_path.read_text(encoding="utf-8"), encoding="utf-8")
        table_path.unlink()
    else:
        md_path.write_text(preamble, encoding="utf-8")


def diagnose_regression_triage(*, harness_overall: str, wallclock_regressed: bool) -> str:
    """Classify regression source using harness verdict and wall-clock signal (D-23).

    Returns
    -------
    str
        One of ``algorithm_or_env_drift``, ``pipeline_orchestration_issue``, or
        ``no_regression``.
    """
    if harness_overall == "reject":
        return "algorithm_or_env_drift"
    if harness_overall == "promote":
        if wallclock_regressed:
            return "pipeline_orchestration_issue"
        return "no_regression"
    msg = f"unexpected harness_overall value: {harness_overall!r}; expected 'promote' or 'reject'"
    raise ValueError(msg)


def warn_git_commit_drift(*, manifest_git_commit: str | None, current_git_commit: str) -> str | None:
    """Return a non-blocking drift warning when manifest and HEAD commits differ (D-08).

    Returns
    -------
    str or None
        Warning message when commits differ; ``None`` when they match or manifest
        commit is unset.
    """
    if manifest_git_commit is None or manifest_git_commit == current_git_commit:
        return None
    return f"phase5-fast-path git_commit {manifest_git_commit!r} differs from current HEAD {current_git_commit!r}"


def build_phase7_integration_summary(
    *,
    baseline_id: str,
    phase5_fast_path_ref: str,
    harness_compare_overall: str,
    wallclock_regressed: bool,
    env_snapshot: dict[str, str],
    current_git_commit: str,
    evidence_artifact_ids: Sequence[str],
    fast_path_git_commit: str | None = None,
    strategy_metadata: dict[str, Any] | None = None,
    nextflow_wallclock_ms: float | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build Phase 7 integration summary manifest for env audit and regression triage (NFLO-01, NFLO-05).

    Returns
    -------
    dict[str, Any]
        Integration summary with env snapshot, harness verdict, regression triage,
        and optional Nextflow wall-clock timing.
    """
    from datetime import UTC, datetime

    manifest: dict[str, Any] = {
        "schema_version": "1",
        "baseline_id": baseline_id,
        "timestamp": timestamp or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "phase5_fast_path_ref": phase5_fast_path_ref,
        "env_snapshot": dict(env_snapshot),
        "git_commit": current_git_commit,
        "git_commit_drift_warning": warn_git_commit_drift(
            manifest_git_commit=fast_path_git_commit,
            current_git_commit=current_git_commit,
        ),
        "harness_compare_overall": harness_compare_overall,
        "wallclock_regressed": wallclock_regressed,
        "regression_triage": diagnose_regression_triage(
            harness_overall=harness_compare_overall,
            wallclock_regressed=wallclock_regressed,
        ),
        "evidence_artifact_ids": list(evidence_artifact_ids),
    }
    if strategy_metadata is not None:
        manifest["strategy_metadata"] = dict(strategy_metadata)
    if nextflow_wallclock_ms is not None:
        manifest["nextflow_wallclock_ms"] = nextflow_wallclock_ms
    return manifest
