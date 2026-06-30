"""Profiler bottleneck taxonomy and ranked optimization lever reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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
    "LeverAttemptRow",
    "LeverAttemptTable",
    "RankedLever",
    "build_bottleneck_report",
    "build_lever_attempt_table",
    "build_phase3_handoff_config",
    "build_phase5_backlog",
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
        "tile-subsampling",
        "linum_basic/tuning.py",
        "high",
        "Last-resort tile subsampling when prepare/update dominates (D-07).",
    ),
)

_PRIMARY_LEVER_ORDER: dict[BottleneckClass, tuple[str, ...]] = {
    BOTTLENECK_COMPUTE: (
        "dct-kernel-tuning",
        "sync-cadence",
        "compile-surfacing",
        "reweighting-tolerance",
        "tile-subsampling",
    ),
    BOTTLENECK_SYNCHRONIZATION: (
        "sync-cadence",
        "reweighting-tolerance",
        "dct-kernel-tuning",
        "compile-surfacing",
        "tile-subsampling",
    ),
    BOTTLENECK_MEMORY_BANDWIDTH: (
        "tile-subsampling",
        "sync-cadence",
        "dct-kernel-tuning",
        "compile-surfacing",
        "reweighting-tolerance",
    ),
    BOTTLENECK_COMPILE_SHAPE: (
        "compile-surfacing",
        "sync-cadence",
        "dct-kernel-tuning",
        "reweighting-tolerance",
        "tile-subsampling",
    ),
    BOTTLENECK_CHUNKING: (
        "tile-subsampling",
        "sync-cadence",
        "dct-kernel-tuning",
        "compile-surfacing",
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
    """

    schema_version: str
    baseline_id: str
    rows: tuple[LeverAttemptRow, ...]
    stacked_overrides: dict[str, Any]

    def __init__(
        self,
        schema_version: str,
        baseline_id: str,
        rows: tuple[LeverAttemptRow, ...],
        stacked_overrides: dict[str, Any],
    ) -> None:
        """Initialise a lever attempt table."""
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "baseline_id", baseline_id)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "stacked_overrides", dict(stacked_overrides))


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

    Returns
    -------
    LeverAttemptTable
        Promote/reject rows and cumulative stacked overrides.
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
            )
        )
        if str(overall) == "promote":
            stacked_overrides.update(overrides_applied)

    return LeverAttemptTable(
        schema_version="1",
        baseline_id=baseline.baseline_id,
        rows=tuple(rows),
        stacked_overrides=stacked_overrides,
    )


PHASE5_DEFERRED_LEVER_IDS: frozenset[str] = frozenset({"inductor-cache-warm-policy"})

PHASE5_DEFERRAL_REASONS: dict[str, str] = {
    "inductor-cache-warm-policy": (
        "Deferred to Phase 5 ALGO-01: inductor cache warm-start policy was not "
        "attempted in Phase 3 (RESEARCH priority 5 unless profiler re-ranks)."
    ),
}


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
