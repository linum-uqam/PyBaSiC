"""Working-size sweep adoption layer combining quality and speed gates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from linum_basic.benchmark.artifacts import BaselineBundle, CandidateArtifact

SPEED_RATIO_THRESHOLD = 1.30

__all__ = [
    "SPEED_RATIO_THRESHOLD",
    "SweepAdoptionVerdict",
    "SweepRow",
    "SweepTable",
    "build_sweep_table",
    "evaluate_sweep_adoption",
]


@dataclass(frozen=True, slots=True, init=False)
class SweepRow:
    """One row in the working_size speed-quality sweep table.

    Attributes
    ----------
    working_size : int
        BaSiC resize resolution for this run.
    source : str
        ``"baseline"`` for the ws=128 reference row or ``"candidate"`` otherwise.
    artifact_id : str
        Baseline or candidate artifact identifier.
    steady_state_ms : float
        Median steady-state fit time in milliseconds.
    speed_ratio : float
        Baseline steady-state ms divided by candidate steady-state ms.
    quality_passed : bool
        Whether the Phase 1 quality gate passed.
    speed_passed : bool
        Whether ``speed_ratio`` meets the adoption threshold.
    candidate_overall : str
        Phase 1 overall verdict label (``promote``, ``reject``, or ``reference``).
    reweight_iterations_median : float | None
        Median reweighting iterations when convergence telemetry is present.
    """

    working_size: int
    source: str
    artifact_id: str
    steady_state_ms: float
    speed_ratio: float
    quality_passed: bool
    speed_passed: bool
    candidate_overall: str
    reweight_iterations_median: float | None = None

    def __init__(
        self,
        working_size: int,
        source: str,
        artifact_id: str,
        steady_state_ms: float,
        speed_ratio: float,
        quality_passed: bool,
        speed_passed: bool,
        candidate_overall: str,
        reweight_iterations_median: float | None = None,
    ) -> None:
        """Initialise one sweep table row.

        Parameters
        ----------
        working_size : int
            BaSiC resize resolution for this run.
        source : str
            ``"baseline"`` for the ws=128 reference row or ``"candidate"`` otherwise.
        artifact_id : str
            Baseline or candidate artifact identifier.
        steady_state_ms : float
            Median steady-state fit time in milliseconds.
        speed_ratio : float
            Baseline steady-state ms divided by candidate steady-state ms.
        quality_passed : bool
            Whether the Phase 1 quality gate passed.
        speed_passed : bool
            Whether ``speed_ratio`` meets the adoption threshold.
        candidate_overall : str
            Phase 1 overall verdict label (``promote``, ``reject``, or ``reference``).
        reweight_iterations_median : float or None, optional
            Median reweighting iterations when convergence telemetry is present.
        """
        object.__setattr__(self, "working_size", working_size)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "steady_state_ms", steady_state_ms)
        object.__setattr__(self, "speed_ratio", speed_ratio)
        object.__setattr__(self, "quality_passed", quality_passed)
        object.__setattr__(self, "speed_passed", speed_passed)
        object.__setattr__(self, "candidate_overall", candidate_overall)
        object.__setattr__(self, "reweight_iterations_median", reweight_iterations_median)


@dataclass(frozen=True, slots=True, init=False)
class SweepAdoptionVerdict:
    """Phase 2 adoption outcome derived from sweep rows.

    Attributes
    ----------
    recommended_ws : int | None
        Fastest ``working_size`` below 128 passing both gates, or ``None``.
    phase3_activate : bool
        Whether ws=128 optimization (Phase 3) should proceed.
    rationale : str
        Human-readable explanation of the adoption decision.
    """

    recommended_ws: int | None
    phase3_activate: bool
    rationale: str

    def __init__(
        self,
        recommended_ws: int | None,
        phase3_activate: bool,
        rationale: str,
    ) -> None:
        """Initialise a sweep adoption verdict.

        Parameters
        ----------
        recommended_ws : int or None
            Fastest ``working_size`` below 128 passing both gates, or ``None``.
        phase3_activate : bool
            Whether ws=128 optimization (Phase 3) should proceed.
        rationale : str
            Human-readable explanation of the adoption decision.
        """
        object.__setattr__(self, "recommended_ws", recommended_ws)
        object.__setattr__(self, "phase3_activate", phase3_activate)
        object.__setattr__(self, "rationale", rationale)


@dataclass(frozen=True, slots=True, init=False)
class SweepTable:
    """Aggregated working_size sweep table with adoption verdict.

    Attributes
    ----------
    schema_version : str
        Table schema version tag.
    baseline_id : str
        Referenced baseline bundle identifier.
    subject_id : str
        Subject identifier shared across rows.
    speed_ratio_threshold : float
        Speed ratio threshold used for ``speed_passed``.
    rows : tuple of SweepRow
        One row per working_size, sorted ascending.
    adoption : SweepAdoptionVerdict
        Phase 2 adoption outcome for the table.
    """

    schema_version: str
    baseline_id: str
    subject_id: str
    speed_ratio_threshold: float
    rows: tuple[SweepRow, ...]
    adoption: SweepAdoptionVerdict

    def __init__(
        self,
        schema_version: str,
        baseline_id: str,
        subject_id: str,
        speed_ratio_threshold: float,
        rows: tuple[SweepRow, ...],
        adoption: SweepAdoptionVerdict,
    ) -> None:
        """Initialise a sweep table.

        Parameters
        ----------
        schema_version : str
            Table schema version tag.
        baseline_id : str
            Referenced baseline bundle identifier.
        subject_id : str
            Subject identifier shared across rows.
        speed_ratio_threshold : float
            Speed ratio threshold used for ``speed_passed``.
        rows : tuple of SweepRow
            One row per working_size, sorted ascending.
        adoption : SweepAdoptionVerdict
            Phase 2 adoption outcome for the table.
        """
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "baseline_id", baseline_id)
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "speed_ratio_threshold", speed_ratio_threshold)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "adoption", adoption)


def _compute_speed_ratio(*, baseline_ms: float, candidate_ms: float) -> float:
    """Return baseline_ms / candidate_ms following benchmark_speedup semantics."""
    if candidate_ms <= 0:
        return float("inf") if baseline_ms > 0 else 1.0
    return baseline_ms / candidate_ms


def _require_telemetry_steady_state_ms(metadata: dict[str, Any], *, artifact_label: str) -> float:
    telemetry = metadata.get("telemetry")
    if not isinstance(telemetry, dict):
        msg = f"{artifact_label} metadata missing telemetry block"
        raise KeyError(msg)
    if "steady_state_ms" not in telemetry:
        msg = f"{artifact_label} telemetry missing steady_state_ms"
        raise KeyError(msg)
    return float(telemetry["steady_state_ms"])


def build_sweep_table(
    baseline: BaselineBundle,
    candidates: Sequence[CandidateArtifact],
    *,
    speed_ratio_threshold: float = SPEED_RATIO_THRESHOLD,
) -> SweepTable:
    """Build a speed-quality sweep table from saved benchmark artifacts.

    The ws=128 reference row is sourced from the baseline bundle without
    re-fitting. Candidate rows compare steady-state timing and Phase 1
    quality verdicts against that baseline.

    Parameters
    ----------
    baseline : BaselineBundle
        Saved production baseline at ``working_size=128``.
    candidates : sequence of CandidateArtifact
        Candidate runs at alternate working sizes.
    speed_ratio_threshold : float, optional
        Minimum speed ratio for ``speed_passed`` (default 1.30).

    Returns
    -------
    SweepTable
        Sorted sweep table with adoption verdict.
    """
    baseline_ms = _require_telemetry_steady_state_ms(baseline.metadata, artifact_label=baseline.baseline_id)
    baseline_ws = int(baseline.strategy_params["working_size"])
    baseline_row = SweepRow(
        working_size=baseline_ws,
        source="baseline",
        artifact_id=baseline.baseline_id,
        steady_state_ms=baseline_ms,
        speed_ratio=1.0,
        quality_passed=True,
        speed_passed=False,
        candidate_overall="reference",
        reweight_iterations_median=None,
    )

    candidate_rows: list[SweepRow] = []
    for candidate in candidates:
        candidate_ms = _require_telemetry_steady_state_ms(
            candidate.metadata,
            artifact_label=candidate.candidate_id,
        )
        quality_verdict = candidate.metadata.get("quality_verdict")
        if not isinstance(quality_verdict, dict) or "passed" not in quality_verdict:
            msg = f"{candidate.candidate_id} metadata missing quality_verdict.passed"
            raise KeyError(msg)
        overall = candidate.metadata.get("overall")
        if overall is None:
            msg = f"{candidate.candidate_id} metadata missing overall verdict"
            raise KeyError(msg)
        working_size = candidate.strategy_params.get("working_size")
        if working_size is None:
            msg = f"{candidate.candidate_id} strategy_params missing working_size"
            raise KeyError(msg)

        speed_ratio = _compute_speed_ratio(baseline_ms=baseline_ms, candidate_ms=candidate_ms)
        convergence = candidate.metadata.get("convergence")
        reweight_median: float | None = None
        if isinstance(convergence, dict):
            raw_median = convergence.get("reweight_iterations_median")
            if raw_median is not None:
                reweight_median = float(raw_median)

        candidate_rows.append(
            SweepRow(
                working_size=int(working_size),
                source="candidate",
                artifact_id=candidate.candidate_id,
                steady_state_ms=candidate_ms,
                speed_ratio=speed_ratio,
                quality_passed=bool(quality_verdict["passed"]),
                speed_passed=speed_ratio >= speed_ratio_threshold,
                candidate_overall=str(overall),
                reweight_iterations_median=reweight_median,
            )
        )

    rows = tuple(sorted((baseline_row, *candidate_rows), key=lambda row: row.working_size))
    adoption = evaluate_sweep_adoption(rows, speed_ratio_threshold=speed_ratio_threshold)
    return SweepTable(
        schema_version="1",
        baseline_id=baseline.baseline_id,
        subject_id=baseline.subject_id,
        speed_ratio_threshold=speed_ratio_threshold,
        rows=rows,
        adoption=adoption,
    )


def evaluate_sweep_adoption(
    rows: Sequence[SweepRow],
    *,
    speed_ratio_threshold: float = SPEED_RATIO_THRESHOLD,
) -> SweepAdoptionVerdict:
    """Pick the fastest eligible candidate or activate Phase 3.

    Eligible rows must have ``working_size < 128``, ``quality_passed``, and
    ``speed_passed``. Among eligible rows the row with the smallest
    ``steady_state_ms`` wins. When no row qualifies, Phase 3 activates.

    Parameters
    ----------
    rows : sequence of SweepRow
        Sweep table rows (typically sorted by ``working_size``).
    speed_ratio_threshold : float, optional
        Minimum speed ratio for ``speed_passed`` (default 1.30).

    Returns
    -------
    SweepAdoptionVerdict
        Adoption recommendation and Phase 3 activation flag.
    """
    del speed_ratio_threshold  # rows carry precomputed speed_passed

    eligible = [row for row in rows if row.working_size < 128 and row.quality_passed and row.speed_passed]
    if not eligible:
        return SweepAdoptionVerdict(
            recommended_ws=None,
            phase3_activate=True,
            rationale="no ws<128 passes both quality and speed gates; Phase 3 activates",
        )

    winner = min(eligible, key=lambda row: row.steady_state_ms)
    return SweepAdoptionVerdict(
        recommended_ws=winner.working_size,
        phase3_activate=False,
        rationale=f"adopt ws={winner.working_size} (fastest ws<128 passing both gates)",
    )
