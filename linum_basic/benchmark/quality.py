"""Per-z quality reports and candidate-vs-baseline deltas for the A/B harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from linum_basic.metrics import evaluate_correction_volume_per_z

if TYPE_CHECKING:
    from linum_basic.fit import MosaicFit
    from linum_basic.mosaic import MosaicGrid

METRIC_DEFINITION_VERSION = "1"
CALIBRATION_POLICY = "mean+3std"
FIRST_CLASS_METRICS: tuple[str, ...] = ("seam_l1", "seam_curvature")

__all__ = [
    "CALIBRATION_POLICY",
    "FIRST_CLASS_METRICS",
    "METRIC_DEFINITION_VERSION",
    "MetricDelta",
    "PerZMetricRow",
    "QualityReport",
    "QualityVerdict",
    "ToleranceSpec",
    "calibrate_tolerances",
    "compute_deltas",
    "compute_quality_report",
    "evaluate_quality_gate",
]


@dataclass(frozen=True, slots=True, init=False)
class PerZMetricRow:
    """Per-z seam quality metrics for one fitted depth plane.

    Attributes
    ----------
    z : int
        Z-index of the fitted depth plane.
    seam_l1 : float
        Mean per-seam relative L1 disagreement after correction.
    seam_curvature : float
        Flatfield focal-curvature residual at this z-level.
    """

    z: int
    seam_l1: float
    seam_curvature: float

    def __init__(self, z: int, seam_l1: float, seam_curvature: float) -> None:
        """Initialise a per-z metric row.

        Parameters
        ----------
        z : int
            Z-index of the fitted depth plane.
        seam_l1 : float
            Mean per-seam relative L1 disagreement after correction.
        seam_curvature : float
            Flatfield focal-curvature residual at this z-level.
        """
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "seam_l1", seam_l1)
        object.__setattr__(self, "seam_curvature", seam_curvature)


@dataclass(frozen=True, slots=True, init=False)
class MetricDelta:
    """Absolute and relative delta for one scalar metric.

    Attributes
    ----------
    abs_delta : float
        Candidate minus baseline (signed).
    rel_delta : float
        Relative delta with a small denominator floor.
    """

    abs_delta: float
    rel_delta: float

    def __init__(self, abs_delta: float, rel_delta: float) -> None:
        """Initialise a metric delta pair.

        Parameters
        ----------
        abs_delta : float
            Candidate minus baseline (signed).
        rel_delta : float
            Relative delta with a small denominator floor.
        """
        object.__setattr__(self, "abs_delta", abs_delta)
        object.__setattr__(self, "rel_delta", rel_delta)


@dataclass(frozen=True, slots=True, init=False)
class QualityReport:
    """Structured per-z quality report with aggregate scalars.

    Attributes
    ----------
    rows : tuple of PerZMetricRow
        One row per fitted z-level.
    aggregates : dict[str, float]
        Volume-level ``seam_l1`` and ``seam_curvature`` scalars.
    metric_definition_version : str
        Version tag for baseline compatibility checks.
    """

    rows: tuple[PerZMetricRow, ...]
    aggregates: dict[str, float]
    metric_definition_version: str

    def __init__(
        self,
        rows: tuple[PerZMetricRow, ...],
        aggregates: dict[str, float],
        metric_definition_version: str,
    ) -> None:
        """Initialise a quality report.

        Parameters
        ----------
        rows : tuple of PerZMetricRow
            One row per fitted z-level.
        aggregates : dict[str, float]
            Volume-level ``seam_l1`` and ``seam_curvature`` scalars.
        metric_definition_version : str
            Version tag for baseline compatibility checks.
        """
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "aggregates", aggregates)
        object.__setattr__(self, "metric_definition_version", metric_definition_version)


def compute_quality_report(
    mosaic: MosaicGrid,
    fit: MosaicFit,
    *,
    epsilon: float = 1e-6,
) -> QualityReport:
    """Build a :class:`QualityReport` from a fitted mosaic.

    Parameters
    ----------
    mosaic : MosaicGrid
        Source mosaic volume.
    fit : MosaicFit
        Fitted flat/dark-fields.
    epsilon : float
        Divisor stabilisation constant forwarded to the metrics helper.

    Returns
    -------
    QualityReport
        Per-z rows plus aggregate ``seam_l1`` and ``seam_curvature``.
    """
    raw = evaluate_correction_volume_per_z(mosaic, fit, epsilon=epsilon)
    rows = tuple(
        PerZMetricRow(z=int(row["z"]), seam_l1=float(row["seam_l1"]), seam_curvature=float(row["seam_curvature"]))
        for row in raw["rows"]
    )
    return QualityReport(
        rows=rows,
        aggregates=dict(raw["aggregates"]),
        metric_definition_version=METRIC_DEFINITION_VERSION,
    )


def _metric_delta(candidate: float, baseline: float, *, rel_floor: float) -> MetricDelta:
    abs_delta = candidate - baseline
    denom = max(abs(baseline), rel_floor)
    rel_delta = abs_delta / denom
    return MetricDelta(abs_delta=abs_delta, rel_delta=rel_delta)


def compute_deltas(
    candidate: QualityReport,
    baseline: QualityReport,
    *,
    rel_floor: float = 1e-6,
) -> dict[str, Any]:
    """Compute candidate-vs-baseline absolute and relative metric deltas.

    Parameters
    ----------
    candidate : QualityReport
        Quality report for the candidate fit.
    baseline : QualityReport
        Quality report for the baseline fit.
    rel_floor : float
        Minimum denominator for relative deltas to avoid divide-by-zero.

    Returns
    -------
    dict
        ``{"aggregate": dict[str, MetricDelta], "per_z": list[dict]}``.

    Raises
    ------
    ValueError
        When candidate and baseline z index sets differ.
    """
    cand_z = {row.z for row in candidate.rows}
    base_z = {row.z for row in baseline.rows}
    if cand_z != base_z:
        msg = f"Candidate and baseline z index sets differ: candidate={sorted(cand_z)}, baseline={sorted(base_z)}"
        raise ValueError(msg)

    aggregate: dict[str, MetricDelta] = {}
    for key in ("seam_l1", "seam_curvature"):
        aggregate[key] = _metric_delta(
            candidate.aggregates[key],
            baseline.aggregates[key],
            rel_floor=rel_floor,
        )

    base_by_z = {row.z: row for row in baseline.rows}
    per_z: list[dict[str, Any]] = []
    for cand_row in candidate.rows:
        base_row = base_by_z[cand_row.z]
        per_z.append(
            {
                "z": cand_row.z,
                "seam_l1": _metric_delta(cand_row.seam_l1, base_row.seam_l1, rel_floor=rel_floor),
                "seam_curvature": _metric_delta(
                    cand_row.seam_curvature,
                    base_row.seam_curvature,
                    rel_floor=rel_floor,
                ),
            }
        )

    return {"aggregate": aggregate, "per_z": per_z}


@dataclass(frozen=True, slots=True, init=False)
class ToleranceSpec:
    """Calibrated tolerance for one first-class quality metric.

    Attributes
    ----------
    metric : str
        Metric name (e.g. ``seam_l1``).
    mean : float
        Mean across baseline repeat aggregates.
    std : float
        Sample standard deviation (ddof=1) across repeats.
    abs_tol : float
        Absolute tolerance floor: ``max(min_abs, sigma * std)``.
    rel_tol : float
        Relative tolerance: ``abs_tol / (abs(mean) + min_abs)``.
    sigma : float
        Standard-deviation multiplier used during calibration.
    min_abs : float
        Minimum absolute tolerance floor.
    """

    metric: str
    mean: float
    std: float
    abs_tol: float
    rel_tol: float
    sigma: float
    min_abs: float

    def __init__(
        self,
        metric: str,
        mean: float,
        std: float,
        abs_tol: float,
        rel_tol: float,
        sigma: float,
        min_abs: float,
    ) -> None:
        """Initialise a tolerance specification."""
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "abs_tol", abs_tol)
        object.__setattr__(self, "rel_tol", rel_tol)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "min_abs", min_abs)


def _calibrate_one_metric(
    values: list[float],
    metric: str,
    *,
    sigma: float,
    min_abs: float,
) -> ToleranceSpec:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
    abs_tol = max(min_abs, sigma * std)
    rel_tol = abs_tol / (abs(mean) + min_abs)
    return ToleranceSpec(
        metric=metric,
        mean=mean,
        std=std,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        sigma=sigma,
        min_abs=min_abs,
    )


def calibrate_tolerances(
    repeat_reports: list[QualityReport],
    *,
    sigma: float = 3.0,
    min_abs: float = 1e-6,
) -> dict[str, ToleranceSpec]:
    """Calibrate per-metric tolerances from baseline repeat quality reports.

    Uses the :data:`CALIBRATION_POLICY` (``mean + sigma * std``) with an
    absolute floor so zero-variance baselines still yield a passable gate.

    Parameters
    ----------
    repeat_reports : list of QualityReport
        One or more baseline repeat reports.
    sigma : float
        Standard-deviation multiplier (default 3.0).
    min_abs : float
        Minimum absolute tolerance floor.

    Returns
    -------
    dict[str, ToleranceSpec]
        Mapping from first-class metric name to calibrated tolerance.
    """
    if not repeat_reports:
        msg = "repeat_reports must contain at least one QualityReport"
        raise ValueError(msg)

    specs: dict[str, ToleranceSpec] = {}
    for metric in FIRST_CLASS_METRICS:
        values = [report.aggregates[metric] for report in repeat_reports]
        specs[metric] = _calibrate_one_metric(values, metric, sigma=sigma, min_abs=min_abs)
    return specs


@dataclass(frozen=True, slots=True, init=False)
class QualityVerdict:
    """QUAL-03 rejection verdict for a candidate vs baseline.

    Attributes
    ----------
    passed : bool
        ``True`` when no aggregate or severe per-z breach occurred.
    failures : tuple of str
        First-class metrics that failed the aggregate gate.
    worst_z : int or None
        Z-index of the worst per-z outlier, if any.
    aggregate_deltas : dict[str, MetricDelta]
        Aggregate deltas for first-class metrics.
    per_z_failures : tuple of dict
        Per-z outlier records with ``z``, ``metric``, and delta fields.
    """

    passed: bool
    failures: tuple[str, ...]
    worst_z: int | None
    aggregate_deltas: dict[str, MetricDelta]
    per_z_failures: tuple[dict[str, Any], ...]

    def __init__(
        self,
        passed: bool,
        failures: tuple[str, ...],
        worst_z: int | None,
        aggregate_deltas: dict[str, MetricDelta],
        per_z_failures: tuple[dict[str, Any], ...],
    ) -> None:
        """Initialise a quality gate verdict."""
        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "failures", failures)
        object.__setattr__(self, "worst_z", worst_z)
        object.__setattr__(self, "aggregate_deltas", aggregate_deltas)
        object.__setattr__(self, "per_z_failures", per_z_failures)


def _per_z_bounds(spec: ToleranceSpec, *, per_z_outlier_sigma: float) -> tuple[float, float]:
    abs_bound = max(spec.min_abs, per_z_outlier_sigma * spec.std)
    rel_bound = abs_bound / (abs(spec.mean) + spec.min_abs)
    return abs_bound, rel_bound


def _delta_exceeds(delta: MetricDelta, abs_tol: float, rel_tol: float) -> bool:
    return delta.abs_delta > abs_tol or delta.rel_delta > rel_tol


def evaluate_quality_gate(
    deltas: dict[str, Any],
    tolerances: dict[str, ToleranceSpec],
    *,
    per_z_outlier_sigma: float = 3.0,
) -> QualityVerdict:
    """Apply calibrated tolerances to candidate deltas (QUAL-03 rejection rule).

    Fails when any first-class metric aggregate delta exceeds absolute **or**
    relative tolerance, or when any per-z delta is a severe outlier. Pearson
    and other diagnostic metrics never flip the verdict.

    Parameters
    ----------
    deltas : dict
        Output of :func:`compute_deltas` with ``aggregate`` and ``per_z`` keys.
    tolerances : dict[str, ToleranceSpec]
        Calibrated tolerances per first-class metric.
    per_z_outlier_sigma : float
        Sigma multiplier for per-z severe-outlier bounds.

    Returns
    -------
    QualityVerdict
        Pass/fail verdict with failure reasons and worst-z outlier index.
    """
    aggregate: dict[str, MetricDelta] = deltas["aggregate"]
    per_z_rows: list[dict[str, Any]] = deltas["per_z"]

    failures: list[str] = []
    for metric in FIRST_CLASS_METRICS:
        if metric not in tolerances:
            continue
        spec = tolerances[metric]
        delta = aggregate[metric]
        if _delta_exceeds(delta, spec.abs_tol, spec.rel_tol):
            failures.append(metric)

    per_z_failures: list[dict[str, Any]] = []
    worst_z: int | None = None
    worst_score = -1.0

    for row in per_z_rows:
        z = int(row["z"])
        for metric in FIRST_CLASS_METRICS:
            if metric not in tolerances:
                continue
            spec = tolerances[metric]
            delta: MetricDelta = row[metric]
            abs_bound, rel_bound = _per_z_bounds(spec, per_z_outlier_sigma=per_z_outlier_sigma)
            if _delta_exceeds(delta, abs_bound, rel_bound):
                score = max(delta.abs_delta / abs_bound, delta.rel_delta / rel_bound)
                per_z_failures.append(
                    {
                        "z": z,
                        "metric": metric,
                        "abs_delta": delta.abs_delta,
                        "rel_delta": delta.rel_delta,
                    }
                )
                if score > worst_score:
                    worst_score = score
                    worst_z = z

    aggregate_first_class = {metric: aggregate[metric] for metric in FIRST_CLASS_METRICS if metric in aggregate}
    passed = not failures and not per_z_failures

    return QualityVerdict(
        passed=passed,
        failures=tuple(failures),
        worst_z=worst_z,
        aggregate_deltas=aggregate_first_class,
        per_z_failures=tuple(per_z_failures),
    )
