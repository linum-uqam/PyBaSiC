"""Tests for per-z quality metrics and benchmark.quality helpers."""

from __future__ import annotations

import numpy as np
import pytest

from linum_basic.metrics import evaluate_correction_volume, evaluate_correction_volume_per_z


def _make_volume_mosaic_and_fit(
    n_rows: int = 2,
    n_cols: int = 3,
    tile_h: int = 16,
    tile_w: int = 16,
    n_z: int = 4,
    field_mode: str = "per-z",
    seed: int = 7,
    z_indices: list[int] | None = None,
) -> tuple:
    """Return a (MosaicGrid, MosaicFit) pair with synthetic data."""
    from linum_basic.fit import MosaicFit
    from linum_basic.mosaic import MosaicGrid

    rng = np.random.default_rng(seed)
    arr = rng.random((n_z, n_rows * tile_h, n_cols * tile_w), dtype=np.float32).astype(np.float32)
    mosaic = MosaicGrid(arr, tile_shape=(tile_h, tile_w), overlap_fraction=0.2)

    if field_mode == "per-z":
        flatfields = np.ones((n_z, tile_h, tile_w), dtype=np.float32)
        darkfields = np.zeros((n_z, tile_h, tile_w), dtype=np.float32)
    else:
        flatfields = np.ones((tile_h, tile_w), dtype=np.float32)
        darkfields = np.zeros((tile_h, tile_w), dtype=np.float32)

    fit = MosaicFit(
        flatfields=flatfields,
        darkfields=darkfields,
        field_mode=field_mode,
        z_indices=z_indices if z_indices is not None else list(range(n_z)),
    )
    return mosaic, fit


class TestEvaluateCorrectionVolumePerZ:
    def test_rows_match_z_indices(self):
        """One row per fitted z with z, seam_l1, seam_curvature keys."""
        mosaic, fit = _make_volume_mosaic_and_fit(n_z=4)
        result = evaluate_correction_volume_per_z(mosaic, fit)
        assert len(result["rows"]) == len(fit.z_indices)
        for row, z in zip(result["rows"], fit.z_indices, strict=True):
            assert row["z"] == z
            assert "seam_l1" in row
            assert "seam_curvature" in row
            assert np.isfinite(row["seam_l1"])
            assert np.isfinite(row["seam_curvature"])

    def test_aggregates_match_scalar_api(self):
        """Aggregate seam_l1 equals mean of per-z values; curvature matches scalar API."""
        mosaic, fit = _make_volume_mosaic_and_fit(n_z=3)
        per_z = evaluate_correction_volume_per_z(mosaic, fit)
        scalar = evaluate_correction_volume(mosaic, fit, metrics=("seam", "curvature"))

        per_z_l1 = [row["seam_l1"] for row in per_z["rows"]]
        assert per_z["aggregates"]["seam_l1"] == pytest.approx(np.nanmean(per_z_l1), abs=1e-6)
        assert per_z["aggregates"]["seam_l1"] == pytest.approx(scalar["seam_l1"], abs=1e-6)
        assert per_z["aggregates"]["seam_curvature"] == pytest.approx(scalar["seam_curvature"], abs=1e-6)

    def test_global_field_mode(self):
        """Global field mode returns one row per fitted z."""
        mosaic, fit = _make_volume_mosaic_and_fit(n_z=3, field_mode="global")
        result = evaluate_correction_volume_per_z(mosaic, fit)
        assert len(result["rows"]) == len(fit.z_indices)
        scalar = evaluate_correction_volume(mosaic, fit, metrics=("seam", "curvature"))
        assert result["aggregates"]["seam_curvature"] == pytest.approx(scalar["seam_curvature"], abs=1e-6)

    def test_empty_z_indices(self):
        """Empty z selection returns empty rows and NaN aggregates."""
        mosaic, fit = _make_volume_mosaic_and_fit(n_z=4, z_indices=[])
        result = evaluate_correction_volume_per_z(mosaic, fit)
        assert result["rows"] == []
        assert np.isnan(result["aggregates"]["seam_l1"])
        assert np.isnan(result["aggregates"]["seam_curvature"])


class TestComputeQualityReport:
    def test_returns_structured_report(self):
        """compute_quality_report wraps per-z metrics with version tag."""
        from linum_basic.benchmark.quality import (
            METRIC_DEFINITION_VERSION,
            PerZMetricRow,
            QualityReport,
            compute_quality_report,
        )

        mosaic, fit = _make_volume_mosaic_and_fit(n_z=3)
        report = compute_quality_report(mosaic, fit)
        assert isinstance(report, QualityReport)
        assert report.metric_definition_version == METRIC_DEFINITION_VERSION
        assert len(report.rows) == len(fit.z_indices)
        assert all(isinstance(row, PerZMetricRow) for row in report.rows)
        assert "seam_l1" in report.aggregates
        assert "seam_curvature" in report.aggregates


class TestComputeDeltas:
    def _make_report(self, n_z: int = 3, l1_offset: float = 0.0, curv_offset: float = 0.0):
        from linum_basic.benchmark.quality import (
            METRIC_DEFINITION_VERSION,
            PerZMetricRow,
            QualityReport,
        )

        rows = [PerZMetricRow(z=z, seam_l1=0.1 + l1_offset + z * 0.01, seam_curvature=0.05 + curv_offset) for z in range(n_z)]
        aggregates = {
            "seam_l1": float(np.mean([r.seam_l1 for r in rows])),
            "seam_curvature": float(np.mean([r.seam_curvature for r in rows])),
        }
        return QualityReport(rows=tuple(rows), aggregates=aggregates, metric_definition_version=METRIC_DEFINITION_VERSION)

    def test_aggregate_and_per_z_deltas(self):
        """compute_deltas returns abs and rel deltas aligned by z."""
        from linum_basic.benchmark.quality import MetricDelta, compute_deltas

        baseline = self._make_report()
        candidate = self._make_report(l1_offset=0.02, curv_offset=0.01)
        deltas = compute_deltas(candidate, baseline)

        assert "seam_l1" in deltas["aggregate"]
        assert isinstance(deltas["aggregate"]["seam_l1"], MetricDelta)
        assert deltas["aggregate"]["seam_l1"].abs_delta != 0.0
        assert deltas["aggregate"]["seam_l1"].rel_delta != 0.0

        assert len(deltas["per_z"]) == len(baseline.rows)
        for entry, base_row, cand_row in zip(deltas["per_z"], baseline.rows, candidate.rows, strict=True):
            assert entry["z"] == base_row.z == cand_row.z
            assert entry["seam_l1"].abs_delta == pytest.approx(cand_row.seam_l1 - base_row.seam_l1)
            assert entry["seam_curvature"].abs_delta == pytest.approx(cand_row.seam_curvature - base_row.seam_curvature)

    def test_raises_on_mismatched_z_indices(self):
        """Mismatched z index sets raise ValueError."""
        from linum_basic.benchmark.quality import compute_deltas

        baseline = self._make_report(n_z=3)
        candidate = self._make_report(n_z=2)
        with pytest.raises(ValueError, match="z"):
            compute_deltas(candidate, baseline)


class TestCalibrateTolerances:
    def _make_report(self, seam_l1: float, seam_curvature: float = 0.05):
        from linum_basic.benchmark.quality import (
            METRIC_DEFINITION_VERSION,
            PerZMetricRow,
            QualityReport,
        )

        rows = (PerZMetricRow(z=0, seam_l1=seam_l1, seam_curvature=seam_curvature),)
        aggregates = {"seam_l1": seam_l1, "seam_curvature": seam_curvature}
        return QualityReport(
            rows=rows,
            aggregates=aggregates,
            metric_definition_version=METRIC_DEFINITION_VERSION,
        )

    def test_three_repeats_tolerance_formula(self):
        """Three repeats yield abs_tol=max(min_abs, 3*std) and matching rel_tol."""
        from linum_basic.benchmark.quality import CALIBRATION_POLICY, calibrate_tolerances

        min_abs = 1e-6
        repeats = [self._make_report(v) for v in (0.1, 0.12, 0.14)]
        specs = calibrate_tolerances(repeats, sigma=3.0, min_abs=min_abs)

        values = np.array([0.1, 0.12, 0.14], dtype=np.float64)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values, ddof=1))
        expected_abs = max(min_abs, 3.0 * std)
        expected_rel = expected_abs / (abs(mean) + min_abs)

        spec = specs["seam_l1"]
        assert spec.mean == pytest.approx(mean, abs=1e-9)
        assert spec.std == pytest.approx(std, abs=1e-9)
        assert spec.abs_tol == pytest.approx(expected_abs, abs=1e-9)
        assert spec.rel_tol == pytest.approx(expected_rel, abs=1e-9)
        assert spec.sigma == 3.0
        assert spec.min_abs == min_abs
        assert CALIBRATION_POLICY == "mean+3std"

    def test_single_repeat_nonzero_floor(self):
        """Single repeat yields std=0 and abs_tol=min_abs."""
        from linum_basic.benchmark.quality import calibrate_tolerances

        min_abs = 1e-6
        specs = calibrate_tolerances([self._make_report(0.2)], min_abs=min_abs)

        spec = specs["seam_l1"]
        assert spec.std == 0.0
        assert spec.abs_tol == pytest.approx(min_abs, abs=1e-12)
        assert spec.rel_tol == pytest.approx(min_abs / (0.2 + min_abs), abs=1e-9)

    def test_calibrates_both_first_class_metrics(self):
        """Both seam_l1 and seam_curvature receive ToleranceSpec entries."""
        from linum_basic.benchmark.quality import calibrate_tolerances

        repeats = [self._make_report(0.1, seam_curvature=0.05) for _ in range(2)]
        specs = calibrate_tolerances(repeats)
        assert "seam_l1" in specs
        assert "seam_curvature" in specs


class TestEvaluateQualityGate:
    def _make_tolerances(self, seam_l1_abs: float = 0.01, seam_l1_rel: float = 0.1) -> dict:
        from linum_basic.benchmark.quality import ToleranceSpec

        return {
            "seam_l1": ToleranceSpec(
                metric="seam_l1",
                mean=0.1,
                std=0.01,
                abs_tol=seam_l1_abs,
                rel_tol=seam_l1_rel,
                sigma=3.0,
                min_abs=1e-6,
            ),
            "seam_curvature": ToleranceSpec(
                metric="seam_curvature",
                mean=0.05,
                std=0.005,
                abs_tol=0.005,
                rel_tol=0.1,
                sigma=3.0,
                min_abs=1e-6,
            ),
        }

    def _make_deltas(
        self,
        *,
        agg_l1_abs: float = 0.0,
        agg_l1_rel: float = 0.0,
        per_z_l1: dict[int, float] | None = None,
        pearson_abs: float = 0.0,
    ) -> dict:
        from linum_basic.benchmark.quality import MetricDelta

        per_z_l1 = per_z_l1 or {}
        per_z = []
        for z, l1_abs in per_z_l1.items():
            per_z.append(
                {
                    "z": z,
                    "seam_l1": MetricDelta(abs_delta=l1_abs, rel_delta=l1_abs / 0.1),
                    "seam_curvature": MetricDelta(abs_delta=0.0, rel_delta=0.0),
                }
            )
        if not per_z:
            per_z = [
                {
                    "z": 0,
                    "seam_l1": MetricDelta(abs_delta=0.0, rel_delta=0.0),
                    "seam_curvature": MetricDelta(abs_delta=0.0, rel_delta=0.0),
                }
            ]

        aggregate = {
            "seam_l1": MetricDelta(abs_delta=agg_l1_abs, rel_delta=agg_l1_rel),
            "seam_curvature": MetricDelta(abs_delta=0.0, rel_delta=0.0),
            "pearson": MetricDelta(abs_delta=pearson_abs, rel_delta=pearson_abs),
        }
        return {"aggregate": aggregate, "per_z": per_z}

    def test_fails_on_aggregate_seam_l1_regression(self):
        """Aggregate seam_l1 beyond abs_tol fails with seam_l1 listed."""
        from linum_basic.benchmark.quality import evaluate_quality_gate

        tolerances = self._make_tolerances(seam_l1_abs=0.01)
        deltas = self._make_deltas(agg_l1_abs=0.02, agg_l1_rel=0.05)
        verdict = evaluate_quality_gate(deltas, tolerances)

        assert verdict.passed is False
        assert "seam_l1" in verdict.failures

    def test_fails_on_severe_per_z_outlier(self):
        """Aggregate within tolerance but severe per-z regression fails with worst_z."""
        from linum_basic.benchmark.quality import evaluate_quality_gate

        tolerances = self._make_tolerances(seam_l1_abs=0.01, seam_l1_rel=0.5)
        deltas = self._make_deltas(agg_l1_abs=0.005, per_z_l1={0: 0.0, 1: 0.05, 2: 0.0})
        verdict = evaluate_quality_gate(deltas, tolerances)

        assert verdict.passed is False
        assert verdict.worst_z == 1

    def test_passes_within_tolerance(self):
        """Candidate within aggregate and per-z tolerance passes."""
        from linum_basic.benchmark.quality import evaluate_quality_gate

        tolerances = self._make_tolerances(seam_l1_abs=0.01)
        deltas = self._make_deltas(agg_l1_abs=0.005, per_z_l1={0: 0.002, 1: 0.003})
        verdict = evaluate_quality_gate(deltas, tolerances)

        assert verdict.passed is True
        assert verdict.failures == ()
        assert verdict.worst_z is None

    def test_pearson_regression_is_diagnostic_only(self):
        """Pearson aggregate regression does not flip the verdict."""
        from linum_basic.benchmark.quality import evaluate_quality_gate

        tolerances = self._make_tolerances()
        deltas = self._make_deltas(pearson_abs=1.0)
        verdict = evaluate_quality_gate(deltas, tolerances)

        assert verdict.passed is True
