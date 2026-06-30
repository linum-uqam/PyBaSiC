"""Unit tests for working_size sweep adoption and table builder."""

from __future__ import annotations

from pathlib import Path

from linum_basic.benchmark.artifacts import (
    SCHEMA_VERSION,
    BaselineBundle,
    CandidateArtifact,
    read_artifact,
    write_artifact,
)
from linum_basic.benchmark.quality import METRIC_DEFINITION_VERSION
from linum_basic.benchmark.sweep import (
    SPEED_RATIO_THRESHOLD,
    SweepRow,
    SweepTable,
    build_sweep_table,
    evaluate_sweep_adoption,
)


def _row(
    *,
    working_size: int,
    steady_state_ms: float,
    quality_passed: bool,
    speed_passed: bool,
    source: str = "candidate",
    artifact_id: str = "candidate-test",
    speed_ratio: float = 1.5,
    candidate_overall: str = "promote",
) -> SweepRow:
    return SweepRow(
        working_size=working_size,
        source=source,
        artifact_id=artifact_id,
        steady_state_ms=steady_state_ms,
        speed_ratio=speed_ratio,
        quality_passed=quality_passed,
        speed_passed=speed_passed,
        candidate_overall=candidate_overall,
    )


class TestEvaluateSweepAdoption:
    def test_fastest_passing_candidate_wins(self) -> None:
        rows = (
            _row(working_size=64, steady_state_ms=100.0, quality_passed=True, speed_passed=True),
            _row(working_size=96, steady_state_ms=120.0, quality_passed=True, speed_passed=True),
        )
        verdict = evaluate_sweep_adoption(rows)
        assert verdict.recommended_ws == 64
        assert verdict.phase3_activate is False
        assert "64" in verdict.rationale

    def test_speed_fail_activates_phase3(self) -> None:
        rows = (_row(working_size=96, steady_state_ms=120.0, quality_passed=True, speed_passed=False),)
        verdict = evaluate_sweep_adoption(rows)
        assert verdict.recommended_ws is None
        assert verdict.phase3_activate is True
        assert "no ws<128 passes both" in verdict.rationale

    def test_no_passing_candidate_activates_phase3(self) -> None:
        rows = (
            _row(working_size=64, steady_state_ms=100.0, quality_passed=False, speed_passed=True),
            _row(working_size=96, steady_state_ms=120.0, quality_passed=True, speed_passed=False),
        )
        verdict = evaluate_sweep_adoption(rows)
        assert verdict.recommended_ws is None
        assert verdict.phase3_activate is True
        assert "no ws<128 passes both" in verdict.rationale

    def test_baseline_reference_row_never_recommended(self) -> None:
        rows = (
            _row(
                working_size=128,
                steady_state_ms=200.0,
                quality_passed=True,
                speed_passed=True,
                source="baseline",
                artifact_id="baseline-test",
                speed_ratio=1.0,
                candidate_overall="reference",
            ),
        )
        verdict = evaluate_sweep_adoption(rows)
        assert verdict.recommended_ws is None
        assert verdict.phase3_activate is True

    def test_speed_ratio_threshold_constant(self) -> None:
        assert SPEED_RATIO_THRESHOLD == 1.30


def _write_sweep_fixtures(
    artifacts_dir: Path,
    *,
    candidate_ws64_ms: float = 100.0,
    candidate_ws96_ms: float = 120.0,
    ws96_convergence_median: float | None = None,
) -> dict[str, object]:
    baseline_id = "baseline-20260101-abc1234-sub22"
    baseline = BaselineBundle(
        baseline_id=baseline_id,
        uuid="550e8400-e29b-41d4-a716-446655440000",
        schema_version=SCHEMA_VERSION,
        metric_definition_version=METRIC_DEFINITION_VERSION,
        subject_id="sub22",
        run_label="production",
        input_fingerprint="sha256:deadbeef",
        z_indices=[0, 1],
        array_shape=[2, 64, 64],
        tile_shape=[64, 64],
        strategy_params={"working_size": 128},
        metrics_rows=[
            {"z": 0, "seam_l1": 0.1, "seam_curvature": 0.02},
            {"z": 1, "seam_l1": 0.05, "seam_curvature": 0.02},
        ],
        metrics_aggregates={"seam_l1": 0.1, "seam_curvature": 0.02},
        repeats=1,
        metadata={"telemetry": {"steady_state_ms": 200.0}},
        timestamp="2026-01-01T12:00:00Z",
    )
    bundle_dir = artifacts_dir / baseline_id
    bundle_dir.mkdir(parents=True)
    write_artifact(bundle_dir / "baseline-bundle.json", baseline)

    def _candidate(
        *,
        candidate_id: str,
        working_size: int,
        steady_state_ms: float,
        convergence_median: float | None = None,
    ) -> CandidateArtifact:
        metadata: dict[str, object] = {
            "telemetry": {"steady_state_ms": steady_state_ms},
            "quality_verdict": {"passed": True, "failures": []},
            "overall": "promote",
        }
        if convergence_median is not None:
            metadata["convergence"] = {"reweight_iterations_median": convergence_median}
        return CandidateArtifact(
            candidate_id=candidate_id,
            baseline_id=baseline_id,
            schema_version=SCHEMA_VERSION,
            metric_definition_version=METRIC_DEFINITION_VERSION,
            subject_id="sub22",
            run_label=f"ws{working_size}",
            input_fingerprint="sha256:deadbeef",
            z_indices=[0, 1],
            array_shape=[2, 64, 64],
            tile_shape=[64, 64],
            strategy_params={"working_size": working_size},
            metrics_rows=[
                {"z": 0, "seam_l1": 0.1, "seam_curvature": 0.02},
                {"z": 1, "seam_l1": 0.05, "seam_curvature": 0.02},
            ],
            metrics_aggregates={"seam_l1": 0.1, "seam_curvature": 0.02},
            repeats=1,
            metadata=metadata,
            environment={},
            timestamp="2026-01-02T12:00:00Z",
        )

    candidate_64 = _candidate(
        candidate_id="candidate-ws64",
        working_size=64,
        steady_state_ms=candidate_ws64_ms,
    )
    candidate_96 = _candidate(
        candidate_id="candidate-ws96",
        working_size=96,
        steady_state_ms=candidate_ws96_ms,
        convergence_median=ws96_convergence_median,
    )
    for candidate in (candidate_64, candidate_96):
        candidate_dir = artifacts_dir / candidate.candidate_id
        candidate_dir.mkdir()
        write_artifact(candidate_dir / "candidate-artifact.json", candidate)

    reloaded_baseline = read_artifact(bundle_dir / "baseline-bundle.json", BaselineBundle)
    reloaded_candidates = [
        read_artifact(
            artifacts_dir / candidate.candidate_id / "candidate-artifact.json",
            CandidateArtifact,
        )
        for candidate in (candidate_64, candidate_96)
    ]
    return {
        "baseline": reloaded_baseline,
        "candidates": reloaded_candidates,
    }


class TestBuildSweepTable:
    def test_build_sweep_table_three_rows(self, tmp_path: Path) -> None:
        fixtures = _write_sweep_fixtures(tmp_path / "artifacts")
        table = build_sweep_table(fixtures["baseline"], fixtures["candidates"])
        assert isinstance(table, SweepTable)
        assert {row.working_size for row in table.rows} == {64, 96, 128}
        assert len(table.rows) == 3
        assert table.rows == tuple(sorted(table.rows, key=lambda row: row.working_size))

    def test_baseline_reference_row(self, tmp_path: Path) -> None:
        fixtures = _write_sweep_fixtures(tmp_path / "artifacts")
        baseline = fixtures["baseline"]
        table = build_sweep_table(baseline, fixtures["candidates"])
        baseline_row = next(row for row in table.rows if row.working_size == 128)
        assert baseline_row.source == "baseline"
        assert baseline_row.artifact_id == baseline.baseline_id
        assert baseline_row.candidate_overall == "reference"
        assert baseline_row.speed_ratio == 1.0
        assert baseline_row.reweight_iterations_median is None

    def test_candidate_reweight_median(self, tmp_path: Path) -> None:
        fixtures = _write_sweep_fixtures(
            tmp_path / "artifacts",
            ws96_convergence_median=7.0,
        )
        table = build_sweep_table(fixtures["baseline"], fixtures["candidates"])
        row_96 = next(row for row in table.rows if row.working_size == 96)
        row_64 = next(row for row in table.rows if row.working_size == 64)
        assert row_96.reweight_iterations_median == 7.0
        assert row_64.reweight_iterations_median is None
