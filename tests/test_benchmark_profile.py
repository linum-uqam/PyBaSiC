"""Unit tests for profiler bottleneck taxonomy and report builder."""

from __future__ import annotations

import numpy as np

from linum_basic.benchmark.profile import (
    BOTTLENECK_CHUNKING,
    BOTTLENECK_COMPILE_SHAPE,
    BOTTLENECK_COMPUTE,
    BOTTLENECK_MEMORY_BANDWIDTH,
    BOTTLENECK_SYNCHRONIZATION,
    BottleneckReport,
    LeverAttemptTable,
    RankedLever,
    build_bottleneck_report,
    build_lever_attempt_table,
    build_phase3_handoff_config,
    build_phase5_backlog,
)
from linum_basic.core import BaSiC


def _write_profiler_fixture(
    events: list[dict[str, object]],
    *,
    summary: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return a dict mimicking torch.profiler key_averages export."""
    payload: dict[str, object] = {"key_averages": events}
    if summary is not None:
        payload["summary"] = summary
    return payload


class TestBuildBottleneckReport:
    def test_compute_primary_limit_from_matmul(self) -> None:
        events = _write_profiler_fixture(
            [
                {"name": "aten::mm", "self_cuda_time_total": 500.0},
                {"name": "aten::add", "self_cuda_time_total": 50.0},
            ]
        )
        report = build_bottleneck_report(events)
        assert isinstance(report, BottleneckReport)
        assert report.primary_limit == BOTTLENECK_COMPUTE
        assert report.working_size == 128
        lever_ids = [lever.lever_id for lever in report.ranked_levers]
        assert "sync-cadence" in lever_ids
        assert "compile-surfacing" in lever_ids
        assert report.ranked_levers == tuple(sorted(report.ranked_levers, key=lambda lever: lever.priority))

    def test_synchronization_primary_limit(self) -> None:
        events = _write_profiler_fixture(
            [
                {"name": "cudaDeviceSynchronize", "self_cuda_time_total": 400.0},
                {"name": "aten::norm", "self_cuda_time_total": 120.0},
                {"name": "aten::mm", "self_cuda_time_total": 30.0},
            ]
        )
        report = build_bottleneck_report(events)
        assert report.primary_limit == BOTTLENECK_SYNCHRONIZATION
        assert report.ranked_levers[0].lever_id == "sync-cadence"

    def test_memory_bandwidth_primary_limit(self) -> None:
        events = _write_profiler_fixture(
            [
                {"name": "aten::copy_", "self_cuda_time_total": 300.0},
                {"name": "aten::cat", "self_cuda_time_total": 250.0},
                {"name": "cudaMemcpyAsync", "self_cuda_time_total": 100.0},
            ]
        )
        report = build_bottleneck_report(events)
        assert report.primary_limit == BOTTLENECK_MEMORY_BANDWIDTH
        assert len(report.ranked_levers) >= 1

    def test_compile_shape_primary_limit(self) -> None:
        events = _write_profiler_fixture(
            [
                {"name": "inductor_compile", "self_cuda_time_total": 800.0},
                {"name": "torch::jit::compile", "self_cuda_time_total": 200.0},
                {"name": "aten::mm", "self_cuda_time_total": 50.0},
            ]
        )
        report = build_bottleneck_report(events)
        assert report.primary_limit == BOTTLENECK_COMPILE_SHAPE
        assert report.ranked_levers[0].lever_id == "compile-surfacing"

    def test_chunking_classified_when_present(self) -> None:
        events = _write_profiler_fixture(
            [
                {"name": "batched_z_chunk_cat", "self_cuda_time_total": 600.0, "category": "chunking"},
                {"name": "aten::mm", "self_cuda_time_total": 100.0},
            ]
        )
        report = build_bottleneck_report(events)
        classes = {hotspot.bottleneck_class for hotspot in report.hotspots}
        assert BOTTLENECK_CHUNKING in classes

    def test_hotspots_sorted_by_self_cuda_time(self) -> None:
        events = _write_profiler_fixture(
            [
                {"name": "aten::mm", "self_cuda_time_total": 100.0},
                {"name": "aten::bmm", "self_cuda_time_total": 300.0},
            ]
        )
        report = build_bottleneck_report(events)
        times = [hotspot.self_cuda_time_ms for hotspot in report.hotspots]
        assert times == sorted(times, reverse=True)

    def test_working_size_override(self) -> None:
        events = _write_profiler_fixture([{"name": "aten::mm", "self_cuda_time_total": 10.0}])
        report = build_bottleneck_report(events, working_size=64)
        assert report.working_size == 64


class TestBuildLeverAttemptTable:
    def test_two_candidate_rows_with_overall_from_metadata(self) -> None:
        from linum_basic.benchmark.artifacts import (
            SCHEMA_VERSION,
            BaselineBundle,
            CandidateArtifact,
        )
        from linum_basic.benchmark.quality import METRIC_DEFINITION_VERSION

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
            metrics_rows=[{"z": 0, "seam_l1": 0.1, "seam_curvature": 0.02}],
            metrics_aggregates={"seam_l1": 0.1, "seam_curvature": 0.02},
            repeats=1,
            metadata={"telemetry": {"steady_state_ms": 200.0}},
            timestamp="2026-01-01T12:00:00Z",
        )

        def _candidate(*, candidate_id: str, overall: str, overrides: dict[str, object]) -> CandidateArtifact:
            return CandidateArtifact(
                candidate_id=candidate_id,
                baseline_id=baseline_id,
                schema_version=SCHEMA_VERSION,
                metric_definition_version=METRIC_DEFINITION_VERSION,
                subject_id="sub22",
                run_label=candidate_id,
                input_fingerprint="sha256:deadbeef",
                z_indices=[0, 1],
                array_shape=[2, 64, 64],
                tile_shape=[64, 64],
                strategy_params={"working_size": 128, **overrides},
                metrics_rows=[{"z": 0, "seam_l1": 0.1, "seam_curvature": 0.02}],
                metrics_aggregates={"seam_l1": 0.1, "seam_curvature": 0.02},
                repeats=1,
                metadata={
                    "telemetry": {"steady_state_ms": 100.0},
                    "quality_verdict": {"passed": overall == "promote", "failures": []},
                    "overall": overall,
                    "overrides_applied": overrides,
                },
                environment={},
                timestamp="2026-01-02T12:00:00Z",
            )

        candidates = [
            _candidate(
                candidate_id="candidate-sync",
                overall="promote",
                overrides={"convergence_check_every": 20},
            ),
            _candidate(
                candidate_id="candidate-reweight",
                overall="reject",
                overrides={"reweighting_tolerance": 0.005},
            ),
        ]
        ranked_levers = (
            RankedLever("sync-cadence", "linum_basic/_alm.py", 1, "medium", "notes"),
            RankedLever("reweighting-tolerance", "linum_basic/core.py", 2, "medium", "notes"),
        )

        table = build_lever_attempt_table(baseline, candidates, ranked_levers=ranked_levers)
        assert isinstance(table, LeverAttemptTable)
        assert table.baseline_id == baseline_id
        assert len(table.rows) == 2
        assert table.rows[0].lever_id == "sync-cadence"
        assert table.rows[0].overall == "promote"
        assert table.rows[0].overrides_applied == {"convergence_check_every": 20}
        assert table.rows[1].lever_id == "reweighting-tolerance"
        assert table.rows[1].overall == "reject"
        assert table.stacked_overrides == {"convergence_check_every": 20}


class TestBuildPhase5Backlog:
    def test_all_reject_table_returns_non_empty_backlog_with_deferred_inductor(self) -> None:
        from linum_basic.benchmark.artifacts import (
            SCHEMA_VERSION,
            BaselineBundle,
            CandidateArtifact,
        )
        from linum_basic.benchmark.quality import METRIC_DEFINITION_VERSION

        baseline_id = "baseline-20260630T163351-e7c47a4-sub-22"
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
            metrics_rows=[{"z": 0, "seam_l1": 0.1, "seam_curvature": 0.02}],
            metrics_aggregates={"seam_l1": 0.1, "seam_curvature": 0.02},
            repeats=1,
            metadata={"telemetry": {"steady_state_ms": 200.0}},
            timestamp="2026-06-30T16:33:51Z",
        )

        def _candidate(*, candidate_id: str, overall: str, overrides: dict[str, object]) -> CandidateArtifact:
            return CandidateArtifact(
                candidate_id=candidate_id,
                baseline_id=baseline_id,
                schema_version=SCHEMA_VERSION,
                metric_definition_version=METRIC_DEFINITION_VERSION,
                subject_id="sub22",
                run_label=candidate_id,
                input_fingerprint="sha256:deadbeef",
                z_indices=[0, 1],
                array_shape=[2, 64, 64],
                tile_shape=[64, 64],
                strategy_params={"working_size": 128, **overrides},
                metrics_rows=[{"z": 0, "seam_l1": 0.1, "seam_curvature": 0.02}],
                metrics_aggregates={"seam_l1": 0.1, "seam_curvature": 0.02},
                repeats=1,
                metadata={
                    "telemetry": {"steady_state_ms": 150.0},
                    "quality_verdict": {"passed": False, "failures": ["seam_l1"]},
                    "overall": overall,
                    "overrides_applied": overrides,
                },
                environment={},
                timestamp="2026-06-30T17:00:00Z",
            )

        candidates = [
            _candidate(
                candidate_id="candidate-sync",
                overall="reject",
                overrides={"convergence_check_every": 20},
            ),
            _candidate(
                candidate_id="candidate-reweight",
                overall="reject",
                overrides={"reweighting_tolerance": 0.005},
            ),
        ]
        ranked_levers = (
            RankedLever("sync-cadence", "linum_basic/_alm.py", 1, "medium", "notes"),
            RankedLever("reweighting-tolerance", "linum_basic/core.py", 2, "medium", "notes"),
        )
        table = build_lever_attempt_table(baseline, candidates, ranked_levers=ranked_levers)
        backlog = build_phase5_backlog(ranked_levers, table)

        assert backlog["baseline_id"] == baseline_id
        assert len(backlog["entries"]) >= 2
        rejected_ids = {entry["lever_id"] for entry in backlog["entries"] if entry["status"] == "rejected"}
        assert "sync-cadence" in rejected_ids
        assert "reweighting-tolerance" in rejected_ids
        deferred = [entry for entry in backlog["entries"] if entry["lever_id"] == "inductor-cache-warm-policy"]
        assert len(deferred) == 1
        assert deferred[0]["status"] == "deferred"
        assert deferred[0]["deferral_reason"]
        assert "Phase 5" in deferred[0]["deferral_reason"]


class TestBuildPhase3HandoffConfig:
    def test_handoff_config_merges_stacked_overrides(self) -> None:
        config = build_phase3_handoff_config(
            {"convergence_check_every": 20, "reweighting_tolerance": 0.005},
            baseline_id="baseline-20260630T163351-e7c47a4-sub-22",
            timestamp="2026-06-30T18:00:00Z",
        )
        assert config["working_size"] == 128
        assert config["baseline_id"] == "baseline-20260630T163351-e7c47a4-sub-22"
        assert config["stacked_overrides"] == {
            "convergence_check_every": 20,
            "reweighting_tolerance": 0.005,
        }
        assert config["timestamp"] == "2026-06-30T18:00:00Z"


class TestTileSubsampleRatio:
    def test_default_none_preserves_tile_count_in_prepare(self) -> None:
        stack = np.ones((20, 32, 32), dtype=np.float32)
        model = BaSiC(stack)
        model.tile_subsample_ratio = None
        model.prepare()
        assert model.n_images == 20

    def test_ratio_subsamples_evenly_in_prepare(self) -> None:
        stack = np.ones((20, 32, 32), dtype=np.float32)
        model = BaSiC(stack)
        model.tile_subsample_ratio = 0.5
        model.prepare()
        assert model.n_images == 10
