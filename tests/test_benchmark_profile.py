"""Unit tests for profiler bottleneck taxonomy and report builder."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from linum_basic.benchmark.profile import (
    BOTTLENECK_CHUNKING,
    BOTTLENECK_COMPILE_SHAPE,
    BOTTLENECK_COMPUTE,
    BOTTLENECK_MEMORY_BANDWIDTH,
    BOTTLENECK_SYNCHRONIZATION,
    BottleneckReport,
    HistoricalBaseline,
    LeverAttemptTable,
    RankedLever,
    build_bottleneck_report,
    build_forensics_change_attribution,
    build_forensics_recovery_levers,
    build_forensics_report,
    build_lever_attempt_table,
    build_phase3_handoff_config,
    build_phase5_backlog,
    build_phase5_fast_path,
    build_phase6_concurrency_verdict,
    build_phase7_integration_summary,
    build_regression_triage_result,
    diagnose_regression_triage,
    is_fast_era,
    load_historical_baselines_from_iteration_ab,
    warn_git_commit_drift,
    write_forensics_report_bundle,
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


class TestStackSpeedRatio:
    def test_stack_speed_ratio_against_phase5_baseline(self) -> None:
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

        def _candidate(
            *, candidate_id: str, overall: str, overrides: dict[str, object], steady_state_ms: float
        ) -> CandidateArtifact:
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
                    "telemetry": {"steady_state_ms": steady_state_ms},
                    "quality_verdict": {"passed": overall == "promote", "failures": []},
                    "overall": overall,
                    "overrides_applied": overrides,
                },
                environment={},
                timestamp="2026-06-30T17:00:00Z",
            )

        candidates = [
            _candidate(
                candidate_id="candidate-sync",
                overall="promote",
                overrides={"convergence_check_every": 20},
                steady_state_ms=100.0,
            ),
        ]
        ranked_levers = (RankedLever("sync-cadence", "linum_basic/_alm.py", 1, "medium", "notes"),)
        stacked_overrides = {"convergence_check_every": 20}
        stack_candidate = _candidate(
            candidate_id="candidate-stack",
            overall="promote",
            overrides=stacked_overrides,
            steady_state_ms=80.0,
        )

        table = build_lever_attempt_table(
            baseline,
            candidates,
            ranked_levers=ranked_levers,
            stack_candidate=stack_candidate,
        )

        assert table.baseline_id == baseline_id
        assert table.stacked_overrides == stacked_overrides
        assert table.stack_speed_ratio == 200.0 / 80.0


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
        deferred_ids = {entry["lever_id"] for entry in backlog["entries"] if entry["status"] == "deferred"}
        assert "inductor-cache-warm-policy" not in deferred_ids


class TestCodePathLeverCatalog:
    def test_code_path_levers_registered_with_target_files(self) -> None:
        from linum_basic.benchmark.profile import _LEVER_DEFINITIONS

        catalog = {lever_id: (target, description) for lever_id, target, _cost, description in _LEVER_DEFINITIONS}
        assert catalog["dct-kernel-tuning"] == (
            "linum_basic/backend.py",
            "Optimize DCT matmul path inside compiled ALM step.",
        )
        assert catalog["compile-shape-stability"][0] == "linum_basic/_alm.py"
        assert catalog["compile-shape-stability"][1]
        assert catalog["inductor-cache-warm-policy"] == (
            "linum_basic/_torch_cache.py",
            "Extra untimed warm fits before measured benchmark repeats for steady-state timing.",
        )


class TestBuildPhase5OptimizationReport:
    def test_ranked_report_includes_per_lever_and_stack_metrics(self) -> None:
        from linum_basic.benchmark import build_phase5_optimization_report
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

        def _candidate(
            *,
            candidate_id: str,
            overall: str,
            overrides: dict[str, object],
            steady_state_ms: float,
            failures: list[str] | None = None,
        ) -> CandidateArtifact:
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
                    "telemetry": {"steady_state_ms": steady_state_ms},
                    "quality_verdict": {
                        "passed": overall == "promote",
                        "failures": failures or [],
                    },
                    "overall": overall,
                    "overrides_applied": overrides,
                },
                environment={},
                timestamp="2026-06-30T17:00:00Z",
            )

        candidates = [
            _candidate(
                candidate_id="candidate-sync",
                overall="promote",
                overrides={"convergence_check_every": 20},
                steady_state_ms=100.0,
            ),
            _candidate(
                candidate_id="candidate-reweight",
                overall="reject",
                overrides={"reweighting_tolerance": 0.005},
                steady_state_ms=150.0,
                failures=["seam_l1"],
            ),
        ]
        ranked_levers = (
            RankedLever("sync-cadence", "linum_basic/_alm.py", 1, "medium", "sync gate notes"),
            RankedLever("reweighting-tolerance", "linum_basic/core.py", 2, "medium", "reweight notes"),
        )
        stack_candidate = _candidate(
            candidate_id="candidate-stack",
            overall="promote",
            overrides={"convergence_check_every": 20},
            steady_state_ms=80.0,
        )
        table = build_lever_attempt_table(
            baseline,
            candidates,
            ranked_levers=ranked_levers,
            stack_candidate=stack_candidate,
        )
        bottleneck = build_bottleneck_report({"key_averages": [{"name": "aten::mm", "self_cuda_time_total": 500.0}]})

        report = build_phase5_optimization_report(
            table,
            bottleneck_report=bottleneck,
        )

        assert report["schema_version"] == "1"
        assert report["baseline_id"] == baseline_id
        assert report["stack_speed_ratio"] == 200.0 / 80.0
        assert report["stacked_overrides"] == {"convergence_check_every": 20}
        assert "entry_class_coverage" in report
        assert set(report["entry_class_coverage"]) >= {
            "convergence-policy",
            "sync-cadence",
            "tile-subsampling",
            "precision-tf32",
            "chunking",
            "compile-shape",
        }

        entries = report["entries"]
        assert len(entries) == 2
        assert entries[0]["lever_id"] == "sync-cadence"
        assert entries[0]["target_file"] == "linum_basic/_alm.py"
        assert entries[0]["speed_ratio"] == 2.0
        assert entries[0]["quality_passed"] is True
        assert entries[0]["seam_l1_passed"] is True
        assert entries[0]["seam_curvature_passed"] is True
        assert entries[0]["overall"] == "promote"
        assert entries[0]["bottleneck_evidence"]

        reject_entry = entries[1]
        assert reject_entry["lever_id"] == "reweighting-tolerance"
        assert reject_entry["overall"] == "reject"
        assert reject_entry["quality_passed"] is False
        assert reject_entry["seam_l1_passed"] is False
        assert reject_entry["reject_rationale"]


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


class TestBuildPhase5FastPath:
    def test_promote_case_carries_stack_and_code_path_flags(self) -> None:
        manifest = build_phase5_fast_path(
            {"convergence_check_every": 20},
            baseline_id="baseline-20260630T163351-e7c47a4-sub-22",
            lever_stack=["sync-cadence"],
            code_path_flags={
                "dct_kernel": "tuned",
                "compile_mode": "reduce-overhead",
                "inductor_warm_passes": 2,
            },
            evidence_artifact_ids=[
                "baseline-20260630T163351-e7c47a4-sub-22",
                "candidate-sync",
            ],
            git_commit="abc1234",
            stack_speed_ratio=2.0,
            end_to_end_ms=100.0,
            timestamp="2026-06-30T18:00:00Z",
        )
        assert manifest["schema_version"] == "1"
        assert manifest["working_size"] == 128
        assert manifest["baseline_id"] == "baseline-20260630T163351-e7c47a4-sub-22"
        assert manifest["stacked_overrides"] == {"convergence_check_every": 20}
        assert manifest["lever_stack"] == ["sync-cadence"]
        assert manifest["code_path_flags"] == {
            "dct_kernel": "tuned",
            "compile_mode": "reduce-overhead",
            "inductor_warm_passes": 2,
        }
        assert manifest["evidence_artifact_ids"] == [
            "baseline-20260630T163351-e7c47a4-sub-22",
            "candidate-sync",
        ]
        assert manifest["git_commit"] == "abc1234"
        assert manifest["stack_speed_ratio"] == 2.0
        assert manifest["end_to_end_ms"] == 100.0
        assert manifest["no_optimization"] is False
        assert manifest["timestamp"] == "2026-06-30T18:00:00Z"

    def test_zero_promote_fallback_uses_production_defaults(self) -> None:
        manifest = build_phase5_fast_path(
            {},
            baseline_id="baseline-20260630T163351-e7c47a4-sub-22",
            lever_stack=[],
            code_path_flags=None,
            evidence_artifact_ids=["baseline-20260630T163351-e7c47a4-sub-22"],
            timestamp="2026-06-30T18:00:00Z",
        )
        assert manifest["stacked_overrides"] == {}
        assert manifest["lever_stack"] == []
        assert manifest["no_optimization"] is True
        assert manifest["code_path_flags"] == {
            "dct_kernel": "default",
            "compile_mode": "default",
            "inductor_warm_passes": 0,
        }


def _mode(
    *,
    strategy: str,
    end_to_end_ms: float,
    quality_passed: bool,
    peak_vram_bytes: int = 0,
    fork_model: str | None = None,
    artifact_id: str = "candidate-test",
) -> dict:
    fork_models = {
        "multi": "maxForks_2_scalar_per_gpu",
        "batched": "maxForks_1_batched_multi_gpu",
    }
    return {
        "strategy": strategy,
        "fork_model": fork_model or fork_models.get(strategy, strategy),
        "end_to_end_ms": end_to_end_ms,
        "per_z_ms": end_to_end_ms / 5,
        "steady_state_ms": end_to_end_ms * 0.9,
        "quality_verdict": {"passed": quality_passed, "failures": ()},
        "artifact_id": artifact_id,
        "peak_vram_bytes": peak_vram_bytes,
        "gpu_map": {"cuda:0": "worker-0", "cuda:1": "worker-1"},
    }


class TestBuildPhase6ConcurrencyVerdict:
    _BASELINE = "baseline-20260701T020128-be1e880-sub-22"
    _FAST_PATH = "/scratch/ws128-opt/27/phase5-fast-path.json"

    @staticmethod
    def _evidence() -> list[str]:
        return [
            TestBuildPhase6ConcurrencyVerdict._BASELINE,
            "candidate-multi",
            "candidate-batched",
        ]

    def test_lowest_end_to_end_ms_wins_among_passing_modes(self) -> None:
        manifest = build_phase6_concurrency_verdict(
            [
                _mode(strategy="multi", end_to_end_ms=120.0, quality_passed=True, artifact_id="candidate-multi"),
                _mode(
                    strategy="batched",
                    end_to_end_ms=100.0,
                    quality_passed=True,
                    artifact_id="candidate-batched",
                ),
            ],
            baseline_id=self._BASELINE,
            phase5_fast_path_ref=self._FAST_PATH,
            evidence_artifact_ids=self._evidence(),
            timestamp="2026-07-01T12:00:00Z",
        )
        assert manifest["winner"] == {
            "strategy": "batched",
            "fork_model": "maxForks_1_batched_multi_gpu",
            "rationale": "lowest end_to_end_ms among quality-passing modes",
        }
        assert manifest["recommended_max_forks"] == 1

    def test_fastest_mode_excluded_when_quality_fails(self) -> None:
        manifest = build_phase6_concurrency_verdict(
            [
                _mode(strategy="multi", end_to_end_ms=80.0, quality_passed=False, artifact_id="candidate-multi"),
                _mode(
                    strategy="batched",
                    end_to_end_ms=110.0,
                    quality_passed=True,
                    artifact_id="candidate-batched",
                ),
            ],
            baseline_id=self._BASELINE,
            phase5_fast_path_ref=self._FAST_PATH,
            evidence_artifact_ids=self._evidence(),
            timestamp="2026-07-01T12:00:00Z",
        )
        assert manifest["winner"]["strategy"] == "batched"
        assert manifest["recommended_max_forks"] == 1

    def test_equal_end_to_end_ms_tie_breaks_on_peak_vram(self) -> None:
        manifest = build_phase6_concurrency_verdict(
            [
                _mode(
                    strategy="multi",
                    end_to_end_ms=100.0,
                    quality_passed=True,
                    peak_vram_bytes=8_000_000_000,
                    artifact_id="candidate-multi",
                ),
                _mode(
                    strategy="batched",
                    end_to_end_ms=100.0,
                    quality_passed=True,
                    peak_vram_bytes=4_000_000_000,
                    artifact_id="candidate-batched",
                ),
            ],
            baseline_id=self._BASELINE,
            phase5_fast_path_ref=self._FAST_PATH,
            evidence_artifact_ids=self._evidence(),
            timestamp="2026-07-01T12:00:00Z",
        )
        assert manifest["winner"]["strategy"] == "batched"
        assert "peak_vram" in manifest["winner"]["rationale"].lower() or "vram" in manifest["winner"]["rationale"].lower()

    def test_all_modes_fail_quality_yields_no_winner(self) -> None:
        manifest = build_phase6_concurrency_verdict(
            [
                _mode(strategy="multi", end_to_end_ms=80.0, quality_passed=False),
                _mode(strategy="batched", end_to_end_ms=90.0, quality_passed=False),
            ],
            baseline_id=self._BASELINE,
            phase5_fast_path_ref=self._FAST_PATH,
            evidence_artifact_ids=self._evidence(),
            timestamp="2026-07-01T12:00:00Z",
        )
        assert manifest["winner"] is None
        assert manifest["recommended_max_forks"] is None
        assert "all modes failed" in manifest["selection_rationale"].lower()

    def test_required_schema_keys_present(self) -> None:
        manifest = build_phase6_concurrency_verdict(
            [
                _mode(strategy="multi", end_to_end_ms=100.0, quality_passed=True),
            ],
            baseline_id=self._BASELINE,
            phase5_fast_path_ref=self._FAST_PATH,
            evidence_artifact_ids=self._evidence(),
            git_commit="abc1234",
            timestamp="2026-07-01T12:00:00Z",
        )
        assert manifest["schema_version"] == "1"
        assert manifest["baseline_id"] == self._BASELINE
        assert manifest["timestamp"] == "2026-07-01T12:00:00Z"
        assert manifest["phase5_fast_path_ref"] == self._FAST_PATH
        assert manifest["evidence_artifact_ids"] == self._evidence()
        assert manifest["git_commit"] == "abc1234"
        assert isinstance(manifest["modes"], list)
        assert len(manifest["modes"]) == 1
        mode = manifest["modes"][0]
        for key in (
            "strategy",
            "fork_model",
            "end_to_end_ms",
            "per_z_ms",
            "steady_state_ms",
            "quality_verdict",
            "artifact_id",
            "peak_vram_bytes",
        ):
            assert key in mode
        assert "winner" in manifest
        assert "recommended_max_forks" in manifest
        assert "gpu_allocation_map" in manifest


class TestHistoricalBaseline:
    def test_is_fast_era_true_at_1_20_boundary(self) -> None:
        assert is_fast_era(120.0, 100.0) is True

    def test_is_fast_era_false_at_1_19_boundary(self) -> None:
        assert is_fast_era(119.0, 100.0) is False

    def test_is_fast_era_false_when_baseline_non_positive(self) -> None:
        assert is_fast_era(100.0, 0.0) is False
        assert is_fast_era(100.0, -1.0) is False

    def test_historical_baseline_asdict_round_trip(self) -> None:
        baseline = HistoricalBaseline(
            git_commit="511c88c",
            steady_state_ms=5000.0,
            end_to_end_ms=None,
            artifact_id="iteration-ab-z27",
            change_class="torch_compile_inductor",
            is_fast_era=True,
        )
        payload = asdict(baseline)
        assert payload["end_to_end_ms"] is None
        assert payload["change_class"] == "torch_compile_inductor"
        assert payload["is_fast_era"] is True


class TestHistoricalRegressionTriage:
    def test_reject_preserves_algorithm_or_env_drift(self) -> None:
        result = build_regression_triage_result(
            harness_overall="reject",
            wallclock_regressed=True,
            current_steady_state_ms=229_000.0,
            historical_baselines=(),
        )
        assert result["regression_triage"] == "algorithm_or_env_drift"
        assert result["historical_regression_detected"] is False
        assert result["fast_era_commits"] == []
        assert result["best_fast_era"] is None

    def test_promote_with_wallclock_regressed_is_pipeline_issue(self) -> None:
        result = build_regression_triage_result(
            harness_overall="promote",
            wallclock_regressed=True,
            current_steady_state_ms=229_000.0,
            historical_baselines=(),
        )
        assert result["regression_triage"] == "pipeline_orchestration_issue"

    def test_promote_without_wallclock_regressed_is_no_regression(self) -> None:
        result = build_regression_triage_result(
            harness_overall="promote",
            wallclock_regressed=False,
            current_steady_state_ms=5_000.0,
            historical_baselines=(),
        )
        assert result["regression_triage"] == "no_regression"

    def test_fast_era_baselines_populate_historical_regression(self) -> None:
        baselines = (
            HistoricalBaseline(
                git_commit="be1e880",
                steady_state_ms=229_000.0,
                end_to_end_ms=250_000.0,
                artifact_id="iteration-ab-z27",
                change_class="torch_compile_inductor",
                is_fast_era=False,
            ),
            HistoricalBaseline(
                git_commit="511c88c",
                steady_state_ms=5_000.0,
                end_to_end_ms=6_000.0,
                artifact_id="iteration-ab-z27",
                change_class="linumpy_config",
                is_fast_era=True,
            ),
        )
        result = build_regression_triage_result(
            harness_overall="promote",
            wallclock_regressed=False,
            current_steady_state_ms=229_000.0,
            historical_baselines=baselines,
        )
        assert result["historical_regression_detected"] is True
        assert result["fast_era_commits"] == ["511c88c"]
        assert result["best_fast_era"]["git_commit"] == "511c88c"
        assert result["best_fast_era"]["steady_state_ms"] == 5_000.0

    def test_no_fast_era_baselines_when_current_is_fast(self) -> None:
        baselines = (
            HistoricalBaseline(
                git_commit="be1e880",
                steady_state_ms=229_000.0,
                end_to_end_ms=None,
                artifact_id="iteration-ab-z27",
                change_class="torch_compile_inductor",
                is_fast_era=False,
            ),
            HistoricalBaseline(
                git_commit="511c88c",
                steady_state_ms=5_000.0,
                end_to_end_ms=None,
                artifact_id="iteration-ab-z27",
                change_class="linumpy_config",
                is_fast_era=False,
            ),
        )
        result = build_regression_triage_result(
            harness_overall="promote",
            wallclock_regressed=False,
            current_steady_state_ms=5_000.0,
            historical_baselines=baselines,
        )
        assert result["historical_regression_detected"] is False
        assert result["fast_era_commits"] == []
        assert result["best_fast_era"] is None

    def test_invalid_harness_overall_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="harness_overall"):
            build_regression_triage_result(
                harness_overall="unknown",
                wallclock_regressed=False,
                current_steady_state_ms=5_000.0,
                historical_baselines=(),
            )


class TestLoadIterationAbBaselines:
    @pytest.fixture
    def fixture_path(self, tmp_path: pytest.TempPathFactory) -> object:
        from pathlib import Path

        repo_fixture = Path(__file__).resolve().parent / "fixtures" / "iteration-ab-z27.json"
        if repo_fixture.is_file():
            return repo_fixture
        fixture = tmp_path / "iteration-ab-z27.json"
        fixture.write_text(
            """{
  "slice_id": 27,
  "pre_fix": {
    "git_commit": "be1e880",
    "steady_state_ms": 229000.0,
    "end_to_end_ms": 250000.0,
    "backend": "numpy",
    "change_class": "torch_compile_inductor",
    "linumpy_config": {"fix_illum_smoothness_flatfield": 0.05}
  },
  "post_fix": {
    "git_commit": "511c88c",
    "steady_state_ms": 5000.0,
    "end_to_end_ms": 6000.0,
    "backend": "torch",
    "change_class": "linumpy_config",
    "linumpy_config": {"fix_illum_smoothness_flatfield": null}
  }
}""",
            encoding="utf-8",
        )
        return fixture

    def test_loads_pre_and_post_fix_rows(self, fixture_path: object) -> None:
        baselines = load_historical_baselines_from_iteration_ab(
            fixture_path,
            current_steady_state_ms=229_000.0,
        )
        assert len(baselines) == 2
        commits = {row.git_commit for row in baselines}
        assert "be1e880" in commits
        assert "511c88c" in commits

    def test_post_fix_is_fast_era_against_pre_fix_current(self, fixture_path: object) -> None:
        baselines = load_historical_baselines_from_iteration_ab(
            fixture_path,
            current_steady_state_ms=229_000.0,
        )
        post_fix = next(row for row in baselines if row.git_commit.startswith("511c88c"))
        assert post_fix.is_fast_era is True
        assert post_fix.change_class == "linumpy_config"

    def test_pre_fix_maps_torch_compile_change_class(self, fixture_path: object) -> None:
        baselines = load_historical_baselines_from_iteration_ab(
            fixture_path,
            current_steady_state_ms=229_000.0,
        )
        pre_fix = next(row for row in baselines if row.git_commit.startswith("be1e880"))
        assert pre_fix.change_class == "torch_compile_inductor"

    def test_missing_required_keys_raises_value_error(self, tmp_path: pytest.TempPathFactory) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text('{"slice_id": 27}', encoding="utf-8")
        with pytest.raises(ValueError, match=r"bad\.json"):
            load_historical_baselines_from_iteration_ab(bad, current_steady_state_ms=100.0)


class TestForensicsRecoveryLevers:
    def test_recovery_levers_ordered_with_required_fields(self) -> None:
        levers = build_forensics_recovery_levers()
        assert len(levers) >= 2
        assert levers[0].lever_id == "worker-compile-off"
        assert levers[0].priority == 1
        assert levers[1].lever_id == "auto-l-s-config"
        assert levers[1].priority == 2
        for lever in levers:
            assert lever.target_file
            assert lever.expected_risk
            assert lever.gate_notes

    def test_change_attribution_documents_fore02_classes(self) -> None:
        baselines = (
            HistoricalBaseline(
                git_commit="be1e880",
                steady_state_ms=229_000.0,
                end_to_end_ms=None,
                artifact_id="iteration-ab-z27",
                change_class="torch_compile_inductor",
                is_fast_era=False,
            ),
            HistoricalBaseline(
                git_commit="511c88c",
                steady_state_ms=5_000.0,
                end_to_end_ms=None,
                artifact_id="iteration-ab-z27",
                change_class="linumpy_config",
                is_fast_era=True,
            ),
        )
        attribution = build_forensics_change_attribution(baselines)
        classes = {entry["change_class"] for entry in attribution}
        assert "torch_compile_inductor" in classes
        assert "linumpy_config" in classes
        assert all("evidence_summary" in entry for entry in attribution)


class TestForensicsReport:
    @pytest.fixture
    def baselines(self) -> tuple[HistoricalBaseline, ...]:
        from pathlib import Path

        fixture = Path(__file__).resolve().parent / "fixtures" / "iteration-ab-z27.json"
        return tuple(load_historical_baselines_from_iteration_ab(fixture, current_steady_state_ms=229_000.0))

    def test_report_schema_keys(self, baselines: tuple[HistoricalBaseline, ...]) -> None:
        report = build_forensics_report(
            slice_id=27,
            historical_baselines=baselines,
            current_steady_state_ms=229_000.0,
            current_git_commit="511c88c",
            harness_compare_overall="promote",
            wallclock_regressed=False,
            evidence_artifact_ids=["iteration-ab-z27.json"],
        )
        for key in (
            "schema_version",
            "slice_id",
            "workload",
            "current_git_commit",
            "historical_baselines",
            "regression_triage",
            "change_attribution",
            "ranked_recovery_levers",
            "evidence_artifact_ids",
            "timestamp",
            "fast_era_commits",
        ):
            assert key in report
        assert report["workload"]["working_size"] == 128
        assert report["fast_era_commits"]
        assert "timeline_commits" not in report
        assert "basicpy_reference" not in report
        assert "bisect_windows" not in report

    def test_ranked_recovery_levers_match_ranked_lever_schema(self, baselines: tuple[HistoricalBaseline, ...]) -> None:
        report = build_forensics_report(
            slice_id=27,
            historical_baselines=baselines,
            current_steady_state_ms=229_000.0,
            current_git_commit="511c88c",
            evidence_artifact_ids=["iteration-ab-z27.json"],
        )
        levers = report["ranked_recovery_levers"]
        assert levers[0]["lever_id"] == "worker-compile-off"
        for lever in levers:
            assert {"lever_id", "target_file", "priority", "expected_risk", "gate_notes"} <= set(lever)


class TestForensicsReportBundle:
    def test_writes_json_and_markdown(
        self, tmp_path: pytest.TempPathFactory, baselines: tuple[HistoricalBaseline, ...] | None = None
    ) -> None:
        from pathlib import Path

        if baselines is None:
            fixture = Path(__file__).resolve().parent / "fixtures" / "iteration-ab-z27.json"
            baselines = tuple(load_historical_baselines_from_iteration_ab(fixture, current_steady_state_ms=229_000.0))
        report = build_forensics_report(
            slice_id=27,
            historical_baselines=baselines,
            current_steady_state_ms=229_000.0,
            current_git_commit="511c88c",
            evidence_artifact_ids=["iteration-ab-z27.json"],
        )
        out = tmp_path / "forensics"
        write_forensics_report_bundle(out, report)
        json_path = out / "forensics-report.json"
        md_path = out / "forensics-report.md"
        assert json_path.is_file()
        assert md_path.is_file()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["slice_id"] == 27
        assert (
            "worker compile" in md_path.read_text(encoding="utf-8").lower()
            or "compile" in md_path.read_text(encoding="utf-8").lower()
        )


class TestRegression_triage:
    def test_reject_with_wallclock_regressed_is_algorithm_or_env_drift(self) -> None:
        assert diagnose_regression_triage(harness_overall="reject", wallclock_regressed=True) == "algorithm_or_env_drift"

    def test_reject_without_wallclock_regressed_is_algorithm_or_env_drift(self) -> None:
        assert diagnose_regression_triage(harness_overall="reject", wallclock_regressed=False) == "algorithm_or_env_drift"

    def test_promote_with_wallclock_regressed_is_pipeline_orchestration_issue(self) -> None:
        assert (
            diagnose_regression_triage(harness_overall="promote", wallclock_regressed=True) == "pipeline_orchestration_issue"
        )

    def test_promote_without_wallclock_regressed_is_no_regression(self) -> None:
        assert diagnose_regression_triage(harness_overall="promote", wallclock_regressed=False) == "no_regression"

    def test_invalid_harness_overall_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="harness_overall"):
            diagnose_regression_triage(harness_overall="unknown", wallclock_regressed=False)


class TestGit_commit_drift:
    def test_none_manifest_commit_returns_none(self) -> None:
        assert warn_git_commit_drift(manifest_git_commit=None, current_git_commit="abc1234") is None

    def test_matching_commits_returns_none(self) -> None:
        assert warn_git_commit_drift(manifest_git_commit="abc1234", current_git_commit="abc1234") is None

    def test_differing_commits_returns_warning_with_both_hashes(self) -> None:
        warning = warn_git_commit_drift(manifest_git_commit="oldcommit", current_git_commit="newcommit")
        assert warning is not None
        assert "oldcommit" in warning
        assert "newcommit" in warning


class TestPhase7_integration_summary:
    _BASELINE = "baseline-20260701T020128-be1e880-sub-22"
    _FAST_PATH = "/scratch/ws128-opt/27/phase5-fast-path.json"

    @staticmethod
    def _env() -> dict[str, str]:
        return {
            "LINUM_BASIC_DCT_KERNEL": "scipy",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor",
            "CUDA_VISIBLE_DEVICES": "0",
        }

    @staticmethod
    def _evidence() -> list[str]:
        return [
            TestPhase7_integration_summary._BASELINE,
            "candidate-integration",
        ]

    def test_required_schema_keys_present(self) -> None:
        manifest = build_phase7_integration_summary(
            baseline_id=self._BASELINE,
            phase5_fast_path_ref=self._FAST_PATH,
            harness_compare_overall="promote",
            wallclock_regressed=False,
            env_snapshot=self._env(),
            current_git_commit="current123",
            evidence_artifact_ids=self._evidence(),
            fast_path_git_commit="current123",
            timestamp="2026-07-01T12:00:00Z",
        )
        assert manifest["schema_version"] == "1"
        assert manifest["baseline_id"] == self._BASELINE
        assert manifest["timestamp"] == "2026-07-01T12:00:00Z"
        assert manifest["phase5_fast_path_ref"] == self._FAST_PATH
        assert manifest["env_snapshot"] == self._env()
        assert manifest["git_commit"] == "current123"
        assert manifest["git_commit_drift_warning"] is None
        assert manifest["harness_compare_overall"] == "promote"
        assert manifest["wallclock_regressed"] is False
        assert manifest["regression_triage"] == "no_regression"
        assert manifest["evidence_artifact_ids"] == self._evidence()
        assert "strategy_metadata" not in manifest
        assert "nextflow_wallclock_ms" not in manifest

    def test_env_snapshot_is_defensive_copy(self) -> None:
        env = self._env()
        manifest = build_phase7_integration_summary(
            baseline_id=self._BASELINE,
            phase5_fast_path_ref=self._FAST_PATH,
            harness_compare_overall="promote",
            wallclock_regressed=False,
            env_snapshot=env,
            current_git_commit="current123",
            evidence_artifact_ids=self._evidence(),
            timestamp="2026-07-01T12:00:00Z",
        )
        assert manifest["env_snapshot"] == env
        assert manifest["env_snapshot"] is not env
        env["LINUM_BASIC_DCT_KERNEL"] = "mutated"
        assert manifest["env_snapshot"]["LINUM_BASIC_DCT_KERNEL"] == "scipy"

    def test_optional_fields_present_when_provided(self) -> None:
        strategy = {"strategy": "batched", "fork_model": "maxForks_1"}
        manifest = build_phase7_integration_summary(
            baseline_id=self._BASELINE,
            phase5_fast_path_ref=self._FAST_PATH,
            harness_compare_overall="promote",
            wallclock_regressed=True,
            env_snapshot=self._env(),
            current_git_commit="current123",
            evidence_artifact_ids=self._evidence(),
            strategy_metadata=strategy,
            nextflow_wallclock_ms=1500.0,
            timestamp="2026-07-01T12:00:00Z",
        )
        assert manifest["strategy_metadata"] == strategy
        assert manifest["nextflow_wallclock_ms"] == 1500.0
        assert manifest["regression_triage"] == "pipeline_orchestration_issue"

    def test_git_commit_drift_warning_matches_helper(self) -> None:
        manifest = build_phase7_integration_summary(
            baseline_id=self._BASELINE,
            phase5_fast_path_ref=self._FAST_PATH,
            harness_compare_overall="reject",
            wallclock_regressed=False,
            env_snapshot=self._env(),
            current_git_commit="current123",
            evidence_artifact_ids=self._evidence(),
            fast_path_git_commit="frozen456",
            timestamp="2026-07-01T12:00:00Z",
        )
        expected = warn_git_commit_drift(manifest_git_commit="frozen456", current_git_commit="current123")
        assert manifest["git_commit_drift_warning"] == expected
        assert manifest["regression_triage"] == diagnose_regression_triage(
            harness_overall="reject",
            wallclock_regressed=False,
        )


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
