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
    build_phase5_fast_path,
    build_phase6_concurrency_verdict,
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
