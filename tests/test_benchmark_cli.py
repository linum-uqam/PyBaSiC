"""CLI tests for scripts/benchmark_speedup.py baseline subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import benchmark_speedup as bench
from tests.test_cli import _write_synthetic_mosaic_zarr


class TestResolveZSelection:
    def test_z_sample_evenly_spaced(self) -> None:
        indices = bench.resolve_z_selection(20, z_sample=5)
        assert len(indices) == 5
        assert indices == sorted(set(indices))
        assert indices[0] == 0
        assert indices[-1] == 19
        expected = np.linspace(0, 19, num=5, dtype=int).tolist()
        assert indices == expected

    def test_z_indices_parsed(self) -> None:
        indices = bench.resolve_z_selection(20, z_indices="10,0,5")
        assert indices == [0, 5, 10]

    def test_missing_selection_raises(self) -> None:
        with pytest.raises(ValueError, match=r"z-indices|z-sample"):
            bench.resolve_z_selection(20)


class TestLargeRunGuard:
    def test_refuses_large_selection_without_flag(self) -> None:
        with pytest.raises(SystemExit):
            bench.enforce_large_run_guard(
                bench.LARGE_RUN_Z_THRESHOLD + 1,
                allow_large_run=False,
                yes=False,
            )

    def test_allows_with_allow_large_run(self) -> None:
        bench.enforce_large_run_guard(
            bench.LARGE_RUN_Z_THRESHOLD + 1,
            allow_large_run=True,
            yes=False,
        )

    def test_allows_with_yes(self) -> None:
        bench.enforce_large_run_guard(
            bench.LARGE_RUN_Z_THRESHOLD + 1,
            allow_large_run=False,
            yes=True,
        )


class TestBenchmarkPackageExports:
    def test_benchmark_all_exports(self) -> None:
        import linum_basic.benchmark as b

        assert b.__all__
        for name in b.__all__:
            assert hasattr(b, name)


class TestBaselineRequiredArgs:
    def test_non_synthetic_missing_output_dir_exits_nonzero(self, tmp_path: Path) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        zarr_in = tmp_path / "in.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=2, n_cols=2, tile=8)

        rc = bench.main(
            [
                "baseline",
                "--input",
                str(zarr_in),
                "--subject-id",
                "subj",
                "--z-sample",
                "2",
            ]
        )
        assert rc != 0

    def test_real_run_missing_z_selection_exits_nonzero(self, tmp_path: Path) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        zarr_in = tmp_path / "in.ome.zarr"
        out_dir = tmp_path / "out"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=2, n_cols=2, tile=8)

        rc = bench.main(
            [
                "baseline",
                "--input",
                str(zarr_in),
                "--subject-id",
                "subj",
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc != 0


class TestBaselineSyntheticIntegration:
    def test_synthetic_baseline_writes_bundle(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.fit import MosaicFit

        zarr_in = tmp_path / "in.ome.zarr"
        out_dir = tmp_path / "artifacts"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=3, n_rows=2, n_cols=2, tile=8)

        def _stub_fit(mosaic, **kwargs):
            z_indices = kwargs.get("z_indices") or list(range(mosaic.n_z))
            th, tw = mosaic.tile_shape
            n = len(z_indices)
            return MosaicFit(
                flatfields=np.ones((n, th, tw), dtype=np.float32),
                darkfields=np.zeros((n, th, tw), dtype=np.float32),
                field_mode="per-z",
                z_indices=list(z_indices),
                params=dict(kwargs.get("basic_kwargs") or {}),
            )

        monkeypatch.setattr(bench, "fit_mosaic", _stub_fit)
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)

        rc = bench.main(
            [
                "baseline",
                "--input",
                str(zarr_in),
                "--subject-id",
                "syn",
                "--output-dir",
                str(out_dir),
                "--z-sample",
                "2",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "1",
            ]
        )
        assert rc == 0

        baseline_dirs = list(out_dir.glob("baseline-*"))
        assert len(baseline_dirs) == 1
        bundle_dir = baseline_dirs[0]

        bundle_files = list(bundle_dir.glob("*.json"))
        assert any("bundle" in f.name for f in bundle_files)
        assert any("tolerance" in f.name for f in bundle_files)
        assert (bundle_dir / "summary.md").exists()

        bundle_path = next(f for f in bundle_files if "bundle" in f.name)
        bundle_data = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert bundle_data["metadata"]["release_gate"] is False
        assert "git_commit" in bundle_data["metadata"]
        precision = bundle_data["metadata"]["precision"]
        assert "allow_tf32_matmul" in precision
        assert "float32_matmul_precision" in precision
        assert "compile_requested" in precision
        assert bundle_data["z_indices"]
        assert bundle_data["array_shape"]
        assert bundle_data["tile_shape"]
        assert bundle_data["strategy_params"]
        assert bundle_data["timestamp"]

        for key, value in bundle_data.items():
            if isinstance(value, list) and len(value) > 1000:
                pytest.fail(f"unexpected large array payload at {key}")

        tolerance_path = next(f for f in bundle_files if "tolerance" in f.name)
        tolerance_data = json.loads(tolerance_path.read_text(encoding="utf-8"))
        assert tolerance_data["calibration_policy"]
        for metric in ("seam_l1", "seam_curvature"):
            assert "mean" in tolerance_data["tolerances"][metric]
            assert "std" in tolerance_data["tolerances"][metric]
            assert "abs_tol" in tolerance_data["tolerances"][metric]
            assert "rel_tol" in tolerance_data["tolerances"][metric]

    def test_baseline_persists_compile_cache_telemetry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.fit import MosaicFit

        zarr_in = tmp_path / "in.ome.zarr"
        out_dir = tmp_path / "artifacts"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=3, n_rows=2, n_cols=2, tile=8)

        def _stub_fit(mosaic, **kwargs):
            z_indices = kwargs.get("z_indices") or list(range(mosaic.n_z))
            th, tw = mosaic.tile_shape
            n = len(z_indices)
            return MosaicFit(
                flatfields=np.ones((n, th, tw), dtype=np.float32),
                darkfields=np.zeros((n, th, tw), dtype=np.float32),
                field_mode="per-z",
                z_indices=list(z_indices),
                params=dict(kwargs.get("basic_kwargs") or {}),
            )

        monkeypatch.setattr(bench, "fit_mosaic", _stub_fit)
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)

        rc = bench.main(
            [
                "baseline",
                "--input",
                str(zarr_in),
                "--subject-id",
                "syn",
                "--output-dir",
                str(out_dir),
                "--z-sample",
                "2",
                "--strategy",
                "batched",
                "--batched-z-chunk-size",
                "8",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "1",
            ]
        )
        assert rc == 0

        bundle_path = next(out_dir.glob("baseline-*/*bundle*.json"))
        bundle_data = json.loads(bundle_path.read_text(encoding="utf-8"))
        telemetry = bundle_data["metadata"]["telemetry"]
        assert telemetry["chunk_size"] == 8
        assert "compile_status" in telemetry
        assert telemetry["compile_status"] in ("disabled", "unavailable", "enabled")
        assert "inductor_cache_path" in telemetry
        assert "fx_graph_cache_enabled" in telemetry
        assert "steady_state_ms" in telemetry

    def test_cpu_only_no_cuda_required(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.fit import MosaicFit

        zarr_in = tmp_path / "in.ome.zarr"
        out_dir = tmp_path / "artifacts"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=2, n_cols=2, tile=8)

        def _stub_fit(mosaic, **kwargs):
            z_indices = kwargs.get("z_indices") or list(range(mosaic.n_z))
            th, tw = mosaic.tile_shape
            n = len(z_indices)
            return MosaicFit(
                flatfields=np.ones((n, th, tw), dtype=np.float32),
                darkfields=np.zeros((n, th, tw), dtype=np.float32),
                field_mode="per-z",
                z_indices=list(z_indices),
            )

        monkeypatch.setattr(bench, "fit_mosaic", _stub_fit)
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)

        try:
            import torch

            monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        except ImportError:
            pass

        rc = bench.main(
            [
                "baseline",
                "--input",
                str(zarr_in),
                "--subject-id",
                "syn",
                "--output-dir",
                str(out_dir),
                "--z-sample",
                "2",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "0",
            ]
        )
        assert rc == 0


def _stub_fit_factory(*, seam_offset: float = 0.0):
    """Return a CPU-only fit_mosaic stub for harness integration tests."""

    def _stub_fit(mosaic, **kwargs):
        from linum_basic.fit import MosaicFit

        z_indices = kwargs.get("z_indices") or list(range(mosaic.n_z))
        th, tw = mosaic.tile_shape
        n = len(z_indices)
        flat = np.ones((n, th, tw), dtype=np.float32) * (1.0 + seam_offset)
        convergence_per_z = [{"reweighting_iteration": 2, "l_s": 0.5, "l_d": 0.2} for _ in z_indices]
        return MosaicFit(
            flatfields=flat,
            darkfields=np.zeros((n, th, tw), dtype=np.float32),
            field_mode="per-z",
            z_indices=list(z_indices),
            params=dict(kwargs.get("basic_kwargs") or {}),
            convergence_per_z=convergence_per_z,
        )

    return _stub_fit


def _run_stubbed_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, subject_id: str = "syn") -> str:
    """Create a baseline bundle via cmd_baseline with stubbed fit; return baseline_id."""
    pytest.importorskip("zarr")
    pytest.importorskip("ome_zarr")

    zarr_in = tmp_path / "in.ome.zarr"
    out_dir = tmp_path / "artifacts"
    _write_synthetic_mosaic_zarr(zarr_in, n_z=3, n_rows=2, n_cols=2, tile=8)

    monkeypatch.setattr(bench, "fit_mosaic", _stub_fit_factory())
    monkeypatch.setattr(bench, "_cuda_available", lambda: False)

    rc = bench.main(
        [
            "baseline",
            "--input",
            str(zarr_in),
            "--subject-id",
            subject_id,
            "--output-dir",
            str(out_dir),
            "--z-sample",
            "2",
            "--synthetic",
            "--repeats",
            "1",
            "--warmup",
            "1",
        ]
    )
    assert rc == 0
    baseline_dirs = list(out_dir.glob("baseline-*"))
    assert len(baseline_dirs) == 1
    bundle_path = next(baseline_dirs[0].glob("*bundle*.json"))
    bundle_data = json.loads(bundle_path.read_text(encoding="utf-8"))
    return bundle_data["baseline_id"]


def _write_sidecar_per_z_outlier_fixtures(artifacts_dir: Path) -> dict[str, object]:
    """Write baseline, sidecar, and candidate artifacts for per-z relative outlier tests."""
    from linum_basic.benchmark.artifacts import (
        SCHEMA_VERSION,
        BaselineBundle,
        CandidateArtifact,
        ToleranceSidecar,
        write_artifact,
    )
    from linum_basic.benchmark.quality import METRIC_DEFINITION_VERSION

    baseline_id = "baseline-20260101-abc1234-syn"
    bundle_dir = artifacts_dir / baseline_id
    bundle_dir.mkdir(parents=True)

    z_indices = [0, 2]
    baseline_rows = [
        {"z": 0, "seam_l1": 0.1, "seam_curvature": 0.02},
        {"z": 2, "seam_l1": 0.05, "seam_curvature": 0.02},
    ]
    baseline = BaselineBundle(
        baseline_id=baseline_id,
        uuid="550e8400-e29b-41d4-a716-446655440000",
        schema_version=SCHEMA_VERSION,
        metric_definition_version=METRIC_DEFINITION_VERSION,
        subject_id="syn",
        run_label="test",
        input_fingerprint="sha256:deadbeef",
        z_indices=z_indices,
        array_shape=[3, 16, 16],
        tile_shape=[16, 16],
        strategy_params={"working_size": 128},
        metrics_rows=baseline_rows,
        metrics_aggregates={"seam_l1": 0.1, "seam_curvature": 0.02},
        repeats=1,
        metadata={"git_commit": "abc1234567890abcdef", "telemetry": {"steady_state_ms": 100.0}},
        timestamp="2026-01-01T12:00:00Z",
    )
    sidecar = ToleranceSidecar(
        baseline_id=baseline_id,
        schema_version=SCHEMA_VERSION,
        metric_definition_version=METRIC_DEFINITION_VERSION,
        calibration_policy="mean+3std",
        sigma=3.0,
        min_abs=1e-6,
        tolerances={
            "seam_l1": {"mean": 0.1, "std": 0.001, "abs_tol": 0.01, "rel_tol": 0.1},
            "seam_curvature": {"mean": 0.02, "std": 0.001, "abs_tol": 0.01, "rel_tol": 0.1},
        },
    )
    candidate_rows = [
        {"z": 0, "seam_l1": 0.101, "seam_curvature": 0.0205},
        {"z": 2, "seam_l1": 0.0526, "seam_curvature": 0.0205},
    ]
    candidate_id = "candidate-20260102-def5678-syn"
    candidate = CandidateArtifact(
        candidate_id=candidate_id,
        baseline_id=baseline_id,
        schema_version=SCHEMA_VERSION,
        metric_definition_version=METRIC_DEFINITION_VERSION,
        subject_id="syn",
        run_label="batched-v1",
        input_fingerprint="sha256:deadbeef",
        z_indices=z_indices,
        array_shape=[3, 16, 16],
        tile_shape=[16, 16],
        strategy_params={"working_size": 128},
        metrics_rows=candidate_rows,
        metrics_aggregates={"seam_l1": 0.102, "seam_curvature": 0.0205},
        repeats=1,
        metadata={"telemetry": {"steady_state_ms": 50.0}},
        environment={"cuda_version": "12.4"},
        timestamp="2026-01-02T12:00:00Z",
    )

    write_artifact(bundle_dir / "baseline-bundle.json", baseline)
    write_artifact(bundle_dir / "tolerance-sidecar.json", sidecar)
    candidate_dir = artifacts_dir / candidate_id
    candidate_dir.mkdir()
    write_artifact(candidate_dir / "candidate-artifact.json", candidate)

    return {
        "baseline_id": baseline_id,
        "baseline_path": bundle_dir / "baseline-bundle.json",
        "candidate_path": candidate_dir / "candidate-artifact.json",
        "outlier_z": 2,
    }


def _write_promotion_eligible_fixtures(artifacts_dir: Path) -> dict[str, object]:
    """Write baseline/candidate artifacts that pass quality and exceed the speed threshold."""
    from linum_basic.benchmark.artifacts import (
        SCHEMA_VERSION,
        BaselineBundle,
        CandidateArtifact,
        ToleranceSidecar,
        write_artifact,
    )
    from linum_basic.benchmark.quality import METRIC_DEFINITION_VERSION

    baseline_id = "baseline-20260101-abc1234-syn"
    bundle_dir = artifacts_dir / baseline_id
    bundle_dir.mkdir(parents=True)

    z_indices = [0, 2]
    baseline_rows = [
        {"z": 0, "seam_l1": 0.1, "seam_curvature": 0.02},
        {"z": 2, "seam_l1": 0.05, "seam_curvature": 0.02},
    ]
    baseline = BaselineBundle(
        baseline_id=baseline_id,
        uuid="550e8400-e29b-41d4-a716-446655440000",
        schema_version=SCHEMA_VERSION,
        metric_definition_version=METRIC_DEFINITION_VERSION,
        subject_id="syn",
        run_label="test",
        input_fingerprint="sha256:deadbeef",
        z_indices=z_indices,
        array_shape=[3, 16, 16],
        tile_shape=[16, 16],
        strategy_params={"working_size": 128},
        metrics_rows=baseline_rows,
        metrics_aggregates={"seam_l1": 0.1, "seam_curvature": 0.02},
        repeats=1,
        metadata={
            "git_commit": "abc1234567890abcdef",
            "telemetry": {"steady_state_ms": 100.0},
            "operator_timing": {"end_to_end_ms": 120.0},
        },
        timestamp="2026-01-01T12:00:00Z",
    )
    sidecar = ToleranceSidecar(
        baseline_id=baseline_id,
        schema_version=SCHEMA_VERSION,
        metric_definition_version=METRIC_DEFINITION_VERSION,
        calibration_policy="mean+3std",
        sigma=3.0,
        min_abs=1e-6,
        tolerances={
            "seam_l1": {"mean": 0.1, "std": 0.001, "abs_tol": 0.01, "rel_tol": 0.1},
            "seam_curvature": {"mean": 0.02, "std": 0.001, "abs_tol": 0.01, "rel_tol": 0.1},
        },
    )
    candidate_id = "candidate-20260102-def5678-syn"
    candidate = CandidateArtifact(
        candidate_id=candidate_id,
        baseline_id=baseline_id,
        schema_version=SCHEMA_VERSION,
        metric_definition_version=METRIC_DEFINITION_VERSION,
        subject_id="syn",
        run_label="batched-v1",
        input_fingerprint="sha256:deadbeef",
        z_indices=z_indices,
        array_shape=[3, 16, 16],
        tile_shape=[16, 16],
        strategy_params={"working_size": 128},
        metrics_rows=list(baseline_rows),
        metrics_aggregates={"seam_l1": 0.1, "seam_curvature": 0.02},
        repeats=1,
        metadata={
            "telemetry": {"steady_state_ms": 50.0},
            "operator_timing": {"end_to_end_ms": 60.0},
        },
        environment={"cuda_version": "12.4"},
        timestamp="2026-01-02T12:00:00Z",
    )

    write_artifact(bundle_dir / "baseline-bundle.json", baseline)
    write_artifact(bundle_dir / "tolerance-sidecar.json", sidecar)
    candidate_dir = artifacts_dir / candidate_id
    candidate_dir.mkdir()
    write_artifact(candidate_dir / "candidate-artifact.json", candidate)

    return {
        "baseline_id": baseline_id,
        "baseline_path": bundle_dir / "baseline-bundle.json",
        "candidate_path": candidate_dir / "candidate-artifact.json",
    }


def _make_telemetry_record(*, steady_state_ms: float):
    from linum_basic.benchmark.telemetry import TelemetryRecord

    return TelemetryRecord(
        warmup_ms=10.0,
        cold_cache_ms=steady_state_ms,
        warm_cache_ms=steady_state_ms,
        steady_state_ms=steady_state_ms,
        max_memory_allocated_bytes=None,
        max_memory_reserved_bytes=None,
        chunk_size=None,
        compile_status="disabled",
        inductor_cache_path=None,
        device=None,
        backend="numpy",
    )


class TestOperatorTimingMetadata:
    def test_operator_timing_metadata_derives_fields(self) -> None:
        telemetry = _make_telemetry_record(steady_state_ms=600_000.0)

        result = bench._operator_timing_metadata(telemetry, n_z=2, n_tiles=4)

        assert result["end_to_end_ms"] == 600_000.0
        assert result["per_z_ms"] == 300_000.0
        assert result["n_z"] == 2
        assert result["n_tiles"] == 4

    def test_operator_timing_per_z_none_when_n_z_zero(self) -> None:
        telemetry = _make_telemetry_record(steady_state_ms=100.0)

        result = bench._operator_timing_metadata(telemetry, n_z=0, n_tiles=4)

        assert result["per_z_ms"] is None
        assert result["n_z"] == 0
        assert result["n_tiles"] == 4
        assert result["end_to_end_ms"] == 100.0

    def test_candidate_artifact_contains_operator_timing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        out_dir = tmp_path / "artifacts"
        baseline_id = _run_stubbed_baseline(tmp_path, monkeypatch)

        zarr_in = tmp_path / "in.ome.zarr"
        monkeypatch.setattr(bench, "fit_mosaic", _stub_fit_factory())
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)

        rc = bench.main(
            [
                "candidate",
                "--input",
                str(zarr_in),
                "--baseline-id",
                baseline_id,
                "--output-dir",
                str(out_dir),
                "--subject-id",
                "syn",
                "--strategy",
                "batched",
                "--batched-z-chunk-size",
                "8",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "1",
            ]
        )
        assert rc == 0

        candidate_path = next(out_dir.glob("candidate-*/candidate-artifact.json"))
        data = json.loads(candidate_path.read_text(encoding="utf-8"))
        meta = data["metadata"]
        assert "telemetry" in meta
        assert "steady_state_ms" in meta["telemetry"]
        operator_timing = meta["operator_timing"]
        assert isinstance(operator_timing["end_to_end_ms"], (int, float))
        assert operator_timing["n_z"] == len(data["z_indices"])
        from linum_basic.mosaic import MosaicGrid

        mosaic = MosaicGrid.from_ome_zarr(str(zarr_in))
        assert operator_timing["n_tiles"] == mosaic.n_tiles


class TestCandidateSubcommand:
    def test_candidate_writes_verdict_artifact(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        out_dir = tmp_path / "artifacts"
        baseline_id = _run_stubbed_baseline(tmp_path, monkeypatch)

        zarr_in = tmp_path / "in.ome.zarr"
        monkeypatch.setattr(bench, "fit_mosaic", _stub_fit_factory())
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)

        rc = bench.main(
            [
                "candidate",
                "--input",
                str(zarr_in),
                "--baseline-id",
                baseline_id,
                "--output-dir",
                str(out_dir),
                "--subject-id",
                "syn",
                "--strategy",
                "batched",
                "--batched-z-chunk-size",
                "8",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "1",
            ]
        )
        assert rc == 0

        candidate_dirs = list(out_dir.glob("candidate-*"))
        assert len(candidate_dirs) == 1
        candidate_path = candidate_dirs[0] / "candidate-artifact.json"
        assert candidate_path.exists()
        data = json.loads(candidate_path.read_text(encoding="utf-8"))
        assert data["baseline_id"] == baseline_id
        meta = data["metadata"]
        assert "speed_verdict" in meta
        assert "quality_verdict" in meta
        assert meta["overall"] in ("promote", "reject")
        assert (candidate_dirs[0] / "summary.md").exists()
        telemetry = meta["telemetry"]
        assert telemetry["chunk_size"] == 8
        assert "compile_status" in telemetry
        assert telemetry["compile_status"] in ("disabled", "unavailable", "enabled")
        precision = meta["precision"]
        assert "allow_tf32_matmul" in precision
        assert "float32_matmul_precision" in precision
        assert "compile_requested" in precision
        assert "inductor_cache_path" in telemetry
        assert "fx_graph_cache_enabled" in telemetry
        assert "steady_state_ms" in telemetry
        assert "deltas" in meta
        assert "convergence" in meta
        convergence = meta["convergence"]
        assert convergence["reweight_iterations_median"] == 2.0
        assert convergence["max_reweighting_iterations"] == 15
        assert convergence["reweight_iterations_per_z"] == {"0": 2, "2": 2}

    def test_candidate_emits_d15_concurrency_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        out_dir = tmp_path / "artifacts"
        baseline_id = _run_stubbed_baseline(tmp_path, monkeypatch)

        zarr_in = tmp_path / "in.ome.zarr"
        captured_kwargs: list[dict] = []

        def _capturing_fit(mosaic, **kwargs):
            captured_kwargs.append(dict(kwargs))
            return _stub_fit_factory()(mosaic, **kwargs)

        monkeypatch.setattr(bench, "fit_mosaic", _capturing_fit)
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)

        rc = bench.main(
            [
                "candidate",
                "--input",
                str(zarr_in),
                "--baseline-id",
                baseline_id,
                "--output-dir",
                str(out_dir),
                "--subject-id",
                "syn",
                "--strategy",
                "batched",
                "--batched-z-chunk-size",
                "8",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "1",
            ]
        )
        assert rc == 0
        assert captured_kwargs
        assert captured_kwargs[-1].get("strategy") == "batched"

        candidate_dirs = list(out_dir.glob("candidate-*"))
        assert len(candidate_dirs) == 1
        data = json.loads((candidate_dirs[0] / "candidate-artifact.json").read_text(encoding="utf-8"))
        meta = data["metadata"]
        concurrency = meta["concurrency"]
        assert concurrency["strategy"] == "batched"
        assert concurrency["fork_model"] == "maxForks_1_batched_multi_gpu"
        assert "n_gpus" in concurrency
        assert "gpu_map" in concurrency
        assert meta["operator_timing"]["end_to_end_ms"] is not None
        assert meta["telemetry"]["steady_state_ms"] is not None

    def test_regressed_candidate_rejects_with_nonzero_exit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from linum_basic.benchmark.quality import MetricDelta, QualityVerdict

        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        out_dir = tmp_path / "artifacts"
        baseline_id = _run_stubbed_baseline(tmp_path, monkeypatch)

        zarr_in = tmp_path / "in.ome.zarr"
        monkeypatch.setattr(bench, "fit_mosaic", _stub_fit_factory())
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)

        def _failing_gate(deltas, tolerances, **kwargs):
            del deltas, tolerances, kwargs
            return QualityVerdict(
                passed=False,
                failures=("seam_l1",),
                worst_z=0,
                aggregate_deltas={"seam_l1": MetricDelta(1.0, 1.0)},
                per_z_failures=(),
            )

        monkeypatch.setattr(bench, "evaluate_quality_gate", _failing_gate)

        rc = bench.main(
            [
                "candidate",
                "--input",
                str(zarr_in),
                "--baseline-id",
                baseline_id,
                "--output-dir",
                str(out_dir),
                "--subject-id",
                "syn",
                "--strategy",
                "batched",
                "--batched-z-chunk-size",
                "8",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "1",
            ]
        )
        assert rc != 0

        candidate_dirs = list(out_dir.glob("candidate-*"))
        assert candidate_dirs
        data = json.loads((candidate_dirs[0] / "candidate-artifact.json").read_text(encoding="utf-8"))
        assert data["metadata"]["overall"] == "reject"

    def test_candidate_rejects_sidecar_reloaded_per_z_relative_outlier(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from linum_basic.benchmark.quality import METRIC_DEFINITION_VERSION, PerZMetricRow, QualityReport

        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        artifacts_dir = tmp_path / "artifacts"
        fixtures = _write_sidecar_per_z_outlier_fixtures(artifacts_dir)
        outlier_z = int(fixtures["outlier_z"])

        zarr_in = tmp_path / "in.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=3, n_rows=2, n_cols=2, tile=8)

        monkeypatch.setattr(bench, "_input_fingerprint", lambda path, shape: "sha256:deadbeef")
        monkeypatch.setattr(bench, "fit_mosaic", _stub_fit_factory())
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)

        def _crafted_report(mosaic, fit, **kwargs):
            del mosaic, fit, kwargs
            return QualityReport(
                rows=(
                    PerZMetricRow(z=0, seam_l1=0.101, seam_curvature=0.0205),
                    PerZMetricRow(z=2, seam_l1=0.0526, seam_curvature=0.0205),
                ),
                aggregates={"seam_l1": 0.102, "seam_curvature": 0.0205},
                metric_definition_version=METRIC_DEFINITION_VERSION,
            )

        monkeypatch.setattr(bench, "compute_quality_report", _crafted_report)

        rc = bench.main(
            [
                "candidate",
                "--input",
                str(zarr_in),
                "--baseline-id",
                str(fixtures["baseline_id"]),
                "--output-dir",
                str(artifacts_dir),
                "--subject-id",
                "syn",
                "--strategy",
                "batched",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "1",
            ]
        )
        assert rc != 0

        candidate_dirs = sorted(artifacts_dir.glob("candidate-*"), key=lambda p: p.name)
        assert len(candidate_dirs) >= 2
        data = json.loads((candidate_dirs[-1] / "candidate-artifact.json").read_text(encoding="utf-8"))
        assert data["metadata"]["overall"] == "reject"
        quality = data["metadata"]["quality_verdict"]
        assert quality["worst_z"] == outlier_z
        assert quality["per_z_failures"]
        assert any(row["z"] == outlier_z and row["metric"] == "seam_l1" for row in quality["per_z_failures"])

    def test_subject_mismatch_refuses_comparison(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        out_dir = tmp_path / "artifacts"
        baseline_id = _run_stubbed_baseline(tmp_path, monkeypatch, subject_id="baseline-subject")

        zarr_in = tmp_path / "in.ome.zarr"
        monkeypatch.setattr(bench, "fit_mosaic", _stub_fit_factory())
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)

        rc = bench.main(
            [
                "candidate",
                "--input",
                str(zarr_in),
                "--baseline-id",
                baseline_id,
                "--output-dir",
                str(out_dir),
                "--subject-id",
                "different-subject",
                "--strategy",
                "batched",
                "--batched-z-chunk-size",
                "8",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "1",
            ]
        )
        assert rc != 0

    def test_fingerprint_mismatch_refuses_before_fit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        out_dir = tmp_path / "artifacts"
        baseline_id = _run_stubbed_baseline(tmp_path, monkeypatch)

        zarr_in = tmp_path / "in.ome.zarr"
        monkeypatch.setattr(bench, "fit_mosaic", _stub_fit_factory())
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)
        monkeypatch.setattr(bench, "_input_fingerprint", lambda path, shape: "sha256:other-volume")

        rc = bench.main(
            [
                "candidate",
                "--input",
                str(zarr_in),
                "--baseline-id",
                baseline_id,
                "--output-dir",
                str(out_dir),
                "--subject-id",
                "syn",
                "--strategy",
                "batched",
                "--batched-z-chunk-size",
                "8",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "1",
            ]
        )
        assert rc != 0
        assert not list(out_dir.glob("candidate-*"))


class TestPhase11PromotionVerdict:
    @staticmethod
    def _quality_verdict(*, passed: bool):
        from linum_basic.benchmark.quality import MetricDelta, QualityVerdict

        return QualityVerdict(
            passed=passed,
            failures=() if passed else ("seam_l1",),
            worst_z=None,
            aggregate_deltas={"seam_l1": MetricDelta(0.0, 0.0)},
            per_z_failures=(),
        )

    def test_promotion_verdict_both_pass_eligible(self) -> None:
        from linum_basic.benchmark.sweep import SPEED_RATIO_THRESHOLD

        speed_verdict = {"ratio": 1.35, "label": "faster"}
        quality_verdict = self._quality_verdict(passed=True)
        result = bench.build_phase11_promotion_verdict(speed_verdict, quality_verdict)
        assert result["promotion_eligible"] is True
        assert result["speed_passed"] is True
        assert result["quality_passed"] is True
        assert result["speed_ratio"] == 1.35
        assert result["threshold"] == SPEED_RATIO_THRESHOLD
        assert result["metric"] == "steady_state_ms"

    def test_promotion_verdict_quality_pass_speed_fail(self) -> None:
        speed_verdict = {"ratio": 1.10, "label": "faster"}
        quality_verdict = self._quality_verdict(passed=True)
        result = bench.build_phase11_promotion_verdict(speed_verdict, quality_verdict)
        assert result["promotion_eligible"] is False
        assert result["speed_passed"] is False
        assert result["quality_passed"] is True

    def test_promotion_verdict_speed_pass_quality_fail(self) -> None:
        speed_verdict = {"ratio": 1.50, "label": "faster"}
        quality_verdict = self._quality_verdict(passed=False)
        result = bench.build_phase11_promotion_verdict(speed_verdict, quality_verdict)
        assert result["promotion_eligible"] is False
        assert result["speed_passed"] is True
        assert result["quality_passed"] is False

    def test_promotion_verdict_none_ratio(self) -> None:
        speed_verdict = {"ratio": None, "label": "insufficient_telemetry"}
        quality_verdict = self._quality_verdict(passed=True)
        result = bench.build_phase11_promotion_verdict(speed_verdict, quality_verdict)
        assert result["speed_passed"] is False
        assert result["promotion_eligible"] is False
        assert result["quality_passed"] is True


class TestCompareSubcommand:
    def test_compare_writes_summary_from_stored_artifacts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        out_dir = tmp_path / "artifacts"
        baseline_id = _run_stubbed_baseline(tmp_path, monkeypatch)
        baseline_dir = out_dir / baseline_id
        baseline_path = next(baseline_dir.glob("*bundle*.json"))

        zarr_in = tmp_path / "in.ome.zarr"
        monkeypatch.setattr(bench, "fit_mosaic", _stub_fit_factory())
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)

        rc = bench.main(
            [
                "candidate",
                "--input",
                str(zarr_in),
                "--baseline-id",
                baseline_id,
                "--output-dir",
                str(out_dir),
                "--subject-id",
                "syn",
                "--strategy",
                "batched",
                "--batched-z-chunk-size",
                "8",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "1",
            ]
        )
        assert rc == 0
        candidate_path = next(out_dir.glob("candidate-*/candidate-artifact.json"))

        compare_out = tmp_path / "compare"
        rc = bench.main(
            [
                "compare",
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
                "--output-dir",
                str(compare_out),
            ]
        )
        assert rc == 0
        summary_json = compare_out / "compare-summary.json"
        assert summary_json.exists()
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        assert summary["overall"] in ("promote", "reject")
        assert (compare_out / "summary.md").exists() or (compare_out / "summary.csv").exists()

    def test_compare_includes_promotion_verdict(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        fixtures = _write_promotion_eligible_fixtures(tmp_path / "artifacts")
        compare_out = tmp_path / "compare-promotion"
        rc = bench.main(
            [
                "compare",
                "--baseline",
                str(fixtures["baseline_path"]),
                "--candidate",
                str(fixtures["candidate_path"]),
                "--output-dir",
                str(compare_out),
            ]
        )
        assert rc == 0
        summary = json.loads((compare_out / "compare-summary.json").read_text(encoding="utf-8"))
        promotion = summary["promotion_verdict"]
        assert "promotion_eligible" in promotion
        assert promotion["metric"] == "steady_state_ms"
        assert promotion["promotion_eligible"] is True
        assert promotion["speed_passed"] is True
        assert promotion["quality_passed"] is True
        assert promotion["speed_ratio"] == pytest.approx(2.0)
        timing = summary["timing_report"]
        assert timing["primary_metric"] == "steady_state_ms"
        assert timing["candidate_steady_state_ms"] == 50.0
        assert timing["baseline_steady_state_ms"] == 100.0
        assert timing["candidate_end_to_end_ms"] == 60.0
        assert timing["baseline_end_to_end_ms"] == 120.0
        captured = capsys.readouterr()
        assert "Phase 11 promotion eligibility" in captured.out
        assert "promotion_eligible=True" in captured.out

    def test_compare_reports_insufficient_telemetry_when_candidate_telemetry_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from linum_basic.benchmark.artifacts import CandidateArtifact, read_artifact, write_artifact

        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        out_dir = tmp_path / "artifacts"
        baseline_id = _run_stubbed_baseline(tmp_path, monkeypatch)
        baseline_path = next((out_dir / baseline_id).glob("*bundle*.json"))

        zarr_in = tmp_path / "in.ome.zarr"
        monkeypatch.setattr(bench, "fit_mosaic", _stub_fit_factory())
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)
        bench.main(
            [
                "candidate",
                "--input",
                str(zarr_in),
                "--baseline-id",
                baseline_id,
                "--output-dir",
                str(out_dir),
                "--subject-id",
                "syn",
                "--strategy",
                "batched",
                "--batched-z-chunk-size",
                "8",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "1",
            ]
        )
        candidate_path = next(out_dir.glob("candidate-*/candidate-artifact.json"))
        candidate = read_artifact(candidate_path, CandidateArtifact)
        stripped = CandidateArtifact(
            candidate_id=candidate.candidate_id,
            baseline_id=candidate.baseline_id,
            schema_version=candidate.schema_version,
            metric_definition_version=candidate.metric_definition_version,
            subject_id=candidate.subject_id,
            run_label=candidate.run_label,
            input_fingerprint=candidate.input_fingerprint,
            z_indices=candidate.z_indices,
            array_shape=candidate.array_shape,
            tile_shape=candidate.tile_shape,
            strategy_params=candidate.strategy_params,
            metrics_rows=candidate.metrics_rows,
            metrics_aggregates=candidate.metrics_aggregates,
            repeats=candidate.repeats,
            metadata={**candidate.metadata, "telemetry": {}},
            environment=candidate.environment,
            timestamp=candidate.timestamp,
        )
        stripped_path = tmp_path / "stripped-candidate.json"
        write_artifact(stripped_path, stripped)

        compare_out = tmp_path / "compare-missing-telemetry"
        rc = bench.main(
            [
                "compare",
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(stripped_path),
                "--output-dir",
                str(compare_out),
            ]
        )
        assert rc == 0
        summary = json.loads((compare_out / "compare-summary.json").read_text(encoding="utf-8"))
        assert summary["speed_verdict"]["label"] == "insufficient_telemetry"
        assert summary["speed_verdict"]["ratio"] is None

    def test_compare_does_not_call_fit_mosaic(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        out_dir = tmp_path / "artifacts"
        baseline_id = _run_stubbed_baseline(tmp_path, monkeypatch)
        baseline_path = next((out_dir / baseline_id).glob("*bundle*.json"))

        zarr_in = tmp_path / "in.ome.zarr"
        monkeypatch.setattr(bench, "fit_mosaic", _stub_fit_factory())
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)
        bench.main(
            [
                "candidate",
                "--input",
                str(zarr_in),
                "--baseline-id",
                baseline_id,
                "--output-dir",
                str(out_dir),
                "--subject-id",
                "syn",
                "--strategy",
                "batched",
                "--batched-z-chunk-size",
                "8",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "1",
            ]
        )
        candidate_path = next(out_dir.glob("candidate-*/candidate-artifact.json"))

        def _boom(*args, **kwargs):
            raise RuntimeError("fit_mosaic must not be called during compare")

        monkeypatch.setattr(bench, "fit_mosaic", _boom)
        rc = bench.main(
            [
                "compare",
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
                "--output-dir",
                str(tmp_path / "compare"),
            ]
        )
        assert rc == 0

    def test_compare_rejects_sidecar_reloaded_per_z_relative_outlier(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        artifacts_dir = tmp_path / "artifacts"
        fixtures = _write_sidecar_per_z_outlier_fixtures(artifacts_dir)
        outlier_z = int(fixtures["outlier_z"])

        def _boom(*args, **kwargs):
            raise RuntimeError("fit_mosaic must not be called during compare")

        monkeypatch.setattr(bench, "fit_mosaic", _boom)

        compare_out = tmp_path / "compare-outlier"
        rc = bench.main(
            [
                "compare",
                "--baseline",
                str(fixtures["baseline_path"]),
                "--candidate",
                str(fixtures["candidate_path"]),
                "--output-dir",
                str(compare_out),
            ]
        )
        assert rc != 0

        summary = json.loads((compare_out / "compare-summary.json").read_text(encoding="utf-8"))
        assert summary["overall"] == "reject"
        assert summary["worst_z"] == outlier_z
        quality = summary["quality_verdict"]
        assert quality["per_z_failures"]
        assert any(row["z"] == outlier_z and row["metric"] == "seam_l1" for row in quality["per_z_failures"])

    def test_compare_refuses_mismatched_artifacts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from linum_basic.benchmark.artifacts import BaselineBundle, read_artifact, write_artifact

        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        out_dir = tmp_path / "artifacts"
        baseline_id = _run_stubbed_baseline(tmp_path, monkeypatch)
        baseline_path = next((out_dir / baseline_id).glob("*bundle*.json"))

        zarr_in = tmp_path / "in.ome.zarr"
        monkeypatch.setattr(bench, "fit_mosaic", _stub_fit_factory())
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)
        bench.main(
            [
                "candidate",
                "--input",
                str(zarr_in),
                "--baseline-id",
                baseline_id,
                "--output-dir",
                str(out_dir),
                "--subject-id",
                "syn",
                "--strategy",
                "batched",
                "--batched-z-chunk-size",
                "8",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "1",
            ]
        )
        candidate_path = next(out_dir.glob("candidate-*/candidate-artifact.json"))

        baseline = read_artifact(baseline_path, BaselineBundle)
        bad_baseline = BaselineBundle(
            baseline_id=baseline.baseline_id,
            uuid=baseline.uuid,
            schema_version=baseline.schema_version,
            metric_definition_version=baseline.metric_definition_version,
            subject_id="other-subject",
            run_label=baseline.run_label,
            input_fingerprint=baseline.input_fingerprint,
            z_indices=baseline.z_indices,
            array_shape=baseline.array_shape,
            tile_shape=baseline.tile_shape,
            strategy_params=baseline.strategy_params,
            metrics_rows=baseline.metrics_rows,
            metrics_aggregates=baseline.metrics_aggregates,
            repeats=baseline.repeats,
            metadata=baseline.metadata,
            timestamp=baseline.timestamp,
        )
        bad_path = tmp_path / "bad-baseline.json"
        write_artifact(bad_path, bad_baseline)

        rc = bench.main(
            [
                "compare",
                "--baseline",
                str(bad_path),
                "--candidate",
                str(candidate_path),
                "--output-dir",
                str(tmp_path / "compare"),
            ]
        )
        assert rc != 0

    def test_compare_refuses_fingerprint_mismatch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from linum_basic.benchmark.artifacts import CandidateArtifact, read_artifact, write_artifact

        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        out_dir = tmp_path / "artifacts"
        baseline_id = _run_stubbed_baseline(tmp_path, monkeypatch)
        baseline_path = next((out_dir / baseline_id).glob("*bundle*.json"))

        zarr_in = tmp_path / "in.ome.zarr"
        monkeypatch.setattr(bench, "fit_mosaic", _stub_fit_factory())
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)
        bench.main(
            [
                "candidate",
                "--input",
                str(zarr_in),
                "--baseline-id",
                baseline_id,
                "--output-dir",
                str(out_dir),
                "--subject-id",
                "syn",
                "--strategy",
                "batched",
                "--batched-z-chunk-size",
                "8",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "1",
            ]
        )
        candidate_path = next(out_dir.glob("candidate-*/candidate-artifact.json"))

        candidate = read_artifact(candidate_path, CandidateArtifact)
        bad_candidate = CandidateArtifact(
            candidate_id=candidate.candidate_id,
            baseline_id=candidate.baseline_id,
            schema_version=candidate.schema_version,
            metric_definition_version=candidate.metric_definition_version,
            subject_id=candidate.subject_id,
            run_label=candidate.run_label,
            input_fingerprint="sha256:other-volume",
            z_indices=candidate.z_indices,
            array_shape=candidate.array_shape,
            tile_shape=candidate.tile_shape,
            strategy_params=candidate.strategy_params,
            metrics_rows=candidate.metrics_rows,
            metrics_aggregates=candidate.metrics_aggregates,
            repeats=candidate.repeats,
            metadata=candidate.metadata,
            environment=candidate.environment,
            timestamp=candidate.timestamp,
        )
        bad_candidate_path = tmp_path / "bad-candidate.json"
        write_artifact(bad_candidate_path, bad_candidate)

        rc = bench.main(
            [
                "compare",
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(bad_candidate_path),
                "--output-dir",
                str(tmp_path / "compare"),
            ]
        )
        assert rc != 0


class TestIntegrationCheckSubcommand:
    @staticmethod
    def _baseline_and_candidate(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[Path, Path, str]:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        out_dir = tmp_path / "artifacts"
        baseline_id = _run_stubbed_baseline(tmp_path, monkeypatch)
        baseline_path = next((out_dir / baseline_id).glob("*bundle*.json"))

        zarr_in = tmp_path / "in.ome.zarr"
        monkeypatch.setattr(bench, "fit_mosaic", _stub_fit_factory())
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)

        rc = bench.main(
            [
                "candidate",
                "--input",
                str(zarr_in),
                "--baseline-id",
                baseline_id,
                "--output-dir",
                str(out_dir),
                "--subject-id",
                "syn",
                "--strategy",
                "batched",
                "--batched-z-chunk-size",
                "8",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "1",
            ]
        )
        assert rc == 0
        candidate_path = next(out_dir.glob("candidate-*/candidate-artifact.json"))
        return baseline_path, candidate_path, baseline_id

    def test_integration_check_writes_summary_with_git_drift_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from linum_basic.benchmark.metadata import collect_git_commit

        baseline_path, candidate_path, _baseline_id = self._baseline_and_candidate(tmp_path, monkeypatch)

        fast_path = tmp_path / "phase5-fast-path.json"
        fast_path.write_text(
            json.dumps({"schema_version": "1", "git_commit": "deadbeef0000"}) + "\n",
            encoding="utf-8",
        )

        out = tmp_path / "integration-drift"
        rc = bench.main(
            [
                "integration-check",
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
                "--output-dir",
                str(out),
                "--fast-path-ref",
                str(fast_path),
            ]
        )
        assert rc in (0, 1)
        summary_path = out / "phase7-integration-summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["regression_triage"] in {
            "algorithm_or_env_drift",
            "pipeline_orchestration_issue",
            "no_regression",
        }
        assert summary["git_commit_drift_warning"]
        assert summary["git_commit"] == collect_git_commit()
        assert "env_snapshot" in summary
        assert not (out / "compare-summary.json").exists()

    def test_integration_check_without_fast_path_ref_has_no_drift_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        baseline_path, candidate_path, _baseline_id = self._baseline_and_candidate(tmp_path, monkeypatch)

        out = tmp_path / "integration-no-drift"
        rc = bench.main(
            [
                "integration-check",
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
                "--output-dir",
                str(out),
            ]
        )
        assert rc in (0, 1)
        summary = json.loads((out / "phase7-integration-summary.json").read_text(encoding="utf-8"))
        assert summary["git_commit_drift_warning"] is None


class TestSweepSubcommand:
    def test_sweep_writes_table_and_verdict_without_fit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from tests.test_benchmark_sweep import _write_sweep_fixtures

        fixtures = _write_sweep_fixtures(tmp_path / "artifacts")
        baseline = fixtures["baseline"]
        candidates = fixtures["candidates"]
        baseline_path = tmp_path / "artifacts" / baseline.baseline_id / "baseline-bundle.json"
        candidate_paths = [
            tmp_path / "artifacts" / candidate.candidate_id / "candidate-artifact.json" for candidate in candidates
        ]
        out_dir = tmp_path / "sweep-out"

        def _raise_fit(*_args, **_kwargs):
            raise AssertionError("fit_mosaic must not be called by sweep")

        monkeypatch.setattr(bench, "fit_mosaic", _raise_fit)

        rc = bench.main(
            [
                "sweep",
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_paths[0]),
                "--candidate",
                str(candidate_paths[1]),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        assert (out_dir / "sweep-table.json").exists()
        assert (out_dir / "sweep-verdict.json").exists()
        assert (out_dir / "summary.md").exists()

        verdict = json.loads((out_dir / "sweep-verdict.json").read_text(encoding="utf-8"))
        assert "recommended_ws" in verdict
        assert "phase3_activate" in verdict
        assert "rationale" in verdict
        assert verdict["recommended_ws"] == 64
        assert verdict["phase3_activate"] is False


class TestProfileSubcommand:
    def test_profile_help_lists_sequential_mode(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            bench.main(["profile", "--help"])
        captured = capsys.readouterr()
        assert "--mode" in captured.out
        assert "sequential" in captured.out

    def test_profile_sequential_writes_bottleneck_report(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        zarr_in = tmp_path / "in.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=3, n_rows=2, n_cols=2, tile=8)
        out_dir = tmp_path / "profile-out"

        def _fake_fit(*_args, **_kwargs):
            return object()

        def _fake_profiler_events(*_args, **_kwargs):
            return {
                "key_averages": [
                    {"name": "aten::mm", "self_cuda_time_total": 250.0},
                ]
            }

        monkeypatch.setattr(bench, "fit_mosaic", _fake_fit)
        monkeypatch.setattr(bench, "_collect_profiler_events", _fake_profiler_events)

        rc = bench.main(
            [
                "profile",
                "--input",
                str(zarr_in),
                "--subject-id",
                "sub-22",
                "--output-dir",
                str(out_dir),
                "--strategy",
                "baseline",
                "--working-size",
                "128",
                "--max-reweighting-iterations",
                "500",
                "--estimate-darkfield",
                "--repeats",
                "1",
                "--warmup",
                "0",
                "--z-sample",
                "2",
                "--synthetic",
            ]
        )
        assert rc == 0
        report_path = out_dir / "bottleneck-report.json"
        assert report_path.exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["primary_limit"] == "compute"
        assert payload["ranked_levers"]
        assert payload["working_size"] == 128
        precision_path = out_dir / "profile-metadata.json"
        assert precision_path.exists()
        precision_payload = json.loads(precision_path.read_text(encoding="utf-8"))
        assert "allow_tf32_matmul" in precision_payload["precision"]
        assert "float32_matmul_precision" in precision_payload["precision"]

    def test_batched_diagnostic_requires_sequential_report(self, tmp_path: Path) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        zarr_in = tmp_path / "in.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=2, n_cols=2, tile=8)
        out_dir = tmp_path / "profile-out"

        rc = bench.main(
            [
                "profile",
                "--mode",
                "batched-diagnostic",
                "--input",
                str(zarr_in),
                "--subject-id",
                "sub-22",
                "--output-dir",
                str(out_dir),
                "--strategy",
                "baseline",
                "--working-size",
                "128",
                "--z-sample",
                "2",
                "--synthetic",
            ]
        )
        assert rc != 0
        assert not (out_dir / "batched-handoff.json").exists()

    def test_batched_diagnostic_writes_handoff_and_force_batched(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.fit import _allow_batched_cuda_for_params

        zarr_in = tmp_path / "in.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=3, n_rows=2, n_cols=2, tile=8)
        out_dir = tmp_path / "profile-out"
        out_dir.mkdir()
        (out_dir / "bottleneck-report.json").write_text(
            json.dumps({"primary_limit": "chunking", "working_size": 128}),
            encoding="utf-8",
        )

        captured_kwargs: dict[str, object] = {}

        def _fake_fit(*_args, **kwargs):
            captured_kwargs.update(kwargs.get("basic_kwargs", {}))
            return object()

        monkeypatch.setattr(bench, "fit_mosaic", _fake_fit)

        rc = bench.main(
            [
                "profile",
                "--mode",
                "batched-diagnostic",
                "--input",
                str(zarr_in),
                "--subject-id",
                "sub-22",
                "--output-dir",
                str(out_dir),
                "--strategy",
                "baseline",
                "--working-size",
                "128",
                "--z-sample",
                "3",
                "--synthetic",
            ]
        )
        assert rc == 0
        assert captured_kwargs.get("force_batched_cuda") is True
        assert captured_kwargs.get("batched_z_chunk_size") == 3
        assert captured_kwargs.get("working_size") == 128
        assert _allow_batched_cuda_for_params({"working_size": 128}) is False

        handoff_path = out_dir / "batched-handoff.json"
        assert handoff_path.exists()
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        assert handoff["mode"] == "batched-diagnostic"
        assert handoff["n_z"] == 3
        assert handoff["throughput_ratio_vs_sequential"] == 1.0
        assert "peak_memory_bytes" in handoff
        assert "steady_state_ms" in handoff
        assert "primary_limit" in handoff


class TestOptimizeSubcommand:
    def test_optimize_writes_lever_artifacts_from_fixtures(self, tmp_path: Path) -> None:
        from linum_basic.benchmark.artifacts import (
            SCHEMA_VERSION,
            BaselineBundle,
            ToleranceSidecar,
            write_artifact,
        )
        from linum_basic.benchmark.profile import build_bottleneck_report
        from linum_basic.benchmark.quality import CALIBRATION_POLICY, METRIC_DEFINITION_VERSION

        out_dir = tmp_path / "opt-out"
        out_dir.mkdir()
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
        sidecar = ToleranceSidecar(
            baseline_id=baseline_id,
            schema_version=SCHEMA_VERSION,
            metric_definition_version=METRIC_DEFINITION_VERSION,
            calibration_policy=CALIBRATION_POLICY,
            sigma=3.0,
            min_abs=1e-6,
            tolerances={
                "seam_l1": {"mean": 0.1, "std": 0.001, "abs_tol": 0.01, "rel_tol": 0.1},
                "seam_curvature": {"mean": 0.02, "std": 0.001, "abs_tol": 0.01, "rel_tol": 0.1},
            },
        )
        bundle_dir = out_dir / baseline_id
        bundle_dir.mkdir()
        write_artifact(bundle_dir / "baseline-bundle.json", baseline)
        write_artifact(bundle_dir / "tolerance-sidecar.json", sidecar)

        report = build_bottleneck_report({"key_averages": [{"name": "aten::mm", "self_cuda_time_total": 100.0}]})
        (out_dir / "bottleneck-report.json").write_text(
            json.dumps(bench._bottleneck_report_to_dict(report), indent=2) + "\n",
            encoding="utf-8",
        )

        cand_path = out_dir / "candidate-sync.json"
        cand_path.write_text(
            json.dumps(
                {
                    "candidate_id": "candidate-sync",
                    "baseline_id": baseline_id,
                    "schema_version": SCHEMA_VERSION,
                    "metric_definition_version": METRIC_DEFINITION_VERSION,
                    "subject_id": "sub22",
                    "run_label": "candidate-sync",
                    "input_fingerprint": "sha256:deadbeef",
                    "z_indices": [0, 1],
                    "array_shape": [2, 64, 64],
                    "tile_shape": [64, 64],
                    "strategy_params": {"working_size": 128, "convergence_check_every": 20},
                    "metrics_rows": [{"z": 0, "seam_l1": 0.1, "seam_curvature": 0.02}],
                    "metrics_aggregates": {"seam_l1": 0.1, "seam_curvature": 0.02},
                    "repeats": 1,
                    "metadata": {
                        "telemetry": {"steady_state_ms": 100.0},
                        "quality_verdict": {"passed": True, "failures": []},
                        "overall": "promote",
                        "overrides_applied": {"convergence_check_every": 20},
                    },
                    "environment": {},
                    "timestamp": "2026-06-30T17:00:00Z",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        rc = bench.main(
            [
                "optimize",
                "--output-dir",
                str(out_dir),
                "--baseline-id",
                baseline_id,
                "--candidate",
                str(cand_path),
            ]
        )
        assert rc == 0
        assert (out_dir / "lever-attempt-table.json").exists()
        assert (out_dir / "phase5-backlog.json").exists()
        assert (out_dir / "phase3-handoff-config.json").exists()
        assert (out_dir / "phase5-fast-path.json").exists()
        handoff = json.loads((out_dir / "phase3-handoff-config.json").read_text(encoding="utf-8"))
        assert handoff["stacked_overrides"] == {"convergence_check_every": 20}
        fast_path = json.loads((out_dir / "phase5-fast-path.json").read_text(encoding="utf-8"))
        assert fast_path["schema_version"] == "1"
        assert fast_path["baseline_id"] == baseline_id
        assert fast_path["stacked_overrides"] == {"convergence_check_every": 20}
        assert fast_path["code_path_flags"] == {
            "dct_kernel": "default",
            "compile_mode": "default",
            "inductor_warm_passes": 0,
        }
        assert fast_path["no_optimization"] is False
        backlog = json.loads((out_dir / "phase5-backlog.json").read_text(encoding="utf-8"))
        deferred_ids = {entry["lever_id"] for entry in backlog["entries"] if entry["status"] == "deferred"}
        assert "inductor-cache-warm-policy" not in deferred_ids


def _write_concurrency_candidate_fixture(
    path: Path,
    *,
    candidate_id: str,
    baseline_id: str,
    strategy: str,
    fork_model: str,
    end_to_end_ms: float,
    steady_state_ms: float,
    quality_passed: bool,
    peak_vram_bytes: int = 0,
) -> None:
    """Write a candidate-artifact.json with D-15 concurrency metadata for cmd_concurrency tests."""
    from linum_basic.benchmark.artifacts import SCHEMA_VERSION
    from linum_basic.benchmark.quality import METRIC_DEFINITION_VERSION

    per_z_ms = end_to_end_ms / 5.0
    gpu_map = (
        {"cuda:0": "worker-0", "cuda:1": "worker-1"} if strategy == "multi" else {"cuda:0": "batched", "cuda:1": "batched"}
    )
    path.write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "baseline_id": baseline_id,
                "schema_version": SCHEMA_VERSION,
                "metric_definition_version": METRIC_DEFINITION_VERSION,
                "subject_id": "sub22",
                "run_label": candidate_id,
                "input_fingerprint": "sha256:deadbeef",
                "z_indices": [0, 1, 2, 3, 4],
                "array_shape": [5, 64, 64],
                "tile_shape": [64, 64],
                "strategy_params": {"working_size": 128},
                "metrics_rows": [{"z": 0, "seam_l1": 0.1, "seam_curvature": 0.02}],
                "metrics_aggregates": {"seam_l1": 0.1, "seam_curvature": 0.02},
                "repeats": 1,
                "metadata": {
                    "telemetry": {
                        "steady_state_ms": steady_state_ms,
                        "max_memory_allocated_bytes": peak_vram_bytes,
                    },
                    "operator_timing": {
                        "end_to_end_ms": end_to_end_ms,
                        "per_z_ms": per_z_ms,
                        "n_z": 5,
                        "n_tiles": 4,
                    },
                    "quality_verdict": {"passed": quality_passed, "failures": []},
                    "concurrency": {
                        "strategy": strategy,
                        "fork_model": fork_model,
                        "n_gpus": 2,
                        "gpu_map": gpu_map,
                        "inductor_cache_path": "/tmp/inductor-cache",
                        "compile_status": "enabled",
                        "quality_verdict": {"passed": quality_passed, "failures": []},
                    },
                },
                "environment": {},
                "timestamp": "2026-07-01T12:00:00Z",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class TestConcurrencySubcommand:
    def test_concurrency_writes_verdict_from_candidate_fixtures(self, tmp_path: Path) -> None:
        from linum_basic.benchmark.artifacts import (
            SCHEMA_VERSION,
            BaselineBundle,
            ToleranceSidecar,
            write_artifact,
        )
        from linum_basic.benchmark.quality import CALIBRATION_POLICY, METRIC_DEFINITION_VERSION

        out_dir = tmp_path / "conc-out"
        out_dir.mkdir()
        baseline_id = "baseline-20260701T020128-be1e880-sub-22"
        baseline = BaselineBundle(
            baseline_id=baseline_id,
            uuid="550e8400-e29b-41d4-a716-446655440000",
            schema_version=SCHEMA_VERSION,
            metric_definition_version=METRIC_DEFINITION_VERSION,
            subject_id="sub22",
            run_label="production",
            input_fingerprint="sha256:deadbeef",
            z_indices=[0, 1, 2, 3, 4],
            array_shape=[5, 64, 64],
            tile_shape=[64, 64],
            strategy_params={"working_size": 128},
            metrics_rows=[{"z": 0, "seam_l1": 0.1, "seam_curvature": 0.02}],
            metrics_aggregates={"seam_l1": 0.1, "seam_curvature": 0.02},
            repeats=1,
            metadata={"telemetry": {"steady_state_ms": 200.0}},
            timestamp="2026-07-01T02:01:28Z",
        )
        sidecar = ToleranceSidecar(
            baseline_id=baseline_id,
            schema_version=SCHEMA_VERSION,
            metric_definition_version=METRIC_DEFINITION_VERSION,
            calibration_policy=CALIBRATION_POLICY,
            sigma=3.0,
            min_abs=1e-6,
            tolerances={
                "seam_l1": {"mean": 0.1, "std": 0.001, "abs_tol": 0.01, "rel_tol": 0.1},
                "seam_curvature": {"mean": 0.02, "std": 0.001, "abs_tol": 0.01, "rel_tol": 0.1},
            },
        )
        bundle_dir = out_dir / baseline_id
        bundle_dir.mkdir()
        write_artifact(bundle_dir / "baseline-bundle.json", baseline)
        write_artifact(bundle_dir / "tolerance-sidecar.json", sidecar)

        multi_path = out_dir / "candidate-multi.json"
        batched_path = out_dir / "candidate-batched.json"
        _write_concurrency_candidate_fixture(
            multi_path,
            candidate_id="candidate-multi",
            baseline_id=baseline_id,
            strategy="multi",
            fork_model="maxForks_2_scalar_per_gpu",
            end_to_end_ms=120.0,
            steady_state_ms=108.0,
            quality_passed=True,
            peak_vram_bytes=8_000_000_000,
        )
        _write_concurrency_candidate_fixture(
            batched_path,
            candidate_id="candidate-batched",
            baseline_id=baseline_id,
            strategy="batched",
            fork_model="maxForks_1_batched_multi_gpu",
            end_to_end_ms=100.0,
            steady_state_ms=90.0,
            quality_passed=True,
            peak_vram_bytes=4_000_000_000,
        )

        rc = bench.main(
            [
                "concurrency",
                "--output-dir",
                str(out_dir),
                "--baseline-id",
                baseline_id,
                "--candidate",
                str(multi_path),
                "--candidate",
                str(batched_path),
                "--fast-path-ref",
                str(out_dir / "phase5-fast-path.json"),
            ]
        )
        assert rc == 0
        verdict_path = out_dir / "phase6-concurrency-verdict.json"
        assert verdict_path.exists()
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        assert verdict["winner"]["strategy"] == "batched"
        assert verdict["recommended_max_forks"] in (1, 2)
        assert verdict["baseline_id"] == baseline_id


class TestInductorWarmPolicy:
    def test_warm_policy_passes_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from linum_basic._torch_cache import warm_policy_passes

        monkeypatch.delenv("LINUM_BASIC_INDUCTOR_WARM_PASSES", raising=False)
        assert warm_policy_passes() == 0

        monkeypatch.setenv("LINUM_BASIC_INDUCTOR_WARM_PASSES", "3")
        assert warm_policy_passes() == 3

        monkeypatch.setenv("LINUM_BASIC_INDUCTOR_WARM_PASSES", "-2")
        assert warm_policy_passes() == 0

    def test_baseline_records_inductor_warm_passes_and_extra_fits(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.fit import MosaicFit

        zarr_in = tmp_path / "in.ome.zarr"
        out_dir = tmp_path / "artifacts"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=3, n_rows=2, n_cols=2, tile=8)

        fit_calls = 0

        def _stub_fit(mosaic, **kwargs):
            nonlocal fit_calls
            fit_calls += 1
            z_indices = kwargs.get("z_indices") or list(range(mosaic.n_z))
            th, tw = mosaic.tile_shape
            n = len(z_indices)
            return MosaicFit(
                flatfields=np.ones((n, th, tw), dtype=np.float32),
                darkfields=np.zeros((n, th, tw), dtype=np.float32),
                field_mode="per-z",
                z_indices=list(z_indices),
                params=dict(kwargs.get("basic_kwargs") or {}),
            )

        monkeypatch.setattr(bench, "fit_mosaic", _stub_fit)
        monkeypatch.setattr(bench, "_cuda_available", lambda: False)
        monkeypatch.setenv("LINUM_BASIC_INDUCTOR_WARM_PASSES", "2")

        rc = bench.main(
            [
                "baseline",
                "--input",
                str(zarr_in),
                "--subject-id",
                "syn",
                "--output-dir",
                str(out_dir),
                "--z-sample",
                "2",
                "--synthetic",
                "--repeats",
                "1",
                "--warmup",
                "1",
            ]
        )
        assert rc == 0
        assert fit_calls == 4

        baseline_dirs = list(out_dir.glob("baseline-*"))
        bundle_path = next(baseline_dirs[0].glob("*bundle*.json"))
        bundle_data = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert bundle_data["metadata"]["inductor_warm_passes"] == 2
