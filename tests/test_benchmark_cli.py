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
        assert "inductor_cache_path" in telemetry
        assert "fx_graph_cache_enabled" in telemetry
        assert "steady_state_ms" in telemetry
        assert "deltas" in meta
        assert "convergence" in meta
        convergence = meta["convergence"]
        assert convergence["reweight_iterations_median"] == 2.0
        assert convergence["max_reweighting_iterations"] == 15
        assert convergence["reweight_iterations_per_z"] == {"0": 2, "2": 2}

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
