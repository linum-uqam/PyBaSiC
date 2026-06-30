"""Tests for versioned benchmark artifact persistence and comparison guards."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import ClassVar

import pytest

from linum_basic.benchmark.artifacts import (
    SCHEMA_VERSION,
    BaselineBundle,
    CandidateArtifact,
    ToleranceSidecar,
    assert_comparable,
    make_baseline_id,
    read_artifact,
    slugify_label,
    write_artifact,
    write_summary_table,
)
from linum_basic.benchmark.quality import METRIC_DEFINITION_VERSION


def _sample_baseline(*, baseline_id: str = "baseline-20260101-abc1234-subject-a") -> BaselineBundle:
    return BaselineBundle(
        baseline_id=baseline_id,
        uuid="550e8400-e29b-41d4-a716-446655440000",
        schema_version=SCHEMA_VERSION,
        metric_definition_version=METRIC_DEFINITION_VERSION,
        subject_id="subject-a",
        run_label="prod-baseline",
        input_fingerprint="sha256:deadbeef",
        z_indices=[0, 4, 8],
        array_shape=[3, 512, 512],
        tile_shape=[128, 128],
        strategy_params={"working_size": 128, "estimate_darkfield": True},
        metrics_rows=[
            {"z": 0, "seam_l1": 0.01, "seam_curvature": 0.02},
            {"z": 4, "seam_l1": 0.011, "seam_curvature": 0.021},
        ],
        metrics_aggregates={"seam_l1": 0.0105, "seam_curvature": 0.0205},
        repeats=3,
        metadata={"git_commit": "abc1234567890abcdef", "host": "gpu-node-1"},
        timestamp="2026-01-01T12:00:00Z",
    )


def _sample_candidate(*, baseline_id: str = "baseline-20260101-abc1234-subject-a") -> CandidateArtifact:
    return CandidateArtifact(
        candidate_id="candidate-20260102-def5678-subject-a",
        baseline_id=baseline_id,
        schema_version=SCHEMA_VERSION,
        metric_definition_version=METRIC_DEFINITION_VERSION,
        subject_id="subject-a",
        run_label="batched-v1",
        input_fingerprint="sha256:deadbeef",
        z_indices=[0, 4, 8],
        array_shape=[3, 512, 512],
        tile_shape=[128, 128],
        strategy_params={"working_size": 128, "force_batched_cuda": True},
        metrics_rows=[
            {"z": 0, "seam_l1": 0.012, "seam_curvature": 0.022},
        ],
        metrics_aggregates={"seam_l1": 0.012, "seam_curvature": 0.022},
        repeats=3,
        metadata={"git_commit": "def5678901234abcdef", "host": "gpu-node-1"},
        environment={"cuda_version": "12.4", "torch_version": "2.7.0"},
        timestamp="2026-01-02T12:00:00Z",
    )


def _sample_tolerance(*, baseline_id: str = "baseline-20260101-abc1234-subject-a") -> ToleranceSidecar:
    return ToleranceSidecar(
        baseline_id=baseline_id,
        schema_version=SCHEMA_VERSION,
        metric_definition_version=METRIC_DEFINITION_VERSION,
        calibration_policy="mean+3std",
        sigma=3.0,
        min_abs=1e-6,
        tolerances={
            "seam_l1": {"mean": 0.1, "std": 0.0003, "abs_tol": 0.001, "rel_tol": 0.05},
            "seam_curvature": {"mean": 0.05, "std": 0.0006, "abs_tol": 0.002, "rel_tol": 0.05},
        },
    )


class TestArtifactRoundTrip:
    @pytest.mark.parametrize(
        "factory, cls",
        [
            (_sample_baseline, BaselineBundle),
            (_sample_candidate, CandidateArtifact),
            (_sample_tolerance, ToleranceSidecar),
        ],
    )
    def test_write_read_roundtrip(self, tmp_path: Path, factory, cls) -> None:
        payload = factory()
        path = tmp_path / f"{cls.__name__}.json"
        write_artifact(path, payload)
        loaded = read_artifact(path, cls)
        assert loaded == payload


class TestGoldenJson:
    def test_baseline_json_is_deterministic(self, tmp_path: Path) -> None:
        payload = _sample_baseline()
        path = tmp_path / "baseline.json"
        write_artifact(path, payload)
        text = path.read_text(encoding="utf-8")
        parsed = json.loads(text)
        assert parsed == json.loads(
            json.dumps(json.loads(text), indent=2, sort_keys=True),
        )
        assert '"schema_version"' in text
        assert text.index('"array_shape"') < text.index('"baseline_id"')


class TestWriteArtifactOverwrite:
    def test_refuses_overwrite_by_default(self, tmp_path: Path) -> None:
        payload = _sample_baseline()
        path = tmp_path / "baseline.json"
        write_artifact(path, payload)
        with pytest.raises(FileExistsError, match="already exists"):
            write_artifact(path, payload)

    def test_allows_overwrite_when_requested(self, tmp_path: Path) -> None:
        payload = _sample_baseline()
        path = tmp_path / "baseline.json"
        write_artifact(path, payload)
        write_artifact(path, payload, overwrite=True)
        assert read_artifact(path, BaselineBundle) == payload


class TestMakeBaselineId:
    def test_format_and_slug(self) -> None:
        baseline_id = make_baseline_id(
            commit="abc1234567890abcdef",
            subject_id="My Subject/v1",
            timestamp="20260101T120000Z",
        )
        assert baseline_id.startswith("baseline-20260101T120000Z-abc1234-")
        assert "/" not in baseline_id
        assert " " not in baseline_id


class TestSlugifyLabel:
    def test_strips_path_traversal(self) -> None:
        sanitized = slugify_label("../../etc/passwd")
        assert "/" not in sanitized
        assert not sanitized.startswith("..")
        assert sanitized  # non-empty safe label


class TestAssertComparable:
    def test_raises_on_hard_key_mismatch(self) -> None:
        baseline = _sample_baseline()
        candidate = _sample_candidate()
        candidate = CandidateArtifact(
            candidate_id=candidate.candidate_id,
            baseline_id=candidate.baseline_id,
            schema_version=candidate.schema_version,
            metric_definition_version=candidate.metric_definition_version,
            subject_id="other-subject",
            run_label=candidate.run_label,
            input_fingerprint=candidate.input_fingerprint,
            z_indices=[0, 8],
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
        with pytest.raises(ValueError, match="subject_id") as exc_info:
            assert_comparable(candidate, baseline)
        message = str(exc_info.value)
        assert "z_indices" in message

    def test_raises_on_input_fingerprint_mismatch(self) -> None:
        baseline = _sample_baseline()
        candidate = _sample_candidate()
        candidate = CandidateArtifact(
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
        with pytest.raises(ValueError, match="input_fingerprint") as exc_info:
            assert_comparable(candidate, baseline)
        assert "input_fingerprint" in str(exc_info.value)

    def test_warns_on_commit_mismatch_without_raise(self) -> None:
        baseline = _sample_baseline()
        candidate = _sample_candidate()
        warnings = assert_comparable(candidate, baseline)
        assert warnings
        assert any("git commit" in w for w in warnings)


class TestToleranceSidecarSchema:
    def test_each_metric_carries_calibration_statistics(self) -> None:
        payload = _sample_tolerance()
        for metric in ("seam_l1", "seam_curvature"):
            entry = payload.tolerances[metric]
            assert "mean" in entry
            assert "std" in entry
            assert "abs_tol" in entry
            assert "rel_tol" in entry
            assert entry["mean"] != 0.0

    def test_tolerances_from_sidecar_preserves_mean_and_std(self) -> None:
        from scripts.benchmark_speedup import _tolerances_from_sidecar

        sidecar = _sample_tolerance()
        specs = _tolerances_from_sidecar(sidecar)
        assert specs["seam_l1"].mean == pytest.approx(0.1)
        assert specs["seam_l1"].std == pytest.approx(0.0003)
        assert specs["seam_curvature"].mean == pytest.approx(0.05)
        assert specs["seam_curvature"].std == pytest.approx(0.0006)

    def test_tolerances_from_sidecar_legacy_fallback(self) -> None:
        from scripts.benchmark_speedup import _tolerances_from_sidecar

        sidecar = ToleranceSidecar(
            baseline_id="baseline-legacy",
            schema_version=SCHEMA_VERSION,
            metric_definition_version=METRIC_DEFINITION_VERSION,
            calibration_policy="mean+3std",
            sigma=3.0,
            min_abs=1e-6,
            tolerances={
                "seam_l1": {"abs_tol": 0.001, "rel_tol": 0.05},
            },
        )
        specs = _tolerances_from_sidecar(sidecar)
        assert specs["seam_l1"].std == pytest.approx(0.001 / 3.0 - 1e-6 / 3.0, rel=1e-3)
        assert specs["seam_l1"].mean == pytest.approx(0.001 / 0.05 - 1e-6, rel=1e-3)


class TestWriteSummaryTable:
    _ROWS: ClassVar[list[dict[str, object]]] = [
        {"strategy": "baseline", "seam_l1": 0.01, "verdict": "pass"},
        {"strategy": "batched", "seam_l1": 0.012, "verdict": "pass"},
    ]

    def test_csv_format(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.csv"
        write_summary_table(path, self._ROWS, fmt="csv")
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == ["strategy", "seam_l1", "verdict"]
        assert len(rows) == 3

    def test_markdown_format(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.md"
        write_summary_table(path, self._ROWS, fmt="markdown")
        text = path.read_text(encoding="utf-8")
        assert "| strategy | seam_l1 | verdict |" in text
        assert "| --- | --- | --- |" in text
        assert "| batched |" in text

    def test_unknown_format_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.txt"
        with pytest.raises(ValueError, match="Unknown summary format"):
            write_summary_table(path, self._ROWS, fmt="html")
