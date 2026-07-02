"""Tests for supplementary BaSiCPy comparison metrics (informational sidecar only)."""

from __future__ import annotations

import builtins
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from linum_basic.benchmark.basicpy_compare import (
    compute_basicpy_comparison,
    run_basicpy_reference_fit,
    write_basicpy_sidecar,
)


def _field(size: int = 16, scale: float = 1.0) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, size * size, dtype=np.float64).reshape(size, size)
    return grid * scale


class TestComputeBasicpyComparison:
    def test_identical_flatfields_have_unit_pearson_r(self) -> None:
        field = _field()
        result = compute_basicpy_comparison(field, field.copy(), linum_ms=100.0, reference_ms=200.0)
        assert result["flatfield_pearson_r"] == pytest.approx(1.0, abs=1e-9)
        assert result["wall_time_ratio"] == pytest.approx(2.0)
        assert result["supplementary"] is True
        assert result["gate_applicable"] is False

    def test_anti_correlated_flatfields_have_negative_unit_pearson_r(self) -> None:
        field = _field()
        result = compute_basicpy_comparison(field, -field, linum_ms=50.0, reference_ms=25.0)
        assert result["flatfield_pearson_r"] == pytest.approx(-1.0, abs=1e-9)
        assert result["wall_time_ratio"] == pytest.approx(0.5)

    def test_partially_correlated_flatfield(self) -> None:
        field_a = _field()
        field_b = field_a + np.random.default_rng(0).normal(0.0, 0.05, field_a.shape)
        result = compute_basicpy_comparison(field_a, field_b, linum_ms=10.0, reference_ms=10.0)
        assert 0.5 < result["flatfield_pearson_r"] < 1.0
        assert result["gate_applicable"] is False


class TestRunBasicpyReferenceFit:
    def test_raises_import_error_when_basicpy_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(sys.modules, "basicpy", raising=False)
        real_import = builtins.__import__

        def _block_basicpy(name: str, *args: object, **kwargs: object) -> object:
            if name == "basicpy" or name.startswith("basicpy."):
                msg = "No module named 'basicpy'"
                raise ImportError(msg)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_basicpy)
        stack = np.ones((4, 8, 8), dtype=np.float64)
        with pytest.raises(ImportError, match=r"uv sync --extra basicpy"):
            run_basicpy_reference_fit(stack)

    def test_runs_with_monkeypatched_basicpy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stack = np.ones((4, 8, 8), dtype=np.float64)
        expected_flat = np.full((8, 8), 1.25, dtype=np.float64)

        class _FakeBaSiC:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.flatfield: np.ndarray | None = None

            def fit(self, images: np.ndarray) -> None:
                assert images.shape == stack.shape
                self.flatfield = expected_flat

        fake_basicpy = types.SimpleNamespace(BaSiC=_FakeBaSiC)
        monkeypatch.setitem(sys.modules, "basicpy", fake_basicpy)

        result = run_basicpy_reference_fit(stack)
        assert np.allclose(result["flatfield"], expected_flat)
        assert result["elapsed_ms"] >= 0.0


class TestWriteBasicpySidecar:
    def test_sidecar_writes_sorted_json(self, tmp_path: Path) -> None:
        field = _field()
        comparison = compute_basicpy_comparison(field, field, linum_ms=100.0, reference_ms=150.0)
        path = write_basicpy_sidecar(tmp_path, comparison)
        assert path == tmp_path / "basicpy-supplementary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["supplementary"] is True
        assert payload["gate_applicable"] is False
        assert payload["flatfield_pearson_r"] == pytest.approx(1.0, abs=1e-9)
        assert list(payload.keys()) == sorted(payload.keys())
