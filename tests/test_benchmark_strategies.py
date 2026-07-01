"""Tests for benchmark strategy resolution and safe config overrides."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linum_basic.benchmark.strategies import (
    ALLOWED_OVERRIDE_KEYS,
    BUILTIN_STRATEGIES,
    load_overrides,
    resolve_strategy,
)


def _resolve(name: str, **kwargs):
    defaults = {
        "working_size": 128,
        "estimate_darkfield": True,
        "max_reweighting_iterations": 15,
    }
    defaults.update(kwargs)
    return resolve_strategy(name, **defaults)


class TestBuiltinStrategyResolution:
    def test_baseline_matches_production_shape(self) -> None:
        result = _resolve("baseline")
        assert result.basic_kwargs["working_size"] == 128
        assert result.basic_kwargs["estimate_darkfield"] is True
        assert result.basic_kwargs["backend"] == "torch"
        assert result.basic_kwargs["device"] == "cuda:0"
        assert result.basic_kwargs["warm_start_reweighting"] is False
        assert result.force_batched is False

    def test_sequential_matches_benchmark_script(self) -> None:
        result = _resolve("sequential")
        assert result.basic_kwargs["device"] == "cuda:0"
        assert result.basic_kwargs["warm_start_reweighting"] is False
        assert result.force_batched is False

    def test_multi_uses_cuda_and_warm_start(self) -> None:
        result = _resolve("multi")
        assert result.basic_kwargs["device"] == "cuda"
        assert result.basic_kwargs["warm_start_reweighting"] is True
        assert result.force_batched is False

    def test_batched_sets_force_batched(self) -> None:
        result = _resolve("batched")
        assert result.basic_kwargs["device"] == "cuda"
        assert result.basic_kwargs["warm_start_reweighting"] is True
        assert result.basic_kwargs["force_batched_cuda"] is True
        assert result.force_batched is True

    def test_batched_carries_chunk_size(self) -> None:
        result = _resolve("batched", batched_z_chunk_size=8)
        assert result.batched_z_chunk_size == 8
        assert result.basic_kwargs["batched_z_chunk_size"] == 8

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            _resolve("turbo")
        with pytest.raises(ValueError, match="baseline"):
            _resolve("turbo")

    def test_builtin_strategies_tuple(self) -> None:
        assert BUILTIN_STRATEGIES == ("baseline", "sequential", "multi", "batched")

    def test_synthetic_sets_release_gate_false(self) -> None:
        assert _resolve("baseline", is_synthetic=True).release_gate is False

    def test_real_sets_release_gate_true(self) -> None:
        assert _resolve("baseline", is_synthetic=False).release_gate is True


class TestBuiltinTupleUnchanged:
    """Regression: harness tuple stays 4 names; auto is resolver-only (Open Q1)."""

    def test_auto_not_in_builtin_strategies(self) -> None:
        assert BUILTIN_STRATEGIES == ("baseline", "sequential", "multi", "batched")
        assert "auto" not in BUILTIN_STRATEGIES


class TestOverrideParsing:
    def test_json_overrides_apply_to_resolve(self, tmp_path: Path) -> None:
        path = tmp_path / "overrides.json"
        path.write_text(json.dumps({"l_s": 0.5, "epsilon": 0.2}))
        overrides = load_overrides(path)
        result = _resolve("baseline", overrides=overrides)
        assert result.basic_kwargs["l_s"] == 0.5
        assert result.basic_kwargs["epsilon"] == 0.2

    def test_unknown_override_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"evil_key": 1}))
        with pytest.raises(ValueError, match="evil_key"):
            load_overrides(path)

    def test_non_mapping_top_level_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="mapping"):
            load_overrides(path)

    def test_yaml_overrides_via_safe_load(self, tmp_path: Path) -> None:
        yaml = pytest.importorskip("yaml")
        path = tmp_path / "overrides.yaml"
        path.write_text(yaml.dump({"working_size": 64}))
        overrides = load_overrides(path)
        assert overrides["working_size"] == 64
        result = _resolve("baseline", overrides=overrides)
        assert result.basic_kwargs["working_size"] == 64

    def test_allowed_keys_cover_fit_params(self) -> None:
        assert "working_size" in ALLOWED_OVERRIDE_KEYS
        assert "device" in ALLOWED_OVERRIDE_KEYS
        assert "backend" in ALLOWED_OVERRIDE_KEYS

    def test_reweighting_tolerance_override_passes_load_overrides(self, tmp_path: Path) -> None:
        path = tmp_path / "overrides.json"
        path.write_text(json.dumps({"reweighting_tolerance": 0.005}))
        overrides = load_overrides(path)
        assert overrides["reweighting_tolerance"] == 0.005
        result = _resolve("baseline", overrides=overrides)
        assert result.basic_kwargs["reweighting_tolerance"] == 0.005

    def test_convergence_check_every_override_passes_load_overrides(self, tmp_path: Path) -> None:
        path = tmp_path / "overrides.json"
        path.write_text(json.dumps({"convergence_check_every": 20}))
        overrides = load_overrides(path)
        assert overrides["convergence_check_every"] == 20
        result = _resolve("baseline", overrides=overrides)
        assert result.basic_kwargs["convergence_check_every"] == 20
