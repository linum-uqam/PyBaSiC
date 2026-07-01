"""Decision-table and unit tests for the auto strategy resolver."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, Literal, cast

import numpy as np
import pytest

from linum_basic.benchmark.strategies import (
    BASELINE_REFERENCE_ID,
    BATCHED_CUDA_WS_GUARD,
    DEFAULT_PRODUCTION_WORKING_SIZE,
    EVIDENCE_POLICY_ID,
    OVERRIDE_SOURCES,
    REASON_AUTO_WS128_MULTI_GPU_FANOUT,
    REASON_AUTO_WS128_SINGLE_GPU_SEQUENTIAL,
    REASON_CUDA_UNAVAILABLE,
    REASON_GUARD_WS128_BLOCKED_BATCHED,
    REASON_OVERRIDE_ENV,
    REASON_OVERRIDE_USER_KWARGS,
    REASON_POLICY_PHASE2_WS128_MANDATORY,
    AutoStrategyResult,
    WorkloadContext,
    build_workload_context,
    estimate_strategy_vram_bytes,
    resolve_auto_strategy,
)
from linum_basic.fit import (
    _allow_batched_cuda_for_params,
    _resolve_batched_z_chunk_size,
    fit_mosaic,
)
from linum_basic.mosaic import MosaicGrid

# D-19 required keys for strategy_metadata (nested _strategy schema).
STRATEGY_METADATA_REQUIRED_KEYS = frozenset(
    {
        "name",
        "execution_path",
        "chunk_size",
        "device",
        "backend",
        "working_size",
        "n_z",
        "n_tiles",
        "n_gpus",
        "memory_estimate_bytes",
        "batched_cuda_guard",
        "attempted_execution_path",
        "reason_codes",
        "reason_summary",
        "override_source",
        "evidence_policy",
        "baseline_reference_id",
    }
)


class TestPolicyConstantsAndTypes:
    def test_policy_constants_match_phase2_uat(self) -> None:
        assert EVIDENCE_POLICY_ID == "phase2-uat-20260630"
        assert BASELINE_REFERENCE_ID == "baseline-20260630T163351-e7c47a4-sub-22"
        assert DEFAULT_PRODUCTION_WORKING_SIZE == 128
        assert BATCHED_CUDA_WS_GUARD == 128

    def test_override_sources_enum(self) -> None:
        assert OVERRIDE_SOURCES == (
            "auto_resolver",
            "user_kwargs",
            "env",
            "strategy_lock",
        )

    def test_workload_context_is_frozen(self) -> None:
        ctx = WorkloadContext(
            n_z=20,
            n_tiles=100,
            field_mode="per-z",
            working_size=128,
            n_gpus=2,
            cuda_available=True,
            memory_estimate_bytes=None,
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            ctx.n_z = 5  # type: ignore[misc]

    def test_estimate_strategy_vram_bytes(self) -> None:
        # 8 * 100 * 128 * 128 * 4 * 4
        expected = 8 * 100 * 128 * 128 * 4 * 4
        assert estimate_strategy_vram_bytes(8, 100, 128) == expected
        assert estimate_strategy_vram_bytes(8, 100, 128) > 0

    def test_auto_strategy_result_carries_metadata_fields(self) -> None:
        metadata = dict.fromkeys(STRATEGY_METADATA_REQUIRED_KEYS)
        metadata.update(
            {
                "name": "auto",
                "execution_path": "sequential_scalar",
                "reason_codes": ["AUTO_WS128_SINGLE_GPU_SEQUENTIAL"],
                "reason_summary": "Single GPU at ws=128.",
                "override_source": "auto_resolver",
                "evidence_policy": EVIDENCE_POLICY_ID,
                "baseline_reference_id": BASELINE_REFERENCE_ID,
            }
        )
        result = AutoStrategyResult(
            name="auto",
            basic_kwargs={"device": "cuda:0", "backend": "torch"},
            strategy_metadata=metadata,
            execution_path="sequential_scalar",
            override_source="auto_resolver",
            chunk_size=None,
        )
        assert result.execution_path == "sequential_scalar"
        assert result.override_source == "auto_resolver"
        assert set(result.strategy_metadata.keys()) >= STRATEGY_METADATA_REQUIRED_KEYS


def _ctx(
    *,
    n_z: int = 20,
    n_tiles: int = 100,
    working_size: int = 128,
    n_gpus: int = 1,
    cuda_available: bool = True,
) -> WorkloadContext:
    mem = estimate_strategy_vram_bytes(min(n_z, 8), n_tiles, working_size) if cuda_available else None
    return WorkloadContext(
        n_z=n_z,
        n_tiles=n_tiles,
        field_mode="per-z",
        working_size=working_size,
        n_gpus=n_gpus,
        cuda_available=cuda_available,
        memory_estimate_bytes=mem,
    )


def _assert_metadata_complete(result: AutoStrategyResult) -> None:
    meta = result.strategy_metadata
    assert set(meta.keys()) >= STRATEGY_METADATA_REQUIRED_KEYS
    assert meta["name"] == "auto"
    assert meta["reason_summary"]
    assert meta["reason_codes"]
    assert meta["override_source"] in OVERRIDE_SOURCES
    assert meta["evidence_policy"] == EVIDENCE_POLICY_ID
    assert meta["baseline_reference_id"] == BASELINE_REFERENCE_ID


class TestAutoDecisionTable:
    @pytest.mark.parametrize("n_z", [1, 5, 20])
    def test_ws128_single_gpu_sequential(self, n_z: int) -> None:
        result = resolve_auto_strategy(_ctx(n_z=n_z, n_gpus=1, working_size=128))
        assert result.execution_path == "sequential_scalar"
        assert result.basic_kwargs["device"] == "cuda:0"
        assert result.basic_kwargs["backend"] == "torch"
        assert "force_batched_cuda" not in result.basic_kwargs
        assert result.strategy_metadata["batched_cuda_guard"] is None
        assert "batched" not in result.execution_path
        assert REASON_AUTO_WS128_SINGLE_GPU_SEQUENTIAL in result.strategy_metadata["reason_codes"]
        assert REASON_POLICY_PHASE2_WS128_MANDATORY in result.strategy_metadata["reason_codes"]
        assert result.override_source == "auto_resolver"
        _assert_metadata_complete(result)

    @pytest.mark.parametrize("n_z", [1, 5, 20])
    def test_ws128_multi_gpu_fanout(self, n_z: int) -> None:
        result = resolve_auto_strategy(_ctx(n_z=n_z, n_gpus=2, working_size=128))
        assert result.execution_path == "multi_gpu_fanout"
        assert result.basic_kwargs["device"] == "cuda"
        assert "force_batched_cuda" not in result.basic_kwargs
        assert result.strategy_metadata["batched_cuda_guard"] is None
        assert REASON_AUTO_WS128_MULTI_GPU_FANOUT in result.strategy_metadata["reason_codes"]
        _assert_metadata_complete(result)

    @pytest.mark.parametrize("n_gpus", [1, 2])
    @pytest.mark.parametrize("n_z", [1, 5, 20])
    def test_ws128_never_batched(self, n_gpus: int, n_z: int) -> None:
        result = resolve_auto_strategy(
            _ctx(n_z=n_z, n_gpus=n_gpus, working_size=128),
        )
        assert "batched" not in result.execution_path
        assert result.basic_kwargs.get("force_batched_cuda") is not True
        assert result.strategy_metadata["batched_cuda_guard"] is None

    def test_ws64_single_z_sequential(self) -> None:
        result = resolve_auto_strategy(_ctx(n_z=1, n_gpus=1, working_size=64))
        assert result.execution_path == "sequential_scalar"
        assert "force_batched_cuda" not in result.basic_kwargs

    def test_ws64_multi_z_single_gpu_batched(self) -> None:
        result = resolve_auto_strategy(_ctx(n_z=5, n_gpus=1, working_size=64))
        assert result.execution_path == "batched_cuda"
        assert result.basic_kwargs["force_batched_cuda"] is True
        assert result.basic_kwargs["device"] == "cuda:0"

    def test_ws64_multi_z_multi_gpu_batched(self) -> None:
        result = resolve_auto_strategy(_ctx(n_z=5, n_gpus=2, working_size=64))
        assert result.execution_path == "batched_cuda_multi_gpu"
        assert result.basic_kwargs["force_batched_cuda"] is True
        assert result.basic_kwargs["device"] == "cuda"

    def test_cuda_unavailable_fallback(self) -> None:
        result = resolve_auto_strategy(
            _ctx(n_z=20, n_gpus=0, cuda_available=False),
        )
        assert result.execution_path == "sequential_scalar"
        assert result.basic_kwargs["backend"] == "numpy"
        assert REASON_CUDA_UNAVAILABLE in result.strategy_metadata["reason_codes"]
        _assert_metadata_complete(result)


class TestOverridePrecedence:
    def test_no_overrides_auto_resolver(self) -> None:
        result = resolve_auto_strategy(_ctx())
        assert result.override_source == "auto_resolver"

    def test_user_kwargs_device_override(self) -> None:
        result = resolve_auto_strategy(
            _ctx(n_gpus=1, working_size=128),
            user_kwargs={"device": "cuda:1"},
        )
        assert result.override_source == "user_kwargs"
        assert result.basic_kwargs["device"] == "cuda:1"
        assert REASON_OVERRIDE_USER_KWARGS in result.strategy_metadata["reason_codes"]

    def test_env_force_batched_cuda(self) -> None:
        result = resolve_auto_strategy(
            _ctx(n_z=20, n_gpus=1, working_size=128),
            env={"LINUM_BASIC_FORCE_BATCHED_CUDA": "1"},
        )
        assert result.override_source == "env"
        assert result.basic_kwargs.get("force_batched_cuda") is True
        assert REASON_OVERRIDE_ENV in result.strategy_metadata["reason_codes"]

    def test_user_kwargs_beats_env(self) -> None:
        result = resolve_auto_strategy(
            _ctx(working_size=128),
            user_kwargs={"force_batched_cuda": False},
            env={"LINUM_BASIC_FORCE_BATCHED_CUDA": "1"},
        )
        assert result.override_source == "user_kwargs"
        assert result.basic_kwargs.get("force_batched_cuda") is False


@pytest.fixture
def mock_cuda_gpus(monkeypatch: pytest.MonkeyPatch):
    """Monkeypatch ``list_cuda_devices`` in strategies to return *n* GPUs."""

    def _apply(n: int) -> None:
        monkeypatch.setattr(
            "linum_basic.benchmark.strategies.list_cuda_devices",
            lambda device: [f"cuda:{i}" for i in range(n)],
        )

    return _apply


class TestBuildWorkloadContext:
    def test_derives_n_gpus_from_list_cuda_devices(
        self,
        mock_cuda_gpus,
    ) -> None:
        mock_cuda_gpus(2)
        ctx = build_workload_context(
            n_z=10,
            n_tiles=50,
            field_mode="per-z",
            working_size=128,
        )
        assert ctx.n_gpus == 2
        assert ctx.cuda_available is True
        assert ctx.memory_estimate_bytes is not None

    def test_mock_drives_multi_gpu_resolution(self, mock_cuda_gpus) -> None:
        mock_cuda_gpus(2)
        ctx = build_workload_context(
            n_z=5,
            n_tiles=100,
            field_mode="per-z",
            working_size=128,
        )
        result = resolve_auto_strategy(ctx)
        assert result.execution_path == "multi_gpu_fanout"
        assert result.basic_kwargs["device"] == "cuda"


EXECUTION_PATHS = frozenset(
    {
        "sequential_scalar",
        "multi_gpu_fanout",
        "batched_cuda",
        "batched_cuda_multi_gpu",
    }
)


def _make_mosaic(n_z: int, *, n_rows: int = 2, n_cols: int = 2, th: int = 8, tw: int = 8, seed: int = 0) -> MosaicGrid:
    rng = np.random.default_rng(seed)
    array = rng.random((n_z, n_rows * th, n_cols * tw), dtype=np.float64).astype(np.float32)
    return MosaicGrid(array=array, tile_shape=(th, tw), overlap_fraction=0.2)


def _mock_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "linum_basic.benchmark.strategies.list_cuda_devices",
        lambda device: [],
    )
    monkeypatch.setattr(
        "linum_basic.fit.list_cuda_devices",
        lambda device: [],
    )


def _mock_cuda_gpus_fit(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    devices = [f"cuda:{i}" for i in range(n)]

    def _list(device: str | None = None) -> list[str]:
        return list(devices)

    monkeypatch.setattr("linum_basic.benchmark.strategies.list_cuda_devices", _list)
    monkeypatch.setattr("linum_basic.fit.list_cuda_devices", _list)


def _mock_fast_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_fit(
        tiles: np.ndarray,
        params: dict,
        n_extra_rows: int,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        th = tiles.shape[1] - n_extra_rows if n_extra_rows else tiles.shape[1]
        tw = tiles.shape[2]
        return (
            np.ones((th, tw), dtype=np.float32),
            np.zeros((th, tw), dtype=np.float32),
            {"reweighting_iteration": 1, "l_s": 1.0, "l_d": 1.0},
        )

    monkeypatch.setattr("linum_basic.fit._fit_one_z", _fake_fit)

    def _fake_cuda_map(
        tiles: np.ndarray,
        device: str,
        *,
        params: dict,
        n_extra_rows: int,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        return _fake_fit(tiles, params, n_extra_rows)

    monkeypatch.setattr("linum_basic.fit._fit_one_z_cuda_map", _fake_cuda_map)

    def _inline_parallel_map(
        fn,
        items,
        n_workers,
        desc: str = "",
        verbose: bool = False,
    ):
        return [fn(item) for item in items]

    def _inline_cuda_devices_map(
        fn,
        items,
        cuda_devices,
        desc: str = "",
        verbose: bool = False,
    ):
        return [fn(item, cuda_devices[i % len(cuda_devices)]) for i, item in enumerate(items)]

    monkeypatch.setattr("linum_basic.fit.parallel_map", _inline_parallel_map)
    monkeypatch.setattr("linum_basic.fit.parallel_map_cuda_devices", _inline_cuda_devices_map)

    def _fake_batched(
        tile_stacks: list[np.ndarray],
        *,
        params: dict,
        n_extra_rows: int,
        cuda_devices: list[str],
        verbose: bool,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        results: list[tuple[np.ndarray, np.ndarray]] = []
        for stack in tile_stacks:
            th = stack.shape[1] - n_extra_rows if n_extra_rows else stack.shape[1]
            tw = stack.shape[2]
            results.append(
                (
                    np.ones((th, tw), dtype=np.float32),
                    np.zeros((th, tw), dtype=np.float32),
                )
            )
        return results

    monkeypatch.setattr("linum_basic.fit._fit_mosaic_batched_cuda", _fake_batched)


class TestFitMosaicStrategyParams:
    def test_auto_default_records_strategy_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_no_cuda(monkeypatch)
        _mock_fast_fit(monkeypatch)
        mosaic = _make_mosaic(2)
        fit = fit_mosaic(mosaic, basic_kwargs={"backend": "numpy"})
        strat = fit.params["_strategy"]
        assert set(strat.keys()) >= STRATEGY_METADATA_REQUIRED_KEYS
        assert strat["name"] == "auto"
        assert strat["override_source"] == "auto_resolver"
        assert REASON_CUDA_UNAVAILABLE in strat["reason_codes"]
        assert strat["working_size"] == 128
        assert fit.params["backend"] == "numpy"

    def test_strategy_lock_sequential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_cuda_gpus_fit(monkeypatch, 1)
        _mock_fast_fit(monkeypatch)
        mosaic = _make_mosaic(2)
        fit = fit_mosaic(
            mosaic,
            strategy="sequential",
            basic_kwargs={"estimate_darkfield": False, "backend": "numpy"},
        )
        strat = fit.params["_strategy"]
        assert strat["override_source"] == "strategy_lock"
        assert strat["name"] == "sequential"
        assert strat["device"] == "cuda:0"

    def test_strategy_lock_multi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_cuda_gpus_fit(monkeypatch, 2)
        _mock_fast_fit(monkeypatch)
        mosaic = _make_mosaic(2)
        fit = fit_mosaic(
            mosaic,
            strategy="multi",
            basic_kwargs={"estimate_darkfield": False, "backend": "numpy"},
        )
        assert fit.params["_strategy"]["device"] == "cuda"
        assert fit.params["_strategy"]["override_source"] == "strategy_lock"

    def test_strategy_lock_batched_records_force(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_cuda_gpus_fit(monkeypatch, 1)
        _mock_fast_fit(monkeypatch)
        monkeypatch.setattr("linum_basic.fit.should_use_batched_cuda", lambda **kwargs: True)
        mosaic = _make_mosaic(3)
        fit = fit_mosaic(
            mosaic,
            strategy="batched",
            basic_kwargs={
                "working_size": 128,
                "backend": "torch",
                "device": "cuda",
                "estimate_darkfield": False,
            },
        )
        strat = fit.params["_strategy"]
        assert strat["override_source"] == "strategy_lock"
        assert fit.params.get("force_batched_cuda") is True
        assert strat["batched_cuda_guard"] is None
        assert "batched" in strat["execution_path"]

    def test_user_kwargs_override_auto(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_cuda_gpus_fit(monkeypatch, 2)
        _mock_fast_fit(monkeypatch)
        mosaic = _make_mosaic(2)
        fit = fit_mosaic(
            mosaic,
            strategy="auto",
            basic_kwargs={
                "device": "cuda:0",
                "backend": "numpy",
                "estimate_darkfield": False,
            },
        )
        strat = fit.params["_strategy"]
        assert strat["override_source"] == "user_kwargs"
        assert fit.params["device"] == "cuda:0"
        assert REASON_OVERRIDE_USER_KWARGS in strat["reason_codes"]

    def test_basic_kwargs_flat_not_nested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_no_cuda(monkeypatch)
        _mock_fast_fit(monkeypatch)
        mosaic = _make_mosaic(1)
        fit = fit_mosaic(
            mosaic,
            basic_kwargs={"working_size": 64, "backend": "numpy", "estimate_darkfield": False},
        )
        assert fit.params["working_size"] == 64
        assert fit.params["_strategy"]["working_size"] == 64
        assert "device" not in fit.params.get("_strategy", {}).get("basic_kwargs", {})

    def test_invalid_strategy_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_no_cuda(monkeypatch)
        mosaic = _make_mosaic(1)
        with pytest.raises(ValueError, match="turbo"):
            fit_mosaic(mosaic, strategy=cast(Any, "turbo"))


class TestGuardDowngradeMetadata:
    def test_batched_force_bypasses_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_cuda_gpus_fit(monkeypatch, 1)
        _mock_fast_fit(monkeypatch)
        monkeypatch.setattr("linum_basic.fit.should_use_batched_cuda", lambda **kwargs: True)
        mosaic = _make_mosaic(3)
        fit = fit_mosaic(
            mosaic,
            strategy="batched",
            basic_kwargs={
                "working_size": 128,
                "backend": "torch",
                "device": "cuda",
                "estimate_darkfield": False,
            },
        )
        strat = fit.params["_strategy"]
        assert strat["batched_cuda_guard"] is None
        assert "batched" in strat["execution_path"]
        assert fit.params.get("force_batched_cuda") is True

    def test_guard_downgrade_recorded_not_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_cuda_gpus_fit(monkeypatch, 1)
        _mock_fast_fit(monkeypatch)
        monkeypatch.setattr("linum_basic.fit.should_use_batched_cuda", lambda **kwargs: True)
        monkeypatch.setattr("linum_basic.fit._allow_batched_cuda_for_params", lambda params: False)
        mosaic = _make_mosaic(5)
        fit = fit_mosaic(
            mosaic,
            strategy="auto",
            basic_kwargs={
                "working_size": 64,
                "backend": "torch",
                "device": "cuda:0",
                "estimate_darkfield": False,
            },
        )
        strat = fit.params["_strategy"]
        assert strat["batched_cuda_guard"] == "ws>=128_blocked"
        assert strat["attempted_execution_path"] == "batched_cuda"
        assert strat["execution_path"] == "sequential_scalar"
        assert REASON_GUARD_WS128_BLOCKED_BATCHED in strat["reason_codes"]

    def test_guard_downgrade_multi_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_cuda_gpus_fit(monkeypatch, 2)
        _mock_fast_fit(monkeypatch)
        monkeypatch.setattr("linum_basic.fit.should_use_batched_cuda", lambda **kwargs: True)
        monkeypatch.setattr("linum_basic.fit._allow_batched_cuda_for_params", lambda params: False)
        mosaic = _make_mosaic(5)
        fit = fit_mosaic(
            mosaic,
            strategy="auto",
            basic_kwargs={
                "working_size": 64,
                "backend": "torch",
                "device": "cuda",
                "estimate_darkfield": False,
            },
        )
        strat = fit.params["_strategy"]
        assert strat["attempted_execution_path"] == "batched_cuda_multi_gpu"
        assert strat["execution_path"] == "multi_gpu_fanout"


class TestChunkSizeFinalisation:
    def test_chunk_size_from_final_merged_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_cuda_gpus_fit(monkeypatch, 2)
        _mock_fast_fit(monkeypatch)
        monkeypatch.setattr("linum_basic.fit.should_use_batched_cuda", lambda **kwargs: True)
        mosaic = _make_mosaic(20)
        basic_kwargs = {
            "working_size": 64,
            "backend": "torch",
            "device": "cuda",
            "estimate_darkfield": False,
        }
        fit = fit_mosaic(mosaic, strategy="batched", basic_kwargs=basic_kwargs)
        expected = _resolve_batched_z_chunk_size(
            fit.params,
            mosaic.n_z,
            2,
        )
        assert fit.params["_strategy"]["chunk_size"] == expected

    def test_chunk_size_none_for_non_batched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_cuda_gpus_fit(monkeypatch, 1)
        _mock_fast_fit(monkeypatch)
        mosaic = _make_mosaic(3)
        fit = fit_mosaic(
            mosaic,
            strategy="auto",
            basic_kwargs={"working_size": 128, "backend": "numpy", "estimate_darkfield": False},
        )
        assert fit.params["_strategy"]["chunk_size"] is None

    def test_allow_batched_threshold_unchanged(self) -> None:
        assert _allow_batched_cuda_for_params({"working_size": 128}) is False
        assert _allow_batched_cuda_for_params({"working_size": 64}) is True
        assert _allow_batched_cuda_for_params({"working_size": 128, "force_batched_cuda": True}) is True


def _expected_auto_execution_path(*, working_size: int, n_gpus: int, n_z: int) -> str:
    if working_size >= BATCHED_CUDA_WS_GUARD:
        return "multi_gpu_fanout" if n_gpus >= 2 else "sequential_scalar"
    if n_z <= 1:
        return "sequential_scalar"
    return "batched_cuda_multi_gpu" if n_gpus >= 2 else "batched_cuda"


def _expected_named_execution_path(*, strategy: str, n_gpus: int) -> str:
    if strategy == "sequential":
        return "sequential_scalar"
    if strategy == "multi":
        return "multi_gpu_fanout"
    return "batched_cuda_multi_gpu" if n_gpus >= 2 else "batched_cuda"


@pytest.mark.parametrize("working_size", [64, 128])
@pytest.mark.parametrize("n_gpus", [1, 2])
@pytest.mark.parametrize("n_z", [1, 5, 20])
@pytest.mark.parametrize("strategy", ["auto", "sequential", "multi", "batched"])
def test_strategy_decision_table(
    working_size: int,
    n_gpus: int,
    n_z: int,
    strategy: Literal["auto", "sequential", "multi", "batched"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_cuda_gpus_fit(monkeypatch, n_gpus)
    _mock_fast_fit(monkeypatch)

    def _should_use_batched(**kwargs: object) -> bool:
        backend = kwargs.get("backend")
        device = kwargs.get("device")
        field_mode = kwargs.get("field_mode")
        nz = kwargs.get("n_z")
        if field_mode != "per-z" or not isinstance(nz, int) or nz <= 1:
            return False
        if backend not in {"torch", "auto"}:
            return False
        dev = str(device or "cuda").lower()
        return dev.startswith("cuda")

    monkeypatch.setattr("linum_basic.fit.should_use_batched_cuda", _should_use_batched)

    mosaic = _make_mosaic(n_z)
    fit = fit_mosaic(
        mosaic,
        field_mode="per-z",
        strategy=strategy,
        basic_kwargs={"working_size": working_size, "backend": "torch"},
    )
    strat = fit.params["_strategy"]

    assert strat["name"] == strategy
    assert strat["execution_path"] in EXECUTION_PATHS

    if strategy == "auto":
        assert strat["override_source"] == "auto_resolver"
        expected = _expected_auto_execution_path(
            working_size=working_size,
            n_gpus=n_gpus,
            n_z=n_z,
        )
        if working_size >= BATCHED_CUDA_WS_GUARD:
            assert strat["batched_cuda_guard"] is None
            assert "batched" not in strat["execution_path"]
            assert strat["execution_path"] == expected
        elif n_z > 1:
            assert "batched" in strat["execution_path"]
    else:
        assert strat["override_source"] == "strategy_lock"
        if strategy == "sequential":
            assert strat["device"] == "cuda:0"
        elif strategy == "multi":
            assert strat["device"] == "cuda"
        elif strategy == "batched":
            assert fit.params.get("force_batched_cuda") is True
            if working_size >= BATCHED_CUDA_WS_GUARD and n_z > 1:
                assert strat["batched_cuda_guard"] is None
                assert "batched" in strat["execution_path"]
