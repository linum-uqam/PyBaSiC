"""Tests for benchmark CUDA telemetry and reproducibility metadata."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


class _FakeCudaEvent:
    def __init__(self, *, enable_timing: bool = True) -> None:
        self.enable_timing = enable_timing
        self._elapsed_ms = 12.5

    def record(self) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def elapsed_time(self, other: _FakeCudaEvent) -> float:
        return self._elapsed_ms


class _FakeCuda:
    def __init__(self) -> None:
        self._peak_allocated = 1_000_000
        self._peak_reserved = 2_000_000
        self.synchronize_calls: list[str | None] = []
        self.reset_calls: list[str | None] = []

    def is_available(self) -> bool:
        return True

    def Event(self, *, enable_timing: bool = True) -> _FakeCudaEvent:
        return _FakeCudaEvent(enable_timing=enable_timing)

    def synchronize(self, device: str | None = None) -> None:
        self.synchronize_calls.append(device)

    def reset_peak_memory_stats(self, device: str | None = None) -> None:
        self.reset_calls.append(device)

    def max_memory_allocated(self, device: str | None = None) -> int:
        return self._peak_allocated

    def max_memory_reserved(self, device: str | None = None) -> int:
        return self._peak_reserved


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, fake_cuda: _FakeCuda | None = None) -> _FakeCuda:
    fake_cuda = fake_cuda or _FakeCuda()
    torch_mod = ModuleType("torch")
    torch_mod.cuda = fake_cuda  # type: ignore[attr-defined]
    torch_mod.version = SimpleNamespace(cuda="12.4")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    return fake_cuda


class TestTelemetryPhases:
    def test_run_with_phases_populates_all_phases_with_mock_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_cuda = _install_fake_torch(monkeypatch)
        from linum_basic.benchmark import telemetry

        def fn() -> int:
            return 42

        record = telemetry.run_with_phases(
            fn,
            device="cuda:0",
            backend="torch",
            repeats=3,
            warmup=1,
            chunk_size=8,
            compile_status="enabled",
            inductor_cache_path="/tmp/inductor",
        )

        assert record.warmup_ms is not None and record.warmup_ms > 0
        assert record.cold_cache_ms is not None and record.cold_cache_ms > 0
        assert record.warm_cache_ms is not None and record.warm_cache_ms > 0
        assert record.steady_state_ms is not None and record.steady_state_ms > 0
        assert record.max_memory_allocated_bytes == fake_cuda._peak_allocated
        assert record.max_memory_reserved_bytes == fake_cuda._peak_reserved
        assert record.chunk_size == 8
        assert record.compile_status == "enabled"
        assert record.inductor_cache_path == "/tmp/inductor"
        assert fake_cuda.synchronize_calls
        assert fake_cuda.reset_calls

    def test_cpu_fallback_without_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        from linum_basic.benchmark import telemetry

        record = telemetry.run_with_phases(
            lambda: 1,
            device="cpu",
            backend="numpy",
            repeats=2,
            warmup=1,
        )

        assert record.warmup_ms is not None
        assert record.cold_cache_ms is not None
        assert record.warm_cache_ms is not None
        assert record.steady_state_ms is not None
        assert record.max_memory_allocated_bytes is None
        assert record.max_memory_reserved_bytes is None

    def test_timed_call_uses_cuda_events_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_cuda = _install_fake_torch(monkeypatch)
        from linum_basic.benchmark import telemetry

        elapsed_ms, result = telemetry.timed_call(lambda: "ok", device="cuda:0", backend="torch")
        assert result == "ok"
        assert elapsed_ms == pytest.approx(12.5)
        assert fake_cuda.synchronize_calls


class TestRunMetadata:
    def test_collect_run_metadata_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        from linum_basic.benchmark import metadata

        real_import = builtins.__import__

        def _block_torch(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch unavailable")
            return real_import(name, *args, **kwargs)

        for key in list(sys.modules):
            if key == "torch" or key.startswith("torch."):
                monkeypatch.delitem(sys.modules, key, raising=False)
        monkeypatch.setattr(builtins, "__import__", _block_torch)

        meta = metadata.collect_run_metadata(
            strategy_params={"mode": "batched"},
            z_indices=[0, 4, 8],
            cache_mode="persistent",
        )

        assert meta.git_commit
        assert meta.host_node
        assert meta.platform
        assert meta.python_version
        assert meta.numpy_version
        assert meta.torch_version
        assert meta.cuda_available is False
        assert meta.cuda_devices == []
        assert meta.strategy_params == {"mode": "batched"}
        assert meta.z_indices == [0, 4, 8]
        assert meta.cache_mode == "persistent"
        datetime.fromisoformat(meta.timestamp)

    def test_collect_git_commit_unknown_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from linum_basic.benchmark import metadata

        def _raise(*args: Any, **kwargs: Any) -> None:
            raise subprocess.SubprocessError("git missing")

        monkeypatch.setattr(subprocess, "run", _raise)
        assert metadata.collect_git_commit() == "unknown"

    def test_collect_torch_cuda_info_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        from linum_basic.benchmark import metadata

        real_import = builtins.__import__

        def _block_torch(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch unavailable")
            return real_import(name, *args, **kwargs)

        for key in list(sys.modules):
            if key == "torch" or key.startswith("torch."):
                monkeypatch.delitem(sys.modules, key, raising=False)
        monkeypatch.setattr(builtins, "__import__", _block_torch)

        info = metadata.collect_torch_cuda_info()
        assert info["available"] is False
        assert info["devices"] == []


class _FakeMatmulBackend:
    def __init__(self, *, allow_tf32: bool = True) -> None:
        self.allow_tf32 = allow_tf32


class _FakeCudnnBackend:
    def __init__(self, *, allow_tf32: bool = True) -> None:
        self.allow_tf32 = allow_tf32


class _FakeCudaBackends:
    def __init__(self, *, matmul_allow_tf32: bool = True, cudnn_allow_tf32: bool = True) -> None:
        self.matmul = _FakeMatmulBackend(allow_tf32=matmul_allow_tf32)
        self.cudnn = _FakeCudnnBackend(allow_tf32=cudnn_allow_tf32)


def _install_fake_torch_precision(
    monkeypatch: pytest.MonkeyPatch,
    *,
    matmul_precision: str = "high",
    matmul_allow_tf32: bool = True,
    cudnn_allow_tf32: bool = True,
) -> None:
    torch_mod = ModuleType("torch")
    torch_mod.__version__ = "2.7.0"  # type: ignore[attr-defined]
    torch_mod.backends = SimpleNamespace(  # type: ignore[attr-defined]
        cuda=SimpleNamespace(
            matmul=_FakeMatmulBackend(allow_tf32=matmul_allow_tf32),
        ),
        cudnn=_FakeCudnnBackend(allow_tf32=cudnn_allow_tf32),
    )
    torch_mod.get_float32_matmul_precision = lambda: matmul_precision  # type: ignore[attr-defined]
    torch_mod.compile = lambda fn, **kwargs: fn  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_mod)


class TestCollectPrecisionMetadata:
    _REQUIRED_KEYS = (
        "float32_matmul_precision",
        "allow_tf32_matmul",
        "allow_tf32_cudnn",
        "compile_requested",
        "compile_fallback_reason",
    )

    def test_without_torch_returns_unavailable_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def _block_torch(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch unavailable")
            return real_import(name, *args, **kwargs)

        for key in list(sys.modules):
            if key == "torch" or key.startswith("torch."):
                monkeypatch.delitem(sys.modules, key, raising=False)
        monkeypatch.setattr(builtins, "__import__", _block_torch)

        from linum_basic.benchmark import telemetry

        result = telemetry.collect_precision_metadata()
        for key in self._REQUIRED_KEYS:
            assert key in result
        assert result["float32_matmul_precision"] is None
        assert result["allow_tf32_matmul"] is None
        assert result["allow_tf32_cudnn"] is None
        assert result["compile_requested"] is False
        assert result["compile_fallback_reason"] is None

    def test_with_fake_torch_captures_tf32_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_torch_precision(
            monkeypatch,
            matmul_precision="high",
            matmul_allow_tf32=True,
            cudnn_allow_tf32=False,
        )
        from linum_basic.benchmark import telemetry

        result = telemetry.collect_precision_metadata(compile_fallback_reason="inductor OOM")
        for key in self._REQUIRED_KEYS:
            assert key in result
        assert result["float32_matmul_precision"] == "high"
        assert result["allow_tf32_matmul"] is True
        assert result["allow_tf32_cudnn"] is False
        assert result["compile_requested"] is True
        assert result["compile_fallback_reason"] == "inductor OOM"


class TestCollectMultiGpuMemoryStats:
    def test_collect_multi_gpu_memory_stats_cpu_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate a CPU-only environment where ``import torch`` fails. We block
        # the import at ``builtins.__import__`` rather than merely deleting
        # ``sys.modules["torch"]``: once a real torch import has run earlier in
        # the same process (e.g. via test_alm_parity.py), re-importing torch
        # re-executes its ``__init__.py`` and re-registers the C-level "triton"
        # TORCH_LIBRARY namespace, which raises RuntimeError (not ImportError)
        # on the second registration. Blocking at the import boundary short-
        # circuits before torch's module body runs, so the function's existing
        # ``except ImportError: return {}`` guard fires reliably regardless of
        # what other test modules imported torch earlier in the suite.
        import builtins

        real_import = builtins.__import__

        def _block_torch(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch unavailable")
            return real_import(name, *args, **kwargs)

        for key in list(sys.modules):
            if key == "torch" or key.startswith("torch."):
                monkeypatch.delitem(sys.modules, key, raising=False)
        monkeypatch.setattr(builtins, "__import__", _block_torch)

        from linum_basic.benchmark import telemetry

        assert telemetry.collect_multi_gpu_memory_stats(["cuda:0", "cuda:1"]) == {}


class TestCollectConcurrencyMetadata:
    _D15_KEYS = (
        "strategy",
        "fork_model",
        "n_gpus",
        "gpu_map",
        "inductor_cache_path",
        "compile_status",
        "quality_verdict",
    )

    def test_collect_concurrency_metadata_fork_model_mapping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from linum_basic.benchmark import metadata

        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor-cache")

        multi = metadata.collect_concurrency_metadata(
            strategy="multi",
            fork_model=None,
            n_gpus=2,
            gpu_map={"cuda:0": "worker-0", "cuda:1": "worker-1"},
            inductor_cache_path=None,
            compile_status="enabled",
            quality_verdict={"passed": True},
        )
        batched = metadata.collect_concurrency_metadata(
            strategy="batched",
            fork_model=None,
            n_gpus=2,
            gpu_map={"cuda:0": "batched", "cuda:1": "batched"},
            inductor_cache_path="/custom/cache",
            compile_status="enabled",
            quality_verdict={"passed": True},
        )
        custom = metadata.collect_concurrency_metadata(
            strategy="experimental",
            fork_model="custom_fork_label",
            n_gpus=1,
            gpu_map={"cuda:0": "solo"},
            inductor_cache_path="/custom/cache",
        )

        for block in (multi, batched, custom):
            for key in self._D15_KEYS:
                assert key in block

        assert multi["fork_model"] == "maxForks_2_scalar_per_gpu"
        assert batched["fork_model"] == "maxForks_1_batched_multi_gpu"
        assert custom["fork_model"] == "custom_fork_label"
        assert multi["inductor_cache_path"] == "/tmp/inductor-cache"
        assert batched["inductor_cache_path"] == "/custom/cache"
