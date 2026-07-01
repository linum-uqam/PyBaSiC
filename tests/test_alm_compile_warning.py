"""Tests for audible torch.compile fallback in ALM step builders."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from linum_basic import _alm
from linum_basic._alm import (
    _ALM_STEP_BATCHED_CACHE,
    _ALM_STEP_CACHE,
    _build_alm_step,
    _build_alm_step_batched,
    inexact_alm_l1,
    inexact_alm_l1_batched,
)
from linum_basic.backend import Backend, get_xp


def _clear_alm_caches() -> None:
    _ALM_STEP_CACHE.clear()
    _ALM_STEP_BATCHED_CACHE.clear()
    if hasattr(_alm, "_CONVERGENCE_WARNED"):
        _alm._CONVERGENCE_WARNED.clear()
    if hasattr(_alm, "_COMPILE_WARNED"):
        _alm._COMPILE_WARNED.clear()


def _compile_raises(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("compile rejected")


@pytest.fixture(autouse=True)
def _reset_alm_step_cache() -> None:
    _clear_alm_caches()
    yield
    _clear_alm_caches()


class TestAlmCompileWarning:
    def test_build_alm_step_warns_on_compile_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("torch")
        import torch

        monkeypatch.setattr(torch, "compile", _compile_raises)
        xp = get_xp(Backend.TORCH, "cpu")

        with pytest.warns(UserWarning, match="torch.compile failed"):
            step_fn = _build_alm_step(xp, n=4, p=8, q=8, l_s=0.1)

        assert callable(step_fn)

    def test_build_alm_step_batched_warns_on_compile_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("torch")
        import torch

        monkeypatch.setattr(torch, "compile", _compile_raises)
        xp = get_xp(Backend.TORCH, "cpu")

        with pytest.warns(UserWarning, match="torch.compile failed"):
            step_fn = _build_alm_step_batched(xp, z=2, n=4, p=8, q=8)

        assert callable(step_fn)

    def test_build_alm_step_skips_compile_when_mode_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("torch")
        import torch

        monkeypatch.setenv("LINUM_BASIC_ALM_COMPILE_MODE", "off")
        compile_called = {"n": 0}
        real_compile = torch.compile

        def _counting_compile(fn: Any, **kwargs: Any) -> Any:
            compile_called["n"] += 1
            return real_compile(fn, **kwargs)

        monkeypatch.setattr(torch, "compile", _counting_compile)
        xp = get_xp(Backend.TORCH, "cpu")
        _build_alm_step(xp, n=4, p=8, q=8, l_s=0.5, estimate_darkfield=True, l_d=0.2)
        assert compile_called["n"] == 0

    def test_build_alm_step_compile_fallback_warns_once_per_scalar_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("torch")
        import torch

        monkeypatch.setattr(torch, "compile", _compile_raises)
        xp = get_xp(Backend.TORCH, "cpu")

        with pytest.warns(UserWarning, match="torch.compile failed"):
            _build_alm_step(xp, n=4, p=8, q=8, l_s=0.1)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _build_alm_step(xp, n=5, p=8, q=8, l_s=0.1)

        matching = [w for w in rec if issubclass(w.category, UserWarning) and "torch.compile failed" in str(w.message)]
        assert len(matching) == 0

    def test_build_alm_step_batched_compile_fallback_warns_once_per_batched_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("torch")
        import torch

        monkeypatch.setattr(torch, "compile", _compile_raises)
        xp = get_xp(Backend.TORCH, "cpu")

        with pytest.warns(UserWarning, match="torch.compile failed"):
            _build_alm_step_batched(xp, z=2, n=4, p=8, q=8)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _build_alm_step_batched(xp, z=3, n=4, p=8, q=8)

        matching = [w for w in rec if issubclass(w.category, UserWarning) and "torch.compile failed" in str(w.message)]
        assert len(matching) == 0


class TestAlmConvergenceWarning:
    def test_alm_max_iter_warns_once(self) -> None:
        rng = np.random.default_rng(42)
        n, p, q = 4, 8, 8
        imgs = np.clip(rng.normal(1.0, 0.1, (n, p, q)), 0.01, None).astype(np.float32)

        with pytest.warns(UserWarning, match="Maximum ALM iterations"):
            inexact_alm_l1(imgs, l_s=0.5, l_d=0.2, max_iter=1, tol=1e-12, estimate_darkfield=False)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            inexact_alm_l1(imgs, l_s=0.5, l_d=0.2, max_iter=1, tol=1e-12, estimate_darkfield=False)

        matching = [w for w in rec if issubclass(w.category, UserWarning) and "Maximum ALM iterations" in str(w.message)]
        assert len(matching) == 0

    def test_alm_batched_max_iter_warns_once_per_zbatch(self) -> None:
        pytest.importorskip("torch")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        xp = get_xp(Backend.TORCH, "cuda:0")
        rng = np.random.default_rng(42)
        n, p, q = 4, 8, 8
        batched_match = "Maximum batched ALM iterations"

        def run_batched(z: int) -> None:
            imgs = np.clip(rng.normal(1.0, 0.1, (z, n, p, q)), 0.01, None).astype(np.float32)
            inexact_alm_l1_batched(
                imgs,
                l_s=0.5,
                l_d=0.2,
                max_iter=1,
                tol=1e-12,
                estimate_darkfield=False,
                xp=xp,
            )

        with pytest.warns(UserWarning, match=batched_match):
            run_batched(z=2)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            run_batched(z=2)

        matching = [w for w in rec if issubclass(w.category, UserWarning) and batched_match in str(w.message)]
        assert len(matching) == 0

        with pytest.warns(UserWarning, match=batched_match):
            run_batched(z=3)
