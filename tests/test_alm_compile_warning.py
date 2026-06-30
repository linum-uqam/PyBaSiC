"""Tests for audible torch.compile fallback in ALM step builders."""

from __future__ import annotations

from typing import Any

import pytest

from linum_basic._alm import _ALM_STEP_BATCHED_CACHE, _ALM_STEP_CACHE, _build_alm_step, _build_alm_step_batched
from linum_basic.backend import Backend, get_xp


def _clear_alm_caches() -> None:
    _ALM_STEP_CACHE.clear()
    _ALM_STEP_BATCHED_CACHE.clear()


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
