"""Tests for persistent torch inductor cache configuration."""

from __future__ import annotations

import os
from pathlib import Path

from linum_basic._torch_cache import configure_torch_inductor_cache, enable_fx_graph_cache


def test_configure_torch_inductor_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_FX_GRAPH_CACHE", raising=False)
    cache = configure_torch_inductor_cache(tmp_path / "inductor")
    assert cache.is_dir()
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(cache.resolve())
    assert os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"


def test_enable_fx_graph_cache_no_torch(monkeypatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "torch._inductor.config", None)
    enable_fx_graph_cache()
