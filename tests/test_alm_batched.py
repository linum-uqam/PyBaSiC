"""Parity tests for batched vs sequential CUDA BaSiC fitting."""

from __future__ import annotations

import numpy as np
import pytest

from linum_basic.fit import fit_mosaic
from linum_basic.mosaic import MosaicGrid


def _make_mosaic(n_z: int = 4, n_rows: int = 2, n_cols: int = 2, th: int = 8, tw: int = 8, *, seed: int = 11) -> MosaicGrid:
    rng = np.random.default_rng(seed)
    array = rng.random((n_z, n_rows * th, n_cols * tw), dtype=np.float32)
    return MosaicGrid(array=array, tile_shape=(th, tw), overlap_fraction=0.2)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="torch not installed",
)
def test_fit_mosaic_batched_matches_sequential_cuda(monkeypatch) -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import linum_basic.fit as fit_mod

    mosaic = _make_mosaic()
    basic_kwargs = {
        "estimate_darkfield": True,
        "working_size": 8,
        "backend": "torch",
        "device": "cuda:0",
        "max_reweighting_iterations": 3,
        "warm_start_reweighting": True,
    }

    monkeypatch.setattr(fit_mod, "should_use_batched_cuda", lambda **kwargs: False)
    seq = fit_mosaic(mosaic, basic_kwargs=basic_kwargs, n_workers=1)

    monkeypatch.setattr(fit_mod, "should_use_batched_cuda", lambda **kwargs: True)
    bat = fit_mosaic(mosaic, basic_kwargs=dict(basic_kwargs, device="cuda"))

    assert np.allclose(seq.flatfields, bat.flatfields, atol=1e-3, rtol=1e-3)
    assert np.allclose(seq.darkfields, bat.darkfields, atol=1e-3, rtol=1e-3)
