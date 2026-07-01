"""Tests for inner/outer iteration telemetry and alm_max_iter forwarding."""

from __future__ import annotations

import numpy as np
import pytest

from linum_basic.core import BaSiC


def _tiny_stack(n: int = 12, h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    vignette = 0.8 + 0.2 * np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2) / (w / 2)
    return np.stack([(vignette * (0.5 + 0.5 * rng.random((h, w)))).astype(np.float32) for _ in range(n)])


class TestAlmMaxIter:
    def test_alm_max_iter_forwarded_and_tracked(self) -> None:
        stack = _tiny_stack()
        model = BaSiC(stack, estimate_darkfield=True, backend="numpy", verbose=False)
        model.working_size = 32
        model.alm_max_iter = 3
        model.max_reweighting_iterations = 2
        model.prepare()
        model.run()
        assert model.last_alm_iterations <= 3
        assert 1 <= model.reweighting_iteration <= 2

    def test_convergence_telemetry_includes_alm_iterations(self) -> None:
        from linum_basic.fit import fit_mosaic
        from linum_basic.mosaic import MosaicGrid

        rng = np.random.default_rng(1)
        array = rng.random((2, 16, 16), dtype=np.float32) * 0.5 + 0.5
        mosaic = MosaicGrid(array=array, tile_shape=(8, 8), overlap_fraction=0.1)
        fit = fit_mosaic(
            mosaic,
            z_indices=[0, 1],
            field_mode="per-z",
            basic_kwargs={
                "backend": "numpy",
                "working_size": 16,
                "max_reweighting_iterations": 3,
                "alm_max_iter": 5,
            },
            n_workers=1,
        )
        assert fit.convergence_per_z is not None
        for entry in fit.convergence_per_z:
            assert "alm_iterations_last" in entry
            assert 1 <= entry["alm_iterations_last"] <= 5


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="torch not installed",
)
class TestAlmMaxIterGpu:
    def test_gpu_fit_records_alm_iterations(self) -> None:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from linum_basic.fit import fit_mosaic
        from linum_basic.mosaic import MosaicGrid

        rng = np.random.default_rng(2)
        array = rng.random((2, 16, 16), dtype=np.float32) * 0.5 + 0.5
        mosaic = MosaicGrid(array=array, tile_shape=(8, 8), overlap_fraction=0.1)
        fit = fit_mosaic(
            mosaic,
            z_indices=[0],
            field_mode="per-z",
            basic_kwargs={
                "backend": "torch",
                "device": "cuda:0",
                "working_size": 16,
                "max_reweighting_iterations": 3,
                "alm_max_iter": 20,
            },
            n_workers=1,
        )
        assert fit.convergence_per_z is not None
        assert fit.convergence_per_z[0]["alm_iterations_last"] <= 20
