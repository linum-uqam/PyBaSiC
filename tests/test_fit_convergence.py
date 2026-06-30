"""Tests for per-z convergence telemetry on the scalar fit path."""

from __future__ import annotations

import numpy as np

from linum_basic.fit import MosaicFit, fit_mosaic
from linum_basic.mosaic import MosaicGrid


def _tiny_mosaic(*, n_z: int = 2, n_rows: int = 2, n_cols: int = 2, th: int = 8, tw: int = 8) -> MosaicGrid:
    rng = np.random.default_rng(0)
    array = rng.random((n_z, n_rows * th, n_cols * tw), dtype=np.float32) * 0.5 + 0.5
    return MosaicGrid(array=array, tile_shape=(th, tw), overlap_fraction=0.2)


class TestFitConvergencePerZ:
    def test_scalar_path_records_convergence_per_z(self) -> None:
        mosaic = _tiny_mosaic(n_z=2)
        max_iters = 3
        fit = fit_mosaic(
            mosaic,
            z_indices=[0, 1],
            field_mode="per-z",
            basic_kwargs={
                "backend": "numpy",
                "max_reweighting_iterations": max_iters,
                "working_size": 32,
            },
            n_workers=1,
            verbose=False,
        )

        assert fit.convergence_per_z is not None
        assert len(fit.convergence_per_z) == 2
        for entry in fit.convergence_per_z:
            assert isinstance(entry["reweighting_iteration"], int)
            assert 1 <= entry["reweighting_iteration"] <= max_iters
            assert isinstance(entry["l_s"], float)
            assert isinstance(entry["l_d"], float)
            assert entry["l_s"] > 0
            assert entry["l_d"] > 0

    def test_global_field_mode_still_records_convergence_per_z(self) -> None:
        mosaic = _tiny_mosaic(n_z=2)
        max_iters = 3
        fit = fit_mosaic(
            mosaic,
            z_indices=[0, 1],
            field_mode="global",
            basic_kwargs={
                "backend": "numpy",
                "max_reweighting_iterations": max_iters,
                "working_size": 32,
            },
            n_workers=1,
            verbose=False,
        )

        assert fit.convergence_per_z is not None
        assert len(fit.convergence_per_z) == 2
        assert fit.flatfields.ndim == 2

    def test_mosaic_fit_default_convergence_is_none(self) -> None:
        fit = MosaicFit(
            flatfields=np.ones((2, 8, 8), dtype=np.float32),
            darkfields=np.zeros((2, 8, 8), dtype=np.float32),
            field_mode="per-z",
            z_indices=[0, 1],
        )
        assert fit.convergence_per_z is None
