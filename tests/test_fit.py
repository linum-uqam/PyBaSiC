"""Integration tests for the mosaic fit pipeline.

Tests fit_mosaic + apply_fit round-trip on a synthetic (Z, H, W) mosaic.
Verifies that applying the estimated fields reduces the seam-consistency
L1 metric relative to the raw (uncorrected) mosaic.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

try:
    from sbh_simulator.simulator import generate_gaussian_vignette

    _SBH_SIMULATOR_AVAILABLE = True
except ImportError:
    _SBH_SIMULATOR_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _SBH_SIMULATOR_AVAILABLE,
    reason="sbh-simulator not installed. Install with: uv pip install sbh-simulator",
)


def _synthetic_mosaic(n_rows=4, n_cols=5, tile_h=10, tile_w=10, n_z=3, seed=42):
    """Build a synthetic mosaic with a known vignette and shared seam content.

    Adjacent tiles have identical pixel values in their overlap regions
    (physical overlap), so after ideal shading correction the seam-L1
    should be near zero.

    Returns
    -------
    raw : numpy.ndarray, shape (n_z, H, W)
    flatfield : numpy.ndarray, shape (tile_h, tile_w)
        Ground-truth flat-field (Gaussian vignette from sbh-simulator).
    """
    rng = np.random.default_rng(seed)
    flatfield = generate_gaussian_vignette(width=tile_w, height=tile_h, sigma=0.7, rng=random.Random(seed)).astype(np.float32)

    overlap_x = round(0.2 * tile_w)
    overlap_y = round(0.2 * tile_h)
    n_tiles = n_rows * n_cols

    raw = np.zeros((n_z, n_rows * tile_h, n_cols * tile_w), dtype=np.float32)
    for z in range(n_z):
        ff_z = flatfield * (1.0 + 0.05 * z)
        # Generate tiles with correlated seam content at every z-level
        tiles = rng.random((n_tiles, tile_h, tile_w)).astype(np.float32)
        # Horizontal seams: right edge of left tile = left edge of right tile
        for r in range(n_rows):
            for c in range(n_cols - 1):
                shared = rng.random((tile_h, overlap_x)).astype(np.float32)
                tiles[r * n_cols + c, :, -overlap_x:] = shared
                tiles[r * n_cols + c + 1, :, :overlap_x] = shared
        # Vertical seams: bottom edge of top tile = top edge of bottom tile
        for r in range(n_rows - 1):
            for c in range(n_cols):
                shared = rng.random((overlap_y, tile_w)).astype(np.float32)
                tiles[r * n_cols + c, -overlap_y:, :] = shared
                tiles[(r + 1) * n_cols + c, :overlap_y, :] = shared
        # Place tiles into the mosaic array with vignette applied
        for r in range(n_rows):
            for c in range(n_cols):
                idx = r * n_cols + c
                raw[z, r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tiles[idx] * ff_z
    return raw, flatfield


@pytest.fixture
def synthetic_mosaic_grid():
    from linum_basic.mosaic import MosaicGrid

    raw, ff = _synthetic_mosaic()
    mosaic = MosaicGrid(raw, tile_shape=(10, 10), overlap_fraction=0.2)
    return mosaic, ff


class TestFitMosaic:
    def test_returns_mosaic_fit(self, synthetic_mosaic_grid):
        from linum_basic.fit import fit_mosaic

        mosaic, _ = synthetic_mosaic_grid
        fit = fit_mosaic(mosaic, basic_kwargs={"estimate_darkfield": False})
        assert fit.flatfields.shape == (mosaic.n_z, 10, 10)
        assert fit.darkfields.shape == (mosaic.n_z, 10, 10)

    def test_global_field_mode(self, synthetic_mosaic_grid):
        from linum_basic.fit import fit_mosaic

        mosaic, _ = synthetic_mosaic_grid
        fit = fit_mosaic(mosaic, field_mode="global", basic_kwargs={"estimate_darkfield": False})
        assert fit.flatfields.shape == (10, 10)
        assert fit.darkfields.shape == (10, 10)
        assert fit.field_mode == "global"

    def test_z_indices_subset(self, synthetic_mosaic_grid):
        from linum_basic.fit import fit_mosaic

        mosaic, _ = synthetic_mosaic_grid
        fit = fit_mosaic(mosaic, z_indices=[0, 2], basic_kwargs={"estimate_darkfield": False})
        assert fit.z_indices == [0, 2]
        assert fit.flatfields.shape == (2, 10, 10)


class TestApplyFit:
    def test_output_shape(self, synthetic_mosaic_grid):
        from linum_basic.fit import apply_fit, fit_mosaic

        mosaic, _ = synthetic_mosaic_grid
        fit = fit_mosaic(mosaic, basic_kwargs={"estimate_darkfield": False})
        corrected = apply_fit(mosaic, fit)
        assert corrected.shape == mosaic.array.shape

    def test_output_dtype(self, synthetic_mosaic_grid):
        from linum_basic.fit import apply_fit, fit_mosaic

        mosaic, _ = synthetic_mosaic_grid
        fit = fit_mosaic(mosaic, basic_kwargs={"estimate_darkfield": False})
        corrected = apply_fit(mosaic, fit)
        assert corrected.dtype == np.float32

    def test_seam_l1_improves(self, synthetic_mosaic_grid):
        """apply_fit with oracle fields must reduce seam-L1 vs raw."""
        from linum_basic.fit import MosaicFit, apply_fit
        from linum_basic.metrics import seam_l1
        from linum_basic.mosaic import MosaicGrid

        mosaic, ff_oracle = synthetic_mosaic_grid
        seam_pairs = mosaic.seam_pairs()
        th, tw = mosaic.tile_shape

        # Raw seam L1 (average over z)
        raw_l1 = float(np.mean([seam_l1(mosaic.iter_tiles(z), seam_pairs) for z in range(mosaic.n_z)]))

        # Build oracle MosaicFit using the ground-truth flatfield
        # (uniform scale-factor per z doesn't affect seam consistency)
        oracle_fit = MosaicFit(
            flatfields=np.stack([ff_oracle] * mosaic.n_z),
            darkfields=np.zeros((mosaic.n_z, th, tw), dtype=np.float32),
            field_mode="per-z",
            z_indices=list(range(mosaic.n_z)),
        )

        corrected_arr = apply_fit(mosaic, oracle_fit)
        corrected_mosaic = MosaicGrid(
            corrected_arr,
            tile_shape=mosaic.tile_shape,
            overlap_fraction=mosaic.overlap_fraction,
        )

        corrected_l1 = float(
            np.mean([seam_l1(corrected_mosaic.iter_tiles(z), seam_pairs) for z in range(corrected_mosaic.n_z)])
        )

        assert corrected_l1 < raw_l1, f"Oracle correction did not improve seam L1: {corrected_l1:.4f} >= {raw_l1:.4f}"


class TestSaveCorrected:
    def test_save_and_reload(self, tmp_path, synthetic_mosaic_grid):
        """save_corrected produces a valid OME-Zarr that matches apply_fit output."""
        from linum_basic.fit import apply_fit, fit_mosaic, save_corrected
        from linum_basic.io.zarr import load_ome_zarr

        mosaic, _ = synthetic_mosaic_grid
        fit = fit_mosaic(mosaic, basic_kwargs={"estimate_darkfield": False})
        expected = apply_fit(mosaic, fit)

        out = tmp_path / "corrected.ome.zarr"
        save_corrected(mosaic, fit, out, overwrite=True)

        loaded, axes, _ = load_ome_zarr(out)
        np.testing.assert_array_almost_equal(loaded, expected, decimal=5)
        assert axes == ["z", "y", "x"]
