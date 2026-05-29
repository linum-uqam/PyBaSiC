"""Tests for linum_basic.metrics — seam-consistency metrics."""

from __future__ import annotations

import random

import numpy as np
import pytest

from linum_basic.metrics import evaluate_correction, seam_l1, seam_pearson
from linum_basic.mosaic import MosaicGrid, SeamPair

try:
    from sbh_simulator.simulator import generate_gaussian_vignette

    _SBH_SIMULATOR_AVAILABLE = True
except ImportError:
    _SBH_SIMULATOR_AVAILABLE = False


def _make_tiles(n=20, th=10, tw=10, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, th, tw), dtype=np.float64).astype(np.float32)


def _make_seam_pairs(n_rows=4, n_cols=5, tile_h=10, tile_w=10, n_z=1, overlap_fraction=0.2) -> list[SeamPair]:
    arr = np.ones((n_z, n_rows * tile_h, n_cols * tile_w), dtype=np.float32)
    return MosaicGrid(arr, tile_shape=(tile_h, tile_w), overlap_fraction=overlap_fraction).seam_pairs()


class TestSeamL1:
    def test_perfect_match(self):
        """seam_l1 = 0 when all neighbouring overlaps are identical."""
        tiles = np.ones((20, 10, 10), dtype=np.float32)
        pairs = _make_seam_pairs(n_rows=4, n_cols=5)
        assert seam_l1(tiles, pairs) == pytest.approx(0.0)

    def test_increases_with_error(self):
        """Larger discrepancy between seam regions → larger seam_l1."""
        tiles_good = np.ones((20, 10, 10), dtype=np.float32)
        tiles_bad = tiles_good.copy()
        # Introduce a discontinuity at every seam
        overlap_x = round(0.2 * 10)  # 2
        tiles_bad[:, :, -overlap_x:] += 1.0  # right edge of every tile brighter
        pairs = _make_seam_pairs(n_rows=4, n_cols=5)
        l1_good = seam_l1(tiles_good, pairs)
        l1_bad = seam_l1(tiles_bad, pairs)
        assert l1_bad > l1_good

    def test_scale_invariance(self):
        """Multiplying all tiles by a scalar must not change seam_l1."""
        tiles = _make_tiles()
        pairs = _make_seam_pairs()
        l1_ref = seam_l1(tiles, pairs)
        l1_scaled = seam_l1(tiles * 5.0, pairs)
        assert l1_ref == pytest.approx(l1_scaled, rel=1e-4)

    def test_empty_pairs(self):
        """Empty seam_pairs list → 0."""
        tiles = _make_tiles()
        assert seam_l1(tiles, []) == pytest.approx(0.0)


class TestSeamPearson:
    def test_perfect_correlation(self):
        """Seam regions with identical content -> Pearson = 1.

        Use a 1-row grid (only horizontal seams) to avoid corner-pixel
        conflicts between horizontal and vertical seam assignments.
        """
        rng = np.random.default_rng(42)
        n_rows, n_cols, th, tw = 1, 10, 10, 10
        overlap_x = round(0.2 * tw)  # 2
        n_tiles = n_cols

        tiles = rng.random((n_tiles, th, tw)).astype(np.float32)
        for c in range(n_cols - 1):
            shared = rng.random((th, overlap_x)).astype(np.float32)
            tiles[c, :, -overlap_x:] = shared
            tiles[c + 1, :, :overlap_x] = shared

        pairs = _make_seam_pairs(n_rows=n_rows, n_cols=n_cols)
        assert seam_pearson(tiles, pairs) == pytest.approx(1.0, abs=1e-5)

    def test_anticorrelated_overlap(self):
        """Negated overlap region → Pearson < 1."""
        tiles = _make_tiles()
        pairs = _make_seam_pairs()
        # Flip the right column of every tile (seam region of the *right* neighbour)
        overlap_x = round(0.2 * 10)
        tiles_flipped = tiles.copy()
        tiles_flipped[:, :, :overlap_x] = 1.0 - tiles_flipped[:, :, :overlap_x]
        r = seam_pearson(tiles_flipped, pairs)
        assert r < 1.0

    def test_constant_overlap_skipped(self):
        """A seam where one side is constant should be skipped (no divide by zero)."""
        tiles = np.ones((2, 10, 10), dtype=np.float32)
        # Only one seam: horizontal between tile 0 and tile 1
        sp = SeamPair(
            0,
            1,
            (slice(None), slice(8, 10)),  # right 2 cols of tile 0
            (slice(None), slice(0, 2)),  # left 2 cols of tile 1
            "horizontal",
        )
        # tile 0 right is constant 1; tile 1 left gets a gradient → std of left != 0
        tiles[1, :, :2] = np.arange(10, dtype=np.float32).reshape(10, 1).repeat(2, axis=1)
        # tile 0 right edge is still constant → seam should be skipped
        r = seam_pearson(tiles, [sp])
        # Only one seam; its left side is constant → skipped → default 1.0
        assert r == pytest.approx(1.0)

    def test_empty_pairs(self):
        """Empty seam_pairs list → 1.0 (perfect by default)."""
        tiles = _make_tiles()
        assert seam_pearson(tiles, []) == pytest.approx(1.0)


class TestEvaluateCorrection:
    def test_returns_dict_keys(self):
        tiles = _make_tiles(n=20, th=10, tw=10)
        flatfield = np.ones((10, 10), dtype=np.float32)
        darkfield = np.zeros((10, 10), dtype=np.float32)
        pairs = _make_seam_pairs()
        result = evaluate_correction(tiles, flatfield, darkfield, pairs)
        assert "seam_l1" in result
        assert "seam_1minus_pearson" in result

    def test_ideal_flatfield(self):
        """Uniform flatfield + zero darkfield → metrics match raw metrics."""
        tiles = _make_tiles(n=20, th=10, tw=10)
        flatfield = np.ones((10, 10), dtype=np.float32)
        darkfield = np.zeros((10, 10), dtype=np.float32)
        pairs = _make_seam_pairs()
        result = evaluate_correction(tiles, flatfield, darkfield, pairs)
        # With a flat field of 1 and dark of 0, corrected == tiles → same L1
        expected_l1 = seam_l1(tiles, pairs)
        assert result["seam_l1"] == pytest.approx(expected_l1, rel=1e-4)

    @pytest.mark.skipif(
        not _SBH_SIMULATOR_AVAILABLE,
        reason="sbh-simulator not installed",
    )
    def test_correction_improves_seam_l1(self):
        """Applying the true flatfield reduces seam_l1 when seams share content."""
        rng = np.random.default_rng(99)
        n_rows, n_cols, th, tw = 4, 5, 12, 12
        n_tiles = n_rows * n_cols
        overlap_x = round(0.2 * tw)  # 2 px
        overlap_y = round(0.2 * th)  # 2 px

        # Build tiles with correlated seam content (same "tissue" at shared boundaries)
        tiles = rng.random((n_tiles, th, tw)).astype(np.float32)
        for r in range(n_rows):
            for c in range(n_cols - 1):
                shared = rng.random((th, overlap_x)).astype(np.float32)
                tiles[r * n_cols + c, :, -overlap_x:] = shared
                tiles[r * n_cols + c + 1, :, :overlap_x] = shared
        for r in range(n_rows - 1):
            for c in range(n_cols):
                shared = rng.random((overlap_y, tw)).astype(np.float32)
                tiles[r * n_cols + c, -overlap_y:, :] = shared
                tiles[(r + 1) * n_cols + c, :overlap_y, :] = shared

        # Ground-truth flatfield via sbh-simulator (Gaussian vignette, max=1)
        flatfield = generate_gaussian_vignette(width=tw, height=th, sigma=0.7, rng=random.Random(99)).astype(np.float32)
        darkfield = np.zeros_like(flatfield)

        # Apply vignette -- seam regions now look different even though they share tissue
        raw_tiles = tiles * flatfield[np.newaxis]

        pairs = _make_seam_pairs(n_rows=n_rows, n_cols=n_cols, tile_h=th, tile_w=tw)
        result_raw = evaluate_correction(raw_tiles, np.ones_like(flatfield), darkfield, pairs)
        result_cor = evaluate_correction(raw_tiles, flatfield, darkfield, pairs)
        assert result_cor["seam_l1"] < result_raw["seam_l1"]
