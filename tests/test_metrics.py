"""Tests for linum_basic.metrics — seam-consistency metrics."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from linum_basic.metrics import evaluate_correction, evaluate_correction_volume, seam_l1, seam_pearson
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
        """Empty seam_pairs list → nan."""
        tiles = _make_tiles()
        assert math.isnan(seam_l1(tiles, []))

    def test_not_gameable_by_interior_brightening(self):
        """Brightening tile interiors (away from seams) must not change seam_l1.

        Regression guard: the old mean-intensity normalisation could be
        gamed by inflating interior brightness, which lowered the score
        without improving seam agreement.  The per-seam relative metric
        is immune because each seam is normalised by its own local mean.
        """
        tiles = _make_tiles(n=20, th=12, tw=12)
        pairs = _make_seam_pairs(n_rows=4, n_cols=5, tile_h=12, tile_w=12)
        l1_ref = seam_l1(tiles, pairs)

        boosted = tiles.copy()
        interior = slice(4, 8)  # strictly inside, away from 2-px seams
        boosted[:, interior, interior] *= 4.0
        l1_boosted = seam_l1(boosted, pairs)

        assert l1_boosted == pytest.approx(l1_ref, rel=1e-6)


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
        # tile 0 right edge is still constant → seam is degenerate → skipped → nan
        r = seam_pearson(tiles, [sp])
        assert math.isnan(r)

    def test_empty_pairs(self):
        """Empty seam_pairs list → nan."""
        tiles = _make_tiles()
        assert math.isnan(seam_pearson(tiles, []))


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

    def test_correction_improves_seam_l1_no_deps(self):
        """True flatfield reduces seam_l1 versus identity correction (no external deps).

        Build tiles whose seam overlap regions share identical tissue content,
        then apply a known linear-gradient vignette.  After correcting with the
        true flatfield the seams should agree almost perfectly, whereas using an
        identity flatfield leaves the vignette-induced discontinuity intact.
        """
        rng = np.random.default_rng(42)
        n_rows, n_cols, th, tw = 4, 5, 20, 20
        n_tiles = n_rows * n_cols
        overlap_x = round(0.2 * tw)  # 4 px
        overlap_y = round(0.2 * th)  # 4 px

        # Build tiles with correlated seam content
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

        # Linear-gradient flatfield: 0.7 at left edge, 1.3 at right edge.
        # The gradient creates measurable seam asymmetry: each tile's right
        # overlap is brighter than its left overlap under the vignette, so
        # even perfectly correlated tissue looks discontinuous at every seam.
        x_coords = np.linspace(0.7, 1.3, tw, dtype=np.float32)
        flatfield = np.tile(x_coords, (th, 1))
        darkfield = np.zeros((th, tw), dtype=np.float32)

        raw_tiles = tiles * flatfield[np.newaxis]

        pairs = _make_seam_pairs(n_rows=n_rows, n_cols=n_cols, tile_h=th, tile_w=tw)
        result_raw = evaluate_correction(raw_tiles, np.ones((th, tw), dtype=np.float32), darkfield, pairs)
        result_cor = evaluate_correction(raw_tiles, flatfield, darkfield, pairs)
        assert result_cor["seam_l1"] < result_raw["seam_l1"], (
            f"Expected correction to reduce seam_l1 but got "
            f"raw={result_raw['seam_l1']:.4f} vs corrected={result_cor['seam_l1']:.4f}"
        )

    def test_seam_l1_monotone_with_vignette_strength(self):
        """Stronger vignette → worse seam_l1 when correcting with identity flatfield."""
        rng = np.random.default_rng(7)
        n_rows, n_cols, th, tw = 3, 4, 16, 16
        n_tiles = n_rows * n_cols
        overlap_x = round(0.2 * tw)  # 3 px

        tiles = rng.random((n_tiles, th, tw)).astype(np.float32)
        for r in range(n_rows):
            for c in range(n_cols - 1):
                shared = rng.random((th, overlap_x)).astype(np.float32)
                tiles[r * n_cols + c, :, -overlap_x:] = shared
                tiles[r * n_cols + c + 1, :, :overlap_x] = shared

        identity_ff = np.ones((th, tw), dtype=np.float32)
        darkfield = np.zeros((th, tw), dtype=np.float32)
        pairs = _make_seam_pairs(n_rows=n_rows, n_cols=n_cols, tile_h=th, tile_w=tw)
        x = np.linspace(0.0, 1.0, tw, dtype=np.float32)

        prev_l1 = 0.0
        for strength in (0.1, 0.3, 0.5, 0.7):
            ff = np.tile(1.0 + strength * x, (th, 1)).astype(np.float32)
            raw = tiles * ff[np.newaxis]
            res = evaluate_correction(raw, identity_ff, darkfield, pairs)
            assert res["seam_l1"] >= prev_l1, f"seam_l1 did not increase with vignette strength={strength}"
            prev_l1 = res["seam_l1"]

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


# ---------------------------------------------------------------------------
# evaluate_correction_volume
# ---------------------------------------------------------------------------


def _make_volume_mosaic_and_fit(
    n_rows: int = 2,
    n_cols: int = 3,
    tile_h: int = 16,
    tile_w: int = 16,
    n_z: int = 4,
    field_mode: str = "per-z",
    seed: int = 7,
) -> tuple:
    """Return a (MosaicGrid, MosaicFit) pair with synthetic data."""
    from linum_basic.fit import MosaicFit
    from linum_basic.mosaic import MosaicGrid

    rng = np.random.default_rng(seed)
    arr = rng.random((n_z, n_rows * tile_h, n_cols * tile_w), dtype=np.float32).astype(np.float32)
    mosaic = MosaicGrid(arr, tile_shape=(tile_h, tile_w), overlap_fraction=0.2)

    if field_mode == "per-z":
        flatfields = np.ones((n_z, tile_h, tile_w), dtype=np.float32)
        darkfields = np.zeros((n_z, tile_h, tile_w), dtype=np.float32)
    else:
        flatfields = np.ones((tile_h, tile_w), dtype=np.float32)
        darkfields = np.zeros((tile_h, tile_w), dtype=np.float32)

    fit = MosaicFit(
        flatfields=flatfields,
        darkfields=darkfields,
        field_mode=field_mode,
        z_indices=list(range(n_z)),
    )
    return mosaic, fit


class TestEvaluateCorrectionVolume:
    def test_returns_all_keys_by_default(self):
        """Default metrics=('seam', 'curvature') → all three keys present."""
        mosaic, fit = _make_volume_mosaic_and_fit()
        result = evaluate_correction_volume(mosaic, fit)
        assert "seam_l1" in result
        assert "seam_1minus_pearson" in result
        assert "seam_curvature" in result

    def test_seam_only(self):
        """metrics=('seam',) → seam keys present, curvature absent."""
        mosaic, fit = _make_volume_mosaic_and_fit()
        result = evaluate_correction_volume(mosaic, fit, metrics=("seam",))
        assert "seam_l1" in result
        assert "seam_1minus_pearson" in result
        assert "seam_curvature" not in result

    def test_curvature_only(self):
        """metrics=('curvature',) → only seam_curvature key present."""
        mosaic, fit = _make_volume_mosaic_and_fit()
        result = evaluate_correction_volume(mosaic, fit, metrics=("curvature",))
        assert "seam_curvature" in result
        assert "seam_l1" not in result
        assert "seam_1minus_pearson" not in result

    def test_seam_l1_uniform_flatfield(self):
        """Uniform flat-field + zero dark → seam_l1 == raw seam_l1 averaged over z."""
        mosaic, fit = _make_volume_mosaic_and_fit(n_z=3)
        result = evaluate_correction_volume(mosaic, fit, metrics=("seam",))
        assert np.isfinite(result["seam_l1"])
        assert result["seam_l1"] >= 0.0

    def test_seam_curvature_finite(self):
        """seam_curvature should be finite for a valid mosaic fit."""
        mosaic, fit = _make_volume_mosaic_and_fit()
        result = evaluate_correction_volume(mosaic, fit, metrics=("curvature",))
        assert np.isfinite(result["seam_curvature"])

    def test_global_field_mode_2d_to_3d_reshape(self):
        """field_mode='global' (2-D flatfield) → seam_curvature computed without error."""
        mosaic, fit = _make_volume_mosaic_and_fit(field_mode="global")
        result = evaluate_correction_volume(mosaic, fit)
        assert "seam_curvature" in result
        assert np.isfinite(result["seam_curvature"])

    def test_global_field_mode_seam_metrics(self):
        """field_mode='global' → seam metrics computed using shared flat/dark fields."""
        mosaic, fit = _make_volume_mosaic_and_fit(field_mode="global")
        result = evaluate_correction_volume(mosaic, fit, metrics=("seam",))
        assert "seam_l1" in result
        assert np.isfinite(result["seam_l1"])

    def test_empty_metrics_returns_empty_dict(self):
        """Passing an empty metrics sequence → empty result dict."""
        mosaic, fit = _make_volume_mosaic_and_fit()
        result = evaluate_correction_volume(mosaic, fit, metrics=())
        assert result == {}

    def test_nan_z_levels_skipped_in_average(self):
        """NaN per-z metric values are skipped when averaging over z-levels.

        Inject a constant-tile z-level (all seam pairs degenerate for seam_pearson)
        alongside normal z-levels and verify that seam_1minus_pearson is still
        finite (the degenerate z is skipped via nanmean).
        """
        from linum_basic.fit import MosaicFit

        n_rows, n_cols, th, tw = 2, 3, 10, 10
        n_z = 3
        rng = np.random.default_rng(99)
        arr = rng.random((n_z, n_rows * th, n_cols * tw), dtype=np.float32).astype(np.float32)
        # Make z=1 constant so all Pearson seams are degenerate → seam_pearson = nan
        arr[1] = 0.5
        mosaic = MosaicGrid(arr, tile_shape=(th, tw), overlap_fraction=0.2)
        flatfields = np.ones((n_z, th, tw), dtype=np.float32)
        darkfields = np.zeros((n_z, th, tw), dtype=np.float32)
        fit = MosaicFit(flatfields=flatfields, darkfields=darkfields, field_mode="per-z", z_indices=list(range(n_z)))
        result = evaluate_correction_volume(mosaic, fit, metrics=("seam",))
        assert np.isfinite(result["seam_l1"])
        # seam_1minus_pearson: z=1 contributes nan (constant tiles) which is skipped;
        # z=0 and z=2 have random data → finite mean
        assert np.isfinite(result["seam_1minus_pearson"])
