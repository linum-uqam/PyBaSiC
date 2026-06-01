"""Tests for linum_basic.curvature — Gaussian fitting and seam curvature."""

from __future__ import annotations

import numpy as np
import pytest

from linum_basic.curvature import (
    GaussianParams,
    curvature_depth_profile,
    fit_focal_gaussian,
    focal_profile,
    seam_curvature,
    seam_curvature_per_z,
)
from linum_basic.mosaic import MosaicGrid, SeamPair

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _gaussian_profile(
    n: int = 64,
    amplitude: float = 0.5,
    center: float = 32.0,
    sigma: float = 15.0,
    offset: float = 0.5,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Return a 1-D Gaussian profile, optionally with additive noise."""
    x = np.arange(n, dtype=np.float64)
    p = amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset
    if noise > 0:
        rng = np.random.default_rng(seed)
        p = p + rng.normal(0, noise, size=n)
    return p


def _gaussian_field(
    th: int = 32,
    tw: int = 64,
    amplitude: float = 0.5,
    center: float = 32.0,
    sigma: float = 15.0,
    offset: float = 0.5,
    orientation: str = "horizontal",
) -> np.ndarray:
    """Return a 2-D flatfield whose mean profile is a 1-D Gaussian."""
    if orientation == "horizontal":
        profile = _gaussian_profile(tw, amplitude, center, sigma, offset)
        return np.tile(profile, (th, 1)).astype(np.float32)
    profile = _gaussian_profile(th, amplitude, center, sigma, offset)
    return np.tile(profile[:, np.newaxis], (1, tw)).astype(np.float32)


def _make_seam_pairs(
    n_rows: int = 1,
    n_cols: int = 2,
    tile_h: int = 32,
    tile_w: int = 64,
    n_z: int = 1,
    overlap_fraction: float = 0.2,
) -> list[SeamPair]:
    arr = np.ones((n_z, n_rows * tile_h, n_cols * tile_w), dtype=np.float32)
    return MosaicGrid(arr, tile_shape=(tile_h, tile_w), overlap_fraction=overlap_fraction).seam_pairs()


# ---------------------------------------------------------------------------
# fit_focal_gaussian
# ---------------------------------------------------------------------------


class TestFitFocalGaussian:
    def test_recovers_known_params(self):
        """Clean Gaussian → recovered params within 10% of ground truth."""
        a, c, s, o = 0.5, 30.0, 12.0, 0.4
        profile = _gaussian_profile(64, a, c, s, o)
        result = fit_focal_gaussian(profile)
        assert not np.isnan(result.sigma), "Fit should converge on a clean Gaussian"
        assert result.amplitude == pytest.approx(a, rel=0.1)
        assert result.center == pytest.approx(c, rel=0.1)
        assert result.sigma == pytest.approx(s, rel=0.1)
        assert result.offset == pytest.approx(o, rel=0.1)

    def test_recovers_params_with_noise(self):
        """Gaussian + 1% Gaussian noise → fit converges (sigma not NaN)."""
        profile = _gaussian_profile(64, noise=0.01)
        result = fit_focal_gaussian(profile)
        assert not np.isnan(result.sigma)

    def test_off_center_gaussian(self):
        """Gaussian with center far from mid-point → fit converges."""
        profile = _gaussian_profile(64, center=10.0, sigma=5.0)
        result = fit_focal_gaussian(profile)
        assert not np.isnan(result.sigma)
        assert result.center == pytest.approx(10.0, abs=2.0)

    def test_nan_on_too_short_profile(self):
        """Profile shorter than 4 points → GaussianParams with NaN fields."""
        result = fit_focal_gaussian(np.array([1.0, 2.0, 1.0]))
        assert np.isnan(result.sigma)
        assert np.isnan(result.rms_residual)

    def test_rms_residual_near_zero_for_perfect_gaussian(self):
        """Perfect Gaussian profile → rms_residual < 1e-6."""
        profile = _gaussian_profile(64)
        result = fit_focal_gaussian(profile)
        assert not np.isnan(result.rms_residual)
        assert result.rms_residual < 1e-6

    def test_rms_residual_larger_for_non_gaussian(self):
        """Non-Gaussian profile → larger residual than a clean Gaussian."""
        clean = _gaussian_profile(64)
        noisy = clean.copy()
        noisy[20:24] += 0.5  # spike in the middle — clearly non-Gaussian
        res_clean = fit_focal_gaussian(clean).rms_residual
        res_noisy = fit_focal_gaussian(noisy).rms_residual
        assert res_noisy > res_clean

    def test_returns_gaussian_params_type(self):
        result = fit_focal_gaussian(_gaussian_profile(32))
        assert isinstance(result, GaussianParams)


# ---------------------------------------------------------------------------
# focal_profile
# ---------------------------------------------------------------------------


class TestFocalProfile:
    def test_horizontal_returns_column_mean(self):
        """horizontal → mean over rows → shape (tw,)."""
        rng = np.random.default_rng(0)
        field = rng.random((16, 32)).astype(np.float32)
        fp = focal_profile(field, "horizontal")
        np.testing.assert_allclose(fp, field.mean(axis=0))

    def test_vertical_returns_row_mean(self):
        """vertical → mean over cols → shape (th,)."""
        rng = np.random.default_rng(1)
        field = rng.random((16, 32)).astype(np.float32)
        fp = focal_profile(field, "vertical")
        np.testing.assert_allclose(fp, field.mean(axis=1))


# ---------------------------------------------------------------------------
# seam_curvature
# ---------------------------------------------------------------------------


class TestSeamCurvature:
    def test_near_zero_on_perfect_gaussian_flatfield(self):
        """A flatfield that is a perfect Gaussian → curvature ≈ 0."""
        field = _gaussian_field(th=32, tw=64)
        flatfields = field[np.newaxis]  # (1, th, tw)
        seam_pairs = _make_seam_pairs(n_rows=1, n_cols=2, tile_h=32, tile_w=64)
        # Filter to horizontal only
        h_pairs = [s for s in seam_pairs if s.orientation == "horizontal"]
        result = seam_curvature(flatfields, h_pairs)
        assert np.isfinite(result), "metric should be finite for a Gaussian field"
        assert result < 0.01, f"expected < 0.01, got {result:.6f}"

    def test_increases_with_non_gaussian_distortion(self):
        """Distorting a Gaussian field with a spike → curvature increases."""
        field_clean = _gaussian_field(th=32, tw=64)
        field_noisy = field_clean.copy()
        # Add a non-Gaussian spike in the overlap region (last ~20% columns)
        ovl = round(0.2 * 64)
        field_noisy[:, 64 - ovl : 64 - ovl + 2] += 0.5
        seam_pairs = _make_seam_pairs(n_rows=1, n_cols=2, tile_h=32, tile_w=64)
        h_pairs = [s for s in seam_pairs if s.orientation == "horizontal"]
        curv_clean = seam_curvature(field_clean[np.newaxis], h_pairs)
        curv_noisy = seam_curvature(field_noisy[np.newaxis], h_pairs)
        assert curv_noisy > curv_clean, f"noisy field curvature {curv_noisy:.6f} should exceed clean {curv_clean:.6f}"

    def test_off_center_gaussian_gives_finite_result(self):
        """Off-center Gaussian flatfield → finite metric (free mean handles it)."""
        field = _gaussian_field(th=32, tw=64, center=10.0, sigma=8.0)
        seam_pairs = _make_seam_pairs(n_rows=1, n_cols=2, tile_h=32, tile_w=64)
        h_pairs = [s for s in seam_pairs if s.orientation == "horizontal"]
        result = seam_curvature(field[np.newaxis], h_pairs)
        assert np.isfinite(result), "off-center Gaussian should yield a finite metric"

    def test_horizontal_orientation(self):
        """Horizontal seam pairs produce a finite metric."""
        field = _gaussian_field(th=32, tw=64)
        seam_pairs = _make_seam_pairs(n_rows=1, n_cols=2, tile_h=32, tile_w=64)
        h_pairs = [s for s in seam_pairs if s.orientation == "horizontal"]
        assert h_pairs, "Expected at least one horizontal seam pair"
        result = seam_curvature(field[np.newaxis], h_pairs)
        assert np.isfinite(result)

    def test_vertical_orientation(self):
        """Vertical seam pairs produce a finite metric."""
        field = _gaussian_field(th=32, tw=64, orientation="vertical")
        # 2x1 grid → vertical seams (top-bottom)
        seam_pairs = _make_seam_pairs(n_rows=2, n_cols=1, tile_h=32, tile_w=64)
        v_pairs = [s for s in seam_pairs if s.orientation == "vertical"]
        assert v_pairs, "Expected at least one vertical seam pair"
        result = seam_curvature(field[np.newaxis], v_pairs)
        assert np.isfinite(result)

    def test_returns_zero_for_empty_pairs(self):
        """No seam pairs → metric returns 0.0."""
        field = _gaussian_field()
        result = seam_curvature(field[np.newaxis], [])
        assert result == pytest.approx(0.0)

    def test_multi_z(self):
        """Stack of 5 z-levels → scalar metric (average)."""
        flatfields = np.stack([_gaussian_field() for _ in range(5)])
        seam_pairs = _make_seam_pairs(n_rows=1, n_cols=2, tile_h=32, tile_w=64)
        result = seam_curvature(flatfields, seam_pairs)
        assert np.isfinite(result)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# seam_curvature_per_z
# ---------------------------------------------------------------------------


class TestSeamCurvaturePerZ:
    def test_shape(self):
        """seam_curvature_per_z returns array of shape (Z,)."""
        n_z = 5
        flatfields = np.stack([_gaussian_field() for _ in range(n_z)])
        seam_pairs = _make_seam_pairs(n_rows=1, n_cols=2, tile_h=32, tile_w=64)
        result = seam_curvature_per_z(flatfields, seam_pairs)
        assert result.shape == (n_z,), f"Expected ({n_z},), got {result.shape}"

    def test_all_finite_for_gaussian_fields(self):
        """Gaussian flatfields at each z → all per-z values finite."""
        n_z = 4
        flatfields = np.stack([_gaussian_field() for _ in range(n_z)])
        seam_pairs = _make_seam_pairs(n_rows=1, n_cols=2, tile_h=32, tile_w=64)
        result = seam_curvature_per_z(flatfields, seam_pairs)
        assert np.all(np.isfinite(result))

    def test_consistent_with_per_z_scalar(self):
        """seam_curvature_per_z(z) should equal seam_curvature on single z-slice."""
        n_z = 3
        flatfields = np.stack([_gaussian_field(sigma=s) for s in [10.0, 15.0, 20.0]])
        seam_pairs = _make_seam_pairs(n_rows=1, n_cols=2, tile_h=32, tile_w=64)
        per_z = seam_curvature_per_z(flatfields, seam_pairs)
        for i in range(n_z):
            scalar = seam_curvature(flatfields[i : i + 1], seam_pairs)
            assert per_z[i] == pytest.approx(scalar, rel=1e-5), f"z={i}: per_z={per_z[i]:.8f} vs scalar={scalar:.8f}"


# ---------------------------------------------------------------------------
# curvature_depth_profile
# ---------------------------------------------------------------------------


class TestCurvatureDepthProfile:
    def test_returns_one_per_z(self):
        """Returns a list with one GaussianParams per z-level."""
        n_z = 6
        flatfields = np.stack([_gaussian_field() for _ in range(n_z)])
        seam_pairs = _make_seam_pairs(n_rows=1, n_cols=2, tile_h=32, tile_w=64)
        result = curvature_depth_profile(flatfields, seam_pairs, orientation="horizontal")
        assert len(result) == n_z

    def test_all_gaussian_params_type(self):
        """All elements are GaussianParams instances."""
        n_z = 3
        flatfields = np.stack([_gaussian_field() for _ in range(n_z)])
        seam_pairs = _make_seam_pairs(n_rows=1, n_cols=2, tile_h=32, tile_w=64)
        result = curvature_depth_profile(flatfields, seam_pairs, orientation="horizontal")
        for i, gp in enumerate(result):
            assert isinstance(gp, GaussianParams), f"z={i}: expected GaussianParams, got {type(gp)}"

    def test_sigma_changes_with_depth(self):
        """If sigma varies with z, the profile should reflect that."""
        sigmas_in = [8.0, 12.0, 16.0, 20.0]
        flatfields = np.stack([_gaussian_field(sigma=s) for s in sigmas_in])
        seam_pairs = _make_seam_pairs(n_rows=1, n_cols=2, tile_h=32, tile_w=64)
        result = curvature_depth_profile(flatfields, seam_pairs, orientation="horizontal")
        sigmas_out = [gp.sigma for gp in result if not np.isnan(gp.sigma)]
        assert len(sigmas_out) == len(sigmas_in), "All fits should converge on clean Gaussians"
        # Sigma should be monotonically increasing
        assert sigmas_out == sorted(sigmas_out), f"Expected monotone sigma, got {sigmas_out}"

    def test_empty_seam_pairs_uses_orientation_fallback(self):
        """No seam pairs → orientation kwarg used as fallback; still returns n_z params."""
        n_z = 3
        flatfields = np.stack([_gaussian_field() for _ in range(n_z)])
        result = curvature_depth_profile(flatfields, [], orientation="horizontal")
        assert len(result) == n_z
        # Falls back to orientation="horizontal" → fits are on a Gaussian profile → finite
        for gp in result:
            assert isinstance(gp, GaussianParams)
