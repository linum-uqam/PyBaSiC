"""Tests for the Zernike-based synthetic flat-field generator."""

from __future__ import annotations

import numpy as np
import pytest

from linum_basic.data import _zernike, _zernike_radial, zernike_flatfield


def test_flatfield_shape_and_dtype():
    field = zernike_flatfield(64, n_max=4, seed=0)
    assert field.shape == (64, 64)
    assert field.dtype == np.float32


def test_flatfield_normalisation_and_positivity():
    field = zernike_flatfield(96, n_max=5, contrast=0.5, seed=1)
    assert abs(float(field.mean()) - 1.0) < 1e-5
    assert float(field.min()) > 0.0


def test_deterministic_given_seed():
    a = zernike_flatfield(48, n_max=4, seed=7)
    b = zernike_flatfield(48, n_max=4, seed=7)
    c = zernike_flatfield(48, n_max=4, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_radial_defocus_matches_closed_form():
    # Z_2^0 radial polynomial is R = 2*rho**2 - 1 (defocus).
    rho = np.linspace(0.0, 1.0, 50)
    np.testing.assert_allclose(_zernike_radial(2, 0, rho), 2 * rho**2 - 1, atol=1e-12)


def test_radial_piston_is_unity():
    rho = np.linspace(0.0, 1.0, 20)
    np.testing.assert_allclose(_zernike_radial(0, 0, rho), np.ones_like(rho), atol=1e-12)


def test_defocus_field_is_rotationally_symmetric():
    # A pure defocus mode (m=0) has no angular dependence, so the resulting
    # field must be invariant under a 90-degree rotation.
    field = zernike_flatfield(80, n_max=2, coeffs={(2, 0): 1.0}, contrast=0.4)
    rotated = np.rot90(field)
    np.testing.assert_allclose(field, rotated, atol=1e-5)


def test_odd_mode_breaks_symmetry():
    # A tilt mode (Z_1^1, cos(theta)) is antisymmetric left-right about center,
    # so the field must differ from its left-right mirror.
    field = zernike_flatfield(80, n_max=1, coeffs={(1, 1): 1.0}, contrast=0.4)
    assert not np.allclose(field, np.fliplr(field), atol=1e-3)


def test_higher_order_increases_spatial_variation():
    # For modes of equal amplitude, a higher Zernike radial order carries more
    # curvature; the Laplacian magnitude (a curvature proxy) should be larger
    # for spherical aberration (Z_4^0) than for defocus (Z_2^0).
    def roughness(field: np.ndarray) -> float:
        lap = -4 * field + np.roll(field, 1, 0) + np.roll(field, -1, 0) + np.roll(field, 1, 1) + np.roll(field, -1, 1)
        return float(np.std(lap[2:-2, 2:-2]))

    defocus = zernike_flatfield(96, n_max=4, coeffs={(2, 0): 1.0}, contrast=0.4)
    spherical = zernike_flatfield(96, n_max=4, coeffs={(4, 0): 1.0}, contrast=0.4)
    assert roughness(spherical) > roughness(defocus)


def test_angular_part_signs():
    rho = np.array([1.0])
    theta = np.array([0.0])
    # cos(0) = 1 for m >= 0
    assert _zernike(1, 1, rho, theta)[0] == pytest.approx(1.0)
    # sin(0) = 0 for m < 0
    assert _zernike(1, -1, rho, theta)[0] == pytest.approx(0.0)
