"""Tests for the working_size parameter in BaSiC.

Verifies that the output images are always upsampled back to the original
resolution regardless of working_size, and that different working_size values
give consistent flat-field estimates.
"""

from __future__ import annotations

import numpy as np
import pytest

from linum_basic.core import BaSiC

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_field(size: int, sigma: float = 0.5) -> np.ndarray:
    y, x = np.mgrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]  # type: ignore[misc]
    return np.exp(-0.5 * (x**2 + y**2) / sigma**2).astype(np.float32)


def _make_stack(n: int, size: int, *, rng: np.random.Generator) -> np.ndarray:
    flat_field = _gaussian_field(size)
    flat_field /= flat_field.mean()
    B = rng.uniform(0.2, 0.8, (n, 1, 1)).astype(np.float32)
    return (flat_field[None] * B).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("working_size", [32, 64, 128])
def test_output_shape_matches_input(working_size: int) -> None:
    """flatfield_fullsize and darkfield_fullsize match the input resolution.

    The BaSiC solver operates at working_size x working_size internally but
    must upsample results back to the original image shape before exposing
    them via the public attributes.
    """
    rng = np.random.default_rng(0)
    input_size = 64
    stack = _make_stack(20, input_size, rng=rng)

    model = BaSiC(stack, estimate_darkfield=False)
    model.working_size = working_size
    model.prepare()
    model.run()

    assert model.flatfield_fullsize.shape == (input_size, input_size), (
        f"flatfield_fullsize shape {model.flatfield_fullsize.shape} "
        f"!= ({input_size}, {input_size}) for working_size={working_size}"
    )
    assert model.darkfield_fullsize.shape == (input_size, input_size), (
        f"darkfield_fullsize shape {model.darkfield_fullsize.shape} "
        f"!= ({input_size}, {input_size}) for working_size={working_size}"
    )


def test_working_size_smaller_than_input() -> None:
    """working_size < input size triggers downsampling then upsampling."""
    rng = np.random.default_rng(1)
    stack = _make_stack(20, 64, rng=rng)

    model = BaSiC(stack, estimate_darkfield=False)
    model.working_size = 16  # smaller than 64
    model.prepare()
    model.run()

    assert model.flatfield_fullsize.shape == (64, 64)


def test_working_size_equal_to_input() -> None:
    """working_size == input size produces no resize artifacts."""
    rng = np.random.default_rng(2)
    input_size = 32
    flat_field = _gaussian_field(input_size)
    flat_field /= flat_field.mean()
    B = rng.uniform(0.2, 0.8, (20, 1, 1)).astype(np.float32)
    stack = (flat_field[None] * B).astype(np.float32)

    model = BaSiC(stack, estimate_darkfield=False)
    model.working_size = input_size
    model.prepare()
    model.run()

    r = float(np.corrcoef(flat_field.ravel(), model.flatfield_fullsize.ravel())[0, 1])
    assert r > 0.95, f"Flat-field r={r:.3f} is too low for working_size == input_size"


def test_flat_field_normalized() -> None:
    """flatfield_fullsize always has mean ≈ 1.0 after run()."""
    rng = np.random.default_rng(3)
    stack = _make_stack(20, 32, rng=rng)

    model = BaSiC(stack, estimate_darkfield=False)
    model.prepare()
    model.run()

    mean = float(model.flatfield_fullsize.mean())
    assert abs(mean - 1.0) < 0.05, f"Flat-field mean {mean:.4f} is not close to 1.0"
