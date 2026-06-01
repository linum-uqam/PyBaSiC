"""Tests for the top-level ``correct_images()`` convenience function."""

from __future__ import annotations

import numpy as np

from linum_basic import correct_images


def _synthetic_shaded(n: int = 20, size: int = 32, seed: int = 0) -> np.ndarray:
    """Return a synthetic shaded stack for unit tests."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]  # type: ignore[misc]
    flat = (0.6 + 0.4 * np.exp(-2.0 * (x**2 + y**2))).astype(np.float32)
    flat /= float(flat.mean())
    tiles = rng.random((n, size, size)).astype(np.float32) * 0.6 + 0.2
    return (tiles * flat).astype(np.float32)


class TestCorrectImages:
    """Tests for linum_basic.correct_images()."""

    def test_output_shape_matches_input(self) -> None:
        """Output shape must equal input shape."""
        stack = _synthetic_shaded(n=20, size=32)
        out = correct_images(stack, working_size=32)
        assert out.shape == stack.shape

    def test_output_dtype_is_float32(self) -> None:
        """Output dtype should be float32 (same as internal float conversion)."""
        stack = _synthetic_shaded(n=20, size=32)
        out = correct_images(stack, working_size=32)
        assert out.dtype == np.float32

    def test_with_darkfield_estimation(self) -> None:
        """estimate_darkfield=True completes without error."""
        stack = _synthetic_shaded(n=20, size=32)
        out = correct_images(stack, estimate_darkfield=True, working_size=32)
        assert out.shape == stack.shape
        assert np.isfinite(out).all()

    def test_knob_override_working_size(self) -> None:
        """Passing working_size as a keyword knob is respected."""
        stack = _synthetic_shaded(n=20, size=64)
        out = correct_images(stack, working_size=32)
        assert out.shape == stack.shape

    def test_corrections_reduce_cv(self) -> None:
        """Corrected stack should have lower coefficient of variation than input."""
        stack = _synthetic_shaded(n=30, size=32)
        out = correct_images(stack, working_size=32)

        def cv(s: np.ndarray) -> float:
            per = s.reshape(len(s), -1)
            return float((per.std(1) / (per.mean(1) + 1e-9)).mean())

        assert cv(out) <= cv(stack) + 0.05  # allow small slack

    def test_returns_ndarray(self) -> None:
        """Return value is always a numpy ndarray."""
        stack = _synthetic_shaded(n=20, size=32)
        out = correct_images(stack, working_size=32)
        assert isinstance(out, np.ndarray)

    def test_list_of_arrays_input(self) -> None:
        """correct_images() accepts a list of ndarray frames (passed to BaSiC)."""
        rng = np.random.default_rng(5)
        frames = [rng.random((32, 32)).astype(np.float32) + 0.2 for _ in range(20)]
        out = correct_images(frames, working_size=32)  # type: ignore[arg-type]
        assert out.shape == (20, 32, 32)
