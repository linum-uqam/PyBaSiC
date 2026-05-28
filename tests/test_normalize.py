"""Tests for BaSiC.normalize().

Validates the shading correction formula: (img - darkfield) / (flatfield + ε).
"""

from __future__ import annotations

import numpy as np

from linum_basic.core import BaSiC

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(
    flatfield: np.ndarray,
    darkfield: np.ndarray | None = None,
) -> BaSiC:
    """Create a BaSiC instance with pre-set flat/dark fields."""
    # Build a minimal single-image stack that matches the field shape.
    h, w = flatfield.shape
    stack = np.ones((1, h, w), dtype=np.float32)
    model = BaSiC(stack, estimate_darkfield=darkfield is not None)
    model.flatfield_fullsize = flatfield.copy()
    model.darkfield_fullsize = darkfield.copy() if darkfield is not None else np.zeros_like(flatfield)
    return model


# ---------------------------------------------------------------------------
# normalize() formula tests
# ---------------------------------------------------------------------------


class TestNormalizeFormula:
    """Verify (img - dark) / (flat + ε) is applied correctly."""

    def test_identity_correction(self) -> None:
        """With flat == 1 and dark == 0, normalize() is identity (float input)."""
        h, w = 8, 8
        flat = np.ones((h, w), dtype=np.float32)
        dark = np.zeros((h, w), dtype=np.float32)
        img = np.random.default_rng(42).random((h, w)).astype(np.float32)

        model = _make_model(flat, dark)
        corrected = model.normalize(img, clip=False)

        np.testing.assert_allclose(corrected, img, atol=1e-5)

    def test_flat_field_inversion(self) -> None:
        """normalize() removes multiplicative flat-field to recover baseline."""
        h, w = 16, 16
        rng = np.random.default_rng(7)

        # Create a smooth flat-field centred at 1.0
        y, x = np.mgrid[-1 : 1 : h * 1j, -1 : 1 : w * 1j]  # type: ignore[misc]
        flat = (0.7 + 0.3 * np.exp(-0.5 * (x**2 + y**2))).astype(np.float32)
        dark = np.zeros((h, w), dtype=np.float32)

        # Clean baseline image (what we'd ideally recover)
        baseline = rng.uniform(0.4, 0.6, (h, w)).astype(np.float32)

        # Simulate the shading effect: measured = flat * baseline + dark
        measured = (flat * baseline).astype(np.float32)

        model = _make_model(flat, dark)
        corrected = model.normalize(measured, clip=False)

        np.testing.assert_allclose(corrected, baseline, rtol=1e-3, atol=1e-5)

    def test_darkfield_subtraction(self) -> None:
        """normalize() subtracts dark-field before dividing by flat-field."""
        h, w = 8, 8
        flat = np.ones((h, w), dtype=np.float32)
        dark = 0.1 * np.ones((h, w), dtype=np.float32)
        img = 0.6 * np.ones((h, w), dtype=np.float32)

        model = _make_model(flat, dark)
        corrected = model.normalize(img, clip=False)

        expected = (0.6 - 0.1) / (1.0 + 1e-6)
        np.testing.assert_allclose(corrected, np.full((h, w), expected, dtype=np.float32), atol=1e-5)

    def test_epsilon_prevents_division_by_zero(self) -> None:
        """normalize() does not raise with a near-zero flat-field value."""
        h, w = 4, 4
        flat = np.zeros((h, w), dtype=np.float32)
        img = np.ones((h, w), dtype=np.float32)

        model = _make_model(flat)
        # Should not raise; ε guards against zero division
        corrected = model.normalize(img, clip=False)
        assert np.all(np.isfinite(corrected))


# ---------------------------------------------------------------------------
# dtype handling
# ---------------------------------------------------------------------------


class TestNormalizeDtype:
    """Verify output dtype and integer clipping behaviour."""

    def test_float32_input_preserves_dtype(self) -> None:
        """float32 input → float32 output."""
        h, w = 8, 8
        model = _make_model(np.ones((h, w), dtype=np.float32))
        img = np.ones((h, w), dtype=np.float32) * 0.5
        assert model.normalize(img).dtype == np.float32

    def test_uint8_input_preserves_dtype(self) -> None:
        """uint8 input → uint8 output."""
        h, w = 8, 8
        model = _make_model(np.ones((h, w), dtype=np.float32))
        img = np.full((h, w), 128, dtype=np.uint8)
        assert model.normalize(img).dtype == np.uint8

    def test_uint8_clip(self) -> None:
        """Integer overflow is clipped to [0, 255] when clip=True."""
        h, w = 4, 4
        flat = 0.01 * np.ones((h, w), dtype=np.float32)  # tiny flat → huge result
        dark = np.zeros((h, w), dtype=np.float32)
        img = np.full((h, w), 255, dtype=np.uint8)

        model = _make_model(flat, dark)
        corrected = model.normalize(img, clip=True)

        assert corrected.dtype == np.uint8
        assert corrected.max() <= 255, "Clipping should prevent overflow above 255"

    def test_no_clip_for_float(self) -> None:
        """clip=True has no effect on float inputs (out-of-[0,1] values kept)."""
        h, w = 4, 4
        flat = 0.01 * np.ones((h, w), dtype=np.float32)  # amplify by 100x
        img = np.ones((h, w), dtype=np.float32)

        model = _make_model(flat)
        corrected = model.normalize(img, clip=True)

        # Float should not be clipped to integer range
        assert corrected.max() > 1.0
