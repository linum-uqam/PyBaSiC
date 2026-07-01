"""Functional / parity tests for :func:`linum_basic._alm.inexact_alm_l1`.

Generates a synthetic image stack with a known flat-field and sparse noise,
then verifies that the ALM solver recovers the flat-field to within an
acceptable tolerance.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from linum_basic._alm import inexact_alm_l1, shrink
from linum_basic.backend import Backend, get_xp
from linum_basic.core import BaSiC

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_stack() -> tuple[np.ndarray, np.ndarray]:
    """Return an 8x32x32 synthetic image stack and its ground-truth flat-field.

    The flat-field is a smooth bilinear ramp (mean 1.0).  Random sparse noise
    is added on top so that the solver has something to separate out.

    Returns
    -------
    tuple of numpy.ndarray
        ``(stack, flatfield)`` both of dtype ``float32``.
    """
    rng = np.random.default_rng(0)
    n, h, w = 8, 32, 32

    # Smooth flat-field: bilinear ramp, mean-normalised to 1.0
    xs, ys = np.meshgrid(np.linspace(0.8, 1.2, w), np.linspace(0.9, 1.1, h), indexing="xy")
    flatfield = (xs * ys).astype(np.float32)
    flatfield = flatfield / flatfield.mean()

    # Sparse additive noise (≈5% non-zero)
    noise = rng.normal(0, 0.05, (n, h, w)).astype(np.float32)
    mask = rng.random((n, h, w)) < 0.05
    noise *= mask

    stack = (flatfield[None, :, :] + noise).astype(np.float32)
    return stack, flatfield


# ---------------------------------------------------------------------------
# shrink
# ---------------------------------------------------------------------------


class TestShrink:
    """Unit tests for the scalar shrink (soft-threshold) operator."""

    def test_zero_threshold(self) -> None:
        """Shrink with ε=0 is the identity (sign(x)*|x| = x)."""
        xp = get_xp(Backend.NUMPY)
        x = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
        out = xp.to_numpy(shrink(xp, xp.asarray(x), epsilon=0.0))
        np.testing.assert_allclose(out, x, atol=1e-6)

    def test_positive_thresholding(self) -> None:
        """Values below ε are zeroed; others are reduced by ε."""
        xp = get_xp(Backend.NUMPY)
        x = np.array([1.0, 0.5, -1.0, -0.3, 0.0], dtype=np.float32)
        out = xp.to_numpy(shrink(xp, xp.asarray(x), epsilon=0.6))
        expected = np.array([0.4, 0.0, -0.4, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# inexact_alm_l1 — NumPy backend
# ---------------------------------------------------------------------------


class TestAlmNumpy:
    """Functional tests for :func:`inexact_alm_l1` using the NumPy backend."""

    def test_returns_expected_shapes(self, synthetic_stack: tuple[np.ndarray, np.ndarray]) -> None:
        """Output arrays have the same shape as the input stack."""
        stack, _ = synthetic_stack
        _n, h, w = stack.shape
        xp = get_xp(Backend.NUMPY)
        l_s = 0.5
        l_d = 0.2
        Ib, Ir, D, _ = inexact_alm_l1(stack, l_s, l_d, max_iter=10, xp=xp)
        assert Ib.shape == stack.shape
        assert Ir.shape == stack.shape
        assert D.shape == (1, h * w)

    def test_flatfield_recovery_correlation(self, synthetic_stack: tuple[np.ndarray, np.ndarray]) -> None:
        """The estimated flat-field correlates > 0.9 with the ground truth."""
        stack, flatfield = synthetic_stack
        xp = get_xp(Backend.NUMPY)
        Ib, _, _, _ = inexact_alm_l1(stack, l_s=0.5, l_d=0.2, max_iter=200, estimate_darkfield=False, xp=xp)
        estimated_ff = Ib.mean(axis=0).ravel()
        gt = flatfield.ravel()
        correlation = float(np.corrcoef(estimated_ff, gt)[0, 1])
        assert correlation > 0.9, f"Flat-field correlation too low: {correlation:.3f}"

    def test_no_nan_in_output(self, synthetic_stack: tuple[np.ndarray, np.ndarray]) -> None:
        """No NaN or Inf values in any output array."""
        stack, _ = synthetic_stack
        xp = get_xp(Backend.NUMPY)
        Ib, Ir, D, _ = inexact_alm_l1(stack, l_s=0.5, l_d=0.2, max_iter=5, xp=xp)
        for name, arr in [("Ib", Ib), ("Ir", Ir), ("D", D)]:
            assert np.isfinite(arr).all(), f"Non-finite values in {name}"

    def test_darkfield_disabled_is_zero(self, synthetic_stack: tuple[np.ndarray, np.ndarray]) -> None:
        """Dark-field output should be near-zero when estimation is disabled."""
        stack, _ = synthetic_stack
        xp = get_xp(Backend.NUMPY)
        _, _, D, _ = inexact_alm_l1(stack, l_s=0.5, l_d=0.2, max_iter=5, estimate_darkfield=False, xp=xp)
        np.testing.assert_allclose(D, 0.0, atol=1e-6)

    def test_residual_magnitude(self, synthetic_stack: tuple[np.ndarray, np.ndarray]) -> None:
        """The sparse residual should be smaller than the flat-field mean."""
        stack, _ = synthetic_stack
        xp = get_xp(Backend.NUMPY)
        Ib, Ir, _, _ = inexact_alm_l1(stack, l_s=0.5, l_d=0.2, max_iter=200, xp=xp)
        assert float(np.abs(Ir).mean()) < float(np.abs(Ib).mean())


class TestConvergenceCheckEvery:
    def test_basic_default_is_none(self, synthetic_stack: tuple[np.ndarray, np.ndarray]) -> None:
        stack, _ = synthetic_stack
        model = BaSiC(stack)
        assert model.convergence_check_every is None

    def test_basic_forwards_convergence_check_every(
        self,
        synthetic_stack: tuple[np.ndarray, np.ndarray],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stack, _ = synthetic_stack
        captured: dict[str, object] = {}

        def _fake_alm(*_args, **kwargs):
            captured["convergence_check_every"] = kwargs.get("convergence_check_every")
            _n, h, w = stack.shape
            return stack.copy(), np.zeros_like(stack), np.zeros((1, h * w), dtype=np.float32), None

        monkeypatch.setattr("linum_basic.core.inexact_alm_l1", _fake_alm)
        model = BaSiC(stack, estimate_darkfield=False)
        model.working_size = 32
        model.convergence_check_every = 20
        model.run()
        assert captured.get("convergence_check_every") == 20


class TestDctKernelTuningLever:
    @pytest.fixture(autouse=True)
    def _clear_alm_step_cache(self) -> Iterator[None]:
        from linum_basic._alm import _ALM_STEP_CACHE

        _ALM_STEP_CACHE.clear()
        yield
        _ALM_STEP_CACHE.clear()

    def test_dct_kernel_tuning_matches_default_within_tolerance(
        self,
        synthetic_stack: tuple[np.ndarray, np.ndarray],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("torch")
        import torch

        from linum_basic._alm import _ALM_STEP_CACHE

        monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)

        stack, _ = synthetic_stack
        xp = get_xp(Backend.TORCH, "cpu")

        Ib_default, Ir_default, D_default, _ = inexact_alm_l1(stack, l_s=0.5, l_d=0.2, max_iter=10, xp=xp)

        monkeypatch.setenv("LINUM_BASIC_DCT_KERNEL", "tuned")
        _ALM_STEP_CACHE.clear()
        Ib_tuned, Ir_tuned, D_tuned, _ = inexact_alm_l1(stack, l_s=0.5, l_d=0.2, max_iter=10, xp=xp)

        np.testing.assert_allclose(Ib_default, Ib_tuned, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(Ir_default, Ir_tuned, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(D_default, D_tuned, rtol=1e-4, atol=1e-4)
