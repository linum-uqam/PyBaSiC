"""Cross-backend parity tests for :mod:`pybasic.backend`.

Compares NumPy and Torch (CPU) outputs for DCT, norm, SVD, and the full
ALM loop.  All Torch tests are skipped when PyTorch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybasic.backend import ArrayNamespace, Backend, get_xp

torch = pytest.importorskip("torch", reason="PyTorch not installed — skipping backend parity tests.")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a seeded random-number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def rand2d(rng: np.random.Generator) -> np.ndarray:
    """Return a random 16x16 float32 matrix."""
    return rng.random((16, 16)).astype(np.float32)


@pytest.fixture
def rand3d(rng: np.random.Generator) -> np.ndarray:
    """Return a random 4x16x16 float32 array (small image stack)."""
    return rng.random((4, 16, 16)).astype(np.float32)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

ATOL = 1e-4


def _np(xp: ArrayNamespace, arr: object) -> np.ndarray:
    """Convert *arr* to NumPy using *xp.to_numpy*."""
    return xp.to_numpy(arr)


# ---------------------------------------------------------------------------
# Backend creation
# ---------------------------------------------------------------------------


class TestGetXp:
    """Tests for :func:`pybasic.backend.get_xp`."""

    def test_numpy_backend(self) -> None:
        """``get_xp('numpy')`` returns a NumPy namespace."""
        xp = get_xp(Backend.NUMPY)
        arr = xp.zeros((3, 3))
        assert isinstance(xp.to_numpy(arr), np.ndarray)

    def test_torch_backend(self) -> None:
        """``get_xp('torch')`` returns a Torch namespace."""
        xp = get_xp(Backend.TORCH)
        arr = xp.zeros((3, 3))
        assert xp.to_numpy(arr) is not None

    def test_auto_returns_namespace(self) -> None:
        """``get_xp('auto')`` returns a valid namespace without raising."""
        xp = get_xp("auto")
        arr = xp.ones((2, 2))
        assert xp.to_numpy(arr).shape == (2, 2)


# ---------------------------------------------------------------------------
# Element-wise ops
# ---------------------------------------------------------------------------


class TestElementWise:
    """Verify element-wise operations agree across backends."""

    @pytest.mark.parametrize("val", [-1.5, 0.0, 2.3])
    def test_abs_scalar(self, val: float) -> None:
        """Absolute value agrees for positive, zero, and negative scalars."""
        x = np.array([val], dtype=np.float32)
        xp_np = get_xp(Backend.NUMPY)
        xp_th = get_xp(Backend.TORCH)
        np.testing.assert_allclose(
            _np(xp_np, xp_np.abs(xp_np.asarray(x))),
            _np(xp_th, xp_th.abs(xp_th.asarray(x))),
            atol=ATOL,
        )

    def test_sign_array(self, rand2d: np.ndarray) -> None:
        """Sign function agrees element-wise."""
        xp_np = get_xp(Backend.NUMPY)
        xp_th = get_xp(Backend.TORCH)
        np.testing.assert_allclose(
            _np(xp_np, xp_np.sign(xp_np.asarray(rand2d))),
            _np(xp_th, xp_th.sign(xp_th.asarray(rand2d))),
            atol=ATOL,
        )

    def test_maximum_scalar(self, rand2d: np.ndarray) -> None:
        """maximum(x, 0) clamps negatives to zero on both backends."""
        x = rand2d - 0.5  # mix of pos/neg
        xp_np = get_xp(Backend.NUMPY)
        xp_th = get_xp(Backend.TORCH)
        np.testing.assert_allclose(
            _np(xp_np, xp_np.maximum(xp_np.asarray(x), 0.0)),
            _np(xp_th, xp_th.maximum(xp_th.asarray(x), 0.0)),
            atol=ATOL,
        )

    def test_minimum_scalar(self, rand2d: np.ndarray) -> None:
        """minimum(x, 0) clamps positives to zero on both backends."""
        x = rand2d - 0.5
        xp_np = get_xp(Backend.NUMPY)
        xp_th = get_xp(Backend.TORCH)
        np.testing.assert_allclose(
            _np(xp_np, xp_np.minimum(xp_np.asarray(x), 0.0)),
            _np(xp_th, xp_th.minimum(xp_th.asarray(x), 0.0)),
            atol=ATOL,
        )


# ---------------------------------------------------------------------------
# Norms & SVD
# ---------------------------------------------------------------------------


class TestLinearAlgebra:
    """Verify linear-algebra helpers agree across backends."""

    def test_frobenius_norm(self, rand2d: np.ndarray) -> None:
        """Frobenius norms agree to within tolerance."""
        xp_np = get_xp(Backend.NUMPY)
        xp_th = get_xp(Backend.TORCH)
        a = xp_np.norm_fro(xp_np.asarray(rand2d))
        b = xp_th.norm_fro(xp_th.asarray(rand2d))
        assert abs(a - b) < ATOL, f"Frobenius norm mismatch: {a} vs {b}"

    def test_svd_leading_singular(self, rand2d: np.ndarray) -> None:
        """Leading singular value agrees to within tolerance."""
        xp_np = get_xp(Backend.NUMPY)
        xp_th = get_xp(Backend.TORCH)
        s_np = xp_np.svd_leading_singular(xp_np.asarray(rand2d))
        s_th = xp_th.svd_leading_singular(xp_th.asarray(rand2d))
        assert abs(s_np - s_th) < ATOL, f"Leading σ mismatch: {s_np} vs {s_th}"


# ---------------------------------------------------------------------------
# DCT parity
# ---------------------------------------------------------------------------


class TestDCTParity:
    """NumPy vs Torch DCT-II / DCT-III parity."""

    def test_dctn_2d(self, rand2d: np.ndarray) -> None:
        """2-D DCT-II output agrees to within atol=1e-4."""
        xp_np = get_xp(Backend.NUMPY)
        xp_th = get_xp(Backend.TORCH)
        out_np = _np(xp_np, xp_np.dctn(xp_np.asarray(rand2d)))
        out_th = _np(xp_th, xp_th.dctn(xp_th.asarray(rand2d)))
        np.testing.assert_allclose(out_np, out_th, atol=ATOL)

    def test_idctn_2d(self, rand2d: np.ndarray) -> None:
        """2-D inverse DCT output agrees to within atol=1e-4."""
        xp_np = get_xp(Backend.NUMPY)
        xp_th = get_xp(Backend.TORCH)
        out_np = _np(xp_np, xp_np.idctn(xp_np.asarray(rand2d)))
        out_th = _np(xp_th, xp_th.idctn(xp_th.asarray(rand2d)))
        np.testing.assert_allclose(out_np, out_th, atol=ATOL)

    def test_dctn_roundtrip(self, rand2d: np.ndarray) -> None:
        """IDCT(DCT(x)) ≈ x for both backends."""
        for backend in (Backend.NUMPY, Backend.TORCH):
            xp = get_xp(backend)
            x = xp.asarray(rand2d)
            reconstructed = _np(xp, xp.idctn(xp.dctn(x)))
            np.testing.assert_allclose(reconstructed, rand2d, atol=ATOL)

    def test_dctn_3d(self, rand3d: np.ndarray) -> None:
        """3-D DCT-II output agrees for a small stack."""
        xp_np = get_xp(Backend.NUMPY)
        xp_th = get_xp(Backend.TORCH)
        out_np = _np(xp_np, xp_np.dctn(xp_np.asarray(rand3d)))
        out_th = _np(xp_th, xp_th.dctn(xp_th.asarray(rand3d)))
        np.testing.assert_allclose(out_np, out_th, atol=ATOL)


# ---------------------------------------------------------------------------
# ALM parity (NumPy vs Torch CPU)
# ---------------------------------------------------------------------------


class TestAlmParity:
    """Full ALM loop output agrees between NumPy and Torch backends."""

    def test_alm_ib_agrees(self, rand3d: np.ndarray) -> None:
        """Flat-field estimate from NumPy and Torch agree to atol=1e-4."""
        from pybasic._alm import inexact_alm_l1

        xp_np = get_xp(Backend.NUMPY)
        xp_th = get_xp(Backend.TORCH)

        Ib_np, _, _ = inexact_alm_l1(rand3d, xp=xp_np, l_s=0.3, l_d=0.1, max_iter=5, estimate_darkfield=False)
        Ib_th, _, _ = inexact_alm_l1(rand3d, xp=xp_th, l_s=0.3, l_d=0.1, max_iter=5, estimate_darkfield=False)

        np.testing.assert_allclose(Ib_np, Ib_th, atol=1e-4)
