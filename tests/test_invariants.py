"""Direct regression tests for the load-bearing BaSiC algorithm invariants.

These tests pin the exact *behaviour* of the numerical invariants documented in
``CONCERNS.md`` and the ``AGENTS.md`` invariants table.  They are designed so
that a future refactor that silently breaks an invariant (e.g. replacing the B1
guard with a naive clamp, or dropping the second/spatial dark-field shrink) is
caught directly by CI rather than only surfacing as a fuzzy correlation drift
three layers away in an end-to-end recovery test.

Importantly, these tests do **not** change any invariant numerics — they only
observe and assert on the existing behaviour of
:func:`linum_basic._alm._build_alm_step`'s ``_alm_core_step`` closure.

Covered invariants (T01):

* **B1 monotonicity guard** — ``new_b1 = xp.where(b1_new > 0.0, b1_new, b1)``
  preserves the previous positive B1 estimate when the candidate would be
  non-positive, instead of collapsing to zero.
* **Eq. 6 dual DCT+spatial shrink** — the second (spatial) soft-threshold of
  the dark-field residual is load-bearing: ablating it measurably changes the
  estimated ``D_field``.

Covered invariants (T02):

* **OpenCV double-transpose resize** — ``core.py`` resizes every image with
  ``cv2.resize(img.T, new_shape, ...).T``.  The double transpose matches
  OpenCV's ``(width, height)`` column-major convention; removing it introduces
  ~1e-7 float drift that cascades through the iterative solver.  The drift is
  too small to assert against reliably, so the test pins the *call contract*:
  ``cv2.resize`` must always receive the transposed array, never the plain one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import linum_basic.core as core
from linum_basic import _alm
from linum_basic._alm import _ALM_STEP_CACHE, _build_alm_step
from linum_basic._alm import shrink as real_shrink
from linum_basic.backend import Backend, get_xp
from linum_basic.core import BaSiC

# Return-tuple index map of ``_alm_core_step`` (kept in sync with ``_alm.py``):
#   0:new_sf 1:new_s_spatial 2:new_ib 3:new_ir 4:d_minus_ir
#   5:new_b  6:new_r_mean_all 7:d_y    8:new_d_field 9:new_b1
_IDX_NEW_D_FIELD = 8
_IDX_NEW_B1 = 9

# Cross-backend agreement tolerance, matching ``tests/test_backend_parity.py``
# so the σ₁ agreement assertion is consistent with the established parity gate.
ATOL = 1e-4


@pytest.fixture(autouse=True)
def _reset_alm_step_cache() -> None:
    """Ensure each test builds a fresh ``_alm_core_step`` closure.

    ``_build_alm_step`` caches its closure by
    ``(backend, device, n, p, q, l_s, estimate_darkfield, l_d, ...)`` but the
    cache key excludes ``b1_uplimit`` and ``ent2`` (they are captured in the
    closure).  Clearing between tests prevents a stale closure with the wrong
    captured scalars from masking a regression.
    """
    _ALM_STEP_CACHE.clear()
    yield
    _ALM_STEP_CACHE.clear()


def _default_inputs(xp: Any, n: int = 6, p: int = 8, q: int = 8) -> dict[str, Any]:
    """Return the standard ALM step inputs for a small synthetic stack.

    Uses a NumPy ``default_rng`` with a fixed seed so the discriminating
    behaviour (negative B1 cap, non-trivial dark-field residual) is
    deterministic across runs and platforms.
    """
    pq = p * q
    rng = np.random.default_rng(7)
    data = np.clip(rng.normal(1.0, 0.05, (n, pq)), 0.01, None).astype(np.float32)
    return {
        "d": xp.asarray(data),
        "s_spatial": xp.asarray(np.ones((1, pq), dtype=np.float32)),
        "b": xp.asarray(np.ones((n, 1), dtype=np.float32)),
        "d_field": xp.asarray(np.zeros((1, pq), dtype=np.float32)),
        "y": xp.asarray(np.zeros((n, pq), dtype=np.float64)),
        "w": xp.asarray(np.ones((n, pq), dtype=np.float32)),
        "mu": xp.asarray(np.array(0.1, dtype=np.float32)),
    }


# ---------------------------------------------------------------------------
# B1 monotonicity guard
# ---------------------------------------------------------------------------


class TestB1MonotonicityGuard:
    """Pin the B1 guard ``new_b1 = xp.where(b1_new > 0.0, b1_new, b1)``.

    CONCER / AGENTS rationale: B1 oscillates between a valid positive estimate
    (~0.09) and a non-positive candidate on alternate iterations.  The guard
    keeps the previous positive B1 instead of resetting it to zero.  Replacing
    it with ``B1 = max(0.0, min(...))`` silently drops the estimate and breaks
    the dark-field regression behaviour (notably on Linux).
    """

    def test_guard_preserves_previous_positive_estimate_when_candidate_negative(
        self,
    ) -> None:
        """Under a negative candidate cap, the guard keeps ``b1_prev``.

        We force the candidate cap ``b1_uplimit / (s_mean + eps)`` to be
        negative by passing a negative ``b1_uplimit``.  With a positive
        ``b1_prev``:

        * the invariant guard returns ``b1_prev`` (0.1), and
        * a naive ``max(0.0, candidate)`` clamp would return 0.0, and
        * using the candidate directly would return the negative cap.

        Asserting ``new_b1 == b1_prev`` therefore discriminates the invariant
        from both naive regressions.
        """
        xp = get_xp(Backend.NUMPY)
        n, p, q = 6, 8, 8
        inputs = _default_inputs(xp, n, p, q)
        l_s, l_d, ent2 = 0.5, 0.2, 10.0
        b1_prev = xp.asarray(np.array(0.1, dtype=np.float32))
        b1_uplimit = -0.5  # negative → forces a negative candidate cap

        step = _build_alm_step(
            xp,
            n,
            p,
            q,
            l_s,
            estimate_darkfield=True,
            l_d=l_d,
            ent2=ent2,
            b1_uplimit=b1_uplimit,
        )
        out = step(
            inputs["d"],
            inputs["s_spatial"],
            inputs["b"],
            inputs["d_field"],
            inputs["y"],
            inputs["w"],
            inputs["mu"],
            b1_prev,
        )
        new_b1 = float(xp.to_numpy(out[_IDX_NEW_B1]))

        # Reconstruct the candidate cap to prove the scenario is discriminating.
        s_mean = float(xp.to_numpy(xp.mean(inputs["s_spatial"])))
        candidate_cap = b1_uplimit / (s_mean + 1e-9)

        # The cap must be negative, otherwise the test would not exercise the
        # guard branch (a positive cap could pass through without triggering it).
        assert candidate_cap < 0.0, (
            f"Test fixture non-discriminating: candidate cap {candidate_cap:.4f} "
            "must be negative to exercise the B1 guard branch"
        )
        # The guard must preserve the previous positive estimate rather than
        # collapsing to the negative candidate (direct use) or to 0 (naive clamp).
        assert np.isfinite(new_b1), f"new_b1 not finite: {new_b1}"
        assert new_b1 == pytest.approx(0.1, abs=1e-6), (
            f"B1 guard failed to preserve previous positive estimate: "
            f"expected b1_prev=0.1, got new_b1={new_b1:.6f} "
            f"(candidate cap was {candidate_cap:.4f})"
        )

    def test_guard_branch_is_discriminating_from_naive_clamp(self) -> None:
        """A naive ``max(0, candidate)`` would yield 0.0, not ``b1_prev``.

        This companion assertion documents *why* the guard matters: the naive
        clamp (the exact regression the invariant guards against) produces a
        measurably different result on the same fixture, so the main test
        cannot pass under a naive-clamp reimplementation.
        """
        xp = get_xp(Backend.NUMPY)
        n, p, q = 6, 8, 8
        inputs = _default_inputs(xp, n, p, q)
        l_s, l_d, ent2 = 0.5, 0.2, 10.0
        b1_prev = xp.asarray(np.array(0.1, dtype=np.float32))
        b1_uplimit = -0.5

        step = _build_alm_step(
            xp,
            n,
            p,
            q,
            l_s,
            estimate_darkfield=True,
            l_d=l_d,
            ent2=ent2,
            b1_uplimit=b1_uplimit,
        )
        out = step(
            inputs["d"],
            inputs["s_spatial"],
            inputs["b"],
            inputs["d_field"],
            inputs["y"],
            inputs["w"],
            inputs["mu"],
            b1_prev,
        )
        new_b1 = float(xp.to_numpy(out[_IDX_NEW_B1]))

        # The naive clamp the invariant protects against: max(0, b1_prev) is
        # only equal to b1_prev when b1_prev > 0, but max(0, candidate) where
        # candidate is the negative cap yields 0.0 — strictly less than the
        # guard's 0.1.  Confirm the guard's output is not the naive zero.
        assert new_b1 > 0.0, (
            f"B1 collapsed to <= 0; the guard should have preserved the positive b1_prev=0.1, got new_b1={new_b1:.6f}"
        )


# ---------------------------------------------------------------------------
# Eq. 6 dual DCT + spatial shrink (dark-field path)
# ---------------------------------------------------------------------------


class TestDarkfieldDualShrink:
    """Pin the second (spatial) shrink in the dark-field update.

    CONCERNS / AGENTS rationale: the dark-field residual is soft-thresholded
    twice — once in the DCT domain and once in the spatial domain — to
    implement Eq. 6's dual ``‖F(DR)‖_1 + ‖DR‖_1`` penalty.  Removing the
    spatial shrink changes the estimated flat-field.

    The test ablates *only* the ``(1, P*Q)``-shaped shrink call (the spatial
    dark-field shrink) and asserts the estimated ``D_field`` measurably
    changes, proving the spatial shrink is load-bearing rather than dead code.
    """

    @staticmethod
    def _spatial_gradient_stack(n: int, p: int, q: int, seed: int) -> np.ndarray:
        """A stack with real spatial structure so the dark-field residual is non-trivial.

        A uniform stack yields a near-zero dark-field residual that the shrink
        operator leaves untouched, making ablation undetectable.  A bilinear
        spatial gradient with per-image multiplicative variation produces a
        non-uniform residual that the spatial shrink actively shapes.
        """
        rng = np.random.default_rng(seed)
        xs, ys = np.meshgrid(np.linspace(0.7, 1.3, q), np.linspace(0.85, 1.15, p), indexing="xy")
        base = (xs * ys).astype(np.float32)
        img_factors = rng.uniform(0.8, 1.2, n).astype(np.float32)
        stack = base[None] * img_factors[:, None, None]
        noise = rng.normal(0, 0.04, (n, p, q)).astype(np.float32)
        mask = rng.random((n, p, q)) < 0.06
        stack = stack + noise * mask
        return np.clip(stack, 0.01, None).astype(np.float32)

    def test_spatial_shrink_is_load_bearing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ablating the ``(1, P*Q)`` spatial shrink measurably changes ``D_field``.

        The spatial shrink is the unique ``shrink`` call whose input has shape
        ``(1, P*Q)`` (the DCT-domain dark-field shrink operates on the ``(P, Q)``
        reshape, and the flat-field/residual shrinks operate on ``(N, P*Q)`` /
        ``(P, Q)``).  Neutralising only that call must change the output, which
        proves the dual DCT+spatial penalty of Eq. 6 is in effect.
        """
        xp = get_xp(Backend.NUMPY)
        n, p, q = 6, 8, 8
        pq = p * q
        stack = self._spatial_gradient_stack(n, p, q, seed=1)
        data = stack.reshape(n, pq)

        d = xp.asarray(data)
        s_spatial = xp.asarray(np.ones((1, pq), dtype=np.float32))
        b = xp.asarray(np.ones((n, 1), dtype=np.float32))
        d_field = xp.asarray(np.zeros((1, pq), dtype=np.float32))
        y = xp.asarray(np.zeros((n, pq), dtype=np.float64))
        w = xp.asarray(np.ones((n, pq), dtype=np.float32))
        mu = xp.asarray(np.array(0.1, dtype=np.float32))
        b1_prev = xp.asarray(np.array(0.0, dtype=np.float32))

        l_s, l_d, ent2 = 0.5, 0.2, 10.0
        b1_uplimit = float(xp.to_numpy(xp.min(d)))  # realistic positive cap

        step = _build_alm_step(
            xp,
            n,
            p,
            q,
            l_s,
            estimate_darkfield=True,
            l_d=l_d,
            ent2=ent2,
            b1_uplimit=b1_uplimit,
        )

        # Reference dark-field with the real dual shrink in place.
        out_ref = step(d, s_spatial, b, d_field, y, w, mu, b1_prev)
        d_field_ref = xp.to_numpy(out_ref[_IDX_NEW_D_FIELD])

        # Ablate only the spatial (1, P*Q) shrink: pass it through un-thresholded.
        def _spatial_passthrough_shrink(xp_: Any, theta: Any, epsilon: float = 1e-3) -> Any:
            if tuple(xp_.to_numpy(theta).shape) == (1, pq):
                return theta
            return real_shrink(xp_, theta, epsilon)

        monkeypatch.setattr(_alm, "shrink", _spatial_passthrough_shrink)
        out_abl = step(d, s_spatial, b, d_field, y, w, mu, b1_prev)
        d_field_abl = xp.to_numpy(out_abl[_IDX_NEW_D_FIELD])

        # Both estimates must remain finite (ablation must not introduce NaN/Inf).
        assert np.isfinite(d_field_ref).all(), "Reference D_field has non-finite values"
        assert np.isfinite(d_field_abl).all(), "Ablated D_field has non-finite values"

        # The spatial shrink must measurably shape the dark-field.  Empirically
        # the L2 difference on this fixture is ~1.0; 1e-3 is a conservative
        # floor that still reliably detects "shrink removed" (difference ~0).
        diff = float(np.linalg.norm(d_field_ref - d_field_abl))
        assert diff > 1e-3, (
            "Spatial dark-field shrink appears to be dead code: ablating it did "
            f"not change D_field (||D_ref - D_abl||_2 = {diff:.3e})"
        )

    def test_spatial_shrink_identified_by_unique_call_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Confirm the ``(1, P*Q)`` shrink is actually invoked in the step.

        A defensive companion to the ablation test: if a future refactor moves
        or removes the spatial shrink entirely, this assertion fails directly
        (no ``(1, P*Q)`` shrink call observed) rather than relying on the
        ablation difference alone.
        """
        xp = get_xp(Backend.NUMPY)
        n, p, q = 6, 8, 8
        pq = p * q
        stack = self._spatial_gradient_stack(n, p, q, seed=1)
        data = stack.reshape(n, pq)

        d = xp.asarray(data)
        s_spatial = xp.asarray(np.ones((1, pq), dtype=np.float32))
        b = xp.asarray(np.ones((n, 1), dtype=np.float32))
        d_field = xp.asarray(np.zeros((1, pq), dtype=np.float32))
        y = xp.asarray(np.zeros((n, pq), dtype=np.float64))
        w = xp.asarray(np.ones((n, pq), dtype=np.float32))
        mu = xp.asarray(np.array(0.1, dtype=np.float32))
        b1_prev = xp.asarray(np.array(0.0, dtype=np.float32))

        l_s, l_d, ent2 = 0.5, 0.2, 10.0
        b1_uplimit = float(xp.to_numpy(xp.min(d)))

        step = _build_alm_step(
            xp,
            n,
            p,
            q,
            l_s,
            estimate_darkfield=True,
            l_d=l_d,
            ent2=ent2,
            b1_uplimit=b1_uplimit,
        )

        observed_shapes: list[tuple[int, ...]] = []

        def _recording_shrink(xp_: Any, theta: Any, epsilon: float = 1e-3) -> Any:
            observed_shapes.append(tuple(xp_.to_numpy(theta).shape))
            return real_shrink(xp_, theta, epsilon)

        monkeypatch.setattr(_alm, "shrink", _recording_shrink)
        step(d, s_spatial, b, d_field, y, w, mu, b1_prev)

        assert (1, pq) in observed_shapes, (
            "No spatial (1, P*Q) dark-field shrink call observed; the Eq. 6 dual "
            f"penalty may have been removed. Observed shrink shapes: {observed_shapes}"
        )


# ---------------------------------------------------------------------------
# OpenCV double-transpose resize invariant (core._load_images)
# ---------------------------------------------------------------------------


class TestOpenCVDoubleTransposeResize:
    """Pin the ``cv2.resize(img.T, new_shape, ...).T`` double transpose.

    CONCERNS / AGENTS rationale: ``core.py`` resizes every image with the
    ``.T`` applied before *and* after ``cv2.resize`` to match OpenCV's
    ``(width, height)`` column-major convention.  Removing either transpose
    introduces ~1e-7 float drift that cascades through the iterative solver.
    That drift is too small to assert against reliably in the *output*, so
    the test pins the *call contract*: ``cv2.resize``'s source argument must
    be the transposed array, never the plain one.
    """

    @staticmethod
    def _asymmetric_non_square_stack(n: int = 3, h: int = 7, w: int = 11) -> np.ndarray:
        """A non-square, asymmetric-content image stack.

        Two independent properties make the contract assertion discriminating:

        * **Non-square** (``h != w``): ``img`` has shape ``(h, w)`` while
          ``img.T`` has shape ``(w, h)``, so the transpose is detectable by
          shape alone.
        * **Asymmetric content** (an increasing ramp): even for a square
          image ``img != img.T`` element-wise, so the transpose is also
          detectable by content.

        Both axes guard against a future "simplification" of the fixture
        (e.g. a constant or symmetric image) that would silently make the
        main test non-discriminating.
        """
        assert h != w, "fixture must be non-square for shape-level discrimination"
        stack = np.empty((n, h, w), dtype=np.float32)
        for i in range(n):
            stack[i] = np.arange(h * w, dtype=np.float32).reshape(h, w) + i * 1000.0
        return stack

    def test_resize_receives_transposed_array(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``cv2.resize``'s source arg must equal ``img.T``, never ``img``.

        Spies on ``cv2.resize`` during ``BaSiC.prepare()`` and asserts every
        captured source array is the transpose of an input image.  If the
        leading ``.T`` is removed, the plain (non-transposed) image would be
        passed and this assertion fails directly — rather than only surfacing
        as ~1e-7 output drift three layers away in an end-to-end test.
        """
        n, h, w = 3, 7, 11
        stack = self._asymmetric_non_square_stack(n, h, w)

        captured: list[np.ndarray] = []
        real_resize = core.cv2.resize

        def _spy_resize(src: np.ndarray, dsize: tuple[int, int], **kwargs: Any) -> np.ndarray:
            captured.append(np.asarray(src).copy())
            return real_resize(src, dsize, **kwargs)

        monkeypatch.setattr(core.cv2, "resize", _spy_resize)

        model = BaSiC(stack, estimate_darkfield=False)
        model.working_size = 4  # < image_shape[0] → INTER_AREA downscale (production path)
        model.prepare()

        # Sanity: prepare completed and produced the expected resized shape.
        assert model.img_stack_resized.shape == (n, model.working_size, model.working_size)
        assert len(captured) == n, f"expected {n} resize calls (one per image), got {len(captured)}"

        expected_transposes = [stack[i].T for i in range(n)]
        plain_images = [stack[i] for i in range(n)]

        for src in captured:
            # Every source must be the transpose of some input image.
            assert any(np.array_equal(src, t) for t in expected_transposes), (
                "cv2.resize received an array that is not the transpose of any "
                f"input image; the leading '.T' before resize may be broken. "
                f"observed src.shape={src.shape}, expected {(w, h)}"
            )
            # The plain (non-transposed) image must never be passed.
            assert not any(np.array_equal(src, p) for p in plain_images), (
                "cv2.resize received the plain (non-transposed) image; the leading "
                "'.T' was removed, breaking OpenCV's (width, height) convention "
                "and introducing ~1e-7 drift that cascades through the solver."
            )

    def test_fixture_is_discriminating(self) -> None:
        """The fixture is non-square AND asymmetric, so discrimination is robust.

        Documents *why* the main test reliably catches removal of ``.T``: the
        plain image and its transpose differ on *both* shape (non-square) and
        content (asymmetric ramp).  If a future change makes the fixture
        square-and-symmetric, this guard fails, preventing the main contract
        assertion from silently becoming a no-op.
        """
        stack = self._asymmetric_non_square_stack(n=2, h=7, w=11)
        img = stack[0]
        assert img.shape == (7, 11), f"plain image shape must be (h, w)=(7, 11), got {img.shape}"
        assert img.T.shape == (11, 7), f"transposed shape must be (w, h)=(11, 7), got {img.T.shape}"
        assert not np.array_equal(img, img.T), (
            "fixture content is symmetric; img == img.T would make the contract "
            "assertion unable to distinguish the transposed from the plain input"
        )


# ---------------------------------------------------------------------------
# svd_leading_singular power-iteration invariant (backend.ArrayNamespace)
# ---------------------------------------------------------------------------


def _torch_available() -> bool:
    """Return ``True`` if PyTorch is importable.

    The other invariant tests above are NumPy-only and must keep running on
    torch-less environments, so the SVD power-iteration class is guarded with
    a per-class ``skipif`` rather than a module-level ``importorskip``.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _torch_available(),
    reason="PyTorch not installed — skipping SVD power-iteration invariant tests",
)
class TestSvdLeadingSingularPowerIteration:
    """Pin the ``svd_leading_singular`` backend split invariant.

    CONCERNS / AGENTS rationale: NumPy computes the leading singular value via
    a full ``numpy.linalg.svd(compute_uv=False)``; the Torch/GPU path instead
    uses batched power iteration on ``x @ x.T`` (see
    ``_svd_leading_singular_torch_batched``) to avoid a full GPU SVD and its
    device synchronisation.  Reverting the Torch path to ``torch.linalg.svd``
    reintroduces the sync and a ~0.001% σ₁ drift that shifts soft-threshold
    boundaries and breaks the darkfield regression tests.

    The invariant has two conjuncts, both pinned here:

    1. The Torch path **never** calls ``torch.linalg.svd`` / ``svdvals``.
    2. The power-iteration σ₁ agrees with NumPy's full-SVD σ₁ within ``ATOL``.
    """

    @staticmethod
    def _representative_matrix(seed: int = 7) -> np.ndarray:
        """A non-negative, image-like matrix with a genuine spectral gap.

        Mirrors the kind of matrix BaSiC feeds to ``svd_leading_singular``:
        the dark-field residual ``D`` is reshaped from sorted, mostly-positive
        image data, for which the power-iteration's deterministic uniform
        start converges quickly (see
        ``_svd_leading_singular_torch_batched`` docstring).  A pure-rank-1 or
        constant matrix would make the agreement assertion trivially pass; a
        matrix with no spectral gap would make 30 power-iteration steps
        meaningless.  This fixture has a moderate gap (verified by the
        companion ``test_fixture_has_genuine_spectral_gap``) so the agreement
        test both converges and stresses the iteration.
        """
        rng = np.random.default_rng(seed)
        return np.abs(rng.normal(1.0, 0.3, (16, 16))).astype(np.float32)

    def test_torch_path_never_calls_full_svd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Torch path must use power iteration, never a full SVD.

        Spies on ``torch.linalg.svd`` and ``torch.linalg.svdvals`` during a
        single ``svd_leading_singular`` call.  If a future change reverts the
        GPU path to ``torch.linalg.svd`` (the exact regression the invariant
        guards against), this assertion fails directly instead of only
        surfacing as ~0.001% σ₁ drift in a darkfield recovery test.
        """
        import torch

        xp = get_xp(Backend.TORCH)
        mat = self._representative_matrix()

        calls = {"svd": 0, "svdvals": 0}
        real_svd = torch.linalg.svd
        real_svdvals = torch.linalg.svdvals

        def _spy_svd(*args: Any, **kwargs: Any) -> Any:
            calls["svd"] += 1
            return real_svd(*args, **kwargs)

        def _spy_svdvals(*args: Any, **kwargs: Any) -> Any:
            calls["svdvals"] += 1
            return real_svdvals(*args, **kwargs)

        monkeypatch.setattr(torch.linalg, "svd", _spy_svd)
        monkeypatch.setattr(torch.linalg, "svdvals", _spy_svdvals)

        sigma = xp.svd_leading_singular(xp.asarray(mat))

        # The call must produce a finite leading singular value.
        assert np.isfinite(sigma), f"svd_leading_singular returned non-finite σ₁: {sigma}"

        # Neither full-SVD entry point may be invoked on the Torch path.
        assert calls["svd"] == 0, (
            "Torch svd_leading_singular called torch.linalg.svd ("
            f"{calls['svd']} time(s)); the power-iteration path may have been "
            "reverted to a full GPU SVD, reintroducing device sync and σ₁ drift."
        )
        assert calls["svdvals"] == 0, (
            "Torch svd_leading_singular called torch.linalg.svdvals ("
            f"{calls['svdvals']} time(s)); the power-iteration path may have "
            "been reverted to a full GPU SVD, reintroducing device sync and σ₁ drift."
        )

    def test_power_iteration_agrees_with_numpy_full_svd(self) -> None:
        """The power-iteration σ₁ agrees with NumPy's full-SVD σ₁ within ATOL.

        Complements conjunct 1: even if the path avoids ``torch.linalg.svd``, a
        broken iteration (wrong transpose, dropped normalisation, too few
        steps) would produce a σ₁ that disagrees with the NumPy reference.
        "ATOL`` matches ``tests/test_backend_parity.py`` so the invariant test
        is consistent with the established cross-backend parity gate.
        """
        xp_np = get_xp(Backend.NUMPY)
        xp_th = get_xp(Backend.TORCH)
        mat = self._representative_matrix()

        s_np = xp_np.svd_leading_singular(xp_np.asarray(mat))
        s_th = xp_th.svd_leading_singular(xp_th.asarray(mat))

        assert np.isfinite(s_np) and np.isfinite(s_th)
        assert abs(s_np - s_th) < ATOL, (
            f"Leading σ mismatch: numpy full SVD σ₁={s_np:.6f} vs "
            f"torch power-iteration σ₁={s_th:.6f} (Δ={abs(s_np - s_th):.3e}, "
            f"ATOL={ATOL}). The power-iteration math may be broken."
        )

    def test_fixture_has_genuine_spectral_gap(self) -> None:
        """The fixture has σ₁ clearly larger than σ₂, so the test is meaningful.

        Documents *why* the agreement test reliably catches a broken power
        iteration: with a genuine spectral gap (σ₂/σ₁ well below 1), 30
        iterations converge tightly to σ₁, so any broken iteration math
        produces a measurably wrong estimate.  If a future change makes the
        fixture near-rank-1 (σ₂≈0) or isotropic (σ₁≈σ₂), this guard fails,
        preventing the agreement assertion from silently weakening.
        """
        mat = self._representative_matrix()
        singular = np.linalg.svd(mat.astype(np.float64), compute_uv=False)
        sigma1, sigma2 = singular[0], singular[1]
        assert sigma1 > 0.0, "fixture has no signal (σ₁ = 0)"
        # A moderate gap: σ₂ no more than 80% of σ₁.  Empirically this fixture
        # sits around 0.55-0.70; 0.80 is a conservative ceiling.
        assert sigma2 / sigma1 < 0.80, (
            f"fixture spectral gap too small for a meaningful power-iteration test: σ₂/σ₁={sigma2 / sigma1:.3f} (need < 0.80)"
        )
        # And not rank-1 either, which would converge trivially in one step.
        assert sigma2 > 1e-3 * sigma1, (
            f"fixture is near-rank-1: σ₂/σ₁={sigma2 / sigma1:.3e}, which makes "
            "power iteration converge in ~1 step and weakens the agreement test."
        )
