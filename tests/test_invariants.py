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
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from linum_basic import _alm
from linum_basic._alm import _ALM_STEP_CACHE, _build_alm_step
from linum_basic._alm import shrink as real_shrink
from linum_basic.backend import Backend, get_xp

# Return-tuple index map of ``_alm_core_step`` (kept in sync with ``_alm.py``):
#   0:new_sf 1:new_s_spatial 2:new_ib 3:new_ir 4:d_minus_ir
#   5:new_b  6:new_r_mean_all 7:d_y    8:new_d_field 9:new_b1
_IDX_NEW_D_FIELD = 8
_IDX_NEW_B1 = 9


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
