"""Unit tests for dark-field estimation in the BaSiC solver.

These tests exercise the DC (mean) preservation fix in the ALM dark-field
path and verify basic shape-correlation quality on small synthetic stacks.
Vignettes are generated with the sbh-simulator Python API (Gaussian family).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from linum_basic.core import BaSiC

try:
    from sbh_simulator.simulator import (
        generate_gaussian_darkfield,
        generate_gaussian_vignette,
    )

    _SBH_SIMULATOR_AVAILABLE = True
except ImportError:
    _SBH_SIMULATOR_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _SBH_SIMULATOR_AVAILABLE,
    reason="sbh-simulator not installed. Install with: uv pip install sbh-simulator",
)


def _make_stack(
    n: int,
    size: int,
    flat_field: np.ndarray,
    dark_field: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a stack of ``n`` images following the BaSiC model.

    Each image follows ``D(x,t) = S(x) · B(t) + D_field(x)`` with a
    per-image scalar brightness ``B(t)`` drawn uniformly from ``[0.2, 0.8]``.
    Using a scalar-per-image B (rather than per-pixel noise) ensures the stack
    matches the separability assumption BaSiC relies on, giving the dark-field
    estimator a clean signal to recover.
    """
    B = rng.uniform(0.2, 0.8, (n, 1, 1)).astype(np.float32)  # per-image scalar
    stack = flat_field[None] * B + dark_field[None]
    # No clipping: BaSiC operates on arbitrary positive-valued images.
    # Clipping would violate the multiplicative model and corrupt dark-field recovery.
    return stack.astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_darkfield_dc_magnitude_preserved() -> None:
    """Estimated dark-field retains its DC (mean) component.

    Regression test for a bug where ``A1_offset.mean()`` was subtracted from
    the dark-field input before DCT shrinkage.  This caused the estimated
    dark-field to be near-zero (~100x too small) for images with a
    significant constant additive offset, even when the spatial shape was
    partially correct.

    The test passes a stack whose only dark-field is a constant offset of
    ``true_df_mean``.  A healthy estimator should recover at least 25 % of
    that offset in the mean of the estimated dark-field.
    """
    rng = np.random.default_rng(0)
    size = 32
    true_df_mean = 0.04

    # Slightly non-uniform flat-field so BaSiC has spatial contrast to work with
    flat_field = generate_gaussian_vignette(width=size, height=size, sigma=0.7, rng=random.Random(0)).astype(np.float32)
    flat_field /= flat_field.mean()
    dark_field = np.full((size, size), true_df_mean, dtype=np.float32)

    stack = _make_stack(30, size, flat_field, dark_field, rng=rng)

    model = BaSiC(stack, estimate_darkfield=True)
    model.prepare()
    model.run()

    estimated_mean = float(model.darkfield_fullsize.mean())
    assert estimated_mean > true_df_mean * 0.25, (
        f"Estimated dark-field mean {estimated_mean:.5f} is too small; "
        f"expected > {true_df_mean * 0.25:.5f} (true mean = {true_df_mean:.4f}). "
        "The A1_offset DC-stripping bug may have been re-introduced."
    )


def test_darkfield_sensitivity_above_baseline() -> None:
    """Estimated dark-field mean is larger when a dark-field is present than when it is absent.

    The BaSiC dark-field estimator is a heuristic based on the residuals of
    dimmer images; it cannot reliably recover the absolute magnitude or spatial
    shape of the dark-field in all cases.  However, it should consistently
    estimate a LARGER mean when a non-trivial dark-field exists than when one
    is absent.  This is the minimum meaningful "signal present / absent"
    sensitivity requirement and is sufficient to catch the DC-stripping
    regression (which would collapse both estimates to near zero).
    """
    rng = np.random.default_rng(0)
    size = 32

    flat_field = generate_gaussian_vignette(width=size, height=size, sigma=0.7, rng=random.Random(0)).astype(np.float32)
    flat_field /= flat_field.mean()

    B = rng.uniform(0.2, 0.8, (30, 1, 1)).astype(np.float32)
    stack_no_df = flat_field[None] * B
    stack_with_df = flat_field[None] * B + 0.04  # constant 0.04 dark-field

    model_no = BaSiC(stack_no_df, estimate_darkfield=True)
    model_no.prepare()
    model_no.run()

    model_yes = BaSiC(stack_with_df, estimate_darkfield=True)
    model_yes.prepare()
    model_yes.run()

    mean_no = float(model_no.darkfield_fullsize.mean())
    mean_yes = float(model_yes.darkfield_fullsize.mean())

    assert mean_yes > mean_no + 0.005, (
        f"DF estimate with dark-field present (mean={mean_yes:.5f}) is not "
        f"meaningfully larger than without (mean={mean_no:.5f}).  "
        "The estimator appears insensitive to the dark-field signal."
    )


def test_darkfield_does_not_degrade_flatfield() -> None:
    """Enabling dark-field estimation does not significantly degrade flat-field recovery.

    The flat-field and dark-field are jointly estimated in a single optimisation
    pass.  This test guards against regressions where the dark-field coupling
    corrupts the flat-field estimate.  The flat-field Pearson r with dark-field
    estimation enabled must be within 0.10 of the baseline without it.
    """
    rng = np.random.default_rng(0)
    size = 32

    flat_field = generate_gaussian_vignette(width=size, height=size, sigma=0.7, rng=random.Random(0)).astype(np.float32)
    flat_field /= flat_field.mean()
    dark_field = generate_gaussian_darkfield(width=size, height=size, sigma=0.5, max_offset=0.05, rng=random.Random(1)).astype(
        np.float32
    )

    stack = _make_stack(30, size, flat_field, dark_field, rng=rng)

    model_no_df = BaSiC(stack, estimate_darkfield=False)
    model_no_df.prepare()
    model_no_df.run()
    r_no_df = float(np.corrcoef(flat_field.ravel(), model_no_df.flatfield_fullsize.ravel())[0, 1])

    model_df = BaSiC(stack, estimate_darkfield=True)
    model_df.prepare()
    model_df.run()
    r_df = float(np.corrcoef(flat_field.ravel(), model_df.flatfield_fullsize.ravel())[0, 1])

    assert r_df > r_no_df - 0.10, (
        f"Flat-field r dropped from {r_no_df:.3f} (no dark-field) to {r_df:.3f} "
        "(with dark-field), a degradation of more than 0.10."
    )
