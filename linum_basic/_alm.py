"""Inexact augmented Lagrangian method (ALM) for BaSiC shading estimation.

This module provides the core L1-minimisation solver used to decompose an
image stack into a smooth flat-field and a sparse residual.  All heavy
numerical work is dispatched through an :class:`~linum_basic.backend.ArrayNamespace`
so the same code can run on CPU (NumPy) or GPU (PyTorch).
"""

from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from linum_basic.backend import ArrayNamespace, Backend, get_xp

__all__ = ["inexact_alm_l1", "shrink"]


def shrink[ArrayT](xp: ArrayNamespace, theta: ArrayT, epsilon: float = 1e-3) -> ArrayT:
    """Scalar shrink (soft-threshold) operator.

    Computes ``sign(θ) · max(|θ| - ε, 0)`` element-wise, implemented as
    ``copysign(max(|θ| - ε, 0), θ)`` which avoids a separate sign array and
    is significantly faster for large arrays on the NumPy backend.

    Parameters
    ----------
    xp : ArrayNamespace
        Active array namespace (NumPy or Torch).
    theta : array
        Input array to shrink.
    epsilon : float
        Shrinkage threshold.

    Returns
    -------
    object
        Shrunk array with the same shape and backend as *theta*.

    References
    ----------
    .. [1] Candès, E., Li, X., Ma, Y. & Wright, J. "Robust Principal Component
       Analysis?" *J. ACM* 58, 1-37 (2011).
    """
    return xp.copysign(xp.maximum(xp.abs(theta) - epsilon, 0.0), theta)


def inexact_alm_l1(
    imgs: np.ndarray,
    l_s: float,
    l_d: float,
    *,
    tol: float = 1e-6,
    max_iter: int = 500,
    weight: np.ndarray | float = 1.0,
    estimate_darkfield: bool = True,
    rho: float = 1.5,
    verbose: bool = False,
    xp: ArrayNamespace | None = None,
    warm_start: dict | None = None,
    return_state: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict | None]:
    r"""L1 minimisation via the inexact augmented Lagrangian method.

    Decomposes a stack of *N* images into a low-rank flat-field component
    **Ib** and a sparse residual **Ir**, optionally estimating a dark-field
    correction **D_field**.

    The formulation follows Peng et al. (2017) and is adapted from the
    original MATLAB BaSiC implementation.

    Parameters
    ----------
    imgs : numpy.ndarray, shape (N, P, Q)
        Stack of *N* images each of spatial size *P x Q*.
    l_s : float
        Flat-field regularisation parameter.  Controls DCT-domain sparsity
        of the flat-field: larger values produce smoother flat-fields.
        Typically set automatically by
        :meth:`~linum_basic.core.BaSiC.prepare` as ``dct_sum / 800``.
    l_d : float
        Dark-field regularisation parameter.  Larger values push the
        estimated dark-field toward zero.  Typically set as
        ``dct_sum / 2000``.
    tol : float
        Convergence tolerance.  The loop exits when the relative
        Frobenius-norm residual

        .. math::

            \\frac{\\|D - \\text{repmat}(S \\odot b) - E\\|_F}{\\|D\\|_F}
            \\leq \\text{tol}

        Default ``1e-6``; rarely needs adjustment.
    max_iter : int
        Maximum number of ALM iterations per call.  Default ``500``.
    weight : numpy.ndarray or float
        Optional weight matrix for the reweighted L1 norm.  Must broadcast
        to shape *(N, P·Q)* or be a scalar.  Updated externally by
        :meth:`~linum_basic.core.BaSiC.update_weights`.
    estimate_darkfield : bool
        When ``True`` the dark-field component is estimated in addition to
        the flat-field.
    rho : float
        Lagrange-multiplier step-size growth factor.  At each iteration,
        ``μ ← rho · μ``.  Larger ``rho`` (e.g. ``2.0``) converges faster
        but may overshoot and oscillate for ill-conditioned stacks; smaller
        values (e.g. ``1.1``) are more stable but slower.  Default ``1.5``.
    verbose : bool
        Show a ``tqdm`` progress bar.
    xp : ArrayNamespace or None
        Array namespace to use.  Defaults to the NumPy backend when
        ``None``.

    warm_start : dict or None
        Optional warm-start state from a previous call.  When provided the
        inner ALM variables (``Sf``, ``Ir``, ``B``, ``D_field``, ``Y``,
        ``mu``, ``sigma1``) are initialised from this dict instead of zeros.
        Pass the dict returned when ``return_state=True``.
    return_state : bool
        When ``True`` a fifth return value is added — a dict containing the
        final ALM state suitable for passing as ``warm_start`` to the next
        outer reweighting iteration.  Default ``False``.

    Returns
    -------
    Ib : numpy.ndarray, shape (N, P, Q)
        Unnormalised estimated flat-field, stacked over images.
    Ir : numpy.ndarray, shape (N, P, Q)
        Sparse residual images.
    D_field : numpy.ndarray, shape (1, P*Q)
        Estimated dark-field (flat, spatial domain).
    state : dict, optional
        Only returned when ``return_state=True``.  Contains ``Sf``, ``Ir``,
        ``B``, ``D_field``, ``Y``, ``mu``, ``sigma1`` for warm-starting.

    Notes
    -----
    Each ALM iteration performs the following closed-form sub-steps:

    1. **Flat-field update** — soft-threshold the DCT-II coefficients of
       the current flat-field estimate at threshold ``l_s / μ``.
    2. **Residual update** — pixel-wise soft-threshold of the weighted
       residual matrix at threshold ``1 / μ``.
    3. **Baseline update** — project the current estimate onto the
       per-image mean.
    4. **Dark-field update** (if ``estimate_darkfield``) — soft-threshold
       in pixel space at threshold ``l_d / μ``.
    5. **Multiplier update** — gradient-ascent step with step size ``1/μ``.
    6. **μ update** — ``μ ← min(rho · μ, μ_max)`` where ``μ_max`` is a
       stability ceiling derived from the spectral norm of the data.

    The outer reweighting loop (:meth:`~linum_basic.core.BaSiC.update`) calls
    this function multiple times, updating the weight matrix between calls.

    .. rubric:: References

    .. [1] Peng, T. *et al.* "A BaSiC tool for background and shading
       correction of optical microscopy images." *Nat. Commun.* **8**,
       14836 (2017). https://doi.org/10.1038/ncomms14836
    .. [2] Candès, E., Li, X., Ma, Y. & Wright, J. "Robust Principal
       Component Analysis?" *J. ACM* **58**, 1-37 (2011).
    """
    if xp is None:
        xp = get_xp(Backend.NUMPY)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    n, p, q = imgs.shape

    # Move data to the active backend
    D = xp.asarray(imgs.reshape(n, p * q).astype(np.float32))
    W = xp.asarray(
        (np.ones((n, p * q), dtype=np.float32) * weight)
        if isinstance(weight, (int, float))
        else np.asarray(weight).reshape(n, p * q).astype(np.float32)
    )

    d_norm = xp.norm_fro(D)
    B1_uplimit = xp.min(D)

    # Initialise variables — use warm-start values when provided.
    sigma1 = xp.svd_leading_singular(D)
    mu: float = 12.5 / sigma1
    if warm_start is not None:
        Sf = xp.asarray(warm_start["Sf"].reshape(p, q).astype(np.float32))
        Ir = xp.asarray(warm_start["Ir"].reshape(n, p * q).astype(np.float32))
        B = xp.asarray(warm_start["B"].reshape(n, 1).astype(np.float32))
        D_field = xp.asarray(warm_start["D_field"].reshape(1, p * q).astype(np.float32))
        # Reset Y and mu so changed weights don't cause divergence
        Y = xp.zeros((n, p * q), dtype=np.float64)
    else:
        # Initialise variables
        Sf = xp.zeros((p, q), dtype=np.float32)  # DCT of zero mean is zero
        Ir = xp.zeros_like(D)  # sparse residual
        B = xp.ones((n, 1), dtype=np.float32)  # per-image baseline
        D_field = xp.zeros((1, p * q), dtype=np.float32)  # dark-field (spatial)
        Y = xp.zeros((n, p * q), dtype=np.float64)
    mu_bar: float = mu * 1e7
    ent2: float = 10.0
    converged = False
    iteration = 0
    B1: float = 0.0  # ensure B1 is always defined

    pbar: tqdm | None = tqdm(desc="ALM Iteration", total=max_iter, leave=False) if verbose else None
    # Ib initialised to zeros as a safe default if the loop body never executes.
    Ib = xp.zeros_like(D)

    # ------------------------------------------------------------------
    # Pre-compute S_spatial from the initial (or warm-started) Sf so that
    # the first iteration can reuse it directly, eliminating one iDCT per
    # ALM iteration (the value is refreshed at the END of every iteration).
    # ------------------------------------------------------------------
    S_spatial = xp.astype(xp.idctn(Sf, norm="ortho"), np.float32).reshape(1, p * q)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while not converged and iteration < max_iter:
        # Pre-compute Y / mu once per iteration: the quotient is reused in
        # both the Ir update (step 1) and the Sf mean update (step 2),
        # avoiding a second O(N * P*Q) division.
        # Cast Y (float64 accumulator) to float32 at the use site to avoid
        # accumulating rounding errors in the Lagrange multiplier over ~500
        # iterations while keeping all backend tensor ops in float32.
        Y_over_mu = xp.astype(Y / mu, np.float32)

        # 1. Update sparse residual Ir.
        # S_spatial was computed at the end of the previous iteration (or
        # pre-computed above), so no extra iDCT is needed here.
        Ib = S_spatial * B + D_field  # (N, P*Q)
        Ir = shrink(xp, D - Ib + Y_over_mu, W / mu)

        # Cache D - Ir: the same subtraction is needed in the flat-field
        # update (step 2), the baseline update (step 4), and the Lagrange
        # step (step 6).  Computing it once avoids two redundant N x (P*Q)
        # allocations per iteration.
        DminusIr = D - Ir

        # 2. Update flat-field DCT coefficients Sf.
        # The mean reduction stays on the active backend (GPU-friendly):
        # only a (P, Q) slice is transferred to CPU, not the full (N, P*Q)
        # matrix.
        # Subtract D_field (dark-field) before taking the mean so that its
        # DC/low-frequency energy does not bleed into the flat-field estimate.
        # This matches the MATLAB reference: temp_W = D - A1_hat - E + Y/mu,
        # where A1_hat = S*B + D_field. When estimate_darkfield=False,
        # D_field is identically zero so this subtraction is a no-op.
        R_dev = DminusIr - D_field + Y_over_mu  # reuse cached Y/mu
        R_for_sf_mean = xp.mean(R_dev.reshape(n, p, q), axis=0)  # stays on device, shape (p, q)
        dSf = xp.astype(xp.dctn(R_for_sf_mean, norm="ortho"), np.float32)
        Sf = shrink(xp, dSf, l_s / mu)

        # 3. Reconstruct Ib from the updated Sf.
        # S_spatial is stored for reuse at the START of the next iteration.
        S_spatial = xp.astype(xp.idctn(Sf, norm="ortho"), np.float32).reshape(1, p * q)
        Ib = S_spatial * B + D_field

        # 4. Update baseline B (stays on the active backend device).
        # Per-row and global means computed on device; no CPU transfer needed.
        R_mean_row = xp.mean(DminusIr, axis=1, keepdims=True)  # (N, 1), on device
        R_mean_all = float(xp.mean(DminusIr))
        B = xp.astype(xp.maximum(R_mean_row / (R_mean_all + 1e-9), 0.0), np.float32)

        # 5. Dark-field estimation (all operations on the active backend device).
        # PyTorch supports boolean fancy indexing and masked reductions — no
        # CPU round-trip is needed.
        if estimate_darkfield:
            S_mean = float(xp.mean(S_spatial))
            mask_valid_b = (B < 1.0)[:, 0]  # shape (N,) bool, on device
            mask_high_s = S_spatial[0] > (S_mean - 1e-6)  # shape (P*Q,)
            mask_low_s = S_spatial[0] < (S_mean + 1e-6)

            any_valid = bool(mask_valid_b.any())
            R_high = float(xp.mean(DminusIr[mask_valid_b][:, mask_high_s])) if any_valid else 0.0
            R_low = float(xp.mean(DminusIr[mask_valid_b][:, mask_low_s])) if any_valid else 0.0
            b1_cand = (R_high - R_low) / (R_mean_all + 1e-9)

            k_cnt = int(mask_valid_b.sum())
            b_valid = B[mask_valid_b][:, 0]  # shape (k_cnt,), on device
            if k_cnt > 0:
                temp1 = float((b_valid**2).sum())
                temp2 = float(b_valid.sum())
                temp3 = b1_cand * k_cnt
                temp4 = float((b_valid * b1_cand).sum())
                denom = temp2 * temp3 - k_cnt * temp4
                B1_new = (temp1 * temp3 - temp2 * temp4) / denom if denom != 0.0 else B1
                B1_new = min(B1_new, B1_uplimit / (S_mean + 1e-9))
                if B1_new > 0.0:
                    B1 = B1_new
                # else: keep the previous positive B1 estimate

            Z = B1 * (S_mean - S_spatial)  # shape (1, P*Q), float32 on device

            # Compute A1_offset in float64.  Both mean(R_rows) and b_mean*S are
            # O(1), their difference is O(dark-field) — float32 loses all
            # significant digits (catastrophic cancellation).  numpy's default
            # mean upcast (float32→float64) gave this precision for free before;
            # we replicate it explicitly here.
            S64 = xp.astype(S_spatial, np.float64)
            if any_valid:
                D_rows_f64 = xp.astype(DminusIr[mask_valid_b], np.float64)
                A1_offset = xp.mean(D_rows_f64, axis=0, keepdims=True) - float(b_valid.mean()) * S64
            else:
                A1_offset = xp.zeros((1, p * q), dtype=np.float64)
            A_offset = A1_offset - xp.astype(Z, np.float64)  # (1, P*Q) float64

            # Zero-mean A_offset before the proximal (DCT-shrink) step so that
            # the DC component of the dark-field is not killed by the shrinkage
            # thresholds.  This matches the MATLAB reference which explicitly
            # subtracts mean(A1_offset) before the DCT step.  In the degenerate
            # case (B1=0, Z=0) we add the mean back after shrinkage so the DC is
            # preserved.  The second spatial shrink below is kept intact (paper
            # Eq. 6 dual penalty |F(D_R)|_1 + |D_R|_1).
            A_offset_mean = float(xp.mean(A_offset))
            A_offset_centered = xp.astype(A_offset - A_offset_mean, np.float32)

            Dr_f = xp.astype(xp.dctn(A_offset_centered.reshape(p, q), norm="ortho"), np.float32)
            Dr_f_shrunk = shrink(xp, Dr_f, l_d / (ent2 * mu))
            Dr = xp.astype(xp.idctn(Dr_f_shrunk.reshape(p, q), norm="ortho"), np.float32).reshape(1, p * q)
            Dr = shrink(xp, Dr, l_d / (mu * ent2))
            D_field = xp.astype(Dr + A_offset_mean + Z, np.float32)

        # 6. Update Lagrange multiplier Y (primal residual: D - Ib - Ir).
        # Evaluation order matches the original (D - Ib) - Ir, not (D - Ir) - Ib,
        # to preserve float32 accumulation identical to the pre-optimisation code.
        dY = D - Ib - Ir
        # Accumulate the Lagrange multiplier in float64 on the active device
        # so that large mu values (mu grows as rho^iter) don't erode precision.
        # xp.astype keeps the cast on GPU for torch backends — no device transfer.
        Y = Y + mu * xp.astype(dY, np.float64)
        mu = min(mu * rho, mu_bar)
        iteration += 1

        stop_crit = xp.norm_fro(dY) / (d_norm + 1e-9)
        if pbar is not None:
            pbar.update()
        if stop_crit < tol:
            converged = True

    if iteration == max_iter:
        print("Maximum ALM iterations reached without full convergence.")
    if pbar is not None:
        pbar.close()

    # Fold B1 into D_field (stays on device)
    D_field = xp.astype(D_field + B1 * S_spatial, np.float32)

    # Convert outputs back to NumPy
    Ib_np = xp.to_numpy(Ib).reshape(n, p, q)
    Ir_np = xp.to_numpy(Ir).reshape(n, p, q)
    D_field_np = xp.to_numpy(D_field)

    if return_state:
        state: dict | None = {
            "Sf": xp.to_numpy(Sf).astype(np.float32),
            "Ir": Ir_np.copy(),
            "B": xp.to_numpy(B).astype(np.float32),
            "D_field": D_field_np.copy(),
        }
    else:
        state = None

    return Ib_np, Ir_np, D_field_np, state
