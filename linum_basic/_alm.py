"""Inexact augmented Lagrangian method (ALM) for BaSiC shading estimation.

This module provides the core L1-minimisation solver used to decompose an
image stack into a smooth flat-field and a sparse residual.  All heavy
numerical work is dispatched through an :class:`~linum_basic.backend.ArrayNamespace`
so the same code can run on CPU (NumPy) or GPU (PyTorch).
"""

from __future__ import annotations

from typing import Any

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
    # Build (optionally compiled) per-iteration core step.
    # Steps 1-4: Ir, Sf, S_spatial, Ib, B update — pure-tensor ops with no
    # Python control-flow branches on tensor values, so torch.compile can
    # trace through them cleanly.  The darkfield update (step 5) and the
    # Y/mu update (step 6) remain in the caller because they require Python
    # branches on host values.
    #
    # S_spatial is passed as a pre-computed parameter (it is produced at the
    # end of each iteration) to avoid a redundant iDCT on every call.
    # ------------------------------------------------------------------
    def _alm_core_step(
        d: Any,
        s_spatial: Any,
        b: Any,
        d_field: Any,
        y: Any,
        w: Any,
        cur_mu: float,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        y_f32 = xp.astype(y / cur_mu, np.float32)
        # Step 1: update Ir using S_spatial from the previous iteration.
        ib_old = s_spatial * b + d_field
        new_ir = shrink(xp, d - ib_old + y_f32, w / cur_mu)
        d_minus_ir = d - new_ir
        # Step 2: update flat-field DCT coefficients.
        r_dev = d_minus_ir - d_field + y_f32
        r_for_sf_mean = xp.mean(r_dev.reshape(-1, p, q), axis=0)
        d_sf = xp.astype(xp.dctn(r_for_sf_mean, norm="ortho"), np.float32)
        new_sf = shrink(xp, d_sf, l_s / cur_mu)
        # Step 3: reconstruct Ib from updated Sf.
        new_s_spatial = xp.astype(xp.idctn(new_sf, norm="ortho"), np.float32).reshape(1, p * q)
        new_ib = new_s_spatial * b + d_field
        # Step 4: update baseline B.
        r_mean_row = xp.mean(d_minus_ir, axis=1, keepdims=True)
        new_r_mean_all = xp.mean(d_minus_ir)
        new_b = xp.astype(xp.maximum(r_mean_row / (new_r_mean_all + 1e-9), 0.0), np.float32)
        # dY for Lagrange update (caller only needs it after darkfield step).
        d_y = d - new_ib - new_ir
        return new_sf, new_s_spatial, new_ib, new_ir, d_minus_ir, new_b, new_r_mean_all, d_y

    if xp._backend is not Backend.NUMPY:
        try:
            import torch as _torch

            if hasattr(_torch, "compile"):
                _alm_core_step = _torch.compile(_alm_core_step, mode="default", fullgraph=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main loop  (gradient tracking disabled for torch backends)
    # ------------------------------------------------------------------
    with xp.inference_mode():
        while not converged and iteration < max_iter:
            # Steps 1-4: Ir / Sf / S_spatial / Ib / B update (compiled when on GPU).
            Sf, S_spatial, Ib, Ir, DminusIr, B, R_mean_all, dY = _alm_core_step(D, S_spatial, B, D_field, Y, W, mu)

            # 5. Dark-field estimation (all operations on the active backend device).
            if estimate_darkfield:
                # S_mean stays on device; avoid float() round-trip.
                S_mean = xp.mean(S_spatial)
                # mask_valid_b: shape (N,) float32 weight (0 or 1) — replaces
                # boolean fancy indexing so shapes stay static for CUDA graphs.
                mask_valid_b_f = xp.astype((B < 1.0)[:, 0], np.float32)  # (N,)
                mask_high_s_f = xp.astype(S_spatial[0] > (S_mean - 1e-6), np.float32)  # (P*Q,)
                mask_low_s_f = xp.astype(S_spatial[0] < (S_mean + 1e-6), np.float32)  # (P*Q,)

                # Number of valid rows; single sync point.
                k_cnt = float(mask_valid_b_f.sum())
                any_valid = k_cnt > 0.0

                # Masked row-mean: sum(DminusIr * mask_valid_b_f[:, None], axis=0) / k_cnt
                # Then dot with high/low-S masks to get R_high / R_low.
                R_mean_all_f = float(R_mean_all)
                if any_valid:
                    # DminusIr_valid: (N, P*Q) weighted, reduces to row-mean of valid rows.
                    DminusIr_valid_rowsum = xp.sum(DminusIr * mask_valid_b_f[:, None], axis=0, keepdims=True)  # (1, P*Q)
                    n_high = float(mask_high_s_f.sum())
                    n_low = float(mask_low_s_f.sum())
                    R_high = float(xp.sum(DminusIr_valid_rowsum * mask_high_s_f[None, :]) / (k_cnt * n_high + 1e-9))
                    R_low = float(xp.sum(DminusIr_valid_rowsum * mask_low_s_f[None, :]) / (k_cnt * n_low + 1e-9))
                else:
                    R_high = 0.0
                    R_low = 0.0
                    DminusIr_valid_rowsum = xp.zeros((1, p * q), dtype=np.float32)
                S_mean_f = float(S_mean)
                b1_cand = (R_high - R_low) / (R_mean_all_f + 1e-9)

                # b_valid statistics using mask; b_flat shape (N,) float32.
                b_flat = xp.astype(B[:, 0], np.float32)  # (N,)
                sum_b = 0.0  # initialise so the second `if any_valid` block sees it
                if any_valid:
                    # Weighted sums (static shape, no dynamic indexing).
                    sum_b2 = float(xp.sum(b_flat**2 * mask_valid_b_f))
                    sum_b = float(xp.sum(b_flat * mask_valid_b_f))
                    temp1 = sum_b2
                    temp2 = sum_b
                    temp3 = b1_cand * k_cnt
                    temp4 = float(xp.sum(b_flat * b1_cand * mask_valid_b_f))
                    denom = temp2 * temp3 - k_cnt * temp4
                    B1_new = (temp1 * temp3 - temp2 * temp4) / denom if denom != 0.0 else B1
                    B1_new = min(B1_new, B1_uplimit / (S_mean_f + 1e-9))
                    if B1_new > 0.0:
                        B1 = B1_new
                    # else: keep the previous positive B1 estimate

                Z = B1 * (S_mean - S_spatial)  # shape (1, P*Q), float32 on device

                # Compute A1_offset in float64 to avoid catastrophic cancellation.
                S64 = xp.astype(S_spatial, np.float64)
                if any_valid:
                    D_valid_rowsum_f64 = xp.astype(DminusIr_valid_rowsum, np.float64)  # (1, P*Q)
                    b_valid_mean = sum_b / k_cnt
                    A1_offset = D_valid_rowsum_f64 / k_cnt - b_valid_mean * S64
                else:
                    A1_offset = xp.zeros((1, p * q), dtype=np.float64)
                A_offset = A1_offset - xp.astype(Z, np.float64)  # (1, P*Q) float64

                # Zero-mean A_offset before the proximal step (matches MATLAB reference).
                A_offset_mean = float(xp.mean(A_offset))
                A_offset_centered = xp.astype(A_offset - A_offset_mean, np.float32)

                Dr_f = xp.astype(xp.dctn(A_offset_centered.reshape(p, q), norm="ortho"), np.float32)
                Dr_f_shrunk = shrink(xp, Dr_f, l_d / (ent2 * mu))
                Dr = xp.astype(xp.idctn(Dr_f_shrunk.reshape(p, q), norm="ortho"), np.float32).reshape(1, p * q)
                Dr = shrink(xp, Dr, l_d / (mu * ent2))
                D_field = xp.astype(Dr + A_offset_mean + Z, np.float32)

            # 6. Update Lagrange multiplier Y (primal residual: D - Ib - Ir).
            # dY was returned by _alm_core_step; it equals D - Ib - Ir using the
            # new S_spatial/B and the current D_field (before the darkfield update).
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
