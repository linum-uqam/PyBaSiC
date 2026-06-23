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

# ---------------------------------------------------------------------------
# Module-level cache for compiled per-iteration step functions.
# Key: (backend_value, device_str, p, q, l_s)  →  compiled callable.
# Caching here avoids re-compiling on every call to inexact_alm_l1 (which is
# invoked once per reweighting pass, up to max_reweighting_iterations times
# per BaSiC.run()).  The JIT cost is paid at most once per unique problem shape
# and regularisation weight across the entire process lifetime.
# ---------------------------------------------------------------------------
_ALM_STEP_CACHE: dict[tuple, Any] = {}


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


def _build_alm_step(
    xp: ArrayNamespace,
    n: int,
    p: int,
    q: int,
    l_s: float,
    *,
    estimate_darkfield: bool = False,
    l_d: float = 0.0,
    ent2: float = 10.0,
    b1_uplimit: float = 0.0,
) -> Any:
    """Return a (possibly compiled) per-iteration ALM step function.

    The returned callable is cached by ``(backend, device, n, p, q, l_s)`` so
    the same compiled artifact is reused across all reweighting passes and
    across different :class:`~linum_basic.core.BaSiC` instances that share
    the same problem shape and regularisation weight.
    """
    device_key = str(getattr(xp, "_device", ""))
    cache_key = (
        xp._backend.value,
        device_key,
        n,
        p,
        q,
        l_s,
        estimate_darkfield,
        l_d,
    )
    if cache_key in _ALM_STEP_CACHE:
        return _ALM_STEP_CACHE[cache_key]

    # On GPU backends, replace FFT-based DCT with real matrix-multiply DCT.
    # torch.fft.fft produces complex intermediate tensors that Torchinductor
    # cannot compile to Triton kernels, forcing those ops into eager mode.
    # Precomputing the orthonormal DCT-II matrices A_p, A_q and expressing
    # the 2-D DCT as  Y = A_p @ X @ A_q.T  uses only real matmuls that
    # Triton compiles efficiently.  We use float32 for A to stay in the
    # same dtype as the inputs; the matrices are built once and captured.
    if xp._backend is not Backend.NUMPY:
        from linum_basic.backend import _get_dct_matrix as _gcm

        _device = getattr(xp, "_device", "cpu")
        _Ap = _gcm(p, _device).float()  # (p, p) f32, orthonormal DCT-II
        _Aq = _gcm(q, _device).float()  # (q, q) f32, orthonormal DCT-II

        def _dctn2(x: Any) -> Any:
            # 2-D DCT-II: Y = A_p @ X @ A_q.T
            return _Ap @ x @ _Aq.T

        def _idctn2(x: Any) -> Any:
            # 2-D DCT-III (inverse DCT-II): X = A_p.T @ Y @ A_q
            return _Ap.T @ x @ _Aq

    else:
        # NumPy path: delegate to scipy via ArrayNamespace
        def _dctn2(x: Any) -> Any:
            return xp.dctn(x, norm="ortho")

        def _idctn2(x: Any) -> Any:
            return xp.idctn(x, norm="ortho")

    def _alm_core_step(
        d: Any,
        s_spatial: Any,
        b: Any,
        d_field: Any,
        y: Any,
        w: Any,
        cur_mu: Any,
        b1: Any,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
        y_f32 = xp.astype(y / cur_mu, np.float32)
        # Step 1: update Ir using S_spatial from the previous iteration.
        ib_old = s_spatial * b + d_field
        new_ir = shrink(xp, d - ib_old + y_f32, w / cur_mu)
        d_minus_ir = d - new_ir
        # Step 2: update flat-field DCT coefficients.
        r_dev = d_minus_ir - d_field + y_f32
        r_for_sf_mean = xp.mean(r_dev.reshape(-1, p, q), axis=0)
        d_sf = xp.astype(_dctn2(r_for_sf_mean), np.float32)
        new_sf = shrink(xp, d_sf, l_s / cur_mu)
        # Step 3: reconstruct Ib from updated Sf.
        new_s_spatial = xp.astype(_idctn2(new_sf), np.float32).reshape(1, p * q)
        new_ib = new_s_spatial * b + d_field
        # Step 4: update baseline B.
        r_mean_row = xp.mean(d_minus_ir, axis=1, keepdims=True)
        new_r_mean_all = xp.mean(d_minus_ir)
        new_b = xp.astype(xp.maximum(r_mean_row / (new_r_mean_all + 1e-9), 0.0), np.float32)
        # dY for Lagrange update (uses D_field before darkfield refresh).
        d_y = d - new_ib - new_ir
        new_d_field = d_field
        new_b1 = b1
        if estimate_darkfield:
            s_mean = xp.mean(new_s_spatial)
            mask_valid_b_f = xp.astype((new_b < 1.0)[:, 0], np.float32)
            mask_high_s_f = xp.astype(new_s_spatial[0] > (s_mean - 1e-6), np.float32)
            mask_low_s_f = xp.astype(new_s_spatial[0] < (s_mean + 1e-6), np.float32)
            k_cnt = xp.sum(mask_valid_b_f)
            dminus_ir_valid_rowsum = xp.sum(d_minus_ir * mask_valid_b_f[:, None], axis=0, keepdims=True)
            n_high = xp.sum(mask_high_s_f)
            n_low = xp.sum(mask_low_s_f)
            r_high = xp.sum(dminus_ir_valid_rowsum * mask_high_s_f[None, :]) / (k_cnt * n_high + 1e-9)
            r_low = xp.sum(dminus_ir_valid_rowsum * mask_low_s_f[None, :]) / (k_cnt * n_low + 1e-9)
            b1_cand = (r_high - r_low) / (new_r_mean_all + 1e-9)
            b_flat = xp.astype(new_b[:, 0], np.float32)
            sum_b2 = xp.sum(b_flat**2 * mask_valid_b_f)
            sum_b = xp.sum(b_flat * mask_valid_b_f)
            temp1 = sum_b2
            temp2 = sum_b
            temp3 = b1_cand * k_cnt
            temp4 = xp.sum(b_flat * b1_cand * mask_valid_b_f)
            denom = temp2 * temp3 - k_cnt * temp4
            b1_new_raw = (temp1 * temp3 - temp2 * temp4) / (denom + 1e-30)
            b1_new_guarded = xp.where(xp.abs(denom) > 1e-30, b1_new_raw, b1)
            b1_new = xp.minimum(b1_new_guarded, b1_uplimit / (s_mean + 1e-9))
            new_b1 = xp.where(b1_new > 0.0, b1_new, b1)
            z = new_b1 * (s_mean - new_s_spatial)
            k_cnt_f64 = xp.astype(k_cnt, np.float64)
            sum_b_f64 = xp.astype(sum_b, np.float64)
            s64 = xp.astype(new_s_spatial, np.float64)
            d_valid_rowsum_f64 = xp.astype(dminus_ir_valid_rowsum, np.float64)
            b_valid_mean = sum_b_f64 / (k_cnt_f64 + 1e-30)
            a1_offset = d_valid_rowsum_f64 / (k_cnt_f64 + 1e-30) - b_valid_mean * s64
            a_offset = a1_offset - xp.astype(z, np.float64)
            a_offset_mean = xp.mean(a_offset)
            a_offset_centered = xp.astype(a_offset - a_offset_mean, np.float32)
            dr_f = xp.astype(_dctn2(a_offset_centered.reshape(p, q)), np.float32)
            dr_f_shrunk = shrink(xp, dr_f, l_d / (ent2 * cur_mu))
            dr = xp.astype(_idctn2(dr_f_shrunk.reshape(p, q)), np.float32).reshape(1, p * q)
            dr = shrink(xp, dr, l_d / (cur_mu * ent2))
            new_d_field = xp.astype(dr + xp.astype(a_offset_mean, np.float32) + z, np.float32)
        return new_sf, new_s_spatial, new_ib, new_ir, d_minus_ir, new_b, new_r_mean_all, d_y, new_d_field, new_b1

    fn: Any = _alm_core_step
    if xp._backend is not Backend.NUMPY:
        try:
            import torch as _torch

            if hasattr(_torch, "compile"):
                # Enable TF32 for float32 matmul: faster on Ampere+ GPUs with
                # negligible precision loss for the iterative ALM solver.
                _torch.set_float32_matmul_precision("high")
                fn = _torch.compile(_alm_core_step, mode="default", fullgraph=False)
        except Exception:
            pass

    _ALM_STEP_CACHE[cache_key] = fn
    return fn


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
    B1_uplimit: float = xp.min(D)

    # Scalars and per-call configuration — computed before entering
    # inference_mode so they remain plain Python floats / constant device
    # tensors that are never mutated and never fed back into _alm_core_step.
    sigma1 = xp.svd_leading_singular(D)  # returns Python float
    mu_scalar: float = 12.5 / sigma1
    if warm_start is not None:
        Sf = xp.asarray(warm_start["Sf"].reshape(p, q).astype(np.float32))
        Ir = xp.asarray(warm_start["Ir"].reshape(n, p * q).astype(np.float32))
        B = xp.asarray(warm_start["B"].reshape(n, 1).astype(np.float32))
        D_field = xp.asarray(warm_start["D_field"].reshape(1, p * q).astype(np.float32))
        # Reset Y so changed weights don't cause divergence
        Y = xp.zeros((n, p * q), dtype=np.float64)
    else:
        # Initialise variables
        Sf = xp.zeros((p, q), dtype=np.float32)  # DCT of zero mean is zero
        Ir = xp.zeros_like(D)  # sparse residual
        B = xp.ones((n, 1), dtype=np.float32)  # per-image baseline
        D_field = xp.zeros((1, p * q), dtype=np.float32)  # dark-field (spatial)
        Y = xp.zeros((n, p * q), dtype=np.float64)
    mu_bar_scalar: float = mu_scalar * 1e7
    ent2: float = 10.0
    converged = False
    iteration = 0
    # On GPU backends xp.norm_fro forces a GPU→CPU synchronisation via float().
    # Check convergence every N iterations to amortise this cost.
    # On CPU backends every iteration is essentially free.
    convergence_check_every = 10 if xp._backend is not Backend.NUMPY else 1
    B1: Any = xp.zeros((), dtype=np.float32)  # scalar device tensor; 0 is safe default

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
    # Retrieve (or build and compile) the per-iteration core step.
    # Steps 1-4 are a pure-tensor function compiled once per unique
    # (backend, device, n, p, q, l_s) combination and cached at module scope.
    # ------------------------------------------------------------------
    _alm_core_step = _build_alm_step(
        xp,
        n,
        p,
        q,
        l_s,
        estimate_darkfield=estimate_darkfield,
        l_d=l_d,
        ent2=ent2,
        b1_uplimit=B1_uplimit,
    )

    # ------------------------------------------------------------------
    # Main loop  (gradient tracking disabled for torch backends)
    # ------------------------------------------------------------------
    with xp.inference_mode():
        # mu grows by rho=1.5 each iteration; keeping it as a 0-dim
        # InferenceMode tensor ensures torch.compile guards on the stable
        # dispatch key, not the changing float value, preventing per-iteration
        # recompiles that exhaust recompile_limit=8 after 9 steps.
        mu: Any = xp.asarray(np.array(mu_scalar, dtype=np.float32))
        mu_bar: Any = xp.asarray(np.array(mu_bar_scalar, dtype=np.float32))
        while not converged and iteration < max_iter:
            # Steps 1-5: flat-field, baseline, optional dark-field (compiled on GPU).
            Sf, S_spatial, Ib, Ir, _DminusIr, B, _R_mean_all, dY, D_field, B1 = _alm_core_step(
                D, S_spatial, B, D_field, Y, W, mu, B1
            )
            B = xp.clone(B)
            S_spatial = xp.clone(S_spatial)
            # 6. Update Lagrange multiplier Y (primal residual: D - Ib - Ir).
            # dY was returned by _alm_core_step; it equals D - Ib - Ir using the
            # new S_spatial/B and the current D_field (before the darkfield update).
            # Accumulate the Lagrange multiplier in float64 on the active device
            # so that large mu values (mu grows as rho^iter) don't erode precision.
            # xp.astype keeps the cast on GPU for torch backends — no device transfer.
            Y = Y + xp.astype(mu, np.float64) * xp.astype(dY, np.float64)
            mu = xp.minimum(mu * rho, mu_bar)
            iteration += 1

            if iteration % convergence_check_every == 0:
                stop_crit = xp.norm_fro(dY) / (d_norm + 1e-9)
                if stop_crit < tol:
                    converged = True
            if pbar is not None:
                pbar.update()

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
