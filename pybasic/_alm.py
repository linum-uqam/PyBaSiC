"""Inexact augmented Lagrangian method (ALM) for BaSiC shading estimation.

This module provides the core L1-minimisation solver used to decompose an
image stack into a smooth flat-field and a sparse residual.  All heavy
numerical work is dispatched through an :class:`~pybasic.backend.ArrayNamespace`
so the same code can run on CPU (NumPy) or GPU (PyTorch).
"""

from __future__ import annotations

import numpy as np
import tqdm

from pybasic.backend import ArrayNamespace, Backend, get_xp

__all__ = ["inexact_alm_l1", "shrink"]


def shrink(xp: ArrayNamespace, theta: object, epsilon: float = 1e-3) -> object:
    """Scalar shrink (soft-threshold) operator.

    Computes ``sign(θ) · max(|θ| - ε, 0)`` element-wise.

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
    return xp.sign(theta) * xp.maximum(xp.abs(theta) - epsilon, 0.0)  # ty: ignore[unsupported-operator]


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L1 minimisation via the inexact augmented Lagrangian method.

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
        Flat-field regularisation parameter (controls DCT-domain sparsity).
    l_d : float
        Dark-field regularisation parameter.
    tol : float
        Convergence tolerance on the relative Frobenius-norm residual.
    max_iter : int
        Maximum number of ALM iterations.
    weight : numpy.ndarray or float
        Optional weight matrix for the reweighted L1 norm.  Must broadcast
        to shape *(N, P·Q)* or be a scalar.
    estimate_darkfield : bool
        When ``True`` the dark-field component is estimated in addition to
        the flat-field.
    rho : float
        Lagrange-multiplier step-size growth factor.
    verbose : bool
        Show a ``tqdm`` progress bar.
    xp : ArrayNamespace or None
        Array namespace to use.  Defaults to the NumPy backend when
        ``None``.

    Returns
    -------
    Ib : numpy.ndarray, shape (N, P, Q)
        Unnormalised estimated flat-field, stacked over images.
    Ir : numpy.ndarray, shape (N, P, Q)
        Sparse residual images.
    D_field : numpy.ndarray, shape (1, P*Q)
        Estimated dark-field (flat, spatial domain).

    Notes
    -----
    The algorithm solves::

        min_{S,E}  λ_s ‖S̃‖₁ + ‖W ⊙ E‖₁
        s.t.       D = repmat(S · B) + E

    where ``S̃`` are the DCT-II coefficients of the flat-field *S*, **B**
    is a per-image baseline vector, and **W** is the reweighting matrix.

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
    B1_uplimit = float(xp.to_numpy(D).min())

    # Initialise variables
    S = xp.zeros_like(D)  # flat-field (spatial)
    Sf = xp.dctn(xp.to_numpy(S).reshape(n, p, q).mean(axis=0), norm="ortho")
    Sf = xp.asarray(np.asarray(Sf).astype(np.float32))
    Ir = xp.zeros_like(D)  # sparse residual
    B = xp.ones((n, 1), dtype=np.float32)  # per-image baseline
    D_field = xp.zeros((1, p * q), dtype=np.float32)  # dark-field (spatial)
    B1: float = 0.0

    # Lagrange multiplier and step-size
    Y = 0.0
    sigma1 = xp.svd_leading_singular(D)
    mu: float = 12.5 / sigma1
    mu_bar: float = mu * 1e7
    ent2: float = 10.0
    converged = False
    iteration = 0

    if verbose:
        pbar = tqdm.tqdm(desc="ALM Iteration", total=max_iter)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while not converged and iteration < max_iter:
        # 1. Update sparse residual Ir
        Sf_np = xp.to_numpy(Sf).reshape(p, q)
        S_spatial = xp.asarray(
            np.asarray(xp.idctn(Sf_np, norm="ortho")).reshape(1, p * q).astype(np.float32)
        )
        Ib = S_spatial * B + D_field  # (N, P*Q)  # ty: ignore[unsupported-operator]
        Ir = shrink(xp, D - Ib + Y / mu, W / mu)  # type: ignore[operator]  # ty: ignore[unsupported-operator]

        # 2. Update flat-field DCT coefficients Sf
        R_for_sf = xp.to_numpy(D - Ir + Y / mu)  # type: ignore[operator]  # ty: ignore[unsupported-operator]
        R_for_sf_mean = R_for_sf.reshape(n, p, q).mean(axis=0)
        dSf = xp.asarray(np.asarray(xp.dctn(R_for_sf_mean, norm="ortho")).astype(np.float32))
        Sf = xp.asarray(xp.to_numpy(shrink(xp, dSf, l_s / mu)).astype(np.float32))

        # 3. Reconstruct Ib from updated Sf
        Sf_np = xp.to_numpy(Sf).reshape(p, q)
        S_spatial = xp.asarray(
            np.asarray(xp.idctn(Sf_np, norm="ortho")).reshape(1, p * q).astype(np.float32)
        )
        Ib = S_spatial * B + D_field  # ty: ignore[unsupported-operator]

        # 5. Update baseline B
        R = D - Ir  # type: ignore[operator]  # ty: ignore[unsupported-operator]
        R_np = xp.to_numpy(R)
        R_mean_row = R_np.mean(axis=1, keepdims=True)
        R_mean_all = R_np.mean()
        B_np = np.clip(R_mean_row / (R_mean_all + 1e-9), 0, None)
        B = xp.asarray(B_np.astype(np.float32))

        # 6. Dark-field estimation
        if estimate_darkfield:
            S_np = xp.to_numpy(S_spatial)
            mask_valid_b = (B_np < 1)[:, 0]  # shape (N,)
            mask_high_s = S_np[0] > (S_np[0].mean() - 1e-6)
            mask_low_s = S_np[0] < (S_np[0].mean() + 1e-6)

            R_high = (
                float(np.mean(R_np[mask_valid_b][:, mask_high_s])) if mask_valid_b.any() else 0.0
            )
            R_low = float(np.mean(R_np[mask_valid_b][:, mask_low_s])) if mask_valid_b.any() else 0.0
            b1_cand = (R_high - R_low) / (R_mean_all + 1e-9)

            k_cnt = int(mask_valid_b.sum())
            b_valid = B_np[mask_valid_b, 0]
            if k_cnt > 0:
                temp1 = float((b_valid**2).sum())
                temp2 = float(b_valid.sum())
                temp3 = b1_cand * k_cnt
                temp4 = float((b_valid * b1_cand).sum())
                denom = temp2 * temp3 - k_cnt * temp4
                B1 = (temp1 * temp3 - temp2 * temp4) / denom if denom != 0.0 else 0.0
            else:
                B1 = 0.0

            B1 = float(np.clip(B1, 0.0, B1_uplimit / (S_np[0].mean() + 1e-9)))

            Z = B1 * (S_np[0].mean() - S_np[0])
            if mask_valid_b.any():
                A1_offset = (
                    R_np[mask_valid_b].mean(axis=0, keepdims=True) - float(b_valid.mean()) * S_np
                )
            else:
                A1_offset = np.zeros_like(S_np)
            A1_offset = A1_offset - A1_offset.mean()
            A_offset = A1_offset - Z

            Dr_f = np.asarray(xp.dctn(A_offset.reshape(p, q), norm="ortho"))
            Dr_f_shrunk = xp.to_numpy(
                shrink(xp, xp.asarray(Dr_f.astype(np.float32)), l_d / (ent2 * mu))
            )
            Dr = np.asarray(xp.idctn(Dr_f_shrunk.reshape(p, q), norm="ortho")).reshape(1, p * q)
            Dr = xp.to_numpy(shrink(xp, xp.asarray(Dr.astype(np.float32)), l_d / (mu * ent2)))
            D_field = xp.asarray((Dr + Z).astype(np.float32))

        # 6. Update Lagrange multiplier Y (primal residual: D - Ib - Ir)
        dY = D - Ib - Ir  # type: ignore[operator]
        Y = Y + mu * dY  # type: ignore[operator]
        mu = min(mu * rho, mu_bar)
        iteration += 1

        stop_crit = xp.norm_fro(dY) / (d_norm + 1e-9)
        if verbose:
            pbar.update()
        if stop_crit < tol:
            converged = True

    if iteration == max_iter:
        print("Maximum ALM iterations reached without full convergence.")  # noqa: T201
    if verbose:
        pbar.close()

    # Fold B1 into D_field
    S_final_np = xp.to_numpy(S_spatial)
    D_field = xp.asarray((xp.to_numpy(D_field) + B1 * S_final_np).astype(np.float32))

    # Convert outputs back to NumPy
    Ib_np = xp.to_numpy(Ib).reshape(n, p, q)
    Ir_np = xp.to_numpy(Ir).reshape(n, p, q)
    D_field_np = xp.to_numpy(D_field)

    return Ib_np, Ir_np, D_field_np
