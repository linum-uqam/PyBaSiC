"""Batched per-z BaSiC fitting for CUDA mosaic grids."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm

from linum_basic._alm import inexact_alm_l1_batched
from linum_basic.core import DEFAULT_L_D_DIVISOR, DEFAULT_L_S_DIVISOR, dct_energy

__all__ = ["fit_stacks_batched", "prepare_stacks_batched", "should_use_batched_cuda"]


def should_use_batched_cuda(
    *,
    field_mode: str,
    n_z: int,
    backend: str | None,
    device: str | None,
) -> bool:
    """Return whether the batched CUDA path should be used.

    Returns
    -------
    bool
        ``True`` when per-z fitting on CUDA with more than one z-level.
    """
    if field_mode != "per-z" or n_z <= 1:
        return False
    if backend not in {"torch", "auto"}:
        return False
    dev = (device or "cuda").lower()
    if not dev.startswith("cuda"):
        return False
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def prepare_stacks_batched(
    tile_stacks: list[np.ndarray],
    *,
    working_size: int,
    n_extra_rows: int,
    l_s: float | None = None,
    l_d: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resize, crop, and sort tile stacks for batched BaSiC.

    Parameters
    ----------
    tile_stacks : list of numpy.ndarray
        One ``(N, th, tw)`` stack per z-level.
    working_size : int
        Square working resolution for BaSiC.
    n_extra_rows : int
        Galvo-return rows stripped from the top of each tile before fitting.
    l_s, l_d : float or None
        Optional regularisation overrides.  When ``None`` each z-level is
        auto-tuned from its mean image DCT energy.

    Returns
    -------
    img_sort : numpy.ndarray, shape ``(Z, N, ws, ws)``
        Pixel-sorted stacks ready for the ALM solver.
    l_s_arr, l_d_arr : numpy.ndarray, shape ``(Z,)``
        Per-z regularisation weights.
    tile_shape : numpy.ndarray, shape ``(2,)``
        ``(th, tw)`` of the (possibly row-cropped) tiles before upsampling.
    """
    if not tile_stacks:
        msg = "tile_stacks must be non-empty."
        raise ValueError(msg)

    z = len(tile_stacks)
    stacks = [ts[:, n_extra_rows:, :] if n_extra_rows > 0 else ts for ts in tile_stacks]
    n, th, tw = stacks[0].shape
    ws = working_size
    new_shape = (ws, ws)
    interp = cv2.INTER_LINEAR if ws > th else cv2.INTER_AREA

    resized = np.zeros((z, n, ws, ws), dtype=np.float32)
    for zi, stack in enumerate(stacks):
        for i in range(n):
            img = stack[i].squeeze()
            resized[zi, i] = cv2.resize(img.T, new_shape, interpolation=interp).T

    img_sort = np.sort(resized, axis=1)

    l_s_arr = np.empty(z, dtype=np.float32)
    l_d_arr = np.empty(z, dtype=np.float32)
    for zi in range(z):
        dct_sum = dct_energy(img_sort[zi].mean(axis=0))
        l_s_arr[zi] = float(l_s) if l_s is not None else dct_sum / DEFAULT_L_S_DIVISOR
        l_d_arr[zi] = float(l_d) if l_d is not None else dct_sum / DEFAULT_L_D_DIVISOR

    return img_sort, l_s_arr, l_d_arr, np.array([th, tw], dtype=np.int64)


def fit_stacks_batched(
    img_sort: np.ndarray,
    *,
    l_s_arr: np.ndarray,
    l_d_arr: np.ndarray,
    params: dict[str, Any],
    xp: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the outer reweighting loop on batched sorted stacks.

    Returns
    -------
    tuple of numpy.ndarray
        ``(flatfields, darkfields)`` each with shape ``(Z, ws, ws)``.
    """
    z, _n, ws, _ws2 = img_sort.shape
    estimate_darkfield = bool(params.get("estimate_darkfield", False))
    max_reweighting = int(params.get("max_reweighting_iterations", 10))
    reweighting_tol = float(params.get("reweighting_tolerance", 1e-3))
    epsilon = float(params.get("epsilon", 0.1))
    warm_start = bool(params.get("warm_start_reweighting", False))
    verbose = bool(params.get("verbose", False))

    weights = np.ones_like(img_sort, dtype=np.float32)
    flatfields = np.ones((z, ws, ws), dtype=np.float32)
    darkfields = np.zeros((z, ws, ws), dtype=np.float32)
    # Per-z outer-loop freeze: each plane converges independently, exactly as a
    # standalone BaSiC.run() would.  Once a plane's relative flat/dark-field
    # change drops below the tolerance its fields and weights are frozen so it
    # receives no further reweighting iterations (matching the sequential path).
    done = np.zeros(z, dtype=bool)
    alm_state: dict | None = None
    flag = True
    reweight_iter = 0

    pbar: tqdm | None = tqdm(desc="Batched reweighting", total=max_reweighting, leave=False) if verbose else None
    while flag:
        last_ff = flatfields.copy()
        last_df = darkfields.copy()

        ib, ir, d_field, alm_state_out = inexact_alm_l1_batched(
            img_sort,
            l_s_arr,
            l_d_arr,
            weight=weights,
            estimate_darkfield=estimate_darkfield,
            verbose=verbose,
            xp=xp,
            warm_start=alm_state if warm_start else None,
            return_state=warm_start,
        )
        if warm_start:
            alm_state = alm_state_out

        denom = np.abs(ir / (ib.mean(axis=(1, 2, 3), keepdims=True) + 1e-6)) + epsilon
        new_weights = 1.0 / denom
        # Normalise per z-plane so mean(W)=1 within each plane (matches the
        # single-plane BaSiC.update_weights, which uses the per-plane element
        # count, not the whole batch).
        per_z_size = new_weights[0].size
        new_weights = new_weights * per_z_size / new_weights.sum(axis=(1, 2, 3), keepdims=True)

        d_2d = d_field.reshape(z, ws, ws)
        new_ff = ib.mean(axis=1) - d_2d
        new_ff = new_ff / (new_ff.mean(axis=(1, 2), keepdims=True) + 1e-9)
        new_df = d_2d

        # Apply updates only to planes that have not yet converged.
        active = ~done
        weights[active] = new_weights[active]
        flatfields[active] = new_ff[active]
        darkfields[active] = new_df[active]

        mad_flat = np.abs(flatfields - last_ff).sum(axis=(1, 2)) / (np.abs(last_ff).sum(axis=(1, 2)) + 1e-9)
        mad_dark_abs = np.abs(darkfields - last_df).sum(axis=(1, 2))
        last_dark_sum = np.abs(last_df).sum(axis=(1, 2))
        mad_dark = np.where(
            mad_dark_abs < 1e-7,
            0.0,
            np.where(last_dark_sum < 1e-7, 1.0, mad_dark_abs / (last_dark_sum + 1e-9)),
        )

        reweight_iter += 1
        newly_done = active & (np.maximum(mad_flat, mad_dark) <= reweighting_tol)
        done = done | newly_done
        if np.all(done) or reweight_iter >= max_reweighting:
            flag = False
        if pbar is not None:
            pbar.update()
    if pbar is not None:
        pbar.close()

    return flatfields, darkfields


def upsample_fields_batched(
    flatfields: np.ndarray,
    darkfields: np.ndarray,
    tile_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Up-sample working-size fields back to tile resolution.

    Returns
    -------
    tuple of numpy.ndarray
        ``(flatfields, darkfields)`` each with shape ``(Z, th, tw)``.
    """
    th, tw = tile_shape
    z = flatfields.shape[0]
    ff_out = np.empty((z, th, tw), dtype=np.float32)
    df_out = np.empty((z, th, tw), dtype=np.float32)
    for zi in range(z):
        ff = cv2.resize(flatfields[zi], (tw, th), interpolation=cv2.INTER_LINEAR)
        ff_out[zi] = ff / (ff.mean() + 1e-9)
        df_out[zi] = cv2.resize(darkfields[zi], (tw, th), interpolation=cv2.INTER_LINEAR)
    return ff_out, df_out
