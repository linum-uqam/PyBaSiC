"""Per-z mosaic-grid fitting pipeline.

Runs one :class:`~linum_basic.core.BaSiC` model per z-level over all
tiles extracted from a :class:`~linum_basic.mosaic.MosaicGrid`, then
optionally collapses the per-z fields into a single global field.

OCT data typically exhibits a **focal curve** — the illumination profile
shifts slightly with depth because the focal plane moves as z increases.
``field_mode="per-z"`` captures this variation; ``field_mode="global"``
averages it out, which is more robust on noisy data but blurs the depth
dependence.

.. tip::
   For a smooth per-z estimate on noisy data, post-process
   ``MosaicFit.flatfields`` with a 1-D Gaussian kernel along the z-axis
   (e.g. ``scipy.ndimage.gaussian_filter1d(flatfields, sigma=2, axis=0)``)
   before calling :func:`apply_fit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from linum_basic._parallel import list_cuda_devices, parallel_map, parallel_map_cuda_devices, resolve_workers
from linum_basic.core import BaSiC
from linum_basic.mosaic import MosaicGrid

if TYPE_CHECKING:
    pass

__all__ = ["MosaicFit", "apply_fit", "fit_mosaic", "make_model"]

# BaSiC.__init__ keyword-argument names; everything else is a post-init attr.
_BASIC_INIT_PARAMS: frozenset[str] = frozenset({"estimate_darkfield", "extension", "verbose", "backend", "device"})


def make_model(tiles: np.ndarray, params: dict[str, Any]) -> BaSiC:
    """Construct and fully configure a :class:`BaSiC` model from *params*.

    Parameters accepted by ``BaSiC.__init__`` are passed as keyword
    arguments; the rest (``working_size``, ``l_s``, ``l_d``,
    ``epsilon``, …) are set as attributes after construction.

    Parameters
    ----------
    tiles : numpy.ndarray
        Tile stack, shape ``(N, th, tw)``, used to initialise the model.
    params : dict
        BaSiC hyperparameters.  Keys matching ``BaSiC.__init__`` arguments
        are passed to the constructor; remaining keys are set as attributes.

    Returns
    -------
    BaSiC
        The configured model, ready for ``prepare()`` and ``run()``.
    """
    init_kw = {k: v for k, v in params.items() if k in _BASIC_INIT_PARAMS}
    post_kw = {k: v for k, v in params.items() if k not in _BASIC_INIT_PARAMS}
    model = BaSiC(tiles, **init_kw)
    for key, val in post_kw.items():
        setattr(model, key, val)
    return model


def _fit_one_z(
    tiles: np.ndarray,
    params: dict[str, Any],
    n_extra_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a single z-level tile stack and return its flat/dark-fields.

    Module-level (picklable) so it can run inside a worker process. The
    galvo-return rows are stripped before fitting; the caller writes the
    returned fields back into the masked output arrays.

    Parameters
    ----------
    tiles : numpy.ndarray
        Tile stack for one z-level, shape ``(N, th, tw)``.
    params : dict
        BaSiC hyperparameters (see :func:`make_model`).
    n_extra_rows : int
        Number of galvo-return rows to exclude from the top of each tile.

    Returns
    -------
    tuple of numpy.ndarray
        ``(flatfield, darkfield)`` for the (possibly row-cropped) tiles.
    """
    if n_extra_rows > 0:
        tiles = tiles[:, n_extra_rows:, :]
    model = make_model(tiles, params)
    model.prepare()
    model.run()
    return model.get_flatfield(), model.get_darkfield()


def _fit_one_z_on_device(
    tiles: np.ndarray,
    device: str,
    params: dict[str, Any],
    n_extra_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one z-level on a specific CUDA device (picklable entry point)."""
    dev_params = dict(params)
    dev_params["device"] = device
    return _fit_one_z(tiles, dev_params, n_extra_rows)


def _fit_one_z_cuda_map(
    tiles: np.ndarray,
    device: str,
    *,
    params: dict[str, Any],
    n_extra_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Module-level partial target for :func:`parallel_map_cuda_devices`."""
    return _fit_one_z_on_device(tiles, device, params, n_extra_rows)


@dataclass(init=False)
class MosaicFit:
    """Results of a BaSiC fit applied to a mosaic grid.

    Attributes
    ----------
    flatfields : numpy.ndarray
        Shape ``(Z, th, tw)`` for ``field_mode="per-z"``;
        shape ``(th, tw)`` for ``field_mode="global"``.
    darkfields : numpy.ndarray
        Same shape as *flatfields*.
    field_mode : {"per-z", "global"}
        Whether fields are per-z or a single averaged field.
    z_indices : list of int
        Z-levels that were fitted (subset of all z-levels).
    params : dict
        BaSiC hyperparameters used for this fit.
    """

    flatfields: np.ndarray
    darkfields: np.ndarray
    field_mode: str
    z_indices: list[int]
    params: dict[str, Any]

    def __init__(
        self,
        flatfields: np.ndarray,
        darkfields: np.ndarray,
        field_mode: str,
        z_indices: list[int],
        params: dict[str, Any] | None = None,
    ) -> None:
        """Construct a :class:`MosaicFit` result.

        Parameters
        ----------
        flatfields : numpy.ndarray
            Shape ``(Z, th, tw)`` for ``field_mode="per-z"``;
            shape ``(th, tw)`` for ``field_mode="global"``.
        darkfields : numpy.ndarray
            Same shape as *flatfields*.
        field_mode : {"per-z", "global"}
            Whether fields are per-z or a single averaged field.
        z_indices : list of int
            Z-levels that were fitted (subset of all z-levels).
        params : dict or None
            BaSiC hyperparameters used for this fit.  Defaults to ``{}``.
        """
        self.flatfields = flatfields
        self.darkfields = darkfields
        self.field_mode = field_mode
        self.z_indices = z_indices
        self.params = params if params is not None else {}


def fit_mosaic(
    mosaic: MosaicGrid,
    *,
    z_indices: list[int] | None = None,
    field_mode: Literal["per-z", "global"] = "per-z",
    basic_kwargs: dict[str, Any] | None = None,
    n_extra_rows: int = 0,
    n_workers: int | None = None,
    verbose: bool = False,
) -> MosaicFit:
    """Fit one BaSiC model per z-level over all tiles of *mosaic*.

    Parameters
    ----------
    mosaic : MosaicGrid
        The mosaic grid to fit.
    z_indices : list of int or None
        Z-levels to fit.  ``None`` fits every z-level.
    field_mode : {"per-z", "global"}
        ``"per-z"`` stores one flat/dark-field per fitted z-level.
        ``"global"`` averages all per-z fields into a single field.
        Use ``"per-z"`` when OCT focal-plane position varies with depth.
        Use ``"global"`` when the dataset is too noisy for reliable per-z
        estimation.
    basic_kwargs : dict or None
        Hyperparameters forwarded to :class:`~linum_basic.core.BaSiC`.
        May include ``working_size``, ``l_s``, ``l_d``, ``epsilon``,
        ``estimate_darkfield``, ``backend``, ``device``, etc.
    n_extra_rows : int
        Number of rows at the top of each tile to exclude from the BaSiC
        fit (galvo return / flyback signal).  The excluded rows are filled
        with 1.0 (flat-field) and 0.0 (dark-field) in the output so that
        :func:`apply_fit` treats them as uncorrected pass-through pixels.
        Default ``0`` (no masking).
    n_workers : int or None
        Number of worker processes used to fit z-levels in parallel. Each
        z-level is an independent BaSiC solve, so this scales nearly
        linearly on CPU. ``None`` (default) uses ``cpu_count() - 2``.
        ``1`` runs sequentially. Forced to ``1`` on the PyTorch CUDA/MPS
        backend (single-accelerator contention).
    verbose : bool
        Show a progress bar over z-levels.

    Returns
    -------
    MosaicFit
        Fitted flat/dark-fields for each requested z-level.
    """
    params: dict[str, Any] = dict(basic_kwargs or {})
    # Warm-start inner ALM across outer reweighting passes when the cap is high.
    if "warm_start_reweighting" not in params and int(params.get("max_reweighting_iterations", 10)) >= 5:
        params["warm_start_reweighting"] = True

    z_idx = list(z_indices) if z_indices is not None else list(range(mosaic.n_z))

    th, tw = mosaic.tile_shape
    flatfields = np.ones((len(z_idx), th, tw), dtype=np.float32)
    darkfields = np.zeros((len(z_idx), th, tw), dtype=np.float32)

    n_eff = resolve_workers(n_workers, params.get("backend"), params.get("device"))
    cuda_devices = list_cuda_devices(params.get("device"))
    use_cuda_fanout = (
        len(cuda_devices) > 1
        and params.get("backend") in {"torch", "auto"}
        and (params.get("device") or "cuda").lower().startswith("cuda")
    )

    # Pre-extract per-z tile stacks so workers receive only the slice they
    # need (cheap views into the in-memory mosaic) instead of the whole grid.
    tile_stacks = [mosaic.iter_tiles(z) for z in z_idx]
    if use_cuda_fanout:
        cuda_fit = partial(_fit_one_z_cuda_map, params=params, n_extra_rows=n_extra_rows)
        results = parallel_map_cuda_devices(
            cuda_fit,
            tile_stacks,
            cuda_devices,
            desc="Fitting z-levels (multi-GPU)",
            verbose=verbose,
        )
    else:
        results = parallel_map(
            partial(_fit_one_z, params=params, n_extra_rows=n_extra_rows),
            tile_stacks,
            n_eff,
            desc="Fitting z-levels",
            verbose=verbose,
        )
    for i, (flatfield, darkfield) in enumerate(results):
        flatfields[i, n_extra_rows:, :] = flatfield
        darkfields[i, n_extra_rows:, :] = darkfield

    if field_mode == "global":
        flatfields_out: np.ndarray = flatfields.mean(axis=0)
        darkfields_out: np.ndarray = darkfields.mean(axis=0)
    else:
        flatfields_out = flatfields
        darkfields_out = darkfields

    return MosaicFit(
        flatfields=flatfields_out,
        darkfields=darkfields_out,
        field_mode=field_mode,
        z_indices=z_idx,
        params=params,
    )


def apply_fit(
    mosaic: MosaicGrid,
    fit: MosaicFit,
    *,
    epsilon: float = 1e-6,
    n_extra_rows: int = 0,
) -> np.ndarray:
    """Apply a :class:`MosaicFit` to produce a corrected mosaic volume.

    Parameters
    ----------
    mosaic : MosaicGrid
        The original (uncorrected) mosaic.
    fit : MosaicFit
        Flat/dark-field estimates from :func:`fit_mosaic`.
    epsilon : float
        Stability constant added to the flat-field denominator.
    n_extra_rows : int
        Number of galvo-return rows at the top of each tile to replace by
        edge-extension from the first valid row.  Must match the value used
        in :func:`fit_mosaic`.  Default ``0`` (no replacement).

    Returns
    -------
    numpy.ndarray, shape (Z, H, W), dtype float32
        Corrected mosaic where every tile has been divided by its flat-field
        and had its dark-field subtracted.  If *n_extra_rows* > 0 the galvo
        rows are replaced with the first valid row value so that they do not
        create a visible dark notch at tile boundaries.
    """
    nz = mosaic.n_z
    th, tw = mosaic.tile_shape
    nrows, ncols = mosaic.n_rows, mosaic.n_cols

    corrected = mosaic.array.astype(np.float32).copy()

    for z in range(nz):
        if fit.field_mode == "global":
            ff: np.ndarray = fit.flatfields
            df: np.ndarray = fit.darkfields
        else:
            pos = _nearest_z_pos(z, fit.z_indices)
            ff = fit.flatfields[pos]
            df = fit.darkfields[pos]

        # View the (H, W) plane as a (nrows, th, ncols, tw) tile grid and apply
        # the (th, tw) fields to every tile at once via broadcasting.
        view = corrected[z].reshape(nrows, th, ncols, tw)
        view[...] = (view - df[None, :, None, :]) / (ff[None, :, None, :] + epsilon)
        if n_extra_rows > 0:
            first_valid = view[:, n_extra_rows : n_extra_rows + 1, :, :]
            view[:, :n_extra_rows, :, :] = first_valid
        corrected[z] = view.reshape(nrows * th, ncols * tw)

    return corrected


def save_corrected(
    mosaic: MosaicGrid,
    fit: MosaicFit,
    output_path: str | Path,
    *,
    input_path: str | Path | None = None,
    overwrite: bool = False,
    epsilon: float = 1e-6,
) -> None:
    """Correct *mosaic* with *fit* and save the result as OME-Zarr.

    Parameters
    ----------
    mosaic : MosaicGrid
        Original mosaic grid.
    fit : MosaicFit
        Estimated fields.
    output_path : str or Path
        Destination ``.ome.zarr`` path.
    input_path : str or Path or None
        If provided the axes and scale metadata are copied from the input
        OME-Zarr; otherwise defaults ``(z, y, x)`` / 1.0 are used.
    overwrite : bool
        Overwrite *output_path* if it exists.
    epsilon : float
        Forwarded to :func:`apply_fit`.
    """
    from linum_basic.io.zarr import load_ome_zarr, write_ome_zarr

    corrected = apply_fit(mosaic, fit, epsilon=epsilon)

    axes = ["z", "y", "x"]
    scale = [1.0, 1.0, 1.0]
    if input_path is not None:
        _, axes, scale = load_ome_zarr(input_path)

    write_ome_zarr(output_path, corrected, axes=axes, scale=scale, overwrite=overwrite)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nearest_z_pos(z: int, z_indices: list[int]) -> int:
    """Index into *z_indices* of the element closest to *z*."""
    arr = np.array(z_indices, dtype=np.int64)
    return int(np.argmin(np.abs(arr - z)))
