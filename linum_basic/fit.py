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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from tqdm.auto import tqdm

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
    model = BaSiC.from_array(tiles, **init_kw)
    for key, val in post_kw.items():
        setattr(model, key, val)
    return model


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
    verbose : bool
        Show a progress bar over z-levels.

    Returns
    -------
    MosaicFit
        Fitted flat/dark-fields for each requested z-level.
    """
    params: dict[str, Any] = dict(basic_kwargs or {})
    z_idx = list(z_indices) if z_indices is not None else list(range(mosaic.n_z))

    th, tw = mosaic.tile_shape
    flatfields = np.ones((len(z_idx), th, tw), dtype=np.float32)
    darkfields = np.zeros((len(z_idx), th, tw), dtype=np.float32)

    for i, z in enumerate(tqdm(z_idx, desc="Fitting z-levels", disable=not verbose)):
        tiles = mosaic.iter_tiles(z)
        if n_extra_rows > 0:
            tiles = tiles[:, n_extra_rows:, :]
        model = make_model(tiles, params)
        model.prepare()
        model.run()
        flatfields[i, n_extra_rows:, :] = model.get_flatfield()
        darkfields[i, n_extra_rows:, :] = model.get_darkfield()

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

        for r in range(nrows):
            for c in range(ncols):
                tile = corrected[z, r * th : (r + 1) * th, c * tw : (c + 1) * tw]
                corrected[z, r * th : (r + 1) * th, c * tw : (c + 1) * tw] = (tile - df) / (ff + epsilon)
                if n_extra_rows > 0:
                    first_valid = corrected[z, r * th + n_extra_rows, c * tw : (c + 1) * tw]
                    corrected[z, r * th : r * th + n_extra_rows, c * tw : (c + 1) * tw] = first_valid[np.newaxis, :]

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
