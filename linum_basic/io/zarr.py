"""OME-Zarr I/O for linum-basic.

Read and write OME-Zarr v0.5 image pyramids via the ``ome-zarr`` library.
The write path supports multi-resolution pyramids via the *n_levels* parameter.
Only the minimal surface needed by :mod:`linum_basic.fit` and the
``basic fit`` / ``basic tune`` sub-commands is exposed here.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import zarr
import zarr.storage
from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Reader
from ome_zarr.writer import write_image

__all__ = ["load_ome_zarr", "write_ome_zarr"]


def load_ome_zarr(path: str | Path, *, lazy: bool = False) -> tuple[np.ndarray | zarr.Array, list[str], list[float]]:
    """Load an OME-Zarr file (multiscale level 0).

    By default the level-0 volume is fully materialised into a NumPy array
    (eager read).  Pass ``lazy=True`` to get back the underlying
    :class:`zarr.Array` handle instead — on-disk and chunked — so callers can
    stream planes or tiles without loading the whole volume into memory.  The
    lazy handle supports NumPy-style indexing (``handle[z]``,
    ``handle[z, y0:y1, x0:x1]``); each index materialises only the requested
    region into a :class:`numpy.ndarray`.

    Parameters
    ----------
    path : str or Path
        Path to the ``.ome.zarr`` directory.
    lazy : bool, optional
        When ``True`` (default ``False``), return the level-0
        :class:`zarr.Array` handle instead of a materialised NumPy array.
        The handle shares the on-disk store's chunk grid; nothing is read
        until it is indexed.

    Returns
    -------
    array : numpy.ndarray or zarr.Array
        When ``lazy`` is ``False`` (default), the full-resolution volume
        (level 0) loaded into memory as a :class:`numpy.ndarray`.
        When ``lazy`` is ``True``, the level-0 :class:`zarr.Array` handle
        (on-disk, chunked) — index it to read specific planes or regions.
    axes : list of str
        Axis names extracted from the multiscale metadata
        (e.g. ``["z", "y", "x"]``).
    scale : list of float
        Physical voxel size for each axis at level 0.

    Raises
    ------
    FileNotFoundError
        If *path* is not a valid OME-Zarr store.
    ValueError
        If no Multiscales specification is found in the metadata.
    """
    node = parse_url(str(path))
    if node is None:
        raise FileNotFoundError(f"Not a valid OME-Zarr store: {path}")

    image_node = next(iter(Reader(node)()))
    multiscale: Multiscales | None = next((s for s in image_node.specs if isinstance(s, Multiscales)), None)
    if multiscale is None:
        raise ValueError(f"No Multiscales spec found in: {path}")

    # Level-0 array handle.  Kept as the on-disk zarr.Array so callers can opt
    # into lazy streaming (indexing materialises only the requested region).
    arr = zarr.open_array(str(Path(path) / multiscale.datasets[0]), mode="r")
    array: np.ndarray | zarr.Array = arr if lazy else np.asarray(arr[:])

    # Axes and scale from the OME-Zarr attrs.  The spec keeps metadata under
    # root.attrs["ome"]["multiscales"] (v0.4+), but some writers (and older
    # stores) use the bare top-level "multiscales" key instead.  Try both so
    # we don't silently fall back to guessed axes/scale for valid stores.
    root_attrs: dict = dict(zarr.open_group(str(path), mode="r").attrs)  # type: ignore[arg-type]
    ms_meta: dict = (root_attrs.get("ome", {}).get("multiscales") or root_attrs.get("multiscales") or [{}])[0]

    axes_meta = ms_meta.get("axes", [])
    # When axes metadata is absent, derive a fallback from ndim so 2-D data
    # gets ["y", "x"] instead of always returning the 3-D ["z", "y", "x"] list.
    _fallback_axes = ["t", "c", "z", "y", "x"]
    ndim = arr.ndim
    axes: list[str] = [ax["name"] for ax in axes_meta] if axes_meta else _fallback_axes[-ndim:]

    scale: list[float] = [1.0] * ndim
    for tr in (ms_meta.get("datasets", [{}])[0]).get("coordinateTransformations", []):
        if tr.get("type") == "scale":
            scale = list(tr["scale"])
            break

    return array, axes, scale


def write_ome_zarr(
    path: str | Path,
    array: np.ndarray,
    *,
    axes: list[str],
    scale: list[float],
    n_levels: int = 1,
    volumetric: bool = False,
    overwrite: bool = False,
) -> None:
    """Write an array as a multi-resolution OME-Zarr v0.5 file.

    Uses :func:`ome_zarr.writer.write_image` to build the image pyramid.
    Each successive level downsamples by a factor of 2 using the ``resize``
    method.

    Parameters
    ----------
    path : str or Path
        Output path (e.g. ``corrected.ome.zarr``).
    array : numpy.ndarray
        Volume to write.  Stored as ``float32``.
    axes : list of str
        Axis names, e.g. ``["z", "y", "x"]``.
    scale : list of float
        Physical voxel size for each axis at level 0.
    n_levels : int
        Number of resolution levels to write (including the full-resolution
        level 0).  Must be >= 1.  When *n_levels* is 1 (default) a
        single-scale store is written with no pyramid.
    volumetric : bool
        When ``True`` all spatial axes (including ``z``) are downsampled 2x
        at each pyramid level.  When ``False`` (default) only the in-plane
        axes (``y``/``x``) are downsampled, keeping the z-stack intact.
    overwrite : bool
        Remove *path* if it already exists.

    Raises
    ------
    FileExistsError
        If *path* exists and *overwrite* is ``False``.
    ValueError
        If *axes* / *scale* lengths do not match *array.ndim*, or
        *n_levels* < 1.
    """
    if len(axes) != array.ndim or len(scale) != array.ndim:
        raise ValueError(
            f"axes ({len(axes)}) and scale ({len(scale)}) must each have length equal to array.ndim ({array.ndim})."
        )
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}.")

    out_path = Path(path)
    if out_path.exists():
        if overwrite:
            shutil.rmtree(out_path)
        else:
            raise FileExistsError(f"Output path already exists: {path}. Set overwrite=True to overwrite.")

    store = zarr.storage.LocalStore(str(out_path))
    root = zarr.open_group(store, mode="w", zarr_format=3)

    axes_dicts = [
        {"name": ax, "type": _axis_type(ax)} | ({"unit": "millimeter"} if _axis_type(ax) == "space" else {}) for ax in axes
    ]

    # Build scale_factors for write_image.
    # volumetric=True: all spatial axes downsampled 2x per level (cumulative dict format).
    # volumetric=False: integer format -> write_image only downsamples y/x.
    if n_levels == 1:
        extra_sf: tuple[()] | list[dict[str, int]] = ()
    elif volumetric:
        spatial_axes = [ax for ax in axes if ax not in ("c", "t")]
        extra_sf = [dict.fromkeys(spatial_axes, 2**i) for i in range(1, n_levels)]
    else:
        extra_sf = tuple(2**i for i in range(1, n_levels))  # type: ignore[assignment]

    # write_image auto-generates coordinate_transformations based on shape
    # ratios; we then patch them to reflect the real physical scale.
    write_image(
        array.astype(np.float32),
        root,
        scale_factors=extra_sf,
        axes=axes_dicts,
    )

    # Patch coordinate_transformations so every level carries the correct
    # physical voxel size (scale_i = scale_0 * shape_0[d] / shape_i[d]).
    # Try both the OME-NGFF v0.4+ wrapper key and the bare top-level key.
    root_attrs_w: dict = dict(root.attrs)  # type: ignore[arg-type]
    ms_meta: dict = (root_attrs_w.get("ome", {}).get("multiscales") or root_attrs_w.get("multiscales") or [{}])[0]
    ms_meta["axes"] = axes_dicts
    s0_shape = np.array(array.shape, dtype=float)
    for ds in ms_meta.get("datasets", []):
        lvl_arr = zarr.open_array(str(out_path / ds["path"]), mode="r")
        lvl_shape = np.array(lvl_arr.shape, dtype=float)
        phys_scale = [float(scale[d] * s0_shape[d] / lvl_shape[d]) for d in range(array.ndim)]
        ds["coordinateTransformations"] = [{"type": "scale", "scale": phys_scale}]

    root.attrs["ome"] = {"version": "0.5", "multiscales": [ms_meta]}


def _axis_type(name: str) -> str:
    """Return the OME-Zarr axis type string for a given axis name."""
    if name == "c":
        return "channel"
    if name == "t":
        return "time"
    return "space"
