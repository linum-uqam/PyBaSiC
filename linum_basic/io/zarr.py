"""OME-Zarr I/O for linum-basic (read / write single-scale volumes).

Only the minimal surface needed by :mod:`linum_basic.fit` and the
``basic_fit`` / ``basic_tune`` CLIs is exposed here.  No dependency on
linumpy is introduced; the implementation follows the same OME-Zarr v0.5
patterns used by that library.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import zarr

__all__ = ["load_ome_zarr", "write_ome_zarr"]


def load_ome_zarr(path: str | Path) -> tuple[np.ndarray, list[str], list[float]]:
    """Load an OME-Zarr file into a NumPy array (multiscale level 0).

    Parameters
    ----------
    path : str or Path
        Path to the ``.ome.zarr`` directory.

    Returns
    -------
    array : numpy.ndarray
        The full-resolution volume loaded into memory.
    axes : list of str
        Axis names extracted from the multiscale metadata
        (e.g. ``["z", "y", "x"]``).
    scale : list of float
        Voxel size for each axis at level 0.

    Raises
    ------
    FileNotFoundError
        If *path* is not a valid OME-Zarr store.
    ValueError
        If no Multiscales specification is found in the metadata.
    """
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Multiscales, Reader

    node = parse_url(str(path))
    if node is None:
        raise FileNotFoundError(f"Not a valid OME-Zarr store: {path}")

    reader = Reader(node)
    nodes = list(reader())
    image_node = nodes[0]

    multiscale: Multiscales | None = None
    for spec in image_node.specs:
        if isinstance(spec, Multiscales):
            multiscale = spec
            break
    if multiscale is None:
        raise ValueError(f"No Multiscales spec found in: {path}")

    # Axis names
    axes: list[str] = [ax["name"] for ax in image_node.metadata.get("axes", [])]
    if not axes:
        axes = ["z", "y", "x"]  # safe fallback for 3-D volumes

    # Level-0 array
    level0_path = Path(path) / multiscale.datasets[0]
    arr = zarr.open_array(str(level0_path), mode="r")
    array = np.asarray(arr[:])

    # Voxel scale at level 0
    scale: list[float] = [1.0] * array.ndim
    coord_transforms = image_node.metadata.get("coordinateTransformations", [])
    if coord_transforms:
        for tr in coord_transforms[0]:
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
    overwrite: bool = False,
) -> None:
    """Write an array as a single-scale OME-Zarr v0.5 file (zarr format 3).

    Parameters
    ----------
    path : str or Path
        Output path (e.g. ``corrected.ome.zarr``).
    array : numpy.ndarray
        Volume to write.  Cast to ``float32`` before writing.
    axes : list of str
        Axis names, e.g. ``["z", "y", "x"]``.
    scale : list of float
        Voxel size for each axis.
    overwrite : bool
        Remove *path* if it already exists.

    Raises
    ------
    FileExistsError
        If *path* exists and *overwrite* is ``False``.
    ValueError
        If *axes* and *scale* lengths do not match *array.ndim*.
    """
    if len(axes) != array.ndim or len(scale) != array.ndim:
        raise ValueError(
            f"axes ({len(axes)}) and scale ({len(scale)}) must each have length equal to array.ndim ({array.ndim})."
        )

    out_path = Path(path)
    if out_path.exists():
        if overwrite:
            shutil.rmtree(out_path)
        else:
            raise FileExistsError(f"Output path already exists: {path}. Set overwrite=True to overwrite.")

    store = zarr.storage.LocalStore(str(out_path))
    root = zarr.open_group(store, mode="w", zarr_format=3)
    root.create_array("s0", data=array.astype(np.float32))

    axes_dicts = [{"name": ax, "type": _axis_type(ax), "unit": "millimeter"} for ax in axes]
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "datasets": [
                    {
                        "path": "s0",
                        "coordinateTransformations": [{"type": "scale", "scale": list(scale)}],
                    }
                ],
                "axes": axes_dicts,
            }
        ],
    }


def _axis_type(name: str) -> str:
    """Return the OME-Zarr axis type string for a given axis name."""
    return "channel" if name == "c" else "space"
