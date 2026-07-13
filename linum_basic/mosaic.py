"""Mosaic-grid utilities for tile extraction and seam-pair enumeration.

A *mosaic grid* is a 3-D volume of shape ``(Z, H, W)`` where tiles of
shape ``(th, tw)`` are packed edge-to-edge (no gaps).  Adjacent tiles
physically overlap by a fraction of their size (default 20 %), so the
last ``overlap_px`` columns of the left tile image the same tissue as
the first ``overlap_px`` columns of the right tile.  :class:`MosaicGrid`
exposes these *seam pairs* so downstream code can measure how well a
shading correction aligns neighbouring tiles across their physical overlap
region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import zarr

__all__ = ["MosaicGrid", "SeamPair"]


@dataclass(frozen=True, init=False)
class SeamPair:
    """A pair of adjacent tiles sharing a physical overlap region.

    Attributes
    ----------
    idx_a : int
        Row-major tile index of the *left* (horizontal) or *top* (vertical)
        tile.
    idx_b : int
        Row-major tile index of the *right* (horizontal) or *bottom*
        (vertical) tile.
    slice_a : tuple of slice
        ``(y_slice, x_slice)`` selecting the overlap region *within* tile
        ``idx_a``.
    slice_b : tuple of slice
        ``(y_slice, x_slice)`` selecting the matching overlap region
        *within* tile ``idx_b``.
    orientation : {"horizontal", "vertical"}
        Direction of the seam.
    """

    idx_a: int
    idx_b: int
    slice_a: tuple[slice, slice]
    slice_b: tuple[slice, slice]
    orientation: Literal["horizontal", "vertical"]

    def __init__(
        self,
        idx_a: int,
        idx_b: int,
        slice_a: tuple[slice, slice],
        slice_b: tuple[slice, slice],
        orientation: Literal["horizontal", "vertical"],
    ) -> None:
        """Construct a :class:`SeamPair`.

        Parameters
        ----------
        idx_a : int
            Row-major tile index of the *left* (horizontal) or *top*
            (vertical) tile.
        idx_b : int
            Row-major tile index of the *right* (horizontal) or *bottom*
            (vertical) tile.
        slice_a : tuple of slice
            ``(y_slice, x_slice)`` selecting the overlap region within
            tile ``idx_a``.
        slice_b : tuple of slice
            ``(y_slice, x_slice)`` selecting the matching overlap region
            within tile ``idx_b``.
        orientation : {"horizontal", "vertical"}
            Direction of the seam.
        """
        object.__setattr__(self, "idx_a", idx_a)
        object.__setattr__(self, "idx_b", idx_b)
        object.__setattr__(self, "slice_a", slice_a)
        object.__setattr__(self, "slice_b", slice_b)
        object.__setattr__(self, "orientation", orientation)


@dataclass
class MosaicGrid:
    """A mosaic-grid volume with tile extraction and seam enumeration.

    Parameters
    ----------
    array : numpy.ndarray or zarr.Array, shape (Z, H, W)
        The assembled mosaic volume.  Tiles are packed edge-to-edge; there
        are no gaps between tiles in the stored data.  By default this is a
        materialised :class:`numpy.ndarray`; pass ``lazy=True`` to
        :meth:`from_ome_zarr` to keep the backing array as the level-0
        :class:`zarr.Array` handle (on-disk, chunked) so the volume is not
        loaded into memory up front.
    tile_shape : tuple of int
        ``(tile_height, tile_width)`` of each tile.  Must divide *H* and
        *W* evenly.
    overlap_fraction : float
        Physical overlap between adjacent tiles as a fraction of the tile
        dimension.  Default ``0.2`` (20 %).  This controls how many pixels
        are compared at each seam.

    Notes
    -----
    For the sample OCT mosaic (shape ``(55, 2325, 1200)``, chunks
    ``(55, 75, 75)``) the tile shape is ``(75, 75)`` and the grid is
    31 x 16 tiles giving 496 tiles per z-level.

    The tile at grid position ``(row, col)`` occupies::

        array[z, row*th:(row+1)*th, col*tw:(col+1)*tw]

    and physically overlaps its right neighbour by
    ``round(overlap_fraction * tw)`` pixels along x, and its bottom
    neighbour by ``round(overlap_fraction * th)`` pixels along y.
    """

    array: np.ndarray | zarr.Array
    tile_shape: tuple[int, int]
    overlap_fraction: float = 0.2
    _seam_pairs_cache: list[SeamPair] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:  # noqa: D105
        _z, h, w = self.array.shape
        th, tw = self.tile_shape
        if h % th != 0:
            raise ValueError(f"Mosaic height {h} is not divisible by tile height {th}.")
        if w % tw != 0:
            raise ValueError(f"Mosaic width {w} is not divisible by tile width {tw}.")
        if not (0.0 < self.overlap_fraction < 1.0):
            raise ValueError(f"overlap_fraction must be in (0, 1), got {self.overlap_fraction}.")

    # ------------------------------------------------------------------
    # Grid properties
    # ------------------------------------------------------------------

    @property
    def n_rows(self) -> int:
        """Number of tile rows in the mosaic grid."""
        return self.array.shape[1] // self.tile_shape[0]

    @property
    def n_cols(self) -> int:
        """Number of tile columns in the mosaic grid."""
        return self.array.shape[2] // self.tile_shape[1]

    @property
    def n_tiles(self) -> int:
        """Total number of tiles (``n_rows * n_cols``)."""
        return self.n_rows * self.n_cols

    @property
    def n_z(self) -> int:
        """Number of z-levels."""
        return self.array.shape[0]

    # ------------------------------------------------------------------
    # Tile access
    # ------------------------------------------------------------------

    def iter_tiles(self, z: int) -> np.ndarray:
        """Extract all tiles for z-level *z* as a stacked array.

        Parameters
        ----------
        z : int
            Z-index into ``self.array``.

        Returns
        -------
        numpy.ndarray, shape (n_tiles, th, tw)
            Tiles in row-major order: tile ``r * n_cols + c`` corresponds
            to grid position ``(row=r, col=c)``.
        """
        th, tw = self.tile_shape
        nrows, ncols = self.n_rows, self.n_cols
        # Reshape-transpose is equivalent to the nested loop but avoids Python
        # iteration over all nrows*ncols tiles. The transpose creates a
        # non-contiguous view; the final reshape forces a contiguous copy.
        # Materialise the z-plane to a concrete ndarray: indexing a zarr.Array
        # reads only that plane from disk; np.asarray is a no-op for an already
        # materialised ndarray (eager path).
        plane: np.ndarray = np.asarray(self.array[z])  # (nrows*th, ncols*tw)
        reshaped: np.ndarray = plane.reshape(nrows, th, ncols, tw)
        # Reorder tile-grid axes (nrows, th, ncols, tw) -> (nrows, ncols, th, tw)
        # so flattening the first two yields row-major tile order.
        transposed = reshaped.transpose(0, 2, 1, 3)
        return transposed.reshape(nrows * ncols, th, tw)

    def get_tile(self, z: int, row: int, col: int) -> np.ndarray:
        """Return the (th, tw) tile at grid position ``(row, col)``.

        Parameters
        ----------
        z : int
            Z-index.
        row : int
            Tile row (0-based).
        col : int
            Tile column (0-based).

        Returns
        -------
        numpy.ndarray
            Shape ``(th, tw)``.
        """
        th, tw = self.tile_shape
        # Indexing a zarr.Array reads only the requested region from disk;
        # np.asarray normalises to ndarray (no-op for the eager ndarray path).
        return np.asarray(self.array[z, row * th : (row + 1) * th, col * tw : (col + 1) * tw])

    # ------------------------------------------------------------------
    # Seam enumeration
    # ------------------------------------------------------------------

    def seam_pairs(self) -> list[SeamPair]:
        """Return all seam pairs for the grid.

        Results are cached after the first call.

        Returns
        -------
        list of SeamPair
            Horizontal seams (left-right neighbours) followed by vertical
            seams (top-bottom neighbours).  Total count:
            ``n_rows*(n_cols-1) + (n_rows-1)*n_cols``.
        """
        if self._seam_pairs_cache is not None:
            return self._seam_pairs_cache

        th, tw = self.tile_shape
        nrows, ncols = self.n_rows, self.n_cols
        overlap_y = max(1, round(self.overlap_fraction * th))
        overlap_x = max(1, round(self.overlap_fraction * tw))

        pairs: list[SeamPair] = []

        # Horizontal seams: tile (r, c) right edge ↔ tile (r, c+1) left edge
        for r in range(nrows):
            for c in range(ncols - 1):
                idx_a = r * ncols + c
                idx_b = r * ncols + c + 1
                # Last overlap_x columns of left tile
                slice_a = (slice(None), slice(tw - overlap_x, None))
                # First overlap_x columns of right tile
                slice_b = (slice(None), slice(0, overlap_x))
                pairs.append(SeamPair(idx_a, idx_b, slice_a, slice_b, "horizontal"))

        # Vertical seams: tile (r, c) bottom edge ↔ tile (r+1, c) top edge
        for r in range(nrows - 1):
            for c in range(ncols):
                idx_a = r * ncols + c
                idx_b = (r + 1) * ncols + c
                # Last overlap_y rows of top tile
                slice_a = (slice(th - overlap_y, None), slice(None))
                # First overlap_y rows of bottom tile
                slice_b = (slice(0, overlap_y), slice(None))
                pairs.append(SeamPair(idx_a, idx_b, slice_a, slice_b, "vertical"))

        # Cache so repeated calls are O(1).
        self._seam_pairs_cache = pairs
        return pairs

    # ------------------------------------------------------------------
    # Convenience class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_ome_zarr(
        cls,
        path: str,
        *,
        overlap_fraction: float = 0.2,
        lazy: bool = False,
    ) -> MosaicGrid:
        """Load a mosaic grid from an OME-Zarr file.

        Parameters
        ----------
        path : str or Path
            Path to the ``.ome.zarr`` directory.
        overlap_fraction : float
            Physical overlap fraction between adjacent tiles.  Default
            ``0.2``.
        lazy : bool, optional
            When ``True`` (default ``False``) the backing array is the
            level-0 :class:`zarr.Array` handle (on-disk, chunked) instead of a
            fully materialised :class:`numpy.ndarray`.  Tile geometry and
            per-z / per-tile reads (``iter_tiles``, ``get_tile``, indexing)
            work identically; nothing is read from disk until those calls are
            made.  Use this for large volumes that should not be loaded into
            memory up front (see the streaming fit path).

        Returns
        -------
        MosaicGrid

        Notes
        -----
        The tile shape is inferred from the zarr chunk shape at level 0
        (last two dimensions, i.e. ``chunk_shape[-2:]``).  This matches
        the convention used by linumpy's mosaic-grid writer where tiles
        are chunked individually along y and x but the full z-stack is a
        single chunk.

        When ``lazy`` is ``False`` (default) the full volume is read into
        memory exactly as before (backward compatible).  When ``lazy`` is
        ``True`` the returned :class:`zarr.Array` is reused to read the
        chunk grid, so the store is opened once rather than twice.
        """
        import zarr

        from linum_basic.io.zarr import load_ome_zarr

        # When lazy, load_ome_zarr returns the level-0 zarr.Array handle;
        # otherwise it materialises the full volume into an ndarray.
        array, _axes, _scale = load_ome_zarr(path, lazy=lazy)

        # Infer tile shape from the level-0 chunk grid.  When lazy, the
        # returned handle IS the level-0 zarr.Array and exposes ``.chunks``
        # directly; when eager (ndarray) resolve the level-0 sub-path via
        # OME-NGFF metadata and re-open the array purely for its chunks.
        if isinstance(array, zarr.Array):
            chunk_shape = array.chunks
        else:
            from pathlib import Path as _Path

            from ome_zarr.io import parse_url
            from ome_zarr.reader import Multiscales, Reader

            node = parse_url(str(path))
            if node is None:
                raise FileNotFoundError(f"Not a valid OME-Zarr store: {path}")
            reader = Reader(node)
            image_node = next(iter(reader()))
            level0_subpath = "s0"  # fallback
            for spec in image_node.specs:
                if isinstance(spec, Multiscales):
                    level0_subpath = spec.datasets[0]
                    break
            arr_meta = zarr.open_array(str(_Path(path) / level0_subpath), mode="r")
            chunk_shape = arr_meta.chunks  # tuple, e.g. (55, 75, 75)

        tile_shape: tuple[int, int] = (int(chunk_shape[-2]), int(chunk_shape[-1]))

        return cls(array=array, tile_shape=tile_shape, overlap_fraction=overlap_fraction)
