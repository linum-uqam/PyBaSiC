"""Tests for linum_basic.mosaic — MosaicGrid and SeamPair."""

from __future__ import annotations

import numpy as np
import pytest

from linum_basic.mosaic import MosaicGrid, SeamPair


def _make_mosaic(n_rows=4, n_cols=5, tile_h=10, tile_w=8, n_z=3, overlap_fraction=0.2) -> MosaicGrid:
    """Build a synthetic mosaic with a simple gradient pattern."""
    h = n_rows * tile_h
    w = n_cols * tile_w
    rng = np.random.default_rng(42)
    arr = rng.random((n_z, h, w), dtype=np.float32)
    return MosaicGrid(arr, tile_shape=(tile_h, tile_w), overlap_fraction=overlap_fraction)


def _write_mosaic_store(
    store_path,
    *,
    n_z: int,
    n_rows: int,
    n_cols: int,
    th: int,
    tw: int,
    seed: int = 42,
) -> np.ndarray:
    """Write a synthetic mosaic as an OME-Zarr store chunked by tile.

    The full z-stack is a single chunk along z; y/x chunks equal the tile
    shape so ``MosaicGrid.from_ome_zarr`` can infer ``tile_shape`` from the
    chunk grid. Returns the underlying ndarray for value comparison.
    """
    import zarr

    rng = np.random.default_rng(seed)
    arr = rng.random((n_z, n_rows * th, n_cols * tw), dtype=np.float32)
    grp = zarr.open_group(str(store_path), mode="w", zarr_format=3)
    grp.create_array("s0", data=arr, chunks=(n_z, th, tw))
    grp.attrs["ome"] = {
        "multiscales": [
            {
                "version": "0.5",
                "axes": [{"name": ax, "type": "space"} for ax in ["z", "y", "x"]],
                "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]}],
            }
        ]
    }
    return arr


class TestMosaicGridProperties:
    def test_n_rows(self):
        m = _make_mosaic(n_rows=4, n_cols=5, tile_h=10, tile_w=8)
        assert m.n_rows == 4

    def test_n_cols(self):
        m = _make_mosaic(n_rows=4, n_cols=5, tile_h=10, tile_w=8)
        assert m.n_cols == 5

    def test_n_tiles(self):
        m = _make_mosaic(n_rows=4, n_cols=5)
        assert m.n_tiles == 20

    def test_n_z(self):
        m = _make_mosaic(n_z=7)
        assert m.n_z == 7

    def test_bad_height(self):
        arr = np.zeros((1, 11, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="height"):
            MosaicGrid(arr, tile_shape=(10, 8))

    def test_bad_width(self):
        arr = np.zeros((1, 10, 9), dtype=np.float32)
        with pytest.raises(ValueError, match="width"):
            MosaicGrid(arr, tile_shape=(10, 8))

    def test_bad_overlap_fraction(self):
        arr = np.zeros((1, 10, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="overlap_fraction"):
            MosaicGrid(arr, tile_shape=(10, 8), overlap_fraction=1.5)


class TestIterTiles:
    def test_shape(self):
        m = _make_mosaic(n_rows=4, n_cols=5, tile_h=10, tile_w=8, n_z=3)
        tiles = m.iter_tiles(0)
        assert tiles.shape == (20, 10, 8)

    def test_row_major_order(self):
        """Tile at (row, col) must be tiles[row * n_cols + col]."""
        m = _make_mosaic(n_rows=4, n_cols=5, tile_h=10, tile_w=8, n_z=3)
        for z in range(m.n_z):
            tiles = m.iter_tiles(z)
            for r in range(m.n_rows):
                for c in range(m.n_cols):
                    expected = m.get_tile(z, r, c)
                    np.testing.assert_array_equal(tiles[r * m.n_cols + c], expected)

    def test_get_tile_matches_array_slice(self):
        m = _make_mosaic(n_rows=3, n_cols=4, tile_h=6, tile_w=6, n_z=2)
        th, tw = m.tile_shape
        for z in range(m.n_z):
            for r in range(m.n_rows):
                for c in range(m.n_cols):
                    tile = m.get_tile(z, r, c)
                    direct = m.array[z, r * th : (r + 1) * th, c * tw : (c + 1) * tw]
                    np.testing.assert_array_equal(tile, direct)


class TestSeamPairs:
    def test_count(self):
        """n_rows*(n_cols-1) horizontal + (n_rows-1)*n_cols vertical seams."""
        n_rows, n_cols = 4, 5
        m = _make_mosaic(n_rows=n_rows, n_cols=n_cols, tile_h=10, tile_w=8)
        expected = n_rows * (n_cols - 1) + (n_rows - 1) * n_cols
        assert len(m.seam_pairs()) == expected

    def test_horizontal_overlap_size(self):
        """Each horizontal seam should expose overlap_x columns on both sides."""
        m = _make_mosaic(n_rows=3, n_cols=4, tile_h=10, tile_w=10, overlap_fraction=0.3)
        overlap_x = round(0.3 * 10)  # 3
        tiles = m.iter_tiles(0)
        for sp in m.seam_pairs():
            if sp.orientation != "horizontal":
                continue
            a_patch = tiles[sp.idx_a][sp.slice_a]
            b_patch = tiles[sp.idx_b][sp.slice_b]
            assert a_patch.shape[1] == overlap_x
            assert b_patch.shape[1] == overlap_x

    def test_vertical_overlap_size(self):
        """Each vertical seam should expose overlap_y rows on both sides."""
        m = _make_mosaic(n_rows=4, n_cols=3, tile_h=12, tile_w=8, overlap_fraction=0.25)
        overlap_y = round(0.25 * 12)  # 3
        tiles = m.iter_tiles(0)
        for sp in m.seam_pairs():
            if sp.orientation != "vertical":
                continue
            a_patch = tiles[sp.idx_a][sp.slice_a]
            b_patch = tiles[sp.idx_b][sp.slice_b]
            assert a_patch.shape[0] == overlap_y
            assert b_patch.shape[0] == overlap_y

    def test_caching(self):
        """seam_pairs() returns the same list object on repeated calls."""
        m = _make_mosaic()
        pairs_first = m.seam_pairs()
        pairs_second = m.seam_pairs()
        # Same object (no recomputation) and same content
        assert pairs_first is pairs_second
        assert len(pairs_first) > 0

    def test_seam_pair_frozen(self):
        """SeamPair is immutable (frozen dataclass)."""
        sp = SeamPair(0, 1, (slice(None), slice(0, 2)), (slice(None), slice(7, 9)), "horizontal")
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            sp.idx_a = 99  # type: ignore[misc]

    def test_horizontal_index_adjacency(self):
        """Left tile idx_a and right tile idx_b are always column-adjacent."""
        m = _make_mosaic(n_rows=3, n_cols=4, tile_h=8, tile_w=8)
        for sp in m.seam_pairs():
            if sp.orientation != "horizontal":
                continue
            row_a, col_a = divmod(sp.idx_a, m.n_cols)
            row_b, col_b = divmod(sp.idx_b, m.n_cols)
            assert row_a == row_b
            assert col_b == col_a + 1

    def test_vertical_index_adjacency(self):
        """Top tile idx_a and bottom tile idx_b are always row-adjacent."""
        m = _make_mosaic(n_rows=3, n_cols=4, tile_h=8, tile_w=8)
        for sp in m.seam_pairs():
            if sp.orientation != "vertical":
                continue
            row_a, col_a = divmod(sp.idx_a, m.n_cols)
            row_b, col_b = divmod(sp.idx_b, m.n_cols)
            assert col_a == col_b
            assert row_b == row_a + 1


class TestFromOmeZarr:
    def test_from_ome_zarr(self, tmp_path):
        """MosaicGrid.from_ome_zarr infers tile_shape from chunk metadata."""
        n_z, n_rows, n_cols, th, tw = 3, 4, 5, 10, 8
        store = str(tmp_path / "mosaic3.ome.zarr")
        _write_mosaic_store(store, n_z=n_z, n_rows=n_rows, n_cols=n_cols, th=th, tw=tw)
        mosaic = MosaicGrid.from_ome_zarr(store, overlap_fraction=0.2)
        assert mosaic.tile_shape == (th, tw)
        assert mosaic.n_rows == n_rows
        assert mosaic.n_cols == n_cols
        assert mosaic.n_z == n_z


class TestFromOmeZarrLazy:
    """lazy=True builds a MosaicGrid over a zarr.Array handle, not a materialised volume."""

    def test_lazy_backing_array_is_zarr_handle(self, tmp_path):
        """from_ome_zarr(lazy=True) stores a zarr.Array, not an ndarray."""
        import zarr

        store = tmp_path / "lazy.ome.zarr"
        _write_mosaic_store(store, n_z=3, n_rows=4, n_cols=5, th=10, tw=8)
        mosaic = MosaicGrid.from_ome_zarr(str(store), lazy=True)

        assert isinstance(mosaic.array, zarr.Array)
        assert not isinstance(mosaic.array, np.ndarray)

    def test_eager_default_backing_array_is_ndarray(self, tmp_path):
        """Without lazy=, from_ome_zarr materialises the full volume (backward compat)."""
        store = tmp_path / "eager.ome.zarr"
        _write_mosaic_store(store, n_z=2, n_rows=3, n_cols=4, th=8, tw=8)
        mosaic = MosaicGrid.from_ome_zarr(str(store))

        assert isinstance(mosaic.array, np.ndarray)

    def test_lazy_tile_shape_inferred_from_chunks(self, tmp_path):
        """Tile shape is inferred from chunk metadata without reading pixel data."""
        n_z, n_rows, n_cols, th, tw = 3, 4, 5, 10, 8
        store = tmp_path / "m.ome.zarr"
        _write_mosaic_store(store, n_z=n_z, n_rows=n_rows, n_cols=n_cols, th=th, tw=tw)
        mosaic = MosaicGrid.from_ome_zarr(str(store), lazy=True)

        assert mosaic.tile_shape == (th, tw)

    def test_lazy_grid_geometry_matches_eager(self, tmp_path):
        """n_rows/n_cols/n_z/n_tiles are identical between lazy and eager grids."""
        n_z, n_rows, n_cols, th, tw = 4, 3, 6, 12, 10
        store = tmp_path / "geo.ome.zarr"
        _write_mosaic_store(store, n_z=n_z, n_rows=n_rows, n_cols=n_cols, th=th, tw=tw)

        lazy = MosaicGrid.from_ome_zarr(str(store), lazy=True)
        eager = MosaicGrid.from_ome_zarr(str(store))

        assert (lazy.n_rows, lazy.n_cols, lazy.n_z, lazy.n_tiles) == (
            eager.n_rows,
            eager.n_cols,
            eager.n_z,
            eager.n_tiles,
        )
        assert (lazy.n_rows, lazy.n_cols, lazy.n_z, lazy.n_tiles) == (n_rows, n_cols, n_z, n_rows * n_cols)

    def test_lazy_iter_tiles_matches_eager(self, tmp_path):
        """iter_tiles(z) reads identical tile values through the lazy handle as eager."""
        n_z, n_rows, n_cols, th, tw = 3, 4, 5, 10, 8
        store = tmp_path / "tiles.ome.zarr"
        _write_mosaic_store(store, n_z=n_z, n_rows=n_rows, n_cols=n_cols, th=th, tw=tw)

        lazy = MosaicGrid.from_ome_zarr(str(store), lazy=True)
        eager = MosaicGrid.from_ome_zarr(str(store))

        for z in range(n_z):
            lazy_tiles = lazy.iter_tiles(z)
            eager_tiles = eager.iter_tiles(z)
            # iter_tiles always materialises a contiguous ndarray of tiles.
            assert isinstance(lazy_tiles, np.ndarray)
            assert lazy_tiles.shape == (n_rows * n_cols, th, tw)
            np.testing.assert_array_equal(lazy_tiles, eager_tiles)

    def test_lazy_get_tile_matches_eager(self, tmp_path):
        """get_tile reads a single tile through the lazy handle correctly."""
        n_z, n_rows, n_cols, th, tw = 2, 3, 4, 8, 8
        store = tmp_path / "onetile.ome.zarr"
        _write_mosaic_store(store, n_z=n_z, n_rows=n_rows, n_cols=n_cols, th=th, tw=tw)

        lazy = MosaicGrid.from_ome_zarr(str(store), lazy=True)
        eager = MosaicGrid.from_ome_zarr(str(store))

        for z in range(n_z):
            for r in range(n_rows):
                for c in range(n_cols):
                    np.testing.assert_array_equal(
                        lazy.get_tile(z, r, c),
                        eager.get_tile(z, r, c),
                    )

    def test_lazy_invalid_store_raises(self, tmp_path):
        """lazy=True still raises FileNotFoundError for a missing store."""
        with pytest.raises(FileNotFoundError):
            MosaicGrid.from_ome_zarr(str(tmp_path / "nope.ome.zarr"), lazy=True)
