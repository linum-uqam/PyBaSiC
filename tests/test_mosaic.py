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
        import zarr

        n_z, n_rows, n_cols, th, tw = 3, 4, 5, 10, 8
        arr = np.random.rand(n_z, n_rows * th, n_cols * tw).astype(np.float32)
        # Write with explicit zarr chunks = tile_shape; bypass write_ome_zarr to control chunks
        store2 = str(tmp_path / "mosaic3.ome.zarr")
        grp = zarr.open_group(store2, mode="w", zarr_format=3)
        grp.create_array("s0", data=arr, chunks=(n_z, th, tw))
        # Write OME metadata
        grp.attrs["ome"] = {
            "multiscales": [
                {
                    "version": "0.5",
                    "axes": [{"name": ax, "type": "space"} for ax in ["z", "y", "x"]],
                    "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]}],
                }
            ]
        }
        mosaic = MosaicGrid.from_ome_zarr(store2, overlap_fraction=0.2)
        assert mosaic.tile_shape == (th, tw)
        assert mosaic.n_rows == n_rows
        assert mosaic.n_cols == n_cols
        assert mosaic.n_z == n_z
