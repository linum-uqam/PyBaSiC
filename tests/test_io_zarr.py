"""Tests for linum_basic.io.zarr — OME-Zarr read/write round-trip."""

from __future__ import annotations

import numpy as np
import pytest


def test_write_read_roundtrip(tmp_path):
    """write_ome_zarr + load_ome_zarr round-trips shape, dtype, axes, scale."""
    from linum_basic.io.zarr import load_ome_zarr, write_ome_zarr

    arr = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    axes = ["z", "y", "x"]
    scale = [0.5, 0.01, 0.01]
    out = tmp_path / "test.ome.zarr"

    write_ome_zarr(out, arr, axes=axes, scale=scale)
    loaded, loaded_axes, loaded_scale = load_ome_zarr(out)

    np.testing.assert_array_equal(loaded, arr)
    assert loaded_axes == axes
    assert loaded_scale == pytest.approx(scale)


def test_overwrite_raises_by_default(tmp_path):
    """Writing to an existing path without overwrite=True raises."""
    from linum_basic.io.zarr import write_ome_zarr

    arr = np.ones((2, 3, 4), dtype=np.float32)
    out = tmp_path / "test.ome.zarr"
    write_ome_zarr(out, arr, axes=["z", "y", "x"], scale=[1.0, 1.0, 1.0])

    with pytest.raises(FileExistsError):
        write_ome_zarr(out, arr, axes=["z", "y", "x"], scale=[1.0, 1.0, 1.0])


def test_overwrite_true(tmp_path):
    """overwrite=True silently replaces an existing store."""
    from linum_basic.io.zarr import load_ome_zarr, write_ome_zarr

    arr1 = np.ones((2, 3, 4), dtype=np.float32)
    arr2 = arr1 * 2
    out = tmp_path / "test.ome.zarr"
    write_ome_zarr(out, arr1, axes=["z", "y", "x"], scale=[1.0, 1.0, 1.0])
    write_ome_zarr(out, arr2, axes=["z", "y", "x"], scale=[1.0, 1.0, 1.0], overwrite=True)
    loaded, _, _ = load_ome_zarr(out)
    np.testing.assert_array_equal(loaded, arr2)


def test_dtype_preserved(tmp_path):
    """write_ome_zarr always stores as float32 (by design); values must be preserved."""
    from linum_basic.io.zarr import load_ome_zarr, write_ome_zarr

    arr = np.array([[[100, 200], [300, 400]]], dtype=np.uint16)
    out = tmp_path / "uint16.ome.zarr"
    write_ome_zarr(out, arr, axes=["z", "y", "x"], scale=[1.0, 1.0, 1.0])
    loaded, _, _ = load_ome_zarr(out)
    # Values must survive even though storage dtype is float32
    np.testing.assert_array_equal(loaded, arr.astype(np.float32))
    assert loaded.dtype == np.float32


def test_2d_array(tmp_path):
    """2-D arrays (single-channel images) can be stored and retrieved."""
    from linum_basic.io.zarr import load_ome_zarr, write_ome_zarr

    arr = np.random.rand(8, 8).astype(np.float32)
    out = tmp_path / "2d.ome.zarr"
    write_ome_zarr(out, arr, axes=["y", "x"], scale=[1.0, 1.0])
    loaded, axes, _ = load_ome_zarr(out)
    np.testing.assert_array_equal(loaded, arr)
    assert axes == ["y", "x"]


def test_n_levels_pyramid(tmp_path):
    """n_levels > 1 with volumetric=False only downsamples y/x (default)."""
    import zarr

    from linum_basic.io.zarr import load_ome_zarr, write_ome_zarr

    arr = np.ones((16, 64, 64), dtype=np.float32)
    out = tmp_path / "pyramid.ome.zarr"
    write_ome_zarr(out, arr, axes=["z", "y", "x"], scale=[1.0, 0.5, 0.5], n_levels=3)

    # load_ome_zarr always returns level 0
    loaded, axes, scale = load_ome_zarr(out)
    np.testing.assert_array_equal(loaded, arr)
    assert axes == ["z", "y", "x"]
    assert scale == pytest.approx([1.0, 0.5, 0.5])

    # 3 levels; z unchanged, y/x halved per level
    root = zarr.open_group(str(out), mode="r")
    root_meta: dict = dict(root.attrs)  # type: ignore[arg-type]
    ms = root_meta["ome"]["multiscales"][0]["datasets"]
    assert len(ms) == 3
    assert root[ms[1]["path"]].shape == (16, 32, 32)
    assert root[ms[2]["path"]].shape == (16, 16, 16)
    # Physical scale doubles in y/x at each level; z unchanged
    assert ms[1]["coordinateTransformations"][0]["scale"] == pytest.approx([1.0, 1.0, 1.0])
    assert ms[2]["coordinateTransformations"][0]["scale"] == pytest.approx([1.0, 2.0, 2.0])


def test_n_levels_pyramid_volumetric(tmp_path):
    """volumetric=True downsamples z/y/x at every pyramid level."""
    import zarr

    from linum_basic.io.zarr import write_ome_zarr

    arr = np.ones((32, 64, 64), dtype=np.float32)
    out = tmp_path / "vol_pyramid.ome.zarr"
    write_ome_zarr(
        out,
        arr,
        axes=["z", "y", "x"],
        scale=[2.0, 0.5, 0.5],
        n_levels=3,
        volumetric=True,
    )

    root = zarr.open_group(str(out), mode="r")
    root_meta: dict = dict(root.attrs)  # type: ignore[arg-type]
    ms = root_meta["ome"]["multiscales"][0]["datasets"]
    assert len(ms) == 3
    # All axes downsampled 2x per level
    assert root[ms[1]["path"]].shape == (16, 32, 32)
    assert root[ms[2]["path"]].shape == (8, 16, 16)
    # Physical scale doubles in all dims at each level
    assert ms[1]["coordinateTransformations"][0]["scale"] == pytest.approx([4.0, 1.0, 1.0])
    assert ms[2]["coordinateTransformations"][0]["scale"] == pytest.approx([8.0, 2.0, 2.0])


def test_axes_metadata_units(tmp_path):
    """Per OME-NGFF 0.5: space axes get 'unit', channel and time axes must not."""
    import zarr

    from linum_basic.io.zarr import write_ome_zarr

    arr = np.ones((2, 3, 8, 8), dtype=np.float32)
    out = tmp_path / "tcyx.ome.zarr"
    write_ome_zarr(out, arr, axes=["t", "c", "y", "x"], scale=[1.0, 1.0, 0.5, 0.5])

    root = zarr.open_group(str(out), mode="r")
    root_meta: dict = dict(root.attrs)  # type: ignore[arg-type]
    axes = root_meta["ome"]["multiscales"][0]["axes"]
    by_name = {a["name"]: a for a in axes}

    # time and channel axes MUST NOT have a unit (not a valid space unit)
    assert "unit" not in by_name["t"], "time axis must not carry a space unit"
    assert "unit" not in by_name["c"], "channel axis must not carry a unit"

    # space axes SHOULD have 'millimeter'
    assert by_name["y"]["unit"] == "millimeter"
    assert by_name["x"]["unit"] == "millimeter"


# ----------------------------------------------------------------------
# Lazy read path (load_ome_zarr(lazy=True))
# ----------------------------------------------------------------------


def test_lazy_returns_zarr_array_handle(tmp_path):
    """load_ome_zarr(lazy=True) returns a zarr.Array handle, not a materialized ndarray."""
    import zarr

    from linum_basic.io.zarr import load_ome_zarr, write_ome_zarr

    arr = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    out = tmp_path / "lazy.ome.zarr"
    write_ome_zarr(out, arr, axes=["z", "y", "x"], scale=[1.0, 0.5, 0.5])

    handle, axes, scale = load_ome_zarr(out, lazy=True)

    # The handle is the on-disk zarr array, NOT a materialized numpy array.
    assert isinstance(handle, zarr.Array)
    assert not isinstance(handle, np.ndarray)
    assert tuple(handle.shape) == arr.shape
    assert handle.dtype == arr.dtype
    # Chunked, on-disk storage is the structural proof of laziness.
    assert handle.chunks is not None and len(handle.chunks) == arr.ndim
    assert axes == ["z", "y", "x"]
    assert scale == pytest.approx([1.0, 0.5, 0.5])


def test_lazy_plane_read_materializes_single_slice(tmp_path):
    """Indexing the lazy handle returns a numpy array for just the requested slice."""
    from linum_basic.io.zarr import load_ome_zarr, write_ome_zarr

    arr = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    out = tmp_path / "planes.ome.zarr"
    write_ome_zarr(out, arr, axes=["z", "y", "x"], scale=[1.0, 1.0, 1.0])

    handle, _, _ = load_ome_zarr(out, lazy=True)

    # A single z-plane materializes as a numpy array of that plane's shape only.
    plane = handle[1]
    assert isinstance(plane, np.ndarray)
    assert plane.shape == (8, 8)
    np.testing.assert_array_equal(plane, arr[1])

    # A spatial sub-slice materializes only the requested region.
    tile = handle[2, 0:4, 0:4]
    assert isinstance(tile, np.ndarray)
    assert tile.shape == (4, 4)
    np.testing.assert_array_equal(tile, arr[2, 0:4, 0:4])


def test_lazy_values_match_eager(tmp_path):
    """Full read of the lazy handle matches the eager load exactly."""
    from linum_basic.io.zarr import load_ome_zarr, write_ome_zarr

    rng = np.random.default_rng(0)
    arr = rng.random((4, 6, 7)).astype(np.float32)
    out = tmp_path / "parity.ome.zarr"
    write_ome_zarr(out, arr, axes=["z", "y", "x"], scale=[0.5, 0.01, 0.01])

    eager, e_axes, e_scale = load_ome_zarr(out)
    lazy, l_axes, l_scale = load_ome_zarr(out, lazy=True)

    assert isinstance(eager, np.ndarray)
    assert not isinstance(lazy, np.ndarray)
    np.testing.assert_array_equal(np.asarray(lazy[:]), eager)
    assert l_axes == e_axes
    assert l_scale == pytest.approx(e_scale)


def test_default_load_is_eager(tmp_path):
    """Without lazy=, load_ome_zarr returns a materialized ndarray (backward compat)."""
    from linum_basic.io.zarr import load_ome_zarr, write_ome_zarr

    arr = np.ones((2, 3, 4), dtype=np.float32)
    out = tmp_path / "eager.ome.zarr"
    write_ome_zarr(out, arr, axes=["z", "y", "x"], scale=[1.0, 1.0, 1.0])

    loaded, _, _ = load_ome_zarr(out)
    assert isinstance(loaded, np.ndarray)


def test_lazy_invalid_store_raises(tmp_path):
    """lazy=True still raises FileNotFoundError for a non-existent store."""
    from linum_basic.io.zarr import load_ome_zarr

    with pytest.raises(FileNotFoundError):
        load_ome_zarr(tmp_path / "does_not_exist.ome.zarr", lazy=True)
