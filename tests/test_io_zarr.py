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
