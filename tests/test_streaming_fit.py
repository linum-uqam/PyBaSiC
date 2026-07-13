"""Tests for the sequential streaming mode of :func:`fit_mosaic`.

Covers three must-haves from the S02 slice plan:

* **Numerics parity** — ``streaming=True`` produces identical flat/dark-fields
  to the non-streaming sequential path on the same input.
* **Peak-memory reduction** — a ``tracemalloc`` probe demonstrates streaming
  peak memory is measurably lower than the eager path.
* **Validation** — ``streaming=True`` rejects ``strategy="multi"`` and
  ``"batched"``; defaults to ``False`` (backward compatible).
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_synthetic_volume(
    n_z: int = 12,
    n_rows: int = 4,
    n_cols: int = 4,
    th: int = 64,
    tw: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Build a synthetic mosaic volume with a smooth vignette BaSiC can fit.

    The volume has a smooth Gaussian illumination profile plus low-amplitude
    noise, so the BaSiC solver has a spatially non-uniform flat-field to
    estimate.  No ``sbh-simulator`` dependency — pure NumPy.
    """
    rng = np.random.default_rng(seed)
    H = n_rows * th
    W = n_cols * tw
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W / 2
    sigma = min(H, W) / 3
    vignette = 1.0 + 0.3 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    volume = np.zeros((n_z, H, W), dtype=np.float32)
    for z in range(n_z):
        noise = rng.random((H, W), dtype=np.float32) * 0.1
        volume[z] = (0.5 + noise) * vignette * (1.0 + 0.02 * z)
    return volume


def _write_store(store_path, volume: np.ndarray, th: int, tw: int) -> None:
    """Write *volume* as an OME-Zarr store chunked by tile (z=full, y/x=tile)."""
    import zarr

    n_z = volume.shape[0]
    grp = zarr.open_group(str(store_path), mode="w", zarr_format=3)
    grp.create_array("s0", data=volume, chunks=(n_z, th, tw))
    grp.attrs["ome"] = {
        "multiscales": [
            {
                "version": "0.5",
                "axes": [{"name": ax, "type": "space"} for ax in ["z", "y", "x"]],
                "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]}],
            }
        ]
    }


# Common BaSiC kwargs: small working_size for speed, numpy backend (no CUDA).
_FIT_KW = {"backend": "numpy", "working_size": 32, "estimate_darkfield": False, "max_reweighting_iterations": 3}


@pytest.fixture
def mosaic_grid(tmp_path):
    """Eager (ndarray-backed) MosaicGrid over a synthetic mosaic."""
    from linum_basic.mosaic import MosaicGrid

    n_z, n_rows, n_cols, th, tw = 8, 4, 4, 64, 64
    volume = _make_synthetic_volume(n_z, n_rows, n_cols, th, tw)
    mosaic = MosaicGrid(volume, tile_shape=(th, tw), overlap_fraction=0.2)
    return mosaic


@pytest.fixture
def lazy_mosaic_grid(tmp_path):
    """Lazy (zarr.Array-backed) MosaicGrid loaded from an OME-Zarr store."""
    from linum_basic.mosaic import MosaicGrid

    n_z, n_rows, n_cols, th, tw = 6, 3, 4, 48, 48
    volume = _make_synthetic_volume(n_z, n_rows, n_cols, th, tw)
    store = tmp_path / "lazy_streaming.ome.zarr"
    _write_store(store, volume, th, tw)
    mosaic = MosaicGrid.from_ome_zarr(str(store), lazy=True)
    return mosaic


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestStreamingValidation:
    def test_streaming_rejects_multi(self, mosaic_grid):
        """streaming=True + strategy='multi' raises ValueError."""
        from linum_basic.fit import fit_mosaic

        with pytest.raises(ValueError, match="streaming=True is incompatible with strategy='multi'"):
            fit_mosaic(mosaic_grid, streaming=True, strategy="multi", basic_kwargs=_FIT_KW)

    def test_streaming_rejects_batched(self, mosaic_grid):
        """streaming=True + strategy='batched' raises ValueError."""
        from linum_basic.fit import fit_mosaic

        with pytest.raises(ValueError, match="streaming=True is incompatible with strategy='batched'"):
            fit_mosaic(mosaic_grid, streaming=True, strategy="batched", basic_kwargs=_FIT_KW)

    def test_streaming_allows_auto(self, mosaic_grid):
        """streaming=True + strategy='auto' is accepted (no error at validation)."""
        from linum_basic.fit import fit_mosaic

        fit = fit_mosaic(mosaic_grid, streaming=True, strategy="auto", basic_kwargs=_FIT_KW)
        assert fit.flatfields.shape == (mosaic_grid.n_z, 64, 64)

    def test_streaming_allows_sequential(self, mosaic_grid):
        """streaming=True + strategy='sequential' is accepted."""
        from linum_basic.fit import fit_mosaic

        fit = fit_mosaic(mosaic_grid, streaming=True, strategy="sequential", basic_kwargs=_FIT_KW)
        assert fit.flatfields.shape == (mosaic_grid.n_z, 64, 64)

    def test_streaming_default_is_false(self, mosaic_grid):
        """Without streaming=, fit_mosaic runs the existing eager path (backward compat)."""
        from linum_basic.fit import fit_mosaic

        # Must not raise — streaming is opt-in.
        fit = fit_mosaic(mosaic_grid, strategy="sequential", n_workers=1, basic_kwargs=_FIT_KW)
        assert fit.flatfields.shape == (mosaic_grid.n_z, 64, 64)

    def test_streaming_rejects_unknown_strategy(self, mosaic_grid):
        """An unknown strategy raises before the streaming check (ordering)."""
        from linum_basic.fit import fit_mosaic

        with pytest.raises(ValueError, match="Unknown strategy"):
            fit_mosaic(mosaic_grid, streaming=True, strategy="bogus", basic_kwargs=_FIT_KW)


# ---------------------------------------------------------------------------
# Numerics parity
# ---------------------------------------------------------------------------


class TestStreamingNumericsParity:
    def test_streaming_matches_non_streaming_sequential(self, mosaic_grid):
        """streaming=True and streaming=False produce identical flat/dark-fields.

        Both paths run ``_fit_one_z`` on the same tiles in the same order with
        the same params; only the extraction timing differs (one-at-a-time vs
        all-up-front).  Results must be bit-identical.
        """
        from linum_basic.fit import fit_mosaic

        fit_stream = fit_mosaic(mosaic_grid, streaming=True, strategy="sequential", basic_kwargs=dict(_FIT_KW))
        fit_eager = fit_mosaic(
            mosaic_grid,
            streaming=False,
            strategy="sequential",
            n_workers=1,
            basic_kwargs=dict(_FIT_KW),
        )

        np.testing.assert_array_equal(fit_stream.flatfields, fit_eager.flatfields)
        np.testing.assert_array_equal(fit_stream.darkfields, fit_eager.darkfields)
        assert fit_stream.z_indices == fit_eager.z_indices

    def test_streaming_respects_z_indices(self, mosaic_grid):
        """streaming=True honours a z_indices subset."""
        from linum_basic.fit import fit_mosaic

        fit = fit_mosaic(
            mosaic_grid,
            z_indices=[0, 2, 5],
            streaming=True,
            strategy="sequential",
            basic_kwargs=_FIT_KW,
        )
        assert fit.z_indices == [0, 2, 5]
        assert fit.flatfields.shape == (3, 64, 64)

    def test_streaming_global_field_mode(self, mosaic_grid):
        """streaming=True + field_mode='global' averages per-z fields."""
        from linum_basic.fit import fit_mosaic

        fit = fit_mosaic(
            mosaic_grid,
            field_mode="global",
            streaming=True,
            strategy="sequential",
            basic_kwargs=_FIT_KW,
        )
        assert fit.field_mode == "global"
        assert fit.flatfields.shape == (64, 64)


# ---------------------------------------------------------------------------
# Peak-memory reduction
# ---------------------------------------------------------------------------


class TestStreamingMemory:
    def test_streaming_peak_memory_lower(self, tmp_path):
        """Streaming peak memory is measurably lower than the eager path.

        Uses a volume with many z-levels so the eager
        ``tile_stacks = [mosaic.iter_tiles(z) for z in z_idx]`` list holds
        many simultaneous copies, while streaming holds only one at a time.
        """
        import tracemalloc

        from linum_basic.fit import fit_mosaic
        from linum_basic.mosaic import MosaicGrid

        n_z, n_rows, n_cols, th, tw = 12, 4, 4, 64, 64
        volume = _make_synthetic_volume(n_z, n_rows, n_cols, th, tw)
        mosaic = MosaicGrid(volume, tile_shape=(th, tw), overlap_fraction=0.2)
        kw = {"backend": "numpy", "working_size": 32, "estimate_darkfield": False, "max_reweighting_iterations": 3}

        # --- Non-streaming (eager): all tile stacks live simultaneously. ---
        gc.collect()
        tracemalloc.start()
        fit_eager = fit_mosaic(mosaic, streaming=False, strategy="sequential", n_workers=1, basic_kwargs=kw)
        _, peak_eager = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # --- Streaming: one tile stack at a time. ---
        gc.collect()
        tracemalloc.start()
        fit_stream = fit_mosaic(mosaic, streaming=True, strategy="sequential", basic_kwargs=kw)
        _, peak_stream = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Sanity: both produced valid output of the same shape.
        assert fit_eager.flatfields.shape == fit_stream.flatfields.shape

        # Streaming must use measurably less peak memory.
        assert peak_stream < peak_eager, f"Streaming peak ({peak_stream:,}) should be < eager peak ({peak_eager:,})"

        # The reduction must exceed at least 2 tile-stacks worth of data,
        # proving the eager list-comp is the memory driver (not just noise).
        tile_stack_bytes = n_rows * n_cols * th * tw * 4  # one z-level of tiles
        reduction = peak_eager - peak_stream
        assert reduction > 2 * tile_stack_bytes, (
            f"Memory reduction ({reduction:,}) should exceed 2 tile-stacks ({2 * tile_stack_bytes:,})"
        )


# ---------------------------------------------------------------------------
# Lazy MosaicGrid integration
# ---------------------------------------------------------------------------


class TestStreamingWithLazyGrid:
    def test_streaming_lazy_grid_produces_valid_fit(self, lazy_mosaic_grid):
        """streaming=True works with a lazy (zarr.Array-backed) MosaicGrid."""
        import zarr

        from linum_basic.fit import fit_mosaic

        # Confirm the grid is genuinely lazy.
        assert isinstance(lazy_mosaic_grid.array, zarr.Array)
        assert not isinstance(lazy_mosaic_grid.array, np.ndarray)

        kw = {"backend": "numpy", "working_size": 32, "estimate_darkfield": False, "max_reweighting_iterations": 3}
        fit = fit_mosaic(lazy_mosaic_grid, streaming=True, strategy="sequential", basic_kwargs=kw)

        assert fit.flatfields.shape == (lazy_mosaic_grid.n_z, 48, 48)
        assert fit.darkfields.shape == (lazy_mosaic_grid.n_z, 48, 48)
        # Flat-fields must be finite (no NaNs from a failed solve).
        assert np.all(np.isfinite(fit.flatfields))

    def test_streaming_lazy_matches_eager_lazy(self, lazy_mosaic_grid, tmp_path):
        """streaming=True and streaming=False produce identical results on a lazy grid."""
        from linum_basic.fit import fit_mosaic

        kw = {"backend": "numpy", "working_size": 32, "estimate_darkfield": False, "max_reweighting_iterations": 3}

        fit_stream = fit_mosaic(lazy_mosaic_grid, streaming=True, strategy="sequential", basic_kwargs=dict(kw))
        fit_eager = fit_mosaic(lazy_mosaic_grid, streaming=False, strategy="sequential", n_workers=1, basic_kwargs=dict(kw))

        np.testing.assert_array_equal(fit_stream.flatfields, fit_eager.flatfields)
