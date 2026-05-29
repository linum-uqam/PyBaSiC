"""Smoke tests for scripts/visualize_flatfield_curvature.py.

Uses the same bundled source image and Gaussian vignette pattern as the
CI vignette validation suite (test_vignette_validation.py) to build a
small synthetic mosaic — no OME-Zarr file required.

The tests verify that both figure functions complete without error and
produce an output PNG.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pytest

# Use non-interactive backend before any pyplot import
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Import the script module by path (lives in scripts/, not the package)
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "visualize_flatfield_curvature.py"


@pytest.fixture(scope="module")
def script():
    """Load scripts/visualize_flatfield_curvature.py as a module."""
    spec = importlib.util.spec_from_file_location("visualize_flatfield_curvature", _SCRIPT_PATH)
    assert spec is not None, f"Could not locate script: {_SCRIPT_PATH}"
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Synthetic mosaic helpers (mirror the CI vignette approach)
# ---------------------------------------------------------------------------

_TILE = 32  # small tiles for speed
_N_ROWS = 4
_N_COLS = 4
_N_Z = 3
_OVERLAP = 0.2


def _gaussian_vignette(tile: int, sigma: float = 0.7) -> np.ndarray:
    """Radial Gaussian flat-field, max == 1 (same convention as sbh-simulator)."""
    y, x = np.mgrid[-1 : 1 : complex(tile), -1 : 1 : complex(tile)]
    return np.exp(-(x**2 + y**2) / (2 * sigma**2)).astype(np.float32)


def _make_synthetic_mosaic():
    """Return a :class:`MosaicGrid` with a Gaussian vignette applied per tile.

    The source content comes from the bundled Landsat image (same as
    ``test_vignette_validation.py``).  A small amount of Gaussian noise
    is added so that BaSiC has a realistic signal distribution to work with.
    """
    from linum_basic.data import load_sample_image
    from linum_basic.mosaic import MosaicGrid

    src = load_sample_image().astype(np.float32) / 255.0
    sh, sw = src.shape
    # extract non-overlapping patches of size _TILE x _TILE
    nh, nw = sh // _TILE, sw // _TILE
    patches = src[: nh * _TILE, : nw * _TILE].reshape(nh, _TILE, nw, _TILE).transpose(0, 2, 1, 3).reshape(-1, _TILE, _TILE)

    vignette = _gaussian_vignette(_TILE)
    rng = np.random.default_rng(42)

    H = _N_ROWS * _TILE
    W = _N_COLS * _TILE
    mosaic_arr = np.zeros((_N_Z, H, W), dtype=np.float32)

    for z in range(_N_Z):
        for r in range(_N_ROWS):
            for c in range(_N_COLS):
                idx = (r * _N_COLS + c) % len(patches)
                tile = patches[idx] * vignette + rng.normal(0.0, 0.01, (_TILE, _TILE)).astype(np.float32)
                mosaic_arr[z, r * _TILE : (r + 1) * _TILE, c * _TILE : (c + 1) * _TILE] = np.clip(tile, 0.0, 1.0)

    # Scale to a realistic uint16-equivalent float range
    mosaic_arr = mosaic_arr * 60000.0

    return MosaicGrid(array=mosaic_arr, tile_shape=(_TILE, _TILE), overlap_fraction=_OVERLAP)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mosaic_and_fit():
    """Shared mosaic + fit for all figure tests (computed once)."""
    from linum_basic.fit import fit_mosaic

    mosaic = _make_synthetic_mosaic()
    z_indices = list(range(_N_Z))
    fit = fit_mosaic(mosaic, z_indices=z_indices, basic_kwargs={}, n_extra_rows=0, verbose=False)
    return mosaic, fit


class TestBuildFigure:
    """Smoke tests for _build_figure."""

    def test_produces_png(self, tmp_path, mosaic_and_fit, script):
        """_build_figure writes a PNG to the requested path."""
        mosaic, fit = mosaic_and_fit
        out = tmp_path / "flatfield_curvature.png"
        script._build_figure(
            mosaic,
            fit,
            z_inspect=fit.z_indices[0],
            smooth_sigma=0.0,
            out_path=out,
            n_extra_rows=0,
        )
        assert out.exists(), "Output PNG was not created"
        assert out.stat().st_size > 0, "Output PNG is empty"

    def test_accepts_smooth_sigma(self, tmp_path, mosaic_and_fit, script):
        """_build_figure accepts a non-zero smooth_sigma without error."""
        mosaic, fit = mosaic_and_fit
        out = tmp_path / "flatfield_curvature_smooth.png"
        script._build_figure(
            mosaic,
            fit,
            z_inspect=fit.z_indices[0],
            smooth_sigma=1.0,
            out_path=out,
            n_extra_rows=0,
        )
        assert out.exists()


class TestBuildOrientationFigure:
    """Smoke tests for _build_orientation_figure."""

    def test_produces_png(self, tmp_path, mosaic_and_fit, script):
        """_build_orientation_figure writes a PNG to the requested path."""
        _, fit = mosaic_and_fit
        out = tmp_path / "flatfield_orientation.png"
        script._build_orientation_figure(
            fit,
            z_inspect=fit.z_indices[0],
            smooth_sigma=0.0,
            out_path=out,
        )
        assert out.exists(), "Output PNG was not created"
        assert out.stat().st_size > 0, "Output PNG is empty"
