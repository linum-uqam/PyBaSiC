"""Demo-fit test: BaSiC recovery of a synthetic Gaussian vignette.

Uses sbh-simulator to generate a known flat-field and dark-field, applies the
degradation to tiles of the bundled Landsat image, runs BaSiC, and verifies
recovery via Pearson correlation against ground truth.

When ``LINUM_BASIC_VIGNETTE_ARTIFACT_DIR`` is set, saves a 9-panel diagnostic PNG:

  Row 0: GT flat-field | Estimated flat-field | FF abs error
  Row 1: GT dark-field | Estimated dark-field | DF abs error
  Row 2: Sample tile clean | Sample tile corrupted | Sample tile BaSiC-corrected

Skipped when the ``sbh-simulator`` package cannot be imported.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest

from linum_basic.core import BaSiC
from linum_basic.data import load_sample_image

try:
    from sbh_simulator.simulator import (
        generate_gaussian_darkfield,
        generate_gaussian_vignette,
    )

    _SBH_SIMULATOR_AVAILABLE = True
except ImportError:
    _SBH_SIMULATOR_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _SBH_SIMULATOR_AVAILABLE,
    reason="sbh-simulator not installed. Install with: uv pip install sbh-simulator",
)

_TILE = 128
_FF_CORRELATION_THRESHOLD = 0.85
_DF_CORRELATION_THRESHOLD = 0.30
_ARTIFACT_DIR_ENV = "LINUM_BASIC_VIGNETTE_ARTIFACT_DIR"


def _ground_truth() -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed (flatfield, darkfield) pair from sbh-simulator."""
    rng = random.Random(42)
    ff = generate_gaussian_vignette(width=_TILE, height=_TILE, sigma=0.7, rng=rng).astype(np.float32)
    df = generate_gaussian_darkfield(width=_TILE, height=_TILE, sigma=0.7, max_offset=0.05, rng=rng).astype(np.float32)
    return ff, df


def _tile_source_image() -> np.ndarray:
    """Tile the bundled image into a stack of (_TILE, _TILE) float32 patches."""
    img = load_sample_image()
    h, w = img.shape
    nh, nw = h // _TILE, w // _TILE
    cropped = img[: nh * _TILE, : nw * _TILE].astype(np.float32) / 255.0
    return cropped.reshape(nh, _TILE, nw, _TILE).transpose(0, 2, 1, 3).reshape(nh * nw, _TILE, _TILE)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def _save_figure(
    gt_ff: np.ndarray,
    gt_df: np.ndarray,
    corrupted: np.ndarray,
    model: BaSiC,
    ff_r: float,
    df_r: float,
    out_dir: Path,
) -> None:
    """Write a 9-panel diagnostic PNG into *out_dir*."""
    try:
        import matplotlib
    except ImportError:
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ff_est = model.flatfield_fullsize
    df_est = model.darkfield_fullsize
    corrected = np.stack([model.normalize(corrupted[i]) for i in range(len(corrupted))])
    clean = np.clip((corrupted - gt_df[np.newaxis]) / (gt_ff[np.newaxis] + 1e-9), 0.0, 1.0)
    edge_mask = gt_ff < gt_ff.mean()
    best = int(np.argmax(corrupted[:, edge_mask].mean(axis=-1)))

    ff_vmin = min(float(gt_ff.min()), float(ff_est.min()))
    ff_vmax = max(float(gt_ff.max()), float(ff_est.max()))
    df_vmin = 0.0
    df_vmax = max(float(gt_df.max()), float(df_est.max()))
    tile_vmin = float(clean[best].min())
    tile_vmax = float(clean[best].max())

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(
        f"BaSiC — Gaussian synthetic vignette  |  FF r = {ff_r:.3f}  |  DF r = {df_r:.3f}",
        fontsize=13,
        fontweight="bold",
    )
    panels = [
        (axes[0, 0], gt_ff, "viridis", True, "GT flat-field", ff_vmin, ff_vmax),
        (axes[0, 1], ff_est, "viridis", True, "Estimated flat-field", ff_vmin, ff_vmax),
        (axes[0, 2], np.abs(ff_est - gt_ff), "viridis", False, "FF abs error", None, None),
        (axes[1, 0], gt_df, "inferno", False, "GT dark-field", df_vmin, df_vmax),
        (axes[1, 1], df_est, "inferno", False, "Estimated dark-field", df_vmin, df_vmax),
        (axes[1, 2], np.abs(df_est - gt_df), "inferno", False, "DF abs error", None, None),
        (axes[2, 0], clean[best], "gray", False, "Sample tile: clean", tile_vmin, tile_vmax),
        (axes[2, 1], corrupted[best], "gray", False, "Sample tile: corrupted", tile_vmin, tile_vmax),
        (axes[2, 2], corrected[best], "gray", False, "Sample tile: BaSiC-corrected", tile_vmin, tile_vmax),
    ]
    for ax, data, cmap, contours, title, vmin, vmax in panels:
        kw: dict = {} if vmin is None else {"vmin": vmin, "vmax": vmax}
        im = ax.imshow(data, cmap=cmap, interpolation="nearest", **kw)
        if contours:
            ax.contour(data, levels=8, colors="w", linewidths=0.5, alpha=0.7)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo_fitting.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as _plt

    _plt.close(fig)
    print(f"Saved figure: {out_path}")


def test_demo_fitting() -> None:
    """BaSiC recovers a synthetic Gaussian vignette with Pearson r >= threshold."""
    gt_ff, gt_df = _ground_truth()
    clean = _tile_source_image()
    rng = np.random.default_rng(0)
    corrupted = (clean * gt_ff[np.newaxis] + gt_df[np.newaxis] + rng.normal(0, 0.002, clean.shape)).astype(np.float32)

    model = BaSiC(corrupted, estimate_darkfield=True)
    model.prepare()
    model.run()

    ff_r = _pearson(model.flatfield_fullsize, gt_ff)
    df_r = _pearson(model.darkfield_fullsize, gt_df)

    assert ff_r >= _FF_CORRELATION_THRESHOLD, f"FF Pearson r = {ff_r:.3f} < {_FF_CORRELATION_THRESHOLD}"
    assert df_r >= _DF_CORRELATION_THRESHOLD, f"DF Pearson r = {df_r:.3f} < {_DF_CORRELATION_THRESHOLD}"

    artifact_dir = os.environ.get(_ARTIFACT_DIR_ENV)
    if artifact_dir:
        _save_figure(gt_ff, gt_df, corrupted, model, ff_r, df_r, Path(artifact_dir))
