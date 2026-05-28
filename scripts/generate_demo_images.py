#!/usr/bin/env python3
"""Generate demo figures for the Linum BaSiC documentation.

Produces a 3-panel comparison image at ``docs/_static/demo/demo_comparison.png``
showing:
  - Left:   a sample tile *after* synthetic vignette corruption
  - Centre: the flat-field estimated by BaSiC
  - Right:  the same tile after BaSiC correction

The synthetic vignette is a 2-D Gaussian centred on the tile — no external
simulator required.  The bundled source image (``tests/data/source_image.jpg``)
is tiled into 128 x 128 patches to build the image stack.

Run from the repository root::

    uv run python scripts/generate_demo_images.py

or::

    python scripts/generate_demo_images.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from linum_basic.core import BaSiC
from linum_basic.data import load_sample_image

_REPO_ROOT = Path(__file__).parent.parent
_OUTPUT_DIR = _REPO_ROOT / "docs" / "_static" / "demo"
_TILE = 128


def _make_vignette(size: int, sigma_frac: float = 0.45) -> np.ndarray:
    """Return a normalised Gaussian vignette of shape ``(size, size)``."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cy, cx = (size - 1) / 2, (size - 1) / 2
    sigma = size * sigma_frac
    v = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    return (v / v.max()).astype(np.float32)


def _make_darkfield(size: int, sigma_frac: float = 0.35, max_offset: float = 0.05) -> np.ndarray:
    """Return a smooth Gaussian dark-field offset of shape ``(size, size)``.

    The dark-field peaks at the image centre (simulating auto-fluorescence
    from the objective lens) and falls off to near-zero at the edges.
    """
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cy, cx = (size - 1) / 2, (size - 1) / 2
    sigma = size * sigma_frac
    d = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    return (d * max_offset).astype(np.float32)


def _tile_image(img: np.ndarray, tile: int) -> tuple[np.ndarray, int, int]:
    """Tile a 2-D uint8 image into a float32 stack of (N, tile, tile) patches.

    Returns
    -------
    tuple of (stack, nh, nw) where stack has shape (nh*nw, tile, tile).
    """
    h, w = img.shape
    nh, nw = h // tile, w // tile
    cropped = img[: nh * tile, : nw * tile].astype(np.float32) / 255.0
    return cropped.reshape(nh, tile, nw, tile).transpose(0, 2, 1, 3).reshape(nh * nw, tile, tile), nh, nw


def _stitch(stack: np.ndarray, nh: int, nw: int, tile: int) -> np.ndarray:
    """Reassemble a (nh*nw, tile, tile) stack back into a (nh*tile, nw*tile) image."""
    return stack.reshape(nh, nw, tile, tile).transpose(0, 2, 1, 3).reshape(nh * tile, nw * tile)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_OUTPUT_DIR,
        metavar="DIR",
        help="Output directory for demo figures (default: %(default)s)",
    )
    args = parser.parse_args()
    out: Path = args.output
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build corrupted stack
    # ------------------------------------------------------------------
    tiles, nh, nw = _tile_image(load_sample_image(), _TILE)
    vignette = _make_vignette(_TILE)
    corrupted = tiles * vignette[np.newaxis]  # broadcast over stack axis

    # ------------------------------------------------------------------
    # 2. Run BaSiC
    # ------------------------------------------------------------------
    print(f"Running BaSiC on {len(corrupted)} tiles ({_TILE}x{_TILE})...")
    model = BaSiC(corrupted, estimate_darkfield=False, verbose=True)
    model.prepare()
    model.run()
    flatfield = model.get_flatfield()  # shape (128, 128)

    # ------------------------------------------------------------------
    # 3. Pick the most visually rich tile (highest std dev)
    # ------------------------------------------------------------------
    best = int(np.argmax(corrupted.std(axis=(1, 2))))
    sample_corrupted = corrupted[best]
    sample_corrected = model.normalize(sample_corrupted)

    # ------------------------------------------------------------------
    # 4. Save individual panels (for use in RST figure directives)
    # ------------------------------------------------------------------
    def _save_gray(arr: np.ndarray, name: str) -> None:
        clipped = np.clip(arr, 0, None)
        normalised = (clipped / clipped.max() * 255).astype(np.uint8) if clipped.max() > 0 else clipped.astype(np.uint8)
        cv2.imwrite(str(out / name), normalised)

    _save_gray(sample_corrupted, "tile_corrupted.png")
    _save_gray(flatfield, "flatfield.png")
    _save_gray(sample_corrected, "tile_corrected.png")

    # ------------------------------------------------------------------
    # 4b. Build and save full-resolution images
    # ------------------------------------------------------------------
    corrected_stack = np.array([model.normalize(t) for t in corrupted])
    full_corrupted = _stitch(corrupted, nh, nw, _TILE)
    full_corrected = _stitch(corrected_stack, nh, nw, _TILE)
    _save_gray(full_corrupted, "full_corrupted.png")
    _save_gray(full_corrected, "full_corrected.png")

    # Side-by-side full image comparison figure
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.patch.set_facecolor("#0d1117")
    for ax, data, title in zip(
        axes2,
        [full_corrupted, full_corrected],
        ["Full image — corrupted (synthetic vignette)", "Full image — corrected by Linum BaSiC"],
        strict=True,
    ):
        ax.imshow(data, cmap="gray", interpolation="lanczos")
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.axis("off")
    fig2.tight_layout()
    full_path = out / "full_comparison.png"
    fig2.savefig(full_path, dpi=100, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close(fig2)

    # ------------------------------------------------------------------
    # 5. Save the 3-panel comparison figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    fig.patch.set_facecolor("#0d1117")  # dark background

    panel_data = [
        (sample_corrupted, "gray", "Corrupted tile\n(with vignette)"),
        (flatfield, "inferno", "Estimated flat-field"),
        (sample_corrected, "gray", "Corrected tile"),
    ]
    for ax, (data, cmap, title) in zip(axes, panel_data, strict=True):
        im = ax.imshow(data, cmap=cmap, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.axis("off")

    fig.suptitle(
        "BaSiC illumination correction demo",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    comparison_path = out / "demo_comparison.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved: {comparison_path}")
    print(f"Saved: {out / 'tile_corrupted.png'}")
    print(f"Saved: {out / 'flatfield.png'}")
    print(f"Saved: {out / 'tile_corrected.png'}")
    print(f"Saved: {full_path}")
    print(f"Saved: {out / 'full_corrupted.png'}")
    print(f"Saved: {out / 'full_corrected.png'}")

    # ------------------------------------------------------------------
    # 6. Dark-field demo — joint flat-field + dark-field estimation
    # ------------------------------------------------------------------
    print("\nGenerating darkfield demo images...")
    rng = np.random.default_rng(42)
    gt_darkfield = _make_darkfield(_TILE)
    gt_vignette = _make_vignette(_TILE)
    gt_vignette /= gt_vignette.mean()  # normalise so mean == 1 (matches BaSiC convention)

    # Apply degradation: corrupted = clean * flatfield + darkfield + small noise
    noise_std = 0.005
    corrupted_df = tiles * gt_vignette[np.newaxis] + gt_darkfield[np.newaxis]
    corrupted_df += rng.normal(0, noise_std, corrupted_df.shape).astype(np.float32)
    corrupted_df = np.clip(corrupted_df, 0, 1)

    print(f"Running BaSiC (estimate_darkfield=True) on {len(corrupted_df)} tiles...")
    model_df = BaSiC(corrupted_df, estimate_darkfield=True, verbose=True)
    model_df.prepare()
    model_df.run()
    est_flatfield = model_df.get_flatfield()
    est_darkfield = model_df.get_darkfield()

    # Pick the most visually rich tile
    best_df = int(np.argmax(corrupted_df.std(axis=(1, 2))))
    sample_corrupt_df = corrupted_df[best_df]
    sample_corrected_df = model_df.normalize(sample_corrupt_df)

    # Save individual PNG assets
    _save_gray(gt_darkfield, "darkfield_gt.png")
    _save_gray(est_darkfield, "darkfield_estimated.png")

    # 6-panel comparison: 2 rows x 3 cols
    # Row 0: corrupted tile | GT flat-field | GT dark-field
    # Row 1: corrected tile | estimated flat-field | estimated dark-field
    fig3, axes3 = plt.subplots(2, 3, figsize=(12, 7))
    fig3.patch.set_facecolor("#0d1117")

    # Shared colour limits for flat-field row and dark-field row
    ff_vmin = min(gt_vignette.min(), est_flatfield.min())
    ff_vmax = max(gt_vignette.max(), est_flatfield.max())
    df_vmax = max(gt_darkfield.max(), est_darkfield.max())

    panels_df = [
        (axes3[0, 0], sample_corrupt_df, "gray", "Corrupted tile\n(vignette + dark-field)", None, None),
        (axes3[0, 1], gt_vignette, "viridis", f"GT flat-field\n(mean = {gt_vignette.mean():.2f})", ff_vmin, ff_vmax),
        (axes3[0, 2], gt_darkfield, "inferno", f"GT dark-field\n(max = {gt_darkfield.max():.3f})", 0, df_vmax),
        (axes3[1, 0], sample_corrected_df, "gray", "Corrected tile", None, None),
        (
            axes3[1, 1],
            est_flatfield,
            "viridis",
            f"Estimated flat-field\n(mean = {est_flatfield.mean():.2f})",
            ff_vmin,
            ff_vmax,
        ),
        (
            axes3[1, 2],
            est_darkfield,
            "inferno",
            f"Estimated dark-field\n(max = {est_darkfield.max():.3f})",
            0,
            df_vmax,
        ),
    ]
    for ax, data, cmap, title, vmin, vmax in panels_df:
        kw: dict = {"interpolation": "nearest"}
        if vmin is not None:
            kw["vmin"] = vmin
            kw["vmax"] = vmax
        im3 = ax.imshow(data, cmap=cmap, **kw)
        fig3.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.axis("off")

    fig3.suptitle(
        "BaSiC dark-field estimation demo — ground truth vs estimate",
        color="white",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig3.tight_layout()
    darkfield_path = out / "darkfield_demo_comparison.png"
    fig3.savefig(darkfield_path, dpi=150, bbox_inches="tight", facecolor=fig3.get_facecolor())
    plt.close(fig3)

    print(f"Saved: {darkfield_path}")
    print(f"Saved: {out / 'darkfield_gt.png'}")
    print(f"Saved: {out / 'darkfield_estimated.png'}")
    print("Done.")


if __name__ == "__main__":
    main()
