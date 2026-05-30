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
import numpy as np

from linum_basic import viz
from linum_basic.core import BaSiC
from linum_basic.data import load_sample_image, zernike_flatfield
from linum_basic.fit import apply_fit
from linum_basic.metrics import seam_l1
from linum_basic.mosaic import MosaicGrid

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


def _build_synthetic_mosaic(tile: int = 96, ncols: int = 12, nrows: int = 8) -> MosaicGrid:
    """Build a single-z mosaic of genuinely overlapping, vignette-corrupted tiles.

    Tiles are extracted from the bundled source image with a stride of
    ``tile * (1 - overlap)`` so neighbouring tiles share content in their
    overlap region, then each is multiplied by a low-order Zernike shading
    field whose asymmetry produces pronounced, correctable seams.
    """
    overlap = 0.2
    ov = round(overlap * tile)
    stride = tile - ov
    src = load_sample_image().astype(np.float32) / 255.0

    array = np.empty((nrows * tile, ncols * tile), dtype=np.float32)
    vignette = zernike_flatfield(tile, n_max=4, contrast=0.45, seed=0)
    for r in range(nrows):
        for c in range(ncols):
            patch = src[r * stride : r * stride + tile, c * stride : c * stride + tile]
            array[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = patch * vignette
    return MosaicGrid(array[np.newaxis], (tile, tile), overlap_fraction=overlap)


def _save_flatfield_3d(flatfield: np.ndarray, out: Path) -> None:
    """Render the estimated flat-field as a 3-D surface (illumination landscape)."""
    fig, _ = viz.field_surface_3d(flatfield, title="Estimated flat-field as an illumination surface")
    path = viz.save_figure(fig, out / "flatfield_3d.png", dpi=150)
    print(f"Saved: {path}")


def _save_tuning_demo(out: Path) -> None:
    """Run a short Optuna tune on a synthetic mosaic and plot the training story.

    Produces a 1x3 figure: the optimisation history (best-so-far seam-L1),
    the tuned flat-field, and a raw-vs-tuned seam-consistency bar chart.
    """
    try:
        import optuna  # noqa: F401
    except ImportError:
        print("optuna not installed — skipping tuning demo.")
        return
    from linum_basic.tuning import tune

    print("\nGenerating tuning/training demo (synthetic mosaic)...")
    mosaic = _build_synthetic_mosaic()
    seam_pairs = mosaic.seam_pairs()
    raw_tiles = mosaic.iter_tiles(0)
    seam_raw = seam_l1(raw_tiles, seam_pairs)

    result = tune(
        mosaic,
        n_trials=25,
        z_subsample=1,
        max_tiles=None,
        seed=0,
        run_full_fit=True,
        verbose=False,
    )
    assert result.best_fit is not None
    corrected = apply_fit(mosaic, result.best_fit)[0]
    th, tw = mosaic.tile_shape
    corr_tiles = corrected.reshape(mosaic.n_rows, th, mosaic.n_cols, tw).transpose(0, 2, 1, 3)
    corr_tiles = corr_tiles.reshape(mosaic.n_rows * mosaic.n_cols, th, tw)
    seam_tuned = seam_l1(corr_tiles, seam_pairs)
    improvement = 100.0 * (seam_raw - seam_tuned) / (seam_raw + 1e-12)

    df = result.trials_df
    if df is not None and "value" in df.columns:
        completed = df[df["state"] == "COMPLETE"].reset_index(drop=True) if "state" in df.columns else df
        values = completed["value"].to_numpy()
    else:
        values = np.array([])

    fig = viz.figure_tuning_history(
        trial_values=values,
        flatfield=result.best_fit.flatfields[0],
        seam_raw=seam_raw,
        seam_tuned=seam_tuned,
    )
    path = viz.save_figure(fig, out / "tuning_demo.png", dpi=150)
    print(f"Saved: {path}  (seam L1 {seam_raw:.3f} -> {seam_tuned:.3f}, {improvement:+.0f}%)")


def _save_seam_metric_demo(out: Path, tile: int = 128, overlap: float = 0.2) -> None:
    """Render the seam-consistency demo using the full MosaicGrid + fit_mosaic pipeline.

    Builds a synthetic mosaic that mimics real acquisition data: tiles share
    a physical overlap region and are all dimmed by the same illumination
    field.  BaSiC is fitted on the full tile stack (as in the real pipeline),
    then two horizontally adjacent tiles are extracted from the raw and
    corrected mosaics and shown side-by-side so the seam artefact — and its
    removal — is directly visible as a 2-D image.
    """
    from linum_basic.fit import apply_fit, fit_mosaic

    print("\nGenerating seam-metric demo (MosaicGrid + fit_mosaic pipeline)...")
    mosaic = _build_synthetic_mosaic(tile=tile, ncols=8, nrows=6)

    fit = fit_mosaic(mosaic, z_indices=[0], field_mode="global", n_workers=1, verbose=False)

    # Corrected mosaic: shape (Z, H, W); z=0 slice is (H, W).
    corrected_vol = apply_fit(mosaic, fit)
    raw_z0 = mosaic.array[0]  # (n_rows*tile, n_cols*tile)
    cor_z0 = corrected_vol[0]

    # Pick a central pair of horizontally adjacent tiles.
    row = mosaic.n_rows // 2
    col = mosaic.n_cols // 2 - 1  # left tile of the pair
    th, tw = mosaic.tile_shape
    raw_a = raw_z0[row * th : (row + 1) * th, col * tw : (col + 1) * tw]
    raw_b = raw_z0[row * th : (row + 1) * th, (col + 1) * tw : (col + 2) * tw]
    cor_a = cor_z0[row * th : (row + 1) * th, col * tw : (col + 1) * tw]
    cor_b = cor_z0[row * th : (row + 1) * th, (col + 1) * tw : (col + 2) * tw]

    fig = viz.figure_seam_metric(
        raw_tile_a=raw_a,
        raw_tile_b=raw_b,
        cor_tile_a=cor_a,
        cor_tile_b=cor_b,
        overlap_fraction=overlap,
        orientation="horizontal",
        title="Seam-consistency metric — two adjacent tiles (MosaicGrid + BaSiC)",
    )
    path = viz.save_figure(fig, out / "seam_metric_demo.png", dpi=150)
    print(f"Saved: {path}")


def _seam_discrepancy(tiles: np.ndarray, n_rows: int, n_cols: int, overlap_px: int) -> float:
    """Mean absolute pixel difference in overlap strips between adjacent tiles."""
    diffs = []
    for row in range(n_rows):
        for col in range(n_cols):
            t = tiles[row * n_cols + col]
            if col + 1 < n_cols:
                r = tiles[row * n_cols + col + 1]
                diffs.append(np.abs(t[:, -overlap_px:] - r[:, :overlap_px]).mean())
            if row + 1 < n_rows:
                b = tiles[(row + 1) * n_cols + col]
                diffs.append(np.abs(t[-overlap_px:, :] - b[:overlap_px, :]).mean())
    return float(np.mean(diffs))


def _save_focal_volume_demo(
    out: Path,
    tile: int = 64,
    n_z: int = 13,
    grid: int = 5,
    overlap_px: int = 8,
) -> None:
    """Render a 3-D focal-volume demo showing depth-varying illumination correction.

    Generates a synthetic z-stack with ``n_z`` depth levels, each containing
    a ``grid x grid`` mosaic of tiles.  The illumination field at each z
    follows a lens-like focal profile: nearly flat and bright at the focal
    plane, increasingly vignetted and dim away from focus (Gaussian beam
    envelope).  BaSiC is fitted independently at each z-level and the
    estimated flat-fields are collected.

    The output figure shows the volume *from the side* (lateral x vs depth z)
    so the focal curve is visible before correction and absent after.
    """
    print("\nGenerating focal-volume demo (3-D z-stack with focal curve)...")
    src = load_sample_image().astype(np.float32) / 255.0
    h_src, w_src = src.shape

    focal_z = n_z // 2
    rng = np.random.default_rng(42)

    raw_sides: list[np.ndarray] = []
    est_sides: list[np.ndarray] = []
    corrected_sides: list[np.ndarray] = []
    seam_before: list[float] = []
    seam_after: list[float] = []

    for z in range(n_z):
        defocus = abs(z - focal_z) / (focal_z + 1e-6)
        contrast = 0.05 + 0.40 * defocus**2
        mean_scale = float(0.45 + 0.55 * np.exp(-3.0 * defocus**2))

        field = zernike_flatfield(tile, n_max=4, contrast=contrast, seed=42)
        field = (field / field.mean()).astype(np.float32)

        n_tiles = grid * grid
        tiles_raw = np.empty((n_tiles, tile, tile), dtype=np.float32)
        for i in range(n_tiles):
            r = rng.integers(0, h_src - tile)
            c = rng.integers(0, w_src - tile)
            tiles_raw[i] = src[r : r + tile, c : c + tile] * field * mean_scale

        model = BaSiC(tiles_raw, estimate_darkfield=False)
        model.prepare()
        model.run()
        ff = model.get_flatfield()
        tiles_cor = tiles_raw / (ff[np.newaxis] + 1e-6)

        raw_sides.append((field * mean_scale)[tile // 2, :])
        est_sides.append(ff[tile // 2, :])
        residual = field * mean_scale / (ff + 1e-6)
        corrected_sides.append(residual[tile // 2, :])
        seam_before.append(_seam_discrepancy(tiles_raw, grid, grid, overlap_px))
        seam_after.append(_seam_discrepancy(tiles_cor, grid, grid, overlap_px))

    fig = viz.figure_focal_volume(
        raw_side=np.array(raw_sides),
        est_side=np.array(est_sides),
        corrected_side=np.array(corrected_sides),
        seam_before=np.array(seam_before),
        seam_after=np.array(seam_after),
        focal_z=focal_z,
        title="Illumination focal curve through a synthetic 3-D volume",
    )
    path = viz.save_figure(fig, out / "focal_volume_demo.png", dpi=150)
    print(f"Saved: {path}")


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
    fig2 = viz.figure_panels(
        [
            viz.Panel(full_corrupted, "Full image — corrupted (synthetic vignette)", kind="image"),
            viz.Panel(full_corrected, "Full image — corrected by Linum BaSiC", kind="image"),
        ],
        ncols=2,
        figsize=(14, 5),
    )
    full_path = viz.save_figure(fig2, out / "full_comparison.png", dpi=100)

    # ------------------------------------------------------------------
    # 5. Save the 3-panel comparison figure
    # ------------------------------------------------------------------
    fig = viz.figure_panels(
        [
            viz.Panel(sample_corrupted, "Corrupted tile\n(with vignette)", kind="image"),
            viz.Panel(flatfield, "Estimated flat-field", kind="flatfield"),
            viz.Panel(sample_corrected, "Corrected tile", kind="image"),
        ],
        ncols=3,
        figsize=(11, 3.8),
        suptitle="BaSiC illumination correction demo",
    )
    comparison_path = viz.save_figure(fig, out / "demo_comparison.png", dpi=150)

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
    # Shared colour limits for flat-field row and dark-field row
    ff_vmin = min(gt_vignette.min(), est_flatfield.min())
    ff_vmax = max(gt_vignette.max(), est_flatfield.max())
    df_vmax = max(gt_darkfield.max(), est_darkfield.max())

    fig3 = viz.figure_panels(
        [
            viz.Panel(sample_corrupt_df, "Corrupted tile\n(vignette + dark-field)", kind="image"),
            viz.Panel(
                gt_vignette,
                f"GT flat-field\n(mean = {gt_vignette.mean():.2f})",
                kind="flatfield",
                vmin=ff_vmin,
                vmax=ff_vmax,
            ),
            viz.Panel(
                gt_darkfield,
                f"GT dark-field\n(max = {gt_darkfield.max():.3f})",
                kind="darkfield",
                vmin=0,
                vmax=df_vmax,
            ),
            viz.Panel(sample_corrected_df, "Corrected tile", kind="image"),
            viz.Panel(
                est_flatfield,
                f"Estimated flat-field\n(mean = {est_flatfield.mean():.2f})",
                kind="flatfield",
                vmin=ff_vmin,
                vmax=ff_vmax,
            ),
            viz.Panel(
                est_darkfield,
                f"Estimated dark-field\n(max = {est_darkfield.max():.3f})",
                kind="darkfield",
                vmin=0,
                vmax=df_vmax,
            ),
        ],
        ncols=3,
        figsize=(12, 7),
        suptitle="BaSiC dark-field estimation demo — ground truth vs estimate",
    )
    darkfield_path = viz.save_figure(fig3, out / "darkfield_demo_comparison.png", dpi=150)

    print(f"Saved: {darkfield_path}")
    print(f"Saved: {out / 'darkfield_gt.png'}")
    print(f"Saved: {out / 'darkfield_estimated.png'}")

    # ------------------------------------------------------------------
    # 7. 3-D flat-field surface (illumination landscape)
    # ------------------------------------------------------------------
    print("\nGenerating 3-D flat-field surface...")
    _save_flatfield_3d(flatfield, out)

    # ------------------------------------------------------------------
    # 8. Tuning / training demo on a synthetic overlapping mosaic
    # ------------------------------------------------------------------
    _save_tuning_demo(out)

    # ------------------------------------------------------------------
    # 9. Seam-consistency metric demo — two adjacent tiles, before/after
    # ------------------------------------------------------------------
    _save_seam_metric_demo(out)
    print("Done.")


if __name__ == "__main__":
    main()
