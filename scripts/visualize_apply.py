"""Visualise a BaSiC mosaic correction (``apply_fit``) on real data.

This mirrors the synthetic vignette diagnostic figure but **without the
ground-truth column** — for real acquisitions there is no reference field, so
the figure tells the raw -> corrected story directly and reports the seam-
consistency metric before and after correction.

Usage
-----
    uv run python scripts/visualize_apply.py \
        --input /path/to/mosaic.ome.zarr --z-inspect 27 \
        --working-size 160 --output /path/to/apply_correction.png

The fitted hyperparameters default to BaSiC auto-tuning; pass the values found
by :func:`linum_basic.tuning.tune` to reproduce a tuned correction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from linum_basic.fit import MosaicFit, apply_fit, fit_mosaic
from linum_basic.metrics import seam_l1
from linum_basic.mosaic import MosaicGrid

_DEFAULT_ZARR = "/Users/Frans/Downloads/sub-22/mosaic_grid_z27_focal_fix.ome.zarr"
_DEFAULT_OUT = "apply_correction.png"


def _tiles_from_image(image: np.ndarray, th: int, tw: int, nrows: int, ncols: int) -> np.ndarray:
    """Split a single mosaic image into its (nrows*ncols, th, tw) tile stack."""
    out = np.empty((nrows * ncols, th, tw), dtype=image.dtype)
    for r in range(nrows):
        for c in range(ncols):
            out[r * ncols + c] = image[r * th : (r + 1) * th, c * tw : (c + 1) * tw]
    return out


def _build_figure(
    mosaic: MosaicGrid,
    fit: MosaicFit,
    z_inspect: int,
    out_path: Path,
    zarr_path: str,
    n_extra_rows: int,
) -> None:
    """Write the raw -> corrected diagnostic figure (no ground truth)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th, tw = mosaic.tile_shape
    nrows, ncols = mosaic.n_rows, mosaic.n_cols
    z_pos = fit.z_indices.index(z_inspect) if z_inspect in fit.z_indices else 0
    ff = fit.flatfields[z_pos] if fit.field_mode == "per-z" else fit.flatfields
    df = fit.darkfields[z_pos] if fit.field_mode == "per-z" else fit.darkfields
    has_df = float(df.max()) > 1e-6

    raw_z = mosaic.array[z_inspect].astype(np.float32)
    corrected_z = apply_fit(mosaic, fit, n_extra_rows=n_extra_rows)[z_inspect]

    # Seam metric before / after.
    seam_pairs = mosaic.seam_pairs()
    raw_tiles = mosaic.iter_tiles(z_inspect)
    corr_tiles = _tiles_from_image(corrected_z, th, tw, nrows, ncols)
    seam_raw = seam_l1(raw_tiles, seam_pairs)
    seam_corr = seam_l1(corr_tiles, seam_pairs)
    improvement = 100.0 * (seam_raw - seam_corr) / (seam_raw + 1e-12)

    # Pick a representative edge tile (most peripheral content) for the tile row.
    edge_mask = ff < ff.mean()
    best = int(np.argmax(raw_tiles[:, edge_mask].mean(axis=-1)))
    raw_tile = raw_tiles[best]
    corr_tile = corr_tiles[best]

    # Shared display ranges so raw vs corrected are directly comparable.
    mos_vmax = float(np.percentile(raw_z, 99))
    tile_vmax = float(np.percentile(raw_tile, 99))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"BaSiC mosaic correction  |  {Path(zarr_path).name}  |  z={z_inspect}\n"
        f"seam L1: {seam_raw:.3f} -> {seam_corr:.3f}  ({improvement:+.1f}%)",
        fontsize=13,
        fontweight="bold",
    )

    # Column 0: estimated fields.
    im = axes[0, 0].imshow(ff, cmap="viridis", interpolation="nearest", vmin=0.7, vmax=1.3)
    axes[0, 0].contour(ff, levels=10, colors="w", linewidths=0.6, alpha=0.7)
    axes[0, 0].set_title("Estimated flat-field", fontsize=10)
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    if has_df:
        im = axes[1, 0].imshow(df, cmap="inferno", interpolation="nearest")
        axes[1, 0].set_title("Estimated dark-field", fontsize=10)
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    else:
        # No dark-field estimated: show per-column means to expose seam flattening.
        axes[1, 0].plot(raw_z.mean(axis=0), color="tab:red", lw=1.0, label="raw")
        axes[1, 0].plot(corrected_z.mean(axis=0), color="tab:green", lw=1.0, label="corrected")
        axes[1, 0].set_title("Column-mean intensity", fontsize=10)
        axes[1, 0].set_xlabel("x-pixel")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].margins(x=0)

    # Column 1: full mosaic raw vs corrected.
    im = axes[0, 1].imshow(raw_z, cmap="gray", interpolation="nearest", vmin=0, vmax=mos_vmax)
    axes[0, 1].set_title("Raw mosaic", fontsize=10)
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(corrected_z, cmap="gray", interpolation="nearest", vmin=0, vmax=mos_vmax)
    axes[1, 1].set_title("Corrected mosaic", fontsize=10)
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Column 2: representative tile raw vs corrected.
    im = axes[0, 2].imshow(raw_tile, cmap="gray", interpolation="nearest", vmin=0, vmax=tile_vmax)
    axes[0, 2].set_title("Sample tile: raw", fontsize=10)
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im = axes[1, 2].imshow(corr_tile, cmap="gray", interpolation="nearest", vmin=0, vmax=tile_vmax)
    axes[1, 2].set_title("Sample tile: corrected", fontsize=10)
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]):
        ax.axis("off")
    if has_df:
        axes[1, 0].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}  (seam L1 {seam_raw:.3f} -> {seam_corr:.3f}, {improvement:+.1f}%)")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=_DEFAULT_ZARR, help="Path to the OME-Zarr mosaic.")
    parser.add_argument("--z-inspect", type=int, default=27, help="Z-level to fit and display.")
    parser.add_argument("--overlap", type=float, default=0.2, help="Physical tile-overlap fraction (0-1).")
    parser.add_argument("--working-size", type=int, default=None, help="BaSiC working resolution.")
    parser.add_argument("--epsilon", type=float, default=None, help="BaSiC reweighting stability constant.")
    parser.add_argument("--estimate-darkfield", action="store_true", help="Estimate a dark-field.")
    parser.add_argument("--n-extra", type=int, default=2, help="Galvo fly-back rows to mask per tile.")
    parser.add_argument("--output", default=_DEFAULT_OUT, help="Output PNG path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    mosaic = MosaicGrid.from_ome_zarr(args.input, overlap_fraction=args.overlap)
    params: dict[str, Any] = {"estimate_darkfield": args.estimate_darkfield}
    if args.working_size is not None:
        params["working_size"] = args.working_size
    if args.epsilon is not None:
        params["epsilon"] = args.epsilon

    fit = fit_mosaic(
        mosaic,
        z_indices=[args.z_inspect],
        basic_kwargs=params,
        n_extra_rows=args.n_extra,
        verbose=True,
    )
    _build_figure(mosaic, fit, args.z_inspect, Path(args.output), args.input, args.n_extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
