"""Visualize flat-field focal curvature from an OME-Zarr mosaic.

Fits one BaSiC flat-field per z-level and plots:

  1. YZ side-view  -- flatfield[:, tile_h//2, :]  (z x x, middle row)
  2. XZ side-view  -- flatfield[:, :, tile_w//2]  (z x y, middle col)
  3. Raw vs corrected montage for a chosen z-level

Usage::

    uv run python scripts/visualize_flatfield_curvature.py \\
        --input /path/to/mosaic.ome.zarr \\
        --z-indices 0 10 20 30 40 50 \\
        --z-inspect 27 \\
        --overlap 0.2 \\
        --output docs/_static/demo/flatfield_curvature.png

All arguments have defaults that work with the development test dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_ZARR = "/Users/Frans/Downloads/sub-22/mosaic_grid_z27_focal_fix.ome.zarr"
_DEFAULT_OUT = str(Path(_DEFAULT_ZARR).parent / "flatfield_curvature.png")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", default=_DEFAULT_ZARR, metavar="ZARR", help="Path to the OME-Zarr mosaic (default: %(default)s)."
    )
    p.add_argument("--output", default=_DEFAULT_OUT, metavar="PNG", help="Output PNG path (default: %(default)s).")
    p.add_argument(
        "--z-indices",
        nargs="+",
        type=int,
        default=None,
        metavar="Z",
        help="Z-levels to fit. Default: evenly-spaced 10 levels.",
    )
    p.add_argument(
        "--z-inspect",
        type=int,
        default=None,
        metavar="Z",
        help="Z-level for raw-vs-corrected panel. Default: middle of --z-indices.",
    )
    p.add_argument("--overlap", type=float, default=0.2, metavar="FRAC", help="Tile overlap fraction (default: %(default)s).")
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.0,
        metavar="SIGMA",
        help="Optional Gaussian sigma for depth-smoothing flatfields (0 = off).",
    )
    p.add_argument(
        "--estimate-darkfield", action="store_true", default=False, help="Estimate dark-field in addition to flat-field."
    )
    p.add_argument("--verbose", action="store_true", default=False, help="Show BaSiC progress bars.")
    p.add_argument(
        "--n-extra",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Number of galvo-return / scan-settling rows at the top of each tile to exclude "
            "from the BaSiC fit and replace by edge-extension in the corrected output. "
            "Set to 0 to disable masking. Default: %(default)s (empirically determined from "
            "per-row intensity diagnostics on the acquisition data)."
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _load_and_fit(
    zarr_path: str,
    z_indices: list[int] | None,
    overlap: float,
    estimate_darkfield: bool,
    n_extra_rows: int,
    verbose: bool,
):
    from linum_basic.fit import fit_mosaic
    from linum_basic.mosaic import MosaicGrid

    print(f"Loading mosaic from {zarr_path} …")
    mosaic = MosaicGrid.from_ome_zarr(zarr_path, overlap_fraction=overlap)
    n_z = mosaic.n_z
    print(f"  shape={mosaic.array.shape}  tile={mosaic.tile_shape}  grid={mosaic.n_rows}x{mosaic.n_cols}")

    if z_indices is None:
        z_indices = list(range(n_z))

    print(f"Fitting {len(z_indices)} z-levels: {z_indices}")
    basic_kwargs = {"estimate_darkfield": estimate_darkfield}
    fit = fit_mosaic(mosaic, z_indices=z_indices, basic_kwargs=basic_kwargs, n_extra_rows=n_extra_rows, verbose=verbose)
    print(f"  flatfields shape: {fit.flatfields.shape}")
    return mosaic, fit


def _smooth_fields(flatfields: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return flatfields
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(flatfields, sigma=sigma, axis=0)


def _build_figure(
    mosaic, fit, z_inspect: int, smooth_sigma: float, out_path: Path, zarr_path: str = "", n_extra_rows: int = 0
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    flatfields = _smooth_fields(fit.flatfields.copy(), smooth_sigma)
    _n_z_fit, th, tw = flatfields.shape
    z_indices = fit.z_indices

    # -----------------------------------------------------------------------
    # YZ and XZ side-views
    # -----------------------------------------------------------------------
    mid_row = th // 2
    mid_col = tw // 2
    yz_view = flatfields[:, mid_row, :]  # shape (n_z_fit, tw)  -- z x x
    xz_view = flatfields[:, :, mid_col]  # shape (n_z_fit, th)  -- z x y

    z_labels = np.array(z_indices)

    # -----------------------------------------------------------------------
    # Raw vs corrected at z_inspect
    # -----------------------------------------------------------------------
    from linum_basic.fit import apply_fit

    z_fit_idx = z_indices.index(z_inspect)
    ff = flatfields[z_fit_idx]
    raw_z = mosaic.array[z_inspect]  # (H, W)
    corrected_z = apply_fit(mosaic, fit, n_extra_rows=n_extra_rows)[z_inspect]
    # clip to [0, 99th-percentile] for display
    p99 = float(np.percentile(raw_z[raw_z > 0], 99))
    raw_disp = np.clip(raw_z, 0, p99)
    p99c = float(np.percentile(corrected_z[corrected_z > 0], 99)) if corrected_z.max() > 0 else 1.0
    corr_disp = np.clip(corrected_z, 0, p99c)

    # -----------------------------------------------------------------------
    # Layout: 2x2 + 1 wide
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax_yz = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[1, 0])
    ax_ff = fig.add_subplot(gs[0, 1])
    ax_raw = fig.add_subplot(gs[0, 2])
    ax_corr = fig.add_subplot(gs[1, 2])
    ax_prof = fig.add_subplot(gs[1, 1])

    # --- YZ view ---
    im = ax_yz.imshow(
        yz_view,
        aspect="auto",
        origin="lower",
        extent=(0, float(tw), float(z_labels[0]), float(z_labels[-1])),
        cmap="RdYlGn",
        vmin=0.7,
        vmax=1.3,
    )
    ax_yz.set_xlabel("x-pixel within tile")
    ax_yz.set_ylabel("z-index")
    ax_yz.set_title("Flat-field YZ slice\n(middle row of each z-level)")
    fig.colorbar(im, ax=ax_yz, shrink=0.8, label="flat-field value")

    # --- XZ view ---
    im2 = ax_xz.imshow(
        xz_view,
        aspect="auto",
        origin="lower",
        extent=(0, float(th), float(z_labels[0]), float(z_labels[-1])),
        cmap="RdYlGn",
        vmin=0.7,
        vmax=1.3,
    )
    ax_xz.set_xlabel("y-pixel within tile")
    ax_xz.set_ylabel("z-index")
    ax_xz.set_title("Flat-field XZ slice\n(middle col of each z-level)")
    fig.colorbar(im2, ax=ax_xz, shrink=0.8, label="flat-field value")

    # --- Flat-field at z_inspect ---
    im3 = ax_ff.imshow(ff, cmap="RdYlGn", vmin=0.7, vmax=1.3, origin="lower")
    ax_ff.set_title(f"Flat-field at z={z_inspect}")
    ax_ff.set_xlabel("x-pixel")
    ax_ff.set_ylabel("y-pixel")
    fig.colorbar(im3, ax=ax_ff, shrink=0.8, label="flat-field value")

    # --- Depth profile of center pixel ---
    center_vals = flatfields[:, mid_row, mid_col]
    ax_prof.plot(z_labels, center_vals, "o-", color="steelblue", markersize=4)
    ax_prof.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax_prof.set_xlabel("z-index")
    ax_prof.set_ylabel("flat-field (center pixel)")
    ax_prof.set_title("Depth profile — center pixel")
    ax_prof.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_prof.grid(True, which="major", alpha=0.4)

    # --- Raw mosaic ---
    ax_raw.imshow(raw_disp, cmap="gray", origin="lower")
    ax_raw.set_title(f"Raw mosaic  z={z_inspect}")
    ax_raw.axis("off")

    # --- Corrected mosaic ---
    ax_corr.imshow(corr_disp, cmap="gray", origin="lower")
    ax_corr.set_title(f"Corrected mosaic  z={z_inspect}")
    ax_corr.axis("off")

    dataset_name = Path(zarr_path).name if zarr_path else "dataset"
    fig.suptitle(
        f"BaSiC flat-field focal curvature  |  {dataset_name}",
        fontsize=13,
        fontweight="bold",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    mosaic, fit = _load_and_fit(
        zarr_path=args.input,
        z_indices=args.z_indices,
        overlap=args.overlap,
        estimate_darkfield=args.estimate_darkfield,
        n_extra_rows=args.n_extra,
        verbose=args.verbose,
    )

    z_inspect = args.z_inspect
    if z_inspect is None:
        z_inspect = fit.z_indices[len(fit.z_indices) // 2]

    if z_inspect not in fit.z_indices:
        print(
            f"Error: --z-inspect={z_inspect} was not in the fitted z-indices {fit.z_indices}.\n"
            f"Pass it explicitly via --z-indices.",
            file=sys.stderr,
        )
        return 1

    _build_figure(
        mosaic,
        fit,
        z_inspect=z_inspect,
        smooth_sigma=args.smooth_sigma,
        out_path=Path(args.output),
        zarr_path=args.input,
        n_extra_rows=args.n_extra,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
