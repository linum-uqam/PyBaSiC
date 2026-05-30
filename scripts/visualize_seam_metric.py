"""Visualize the seam-consistency metric on a real OME-Zarr mosaic.

Fits a BaSiC flat-field for one z-level of a mosaic grid, then renders the
seam-metric demo figure (:func:`linum_basic.viz.figure_seam_metric`).  The
figure has three panels: the BaSiC flat-field as a 3-D surface (showing the
illumination curvature that causes seams), a row of raw tiles stitched
side-by-side (seams visible), and the same row after correction (seamless).

Usage::

    uv run python scripts/visualize_seam_metric.py \\
        --input /path/to/mosaic.ome.zarr \\
        --output seam_metric.png \\
        --z 0 --row 2 \\
        --overlap 0.2

``--input`` and ``--output`` are required; all other arguments have sensible
defaults that adapt to the mosaic's grid shape.
"""

from __future__ import annotations

import argparse

from linum_basic import viz

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, metavar="ZARR", help="Path to the OME-Zarr mosaic.")
    p.add_argument("--output", required=True, metavar="PNG", help="Output PNG path.")
    p.add_argument("--z", type=int, default=0, metavar="Z", help="Z-level to inspect (default: %(default)s).")
    p.add_argument("--row", type=int, default=None, metavar="R", help="Tile row to use (default: middle row).")
    p.add_argument(
        "--orientation",
        choices=("horizontal", "vertical"),
        default="horizontal",
        help="Tile adjacency direction (default: %(default)s).",
    )
    p.add_argument("--overlap", type=float, default=0.2, metavar="FRAC", help="Tile overlap fraction (default: %(default)s).")
    p.add_argument(
        "--estimate-darkfield", action="store_true", default=False, help="Estimate dark-field in addition to flat-field."
    )
    p.add_argument(
        "--n-extra",
        type=int,
        default=2,
        metavar="N",
        help="Galvo-return rows at the top of each tile to exclude from the fit (default: %(default)s).",
    )
    p.add_argument("--verbose", action="store_true", default=False, help="Show BaSiC progress bars.")
    return p


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    import numpy as np

    from linum_basic.fit import fit_mosaic
    from linum_basic.mosaic import MosaicGrid

    print(f"Loading mosaic from {args.input} …")
    mosaic = MosaicGrid.from_ome_zarr(args.input, overlap_fraction=args.overlap)
    print(f"  shape={mosaic.array.shape}  tile={mosaic.tile_shape}  grid={mosaic.n_rows}x{mosaic.n_cols}")

    row = args.row if args.row is not None else mosaic.n_rows // 2
    if row >= mosaic.n_rows:
        row = mosaic.n_rows - 1

    print(f"Fitting global flat-field at z={args.z} (darkfield={args.estimate_darkfield}) …")
    fit = fit_mosaic(
        mosaic,
        z_indices=[args.z],
        field_mode="global",
        basic_kwargs={"estimate_darkfield": args.estimate_darkfield},
        n_extra_rows=args.n_extra,
        verbose=args.verbose,
    )
    print(f"  flatfield shape: {fit.flatfields.shape}")

    # Extract all tiles in the chosen row.
    if args.orientation == "horizontal":
        n_tiles = mosaic.n_cols
        tiles_raw = np.stack([mosaic.get_tile(args.z, row, c) for c in range(n_tiles)])
    else:
        n_tiles = mosaic.n_rows
        tiles_raw = np.stack([mosaic.get_tile(args.z, r, row) for r in range(n_tiles)])

    dark = fit.darkfields if args.estimate_darkfield else 0.0
    flat = fit.flatfields  # (th, tw) for field_mode="global"
    tiles_cor = (tiles_raw - dark) / (flat + 1e-6)

    print(f"Extracted {n_tiles} tiles for row={row}, orientation={args.orientation}")

    fig = viz.figure_seam_metric(
        flatfield=flat,
        tiles_raw=tiles_raw,
        tiles_cor=tiles_cor,
        orientation=args.orientation,
        overlap_fraction=args.overlap,
        title=f"Seam-consistency metric on real data (z={args.z})",
    )
    path = viz.save_figure(fig, args.output, dpi=150)
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
