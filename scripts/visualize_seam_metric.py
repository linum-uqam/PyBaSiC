"""Visualize the seam-consistency metric on a real OME-Zarr mosaic.

Fits a BaSiC flat-field for one z-level of a mosaic grid, picks two adjacent
tiles that share a physical overlap, and renders the seam-metric demo figure
(:func:`linum_basic.viz.figure_seam_metric`).  The figure shows the two
stitched tiles as a 2-D image — raw (with visible seam) on the left and
BaSiC-corrected (seamless) on the right.

Usage::

    uv run python scripts/visualize_seam_metric.py \\
        --input /path/to/mosaic.ome.zarr \\
        --output seam_metric.png \\
        --z 0 --row 2 --col 2 \\
        --overlap 0.2

``--input`` and ``--output`` are required; all other arguments have sensible
defaults that adapt to the mosaic's grid shape.
"""

from __future__ import annotations

import argparse
import sys

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
    p.add_argument(
        "--row", type=int, default=None, metavar="R", help="Tile row of the left/top neighbour (default: middle row)."
    )
    p.add_argument(
        "--col", type=int, default=None, metavar="C", help="Tile column of the left/top neighbour (default: middle col)."
    )
    p.add_argument(
        "--orientation",
        choices=("horizontal", "vertical"),
        default="horizontal",
        help="Adjacency of the two tiles (default: %(default)s).",
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

    from linum_basic.fit import fit_mosaic
    from linum_basic.mosaic import MosaicGrid

    print(f"Loading mosaic from {args.input} …")
    mosaic = MosaicGrid.from_ome_zarr(args.input, overlap_fraction=args.overlap)
    print(f"  shape={mosaic.array.shape}  tile={mosaic.tile_shape}  grid={mosaic.n_rows}x{mosaic.n_cols}")

    row = args.row if args.row is not None else mosaic.n_rows // 2
    col = args.col if args.col is not None else mosaic.n_cols // 2

    # Resolve the neighbouring tile and validate the grid has one.
    if args.orientation == "horizontal":
        if col + 1 >= mosaic.n_cols:
            col = mosaic.n_cols - 2
        neighbour = (row, col + 1)
    else:
        if row + 1 >= mosaic.n_rows:
            row = mosaic.n_rows - 2
        neighbour = (row + 1, col)
    if row < 0 or col < 0:
        print("error: mosaic does not contain two adjacent tiles for this orientation.", file=sys.stderr)
        return 1

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

    raw_a = mosaic.get_tile(args.z, row, col)
    raw_b = mosaic.get_tile(args.z, *neighbour)
    print(f"Tiles: A=(row={row}, col={col})  B=(row={neighbour[0]}, col={neighbour[1]})")

    dark = fit.darkfields if args.estimate_darkfield else 0.0
    flat = fit.flatfields  # (th, tw) for field_mode="global"
    epsilon = 1e-6
    cor_a = (raw_a - dark) / (flat + epsilon)
    cor_b = (raw_b - dark) / (flat + epsilon)

    fig = viz.figure_seam_metric(
        raw_tile_a=raw_a,
        raw_tile_b=raw_b,
        cor_tile_a=cor_a,
        cor_tile_b=cor_b,
        orientation=args.orientation,
        overlap_fraction=args.overlap,
        title=f"Seam-consistency metric on real data (z={args.z})",
    )
    path = viz.save_figure(fig, args.output, dpi=150)
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
