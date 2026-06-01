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

from linum_basic import viz
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

    # Pick a representative edge tile (most peripheral content) for the tile row.
    edge_mask = ff < ff.mean()
    best = int(np.argmax(raw_tiles[:, edge_mask].mean(axis=-1)))
    raw_tile = raw_tiles[best]
    corr_tile = corr_tiles[best]

    fig = viz.figure_apply_correction(
        flatfield=ff,
        raw_mosaic=raw_z,
        corrected_mosaic=corrected_z,
        raw_tile=raw_tile,
        corrected_tile=corr_tile,
        seam_raw=seam_raw,
        seam_corrected=seam_corr,
        darkfield=df if has_df else None,
        title=f"BaSiC mosaic correction  |  {Path(zarr_path).name}  |  z={z_inspect}",
    )
    path = viz.save_figure(fig, out_path, dpi=150)
    improvement = 100.0 * (seam_raw - seam_corr) / (seam_raw + 1e-12)
    print(f"Saved to {path}  (seam L1 {seam_raw:.3f} -> {seam_corr:.3f}, {improvement:+.1f}%)")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=_DEFAULT_ZARR, help="Path to the OME-Zarr mosaic.")
    parser.add_argument("--z-inspect", type=int, default=27, help="Z-level to fit and display.")
    parser.add_argument("--overlap", type=float, default=0.2, help="Physical tile-overlap fraction (0-1).")
    parser.add_argument("--params-json", default=None, help="JSON file with BaSiC hyperparameters (output of basic_tune).")
    parser.add_argument("--working-size", type=int, default=None, help="BaSiC working resolution (overrides --params-json).")
    parser.add_argument(
        "--epsilon", type=float, default=None, help="BaSiC reweighting stability constant (overrides --params-json)."
    )
    parser.add_argument("--estimate-darkfield", action="store_true", help="Estimate a dark-field.")
    parser.add_argument("--n-extra", type=int, default=2, help="Galvo fly-back rows to mask per tile.")
    parser.add_argument("--output", default=_DEFAULT_OUT, help="Output PNG path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    mosaic = MosaicGrid.from_ome_zarr(args.input, overlap_fraction=args.overlap)
    params: dict[str, Any] = {}
    if args.params_json is not None:
        import json

        with Path(args.params_json).open() as fh:
            params = json.load(fh)
    params.setdefault("estimate_darkfield", args.estimate_darkfield)
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
