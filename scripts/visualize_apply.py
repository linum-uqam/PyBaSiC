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

Pass ``--n-z N`` to fit and evaluate N evenly-spaced z-levels (default 10) and
print a per-z seam-metric table.  Use ``--all-z`` to fit every z-level.

The fitted hyperparameters default to BaSiC auto-tuning; pass the values found
by :func:`linum_basic.tuning.tune` to reproduce a tuned correction.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from linum_basic import viz
from linum_basic.fit import MosaicFit, fit_mosaic
from linum_basic.metrics import seam_l1, seam_pearson
from linum_basic.mosaic import MosaicGrid

_DEFAULT_ZARR = None
_DEFAULT_OUT = "apply_correction.png"


def _tiles_from_image(image: np.ndarray, th: int, tw: int, nrows: int, ncols: int) -> np.ndarray:
    """Split a single mosaic image into its (nrows*ncols, th, tw) tile stack."""
    return image.reshape(nrows, th, ncols, tw).transpose(0, 2, 1, 3).reshape(nrows * ncols, th, tw)


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
    view = raw_z.reshape(nrows, th, ncols, tw)
    corrected_view = (view - df[None, :, None, :]) / (ff[None, :, None, :] + 1e-6)
    if n_extra_rows > 0:
        first_valid = corrected_view[:, n_extra_rows : n_extra_rows + 1, :, :]
        corrected_view[:, :n_extra_rows, :, :] = first_valid
    corrected_z = corrected_view.reshape(nrows * th, ncols * tw)

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


def _print_validation_table(
    mosaic: MosaicGrid,
    fit: MosaicFit,
    n_extra_rows: int,
) -> None:
    """Print a per-z before/after seam-metric table for all fitted z-levels."""
    th, tw = mosaic.tile_shape
    nrows, ncols = mosaic.n_rows, mosaic.n_cols
    seam_pairs = mosaic.seam_pairs()

    print()
    print(f"{'z':>4}  {'raw L1':>8}  {'cor L1':>8}  {'ΔL1%':>7}  {'raw 1-r':>8}  {'cor 1-r':>8}")
    print("-" * 58)

    raw_l1_vals: list[float] = []
    cor_l1_vals: list[float] = []
    raw_1mr_vals: list[float] = []
    cor_1mr_vals: list[float] = []

    for z_pos, z in enumerate(fit.z_indices):
        raw_z = mosaic.array[z].astype(np.float32)

        if fit.field_mode == "per-z":
            ff = fit.flatfields[z_pos]
            df = fit.darkfields[z_pos]
        else:
            ff = fit.flatfields
            df = fit.darkfields

        view = raw_z.reshape(nrows, th, ncols, tw)
        corrected_view = (view - df[None, :, None, :]) / (ff[None, :, None, :] + 1e-6)
        if n_extra_rows > 0:
            first_valid = corrected_view[:, n_extra_rows : n_extra_rows + 1, :, :]
            corrected_view[:, :n_extra_rows, :, :] = first_valid
        corrected_z = corrected_view.reshape(nrows * th, ncols * tw)

        raw_tiles = mosaic.iter_tiles(z)
        cor_tiles = _tiles_from_image(corrected_z, th, tw, nrows, ncols)

        rl1 = seam_l1(raw_tiles, seam_pairs)
        cl1 = seam_l1(cor_tiles, seam_pairs)
        rp = 1.0 - seam_pearson(raw_tiles, seam_pairs)
        cp = 1.0 - seam_pearson(cor_tiles, seam_pairs)

        def _fmt(v: float) -> str:
            return "  nan   " if math.isnan(v) else f"{v:8.4f}"

        if not math.isnan(rl1) and not math.isnan(cl1):
            delta = 100.0 * (rl1 - cl1) / (rl1 + 1e-12)
            delta_s = f"{delta:+6.1f}%"
        else:
            delta_s = "    n/a"

        print(f"{z:>4}  {_fmt(rl1)}  {_fmt(cl1)}  {delta_s}  {_fmt(rp)}  {_fmt(cp)}")

        if not math.isnan(rl1):
            raw_l1_vals.append(rl1)
        if not math.isnan(cl1):
            cor_l1_vals.append(cl1)
        if not math.isnan(rp):
            raw_1mr_vals.append(rp)
        if not math.isnan(cp):
            cor_1mr_vals.append(cp)

    print("-" * 58)
    if raw_l1_vals and cor_l1_vals:
        mean_rl1 = float(np.mean(raw_l1_vals))
        mean_cl1 = float(np.mean(cor_l1_vals))
        delta_mean = 100.0 * (mean_rl1 - mean_cl1) / (mean_rl1 + 1e-12)
        print(f"{'mean':>4}  {mean_rl1:8.4f}  {mean_cl1:8.4f}  {delta_mean:+6.1f}%", end="")
        if raw_1mr_vals and cor_1mr_vals:
            print(f"  {float(np.mean(raw_1mr_vals)):8.4f}  {float(np.mean(cor_1mr_vals)):8.4f}")
        else:
            print()
    print()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the OME-Zarr mosaic.")
    parser.add_argument("--z-inspect", type=int, default=27, help="Z-level to fit and display.")
    parser.add_argument("--overlap", type=float, default=0.2, help="Physical tile-overlap fraction (0-1).")
    parser.add_argument("--params-json", default=None, help="JSON file with BaSiC hyperparameters (output of `basic tune`)")
    parser.add_argument("--working-size", type=int, default=None, help="BaSiC working resolution (overrides --params-json).")
    parser.add_argument(
        "--epsilon", type=float, default=None, help="BaSiC reweighting stability constant (overrides --params-json)."
    )
    parser.add_argument("--estimate-darkfield", action="store_true", help="Estimate a dark-field.")
    parser.add_argument("--n-extra", type=int, default=2, help="Galvo fly-back rows to mask per tile.")
    parser.add_argument("--output", default=_DEFAULT_OUT, help="Output PNG path.")
    parser.add_argument(
        "--n-z",
        type=int,
        default=10,
        help="Number of evenly-spaced z-levels to fit and evaluate (default 10). Ignored when --all-z is set.",
    )
    parser.add_argument("--all-z", action="store_true", help="Fit every z-level (overrides --n-z).")
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

    # Determine which z-levels to fit
    n_z = mosaic.n_z
    if args.all_z:
        z_indices = list(range(n_z))
    else:
        count = min(args.n_z, n_z)
        z_indices = list(np.unique(np.linspace(0, n_z - 1, count).round().astype(int)))
    # Always include the display z-level
    if args.z_inspect not in z_indices:
        z_indices = sorted({args.z_inspect, *z_indices})

    fit = fit_mosaic(
        mosaic,
        z_indices=z_indices,
        basic_kwargs=params,
        n_extra_rows=args.n_extra,
        verbose=True,
    )
    _build_figure(mosaic, fit, args.z_inspect, Path(args.output), args.input, args.n_extra)
    _print_validation_table(mosaic, fit, args.n_extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
