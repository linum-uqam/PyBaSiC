#!/usr/bin/env python3
"""GPU iteration benchmark on a production-shaped mosaic volume.

Compares outer reweighting caps, inner ALM caps, and flat-field regularisation
(``l_s`` auto vs pipeline override).  Intended for A6000 server runs on sub-22
focal-corrected mosaics.

Example::

    uv run python scripts/iteration_bench_gpu.py \\
        --input /scratch/workspace/sub-22/output/27/fix_focal_curvature/mosaic_grid_z27_focal_fix.ome.zarr \\
        --z-sample 5 --tile-fov-mm 0.875
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="Mosaic-grid OME-Zarr (Z, Y, X).")
    p.add_argument("--tile-fov-mm", type=float, default=0.875, help="Tile FOV in mm [%(default)s].")
    p.add_argument("--z-sample", type=int, default=5, help="Number of z-planes to fit (uniform sample).")
    p.add_argument("--fit-max-samples", type=int, default=8000, help="Tile budget (matches pipeline).")
    p.add_argument("--device", default="cuda", help="Torch device [%(default)s].")
    p.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return p


def _load_volume(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    from linum_basic.io.zarr import load_ome_zarr

    array, _axes, scale = load_ome_zarr(path)
    if np.iscomplexobj(array):
        array = np.abs(array)
    # scale is [z, y, x] physical spacing; use y pixel size in mm if present
    pixel_size_mm = float(scale[1]) if len(scale) > 1 else 0.01
    resolution = (float(scale[0]), pixel_size_mm, pixel_size_mm)
    return array.astype(np.float32, copy=False), resolution


def _crop_to_tiles(array: np.ndarray, tile_shape: tuple[int, int]) -> np.ndarray:
    th, tw = tile_shape
    h_crop = (array.shape[1] // th) * th
    w_crop = (array.shape[2] // tw) * tw
    if h_crop != array.shape[1] or w_crop != array.shape[2]:
        print(f"Cropping ({array.shape[1]},{array.shape[2]}) -> ({h_crop},{w_crop})")
    return array[:, :h_crop, :w_crop]


def _z_indices(n_z: int, tiles_per_plane: int, fit_max_samples: int, z_sample: int) -> list[int]:
    n_planes_for_fit = min(n_z, max(1, fit_max_samples // tiles_per_plane))
    n_planes_for_fit = min(n_planes_for_fit, z_sample)
    if n_planes_for_fit >= n_z:
        return list(range(n_z))
    return np.linspace(0, n_z - 1, n_planes_for_fit, dtype=int).tolist()


def _run_case(
    label: str,
    mosaic,
    z_indices: list[int],
    *,
    basic_kwargs: dict,
    n_workers: int = 2,
) -> dict:
    from linum_basic.fit import fit_mosaic
    from linum_basic.metrics import evaluate_correction_volume

    t0 = time.perf_counter()
    fit = fit_mosaic(
        mosaic,
        z_indices=z_indices,
        field_mode="per-z",
        basic_kwargs=basic_kwargs,
        n_workers=n_workers,
        verbose=True,
    )
    wall_s = time.perf_counter() - t0
    metrics = evaluate_correction_volume(mosaic, fit, metrics=("seam", "curvature"))
    conv = fit.convergence_per_z or []
    reweight_iters = [int(c["reweighting_iteration"]) for c in conv]
    alm_iters = [int(c.get("alm_iterations_last", 0)) for c in conv]
    l_s_vals = [c.get("l_s") for c in conv if c.get("l_s") is not None]
    row = {
        "label": label,
        "wall_s": wall_s,
        "seam_l1": float(metrics["seam_l1"]),
        "seam_curvature": float(metrics["seam_curvature"]),
        "reweight_iters_median": float(np.median(reweight_iters)) if reweight_iters else None,
        "reweight_iters_max": int(max(reweight_iters)) if reweight_iters else None,
        "reweight_iters_per_z": reweight_iters,
        "alm_iters_median": float(np.median(alm_iters)) if alm_iters else None,
        "alm_iters_max": int(max(alm_iters)) if alm_iters else None,
        "l_s_median": float(np.median(l_s_vals)) if l_s_vals else None,
        "per_z_s": wall_s / max(len(z_indices), 1),
        "basic_kwargs": {k: basic_kwargs[k] for k in sorted(basic_kwargs) if k != "device"},
    }
    print(
        f"[{label}] {wall_s:.1f}s ({row['per_z_s']:.1f}s/z) "
        f"seam_l1={row['seam_l1']:.4f} reweight median={row['reweight_iters_median']} "
        f"alm median={row['alm_iters_median']}"
    )
    return row


def main() -> None:
    args = _build_parser().parse_args()
    array, resolution = _load_volume(args.input)
    pixel_size_mm = float(resolution[1])
    tile_px = round(args.tile_fov_mm / pixel_size_mm)
    tile_shape = (tile_px, tile_px)
    array = _crop_to_tiles(array, tile_shape)

    from linum_basic.mosaic import MosaicGrid

    mosaic = MosaicGrid(array=array, tile_shape=tile_shape)
    tiles_per_plane = mosaic.n_rows * mosaic.n_cols
    z_indices = _z_indices(mosaic.n_z, tiles_per_plane, args.fit_max_samples, args.z_sample)
    print(f"Mosaic: {mosaic.n_z} planes, {mosaic.n_rows}x{mosaic.n_cols} tiles @ {tile_shape}, fitting z={z_indices}")

    base = {
        "backend": "torch",
        "device": args.device,
        "estimate_darkfield": True,
        "working_size": 128,
        "verbose": False,
    }
    cases = [
        # A/B: pipeline flatfield smoothness (l_s) and reweighting caps
        ("A_prod_ls005_cap500", {**base, "max_reweighting_iterations": 500, "alm_max_iter": 500, "l_s": 0.05}),
        ("B_auto_ls_cap500", {**base, "max_reweighting_iterations": 500, "alm_max_iter": 500}),
        ("C_auto_ls_cap15", {**base, "max_reweighting_iterations": 15, "alm_max_iter": 100}),
        ("D_prod_ls005_cap15", {**base, "max_reweighting_iterations": 15, "alm_max_iter": 100, "l_s": 0.05}),
        (
            "E_prod_ls005_tol1e2",
            {**base, "max_reweighting_iterations": 500, "alm_max_iter": 500, "l_s": 0.05, "reweighting_tolerance": 1e-2},
        ),
        ("F_auto_ls_tol1e2", {**base, "max_reweighting_iterations": 500, "alm_max_iter": 500, "reweighting_tolerance": 1e-2}),
    ]

    report = {
        "input": str(args.input),
        "mosaic_shape": list(array.shape),
        "tile_shape": list(tile_shape),
        "z_indices": z_indices,
        "cases": [],
    }
    for label, kw in cases:
        report["cases"].append(_run_case(label, mosaic, z_indices, basic_kwargs=kw, n_workers=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
