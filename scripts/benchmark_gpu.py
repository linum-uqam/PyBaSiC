#!/usr/bin/env python3
"""GPU benchmark for the BaSiC ALM solver.

Compares NumPy (CPU) vs. Torch-CUDA performance across a range of stack sizes
and working resolutions.  Prints a markdown table with per-iteration time,
total wall time, peak VRAM, and speedup.

Synthetic mode (default)::

    uv run python scripts/benchmark_gpu.py [--n 8 32 128] [--size 64 128] \\
                                            [--iters 50] [--no-darkfield]

Real-data mode (tunes first, then benchmarks)::

    uv run python scripts/benchmark_gpu.py --input /path/to/mosaic.ome.zarr \\
                                            [--overlap 0.2] [--z-level 27] \\
                                            [--tune-trials 30] [--tune-z-subsample 4]

Requirements: CUDA-capable GPU, PyTorch with CUDA support.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

_WARMUP_ITERS = 3  # ALM iterations used for torch.compile warmup (triggers compilation)


def _make_stack(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a synthetic image stack with a mild vignette and noise."""
    cy, cx = np.mgrid[0:size, 0:size]
    r2 = ((cy - size / 2) ** 2 + (cx - size / 2) ** 2) / (size / 2) ** 2
    flatfield = 1.0 - 0.3 * r2  # radially-decaying vignette
    imgs = np.empty((n, size, size), dtype=np.float32)
    for i in range(n):
        imgs[i] = flatfield * (0.8 + 0.4 * rng.random((size, size), dtype=np.float32))
    return imgs


def _load_zarr_mosaic(path: str, overlap_fraction: float) -> Any:
    """Load an OME-Zarr file as a MosaicGrid."""
    from linum_basic.mosaic import MosaicGrid

    return MosaicGrid.from_ome_zarr(path, overlap_fraction=overlap_fraction)


def _extract_tiles_at_z(mosaic: Any, z: int) -> np.ndarray:
    """Extract all tiles at a given z-level into a (n_tiles, th, tw) stack."""
    n_rows, n_cols = mosaic.n_rows, mosaic.n_cols
    tiles = np.stack(
        [mosaic.get_tile(z, r, c) for r in range(n_rows) for c in range(n_cols)],
        axis=0,
    )
    return tiles.astype(np.float32)


def _run_tuning(
    mosaic: Any,
    *,
    n_trials: int,
    z_subsample: int,
    backend: str,
    device: str | None,
) -> dict[str, Any]:
    """Run Optuna tuning on *mosaic* and return best BaSiC params."""
    from linum_basic.tuning import tune

    print(f"Running tuning ({n_trials} trials, z_subsample={z_subsample}) …")
    result = tune(
        mosaic,
        n_trials=n_trials,
        z_subsample=z_subsample,
        backend=backend,
        device=device,
        run_full_fit=False,
        verbose=True,
    )
    print(f"Best params (objective={result.objective}, value={result.best_value:.6f}):")
    for k, v in result.best_params.items():
        print(f"  {k}: {v}")
    return result.best_params


def _run_once(
    imgs: np.ndarray,
    *,
    estimate_darkfield: bool,
    backend: str,
    device: str | None,
    max_iter: int,
    l_s: float | None = None,
    l_d: float | None = None,
) -> dict[str, Any]:
    """Run inexact_alm_l1 once and return timing + VRAM stats."""
    from linum_basic._alm import inexact_alm_l1
    from linum_basic.backend import get_xp

    xp = get_xp(backend, device=device)

    # Compute l_s and l_d the same way BaSiC.prepare() does (unless overridden).
    if l_s is None or l_d is None:
        mean_img = imgs.mean(axis=0)
        from scipy.fft import dctn as _dctn  # type: ignore[import-untyped]

        dct_sum = float(np.abs(_dctn(mean_img / (mean_img.mean() + 1e-9), norm="ortho")).sum())
        if l_s is None:
            l_s = dct_sum / 800.0
        if l_d is None:
            l_d = dct_sum / 2000.0

    # Reset VRAM stats.
    vram_before = 0.0
    if backend == "torch" and device and device.startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    inexact_alm_l1(
        imgs,
        l_s,
        l_d,
        estimate_darkfield=estimate_darkfield,
        max_iter=max_iter,
        xp=xp,
    )
    if backend == "torch" and device and device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    vram_mb = 0.0
    if backend == "torch" and device and device.startswith("cuda"):
        import torch

        vram_mb = torch.cuda.max_memory_allocated(device) / 1e6 - vram_before

    return {
        "elapsed_s": t1 - t0,
        "vram_mb": vram_mb,
        "iter_ms": (t1 - t0) * 1000.0 / max_iter,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- real-data source ---
    parser.add_argument(
        "--input",
        default=None,
        metavar="PATH",
        help="Path to OME-Zarr mosaic. If given, load real data and run tuning before benchmarking.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        metavar="FRAC",
        help="Tile overlap fraction for MosaicGrid (default: 0.2).",
    )
    parser.add_argument(
        "--z-level",
        type=int,
        default=None,
        metavar="Z",
        help="Z-level used to extract tiles for benchmarking (default: middle z).",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=30,
        metavar="N",
        help="Number of Optuna trials for tuning (default: 30).",
    )
    parser.add_argument(
        "--tune-z-subsample",
        type=int,
        default=4,
        metavar="N",
        help="Z-levels sampled per tuning trial (default: 4).",
    )
    # --- synthetic-mode controls ---
    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=[8, 32, 128],
        metavar="N",
        help="[synthetic] Stack depths to benchmark (default: 8 32 128).",
    )
    parser.add_argument(
        "--size",
        nargs="+",
        type=int,
        default=[64, 128],
        metavar="SIZE",
        help="[synthetic] Image sizes to benchmark (default: 64 128).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        metavar="ITERS",
        help="ALM max_iter for each benchmark run (default: 50).",
    )
    parser.add_argument(
        "--no-darkfield",
        dest="darkfield",
        action="store_false",
        default=True,
        help="Disable darkfield estimation during benchmarking.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device string, e.g. cuda or cuda:0 (default: cuda).",
    )
    args = parser.parse_args()

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False

    if not cuda_ok:
        print("No CUDA device available — only NumPy results will be shown.")

    # ------------------------------------------------------------------
    # Build benchmark configurations
    # ------------------------------------------------------------------
    # Each config: (label, imgs, l_s_override, l_d_override, estimate_darkfield)
    configs: list[tuple[str, np.ndarray, float | None, float | None, bool]] = []

    if args.input is not None:
        print(f"Loading mosaic from {args.input} …")
        mosaic = _load_zarr_mosaic(args.input, overlap_fraction=args.overlap)
        nz = mosaic.array.shape[0]
        z = args.z_level if args.z_level is not None else nz // 2
        print(f"  Mosaic: {nz}z x {mosaic.n_rows}r x {mosaic.n_cols}c  tile {mosaic.tile_shape}")
        print(f"  Using z-level {z} for benchmarking.")

        # Tune to get best parameters
        best_params = _run_tuning(
            mosaic,
            n_trials=args.tune_trials,
            z_subsample=args.tune_z_subsample,
            backend="numpy",  # tune on CPU to avoid GPU memory contention during tuning
            device=None,
        )

        imgs = _extract_tiles_at_z(mosaic, z)
        n_tiles, th, tw = imgs.shape
        estimate_darkfield = bool(best_params.get("estimate_darkfield", args.darkfield))
        label = f"{n_tiles}tiles/{th}x{tw}"
        configs.append((label, imgs, best_params.get("l_s"), best_params.get("l_d"), estimate_darkfield))
    else:
        rng = np.random.default_rng(42)
        for n in args.n:
            for sz in args.size:
                imgs = _make_stack(n, sz, rng)
                configs.append((f"{n}x{sz}x{sz}", imgs, None, None, args.darkfield))

    header = f"{'config':>20}  {'backend':>8}  {'total_s':>8}  {'iter_ms':>8}  {'VRAM_MB':>8}  {'speedup':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for label, imgs, l_s_ov, l_d_ov, estimate_darkfield in configs:
        np_stats = _run_once(
            imgs,
            estimate_darkfield=estimate_darkfield,
            backend="numpy",
            device=None,
            max_iter=args.iters,
            l_s=l_s_ov,
            l_d=l_d_ov,
        )

        cuda_stats: dict[str, Any] | None = None
        if cuda_ok:
            # Warmup: force torch.compile to trace and compile kernels before
            # the timed run.  Without this, the first call pays the ~2-3 s
            # TorchDynamo trace + Triton compilation overhead, masking the
            # actual GPU execution speed.
            _run_once(
                imgs,
                estimate_darkfield=estimate_darkfield,
                backend="torch",
                device=args.device,
                max_iter=_WARMUP_ITERS,
                l_s=l_s_ov,
                l_d=l_d_ov,
            )
            cuda_stats = _run_once(
                imgs,
                estimate_darkfield=estimate_darkfield,
                backend="torch",
                device=args.device,
                max_iter=args.iters,
                l_s=l_s_ov,
                l_d=l_d_ov,
            )

        speedup = np_stats["elapsed_s"] / cuda_stats["elapsed_s"] if cuda_stats else float("nan")
        vram = cuda_stats["vram_mb"] if cuda_stats else float("nan")
        cuda_total = cuda_stats["elapsed_s"] if cuda_stats else float("nan")
        cuda_iter = cuda_stats["iter_ms"] if cuda_stats else float("nan")

        print(f"{label:>20}  {'numpy':>8}  {np_stats['elapsed_s']:>8.3f}  {np_stats['iter_ms']:>8.2f}  {'':>8}  {'':>8}")
        if cuda_ok:
            print(f"{label:>20}  {'cuda':>8}  {cuda_total:>8.3f}  {cuda_iter:>8.2f}  {vram:>8.1f}  {speedup:>8.1f}x")

    print(sep)
    print(f"(max_iter={args.iters})")


if __name__ == "__main__":
    main()
