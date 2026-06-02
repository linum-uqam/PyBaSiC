#!/usr/bin/env python3
"""GPU benchmark for the BaSiC ALM solver.

Compares NumPy (CPU) vs. Torch-CUDA performance across a range of stack sizes
and working resolutions.  Prints a markdown table with per-iteration time,
total wall time, peak VRAM, and speedup.

Usage::

    uv run python scripts/benchmark_gpu.py [--n 8 32 128] [--size 64 128] \\
                                            [--iters 50] [--no-darkfield]

Requirements: CUDA-capable GPU, PyTorch with CUDA support.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

_WARMUP_ITERS = 3  # BaSiC reweighting iterations counted as warmup


def _make_stack(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a synthetic image stack with a mild vignette and noise."""
    cy, cx = np.mgrid[0:size, 0:size]
    r2 = ((cy - size / 2) ** 2 + (cx - size / 2) ** 2) / (size / 2) ** 2
    flatfield = 1.0 - 0.3 * r2  # radially-decaying vignette
    imgs = np.empty((n, size, size), dtype=np.float32)
    for i in range(n):
        imgs[i] = flatfield * (0.8 + 0.4 * rng.random((size, size), dtype=np.float32))
    return imgs


def _run_once(
    imgs: np.ndarray,
    *,
    estimate_darkfield: bool,
    backend: str,
    device: str | None,
    max_iter: int,
) -> dict[str, Any]:
    """Run inexact_alm_l1 once and return timing + VRAM stats."""
    from linum_basic._alm import inexact_alm_l1
    from linum_basic.backend import get_xp

    xp = get_xp(backend, device=device)

    # Compute l_s and l_d the same way BaSiC.prepare() does.
    mean_img = imgs.mean(axis=0)
    from scipy.fft import dctn as _dctn  # type: ignore[import-untyped]

    dct_sum = float(np.abs(_dctn(mean_img / (mean_img.mean() + 1e-9), norm="ortho")).sum())
    l_s = dct_sum / 800.0
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=[8, 32, 128],
        metavar="N",
        help="Stack depths to benchmark (default: 8 32 128).",
    )
    parser.add_argument(
        "--size",
        nargs="+",
        type=int,
        default=[64, 128],
        metavar="SIZE",
        help="Image working_size values to benchmark (default: 64 128).",
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

    rng = np.random.default_rng(42)

    header = f"{'N':>5}  {'size':>6}  {'backend':>8}  {'total_s':>8}  {'iter_ms':>8}  {'VRAM_MB':>8}  {'speedup':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    configs = [(n, sz) for n in args.n for sz in args.size]
    for n, sz in configs:
        imgs = _make_stack(n, sz, rng)

        np_stats = _run_once(
            imgs,
            estimate_darkfield=args.darkfield,
            backend="numpy",
            device=None,
            max_iter=args.iters,
        )

        cuda_stats: dict[str, Any] | None = None
        if cuda_ok:
            cuda_stats = _run_once(
                imgs,
                estimate_darkfield=args.darkfield,
                backend="torch",
                device=args.device,
                max_iter=args.iters,
            )

        speedup = np_stats["elapsed_s"] / cuda_stats["elapsed_s"] if cuda_stats else float("nan")
        vram = cuda_stats["vram_mb"] if cuda_stats else float("nan")
        cuda_total = cuda_stats["elapsed_s"] if cuda_stats else float("nan")
        cuda_iter = cuda_stats["iter_ms"] if cuda_stats else float("nan")

        print(f"{n:>5}  {sz:>6}  {'numpy':>8}  {np_stats['elapsed_s']:>8.3f}  {np_stats['iter_ms']:>8.2f}  {'':>8}  {'':>8}")
        if cuda_ok:
            print(f"{n:>5}  {sz:>6}  {'cuda':>8}  {cuda_total:>8.3f}  {cuda_iter:>8.2f}  {vram:>8.1f}  {speedup:>8.1f}x")

    print(sep)
    print(f"(max_iter={args.iters}, estimate_darkfield={args.darkfield})")


if __name__ == "__main__":
    main()
