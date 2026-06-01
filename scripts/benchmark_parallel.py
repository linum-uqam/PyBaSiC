#!/usr/bin/env python3
"""Parallelism benchmark for :func:`linum_basic.fit.fit_mosaic`.

Measures wall-clock time for ``fit_mosaic`` at several worker counts
(1, 2, 4, 6, 10) on either a real OME-Zarr mosaic or a synthetic dataset,
then writes a JSON result file and prints a summary table.

Run with real data (local)::

    uv run python scripts/benchmark_parallel.py \\
        --input /path/to/mosaic.ome.zarr

Run with synthetic data (no file required)::

    uv run python scripts/benchmark_parallel.py --synthetic

Run on a GPU backend (remote machine)::

    uv run python scripts/benchmark_parallel.py \\
        --input /path/to/mosaic.ome.zarr \\
        --backend torch --device cuda:0

Results are written to ``benchmark_results.json`` (or ``--output``).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from linum_basic.mosaic import MosaicGrid

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORKER_COUNTS = [1, 2, 4, 6, 10]


def _synthetic_mosaic(n_z: int, n_rows: int, n_cols: int, th: int, tw: int) -> MosaicGrid:
    from linum_basic.mosaic import MosaicGrid

    rng = np.random.default_rng(0)
    arr = rng.standard_normal((n_z, n_rows * th, n_cols * tw)).astype(np.float32)
    arr = np.clip(arr * 0.2 + 1.0, 0.01, None)
    return MosaicGrid(array=arr, tile_shape=(th, tw), overlap_fraction=0.2)


def _load_real_mosaic(path: str) -> MosaicGrid:
    from linum_basic.mosaic import MosaicGrid

    return MosaicGrid.from_ome_zarr(path, overlap_fraction=0.2)


def _gpu_info() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if torch.backends.mps.is_available():
            return "Apple MPS"
    except ImportError:
        pass
    return "none"


def _cpu_info() -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        result = subprocess.run(
            ["cat", "/proc/cpuinfo"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[-1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _time_fit(mosaic: MosaicGrid, basic_kwargs: dict[str, Any], n_workers: int) -> float:
    """Return wall-clock seconds for fit_mosaic with *n_workers*."""
    from linum_basic.fit import fit_mosaic

    t0 = time.perf_counter()
    fit_mosaic(mosaic, basic_kwargs=basic_kwargs, n_workers=n_workers, verbose=False)
    return time.perf_counter() - t0


def run_benchmark(
    mosaic: MosaicGrid,
    basic_kwargs: dict[str, Any],
    worker_counts: list[int],
    n_repeats: int,
) -> dict[str, Any]:
    """Run the benchmark and return a result dict."""

    results: dict[str, Any] = {}
    for nw in worker_counts:
        times = []
        for rep in range(n_repeats):
            elapsed = _time_fit(mosaic, basic_kwargs, nw)
            times.append(elapsed)
            print(f"  workers={nw:2d}  rep={rep + 1}/{n_repeats}  {elapsed:.2f}s")
        results[str(nw)] = {
            "times": times,
            "mean": float(np.mean(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
        }
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input", default=None, metavar="PATH", help="Path to OME-Zarr mosaic (local or mounted).")
    src.add_argument("--synthetic", action="store_true", help="Use synthetic data (no file needed).")
    p.add_argument("--backend", default="numpy", choices=["numpy", "torch", "auto"], help="BaSiC compute backend.")
    p.add_argument("--device", default=None, help="PyTorch device string, e.g. 'cuda:0' or 'mps'.")
    p.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Worker counts to test (default: 1 2 4 6 10).",
    )
    p.add_argument("--repeats", type=int, default=2, metavar="N", help="Repetitions per worker count (default: 2).")
    p.add_argument("--working-size", type=int, default=128, help="BaSiC working_size (default: 128).")
    p.add_argument("--output", default="benchmark_results.json", metavar="PATH", help="JSON output file.")
    # Synthetic dataset dimensions
    p.add_argument("--syn-z", type=int, default=40, help="Synthetic: number of z-levels.")
    p.add_argument("--syn-rows", type=int, default=4, help="Synthetic: tile rows.")
    p.add_argument("--syn-cols", type=int, default=4, help="Synthetic: tile columns.")
    p.add_argument("--syn-th", type=int, default=256, help="Synthetic: tile height (pixels).")
    p.add_argument("--syn-tw", type=int, default=256, help="Synthetic: tile width (pixels).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    from linum_basic._parallel import default_workers

    # ------------------------------------------------------------------
    # Load / build mosaic
    # ------------------------------------------------------------------
    if args.synthetic:
        print(
            f"Building synthetic mosaic "
            f"({args.syn_z}z x {args.syn_rows}r x {args.syn_cols}c x {args.syn_th}x{args.syn_tw}px) ..."
        )
        mosaic = _synthetic_mosaic(args.syn_z, args.syn_rows, args.syn_cols, args.syn_th, args.syn_tw)
        data_source = "synthetic"
    else:
        path = args.input
        if path is None:
            print(
                "No zarr file specified. Pass --input PATH or use --synthetic.",
                file=sys.stderr,
            )
            return 1
        print(f"Loading mosaic from {path} …")
        mosaic = _load_real_mosaic(path)
        data_source = path

    nz, h, w = mosaic.array.shape
    th, tw = mosaic.tile_shape
    print(
        f"  Mosaic: {nz}z x {mosaic.n_rows}r x {mosaic.n_cols}c  "
        f"tile {th}x{tw}  ({nz * mosaic.n_rows * mosaic.n_cols} total solves)"
    )

    # ------------------------------------------------------------------
    # BaSiC kwargs
    # ------------------------------------------------------------------
    basic_kwargs: dict[str, Any] = {
        "backend": args.backend,
        "working_size": args.working_size,
    }
    if args.device is not None:
        basic_kwargs["device"] = args.device

    worker_counts = args.workers or _WORKER_COUNTS
    # Clamp to available CPUs (no point testing more workers than cores)
    max_workers = os.cpu_count() or 4
    worker_counts = sorted({min(nw, max_workers) for nw in worker_counts})

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    metadata: dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu": _cpu_info(),
        "cpu_count": os.cpu_count(),
        "gpu": _gpu_info(),
        "default_workers": default_workers(),
        "data_source": str(data_source),
        "mosaic_shape": [nz, h, w],
        "tile_shape": [th, tw],
        "n_rows": mosaic.n_rows,
        "n_cols": mosaic.n_cols,
        "total_solves": nz * mosaic.n_rows * mosaic.n_cols,
        "backend": args.backend,
        "device": args.device,
        "working_size": args.working_size,
        "repeats": args.repeats,
    }

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    print(f"\nBackend: {args.backend}  device: {args.device or 'default'}  working_size: {args.working_size}")
    print(f"Worker counts: {worker_counts}  repeats: {args.repeats}")
    print()

    timing = run_benchmark(mosaic, basic_kwargs, worker_counts, args.repeats)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    baseline = timing.get("1", {}).get("mean", None)
    print("\n" + "=" * 60)
    print(f"{'workers':>8}  {'mean(s)':>8}  {'min(s)':>7}  {'speedup':>8}")
    print("-" * 60)
    for nw_str, t in timing.items():
        spd = f"{baseline / t['mean']:.2f}x" if baseline else "—"
        print(f"{nw_str:>8}  {t['mean']:>8.2f}  {t['min']:>7.2f}  {spd:>8}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    out = {**metadata, "timing": timing}
    out_path = Path(args.output)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
