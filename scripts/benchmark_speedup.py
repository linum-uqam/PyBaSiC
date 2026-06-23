#!/usr/bin/env python3
"""Compare fit_mosaic wall time: single-GPU vs multi-GPU fan-out.

Designed for CUDA hosts with >= 2 GPUs.  Uses synthetic OCT-like mosaics by
default so no data mount is required.

Example (remote server)::

    uv run python scripts/benchmark_speedup.py \\
        --syn-z 16 --syn-rows 31 --syn-cols 16 --syn-th 75 --syn-tw 75 \\
        --repeats 3 --max-reweighting-iterations 15 --estimate-darkfield
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _synthetic_mosaic(n_z: int, n_rows: int, n_cols: int, th: int, tw: int) -> Any:
    from linum_basic.mosaic import MosaicGrid

    rng = np.random.default_rng(0)
    arr = rng.standard_normal((n_z, n_rows * th, n_cols * tw)).astype(np.float32)
    arr = np.clip(arr * 0.2 + 1.0, 0.01, None)
    return MosaicGrid(array=arr, tile_shape=(th, tw), overlap_fraction=0.2)


def _time_fit(mosaic: Any, basic_kwargs: dict[str, Any]) -> float:
    from linum_basic.fit import fit_mosaic

    t0 = time.perf_counter()
    fit_mosaic(mosaic, basic_kwargs=basic_kwargs, n_workers=1, verbose=False)
    return time.perf_counter() - t0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--syn-z", type=int, default=16)
    p.add_argument("--syn-rows", type=int, default=31)
    p.add_argument("--syn-cols", type=int, default=16)
    p.add_argument("--syn-th", type=int, default=75)
    p.add_argument("--syn-tw", type=int, default=75)
    p.add_argument("--working-size", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-reweighting-iterations", type=int, default=15)
    p.add_argument("--estimate-darkfield", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output", default="/tmp/benchmark_speedup.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        import torch
    except ImportError:
        print("PyTorch required", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1

    from linum_basic._parallel import list_cuda_devices

    devices = list_cuda_devices("cuda")
    print(f"Host: {platform.node()}  GPUs: {devices}")

    mosaic = _synthetic_mosaic(args.syn_z, args.syn_rows, args.syn_cols, args.syn_th, args.syn_tw)
    print(
        f"Mosaic: {mosaic.n_z}z x {mosaic.n_rows}x{mosaic.n_cols} tiles ({mosaic.n_tiles}/z)  tile {args.syn_th}x{args.syn_tw}"
    )

    single_kwargs: dict[str, Any] = {
        "backend": "torch",
        "device": "cuda:0",
        "working_size": args.working_size,
        "estimate_darkfield": args.estimate_darkfield,
        "max_reweighting_iterations": args.max_reweighting_iterations,
        "warm_start_reweighting": False,
    }
    multi_kwargs: dict[str, Any] = {
        **single_kwargs,
        "device": "cuda",
        "warm_start_reweighting": True,
    }

    single_times: list[float] = []
    multi_times: list[float] = []
    for rep in range(args.repeats):
        t_single = _time_fit(mosaic, single_kwargs)
        single_times.append(t_single)
        print(f"  single-GPU rep {rep + 1}/{args.repeats}: {t_single:.2f}s")
    for rep in range(args.repeats):
        t_multi = _time_fit(mosaic, multi_kwargs)
        multi_times.append(t_multi)
        print(f"  multi-GPU  rep {rep + 1}/{args.repeats}: {t_multi:.2f}s")

    single_mean = float(np.mean(single_times))
    multi_mean = float(np.mean(multi_times))
    speedup = single_mean / multi_mean if multi_mean > 0 else float("nan")

    print("\n" + "=" * 50)
    print(f"single-GPU (cuda:0):  {single_mean:.2f}s mean")
    print(f"multi-GPU  (cuda):    {multi_mean:.2f}s mean")
    print(f"speedup:              {speedup:.2f}x")
    print("=" * 50)

    out = {
        "host": platform.node(),
        "gpus": devices,
        "single_gpu_mean_s": single_mean,
        "multi_gpu_mean_s": multi_mean,
        "speedup": speedup,
        "single_times": single_times,
        "multi_times": multi_times,
        "kwargs": {"single": single_kwargs, "multi": multi_kwargs},
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
