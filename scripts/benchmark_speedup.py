#!/usr/bin/env python3
"""Compare fit_mosaic wall time across CUDA execution strategies.

Modes
-----
sequential
    Single GPU, one z-level at a time (legacy path).
multi
    Multi-GPU process fan-out across z-levels.
batched
    Single batched CUDA solve over all z-planes (default when CUDA available).
all
    Run every mode and print a comparison table.

Example (production-like params on the server)::

    uv run python scripts/benchmark_speedup.py \\
        --syn-z 55 --syn-rows 13 --syn-cols 11 --syn-th 88 --syn-tw 88 \\
        --max-reweighting-iterations 500 --estimate-darkfield \\
        --mode all --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np


def _synthetic_mosaic(n_z: int, n_rows: int, n_cols: int, th: int, tw: int) -> Any:
    from linum_basic.mosaic import MosaicGrid

    rng = np.random.default_rng(0)
    arr = rng.standard_normal((n_z, n_rows * th, n_cols * tw)).astype(np.float32)
    arr = np.clip(arr * 0.2 + 1.0, 0.01, None)
    return MosaicGrid(array=arr, tile_shape=(th, tw), overlap_fraction=0.2)


def _time_fit(mosaic: Any, basic_kwargs: dict[str, Any], *, force_batched: bool | None) -> float:
    import linum_basic.fit as fit_mod

    original = fit_mod.should_use_batched_cuda

    def _selector(*, field_mode: str, n_z: int, backend: str | None, device: str | None) -> bool:
        if force_batched is True:
            return True
        if force_batched is False:
            return False
        return original(field_mode=field_mode, n_z=n_z, backend=backend, device=device)

    fit_mod.should_use_batched_cuda = cast(Any, _selector)
    try:
        from linum_basic.fit import fit_mosaic

        t0 = time.perf_counter()
        fit_mosaic(mosaic, basic_kwargs=basic_kwargs, n_workers=1, verbose=False)
        return time.perf_counter() - t0
    finally:
        fit_mod.should_use_batched_cuda = cast(Any, original)


def _run_mode(
    mode: str,
    mosaic: Any,
    *,
    working_size: int,
    max_reweighting_iterations: int,
    estimate_darkfield: bool,
    repeats: int,
    batched_z_chunk_size: int | None,
) -> dict[str, Any]:
    if mode == "sequential":
        kwargs: dict[str, Any] = {
            "backend": "torch",
            "device": "cuda:0",
            "working_size": working_size,
            "estimate_darkfield": estimate_darkfield,
            "max_reweighting_iterations": max_reweighting_iterations,
            "warm_start_reweighting": False,
        }
        force_batched = False
    elif mode == "multi":
        kwargs = {
            "backend": "torch",
            "device": "cuda",
            "working_size": working_size,
            "estimate_darkfield": estimate_darkfield,
            "max_reweighting_iterations": max_reweighting_iterations,
            "warm_start_reweighting": True,
        }
        force_batched = False
    elif mode == "batched":
        kwargs = {
            "backend": "torch",
            "device": "cuda",
            "working_size": working_size,
            "estimate_darkfield": estimate_darkfield,
            "max_reweighting_iterations": max_reweighting_iterations,
            "warm_start_reweighting": True,
        }
        if batched_z_chunk_size is not None:
            kwargs["batched_z_chunk_size"] = batched_z_chunk_size
        force_batched = True
    else:
        msg = f"Unknown mode: {mode}"
        raise ValueError(msg)

    times: list[float] = []
    for rep in range(repeats):
        elapsed = _time_fit(mosaic, kwargs, force_batched=force_batched)
        times.append(elapsed)
        print(f"  {mode:10s} rep {rep + 1}/{repeats}: {elapsed:.2f}s")
    mean = float(np.mean(times))
    return {"mode": mode, "mean_s": mean, "times_s": times, "kwargs": kwargs}


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
    p.add_argument(
        "--batched-z-chunk-size",
        type=int,
        default=None,
        help="Optional z chunk size for batched CUDA mode; <=0 disables chunking.",
    )
    p.add_argument(
        "--mode",
        choices=("sequential", "multi", "batched", "all"),
        default="all",
        help="Execution strategy to benchmark. [%(default)s]",
    )
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

    if os.environ.get("LINUM_BASIC_TORCH_CACHE", "1") != "0":
        from linum_basic._torch_cache import configure_torch_inductor_cache, enable_fx_graph_cache

        cache_dir = configure_torch_inductor_cache()
        enable_fx_graph_cache()
        print(f"Inductor cache: {cache_dir}")

    from linum_basic._parallel import list_cuda_devices

    devices = list_cuda_devices("cuda")
    print(f"Host: {platform.node()}  GPUs: {devices}")

    mosaic = _synthetic_mosaic(args.syn_z, args.syn_rows, args.syn_cols, args.syn_th, args.syn_tw)
    print(
        f"Mosaic: {mosaic.n_z}z x {mosaic.n_rows}x{mosaic.n_cols} tiles ({mosaic.n_tiles}/z)  tile {args.syn_th}x{args.syn_tw}"
    )

    modes = ["sequential", "multi", "batched"] if args.mode == "all" else [args.mode]
    results: list[dict[str, Any]] = []
    for mode in modes:
        if mode == "multi" and len(devices) < 2:
            print(f"  {mode:10s} skipped (needs >= 2 GPUs)")
            continue
        results.append(
            _run_mode(
                mode,
                mosaic,
                working_size=args.working_size,
                max_reweighting_iterations=args.max_reweighting_iterations,
                estimate_darkfield=args.estimate_darkfield,
                repeats=args.repeats,
                batched_z_chunk_size=args.batched_z_chunk_size,
            )
        )

    print("\n" + "=" * 60)
    baseline = results[0]["mean_s"] if results else float("nan")
    for row in results:
        speedup = baseline / row["mean_s"] if row["mean_s"] > 0 else float("nan")
        print(f"{row['mode']:10s}  {row['mean_s']:8.2f}s mean  ({speedup:.2f}x vs first mode)")
    print("=" * 60)
    if len(devices) >= 2:
        print(
            "Concurrency note: with batched mode one slice may saturate a single GPU; "
            "benchmark maxForks=2 (1 GPU/slice) vs maxForks=1 (batched) on full subjects."
        )

    out = {
        "host": platform.node(),
        "gpus": devices,
        "results": results,
        "mosaic": {
            "n_z": mosaic.n_z,
            "n_rows": mosaic.n_rows,
            "n_cols": mosaic.n_cols,
            "n_tiles": mosaic.n_tiles,
            "tile_shape": [args.syn_th, args.syn_tw],
        },
        "working_size": args.working_size,
        "max_reweighting_iterations": args.max_reweighting_iterations,
        "batched_z_chunk_size": args.batched_z_chunk_size,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
