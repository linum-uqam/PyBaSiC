#!/usr/bin/env python3
"""Measure OS-level peak RSS (and GPU VRAM) for one ``fit_mosaic`` invocation.

This is the operator-run probe for the M007/S01 production-scale memory
measurement. It runs a *single* ``fit_mosaic`` call in one fresh process and
records the real OS-level peak resident set size
(``resource.getrusage(resource.RUSAGE_SELF).ru_maxrss``) plus, when a CUDA
device is selected, the GPU peak VRAM via the existing
:func:`linum_basic.benchmark.telemetry.collect_memory_stats` helper.

Why a separate process per mode
-------------------------------
Peak RSS is a *monotonic per-process high-water mark*. Eager and streaming
must therefore be measured in independent OS processes (one invocation each)
so the eager peak cannot contaminate the streaming measurement. This is why
the script takes a single ``--mode`` and T02/T03 each invoke it once.

Why ``resource.getrusage`` rather than ``tracemalloc``
------------------------------------------------------
``tracemalloc`` only sees Python-object allocations inside one interpreter.
It cannot see native buffers (NumPy/BLAS scratch, PyTorch CUDA host-pinned
memory, OpenCV internals). ``ru_maxrss`` is the kernel's accounting of the
whole process, which is what an operator watching ``top``/``htop`` actually
cares about at production scale.

Modes
-----
eager
    Load the full volume into memory (``lazy=False``) and fit without
    streaming (``streaming=False``): every per-z tile stack is extracted up
    front and held live simultaneously. This is the worst-case host-memory
    path and the R051 baseline.
streaming
    Open the OME-Zarr lazily (``lazy=True``) and fit one z-level at a time
    (``streaming=True``): peak host memory is bounded by a single plane
    instead of the whole z-stack. This is the MVP streaming path.

Example (eager baseline on the A6000)::

    uv run python scripts/streaming_memory_probe.py \\
        --input subject.ome.zarr \\
        --mode eager \\
        --output scripts/experiments/s01_artifacts/eager-probe.json \\
        --z-sample 5 --working-size 128 \\
        --backend torch --device cuda:0 --estimate-darkfield --yes

Example (streaming, same z-selection/working_size, separate process)::

    uv run python scripts/streaming_memory_probe.py \\
        --input subject.ome.zarr \\
        --mode streaming \\
        --output scripts/experiments/s01_artifacts/streaming-probe.json \\
        --z-sample 5 --working-size 128 \\
        --backend torch --device cuda:0 --estimate-darkfield --yes
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Reused (not duplicated) provenance + telemetry helpers from the benchmark
# package. ``collect_memory_stats`` reads peak CUDA allocator counters; the
# provenance helpers stamp the artifact with git commit and host info.
from linum_basic.benchmark import (
    collect_git_commit,
    collect_host_info,
    collect_memory_stats,
)
from scripts.benchmark_speedup import (
    LARGE_RUN_Z_THRESHOLD,
    enforce_large_run_guard,
    resolve_z_selection,
)

SCHEMA_VERSION = 1
PROBE_MODES = ("eager", "streaming")


def _peak_rss_bytes() -> int:
    """Return peak resident set size of this process in bytes.

    ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` is the monotonic per-process
    high-water mark. Its unit is platform dependent: kilobytes on Linux and
    bytes on macOS/BSD, so it is normalised to bytes here.
    """
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Linux":
        # Linux reports ru_maxrss in kilobytes.
        return int(ru_maxrss) * 1024
    # macOS / BSD report ru_maxrss in bytes already.
    return int(ru_maxrss)


def _reset_cuda_peak_stats(device: str | None) -> None:
    """Reset CUDA peak-memory counters so the fit region is measured cleanly.

    ``collect_memory_stats`` reads the allocator's running peak, so the peak
    must be reset immediately before the fit and read immediately after.
    No-op when no CUDA device is selected or torch is unavailable.
    """
    if not device or not device.lower().startswith("cuda"):
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    index = 0
    if device.startswith("cuda:"):
        index = int(device.split(":", 1)[1])
    try:
        torch.cuda.set_device(index)
    except RuntimeError, ValueError:
        return
    torch.cuda.reset_peak_memory_stats(device)


def _build_basic_kwargs(
    *,
    working_size: int,
    estimate_darkfield: bool,
    max_reweighting_iterations: int,
    backend: str | None,
    device: str | None,
) -> dict[str, Any]:
    """Assemble the ``basic_kwargs`` dict forwarded to ``fit_mosaic``."""
    kwargs: dict[str, Any] = {
        "working_size": working_size,
        "estimate_darkfield": estimate_darkfield,
        "max_reweighting_iterations": max_reweighting_iterations,
    }
    if backend is not None:
        kwargs["backend"] = backend
    if device is not None:
        kwargs["device"] = device
    return kwargs


def run_probe(
    input_path: str | Path,
    *,
    mode: str,
    z_indices: str | None = None,
    z_sample: int | None = None,
    working_size: int = 128,
    backend: str | None = None,
    device: str | None = None,
    estimate_darkfield: bool = False,
    max_reweighting_iterations: int = 15,
    overlap: float = 0.2,
    allow_large_run: bool = False,
    yes: bool = False,
) -> dict[str, Any]:
    """Run one ``fit_mosaic`` invocation in the requested *mode* and measure it.

    This is the testable core (no argparse, no file writing). It performs the
    fit, reads peak RSS / VRAM, and returns the JSON-serialisable artifact
    dict. Callers (``main`` and the contract test) are responsible for writing
    it to disk if desired.

    Parameters
    ----------
    input_path : str or Path
        Path to the OME-Zarr subject volume.
    mode : {"eager", "streaming"}
        ``eager`` loads the full volume and fits without streaming (worst-case
        host memory); ``streaming`` opens lazily and fits one z at a time
        (bounded peak memory).
    z_indices, z_sample :
        Forwarded to :func:`scripts.benchmark_speedup.resolve_z_selection`.
        Exactly one must be provided so the measurement scope is explicit.
    working_size : int
        BaSiC working resolution (production default 128).
    backend, device :
        Optional BaSiC backend/device forwarded into ``basic_kwargs``.
        Selecting a ``cuda`` device enables GPU VRAM measurement.
    estimate_darkfield, max_reweighting_iterations, overlap :
        BaSiC / mosaic fit knobs, recorded in the artifact for reproducibility.
    allow_large_run, yes :
        Forwarded to :func:`scripts.benchmark_speedup.enforce_large_run_guard`.

    Returns
    -------
    dict
        JSON-serialisable artifact with the schema documented in the module
        docstring and the S01 slice verification contract.

    Raises
    ------
    SystemExit
        When the z selection exceeds the large-run guard without confirmation.
    ValueError
        When *mode* is unknown or the z selection is invalid (propagated from
        ``resolve_z_selection``).
    FileNotFoundError
        When *input_path* does not exist.
    """
    if mode not in PROBE_MODES:
        msg = f"Unknown mode {mode!r}; choose from {PROBE_MODES}"
        raise ValueError(msg)

    path = Path(input_path)
    if not path.exists():
        msg = f"Input not found: {path}"
        raise FileNotFoundError(msg)

    streaming = mode == "streaming"
    lazy = streaming  # streaming pairs with a lazy backing volume.

    # Lazy import keeps ``--help`` and argparse fast and avoids an import-time
    # CUDA context on machines without a GPU.
    from linum_basic.fit import fit_mosaic
    from linum_basic.mosaic import MosaicGrid

    # Load eagerly to discover n_z for z-selection resolution, regardless of the
    # mode. This metadata read is cheap (OME-NGFF attrs + zarr chunk grid) and
    # does not materialise tile pixels when lazy-loading later.
    probe_grid = MosaicGrid.from_ome_zarr(str(path), overlap_fraction=overlap, lazy=True)
    n_z = probe_grid.n_z
    resolved_z = resolve_z_selection(n_z, z_indices=z_indices, z_sample=z_sample)
    enforce_large_run_guard(len(resolved_z), allow_large_run=allow_large_run, yes=yes)

    basic_kwargs = _build_basic_kwargs(
        working_size=working_size,
        estimate_darkfield=estimate_darkfield,
        max_reweighting_iterations=max_reweighting_iterations,
        backend=backend,
        device=device,
    )

    # --- measurement region -------------------------------------------------
    # Reset CUDA peak counters *after* context init but *before* the fit so the
    # measured peak isolates the fit_mosaic call (the eager volume load below is
    # part of the eager path's characteristic cost and must stay in scope).
    _reset_cuda_peak_stats(device)

    t0 = time.perf_counter()
    mosaic = MosaicGrid.from_ome_zarr(str(path), overlap_fraction=overlap, lazy=lazy)
    fit = fit_mosaic(
        mosaic,
        z_indices=resolved_z,
        streaming=streaming,
        strategy="sequential",
        basic_kwargs=basic_kwargs,
        verbose=False,
    )
    wall_s = time.perf_counter() - t0
    # Peak RSS is monotonic for the whole process; reading after the fit captures
    # the high-water mark including the eager volume load and all native scratch.
    peak_rss_bytes = _peak_rss_bytes()
    gpu_memory = collect_memory_stats(device)
    peak_vram_bytes = gpu_memory.max_memory_allocated_bytes if gpu_memory is not None else None
    # --- end measurement region --------------------------------------------

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "streaming": streaming,
        "lazy_load": lazy,
        "strategy": "sequential",
        "peak_rss_bytes": peak_rss_bytes,
        "peak_vram_bytes": peak_vram_bytes,
        "gpu_memory": (
            {
                "max_memory_allocated_bytes": gpu_memory.max_memory_allocated_bytes,
                "max_memory_reserved_bytes": gpu_memory.max_memory_reserved_bytes,
            }
            if gpu_memory is not None
            else None
        ),
        "wall_ms": round(wall_s * 1000.0, 3),
        "input_path": str(path),
        "z_indices": list(resolved_z),
        "n_z": n_z,
        "working_size": working_size,
        "backend": backend,
        "device": device,
        "estimate_darkfield": estimate_darkfield,
        "max_reweighting_iterations": max_reweighting_iterations,
        "overlap": overlap,
        "flatfields_shape": list(fit.flatfields.shape),
        "darkfields_shape": list(fit.darkfields.shape),
        "git_commit": collect_git_commit(),
        "host": collect_host_info(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return artifact


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure peak RSS (and GPU VRAM) for one eager vs streaming "
            "fit_mosaic invocation. Run once per mode as a fresh process."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="OME-Zarr subject path.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=PROBE_MODES,
        help="eager = full volume + non-streaming fit (baseline); streaming = lazy volume + one-z-at-a-time fit.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the JSON artifact (parent dirs created).",
    )
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument("--z-indices", default=None, help='Comma-separated z indices (e.g. "0,5,10").')
    select.add_argument("--z-sample", type=int, default=None, help="Evenly spaced z sample count.")
    parser.add_argument("--working-size", type=int, default=128, help="BaSiC working resolution (default 128).")
    parser.add_argument("--backend", default=None, help="BaSiC backend (auto|numpy|torch).")
    parser.add_argument(
        "--device",
        default=None,
        help="BaSiC device (e.g. cuda:0). A cuda device enables VRAM measurement.",
    )
    parser.add_argument(
        "--estimate-darkfield",
        action="store_true",
        help="Estimate a dark-field in addition to the flat-field.",
    )
    parser.add_argument(
        "--max-reweighting-iterations",
        type=int,
        default=15,
        help="Outer reweighting cap forwarded to BaSiC.",
    )
    parser.add_argument("--overlap", type=float, default=0.2, help="Mosaic tile overlap fraction.")
    parser.add_argument(
        "--allow-large-run",
        action="store_true",
        help=f"Allow >{LARGE_RUN_Z_THRESHOLD} selected z-planes without confirm.",
    )
    parser.add_argument("--yes", action="store_true", help="Alias for --allow-large-run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    try:
        artifact = run_probe(
            input_path,
            mode=args.mode,
            z_indices=args.z_indices,
            z_sample=args.z_sample,
            working_size=args.working_size,
            backend=args.backend,
            device=args.device,
            estimate_darkfield=bool(args.estimate_darkfield),
            max_reweighting_iterations=args.max_reweighting_iterations,
            overlap=args.overlap,
            allow_large_run=bool(args.allow_large_run),
            yes=bool(args.yes),
        )
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(
        f"mode={artifact['mode']} peak_rss_bytes={artifact['peak_rss_bytes']:,} "
        f"peak_vram_bytes={artifact['peak_vram_bytes']} wall_ms={artifact['wall_ms']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
