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

Harness subcommands
-------------------
baseline
    Run a production-shaped baseline fit, calibrate tolerances, and write a
    versioned artifact bundle (walking skeleton for the A/B harness).
candidate
    Run a candidate strategy against a saved baseline bundle and emit speed,
    quality, and overall promote/reject verdicts.
compare
    Recompute promote/reject summary from saved baseline and candidate artifacts.
concurrency
    Canonical Phase 6 multi vs batched A/B aggregator; legacy ``--mode`` parser
    is retained for reference without deprecation warnings.

Example (production-like params on the server)::

    uv run python scripts/benchmark_speedup.py \\
        --syn-z 55 --syn-rows 13 --syn-cols 11 --syn-th 88 --syn-tw 88 \\
        --max-reweighting-iterations 500 --estimate-darkfield \\
        --mode all --repeats 3

Example (synthetic-smoke baseline bundle)::

    uv run python scripts/benchmark_speedup.py baseline \\
        --input subject.ome.zarr --subject-id sub-22 --output-dir ./runs \\
        --z-sample 5 --strategy baseline --synthetic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from linum_basic._torch_cache import warm_policy_passes
from linum_basic.benchmark.artifacts import (
    BaselineBundle,
    CandidateArtifact,
    ToleranceSidecar,
    assert_comparable,
    make_baseline_id,
    read_artifact,
    slugify_label,
    write_artifact,
    write_summary_table,
)
from linum_basic.benchmark.metadata import (
    RunMetadata,
    collect_concurrency_metadata,
    collect_git_commit,
    collect_run_metadata,
)
from linum_basic.benchmark.profile import (
    BottleneckReport,
    LeverAttemptTable,
    RankedLever,
    build_bottleneck_report,
    build_lever_attempt_table,
    build_phase3_handoff_config,
    build_phase5_backlog,
    build_phase5_fast_path,
    build_phase6_concurrency_verdict,
)
from linum_basic.benchmark.quality import (
    CALIBRATION_POLICY,
    METRIC_DEFINITION_VERSION,
    PerZMetricRow,
    QualityReport,
    QualityVerdict,
    ToleranceSpec,
    calibrate_tolerances,
    compute_deltas,
    compute_quality_report,
    evaluate_quality_gate,
)
from linum_basic.benchmark.strategies import StrategyResult, load_overrides, resolve_strategy
from linum_basic.benchmark.sweep import SPEED_RATIO_THRESHOLD, build_sweep_table
from linum_basic.benchmark.telemetry import (
    TelemetryRecord,
    collect_memory_stats,
    collect_multi_gpu_memory_stats,
    collect_precision_metadata,
    peak_single_gpu_vram_bytes,
    run_with_phases,
)
from linum_basic.fit import MosaicFit, fit_mosaic

LARGE_RUN_Z_THRESHOLD = 8

_HARNESS_SUBCOMMANDS = frozenset({"baseline", "candidate", "compare", "concurrency", "profile", "sweep", "optimize"})


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _resolve_compile_status(strategy: StrategyResult, run_meta: RunMetadata) -> str:
    """Return a CPU-safe torch.compile status label for artifact telemetry."""
    backend = strategy.basic_kwargs.get("backend", "torch")
    if backend == "numpy":
        return "disabled"
    if not run_meta.cuda_available:
        return "unavailable"
    return "enabled"


def _telemetry_metadata(telemetry: TelemetryRecord, run_meta: RunMetadata) -> dict[str, Any]:
    """Serialize timing, memory, and compile/cache fields for artifact metadata."""
    return {
        "warmup_ms": telemetry.warmup_ms,
        "cold_cache_ms": telemetry.cold_cache_ms,
        "warm_cache_ms": telemetry.warm_cache_ms,
        "steady_state_ms": telemetry.steady_state_ms,
        "max_memory_allocated_bytes": telemetry.max_memory_allocated_bytes,
        "max_memory_reserved_bytes": telemetry.max_memory_reserved_bytes,
        "chunk_size": telemetry.chunk_size,
        "compile_status": telemetry.compile_status,
        "inductor_cache_path": telemetry.inductor_cache_path,
        "fx_graph_cache_enabled": run_meta.fx_graph_cache_enabled,
    }


def _precision_metadata() -> dict[str, Any]:
    """Capture TF32 and compile-fallback precision flags for artifact metadata."""
    from linum_basic._alm import last_compile_fallback

    return collect_precision_metadata(compile_fallback_reason=last_compile_fallback)


def _operator_timing_metadata(
    telemetry: TelemetryRecord,
    *,
    n_z: int,
    n_tiles: int,
) -> dict[str, Any]:
    """Serialize operator-facing per-z fit wall-clock derived from steady_state_ms."""
    end_to_end_ms = telemetry.steady_state_ms
    if n_z > 0 and end_to_end_ms is not None:
        per_z_ms: float | None = end_to_end_ms / float(n_z)
    else:
        per_z_ms = None
    return {
        "end_to_end_ms": end_to_end_ms,
        "per_z_ms": per_z_ms,
        "n_z": n_z,
        "n_tiles": n_tiles,
    }


def _convergence_metadata(
    fit: MosaicFit,
    z_indices: list[int],
    *,
    max_reweighting_iterations: int,
) -> dict[str, Any]:
    """Summarise per-z reweighting telemetry for candidate artifact metadata."""
    per_z = fit.convergence_per_z
    if per_z is None:
        return {
            "reweight_iterations_per_z": None,
            "reweight_iterations_median": None,
            "max_reweighting_iterations": max_reweighting_iterations,
        }

    reweight_iterations_per_z = {
        str(z): int(entry["reweighting_iteration"]) for z, entry in zip(z_indices, per_z, strict=True)
    }
    iterations = [int(entry["reweighting_iteration"]) for entry in per_z]
    block: dict[str, Any] = {
        "reweight_iterations_per_z": reweight_iterations_per_z,
        "reweight_iterations_median": float(statistics.median(iterations)) if iterations else None,
        "max_reweighting_iterations": max_reweighting_iterations,
    }
    l_s_vals = [float(entry["l_s"]) for entry in per_z if entry.get("l_s") is not None]
    l_d_vals = [float(entry["l_d"]) for entry in per_z if entry.get("l_d") is not None]
    if l_s_vals:
        block["l_s"] = float(statistics.median(l_s_vals))
    if l_d_vals:
        block["l_d"] = float(statistics.median(l_d_vals))
    return block


def resolve_z_selection(
    n_z: int,
    *,
    z_indices: str | None = None,
    z_sample: int | None = None,
) -> list[int]:
    """Resolve explicit z-plane indices for a real-subject benchmark run.

    Parameters
    ----------
    n_z : int
        Total number of z-planes in the subject volume.
    z_indices : str or None
        Comma-separated explicit indices (for example ``"0,5,10"``).
    z_sample : int or None
        Evenly spaced sample count across depth.

    Returns
    -------
    list of int
        Sorted unique z indices in ``[0, n_z)``.

    Raises
    ------
    ValueError
        When neither *z_indices* nor *z_sample* is provided, or indices are invalid.
    """
    if z_indices is not None and z_sample is not None:
        msg = "Specify only one of --z-indices or --z-sample"
        raise ValueError(msg)

    if z_indices is not None:
        raw = [part.strip() for part in z_indices.split(",") if part.strip()]
        if not raw:
            msg = "Provide at least one z index via --z-indices"
            raise ValueError(msg)
        indices = sorted({int(part) for part in raw})
    elif z_sample is not None:
        if z_sample < 1:
            msg = f"--z-sample must be >= 1, got {z_sample}"
            raise ValueError(msg)
        if n_z < 1:
            msg = f"Subject has no z-planes (n_z={n_z})"
            raise ValueError(msg)
        count = min(z_sample, n_z)
        indices = np.linspace(0, n_z - 1, num=count, dtype=int).tolist()
        indices = sorted(set(indices))
    else:
        msg = "Real runs require explicit z selection via --z-indices or --z-sample"
        raise ValueError(msg)

    for z in indices:
        if z < 0 or z >= n_z:
            msg = f"z index {z} out of range for n_z={n_z}"
            raise ValueError(msg)
    return indices


def enforce_large_run_guard(
    n_selected: int,
    *,
    allow_large_run: bool,
    yes: bool,
) -> None:
    """Refuse large z selections unless the operator explicitly confirms.

    Raises
      ------
      SystemExit
          When the selection exceeds :data:`LARGE_RUN_Z_THRESHOLD` without confirmation.
    """
    if n_selected <= LARGE_RUN_Z_THRESHOLD:
        return
    if allow_large_run or yes:
        return
    print(
        f"Selected {n_selected} z-planes exceeds threshold {LARGE_RUN_Z_THRESHOLD}. "
        "Pass --allow-large-run or --yes to continue.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _add_shared_run_group(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--strategy",
        choices=("baseline", "sequential", "multi", "batched"),
        default="baseline",
        help="Built-in fit strategy name.",
    )
    parser.add_argument("--config", default=None, help="Optional JSON/YAML override file.")
    parser.add_argument("--working-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3, help="Measured repeats after warmup.")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warmup iterations.")
    parser.add_argument("--max-reweighting-iterations", type=int, default=15)
    parser.add_argument("--estimate-darkfield", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--batched-z-chunk-size",
        type=int,
        default=None,
        help="Optional z chunk size for batched CUDA mode; <=0 disables chunking.",
    )


def _add_real_input_group(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="OME-Zarr subject path.")
    parser.add_argument("--subject-id", default=None, help="Human-readable subject identifier.")
    parser.add_argument("--run-label", default=None, help="Operator run label.")
    parser.add_argument("--output-dir", default=None, help="Directory for harness artifacts.")
    parser.add_argument("--z-indices", default=None, help='Comma-separated z indices (e.g. "0,5,10").')
    parser.add_argument("--z-sample", type=int, default=None, help="Evenly spaced z sample count.")
    parser.add_argument(
        "--allow-large-run",
        action="store_true",
        help=f"Allow >{LARGE_RUN_Z_THRESHOLD} selected z-planes without interactive confirm.",
    )
    parser.add_argument("--yes", action="store_true", help="Alias for --allow-large-run.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Smoke-only run; never eligible as a release gate (QUAL-04).",
    )


def _build_subcommand_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="linum-basic A/B benchmark harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser(
        "baseline",
        help="Run production-shaped baseline calibration and write artifact bundle.",
    )
    _add_shared_run_group(baseline)
    _add_real_input_group(baseline)
    baseline.set_defaults(handler=cmd_baseline)

    candidate = subparsers.add_parser(
        "candidate",
        help="Run a candidate fit against a saved baseline and write verdict artifact.",
    )
    _add_shared_run_group(candidate)
    _add_real_input_group(candidate)
    candidate.add_argument(
        "--baseline-id",
        required=True,
        help="Saved baseline bundle id under --output-dir.",
    )
    candidate.set_defaults(handler=cmd_candidate)

    compare = subparsers.add_parser(
        "compare",
        help="Recompute promote/reject summary from saved baseline and candidate JSON.",
    )
    compare.add_argument("--baseline", required=True, help="Path to baseline-bundle.json.")
    compare.add_argument("--candidate", required=True, help="Path to candidate-artifact.json.")
    compare.add_argument("--output-dir", required=True, help="Directory for comparison summary.")
    compare.set_defaults(handler=cmd_compare)

    sweep = subparsers.add_parser(
        "sweep",
        help="Aggregate saved baseline and candidate artifacts into a speed-quality sweep table.",
    )
    sweep.add_argument("--baseline", required=True, help="Path to baseline-bundle.json.")
    sweep.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Path to candidate-artifact.json (repeatable).",
    )
    sweep.add_argument("--output-dir", required=True, help="Directory for sweep outputs.")
    sweep.add_argument(
        "--speed-ratio-threshold",
        type=float,
        default=SPEED_RATIO_THRESHOLD,
        help=f"Minimum speed ratio for adoption (default {SPEED_RATIO_THRESHOLD}).",
    )
    sweep.set_defaults(handler=cmd_sweep)

    profile = subparsers.add_parser(
        "profile",
        help="Profile sequential ws=128 fit and write bottleneck-report.json.",
    )
    _add_shared_run_group(profile)
    _add_real_input_group(profile)
    profile.add_argument(
        "--mode",
        choices=("sequential", "batched-diagnostic"),
        default="sequential",
        help="Profiling mode: sequential ws=128 baseline or batched-diagnostic Phase 6 handoff.",
    )
    profile.add_argument(
        "--require-sequential-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require bottleneck-report.json before batched-diagnostic (D-04).",
    )
    profile.add_argument(
        "--baseline-id",
        default=None,
        help="Saved baseline bundle id for z_indices inheritance (under --output-dir).",
    )
    profile.set_defaults(max_reweighting_iterations=500, handler=cmd_profile)

    optimize = subparsers.add_parser(
        "optimize",
        help="Aggregate lever attempt artifacts into handoff and backlog JSON (D-10).",
    )
    optimize.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing bottleneck-report.json and baseline bundle.",
    )
    optimize.add_argument(
        "--baseline-id",
        required=True,
        help="Saved baseline bundle id under --output-dir.",
    )
    optimize.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Path to candidate-artifact.json (repeatable, ordered by ranked_levers).",
    )
    optimize.set_defaults(handler=cmd_optimize)

    concurrency = subparsers.add_parser(
        "concurrency",
        help="Aggregate multi vs batched candidate artifacts into phase6-concurrency-verdict.json (canonical).",
    )
    concurrency.add_argument("--output-dir", required=True, help="Directory for phase6-concurrency-verdict.json.")
    concurrency.add_argument(
        "--baseline-id",
        required=True,
        help="Saved baseline bundle id referenced by the candidate artifacts.",
    )
    concurrency.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Path to candidate-artifact.json (repeatable; exactly two: multi and batched arms).",
    )
    concurrency.add_argument(
        "--fast-path-ref",
        default="",
        help="Path or commit of phase5-fast-path.json frozen for this A/B (D-13).",
    )
    concurrency.set_defaults(handler=cmd_concurrency)

    return parser


def _input_fingerprint(path: str, shape: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).resolve().as_posix().encode())
    digest.update(repr(shape).encode())
    return f"sha256:{digest.hexdigest()[:16]}"


def _validate_baseline_args(args: argparse.Namespace) -> str | None:
    """Return an error message when required real-run args are missing."""
    if not args.synthetic:
        if not args.output_dir:
            return "Real runs require --output-dir"
        if not args.subject_id and not args.run_label:
            return "Real runs require --subject-id or --run-label"
    else:
        if not args.output_dir:
            return "Baseline runs require --output-dir"
        if not args.subject_id and not args.run_label:
            return "Baseline runs require --subject-id or --run-label"
    return None


def _apply_strategy_batched_patch(strategy_force_batched: bool) -> Any:
    import linum_basic.fit as fit_mod

    original = fit_mod.should_use_batched_cuda

    def _selector(*, field_mode: str, n_z: int, backend: str | None, device: str | None) -> bool:
        if strategy_force_batched:
            return True
        return original(field_mode=field_mode, n_z=n_z, backend=backend, device=device)

    fit_mod.should_use_batched_cuda = cast(Any, _selector)
    return original


def cmd_baseline(args: argparse.Namespace) -> int:
    """Run baseline calibration and write a versioned artifact bundle."""
    err = _validate_baseline_args(args)
    if err:
        print(err, file=sys.stderr)
        return 1

    if not args.input:
        print("Real runs require --input", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    try:
        from linum_basic.mosaic import MosaicGrid

        mosaic = MosaicGrid.from_ome_zarr(str(input_path))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Failed to load OME-Zarr: {exc}", file=sys.stderr)
        return 1

    try:
        z_indices = resolve_z_selection(
            mosaic.n_z,
            z_indices=args.z_indices,
            z_sample=args.z_sample,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        enforce_large_run_guard(
            len(z_indices),
            allow_large_run=bool(args.allow_large_run),
            yes=bool(args.yes),
        )
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1

    overrides: dict[str, Any] = {}
    if args.config:
        try:
            overrides = load_overrides(args.config)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1

    strategy = resolve_strategy(
        args.strategy,
        working_size=args.working_size,
        estimate_darkfield=args.estimate_darkfield,
        max_reweighting_iterations=args.max_reweighting_iterations,
        batched_z_chunk_size=args.batched_z_chunk_size,
        overrides=overrides,
        is_synthetic=bool(args.synthetic),
    )

    subject_id = args.subject_id or args.run_label or "unknown"
    run_label = args.run_label or args.subject_id or "unknown"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_batched = _apply_strategy_batched_patch(strategy.force_batched)
    captured_fits: list[MosaicFit] = []

    run_meta = collect_run_metadata(
        strategy_params=dict(strategy.basic_kwargs),
        z_indices=z_indices,
        cache_mode="persistent" if os.environ.get("LINUM_BASIC_TORCH_CACHE", "1") != "0" else "disabled",
        configure_cache=False,
    )
    compile_status = _resolve_compile_status(strategy, run_meta)

    def _fit_once() -> MosaicFit:
        fit = fit_mosaic(
            mosaic,
            z_indices=z_indices,
            basic_kwargs=strategy.basic_kwargs,
            n_workers=1,
            verbose=False,
        )
        captured_fits.append(fit)
        return fit

    inductor_warm_passes = warm_policy_passes()
    for _ in range(inductor_warm_passes):
        _fit_once()

    try:
        telemetry = run_with_phases(
            _fit_once,
            device=strategy.basic_kwargs.get("device"),
            backend=strategy.basic_kwargs.get("backend"),
            repeats=args.repeats,
            warmup=args.warmup,
            chunk_size=strategy.batched_z_chunk_size,
            compile_status=compile_status,
            inductor_cache_path=run_meta.inductor_cache_path,
        )
    finally:
        import linum_basic.fit as fit_mod

        fit_mod.should_use_batched_cuda = original_batched

    measured_fits = captured_fits[args.warmup + inductor_warm_passes :]
    if not measured_fits:
        print("fit_mosaic did not produce measured repeats", file=sys.stderr)
        return 1

    repeat_reports = [compute_quality_report(mosaic, fit) for fit in measured_fits]
    tolerance_specs = calibrate_tolerances(repeat_reports)
    primary_report = repeat_reports[-1]

    ts_compact = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    baseline_id = make_baseline_id(
        commit=run_meta.git_commit,
        subject_id=subject_id,
        timestamp=ts_compact,
    )

    array_shape = [mosaic.n_z, mosaic.array.shape[1], mosaic.array.shape[2]]
    tile_shape = [mosaic.tile_shape[0], mosaic.tile_shape[1]]
    fingerprint = _input_fingerprint(str(input_path), tuple(array_shape))

    metadata = {
        "git_commit": run_meta.git_commit,
        "host_node": run_meta.host_node,
        "platform": run_meta.platform,
        "python_version": run_meta.python_version,
        "numpy_version": run_meta.numpy_version,
        "torch_version": run_meta.torch_version,
        "cuda_version": run_meta.cuda_version,
        "cuda_available": run_meta.cuda_available,
        "cuda_devices": run_meta.cuda_devices,
        "cache_mode": run_meta.cache_mode,
        "inductor_warm_passes": inductor_warm_passes,
        "inductor_cache_path": run_meta.inductor_cache_path,
        "fx_graph_cache_enabled": run_meta.fx_graph_cache_enabled,
        "telemetry": _telemetry_metadata(telemetry, run_meta),
        "operator_timing": _operator_timing_metadata(
            telemetry,
            n_z=len(z_indices),
            n_tiles=mosaic.n_tiles,
        ),
        "precision": _precision_metadata(),
        "release_gate": strategy.release_gate,
    }

    bundle = BaselineBundle(
        baseline_id=baseline_id,
        uuid=str(uuid.uuid4()),
        schema_version="1",
        metric_definition_version=METRIC_DEFINITION_VERSION,
        subject_id=subject_id,
        run_label=run_label,
        input_fingerprint=fingerprint,
        z_indices=z_indices,
        array_shape=array_shape,
        tile_shape=tile_shape,
        strategy_params=dict(strategy.basic_kwargs),
        metrics_rows=[
            {"z": row.z, "seam_l1": row.seam_l1, "seam_curvature": row.seam_curvature} for row in primary_report.rows
        ],
        metrics_aggregates=dict(primary_report.aggregates),
        repeats=args.repeats,
        metadata=metadata,
        timestamp=run_meta.timestamp,
    )

    bundle_dir = output_dir / baseline_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    write_artifact(bundle_dir / "baseline-bundle.json", bundle)
    write_artifact(
        bundle_dir / "tolerance-sidecar.json",
        ToleranceSidecar(
            baseline_id=baseline_id,
            schema_version="1",
            metric_definition_version=METRIC_DEFINITION_VERSION,
            calibration_policy=CALIBRATION_POLICY,
            sigma=3.0,
            min_abs=1e-6,
            tolerances={
                metric: {
                    "mean": spec.mean,
                    "std": spec.std,
                    "abs_tol": spec.abs_tol,
                    "rel_tol": spec.rel_tol,
                }
                for metric, spec in tolerance_specs.items()
            },
        ),
    )

    summary_rows = [
        {
            "z": row.z,
            "seam_l1": f"{row.seam_l1:.6f}",
            "seam_curvature": f"{row.seam_curvature:.6f}",
        }
        for row in primary_report.rows
    ]
    summary_rows.append(
        {
            "z": "aggregate",
            "seam_l1": f"{primary_report.aggregates['seam_l1']:.6f}",
            "seam_curvature": f"{primary_report.aggregates['seam_curvature']:.6f}",
        }
    )
    write_summary_table(bundle_dir / "summary.md", summary_rows, fmt="markdown")

    print(f"Wrote baseline bundle to {bundle_dir}")
    return 0


def _load_baseline_bundle(output_dir: Path, baseline_id: str) -> tuple[BaselineBundle, ToleranceSidecar]:
    """Load a saved baseline bundle and tolerance sidecar from *output_dir*."""
    bundle_dir = output_dir / baseline_id
    if not bundle_dir.is_dir():
        msg = f"Baseline bundle not found: {bundle_dir}"
        raise FileNotFoundError(msg)
    bundle = read_artifact(bundle_dir / "baseline-bundle.json", BaselineBundle)
    sidecar = read_artifact(bundle_dir / "tolerance-sidecar.json", ToleranceSidecar)
    return bundle, sidecar


def _quality_report_from_bundle(bundle: BaselineBundle | CandidateArtifact) -> QualityReport:
    rows = tuple(
        PerZMetricRow(
            z=int(row["z"]),
            seam_l1=float(row["seam_l1"]),
            seam_curvature=float(row["seam_curvature"]),
        )
        for row in bundle.metrics_rows
    )
    return QualityReport(
        rows=rows,
        aggregates=dict(bundle.metrics_aggregates),
        metric_definition_version=bundle.metric_definition_version,
    )


def _tolerances_from_sidecar(sidecar: ToleranceSidecar) -> dict[str, ToleranceSpec]:
    specs: dict[str, ToleranceSpec] = {}
    for metric, tol in sidecar.tolerances.items():
        abs_tol = float(tol["abs_tol"])
        rel_tol = float(tol["rel_tol"])
        if "std" in tol:
            std = float(tol["std"])
        else:
            std = max(0.0, (abs_tol - sidecar.min_abs) / sidecar.sigma) if sidecar.sigma else 0.0
        if "mean" in tol:
            mean = float(tol["mean"])
        elif rel_tol > 0.0:
            mean = abs_tol / rel_tol - sidecar.min_abs
        else:
            mean = 0.0
        specs[metric] = ToleranceSpec(
            metric=metric,
            mean=mean,
            std=std,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            sigma=sidecar.sigma,
            min_abs=sidecar.min_abs,
        )
    return specs


def _serialize_metric_delta(delta: Any) -> dict[str, float]:
    return {"abs_delta": float(delta.abs_delta), "rel_delta": float(delta.rel_delta)}


def _serialize_deltas(deltas: dict[str, Any]) -> dict[str, Any]:
    aggregate = {key: _serialize_metric_delta(val) for key, val in deltas["aggregate"].items()}
    per_z = [
        {
            "z": row["z"],
            **{metric: _serialize_metric_delta(row[metric]) for metric in ("seam_l1", "seam_curvature")},
        }
        for row in deltas["per_z"]
    ]
    return {"aggregate": aggregate, "per_z": per_z}


def _serialize_quality_verdict(verdict: QualityVerdict) -> dict[str, Any]:
    return {
        "passed": verdict.passed,
        "failures": list(verdict.failures),
        "worst_z": verdict.worst_z,
        "aggregate_deltas": {metric: _serialize_metric_delta(delta) for metric, delta in verdict.aggregate_deltas.items()},
        "per_z_failures": [dict(row) for row in verdict.per_z_failures],
    }


def compute_speed_verdict(candidate_telemetry: dict[str, Any], baseline_telemetry: dict[str, Any]) -> dict[str, Any]:
    """Derive a speed verdict from candidate vs baseline steady-state timing."""
    if not candidate_telemetry or candidate_telemetry.get("steady_state_ms") is None:
        base_ms = baseline_telemetry.get("steady_state_ms")
        return {
            "ratio": None,
            "label": "insufficient_telemetry",
            "candidate_ms": None,
            "baseline_ms": float(base_ms) if base_ms is not None else None,
        }

    cand_ms = float(candidate_telemetry["steady_state_ms"])
    base_ms = float(baseline_telemetry.get("steady_state_ms", 0.0))
    ratio = (float("inf") if base_ms > 0 else 1.0) if cand_ms <= 0 else base_ms / cand_ms
    if ratio > 1.05:
        label = "faster"
    elif ratio < 0.95:
        label = "slower"
    else:
        label = "neutral"
    return {
        "ratio": ratio,
        "label": label,
        "candidate_ms": cand_ms,
        "baseline_ms": base_ms,
    }


def build_overall_verdict(speed_verdict: dict[str, Any], quality_verdict: QualityVerdict) -> str:
    """Return promote/reject; quality gate failure always rejects regardless of speed."""
    del speed_verdict
    return "promote" if quality_verdict.passed else "reject"


def _make_candidate_id(*, commit: str, subject_id: str, timestamp: str) -> str:
    short_commit = commit[:7]
    subject_slug = slugify_label(subject_id)
    return f"candidate-{timestamp}-{short_commit}-{subject_slug}"


def _probe_candidate(
    *,
    baseline_id: str,
    subject_id: str,
    run_label: str,
    input_fingerprint: str,
    z_indices: list[int],
    array_shape: list[int],
    tile_shape: list[int],
    strategy_params: dict[str, Any],
) -> CandidateArtifact:
    return CandidateArtifact(
        candidate_id="probe",
        baseline_id=baseline_id,
        schema_version="1",
        metric_definition_version=METRIC_DEFINITION_VERSION,
        subject_id=subject_id,
        run_label=run_label,
        input_fingerprint=input_fingerprint,
        z_indices=z_indices,
        array_shape=array_shape,
        tile_shape=tile_shape,
        strategy_params=strategy_params,
        metrics_rows=[],
        metrics_aggregates={"seam_l1": 0.0, "seam_curvature": 0.0},
        repeats=1,
        metadata={},
        environment={},
        timestamp="",
    )


def _build_comparison_summary(
    *,
    baseline: BaselineBundle,
    candidate: CandidateArtifact,
    deltas: dict[str, Any],
    speed_verdict: dict[str, Any],
    quality_verdict: QualityVerdict,
    overall: str,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "baseline_id": baseline.baseline_id,
        "candidate_id": candidate.candidate_id,
        "overall": overall,
        "speed_verdict": speed_verdict,
        "quality_verdict": _serialize_quality_verdict(quality_verdict),
        "deltas": _serialize_deltas(deltas),
        "warnings": warnings,
        "worst_z": quality_verdict.worst_z,
    }


def _comparison_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, delta in summary["deltas"]["aggregate"].items():
        rows.append(
            {
                "metric": metric,
                "abs_delta": f"{delta['abs_delta']:.6f}",
                "rel_delta": f"{delta['rel_delta']:.6f}",
            }
        )
    rows.append(
        {
            "metric": "speed",
            "abs_delta": (
                f"{summary['speed_verdict']['ratio']:.4f}x" if summary["speed_verdict"]["ratio"] is not None else "n/a"
            ),
            "rel_delta": summary["speed_verdict"]["label"],
        }
    )
    rows.append(
        {
            "metric": "overall",
            "abs_delta": summary["overall"],
            "rel_delta": "pass" if summary["quality_verdict"]["passed"] else "fail",
        }
    )
    if summary.get("worst_z") is not None:
        rows.append(
            {
                "metric": "worst_z",
                "abs_delta": str(summary["worst_z"]),
                "rel_delta": "",
            }
        )
    return rows


def _run_candidate_comparison(
    baseline: BaselineBundle,
    sidecar: ToleranceSidecar,
    candidate_report: QualityReport,
    candidate_telemetry: dict[str, Any],
    candidate_artifact: CandidateArtifact,
) -> tuple[dict[str, Any], dict[str, Any], QualityVerdict, str, list[str]]:
    warnings = assert_comparable(candidate_artifact, baseline)
    baseline_report = _quality_report_from_bundle(baseline)
    deltas = compute_deltas(candidate_report, baseline_report)
    tolerances = _tolerances_from_sidecar(sidecar)
    quality_verdict = evaluate_quality_gate(deltas, tolerances)
    baseline_telemetry = baseline.metadata.get("telemetry", {})
    speed_verdict = compute_speed_verdict(candidate_telemetry, baseline_telemetry)
    overall = build_overall_verdict(speed_verdict, quality_verdict)
    summary = _build_comparison_summary(
        baseline=baseline,
        candidate=candidate_artifact,
        deltas=deltas,
        speed_verdict=speed_verdict,
        quality_verdict=quality_verdict,
        overall=overall,
        warnings=warnings,
    )
    return summary, speed_verdict, quality_verdict, overall, warnings


def cmd_candidate(args: argparse.Namespace) -> int:
    """Run a candidate strategy against a saved baseline and write verdict artifacts."""
    if not args.output_dir:
        print("Candidate runs require --output-dir", file=sys.stderr)
        return 1
    if not args.baseline_id:
        print("Candidate runs require --baseline-id", file=sys.stderr)
        return 1
    if not args.input:
        print("Candidate runs require --input", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    try:
        baseline, sidecar = _load_baseline_bundle(output_dir, args.baseline_id)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if sidecar.baseline_id != baseline.baseline_id:
        print(
            f"Tolerance sidecar baseline_id {sidecar.baseline_id!r} does not match bundle {baseline.baseline_id!r}",
            file=sys.stderr,
        )
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    try:
        from linum_basic.mosaic import MosaicGrid

        mosaic = MosaicGrid.from_ome_zarr(str(input_path))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Failed to load OME-Zarr: {exc}", file=sys.stderr)
        return 1

    subject_id = args.subject_id or baseline.subject_id
    run_label = args.run_label or args.subject_id or baseline.run_label
    z_indices = list(baseline.z_indices)

    if args.z_indices is not None or args.z_sample is not None:
        try:
            selected = resolve_z_selection(
                mosaic.n_z,
                z_indices=args.z_indices,
                z_sample=args.z_sample,
            )
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        if selected != z_indices:
            print(
                f"Selected z-planes differ from baseline bundle; expected {z_indices}, got {selected}",
                file=sys.stderr,
            )
            return 1

    try:
        enforce_large_run_guard(
            len(z_indices),
            allow_large_run=bool(args.allow_large_run),
            yes=bool(args.yes),
        )
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1

    overrides: dict[str, Any] = {}
    if args.config:
        try:
            overrides = load_overrides(args.config)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1

    strategy = resolve_strategy(
        args.strategy,
        working_size=args.working_size,
        estimate_darkfield=args.estimate_darkfield,
        max_reweighting_iterations=args.max_reweighting_iterations,
        batched_z_chunk_size=args.batched_z_chunk_size,
        overrides=overrides,
        is_synthetic=bool(args.synthetic),
    )

    array_shape = [mosaic.n_z, mosaic.array.shape[1], mosaic.array.shape[2]]
    tile_shape = [mosaic.tile_shape[0], mosaic.tile_shape[1]]
    fingerprint = _input_fingerprint(str(input_path), tuple(array_shape))

    probe = _probe_candidate(
        baseline_id=baseline.baseline_id,
        subject_id=subject_id,
        run_label=run_label,
        input_fingerprint=fingerprint,
        z_indices=z_indices,
        array_shape=array_shape,
        tile_shape=tile_shape,
        strategy_params=dict(strategy.basic_kwargs),
    )
    try:
        warnings = assert_comparable(probe, baseline)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    original_batched = _apply_strategy_batched_patch(strategy.force_batched)
    captured_fits: list[MosaicFit] = []

    run_meta = collect_run_metadata(
        strategy_params=dict(strategy.basic_kwargs),
        z_indices=z_indices,
        cache_mode="persistent" if os.environ.get("LINUM_BASIC_TORCH_CACHE", "1") != "0" else "disabled",
        configure_cache=False,
    )
    compile_status = _resolve_compile_status(strategy, run_meta)

    def _fit_once() -> MosaicFit:
        fit = fit_mosaic(
            mosaic,
            z_indices=z_indices,
            basic_kwargs=strategy.basic_kwargs,
            strategy=cast(Literal["auto", "sequential", "multi", "batched"], strategy.name),
            n_workers=1,
            verbose=False,
        )
        captured_fits.append(fit)
        return fit

    inductor_warm_passes = warm_policy_passes()
    for _ in range(inductor_warm_passes):
        _fit_once()

    try:
        telemetry = run_with_phases(
            _fit_once,
            device=strategy.basic_kwargs.get("device"),
            backend=strategy.basic_kwargs.get("backend"),
            repeats=args.repeats,
            warmup=args.warmup,
            chunk_size=strategy.batched_z_chunk_size,
            compile_status=compile_status,
            inductor_cache_path=run_meta.inductor_cache_path,
        )
    finally:
        import linum_basic.fit as fit_mod

        fit_mod.should_use_batched_cuda = original_batched

    measured_fits = captured_fits[args.warmup + inductor_warm_passes :]
    if not measured_fits:
        print("fit_mosaic did not produce measured repeats", file=sys.stderr)
        return 1

    primary_fit = measured_fits[-1]
    candidate_report = compute_quality_report(mosaic, primary_fit)

    ts_compact = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    candidate_id = _make_candidate_id(
        commit=run_meta.git_commit,
        subject_id=subject_id,
        timestamp=ts_compact,
    )

    environment = {
        "cuda_version": run_meta.cuda_version,
        "torch_version": run_meta.torch_version,
        "python_version": run_meta.python_version,
        "platform": run_meta.platform,
    }

    candidate_artifact = CandidateArtifact(
        candidate_id=candidate_id,
        baseline_id=baseline.baseline_id,
        schema_version="1",
        metric_definition_version=METRIC_DEFINITION_VERSION,
        subject_id=subject_id,
        run_label=run_label,
        input_fingerprint=fingerprint,
        z_indices=z_indices,
        array_shape=array_shape,
        tile_shape=tile_shape,
        strategy_params=dict(strategy.basic_kwargs),
        metrics_rows=[
            {"z": row.z, "seam_l1": row.seam_l1, "seam_curvature": row.seam_curvature} for row in candidate_report.rows
        ],
        metrics_aggregates=dict(candidate_report.aggregates),
        repeats=args.repeats,
        metadata={},
        environment=environment,
        timestamp=run_meta.timestamp,
    )

    candidate_telemetry = _telemetry_metadata(telemetry, run_meta)
    max_reweight_iters = int(strategy.basic_kwargs.get("max_reweighting_iterations", 10))
    convergence_meta = _convergence_metadata(
        primary_fit,
        z_indices,
        max_reweighting_iterations=max_reweight_iters,
    )

    summary, speed_verdict, quality_verdict, overall, comparison_warnings = _run_candidate_comparison(
        baseline,
        sidecar,
        candidate_report,
        candidate_telemetry,
        candidate_artifact,
    )
    all_warnings = [*warnings, *comparison_warnings]

    visible_devices = list(run_meta.cuda_devices)
    gpu_map = _gpu_map_for_strategy(strategy.name, visible_devices)
    serialized_quality = _serialize_quality_verdict(quality_verdict)
    multi_gpu_stats = collect_multi_gpu_memory_stats(visible_devices)
    peak_vram_bytes = peak_single_gpu_vram_bytes(multi_gpu_stats)
    if peak_vram_bytes == 0:
        peak_vram_bytes = int(candidate_telemetry.get("max_memory_allocated_bytes") or 0)
    concurrency_meta = collect_concurrency_metadata(
        strategy=strategy.name,
        fork_model=None,
        n_gpus=len(visible_devices),
        gpu_map=gpu_map,
        inductor_cache_path=run_meta.inductor_cache_path,
        compile_status=compile_status,
        quality_verdict=serialized_quality,
    )
    concurrency_meta["peak_vram_bytes"] = peak_vram_bytes

    candidate_artifact = CandidateArtifact(
        candidate_id=candidate_id,
        baseline_id=baseline.baseline_id,
        schema_version="1",
        metric_definition_version=METRIC_DEFINITION_VERSION,
        subject_id=subject_id,
        run_label=run_label,
        input_fingerprint=fingerprint,
        z_indices=z_indices,
        array_shape=array_shape,
        tile_shape=tile_shape,
        strategy_params=dict(strategy.basic_kwargs),
        metrics_rows=[
            {"z": row.z, "seam_l1": row.seam_l1, "seam_curvature": row.seam_curvature} for row in candidate_report.rows
        ],
        metrics_aggregates=dict(candidate_report.aggregates),
        repeats=args.repeats,
        metadata={
            "git_commit": run_meta.git_commit,
            "host_node": run_meta.host_node,
            "platform": run_meta.platform,
            "python_version": run_meta.python_version,
            "numpy_version": run_meta.numpy_version,
            "torch_version": run_meta.torch_version,
            "cuda_version": run_meta.cuda_version,
            "cuda_available": run_meta.cuda_available,
            "cuda_devices": run_meta.cuda_devices,
            "cache_mode": run_meta.cache_mode,
            "inductor_warm_passes": inductor_warm_passes,
            "inductor_cache_path": run_meta.inductor_cache_path,
            "fx_graph_cache_enabled": run_meta.fx_graph_cache_enabled,
            "telemetry": candidate_telemetry,
            "operator_timing": _operator_timing_metadata(
                telemetry,
                n_z=len(z_indices),
                n_tiles=mosaic.n_tiles,
            ),
            "precision": _precision_metadata(),
            "convergence": convergence_meta,
            "concurrency": concurrency_meta,
            "speed_verdict": speed_verdict,
            "quality_verdict": _serialize_quality_verdict(quality_verdict),
            "overall": overall,
            "deltas": summary["deltas"],
            "warnings": all_warnings,
        },
        environment=environment,
        timestamp=run_meta.timestamp,
    )

    candidate_dir = output_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    write_artifact(candidate_dir / "candidate-artifact.json", candidate_artifact)
    write_summary_table(
        candidate_dir / "summary.md",
        _comparison_summary_rows(summary),
        fmt="markdown",
    )

    print(f"Wrote candidate artifact to {candidate_dir} (overall={overall})")
    for warning in all_warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    return 1 if overall == "reject" else 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Recompute promote/reject summary from saved baseline and candidate artifacts."""
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    output_dir = Path(args.output_dir)

    if not baseline_path.is_file():
        print(f"Baseline artifact not found: {baseline_path}", file=sys.stderr)
        return 1
    if not candidate_path.is_file():
        print(f"Candidate artifact not found: {candidate_path}", file=sys.stderr)
        return 1

    baseline = read_artifact(baseline_path, BaselineBundle)
    candidate = read_artifact(candidate_path, CandidateArtifact)

    sidecar_path = baseline_path.parent / "tolerance-sidecar.json"
    if not sidecar_path.is_file():
        print(f"Tolerance sidecar not found beside baseline: {sidecar_path}", file=sys.stderr)
        return 1
    sidecar = read_artifact(sidecar_path, ToleranceSidecar)

    try:
        candidate_report = _quality_report_from_bundle(candidate)
        candidate_telemetry = candidate.metadata.get("telemetry", {})
        summary, _speed, _quality, overall, warnings = _run_candidate_comparison(
            baseline,
            sidecar,
            candidate_report,
            candidate_telemetry,
            candidate,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "compare-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_table(
        output_dir / "summary.md",
        _comparison_summary_rows(summary),
        fmt="markdown",
    )

    print(f"Wrote comparison summary to {output_dir} (overall={overall})")
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    return 1 if overall == "reject" else 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Aggregate saved baseline and candidate artifacts into sweep table + verdict."""
    baseline_path = Path(args.baseline)
    output_dir = Path(args.output_dir)

    if not baseline_path.is_file():
        print(f"Baseline artifact not found: {baseline_path}", file=sys.stderr)
        return 1

    candidates: list[CandidateArtifact] = []
    for candidate_arg in args.candidate:
        candidate_path = Path(candidate_arg)
        if not candidate_path.is_file():
            print(f"Candidate artifact not found: {candidate_path}", file=sys.stderr)
            return 1
        candidates.append(read_artifact(candidate_path, CandidateArtifact))

    baseline = read_artifact(baseline_path, BaselineBundle)
    table = build_sweep_table(
        baseline,
        candidates,
        speed_ratio_threshold=float(args.speed_ratio_threshold),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sweep-table.json").write_text(
        json.dumps(asdict(table), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sweep-verdict.json").write_text(
        json.dumps(asdict(table.adoption), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_rows = [
        {
            "working_size": row.working_size,
            "steady_state_ms": row.steady_state_ms,
            "speed_ratio": row.speed_ratio,
            "quality_passed": row.quality_passed,
            "speed_passed": row.speed_passed,
            "overall": row.candidate_overall,
        }
        for row in table.rows
    ]
    write_summary_table(output_dir / "summary.md", summary_rows, fmt="markdown")

    print(
        f"Wrote sweep outputs to {output_dir} "
        f"(recommended_ws={table.adoption.recommended_ws}, phase3_activate={table.adoption.phase3_activate})"
    )
    return 0


def _export_profiler_key_averages(prof: Any) -> dict[str, Any]:
    """Convert torch.profiler key_averages to build_bottleneck_report input."""
    events: list[dict[str, Any]] = []
    for item in prof.key_averages():
        self_cuda_us = float(getattr(item, "self_cuda_time_total", 0.0))
        if self_cuda_us <= 0.0:
            continue
        events.append(
            {
                "name": str(item.key),
                "self_cuda_time_total": self_cuda_us / 1000.0,
            }
        )
    return {"key_averages": events, "summary": {"event_count": len(events)}}


def _collect_profiler_events(
    fit_fn: Callable[[], MosaicFit],
    *,
    warmup: int,
) -> dict[str, Any]:
    """Run *fit_fn* under torch.profiler or return a CPU smoke fixture."""
    if _cuda_available():
        import torch

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=warmup, active=1, repeat=1),
            record_shapes=True,
            with_stack=True,
        ) as prof:
            fit_fn()
            prof.step()
        return _export_profiler_key_averages(prof)

    fit_fn()
    return {
        "key_averages": [{"name": "aten::mm", "self_cuda_time_total": 1.0}],
        "summary": {"cuda_available": False},
    }


def _bottleneck_report_to_dict(report: BottleneckReport) -> dict[str, Any]:
    return cast(dict[str, Any], asdict(report))


def _lever_attempt_table_to_dict(table: LeverAttemptTable) -> dict[str, Any]:
    return cast(dict[str, Any], asdict(table))


def _ranked_levers_from_report(payload: dict[str, Any]) -> tuple[RankedLever, ...]:
    raw = payload.get("ranked_levers")
    if not isinstance(raw, list):
        msg = "bottleneck-report.json missing ranked_levers list"
        raise KeyError(msg)
    levers: list[RankedLever] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        levers.append(
            RankedLever(
                lever_id=str(entry["lever_id"]),
                target_file=str(entry["target_file"]),
                priority=int(entry["priority"]),
                expected_risk=str(entry["expected_risk"]),
                gate_notes=str(entry["gate_notes"]),
            )
        )
    if not levers:
        msg = "bottleneck-report.json ranked_levers is empty"
        raise ValueError(msg)
    return tuple(levers)


def _resolve_code_path_flags() -> dict[str, Any]:
    """Read active code-path lever env flags for the fast-path manifest."""
    from linum_basic.backend import read_dct_kernel_mode

    raw_compile = os.environ.get("LINUM_BASIC_ALM_COMPILE_MODE", "default").strip().lower()
    compile_mode = "default" if raw_compile in ("", "default") else raw_compile
    return {
        "dct_kernel": read_dct_kernel_mode(),
        "compile_mode": compile_mode,
        "inductor_warm_passes": warm_policy_passes(),
    }


def _promoted_lever_stack(attempt_table: LeverAttemptTable) -> list[str]:
    """Return ordered promoted lever ids from an attempt table."""
    return [row.lever_id for row in attempt_table.rows if row.overall == "promote"]


def _fast_path_end_to_end_ms(attempt_table: LeverAttemptTable) -> float | None:
    """Return steady-state ms for the last promoted lever attempt, if any."""
    promoted_rows = [row for row in attempt_table.rows if row.overall == "promote"]
    if not promoted_rows:
        return None
    return promoted_rows[-1].steady_state_ms


def cmd_optimize(args: argparse.Namespace) -> int:
    """Aggregate lever candidate artifacts into D-10 JSON deliverables without fit."""
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"--output-dir must be a directory: {output_dir}", file=sys.stderr)
        return 1

    report_path = output_dir / "bottleneck-report.json"
    if not report_path.is_file():
        print(
            f"optimize requires bottleneck-report.json in {output_dir}; run profile --mode sequential first.",
            file=sys.stderr,
        )
        return 1

    try:
        baseline, _sidecar = _load_baseline_bundle(output_dir, args.baseline_id)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        ranked_levers = _ranked_levers_from_report(report_payload)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"Failed to read bottleneck report: {exc}", file=sys.stderr)
        return 1

    candidate_paths = [Path(path) for path in args.candidate]
    candidates: list[CandidateArtifact] = []
    for path in candidate_paths:
        if not path.is_file():
            print(f"Candidate artifact not found: {path}", file=sys.stderr)
            return 1
        try:
            candidates.append(read_artifact(path, CandidateArtifact))
        except (ValueError, KeyError, TypeError) as exc:
            print(f"Failed to read candidate {path}: {exc}", file=sys.stderr)
            return 1

    if len(candidates) > len(ranked_levers):
        print(
            f"Too many candidates ({len(candidates)}) for ranked_levers count ({len(ranked_levers)})",
            file=sys.stderr,
        )
        return 1

    paired_levers = ranked_levers[: len(candidates)]
    try:
        attempt_table = build_lever_attempt_table(
            baseline,
            candidates,
            ranked_levers=paired_levers,
        )
    except (KeyError, ValueError) as exc:
        print(f"Failed to build lever attempt table: {exc}", file=sys.stderr)
        return 1

    backlog = build_phase5_backlog(ranked_levers, attempt_table)
    handoff = build_phase3_handoff_config(
        attempt_table.stacked_overrides,
        baseline_id=baseline.baseline_id,
    )
    lever_stack = _promoted_lever_stack(attempt_table)
    evidence_ids = [baseline.baseline_id]
    evidence_ids.extend(row.artifact_id for row in attempt_table.rows if row.overall == "promote")
    fast_path = build_phase5_fast_path(
        attempt_table.stacked_overrides,
        baseline_id=baseline.baseline_id,
        lever_stack=lever_stack,
        code_path_flags=_resolve_code_path_flags(),
        evidence_artifact_ids=evidence_ids,
        git_commit=collect_git_commit(),
        stack_speed_ratio=attempt_table.stack_speed_ratio,
        end_to_end_ms=_fast_path_end_to_end_ms(attempt_table),
    )

    table_path = output_dir / "lever-attempt-table.json"
    backlog_path = output_dir / "phase5-backlog.json"
    handoff_path = output_dir / "phase3-handoff-config.json"
    fast_path_path = output_dir / "phase5-fast-path.json"
    table_path.write_text(
        json.dumps(_lever_attempt_table_to_dict(attempt_table), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    backlog_path.write_text(json.dumps(backlog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    handoff_path.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fast_path_path.write_text(json.dumps(fast_path, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote lever artifacts to {output_dir} "
        f"(rows={len(attempt_table.rows)}, backlog={len(backlog['entries'])}, "
        f"promoted_keys={len(handoff['stacked_overrides'])}, "
        f"no_optimization={fast_path['no_optimization']})"
    )
    return 0


def _gpu_map_for_strategy(strategy_name: str, devices: list[str]) -> dict[str, str]:
    """Map visible CUDA devices to worker or batched allocation labels (D-08)."""
    if strategy_name == "batched":
        return dict.fromkeys(devices, "batched")
    return {device: f"worker-{index}" for index, device in enumerate(devices)}


def _concurrency_mode_from_candidate(candidate: CandidateArtifact) -> dict[str, Any]:
    """Extract a build_phase6_concurrency_verdict mode record from a candidate artifact."""
    meta = candidate.metadata
    concurrency = meta.get("concurrency") or {}
    operator_timing = meta.get("operator_timing") or {}
    telemetry = meta.get("telemetry") or {}

    strategy = concurrency.get("strategy")
    if not strategy:
        msg = f"Candidate {candidate.candidate_id} missing concurrency.strategy metadata"
        raise ValueError(msg)

    end_to_end_ms = operator_timing.get("end_to_end_ms")
    per_z_ms = operator_timing.get("per_z_ms")
    steady_state_ms = telemetry.get("steady_state_ms")
    if end_to_end_ms is None or per_z_ms is None or steady_state_ms is None:
        msg = (
            f"Candidate {candidate.candidate_id} missing operator_timing or telemetry "
            "fields required for concurrency verdict aggregation"
        )
        raise ValueError(msg)

    peak_vram = concurrency.get("peak_vram_bytes")
    if peak_vram is None:
        peak_vram = telemetry.get("max_memory_allocated_bytes", 0)

    return {
        "strategy": str(strategy),
        "fork_model": concurrency.get("fork_model"),
        "end_to_end_ms": float(end_to_end_ms),
        "per_z_ms": float(per_z_ms),
        "steady_state_ms": float(steady_state_ms),
        "quality_verdict": meta.get("quality_verdict"),
        "artifact_id": candidate.candidate_id,
        "peak_vram_bytes": int(peak_vram or 0),
        "gpu_map": concurrency.get("gpu_map"),
    }


def cmd_concurrency(args: argparse.Namespace) -> int:
    """Aggregate multi and batched candidate artifacts into phase6-concurrency-verdict.json."""
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"--output-dir must be a directory: {output_dir}", file=sys.stderr)
        return 1

    candidate_paths = [Path(path) for path in args.candidate]
    if len(candidate_paths) != 2:
        print(
            f"concurrency requires exactly 2 --candidate paths, got {len(candidate_paths)}",
            file=sys.stderr,
        )
        return 1

    candidates: list[CandidateArtifact] = []
    for path in candidate_paths:
        if not path.is_file():
            print(f"Candidate artifact not found: {path}", file=sys.stderr)
            return 1
        try:
            candidates.append(read_artifact(path, CandidateArtifact))
        except (ValueError, KeyError, TypeError) as exc:
            print(f"Failed to read candidate {path}: {exc}", file=sys.stderr)
            return 1

    try:
        modes = [_concurrency_mode_from_candidate(candidate) for candidate in candidates]
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    evidence_ids = [args.baseline_id, *[candidate.candidate_id for candidate in candidates]]
    verdict = build_phase6_concurrency_verdict(
        modes,
        baseline_id=args.baseline_id,
        phase5_fast_path_ref=args.fast_path_ref or "",
        evidence_artifact_ids=evidence_ids,
        git_commit=collect_git_commit(),
    )

    verdict_path = output_dir / "phase6-concurrency-verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    winner = verdict.get("winner")
    winner_strategy = winner.get("strategy") if isinstance(winner, dict) else None
    print(
        f"Wrote phase6 concurrency verdict to {verdict_path} "
        f"(winner={winner_strategy}, recommended_max_forks={verdict.get('recommended_max_forks')})"
    )
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """Profile ws=128 fit and write harness JSON artifacts."""
    if args.mode == "batched-diagnostic":
        return _cmd_profile_batched_diagnostic(args)

    if args.mode != "sequential":
        print(f"Unsupported profile mode: {args.mode}", file=sys.stderr)
        return 1

    return _cmd_profile_sequential(args)


def _validate_profile_common(args: argparse.Namespace) -> tuple[Path, Path] | int:
    """Validate shared profile CLI args; return (output_dir, input_path) or exit code."""
    if not args.output_dir:
        print("Profile runs require --output-dir", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        print(f"--output-dir must be a directory: {output_dir}", file=sys.stderr)
        return 1

    if not args.input:
        print("Profile runs require --input", file=sys.stderr)
        return 1

    if args.strategy != "baseline":
        print("Profile mode requires --strategy baseline", file=sys.stderr)
        return 1

    if args.working_size != 128:
        print("Profile mode requires --working-size 128", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    return output_dir, input_path


def _resolve_profile_z_indices(args: argparse.Namespace, mosaic: Any) -> list[int] | int:
    """Resolve z-plane selection for profile commands."""
    z_indices: list[int]
    if args.baseline_id:
        try:
            baseline, _sidecar = _load_baseline_bundle(Path(args.output_dir), args.baseline_id)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        z_indices = list(baseline.z_indices)
        if args.z_indices is not None or args.z_sample is not None:
            try:
                selected = resolve_z_selection(
                    mosaic.n_z,
                    z_indices=args.z_indices,
                    z_sample=args.z_sample,
                )
            except ValueError as exc:
                print(exc, file=sys.stderr)
                return 1
            if selected != z_indices:
                print(
                    f"Selected z-planes differ from baseline bundle; expected {z_indices}, got {selected}",
                    file=sys.stderr,
                )
                return 1
    else:
        try:
            z_indices = resolve_z_selection(
                mosaic.n_z,
                z_indices=args.z_indices,
                z_sample=args.z_sample,
            )
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1

    try:
        enforce_large_run_guard(
            len(z_indices),
            allow_large_run=bool(args.allow_large_run),
            yes=bool(args.yes),
        )
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1

    return z_indices


def _cmd_profile_batched_diagnostic(args: argparse.Namespace) -> int:
    """Run full-z batched CUDA diagnostic after sequential bottleneck report (D-02-D-04)."""
    common = _validate_profile_common(args)
    if isinstance(common, int):
        return common
    output_dir, input_path = common

    require_report = bool(getattr(args, "require_sequential_report", True))
    report_path = output_dir / "bottleneck-report.json"
    if require_report and not report_path.is_file():
        print(
            "batched-diagnostic requires bottleneck-report.json in --output-dir; run profile --mode sequential first (D-04).",
            file=sys.stderr,
        )
        return 1

    primary_limit = "unknown"
    if report_path.is_file():
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            raw_limit = report_payload.get("primary_limit")
            if isinstance(raw_limit, str) and raw_limit:
                primary_limit = raw_limit
        except json.JSONDecodeError:
            pass

    try:
        from linum_basic.mosaic import MosaicGrid

        mosaic = MosaicGrid.from_ome_zarr(str(input_path))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Failed to load OME-Zarr: {exc}", file=sys.stderr)
        return 1

    z_indices = _resolve_profile_z_indices(args, mosaic)
    if isinstance(z_indices, int):
        return z_indices

    overrides: dict[str, Any] = {
        "force_batched_cuda": True,
        "batched_z_chunk_size": len(z_indices),
        "working_size": 128,
    }
    if args.config:
        try:
            overrides.update(load_overrides(args.config))
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
    overrides["force_batched_cuda"] = True
    overrides["batched_z_chunk_size"] = len(z_indices)
    overrides["working_size"] = 128

    strategy = resolve_strategy(
        "baseline",
        working_size=128,
        estimate_darkfield=args.estimate_darkfield,
        max_reweighting_iterations=args.max_reweighting_iterations,
        batched_z_chunk_size=len(z_indices),
        overrides=overrides,
        is_synthetic=bool(args.synthetic),
    )

    original_batched = _apply_strategy_batched_patch(True)

    def _fit_once() -> MosaicFit:
        return fit_mosaic(
            mosaic,
            z_indices=z_indices,
            basic_kwargs=strategy.basic_kwargs,
            n_workers=1,
            verbose=False,
        )

    try:
        t0 = time.perf_counter()
        _fit_once()
        steady_state_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        import linum_basic.fit as fit_mod

        fit_mod.should_use_batched_cuda = original_batched

    memory = collect_memory_stats("cuda:0")
    peak_memory_bytes = memory.max_memory_allocated_bytes if memory is not None else 0

    handoff = {
        "mode": "batched-diagnostic",
        "n_z": len(z_indices),
        "peak_memory_bytes": peak_memory_bytes,
        "steady_state_ms": steady_state_ms,
        "throughput_ratio_vs_sequential": 1.0,
        "primary_limit": primary_limit,
        "primary_limit_chunking_note": (
            "Full-z batched CUDA diagnostic for Phase 6 bandwidth evidence; "
            "production guard keeps ws=128 on scalar path unless force_batched_cuda is set."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    handoff_path = output_dir / "batched-handoff.json"
    handoff_path.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote batched handoff to {handoff_path} (n_z={handoff['n_z']}, primary_limit={primary_limit})")
    return 0


def _cmd_profile_sequential(args: argparse.Namespace) -> int:
    """Profile sequential ws=128 fit and write bottleneck-report.json."""
    common = _validate_profile_common(args)
    if isinstance(common, int):
        return common
    output_dir, input_path = common

    try:
        from linum_basic.mosaic import MosaicGrid

        mosaic = MosaicGrid.from_ome_zarr(str(input_path))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Failed to load OME-Zarr: {exc}", file=sys.stderr)
        return 1

    z_indices = _resolve_profile_z_indices(args, mosaic)
    if isinstance(z_indices, int):
        return z_indices

    overrides: dict[str, Any] = {}
    if args.config:
        try:
            overrides = load_overrides(args.config)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1

    strategy = resolve_strategy(
        "baseline",
        working_size=args.working_size,
        estimate_darkfield=args.estimate_darkfield,
        max_reweighting_iterations=args.max_reweighting_iterations,
        batched_z_chunk_size=args.batched_z_chunk_size,
        overrides=overrides,
        is_synthetic=bool(args.synthetic),
    )

    original_batched = _apply_strategy_batched_patch(strategy.force_batched)

    def _fit_once() -> MosaicFit:
        return fit_mosaic(
            mosaic,
            z_indices=z_indices,
            basic_kwargs=strategy.basic_kwargs,
            n_workers=1,
            verbose=False,
        )

    try:
        profiler_events = _collect_profiler_events(_fit_once, warmup=args.warmup)
    finally:
        import linum_basic.fit as fit_mod

        fit_mod.should_use_batched_cuda = original_batched

    report = build_bottleneck_report(profiler_events, working_size=args.working_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "bottleneck-report.json"
    report_path.write_text(
        json.dumps(_bottleneck_report_to_dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Precision metadata is stored in a companion file so bottleneck-report.json
    # remains a pure BottleneckReport schema for Plan 03-01 consumers.
    precision_path = output_dir / "profile-metadata.json"
    precision_path.write_text(
        json.dumps({"precision": _precision_metadata()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote bottleneck report to {report_path} (primary_limit={report.primary_limit}, levers={len(report.ranked_levers)})"
    )
    return 0


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
            "force_batched_cuda": True,
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


def _build_legacy_parser() -> argparse.ArgumentParser:
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


def _main_legacy(argv: list[str] | None = None) -> int:
    args = _build_legacy_parser().parse_args(argv)
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


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if argv_list and argv_list[0] in _HARNESS_SUBCOMMANDS:
        args = _build_subcommand_parser().parse_args(argv_list)
        handler = getattr(args, "handler", None)
        if handler is None:
            print("No subcommand handler", file=sys.stderr)
            return 1
        return int(handler(args))
    return _main_legacy(argv_list)


if __name__ == "__main__":
    sys.exit(main())
