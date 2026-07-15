#!/usr/bin/env python3
"""Run the D023 guarded auto-apply loop and persist the full ``.gate`` dict.

This is the operator-run probe for the M008/S03 real-subject validation of the
auto-apply safety gate (R057). It runs :func:`linum_basic.tuning.auto_tune` on
a real OME-Zarr mosaic and writes the complete ``.gate`` explainability
sub-dict (plus reproducibility provenance) to a JSON artifact — the exact
fields the dated Validation Log entry in ``docs/tuning.md`` must cite
(``gate_verdict`` / ``applied`` / ``deltas`` / ``failing_metrics`` /
``fallback_reason``).

Why a probe and not just ``basic tune --auto-apply``?
-----------------------------------------------------
``basic tune --auto-apply`` composes the gate with the existing output flags
(``--out-json`` writes best params, ``--bounds-json`` writes the narrowed
recommendation, ``--apply`` writes the winning corrected volume). It only
*prints* the delta table under ``--verbose`` and never persists the
``baseline`` / ``candidate`` aggregates or the ``deltas`` block — the very
fields a Validation Log entry must record for R057 to be reproducible. This
probe captures the whole :class:`AutoTuneResult.gate` dict so the validation
evidence survives the run.

It runs in *exactly* the same code path the CLI uses
(:func:`auto_tune` → :func:`tune` → :func:`recommend_bounds` → two
:func:`fit_mosaic` calls → non-regression gate) and adds no solver code (K02).

Pass-path validation (T04)::

    CUDA_VISIBLE_DEVICES=0 uv run python \\
        scripts/experiments/m008_s03/auto_apply_validation.py \\
        --input /scratch/workspace/sub-22/output/27/resample_mosaic_grid \\
        --output scripts/experiments/m008_s03/auto-apply-pass.json \\
        --n-trials 50 --z-subsample 4 --backend torch --device cuda:0 \\
        --max-tiles 64 --verbose

Local smoke test (no CUDA; proves the plumbing end-to-end on a tiny synthetic
mosaic, NumPy backend, 2 trials) — run from the repo root::

    uv run python scripts/experiments/m008_s03/auto_apply_validation.py \\
        --smoke --output /tmp/auto-apply-smoke.json --verbose
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Reused (not duplicated) provenance helpers from the benchmark package, the
# same helpers scripts/streaming_memory_probe.py stamps its artifacts with.
from linum_basic.benchmark import collect_git_commit, collect_host_info
from linum_basic.mosaic import MosaicGrid
from linum_basic.tuning import AutoApplyError, auto_tune

SCHEMA_VERSION = 1


def _collect_software_versions() -> dict[str, Any]:
    """Collect torch / CUDA versions (best-effort) for provenance."""
    info: dict[str, Any] = {"python_version": platform.python_version()}
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
    except ImportError, RuntimeError:  # pragma: no cover - torch optional
        info["torch_version"] = "unavailable"
        info["cuda_available"] = False
        info["cuda_version"] = None
    return info


def _gate_to_jsonable(gate: dict[str, Any]) -> dict[str, Any]:
    """Make the ``.gate`` dict JSON-serialisable (it already is, but be safe)."""
    out: dict[str, Any] = {}
    for key, value in gate.items():
        # The baseline / candidate sub-dicts may carry numpy floats inside
        # their ``aggregates``; coerce recursively.
        out[key] = _coerce(value)
    return out


def _coerce(value: Any) -> Any:
    """Recursively coerce numpy scalars to plain Python for JSON."""
    if isinstance(value, dict):
        return {k: _coerce(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    # numpy float / int scalars and plain floats
    try:
        return float(value)
    except TypeError, ValueError:
        return str(value)


def run_validation(
    mosaic: MosaicGrid,
    *,
    n_trials: int,
    z_subsample: int,
    margin: float,
    seed: int,
    backend: str,
    device: str | None,
    max_tiles: int | None,
    n_extra_rows: int,
    verbose: bool,
    provenance_extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run ``auto_tune`` and return ``(gate_dict, full_artifact)``.

    The full artifact bundles the gate dict with provenance so the caller can
    write it to disk unchanged.
    """
    t0 = time.perf_counter()
    try:
        result = auto_tune(
            mosaic,
            n_trials=n_trials,
            z_subsample=z_subsample,
            margin=margin,
            seed=seed,
            backend=backend,
            device=device,
            max_tiles=max_tiles,
            n_extra_rows=n_extra_rows,
            verbose=verbose,
        )
    except AutoApplyError as exc:
        # The baseline-fit-failure path: auto_tune raises rather than falls
        # back. Record it honestly in the same artifact shape.
        elapsed = time.perf_counter() - t0
        gate = {
            "gate_verdict": "error",
            "applied": None,
            "fallback_reason": "auto-apply-error",
            "error": str(exc),
        }
        artifact = _build_artifact(gate, elapsed, backend, device, provenance_extra, error=True)
        return gate, artifact

    elapsed = time.perf_counter() - t0
    gate = _gate_to_jsonable(result.gate)
    artifact = _build_artifact(gate, elapsed, backend, device, provenance_extra)
    return gate, artifact


def _build_artifact(
    gate: dict[str, Any],
    elapsed: float,
    backend: str,
    device: str | None,
    provenance_extra: dict[str, Any] | None,
    *,
    error: bool = False,
) -> dict[str, Any]:
    """Assemble the full JSON artifact: provenance + gate + timing."""
    host = collect_host_info()
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "provenance": {
            "git_commit": collect_git_commit(),
            "host_node": host["host_node"],
            "platform": host["platform"],
            "software": _collect_software_versions(),
            "backend": backend,
            "device": device,
            **(provenance_extra or {}),
        },
        "gate": gate,
        "wall_clock_seconds": round(elapsed, 3),
    }
    if error:
        artifact["outcome"] = "auto-apply-error"
    return artifact


def _build_synthetic_mosaic() -> MosaicGrid:
    """Tiny synthetic mosaic for the ``--smoke`` plumbing test (no CUDA).

    Mirrors the fixture in ``tests/test_tuning.py`` so the smoke run exercises
    the real :func:`auto_tune` path on the NumPy backend without needing a GPU.
    """
    import numpy as np

    rng = np.random.default_rng(7)
    tile_h, tile_w = 12, 12
    n_rows, n_cols, n_z = 4, 5, 4
    overlap_x = round(0.2 * tile_w)
    overlap_y = round(0.2 * tile_h)
    n_tiles = n_rows * n_cols
    flatfield = np.ones((tile_h, tile_w), dtype=np.float32)

    raw = np.zeros((n_z, n_rows * tile_h, n_cols * tile_w), dtype=np.float32)
    for z in range(n_z):
        tiles = rng.random((n_tiles, tile_h, tile_w)).astype(np.float32) + 0.5
        for r in range(n_rows):
            for c in range(n_cols - 1):
                shared = rng.random((tile_h, overlap_x)).astype(np.float32) + 0.5
                tiles[r * n_cols + c, :, -overlap_x:] = shared
                tiles[r * n_cols + c + 1, :, :overlap_x] = shared
        for r in range(n_rows - 1):
            for c in range(n_cols):
                shared = rng.random((overlap_y, tile_w)).astype(np.float32) + 0.5
                tiles[r * n_cols + c, -overlap_y:, :] = shared
                tiles[(r + 1) * n_cols + c, :overlap_y, :] = shared
        for r in range(n_rows):
            for c in range(n_cols):
                idx = r * n_cols + c
                raw[z, r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tiles[idx] * flatfield
    return MosaicGrid(raw, tile_shape=(tile_h, tile_w), overlap_fraction=0.2)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the D023 auto-apply gate and persist the full .gate dict (M008/S03, R057).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", metavar="ZARR", type=Path, help="Path to the input OME-Zarr mosaic grid (required unless --smoke)."
    )
    p.add_argument(
        "--output", metavar="JSON", required=True, type=Path, help="Write the full gate dict + provenance JSON here."
    )
    p.add_argument(
        "--smoke", action="store_true", help="Run on a tiny synthetic mosaic (NumPy, 2 trials) to prove the plumbing."
    )
    p.add_argument("--n-trials", metavar="N", type=int, default=50, help="Optuna trials (forwarded to auto_tune).")
    p.add_argument("--z-subsample", metavar="N", type=int, default=4, help="Z-levels per trial (forwarded to auto_tune).")
    p.add_argument(
        "--bounds-margin",
        metavar="FRAC",
        type=float,
        default=0.10,
        help="Relative near-optimal band width for recommend_bounds.",
    )
    p.add_argument("--seed", metavar="N", type=int, default=0, help="Random seed.")
    p.add_argument("--backend", default="torch", help="Array backend (torch | numpy).")
    p.add_argument("--device", default=None, help="Device (e.g. cuda:0); omit for CPU/NumPy.")
    p.add_argument("--max-tiles", metavar="N", type=int, default=64, help="Tiles per trial seam metric; 0 = all.")
    p.add_argument("--n-extra-rows", metavar="N", type=int, default=0, help="Leading rows to drop per tile.")
    p.add_argument("--overlap", metavar="FRAC", type=float, default=0.2, help="Physical tile-overlap fraction.")
    p.add_argument("--verbose", action="store_true", help="Forwarded to auto_tune.")
    p.add_argument("--label", default=None, help="Free-form provenance label (e.g. 'sub-22 slice 27').")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.smoke:
        mosaic = _build_synthetic_mosaic()
        n_trials, backend, device, max_tiles = 2, "numpy", None, 8
        z_subsample = 2
    else:
        if args.input is None:
            print("error: --input ZARR is required (or pass --smoke)", file=sys.stderr)
            return 2
        mosaic = MosaicGrid.from_ome_zarr(str(args.input), overlap_fraction=args.overlap)
        n_trials = args.n_trials
        z_subsample = args.z_subsample
        backend = args.backend
        device = args.device
        max_tiles = args.max_tiles if args.max_tiles > 0 else None

    provenance_extra = {"label": args.label, "n_trials": n_trials, "z_subsample": z_subsample}

    gate, artifact = run_validation(
        mosaic,
        n_trials=n_trials,
        z_subsample=z_subsample,
        margin=args.bounds_margin,
        seed=args.seed,
        backend=backend,
        device=device,
        max_tiles=max_tiles,
        n_extra_rows=args.n_extra_rows,
        verbose=args.verbose,
        provenance_extra=provenance_extra,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(artifact, fh, indent=2)

    verdict = gate.get("gate_verdict", "unknown")
    applied = gate.get("applied")
    fallback_reason = gate.get("fallback_reason")
    print(f"gate_verdict: {verdict}")
    print(f"applied: {applied}")
    if fallback_reason is not None:
        print(f"fallback_reason: {fallback_reason}")
    deltas = gate.get("deltas") or {}
    for metric in ("seam_l1", "seam_curvature"):
        if metric in deltas:
            d = deltas[metric]
            print(f"  {metric}: abs_delta={d.get('abs_delta'):+.6f} rel_delta={d.get('rel_delta'):+.6f}")
    print(f"wall_clock: {artifact['wall_clock_seconds']}s")
    print(f"Wrote gate artifact to '{args.output}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
