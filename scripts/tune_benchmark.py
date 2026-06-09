"""Expansive BaSiC tuning benchmark for linumpy pipeline timing.

Runs a high-quality Optuna search to find the best BaSiC hyperparameters
and reports detailed timing so the caller can size the optimisation step
in a 3-D reconstruction pipeline.

Usage (on the CUDA host)::

    PYTHONUNBUFFERED=1 uv run python scripts/tune_benchmark.py \\
        --input /path/to/mosaic.ome.zarr \\
        [--output-json results.json] [--db-url sqlite:///tune.db]

Outputs
-------
<output-json> (default: tune_benchmark_results.json in CWD)
    Best params, timing stats, per-trial convergence trace.
<db-url> (default: sqlite:///tune_benchmark.db in CWD)
    Optuna SQLite study (resumable).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, metavar="ZARR", help="Path to the OME-Zarr mosaic to tune on.")
    p.add_argument(
        "--output-json",
        default="tune_benchmark_results.json",
        metavar="JSON",
        help="Output path for the results JSON (default: %(default)s in CWD).",
    )
    p.add_argument(
        "--db-url",
        default="sqlite:///tune_benchmark.db",
        metavar="URL",
        help="Optuna storage URL (default: %(default)s in CWD).",
    )
    p.add_argument(
        "--study-name",
        default="linumpy-benchmark",
        metavar="NAME",
        help="Optuna study name (default: %(default)s).",
    )
    return p


# Tuning budget — "expansive" settings
N_TRIALS = 300
Z_SUBSAMPLE = 10  # from 55 z-levels (evenly spaced)
MAX_TILES = 256  # from 496 tiles total; 4x default 64
OBJECTIVE = "composite"
COMPOSITE_WEIGHTS = (1.0, 1.0)
SEED = 42
OVERLAP = 0.2

# Backend: torch+cuda:0 — the A6000 GPU cuts each BaSiC fit from ~18 s to ~2 s,
# making 300 trials feasible in ~2 hours instead of ~15.
# n_workers is forced to 1 for the torch backend (GPU contention);
# that also enables Optuna's MedianPruner to skip bad trials early.
BACKEND = "torch"
DEVICE = "cuda:0"
N_WORKERS = 1  # sequential + pruning (forced to 1 for torch backend)

# Extended search space (wider than defaults for a proper exploration)
SEARCH_SPACE: dict[str, list | tuple] = {
    "working_size": [64, 96, 128, 160, 192, 256],
    "l_s_divisor": (50.0, 10000.0),
    "l_d_divisor": (200.0, 20000.0),
    "epsilon": (0.001, 2.0),
    "estimate_darkfield": [True, False],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f} min"


def _single_z_pilot(mosaic, backend: str, device: str | None = None) -> float:
    """Time one BaSiC fit at working_size=128 (representative mid-point)."""
    from linum_basic.fit import make_model

    tiles = mosaic.iter_tiles(0)
    # Subsample to MAX_TILES for fair comparison with the actual tuning loop
    idx = np.linspace(0, len(tiles) - 1, min(MAX_TILES, len(tiles)), dtype=int)
    fit_tiles = tiles[idx]

    # Estimate dct_sum so we can set l_s / l_d properly
    from linum_basic.core import dct_energy

    dct_sum = dct_energy(fit_tiles.mean(axis=0))
    params = {
        "working_size": 128,
        "l_s": dct_sum / 800.0,
        "l_d": dct_sum / 2000.0,
        "epsilon": 0.1,
        "estimate_darkfield": True,
        "backend": backend,
    }
    if device is not None:
        params["device"] = device

    t0 = time.perf_counter()
    model = make_model(fit_tiles, params)
    model.prepare()
    model.run()
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _build_parser().parse_args()

    from linum_basic.mosaic import MosaicGrid
    from linum_basic.tuning import tune

    print("=" * 60)
    print("BaSiC Expansive Tuning Benchmark")
    print("=" * 60)
    print(f"zarr   : {args.input}")
    print(f"trials : {N_TRIALS}  z-sub : {Z_SUBSAMPLE}  tiles : {MAX_TILES}")
    print(f"obj    : {OBJECTIVE}  backend : {BACKEND}  seed : {SEED}")
    print()

    # ---- Load mosaic ----
    print("Loading mosaic...", flush=True)
    t_load = time.perf_counter()
    mosaic = MosaicGrid.from_ome_zarr(args.input, overlap_fraction=OVERLAP)
    t_load = time.perf_counter() - t_load
    print(
        f"  shape: {mosaic.array.shape}  "
        f"tiles: {mosaic.n_cols * mosaic.n_rows}  "
        f"z-levels: {mosaic.n_z}  "
        f"seam pairs: {len(mosaic.seam_pairs())}"
    )
    print(f"  load: {_fmt_time(t_load)}")
    print()

    # ---- Pilot ----
    device_str = DEVICE if BACKEND == "torch" else None
    print(
        f"Pilot: timing one BaSiC fit (working_size=128, darkfield=True, backend={BACKEND})...",
        flush=True,
    )
    t_single = _single_z_pilot(mosaic, BACKEND, device_str)
    t_est_seq = t_single * Z_SUBSAMPLE
    t_est_total_no_prune = t_est_seq * N_TRIALS
    # With MedianPruner ~40-60 % of trials are pruned after step 1-2;
    # assume on average pruned trials run 25 % of z-levels.
    pruned_fraction = 0.50
    t_est_prune = N_TRIALS * (1 - pruned_fraction) * t_est_seq + N_TRIALS * pruned_fraction * t_single * 2
    print(f"  single z-level: {_fmt_time(t_single)}")
    print(f"  per trial (sequential, {Z_SUBSAMPLE} z): {_fmt_time(t_est_seq)}")
    print(f"  estimated total without pruning: {_fmt_time(t_est_total_no_prune)}")
    print(f"  estimated total with ~50% pruning: {_fmt_time(t_est_prune)}")
    print()

    # ---- Main tuning run ----
    print(f"Starting {N_TRIALS}-trial optimisation...", flush=True)
    t_tune_start = time.perf_counter()

    tune_kwargs: dict = {
        "n_trials": N_TRIALS,
        "z_subsample": Z_SUBSAMPLE,
        "max_tiles": MAX_TILES,
        "search_space": SEARCH_SPACE,
        "objective": OBJECTIVE,
        "composite_weights": COMPOSITE_WEIGHTS,
        "seed": SEED,
        "n_workers": N_WORKERS,
        "backend": BACKEND,
        "storage": args.db_url,
        "study_name": args.study_name,
        "run_full_fit": False,
        "verbose": True,
    }
    if BACKEND == "torch" and DEVICE is not None:
        tune_kwargs["device"] = DEVICE

    result = tune(mosaic, **tune_kwargs)

    t_tune_total = time.perf_counter() - t_tune_start

    # ---- Report ----
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  total time        : {_fmt_time(t_tune_total)}")
    print(f"  avg per trial     : {_fmt_time(t_tune_total / N_TRIALS)}")
    print(f"  best ({OBJECTIVE:10s}) : {result.best_value:.6f}")
    print("  best params:")
    for k, v in result.best_params.items():
        if isinstance(v, float):
            print(f"    {k:<20s}: {v:.5g}")
        else:
            print(f"    {k:<20s}: {v}")

    # ---- Convergence analysis ----
    convergence: dict = {}
    n_completed = 0
    n_pruned = 0
    first_best_trial = -1

    if result.trials_df is not None:
        df = result.trials_df
        completed = df[df["state"] == "COMPLETE"]
        pruned = df[df["state"] == "PRUNED"]
        n_completed = len(completed)
        n_pruned = len(pruned)
        print(f"\n  completed trials  : {n_completed}")
        print(f"  pruned trials     : {n_pruned} ({100 * n_pruned / N_TRIALS:.0f}%)")

        # Best-value trace (over completed trials in submission order)
        vals = completed["value"].values
        best_trace = np.minimum.accumulate(vals)
        first_best_trial = int(np.argmin(best_trace == best_trace[-1]))
        print(f"  best found at trial #{first_best_trial} (of {n_completed} completed)")

        # Convergence: how much improvement after 50 / 100 / 150 trials?
        for milestone in [50, 100, 150, 200, 250]:
            if milestone < n_completed:
                frac_imp = (best_trace[0] - best_trace[milestone]) / (best_trace[0] + 1e-9)
                print(f"  improvement at trial {milestone:3d}: {100 * frac_imp:.1f}%")

        convergence = {
            "trial_index": list(range(len(best_trace))),
            "best_value_trace": best_trace.tolist(),
        }
    else:
        print("  (pandas not available — no convergence trace)")

    # ---- Save JSON ----
    output = {
        "dataset": args.input,
        "config": {
            "n_trials": N_TRIALS,
            "z_subsample": Z_SUBSAMPLE,
            "max_tiles": MAX_TILES,
            "objective": OBJECTIVE,
            "composite_weights": list(COMPOSITE_WEIGHTS),
            "search_space": {k: list(v) for k, v in SEARCH_SPACE.items()},
            "backend": BACKEND,
            "n_workers": N_WORKERS,
            "seed": SEED,
        },
        "timing": {
            "load_seconds": t_load,
            "single_z_fit_seconds": t_single,
            "total_seconds": t_tune_total,
            "total_minutes": t_tune_total / 60,
            "avg_seconds_per_trial": t_tune_total / N_TRIALS,
            "estimated_seq_no_prune_seconds": t_est_total_no_prune,
        },
        "stats": {
            "n_completed": n_completed,
            "n_pruned": n_pruned,
            "pruned_fraction": n_pruned / N_TRIALS if N_TRIALS else 0,
            "first_best_trial": first_best_trial,
        },
        "best_value": result.best_value,
        "best_params": {k: float(v) if isinstance(v, float) else v for k, v in result.best_params.items()},
        "convergence": convergence,
    }

    with Path(args.output_json).open("w") as fh:
        json.dump(output, fh, indent=2)

    print(f"\nResults saved to {args.output_json}")
    print(f"Optuna study DB  : {args.db_url}")


if __name__ == "__main__":
    main()
