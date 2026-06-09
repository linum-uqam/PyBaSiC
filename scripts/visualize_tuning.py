"""Visualize BaSiC hyperparameter tuning results on an OME-Zarr mosaic.

Runs Optuna tuning on the mosaic and produces two output figures:

  1. ``tune_optimization.png``
     Optimization history (seam-L1 per trial, best-so-far line) and a table
     of the best parameters found.

  2. ``tune_seam_comparison.png``
     Tile-level seam quality at ``--z-inspect``: raw tiles, BaSiC defaults,
     and tuned params side-by-side with seam-L1 annotations.

Usage::

    uv run python scripts/visualize_tuning.py --input /path/to/mosaic.ome.zarr
    uv run python scripts/visualize_tuning.py --input /path/to/mosaic.ome.zarr --n-trials 50 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from linum_basic import viz

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        metavar="ZARR",
        help="Path to the OME-Zarr mosaic.",
    )
    p.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory for output PNGs and JSON (default: current directory).",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=30,
        metavar="N",
        help="Number of Optuna trials (default: %(default)s).",
    )
    p.add_argument(
        "--z-subsample",
        type=int,
        default=8,
        metavar="N",
        help="Z-levels evaluated per trial (default: %(default)s).",
    )
    p.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        metavar="FRAC",
        help="Tile overlap fraction (default: %(default)s).",
    )
    p.add_argument(
        "--z-inspect",
        type=int,
        default=27,
        metavar="Z",
        help="Z-level used for the seam-comparison figure (default: %(default)s).",
    )
    p.add_argument(
        "--n-extra",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Galvo-return rows to exclude from BaSiC fitting (default: %(default)s). "
            "Rows 0..(N-1) are masked before fitting and edge-extended in the output."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Optuna random seed (default: %(default)s).",
    )
    p.add_argument(
        "--storage",
        default=None,
        metavar="URL",
        help="Optuna storage URL, e.g. 'sqlite:///tune.db'. None uses in-memory storage.",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        metavar="N",
        help="Threads for parallel z-level evaluation per trial (default: n_cpus - 2).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show BaSiC and Optuna progress.",
    )
    return p


# ---------------------------------------------------------------------------
# BaSiC helpers
# ---------------------------------------------------------------------------


def _fit(tiles: np.ndarray, params: dict, n_extra: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit BaSiC with galvo-row masking and return (flatfield, darkfield)."""
    from linum_basic.fit import make_model

    masked = tiles.copy()
    if n_extra > 0:
        masked[:, :n_extra, :] = masked[:, n_extra : n_extra + 1, :]
    model = make_model(masked, params)
    model.prepare()
    model.run()
    return model.get_flatfield(), model.get_darkfield()


def _apply(tiles: np.ndarray, ff: np.ndarray, df: np.ndarray) -> np.ndarray:
    return (tiles - df) / np.clip(ff, 1e-6, None)


# ---------------------------------------------------------------------------
# Figure 1 — optimization history
# ---------------------------------------------------------------------------


def _build_optimization_figure(result, out_path: Path) -> None:
    """Optimization history and best-parameter table."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    viz.set_theme()

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.8, 1], wspace=0.35)

    ax_hist = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    # --- history ---
    df = result.trials_df
    if df is not None and "value" in df.columns and "state" in df.columns:
        completed = df[df["state"] == "COMPLETE"].copy().reset_index(drop=True)
        pruned = df[df["state"] == "PRUNED"].copy().reset_index(drop=True)

        if not completed.empty:
            ax_hist.scatter(
                completed.index,
                completed["value"],
                c="steelblue",
                s=35,
                alpha=0.75,
                zorder=3,
                label="completed",
            )
            best_so_far = completed["value"].cummin()
            ax_hist.step(
                completed.index,
                best_so_far,
                where="post",
                color="firebrick",
                lw=2,
                zorder=4,
                label="best so far",
            )
        if not pruned.empty:
            ax_hist.scatter(
                pruned.index,
                [float(df["value"].dropna().max()) * 1.05] * len(pruned),
                c="lightgray",
                s=25,
                alpha=0.6,
                marker="x",
                zorder=2,
                label=f"pruned ({len(pruned)})",
            )
        ax_hist.axhline(result.best_value, color="firebrick", ls="--", lw=1, alpha=0.4)
        ax_hist.set_xlabel("Completed trial index", fontsize=10)
        ax_hist.set_ylabel("Seam L1 (normalised)", fontsize=10)
        ax_hist.set_title("Optuna optimization history", fontsize=11)
        ax_hist.legend(fontsize=9)
    else:
        ax_hist.text(0.5, 0.5, "No trial data available", ha="center", va="center", transform=ax_hist.transAxes)

    # --- best params table ---
    bp = result.best_params
    param_fmt: dict[str, str] = {
        "working_size": str(int(bp.get("working_size", "—"))),
        "l_s": f"{bp.get('l_s', 0):.5f}",
        "l_d": f"{bp.get('l_d', 0):.5f}",
        "epsilon": f"{bp.get('epsilon', 0):.4f}",
        "estimate_darkfield": str(bp.get("estimate_darkfield", "—")),
    }
    rows = [[k, v] for k, v in param_fmt.items() if k in bp]
    rows.append(["seam L1", f"{result.best_value:.6f}"])

    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=["Parameter", "Best value"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.65)
    ax_tbl.set_title("Best trial parameters", fontsize=11, pad=12)

    fig.suptitle("BaSiC Hyperparameter Tuning", fontsize=13, fontweight="bold", y=1.01)
    viz.save_figure(fig, out_path, dpi=150)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — seam comparison
# ---------------------------------------------------------------------------


def _montage_strip(tiles: np.ndarray, sp, n_tiles: int = 6) -> np.ndarray:
    """Return a horizontal strip of `n_tiles` tiles centred on seam pair `sp`."""
    # Use the row containing idx_a; pick tiles around it
    n = len(tiles)
    start = max(0, sp.idx_a - n_tiles // 2)
    end = min(n, start + n_tiles)
    strip = tiles[start:end]
    return np.concatenate(strip, axis=1)  # (height, n_tiles * width)


def _build_seam_figure(
    mosaic,
    default_params: dict,
    tuned_params: dict,
    z_inspect: int,
    n_extra: int,
    out_path: Path,
) -> None:
    """Seam-quality comparison: raw | default | tuned at z_inspect."""
    import matplotlib.pyplot as plt

    from linum_basic.metrics import evaluate_correction

    viz.set_theme()

    print(f"  Fitting default params at z={z_inspect} …")
    tiles_raw = mosaic.iter_tiles(z_inspect)
    seam_pairs = mosaic.seam_pairs()

    ff_def, df_def = _fit(tiles_raw, default_params, n_extra)
    tiles_def = _apply(tiles_raw, ff_def, df_def)
    m_def = evaluate_correction(tiles_raw, ff_def, df_def, seam_pairs)

    print(f"  Fitting tuned params at z={z_inspect} …")
    ff_tun, df_tun = _fit(tiles_raw, tuned_params, n_extra)
    tiles_tun = _apply(tiles_raw, ff_tun, df_tun)
    m_tun = evaluate_correction(tiles_raw, ff_tun, df_tun, seam_pairs)

    from linum_basic.metrics import seam_l1

    sl1_raw = seam_l1(tiles_raw, seam_pairs)

    # Pick a seam pair with good contrast for the strip panel
    sp = seam_pairs[len(seam_pairs) // 2]
    n_strip = 6

    def _strip(t: np.ndarray) -> np.ndarray:
        start = max(0, sp.idx_a - n_strip // 2)
        end = min(len(t), start + n_strip)
        return np.concatenate(t[start:end], axis=1)

    strip_raw = _strip(tiles_raw)
    strip_def = _strip(tiles_def)
    strip_tun = _strip(tiles_tun)

    # Shared colour scale (percentile clip of raw)
    vmin = float(np.percentile(strip_raw, 1))
    vmax = float(np.percentile(strip_raw, 99))

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.suptitle(f"Seam quality at z={z_inspect}", fontsize=13, fontweight="bold")

    cmap = viz.INTENSITY_CMAP

    def _seam_l1_label(v: float) -> str:
        return f"seam L1 = {v:.5f}"

    # --- tile strips ---
    for ax, strip, title, sl1 in [
        (axes[0, 0], strip_raw, "Raw (uncorrected)", sl1_raw),
        (axes[1, 0], strip_def, "Default BaSiC params", m_def["seam_l1"]),
        (axes[2, 0], strip_tun, "Tuned params", m_tun["seam_l1"]),
    ]:
        ax.imshow(strip, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"{title}\n{_seam_l1_label(sl1)}", fontsize=10)
        ax.axis("off")

    # --- flat-fields ---
    for ax, ff, title in [
        (axes[0, 1], np.ones_like(ff_def), "Flat-field (identity — no correction)"),
        (axes[1, 1], ff_def, "Flat-field — default"),
        (axes[2, 1], ff_tun, "Flat-field — tuned"),
    ]:
        im = ax.imshow(ff, cmap=viz.FLATFIELD_CMAP, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correction factor")

    # --- summary bar chart in axes[0,1] (replace identity flat-field) ---
    axes[0, 1].remove()
    ax_bar = fig.add_subplot(3, 2, 2)
    labels = ["Raw", "Default", "Tuned"]
    values = [sl1_raw, m_def["seam_l1"], m_tun["seam_l1"]]
    colors = ["#aec6cf", "#8fa8d6", "#4e79a7"]
    bars = ax_bar.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    ax_bar.set_ylabel("Seam L1 (normalised, lower is better)", fontsize=9)
    ax_bar.set_title("Seam L1 comparison", fontsize=10)
    for bar, val in zip(bars, values, strict=True):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.01,
            f"{val:.5f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    viz.save_figure(fig, out_path, dpi=150)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from linum_basic.mosaic import MosaicGrid
    from linum_basic.tuning import tune

    print(f"Loading mosaic from {args.input} …")
    mosaic = MosaicGrid.from_ome_zarr(str(args.input), overlap_fraction=args.overlap)
    print(f"  shape={mosaic.array.shape}  tile={mosaic.tile_shape}  grid={mosaic.n_rows}x{mosaic.n_cols}  n_z={mosaic.n_z}")

    print(f"\nRunning Optuna tuning: {args.n_trials} trials, {args.z_subsample} z-levels/trial …")
    result = tune(
        mosaic,
        n_trials=args.n_trials,
        z_subsample=args.z_subsample,
        seed=args.seed,
        storage=args.storage,
        n_workers=args.n_workers,
        run_full_fit=False,
        verbose=args.verbose,
    )
    print(f"  Best seam L1: {result.best_value:.6f}")
    print(f"  Best params: {result.best_params}")

    # Save best params as JSON
    json_path = out_dir / "tune_best_params.json"
    with json_path.open("w") as fh:
        json.dump(result.best_params, fh, indent=2)
    print(f"Saved: {json_path}")

    print("\nBuilding optimization history figure …")
    _build_optimization_figure(result, out_dir / "tune_optimization.png")

    print("\nBuilding seam comparison figure …")
    # Default params: let BaSiC auto-tune l_s / l_d (pass nothing)
    default_params: dict = {}
    _build_seam_figure(
        mosaic,
        default_params,
        result.best_params,
        z_inspect=min(args.z_inspect, mosaic.n_z - 1),
        n_extra=args.n_extra,
        out_path=out_dir / "tune_seam_comparison.png",
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
