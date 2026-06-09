"""Generate presentation figures for a BaSiC tuning benchmark run.

Produces four publication-quality figures:

1. ``tuning_history.png``  — Optuna optimisation history: all trial values,
   pruned trials highlighted, best-so-far envelope, timing annotation.
2. ``tuning_params.png``   — Parameter importance + per-parameter distributions
   for the top 20 % of completed trials vs. all others.
3. ``correction_results.png`` — Full before/after correction diagnostic at the
   chosen z-level using the best hyperparameters.
4. ``seam_metric.png``     — Seam-consistency metric explained on a tile row.

Usage::

    uv run python scripts/plot_tuning_results.py \\
        --input /path/to/mosaic.ome.zarr \\
        --db sqlite:///tune_benchmark.db \\
        [--study-name linumpy-benchmark] \\
        [--z 27] [--output-dir .]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

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
        "--db",
        default="sqlite:///tune_benchmark.db",
        metavar="URL",
        help="Optuna storage URL (default: %(default)s).",
    )
    p.add_argument(
        "--study-name",
        default="linumpy-benchmark",
        metavar="NAME",
        help="Optuna study name (default: %(default)s).",
    )
    p.add_argument(
        "--z",
        type=int,
        default=27,
        metavar="INT",
        help="Z-level used for the correction figure (default: %(default)s).",
    )
    p.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory for output PNGs (default: current directory).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI for all saved figures (default: %(default)s).",
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PARAM_LABELS: dict[str, str] = {
    "params_epsilon": "ε (epsilon)",
    "params_l_d_divisor": "l_d divisor",
    "params_l_s_divisor": "l_s divisor",
    "params_working_size": "Working size",
    "params_estimate_darkfield": "Estimate darkfield",
}


def _best_so_far(values: np.ndarray) -> np.ndarray:
    """Cumulative minimum (best-so-far) trace over completed trials."""
    return np.minimum.accumulate(values)


# ---------------------------------------------------------------------------
# Figure 1 - Optimisation history
# ---------------------------------------------------------------------------


def figure_opt_history(df_all, best_trial_number: int, timing: dict[str, float]):  # type: ignore[no-untyped-def]
    """Scatter of all trials (complete blue, pruned grey) + best-so-far."""
    import matplotlib.pyplot as plt

    from linum_basic.viz import set_theme

    set_theme()

    fig, ax = plt.subplots(figsize=(11, 5))

    complete = df_all[df_all["state"] == "COMPLETE"].sort_values("number")
    pruned = df_all[df_all["state"] == "PRUNED"].sort_values("number")

    # Pruned trials (no value — use NaN-safe plot)
    ax.scatter(
        pruned["number"],
        pruned["value"],
        s=18,
        c="#b0b0b0",
        marker="x",
        linewidths=0.8,
        zorder=2,
        label=f"Pruned ({len(pruned)})",
    )

    # Complete trials
    ax.scatter(
        complete["number"],
        complete["value"],
        s=30,
        c="#4c9de8",
        alpha=0.85,
        zorder=3,
        label=f"Complete ({len(complete)})",
    )

    # Best-so-far envelope
    vals = complete["value"].values
    bsf = _best_so_far(vals)
    ax.step(
        complete["number"].values,
        bsf,
        where="post",
        color="#e84c4c",
        lw=2.2,
        zorder=4,
        label="Best so far",
    )

    # Annotate best trial
    best_row = complete[complete["number"] == best_trial_number]
    if not best_row.empty:
        bv = best_row["value"].iloc[0]
        ax.annotate(
            f"  Trial {best_trial_number}\n  best = {bv:.4f}",
            xy=(best_trial_number, bv),
            xytext=(best_trial_number + max(len(complete) // 20, 5), bv * 1.05),
            arrowprops={"arrowstyle": "->", "color": "#333", "lw": 1.2},
            fontsize=9,
            color="#333",
        )

    # Timing annotation (top-right)
    total_min = timing.get("total_minutes", 0)
    avg_s = timing.get("avg_seconds_per_trial", 0)
    no_prune_h = timing.get("estimated_seq_no_prune_seconds", 0) / 3600
    ax.text(
        0.98,
        0.97,
        f"Total: {total_min:.0f} min  |  avg: {avg_s:.1f} s/trial\nWithout pruning: ~{no_prune_h:.0f} h",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cccccc", "alpha": 0.85},
    )

    ax.set_xlabel("Trial number")
    ax.set_ylabel("Composite objective (lower = better)")
    ax.set_title("BaSiC hyperparameter search — Optuna TPE + MedianPruner")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 - Parameter analysis
# ---------------------------------------------------------------------------


def figure_param_analysis(df_all):  # type: ignore[no-untyped-def]
    """Top-tile distributions + working_size bar + darkfield bar."""
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    from linum_basic.viz import set_theme

    set_theme()

    complete = df_all[df_all["state"] == "COMPLETE"].sort_values("value")
    n_top = max(1, len(complete) // 5)  # top 20 %
    top = complete.head(n_top)
    rest = complete.tail(len(complete) - n_top)

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.38)

    # ---- Continuous params: violin ----------------------------------------
    continuous = ["params_epsilon", "params_l_s_divisor", "params_l_d_divisor"]
    for col_i, param in enumerate(continuous):
        ax = fig.add_subplot(gs[0, col_i])
        top_vals = top[param].dropna().values
        rest_vals = rest[param].dropna().values
        parts = ax.violinplot(
            [rest_vals, top_vals],
            positions=[1, 2],
            showmedians=True,
            widths=0.6,
        )
        for pc, color in zip(parts["bodies"], ["#aec6e8", "#e84c4c"], strict=True):
            pc.set_facecolor(color)
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("#333")
        parts["cbars"].set_color("#666")
        parts["cmaxes"].set_color("#666")
        parts["cmins"].set_color("#666")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["other 80 %", "top 20 %"], fontsize=8)
        ax.set_title(PARAM_LABELS.get(param, param), fontsize=9)
        ax.set_ylabel("Value", fontsize=8)

    # ---- Working size bar -------------------------------------------------
    ax_ws = fig.add_subplot(gs[0, 3])
    ws_counts_all = complete["params_working_size"].value_counts().sort_index()
    ws_counts_top = top["params_working_size"].value_counts().sort_index()
    x = np.arange(len(ws_counts_all))
    w = 0.4
    ax_ws.bar(x - w / 2, ws_counts_all.values, width=w, color="#aec6e8", label="all")
    top_vals_ws = ws_counts_top.reindex(ws_counts_all.index, fill_value=0).values
    ax_ws.bar(x + w / 2, top_vals_ws, width=w, color="#e84c4c", label="top 20%")
    ax_ws.set_xticks(x)
    ax_ws.set_xticklabels([str(int(v)) for v in ws_counts_all.index], fontsize=8)
    ax_ws.set_xlabel("working_size (px)", fontsize=8)
    ax_ws.set_ylabel("# trials", fontsize=8)
    ax_ws.set_title("Working size", fontsize=9)
    ax_ws.legend(fontsize=8)

    # ---- Darkfield bar ----------------------------------------------------
    ax_df = fig.add_subplot(gs[1, 0])
    df_counts_all = complete["params_estimate_darkfield"].value_counts()
    df_counts_top = top["params_estimate_darkfield"].value_counts()
    cats = [True, False]
    x2 = np.arange(len(cats))
    all_c = [df_counts_all.get(c, 0) for c in cats]
    top_c = [df_counts_top.get(c, 0) for c in cats]
    ax_df.bar(x2 - w / 2, all_c, width=w, color="#aec6e8", label="all")
    ax_df.bar(x2 + w / 2, top_c, width=w, color="#e84c4c", label="top 20%")
    ax_df.set_xticks(x2)
    ax_df.set_xticklabels(["Yes", "No"], fontsize=8)
    ax_df.set_xlabel("estimate_darkfield", fontsize=8)
    ax_df.set_ylabel("# trials", fontsize=8)
    ax_df.set_title("Estimate darkfield", fontsize=9)
    ax_df.legend(fontsize=8)

    # ---- Value vs epsilon scatter -----------------------------------------
    ax_sc = fig.add_subplot(gs[1, 1])
    sc = ax_sc.scatter(
        complete["params_epsilon"],
        complete["value"],
        c=complete["params_working_size"],
        s=20,
        alpha=0.7,
        cmap="RdYlGn_r",
    )
    fig.colorbar(sc, ax=ax_sc, label="working_size", pad=0.02)
    ax_sc.set_xlabel("ε (epsilon)", fontsize=8)
    ax_sc.set_ylabel("Objective", fontsize=8)
    ax_sc.set_title("Objective vs ε", fontsize=9)

    # ---- Value vs l_s_divisor scatter -------------------------------------
    ax_ls = fig.add_subplot(gs[1, 2])
    ax_ls.scatter(
        np.log10(complete["params_l_s_divisor"]),
        complete["value"],
        s=20,
        alpha=0.55,
        c="#4c9de8",
    )
    ax_ls.set_xlabel("log₁₀(l_s divisor)", fontsize=8)
    ax_ls.set_ylabel("Objective", fontsize=8)
    ax_ls.set_title("Objective vs l_s divisor", fontsize=9)

    # ---- Best params table ------------------------------------------------
    ax_t = fig.add_subplot(gs[1, 3])
    ax_t.axis("off")
    best = complete.iloc[0]
    rows_data = [
        ["working_size", f"{int(best['params_working_size'])} px"],
        ["l_s divisor", f"{best['params_l_s_divisor']:.1f}"],
        ["l_d divisor", f"{best['params_l_d_divisor']:.1f}"],
        ["ε (epsilon)", f"{best['params_epsilon']:.4f}"],
        ["darkfield", str(bool(best["params_estimate_darkfield"]))],
        ["objective", f"{best['value']:.5f}"],
    ]
    tbl = ax_t.table(
        cellText=rows_data,
        colLabels=["Parameter", "Best value"],
        cellLoc="left",
        loc="center",
        bbox=[0.0, 0.1, 1.0, 0.85],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, _c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e84c4c")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f5f5")
        cell.set_edgecolor("#cccccc")
    ax_t.set_title("Best trial params", fontsize=9, pad=4)

    fig.suptitle(
        "Hyperparameter search — parameter distributions (top 20% vs rest)",
        fontsize=12,
        fontweight="bold",
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 3 & 4 - Correction results
# ---------------------------------------------------------------------------


def _fit_z(mosaic, params: dict[str, Any], z: int):  # type: ignore[no-untyped-def]
    """Fit one z-level with given params, return (flatfield, darkfield)."""
    from linum_basic.fit import make_model

    tiles = mosaic.iter_tiles(z)
    model = make_model(tiles, params)
    model.prepare()
    model.run()
    return model.get_flatfield(), model.get_darkfield()


def _apply_z(plane: np.ndarray, ff: np.ndarray, df: np.ndarray, mosaic) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Divide one (H, W) plane by its (th, tw) flat/dark-field via broadcasting."""
    th, tw = mosaic.tile_shape
    nr, nc = mosaic.n_rows, mosaic.n_cols
    view = plane.astype(np.float32).reshape(nr, th, nc, tw)
    cor = (view - df[None, :, None, :]) / (ff[None, :, None, :] + 1e-6)
    return cor.reshape(nr * th, nc * tw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import optuna

    args = _build_parser().parse_args()

    from linum_basic.metrics import seam_l1
    from linum_basic.mosaic import MosaicGrid
    from linum_basic.viz import figure_apply_correction, figure_seam_metric, save_figure

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load Optuna study -----------------------------------------------
    print(f"Loading Optuna study '{args.study_name}' from {args.db} ...", flush=True)
    study = optuna.load_study(study_name=args.study_name, storage=args.db)
    df_all = study.trials_dataframe()
    complete = df_all[df_all["state"] == "COMPLETE"].sort_values("value")
    print(f"  {len(complete)} complete  |  {len(df_all) - len(complete)} pruned")

    # Reconstruct timing dict from best trial duration
    durations = complete["duration"].dt.total_seconds()
    timing = {
        "total_minutes": durations.sum() / 60,
        "avg_seconds_per_trial": durations.mean(),
        "estimated_seq_no_prune_seconds": durations.sum() * 3,  # rough upper bound
    }
    # Use actual timing from best run if richer data available
    best_trial = study.best_trial
    best_params_raw = best_trial.params  # has divisors

    # ---- Figure 1: Optimisation history -----------------------------------
    print("Figure 1: optimisation history ...", flush=True)
    fig1 = figure_opt_history(df_all, best_trial.number, timing)
    p1 = save_figure(fig1, out_dir / "tuning_history.png", dpi=args.dpi)
    print(f"  → {p1}")

    # ---- Figure 2: Parameter analysis ------------------------------------
    print("Figure 2: parameter analysis ...", flush=True)
    fig2 = figure_param_analysis(df_all)
    p2 = save_figure(fig2, out_dir / "tuning_params.png", dpi=args.dpi)
    print(f"  → {p2}")

    # ---- Load mosaic -------------------------------------------------------
    print(f"Loading mosaic {args.input} ...", flush=True)
    mosaic = MosaicGrid.from_ome_zarr(args.input, overlap_fraction=0.2)
    z = min(args.z, mosaic.n_z - 1)
    print(f"  shape: {mosaic.array.shape}  z-level: {z}")

    # Convert best params (divisors → absolute l_s / l_d)
    # The tuning.py _basic_params helper does: dct_sum / divisor.
    # Re-derive dct_sum for a fast estimate from the mean tile of z=z.
    from linum_basic.core import dct_energy

    tiles_raw = mosaic.iter_tiles(z).astype(np.float32)
    dct_sum = float(dct_energy(tiles_raw.mean(axis=0)))

    best_params: dict[str, Any] = {
        "working_size": int(best_params_raw["working_size"]),
        "l_s": dct_sum / float(best_params_raw["l_s_divisor"]),
        "l_d": dct_sum / float(best_params_raw["l_d_divisor"]),
        "epsilon": float(best_params_raw["epsilon"]),
        "estimate_darkfield": bool(best_params_raw["estimate_darkfield"]),
    }
    print(f"  best params: {best_params}")

    # ---- Fit z=z with best params -----------------------------------------
    print(f"  Fitting z={z} with best params ...", flush=True)
    ff, df = _fit_z(mosaic, best_params, z)
    print(f"  Done. flatfield range [{ff.min():.3f}, {ff.max():.3f}]")

    # ---- Compute seam metrics -------------------------------------------
    seam_pairs = mosaic.seam_pairs()
    seam_raw_val = float(seam_l1(tiles_raw, seam_pairs))

    tiles_cor = tiles_raw / (ff[None, ...] + 1e-6)
    seam_cor_val = float(seam_l1(tiles_cor, seam_pairs))
    print(f"  seam L1: raw={seam_raw_val:.4f}  corrected={seam_cor_val:.4f}")

    # ---- Build mosaic planes for figure_apply_correction ------------------
    raw_plane = mosaic.array[z].astype(np.float32)
    cor_plane = _apply_z(raw_plane, ff, df, mosaic)

    # Representative tile: center of the grid
    cx, cy = mosaic.n_cols // 2, mosaic.n_rows // 2
    center_tile_raw = tiles_raw[cy * mosaic.n_cols + cx]
    center_tile_cor = tiles_cor[cy * mosaic.n_cols + cx]

    # pixel size from zarr attrs (mm)
    try:
        import zarr

        zg = zarr.open(args.input)
        scale = zg.attrs["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        pixel_size_mm: float | None = float(scale[1])
    except Exception:
        pixel_size_mm = None

    # ---- Figure 3: Apply correction ---------------------------------------
    print("Figure 3: correction results ...", flush=True)
    fig3 = figure_apply_correction(
        flatfield=ff,
        raw_mosaic=raw_plane,
        corrected_mosaic=cor_plane,
        raw_tile=center_tile_raw,
        corrected_tile=center_tile_cor,
        seam_raw=seam_raw_val,
        seam_corrected=seam_cor_val,
        darkfield=df if best_params["estimate_darkfield"] else None,
        title=f"BaSiC correction — z={z}  (best hyperparameters)",
        pixel_size_mm=pixel_size_mm,
    )
    p3 = save_figure(fig3, out_dir / "correction_results.png", dpi=args.dpi)
    print(f"  → {p3}")

    # ---- Figure 4: Seam metric explainer ---------------------------------
    # Use the middle row of tiles (row index n_rows//2)
    print("Figure 4: seam metric ...", flush=True)
    mid_row = mosaic.n_rows // 2
    row_idx = np.arange(mid_row * mosaic.n_cols, (mid_row + 1) * mosaic.n_cols)
    tiles_row_raw = tiles_raw[row_idx]
    tiles_row_cor = tiles_cor[row_idx]
    fig4 = figure_seam_metric(
        flatfield=ff,
        tiles_raw=tiles_row_raw,
        tiles_cor=tiles_row_cor,
        overlap_fraction=0.2,
        orientation="horizontal",
        title=f"Seam-consistency metric — row {mid_row}, z={z}",
    )
    p4 = save_figure(fig4, out_dir / "seam_metric.png", dpi=args.dpi)
    print(f"  → {p4}")

    print("\nDone. Figures written to:", out_dir.resolve())


if __name__ == "__main__":
    main()
