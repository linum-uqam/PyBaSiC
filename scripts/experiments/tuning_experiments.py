"""Empirical experiments for BaSiC tuning improvements (throwaway harness).

Two experiments, selectable via --experiment:

  metric   Sanity-check the seam_l1 objective. Is it gameable / does it agree
           with the scale-invariant Pearson seam metric? Compares the current
           normalisation (divide by mean(|corrected|)) against a fixed
           raw-mean normalisation, under flat-field scaling and perturbation.

  tiles    Find the point at which using fewer tiles to *fit* BaSiC degrades
           the correction quality measured on ALL tiles. Sweeps the number of
           tiles used for fitting and reports seam_l1 / pearson / runtime.

Run::

    uv run python scripts/experiments/tuning_experiments.py --experiment metric
    uv run python scripts/experiments/tuning_experiments.py --experiment tiles
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from linum_basic.core import DEFAULT_L_D_DIVISOR, DEFAULT_L_S_DIVISOR, dct_energy
from linum_basic.fit import make_model
from linum_basic.metrics import seam_pearson
from linum_basic.mosaic import MosaicGrid, SeamPair

# ---------------------------------------------------------------------------
# Metric variants
# ---------------------------------------------------------------------------


def _seam_abs_mean(tiles: np.ndarray, seam_pairs: list[SeamPair]) -> float:
    """Mean absolute seam difference (un-normalised numerator of seam_l1)."""
    if not seam_pairs:
        return 0.0
    diffs = []
    for sp in seam_pairs:
        a = tiles[sp.idx_a][sp.slice_a].ravel()
        b = tiles[sp.idx_b][sp.slice_b].ravel()
        diffs.append(float(np.abs(a - b).mean()))
    return float(np.mean(diffs))


def seam_l1_corrected_norm(tiles: np.ndarray, seam_pairs: list[SeamPair]) -> float:
    """Current implementation: normalise by mean(|corrected|)."""
    num = _seam_abs_mean(tiles, seam_pairs)
    return num / (float(np.mean(np.abs(tiles))) + 1e-9)


def seam_l1_raw_norm(tiles: np.ndarray, seam_pairs: list[SeamPair], raw_mean: float) -> float:
    """Alternative: normalise by a FIXED reference (raw tile mean)."""
    num = _seam_abs_mean(tiles, seam_pairs)
    return num / (raw_mean + 1e-9)


def seam_l1_relative(tiles: np.ndarray, seam_pairs: list[SeamPair]) -> float:
    """Per-pair relative seam error (scale-invariant, physical).

    For each seam, ``mean|a-b| / mean((|a|+|b|)/2)`` — the mismatch relative
    to the local brightness of that seam — then averaged over pairs.  A
    constant gain offset between two tiles produces the same relative error
    regardless of absolute intensity, and a bright region elsewhere cannot
    dilute the score (each seam is self-referenced).
    """
    if not seam_pairs:
        return 0.0
    rels: list[float] = []
    for sp in seam_pairs:
        a = tiles[sp.idx_a][sp.slice_a].ravel()
        b = tiles[sp.idx_b][sp.slice_b].ravel()
        local = (np.abs(a) + np.abs(b)).mean() / 2.0
        rels.append(float(np.abs(a - b).mean()) / (float(local) + 1e-9))
    return float(np.mean(rels))


def _correct(tiles: np.ndarray, flat: np.ndarray, dark: np.ndarray) -> np.ndarray:
    return (tiles.astype(np.float32) - dark[np.newaxis]) / (flat[np.newaxis] + 1e-6)


# ---------------------------------------------------------------------------
# Fitting helper
# ---------------------------------------------------------------------------


def _fit(tiles: np.ndarray, *, working_size: int = 128, estimate_darkfield: bool = True) -> tuple[np.ndarray, np.ndarray]:
    dct_sum = dct_energy(tiles.mean(axis=0))
    params = {
        "working_size": working_size,
        "l_s": dct_sum / DEFAULT_L_S_DIVISOR,
        "l_d": dct_sum / DEFAULT_L_D_DIVISOR,
        "epsilon": 0.1,
        "estimate_darkfield": estimate_darkfield,
    }
    model = make_model(tiles, params)
    model.prepare()
    model.run()
    return model.get_flatfield(), model.get_darkfield()


# ---------------------------------------------------------------------------
# Experiment: metric sanity check
# ---------------------------------------------------------------------------


def experiment_metric(mosaic: MosaicGrid, z: int) -> None:
    tiles = mosaic.iter_tiles(z)
    seams = mosaic.seam_pairs()
    raw_mean = float(np.mean(np.abs(tiles)))

    print(f"\n=== METRIC SANITY (z={z}, n_tiles={len(tiles)}, n_seams={len(seams)}) ===")
    print(f"raw tile mean |I| = {raw_mean:.4f}")

    flat, dark = _fit(tiles)
    corrected = _correct(tiles, flat, dark)

    base_curr = seam_l1_corrected_norm(corrected, seams)
    base_raw = seam_l1_raw_norm(corrected, seams, raw_mean)
    base_rel = seam_l1_relative(corrected, seams)
    base_pear = seam_pearson(corrected, seams)
    print("\n-- baseline (default fit) --")
    print(f"seam_l1 (corrected-norm) = {base_curr:.5f}")
    print(f"seam_l1 (raw-norm)       = {base_raw:.5f}")
    print(f"seam_l1 (relative)       = {base_rel:.5f}")
    print(f"seam_pearson             = {base_pear:.5f}")

    # (1) Global flat-field scaling: should be invariant for a good metric.
    print("\n-- global flat-field scaling (corrected -> corrected/k) --")
    print(f"{'k':>6} {'l1_corrnorm':>12} {'l1_rawnorm':>12} {'l1_relative':>12} {'pearson':>10}")
    for k in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        c = _correct(tiles, flat * k, dark * k)
        print(
            f"{k:>6.2f} {seam_l1_corrected_norm(c, seams):>12.5f} "
            f"{seam_l1_raw_norm(c, seams, raw_mean):>12.5f} "
            f"{seam_l1_relative(c, seams):>12.5f} {seam_pearson(c, seams):>10.5f}"
        )

    # (2) Center-inflation perturbation: brighten tile interior (no seams there)
    #     without changing seam regions. A gameable metric will DROP because the
    #     denominator (mean |corrected|) grows while seam diffs are unchanged.
    print("\n-- interior brightening (seam regions untouched) --")
    print(f"{'boost':>6} {'l1_corrnorm':>12} {'l1_rawnorm':>12} {'l1_relative':>12} {'pearson':>10}")
    th, tw = corrected.shape[1], corrected.shape[2]
    interior = np.zeros((th, tw), dtype=bool)
    m = max(1, int(0.3 * min(th, tw)))
    interior[m : th - m, m : tw - m] = True
    for boost in [1.0, 1.5, 2.0, 4.0]:
        c = corrected.copy()
        c[:, interior] *= boost
        print(
            f"{boost:>6.2f} {seam_l1_corrected_norm(c, seams):>12.5f} "
            f"{seam_l1_raw_norm(c, seams, raw_mean):>12.5f} "
            f"{seam_l1_relative(c, seams):>12.5f} {seam_pearson(c, seams):>10.5f}"
        )

    print(
        "\nINTERPRETATION: if l1_corrnorm DROPS under interior brightening while "
        "pearson is flat, the corrected-mean normalisation is gameable; raw-norm "
        "should stay flat (numerator unchanged)."
    )


# ---------------------------------------------------------------------------
# Experiment: tile subsampling
# ---------------------------------------------------------------------------


def _evenly_spaced(n_total: int, n_pick: int) -> np.ndarray:
    if n_pick >= n_total:
        return np.arange(n_total)
    return np.unique(np.linspace(0, n_total - 1, n_pick).round().astype(int))


def experiment_tiles(mosaic: MosaicGrid, z: int, counts: list[int]) -> None:
    all_tiles = mosaic.iter_tiles(z)
    seams = mosaic.seam_pairs()
    n_total = len(all_tiles)
    raw_mean = float(np.mean(np.abs(all_tiles)))

    print(f"\n=== TILE SUBSAMPLING (z={z}, n_total={n_total}) ===")
    print("Fit on N evenly-spaced tiles; evaluate seams on ALL tiles.")
    print(f"{'N_fit':>6} {'l1_corrnorm':>12} {'l1_rawnorm':>12} {'pearson':>10} {'fit_s':>8}")

    results = []
    for n in counts:
        idx = _evenly_spaced(n_total, n)
        fit_tiles = all_tiles[idx]
        t0 = time.perf_counter()
        flat, dark = _fit(fit_tiles)
        dt = time.perf_counter() - t0
        corrected = _correct(all_tiles, flat, dark)
        l1c = seam_l1_corrected_norm(corrected, seams)
        l1r = seam_l1_raw_norm(corrected, seams, raw_mean)
        pear = seam_pearson(corrected, seams)
        results.append((len(idx), l1c, l1r, pear, dt))
        print(f"{len(idx):>6} {l1c:>12.5f} {l1r:>12.5f} {pear:>10.5f} {dt:>8.2f}")

    # Report degradation relative to the full-tile fit.
    full = results[-1]
    print(f"\nFull-fit (N={full[0]}) seam_l1_rawnorm = {full[2]:.5f}")
    print("Degradation vs full-fit (raw-norm):")
    for n, _l1c, l1r, _pear, _dt in results:
        deg = 100.0 * (l1r - full[2]) / (full[2] + 1e-12)
        print(f"  N={n:>4}: {deg:+.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Path to the OME-Zarr mosaic.")
    p.add_argument("--experiment", choices=["metric", "tiles"], required=True)
    p.add_argument("--z", type=int, default=27)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=[16, 32, 64, 96, 128, 192, 256, 384, 496],
        help="Tile counts to sweep (tiles experiment).",
    )
    args = p.parse_args()

    print(f"Loading mosaic: {args.input}")
    mosaic = MosaicGrid.from_ome_zarr(str(args.input), overlap_fraction=args.overlap)
    print(f"  shape={mosaic.array.shape} tile={mosaic.tile_shape} grid={mosaic.n_rows}x{mosaic.n_cols} n_z={mosaic.n_z}")
    z = min(args.z, mosaic.n_z - 1)

    if args.experiment == "metric":
        experiment_metric(mosaic, z)
    else:
        experiment_tiles(mosaic, z, args.counts)


if __name__ == "__main__":
    main()
