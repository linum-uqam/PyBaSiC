"""Visualize depth-wise flatfield curvature along seam profiles.

Fits one BaSiC flat-field per z-level and plots:

  1. Depth side-view  -- z x heatmap of the mean flatfield profile in the
     seam direction, overlaid with the best-fit Gaussian curves.
  2. sigma(z)         -- Gaussian width vs depth.
  3. amplitude(z)     -- Gaussian peak amplitude vs depth, with centre(z) on
     a secondary axis.
  4. seam_curvature(z) -- per-z normalised overlap residual (lower = better).

The curvature plot (panel 4) is the depth-resolved version of the scalar
:func:`~linum_basic.curvature.seam_curvature` metric, showing *where* along
the optical axis the flatfield most closely follows the Gaussian optics model
and where it deviates (potential acquisition artefacts or fitting failures).

Usage::

    uv run python scripts/visualize_seam_curvature.py \\
        --input /path/to/mosaic.ome.zarr \\
        --z-indices 0 5 10 15 20 25 \\
        --overlap 0.2 \\
        --orientation horizontal \\
        --output seam_curvature.png

All arguments have defaults that work with the development test dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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
        "--output",
        default="seam_curvature.png",
        metavar="PNG",
        help="Output PNG path (default: %(default)s).",
    )
    p.add_argument(
        "--z-indices",
        nargs="+",
        type=int,
        default=None,
        metavar="Z",
        help="Z-levels to fit.  Default: evenly-spaced up to 12 levels.",
    )
    p.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        metavar="FRAC",
        help="Tile overlap fraction (default: %(default)s).",
    )
    p.add_argument(
        "--orientation",
        choices=["horizontal", "vertical", "both"],
        default="horizontal",
        help=(
            "Seam orientation to visualise in panels 1-3.  Panel 4 always "
            "averages over the orientations present.  (default: %(default)s)"
        ),
    )
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.0,
        metavar="SIGMA",
        help="Gaussian sigma for z-smoothing the flatfields before analysis (0 = off).",
    )
    p.add_argument(
        "--estimate-darkfield",
        action="store_true",
        default=False,
        help="Estimate dark-field in addition to flat-field.",
    )
    p.add_argument(
        "--n-extra",
        type=int,
        default=2,
        metavar="N",
        help=("Leading rows to exclude from the BaSiC fit (galvo fly-back artefact). Default: %(default)s."),
    )
    p.add_argument("--verbose", action="store_true", default=False, help="Show BaSiC progress bars.")
    return p


# ---------------------------------------------------------------------------
# Load + fit helpers
# ---------------------------------------------------------------------------


def _load_and_fit(
    zarr_path: str,
    z_indices: list[int] | None,
    overlap: float,
    estimate_darkfield: bool,
    n_extra_rows: int,
    verbose: bool,
):
    from linum_basic.fit import fit_mosaic
    from linum_basic.mosaic import MosaicGrid

    print(f"Loading mosaic from {zarr_path} …")
    mosaic = MosaicGrid.from_ome_zarr(zarr_path, overlap_fraction=overlap)
    n_z = mosaic.n_z
    print(f"  shape={mosaic.array.shape}  tile={mosaic.tile_shape}  grid={mosaic.n_rows}x{mosaic.n_cols}")

    if z_indices is None:
        step = max(1, n_z // 12)
        z_indices = list(range(0, n_z, step))[:12]

    print(f"Fitting {len(z_indices)} z-levels: {z_indices}")
    basic_kwargs = {"estimate_darkfield": estimate_darkfield}
    fit = fit_mosaic(mosaic, z_indices=z_indices, basic_kwargs=basic_kwargs, n_extra_rows=n_extra_rows, verbose=verbose)
    print(f"  flatfields shape: {fit.flatfields.shape}")
    return mosaic, fit


def _smooth_fields(flatfields: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return flatfields
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(flatfields, sigma=sigma, axis=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot(
    flatfields: np.ndarray,
    z_indices: list[int],
    seam_pairs: list,
    orientation: str,
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from linum_basic.curvature import (
        GaussianParams,
        curvature_depth_profile,
        focal_profile,
        seam_curvature_per_z,
    )

    n_z = flatfields.shape[0]
    z_arr = np.array(z_indices)

    # ------------------------------------------------------------------
    # Compute per-z Gaussian parameters
    # ------------------------------------------------------------------
    # Filter seam_pairs to desired orientation for panels 1-3
    if orientation == "both":
        orient_label = "horizontal + vertical"
        # For panels 1-3 default to horizontal if available
        orient_profile = "horizontal" if any(s.orientation == "horizontal" for s in seam_pairs) else "vertical"
    else:
        orient_profile = orientation
        orient_label = orientation

    gauss_params: list[GaussianParams] = curvature_depth_profile(flatfields, seam_pairs, orientation=orient_profile)
    curv_per_z: np.ndarray = seam_curvature_per_z(flatfields, seam_pairs)

    # Build the z x profile heatmap for panel 1
    profiles = np.stack([focal_profile(flatfields[z], orient_profile) for z in range(n_z)], axis=0)  # (Z, T)

    # ------------------------------------------------------------------
    # Figure layout: 4 panels
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)

    ax_heat = fig.add_subplot(gs[0, 0])
    ax_sigma = fig.add_subplot(gs[0, 1])
    ax_amp = fig.add_subplot(gs[1, 0])
    ax_curv = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------------------------
    # Panel 1: depth x profile heatmap + Gaussian overlays
    # ------------------------------------------------------------------
    extent = (0, profiles.shape[1], z_arr[-1] + 0.5, z_arr[0] - 0.5)
    im = ax_heat.imshow(profiles, aspect="auto", origin="upper", cmap="inferno", extent=extent)
    plt.colorbar(im, ax=ax_heat, label="Flatfield value")

    # Overlay fitted-Gaussian centers: a vertical line at each fitted center
    # position per z.  (The horizontal lines were redundant and removed.)
    for _zi, gp in enumerate(gauss_params):
        if np.isnan(gp.sigma):
            continue
        ax_heat.axvline(gp.center, color="cyan", lw=0.4, alpha=0.2)

    # Scatter the center positions per z
    centers = [gp.center for gp in gauss_params]
    valid_z = [z_arr[i] for i, c in enumerate(centers) if not np.isnan(c)]
    valid_c = [c for c in centers if not np.isnan(c)]
    if valid_c:
        ax_heat.scatter(valid_c, valid_z, s=15, color="cyan", zorder=5, label="Gaussian centre")
        ax_heat.legend(fontsize=7, loc="lower right")

    ax_heat.set_xlabel("Profile pixel (seam direction)")
    ax_heat.set_ylabel("z-index")
    ax_heat.set_title(f"Flatfield profile depth view\n({orient_profile} seam direction)")

    # ------------------------------------------------------------------
    # Panel 2: sigma(z)
    # ------------------------------------------------------------------
    sigmas = np.array([gp.sigma for gp in gauss_params], dtype=float)
    valid_mask = ~np.isnan(sigmas)
    ax_sigma.plot(z_arr[valid_mask], sigmas[valid_mask], "o-", color="steelblue", ms=5)
    if valid_mask.any():
        ax_sigma.axhline(float(np.nanmedian(sigmas)), ls="--", color="steelblue", lw=0.8, alpha=0.6, label="median")
        ax_sigma.legend(fontsize=8)
    ax_sigma.set_xlabel("z-index")
    ax_sigma.set_ylabel("Gaussian σ (px)")
    ax_sigma.set_title("Focal width σ(z)")
    ax_sigma.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 3: amplitude(z) with center(z) on twin axis
    # ------------------------------------------------------------------
    amplitudes = np.array([gp.amplitude for gp in gauss_params], dtype=float)
    ax_amp.plot(z_arr[valid_mask], amplitudes[valid_mask], "o-", color="darkorange", ms=5, label="amplitude")
    ax_amp.set_xlabel("z-index")
    ax_amp.set_ylabel("Gaussian amplitude", color="darkorange")
    ax_amp.tick_params(axis="y", labelcolor="darkorange")
    ax_amp.set_title("Amplitude and centre(z)")
    ax_amp.grid(True, alpha=0.3)

    ax_amp2 = ax_amp.twinx()
    ax_amp2.plot(z_arr[valid_mask], np.array(valid_c), "s--", color="purple", ms=4, alpha=0.7, label="centre")
    ax_amp2.set_ylabel("Gaussian centre (px)", color="purple")
    ax_amp2.tick_params(axis="y", labelcolor="purple")

    lines1, labels1 = ax_amp.get_legend_handles_labels()
    lines2, labels2 = ax_amp2.get_legend_handles_labels()
    ax_amp.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    # ------------------------------------------------------------------
    # Panel 4: seam_curvature per z
    # ------------------------------------------------------------------
    curv_valid = ~np.isnan(curv_per_z)
    ax_curv.plot(z_arr[curv_valid], curv_per_z[curv_valid], "o-", color="firebrick", ms=5)
    if curv_valid.any():
        mean_curv = float(np.nanmean(curv_per_z))
        ax_curv.axhline(mean_curv, ls="--", color="firebrick", lw=0.8, alpha=0.6, label=f"mean={mean_curv:.4f}")
        ax_curv.legend(fontsize=8)
    ax_curv.set_xlabel("z-index")
    ax_curv.set_ylabel("Normalised RMS residual")
    ax_curv.set_title(f"Seam curvature metric per z\n({orient_label})")
    ax_curv.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Overall title + save
    # ------------------------------------------------------------------
    scalar_curv = float(np.nanmean(curv_per_z))
    fig.suptitle(
        f"Seam curvature analysis  |  scalar metric = {scalar_curv:.4f}  |  n_z = {n_z}",
        fontsize=12,
        fontweight="bold",
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    mosaic, fit = _load_and_fit(
        args.input,
        args.z_indices,
        args.overlap,
        args.estimate_darkfield,
        args.n_extra,
        args.verbose,
    )

    flatfields = _smooth_fields(fit.flatfields, args.smooth_sigma)
    seam_pairs = mosaic.seam_pairs()

    _plot(
        flatfields=flatfields,
        z_indices=list(fit.z_indices),
        seam_pairs=seam_pairs,
        orientation=args.orientation,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
