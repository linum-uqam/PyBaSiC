"""Visualize flat-field focal curvature from an OME-Zarr mosaic.

Fits one BaSiC flat-field per z-level and plots:

  1. YZ side-view  -- flatfield[:, tile_h//2, :]  (z x x, middle row)
  2. XZ side-view  -- flatfield[:, :, tile_w//2]  (z x y, middle col)
  3. Raw vs corrected montage for a chosen z-level

Usage::

    uv run python scripts/visualize_flatfield_curvature.py \\
        --input /path/to/mosaic.ome.zarr \\
        --z-indices 0 10 20 30 40 50 \\
        --z-inspect 27 \\
        --overlap 0.2 \\
        --output docs/_static/demo/flatfield_curvature.png

All arguments have defaults that work with the development test dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from linum_basic import viz

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_ZARR = "/Users/Frans/Downloads/sub-22/mosaic_grid_z27_focal_fix.ome.zarr"
_DEFAULT_OUT = str(Path(_DEFAULT_ZARR).parent / "flatfield_curvature.png")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", default=_DEFAULT_ZARR, metavar="ZARR", help="Path to the OME-Zarr mosaic (default: %(default)s)."
    )
    p.add_argument("--output", default=_DEFAULT_OUT, metavar="PNG", help="Output PNG path (default: %(default)s).")
    p.add_argument(
        "--z-indices",
        nargs="+",
        type=int,
        default=None,
        metavar="Z",
        help="Z-levels to fit. Default: evenly-spaced 10 levels.",
    )
    p.add_argument(
        "--z-inspect",
        type=int,
        default=None,
        metavar="Z",
        help="Z-level for raw-vs-corrected panel. Default: middle of --z-indices.",
    )
    p.add_argument("--overlap", type=float, default=0.2, metavar="FRAC", help="Tile overlap fraction (default: %(default)s).")
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.0,
        metavar="SIGMA",
        help="Optional Gaussian sigma for depth-smoothing flatfields (0 = off).",
    )
    p.add_argument(
        "--estimate-darkfield", action="store_true", default=False, help="Estimate dark-field in addition to flat-field."
    )
    p.add_argument("--verbose", action="store_true", default=False, help="Show BaSiC progress bars.")
    p.add_argument(
        "--n-extra",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Number of galvo-return / scan-settling rows at the top of each tile to exclude "
            "from the BaSiC fit and replace by edge-extension in the corrected output. "
            "Set to 0 to disable masking. Default: %(default)s (empirically determined from "
            "per-row intensity diagnostics on the acquisition data)."
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Core logic
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
        z_indices = list(range(n_z))

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
# Orientation analysis helpers
# ---------------------------------------------------------------------------


def _find_principal_angle(flatfields: np.ndarray) -> float:
    """Return the dominant gradient direction (degrees) of the mean flat-field.

    Uses SVD of the pixel-wise gradient vectors to find the in-plane direction
    of maximum illumination variation.  For a radially symmetric beam this is
    arbitrary; for an asymmetric beam it points along the major axis.
    """
    ff_mean = flatfields.mean(axis=0)
    gy, gx = np.gradient(ff_mean)
    grad_stack = np.stack([gx.ravel(), gy.ravel()], axis=1)
    _, _, vt = np.linalg.svd(grad_stack, full_matrices=False)
    dx, dy = float(vt[0, 0]), float(vt[0, 1])
    return float(np.degrees(np.arctan2(dy, dx)))


def _zd_slice(flatfields: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Extract a Z x T diagonal slice from the flat-field volume.

    Parameters
    ----------
    flatfields : numpy.ndarray, shape (n_z, th, tw)
    angle_deg : float
        Cut angle in degrees measured from the positive x-axis
        (0 = horizontal, 90 = vertical, arbitrary values = diagonal).

    Returns
    -------
    t : numpy.ndarray
        Pixel offsets from the tile centre along the cut direction.
    view : numpy.ndarray, shape (n_z, len(t))
        Flat-field values sampled along the cut for every z-level.
    """
    from scipy.ndimage import map_coordinates

    _, th, tw = flatfields.shape
    cy, cx = th // 2, tw // 2
    ar = np.deg2rad(angle_deg)
    half = min(th, tw) // 2 - 1
    t = np.arange(-half, half + 1, dtype=float)
    rows = cy + t * np.sin(ar)
    cols = cx + t * np.cos(ar)
    view = np.stack(
        [map_coordinates(flatfields[i], [rows, cols], order=1, mode="nearest") for i in range(len(flatfields))],
        axis=0,
    )
    return t, view


def _build_orientation_figure(
    fit,
    z_inspect: int,
    smooth_sigma: float,
    out_path: Path,
    zarr_path: str = "",
) -> None:
    """Generate a flat-field orientation comparison figure.

    Shows Z-depth cross-sections at four cut angles: axis-aligned (0°, 90°)
    and along the principal / perpendicular gradient directions computed via
    SVD.  Diagonal cuts slice the illumination volume along its natural
    curvature axis rather than forcing an arbitrary axis-aligned view.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from scipy.ndimage import map_coordinates

    viz.set_theme()
    flatfields = _smooth_fields(fit.flatfields.copy(), smooth_sigma)
    z_indices = fit.z_indices
    _, th, tw = flatfields.shape
    cy, cx = th // 2, tw // 2
    z_labels = np.array(z_indices)

    principal_angle = _find_principal_angle(flatfields)
    perp_angle = principal_angle + 90.0

    # Z-depth views at four cut angles
    t_h, view_h = _zd_slice(flatfields, 0.0)
    t_v, view_v = _zd_slice(flatfields, 90.0)
    t_p, view_p = _zd_slice(flatfields, principal_angle)
    t_perp, view_perp = _zd_slice(flatfields, perp_angle)

    # 1-D profiles through the flat-field at z_inspect
    z_pos = z_indices.index(z_inspect)
    ff_zi = flatfields[z_pos]

    def _profile(ff: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
        ar = np.deg2rad(angle_deg)
        h = min(ff.shape) // 2 - 1
        tp = np.arange(-h, h + 1, dtype=float)
        rp = cy + tp * np.sin(ar)
        cp = cx + tp * np.cos(ar)
        return tp, map_coordinates(ff, [rp, cp], order=1, mode="nearest")

    t1h, p_h = _profile(ff_zi, 0.0)
    t1v, p_v = _profile(ff_zi, 90.0)
    t1p, p_p = _profile(ff_zi, principal_angle)
    t1perp, p_perp = _profile(ff_zi, perp_angle)

    # -----------------------------------------------------------------------
    # Layout: 2 rows x 3 cols
    # Row 0: mean ff + cuts | Z x H (0 deg)   | Z x Principal (diagonal)
    # Row 1: 1-D profiles   | Z x V (90 deg)  | Z x Perp (diagonal)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    vmin, vmax = 0.7, 1.3
    im_kw = {"aspect": "auto", "origin": "lower", "cmap": viz.FLATFIELD_CMAP, "vmin": vmin, "vmax": vmax}

    # --- [0, 0] Mean flat-field with cut-direction overlay ---
    ff_mean = flatfields.mean(axis=0)
    ax = axes[0, 0]
    im = ax.imshow(ff_mean, cmap=viz.FLATFIELD_CMAP, vmin=vmin, vmax=vmax, origin="lower")
    half_line = min(th, tw) // 2 - 1
    for angle, color, lbl in [
        (0.0, "dodgerblue", "H  (0°)"),
        (90.0, "deepskyblue", "V  (90°)"),
        (principal_angle, "crimson", f"Principal  ({principal_angle:.1f}°)"),
        (perp_angle, "tomato", f"Perp  ({perp_angle:.1f}°)"),
    ]:
        ar = np.deg2rad(angle)
        ax.plot(
            [cx - half_line * np.cos(ar), cx + half_line * np.cos(ar)],
            [cy - half_line * np.sin(ar), cy + half_line * np.sin(ar)],
            color=color,
            linewidth=2,
            label=lbl,
        )
    ax.legend(fontsize=7, loc="upper right")
    fig.colorbar(im, ax=ax, shrink=0.8, label="flat-field")
    ax.set_title("Mean flat-field + cut directions")
    ax.set_xlabel("x-pixel")
    ax.set_ylabel("y-pixel")

    # --- [0, 1] Z x H view ---
    ax = axes[0, 1]
    im = ax.imshow(
        view_h,
        **im_kw,
        extent=(float(t_h[0]), float(t_h[-1]), float(z_labels[0]), float(z_labels[-1])),
    )
    ax.set_xlabel("pixel offset from centre")
    ax.set_ylabel("z-index")
    ax.set_title("Z x H  (axis-aligned, 0 deg)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="flat-field")

    # --- [0, 2] Z x Principal view (diagonal) ---
    ax = axes[0, 2]
    im = ax.imshow(
        view_p,
        **im_kw,
        extent=(float(t_p[0]), float(t_p[-1]), float(z_labels[0]), float(z_labels[-1])),
    )
    ax.set_xlabel("pixel offset from centre")
    ax.set_ylabel("z-index")
    ax.set_title(f"Z x Principal  ({principal_angle:.1f} deg, diagonal)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="flat-field")

    # --- [1, 0] 1-D profiles at z_inspect ---
    ax = axes[1, 0]
    ax.plot(t1h, p_h, color="dodgerblue", label="H  (0°)")
    ax.plot(t1v, p_v, color="deepskyblue", label="V  (90°)")
    ax.plot(t1p, p_p, color="crimson", label=f"Principal  ({principal_angle:.1f}°)")
    ax.plot(t1perp, p_perp, color="tomato", label=f"Perp  ({perp_angle:.1f}°)")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("pixel offset from centre")
    ax.set_ylabel("flat-field value")
    ax.set_title(f"1-D profiles at z={z_inspect}")
    ax.legend(fontsize=8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", alpha=0.4)

    # --- [1, 1] Z x V view ---
    ax = axes[1, 1]
    im = ax.imshow(
        view_v,
        **im_kw,
        extent=(float(t_v[0]), float(t_v[-1]), float(z_labels[0]), float(z_labels[-1])),
    )
    ax.set_xlabel("pixel offset from centre")
    ax.set_ylabel("z-index")
    ax.set_title("Z x V  (axis-aligned, 90 deg)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="flat-field")

    # --- [1, 2] Z x Perp view (diagonal) ---
    ax = axes[1, 2]
    im = ax.imshow(
        view_perp,
        **im_kw,
        extent=(float(t_perp[0]), float(t_perp[-1]), float(z_labels[0]), float(z_labels[-1])),
    )
    ax.set_xlabel("pixel offset from centre")
    ax.set_ylabel("z-index")
    ax.set_title(f"Z x Perp  ({perp_angle:.1f} deg, diagonal)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="flat-field")

    dataset_name = Path(zarr_path).name if zarr_path else "dataset"
    fig.suptitle(
        f"Flat-field orientation analysis  |  {dataset_name}",
        fontsize=13,
        fontweight="bold",
    )

    viz.save_figure(fig, out_path, dpi=150)
    print(f"Saved to {out_path}")


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------


def _build_figure(
    mosaic, fit, z_inspect: int, smooth_sigma: float, out_path: Path, zarr_path: str = "", n_extra_rows: int = 0
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    viz.set_theme()

    flatfields = _smooth_fields(fit.flatfields.copy(), smooth_sigma)
    _n_z_fit, th, tw = flatfields.shape
    z_indices = fit.z_indices

    # -----------------------------------------------------------------------
    # YZ and XZ side-views
    # -----------------------------------------------------------------------
    mid_row = th // 2
    mid_col = tw // 2
    yz_view = flatfields[:, mid_row, :]  # shape (n_z_fit, tw)  -- z x x
    xz_view = flatfields[:, :, mid_col]  # shape (n_z_fit, th)  -- z x y

    z_labels = np.array(z_indices)

    # -----------------------------------------------------------------------
    # Raw vs corrected at z_inspect
    # -----------------------------------------------------------------------
    from linum_basic.fit import apply_fit

    z_fit_idx = z_indices.index(z_inspect)
    ff = flatfields[z_fit_idx]
    raw_z = mosaic.array[z_inspect]  # (H, W)
    corrected_z = apply_fit(mosaic, fit, n_extra_rows=n_extra_rows)[z_inspect]
    # clip to [0, 99th-percentile] for display
    p99 = float(np.percentile(raw_z[raw_z > 0], 99))
    raw_disp = np.clip(raw_z, 0, p99)
    p99c = float(np.percentile(corrected_z[corrected_z > 0], 99)) if corrected_z.max() > 0 else 1.0
    corr_disp = np.clip(corrected_z, 0, p99c)

    # -----------------------------------------------------------------------
    # Layout: 2x2 + 1 wide
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax_yz = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[1, 0])
    ax_ff = fig.add_subplot(gs[0, 1])
    ax_raw = fig.add_subplot(gs[0, 2])
    ax_corr = fig.add_subplot(gs[1, 2])
    ax_prof = fig.add_subplot(gs[1, 1])

    # --- YZ view ---
    im = ax_yz.imshow(
        yz_view,
        aspect="auto",
        origin="lower",
        extent=(0, float(tw), float(z_labels[0]), float(z_labels[-1])),
        cmap=viz.FLATFIELD_CMAP,
        vmin=0.7,
        vmax=1.3,
    )
    ax_yz.set_xlabel("x-pixel within tile")
    ax_yz.set_ylabel("z-index")
    ax_yz.set_title("Flat-field YZ slice\n(middle row of each z-level)")
    fig.colorbar(im, ax=ax_yz, shrink=0.8, label="flat-field value")

    # --- XZ view ---
    im2 = ax_xz.imshow(
        xz_view,
        aspect="auto",
        origin="lower",
        extent=(0, float(th), float(z_labels[0]), float(z_labels[-1])),
        cmap=viz.FLATFIELD_CMAP,
        vmin=0.7,
        vmax=1.3,
    )
    ax_xz.set_xlabel("y-pixel within tile")
    ax_xz.set_ylabel("z-index")
    ax_xz.set_title("Flat-field XZ slice\n(middle col of each z-level)")
    fig.colorbar(im2, ax=ax_xz, shrink=0.8, label="flat-field value")

    # --- Flat-field at z_inspect ---
    im3 = ax_ff.imshow(ff, cmap=viz.FLATFIELD_CMAP, vmin=0.7, vmax=1.3, origin="lower")
    ax_ff.set_title(f"Flat-field at z={z_inspect}")
    ax_ff.set_xlabel("x-pixel")
    ax_ff.set_ylabel("y-pixel")
    fig.colorbar(im3, ax=ax_ff, shrink=0.8, label="flat-field value")

    # --- Depth profile of center pixel ---
    center_vals = flatfields[:, mid_row, mid_col]
    ax_prof.plot(z_labels, center_vals, "o-", color="steelblue", markersize=4)
    ax_prof.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax_prof.set_xlabel("z-index")
    ax_prof.set_ylabel("flat-field (center pixel)")
    ax_prof.set_title("Depth profile — center pixel")
    ax_prof.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_prof.grid(True, which="major", alpha=0.4)

    # --- Raw mosaic ---
    ax_raw.imshow(raw_disp, cmap="gray", origin="lower")
    ax_raw.set_title(f"Raw mosaic  z={z_inspect}")
    ax_raw.axis("off")

    # --- Corrected mosaic ---
    ax_corr.imshow(corr_disp, cmap="gray", origin="lower")
    ax_corr.set_title(f"Corrected mosaic  z={z_inspect}")
    ax_corr.axis("off")

    dataset_name = Path(zarr_path).name if zarr_path else "dataset"
    fig.suptitle(
        f"BaSiC flat-field focal curvature  |  {dataset_name}",
        fontsize=13,
        fontweight="bold",
    )

    viz.save_figure(fig, out_path, dpi=150)
    print(f"Saved to {out_path}")


# ---------------------------------------------------------------------------
# 3-D surface figure
# ---------------------------------------------------------------------------


def _build_3d_figure(
    fit,
    z_inspect: int,
    smooth_sigma: float,
    out_path: Path,
    zarr_path: str = "",
    n_stacked: int = 5,
) -> None:
    """Render the flat-field as true 3-D surfaces.

    Two panels are produced: a single high-detail surface of the flat-field at
    ``z_inspect``, and a stacked, translucent set of surfaces sampled across the
    fitted depth range to show how the focal curvature evolves with z.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    viz.set_theme()

    flatfields = _smooth_fields(fit.flatfields.copy(), smooth_sigma)
    n_z_fit, th, tw = flatfields.shape
    z_indices = list(fit.z_indices)

    xx, yy = np.meshgrid(np.arange(tw), np.arange(th))
    vmin, vmax = 0.7, 1.3
    fig = plt.figure(figsize=(16, 7))

    # --- Panel 1: single surface at z_inspect ---
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    z_fit_idx = z_indices.index(z_inspect)
    ff = flatfields[z_fit_idx]
    surf = ax1.plot_surface(
        xx, yy, ff, cmap=viz.FLATFIELD_CMAP, vmin=vmin, vmax=vmax, linewidth=0, antialiased=True, rcount=80, ccount=80
    )
    ax1.set_title(f"Flat-field surface at z={z_inspect}")
    ax1.set_xlabel("x-pixel")
    ax1.set_ylabel("y-pixel")
    ax1.set_zlabel("flat-field")
    ax1.set_zlim(vmin, vmax)
    fig.colorbar(surf, ax=ax1, shrink=0.6, pad=0.1, label="flat-field value")

    # --- Panel 2: stacked translucent surfaces across depth ---
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    sample_idx = np.unique(np.linspace(0, n_z_fit - 1, min(n_stacked, n_z_fit)).round().astype(int))
    cmap = plt.get_cmap("viridis")
    for i in sample_idx:
        frac = i / max(n_z_fit - 1, 1)
        ax2.plot_surface(
            xx,
            yy,
            flatfields[i],
            color=cmap(frac),
            alpha=0.45,
            linewidth=0,
            antialiased=True,
            rcount=40,
            ccount=40,
            shade=False,
        )
    ax2.set_title("Flat-field curvature across depth")
    ax2.set_xlabel("x-pixel")
    ax2.set_ylabel("y-pixel")
    ax2.set_zlabel("flat-field")
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array(np.array([z_indices[i] for i in sample_idx], dtype=float))
    fig.colorbar(mappable, ax=ax2, shrink=0.6, pad=0.1, label="z-index")

    dataset_name = Path(zarr_path).name if zarr_path else "dataset"
    fig.suptitle(
        f"BaSiC flat-field 3-D surface  |  {dataset_name}",
        fontsize=13,
        fontweight="bold",
    )

    viz.save_figure(fig, out_path, dpi=150)
    print(f"Saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    mosaic, fit = _load_and_fit(
        zarr_path=args.input,
        z_indices=args.z_indices,
        overlap=args.overlap,
        estimate_darkfield=args.estimate_darkfield,
        n_extra_rows=args.n_extra,
        verbose=args.verbose,
    )

    z_inspect = args.z_inspect
    if z_inspect is None:
        z_inspect = fit.z_indices[len(fit.z_indices) // 2]

    if z_inspect not in fit.z_indices:
        print(
            f"Error: --z-inspect={z_inspect} was not in the fitted z-indices {fit.z_indices}.\n"
            f"Pass it explicitly via --z-indices.",
            file=sys.stderr,
        )
        return 1

    _build_figure(
        mosaic,
        fit,
        z_inspect=z_inspect,
        smooth_sigma=args.smooth_sigma,
        out_path=Path(args.output),
        zarr_path=args.input,
        n_extra_rows=args.n_extra,
    )
    orient_path = Path(args.output).with_stem(Path(args.output).stem + "_orientation")
    _build_orientation_figure(
        fit,
        z_inspect=z_inspect,
        smooth_sigma=args.smooth_sigma,
        out_path=orient_path,
        zarr_path=args.input,
    )
    threed_path = Path(args.output).with_stem(Path(args.output).stem + "_3d")
    _build_3d_figure(
        fit,
        z_inspect=z_inspect,
        smooth_sigma=args.smooth_sigma,
        out_path=threed_path,
        zarr_path=args.input,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
