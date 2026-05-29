"""Shared visualization theme and figure builders for linum-basic.

This module is the single source of truth for figure styling so that every
diagnostic and documentation figure shares one consistent look.  The palette
is built around :mod:`matplotlib`'s perceptually uniform ``viridis`` colormap:

* flat-fields are shown with ``viridis``;
* dark-fields use its inverse, ``viridis_r``;
* raw / corrected intensity images use grayscale for readability.

Plotting helpers here never run the BaSiC solver or Optuna — they consume
arrays produced elsewhere and render them.  Importing this module requires
matplotlib (installed via the ``viz`` optional dependency group); callers that
must work without matplotlib should import it lazily.

Examples
--------
>>> import numpy as np
>>> from linum_basic import viz
>>> field = np.ones((8, 8), dtype=np.float32)
>>> fig, _ = viz.field_surface_3d(field, title="flat-field")
>>> _ = viz.save_figure(fig, "/tmp/example_flatfield_3d.png")  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

__all__ = [
    "DARKFIELD_CMAP",
    "FLATFIELD_CMAP",
    "INTENSITY_CMAP",
    "PREVIEW_CMAP",
    "Panel",
    "add_scalebar",
    "aip_preview",
    "field_surface_3d",
    "figure_apply_correction",
    "figure_focal_volume",
    "figure_panels",
    "figure_seam_metric",
    "figure_tuning_history",
    "save_figure",
    "set_theme",
    "show_field",
    "show_image",
]

#: Colormap for estimated flat-fields (multiplicative gain).
FLATFIELD_CMAP = "viridis"
#: Colormap for estimated dark-fields (additive offset) — the inverse palette.
DARKFIELD_CMAP = "viridis_r"
#: Colormap for raw / corrected tissue intensity images.
INTENSITY_CMAP = "gray"
#: Colormap for projected volume previews.
PREVIEW_CMAP = "viridis"


def set_theme() -> None:
    """Apply the shared linum-basic figure theme to matplotlib's global state.

    Uses the ``scienceplots`` publication style (without LaTeX) when available
    and falls back to matplotlib's default style otherwise.  Common rc
    parameters (DPI, fonts, default colormap, white background) are then
    overridden so every figure is visually consistent.
    """
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except Exception:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "image.cmap": FLATFIELD_CMAP,
            "image.interpolation": "nearest",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
            "axes.grid": False,
        }
    )


def save_figure(fig: Figure, path: str | Path, *, dpi: int = 200) -> Path:
    """Save *fig* to *path* with consistent settings and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to write.
    path : str or Path
        Output path; parent directories are created if missing.
    dpi : int, optional
        Output resolution in dots per inch (default 200).

    Returns
    -------
    Path
        The resolved output path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def add_scalebar(
    ax: plt.Axes,
    pixel_size_mm: float | None,
    *,
    color: str = "white",
    location: str = "lower right",
) -> object | None:
    """Add a physical scale bar to *ax* if a pixel size is available.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes holding an image whose pixels span ``pixel_size_mm`` millimetres.
    pixel_size_mm : float or None
        Physical size of one pixel in millimetres.  When ``None`` or
        non-positive no scale bar is drawn.
    color : str, optional
        Scale-bar colour (default ``"white"``).
    location : str, optional
        Matplotlib legend-style location string (default ``"lower right"``).

    Returns
    -------
    matplotlib_scalebar.scalebar.ScaleBar or None
        The added artist, or ``None`` if no bar was drawn.
    """
    if pixel_size_mm is None or pixel_size_mm <= 0:
        return None
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
    except ImportError:
        return None
    bar = ScaleBar(
        pixel_size_mm,
        units="mm",
        color=color,
        box_alpha=0.0,
        frameon=False,
        location=location,
    )
    ax.add_artist(bar)
    return bar


def show_field(
    ax: plt.Axes,
    field: np.ndarray,
    *,
    darkfield: bool = False,
    contours: bool = True,
    title: str | None = None,
    colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> matplotlib.image.AxesImage:
    """Display an estimated flat- or dark-field on *ax*.

    Flat-fields use ``viridis`` with optional iso-gain contours; dark-fields
    use the inverse palette ``viridis_r`` and never draw contours.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    field : numpy.ndarray
        2-D field to display.
    darkfield : bool, optional
        Render as a dark-field (inverse palette) when ``True``.
    contours : bool, optional
        Overlay iso-level contours for flat-fields (default ``True``).
    title : str, optional
        Axes title.
    colorbar : bool, optional
        Attach a colorbar (default ``True``).
    vmin, vmax : float, optional
        Display range limits.

    Returns
    -------
    matplotlib.image.AxesImage
        The rendered image handle.
    """
    arr = np.asarray(field)
    cmap = DARKFIELD_CMAP if darkfield else FLATFIELD_CMAP
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    if contours and not darkfield:
        ax.contour(arr, levels=8, colors="w", linewidths=0.5, alpha=0.6)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def show_image(
    ax: plt.Axes,
    img: np.ndarray,
    *,
    title: str | None = None,
    cmap: str = INTENSITY_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    pixel_size_mm: float | None = None,
    scalebar: bool = False,
) -> matplotlib.image.AxesImage:
    """Display a raw or corrected intensity image on *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    img : numpy.ndarray
        2-D intensity image.
    title : str, optional
        Axes title.
    cmap : str, optional
        Colormap (default grayscale).
    vmin, vmax : float, optional
        Display range limits.
    colorbar : bool, optional
        Attach a colorbar (default ``True``).
    pixel_size_mm : float, optional
        Physical pixel size for the scale bar.
    scalebar : bool, optional
        Draw a physical scale bar when ``True`` and ``pixel_size_mm`` is set.

    Returns
    -------
    matplotlib.image.AxesImage
        The rendered image handle.
    """
    im = ax.imshow(np.asarray(img), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if scalebar:
        add_scalebar(ax, pixel_size_mm)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def field_surface_3d(
    field: np.ndarray,
    *,
    darkfield: bool = False,
    title: str | None = None,
    zlabel: str = "gain",
    figsize: tuple[float, float] = (7.0, 6.0),
) -> tuple[Figure, plt.Axes]:
    """Render a field as a 3-D illumination surface.

    Parameters
    ----------
    field : numpy.ndarray
        2-D field to render.
    darkfield : bool, optional
        Use the inverse palette when ``True``.
    title : str, optional
        Figure title.
    zlabel : str, optional
        Label for the vertical axis (default ``"gain"``).
    figsize : tuple of float, optional
        Figure size in inches.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and its 3-D axes.
    """
    set_theme()
    arr = np.asarray(field)
    cmap = DARKFIELD_CMAP if darkfield else FLATFIELD_CMAP
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    yy, xx = np.mgrid[0 : arr.shape[0], 0 : arr.shape[1]]
    surf = ax.plot_surface(xx, yy, arr, cmap=cmap, rcount=80, ccount=80, linewidth=0, antialiased=True)
    ax.set_xlabel("x-pixel")
    ax.set_ylabel("y-pixel")
    ax.set_zlabel(zlabel)
    if title:
        ax.set_title(title)
    ax.view_init(elev=32, azim=-58)
    fig.colorbar(surf, ax=ax, fraction=0.03, pad=0.08)
    fig.tight_layout()
    return fig, ax


def aip_preview(
    volume: np.ndarray,
    *,
    axis: int = 0,
    pixel_size_mm: float | None = None,
    cmap: str = PREVIEW_CMAP,
    title: str | None = None,
    percentile: float = 99.5,
    figsize: tuple[float, float] = (7.0, 7.0),
) -> Figure:
    """Build an average-intensity-projection (AIP) preview of a 3-D volume.

    The volume is averaged along *axis* (the depth axis by default), giving a
    single 2-D projection rendered as a PNG-ready figure.

    Parameters
    ----------
    volume : numpy.ndarray
        3-D image volume.
    axis : int, optional
        Axis to project (average) over (default ``0``, the depth/``z`` axis).
    pixel_size_mm : float, optional
        In-plane physical pixel size for the scale bar.
    cmap : str, optional
        Colormap (default ``viridis``).
    title : str, optional
        Figure title.
    percentile : float, optional
        Upper display percentile for contrast (default ``99.5``); the matching
        lower percentile sets ``vmin``.
    figsize : tuple of float, optional
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The projection figure.

    Raises
    ------
    ValueError
        If *volume* is not 3-dimensional.
    """
    set_theme()
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"aip_preview expects a 3-D volume, got shape {vol.shape}")
    proj = vol.mean(axis=axis)
    if proj.size:
        vmax = float(np.percentile(proj, percentile))
        vmin = float(np.percentile(proj, 100.0 - percentile))
        if vmax <= vmin:
            vmin, vmax = float(proj.min()), float(proj.max())
    else:
        vmin, vmax = 0.0, 1.0
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(proj, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    add_scalebar(ax, pixel_size_mm)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


@dataclass
class Panel:
    """A single panel description for :func:`figure_panels`.

    Attributes
    ----------
    data : numpy.ndarray
        2-D array to display.
    title : str
        Panel title.
    kind : str
        One of ``"image"``, ``"flatfield"`` or ``"darkfield"``.
    cmap : str or None
        Override colormap (only used for ``kind == "image"``).
    vmin, vmax : float or None
        Display range limits.
    contours : bool
        Overlay contours (flat-fields only).
    pixel_size_mm : float or None
        Physical pixel size for an optional scale bar.
    scalebar : bool
        Draw a scale bar (images only).
    """

    data: np.ndarray
    title: str = ""
    kind: str = "image"
    cmap: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    contours: bool = True
    pixel_size_mm: float | None = None
    scalebar: bool = False


def figure_panels(
    panels: Sequence[Panel],
    *,
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
) -> Figure:
    """Lay out a sequence of panels on a consistent grid.

    Parameters
    ----------
    panels : sequence of Panel
        Panels to render, row-major.
    ncols : int, optional
        Number of columns; defaults to ``len(panels)`` (single row).
    figsize : tuple of float, optional
        Figure size; a sensible default is derived from the grid shape.
    suptitle : str, optional
        Overall figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure.
    """
    set_theme()
    n = len(panels)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    figsize = figsize or (4.0 * ncols, 4.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axes.ravel()
    for ax, panel in zip(flat_axes, panels, strict=False):
        if panel.kind == "flatfield":
            show_field(
                ax,
                panel.data,
                darkfield=False,
                contours=panel.contours,
                title=panel.title,
                vmin=panel.vmin,
                vmax=panel.vmax,
            )
        elif panel.kind == "darkfield":
            show_field(
                ax,
                panel.data,
                darkfield=True,
                contours=False,
                title=panel.title,
                vmin=panel.vmin,
                vmax=panel.vmax,
            )
        else:
            show_image(
                ax,
                panel.data,
                title=panel.title,
                cmap=panel.cmap or INTENSITY_CMAP,
                vmin=panel.vmin,
                vmax=panel.vmax,
                pixel_size_mm=panel.pixel_size_mm,
                scalebar=panel.scalebar,
            )
    for ax in flat_axes[n:]:
        ax.set_axis_off()
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def _improvement_pct(before: float, after: float) -> float:
    """Return the percentage reduction from *before* to *after* (higher = better)."""
    return 100.0 * (before - after) / (before + 1e-12)


def _overlap_slices(
    shape: tuple[int, int], orientation: str, overlap_fraction: float
) -> tuple[tuple[slice, slice], tuple[slice, slice], int, str]:
    """Compute the overlap slices for two adjacent tiles.

    Returns the slice into tile A, the slice into tile B, the axis to average
    over to obtain a profile *along* the seam, and a label for that seam axis.
    """
    th, tw = shape
    if orientation == "horizontal":
        ov = max(1, round(overlap_fraction * tw))
        return (slice(None), slice(tw - ov, tw)), (slice(None), slice(0, ov)), 1, "y-pixel (along seam)"
    if orientation == "vertical":
        ov = max(1, round(overlap_fraction * th))
        return (slice(th - ov, th), slice(None)), (slice(0, ov), slice(None)), 0, "x-pixel (along seam)"
    raise ValueError(f"orientation must be 'horizontal' or 'vertical', got {orientation!r}")


def _seam_l1(a: np.ndarray, b: np.ndarray, ovl_a: tuple[slice, slice], ovl_b: tuple[slice, slice]) -> float:
    """Scale-invariant mean L1 disagreement over the shared overlap (0 = perfect)."""
    oa = a[ovl_a].ravel()
    ob = b[ovl_b].ravel()
    local = float((np.abs(oa) + np.abs(ob)).mean()) / 2.0
    return float(np.abs(oa - ob).mean()) / (local + 1e-9)


def figure_apply_correction(
    *,
    flatfield: np.ndarray,
    raw_mosaic: np.ndarray,
    corrected_mosaic: np.ndarray,
    raw_tile: np.ndarray,
    corrected_tile: np.ndarray,
    seam_raw: float,
    seam_corrected: float,
    darkfield: np.ndarray | None = None,
    title: str | None = None,
    pixel_size_mm: float | None = None,
) -> Figure:
    """Build the raw -> corrected mosaic diagnostic figure (no ground truth).

    Layout is a 2x3 grid:

    * column 0: estimated flat-field (top) and dark-field or column-mean
      intensity profiles (bottom);
    * column 1: raw and corrected full mosaics;
    * column 2: a representative tile, raw and corrected.

    Parameters
    ----------
    flatfield : numpy.ndarray
        Estimated flat-field.
    raw_mosaic, corrected_mosaic : numpy.ndarray
        Full mosaic images before and after correction.
    raw_tile, corrected_tile : numpy.ndarray
        A representative tile before and after correction.
    seam_raw, seam_corrected : float
        Seam-consistency metric before and after correction.
    darkfield : numpy.ndarray, optional
        Estimated dark-field.  When omitted (or near-zero) the bottom-left
        panel shows column-mean intensity profiles instead.
    title : str, optional
        Title prefix; the seam metrics are appended automatically.
    pixel_size_mm : float, optional
        Physical pixel size for scale bars on the mosaic panels.

    Returns
    -------
    matplotlib.figure.Figure
        The diagnostic figure.
    """
    set_theme()
    raw_mosaic = np.asarray(raw_mosaic, dtype=np.float32)
    corrected_mosaic = np.asarray(corrected_mosaic, dtype=np.float32)
    has_df = darkfield is not None and float(np.asarray(darkfield).max()) > 1e-6
    improvement = _improvement_pct(seam_raw, seam_corrected)

    mos_vmax = float(np.percentile(raw_mosaic, 99))
    tile_vmax = float(np.percentile(np.asarray(raw_tile), 99))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    suptitle = f"seam L1: {seam_raw:.3f} -> {seam_corrected:.3f}  ({improvement:+.1f}%)"
    fig.suptitle(f"{title}\n{suptitle}" if title else suptitle)

    show_field(axes[0, 0], flatfield, title="Estimated flat-field")

    if has_df:
        show_field(axes[1, 0], np.asarray(darkfield), darkfield=True, title="Estimated dark-field")
    else:
        axes[1, 0].plot(raw_mosaic.mean(axis=0), color="tab:red", lw=1.0, label="raw")
        axes[1, 0].plot(corrected_mosaic.mean(axis=0), color="tab:green", lw=1.0, label="corrected")
        axes[1, 0].set_title("Column-mean intensity")
        axes[1, 0].set_xlabel("x-pixel")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].margins(x=0)

    show_image(axes[0, 1], raw_mosaic, title="Raw mosaic", vmin=0, vmax=mos_vmax, pixel_size_mm=pixel_size_mm, scalebar=True)
    show_image(
        axes[1, 1],
        corrected_mosaic,
        title="Corrected mosaic",
        vmin=0,
        vmax=mos_vmax,
        pixel_size_mm=pixel_size_mm,
        scalebar=True,
    )
    show_image(axes[0, 2], np.asarray(raw_tile), title="Sample tile: raw", vmin=0, vmax=tile_vmax)
    show_image(axes[1, 2], np.asarray(corrected_tile), title="Sample tile: corrected", vmin=0, vmax=tile_vmax)

    fig.tight_layout()
    return fig


def figure_seam_metric(
    *,
    flatfield: np.ndarray,
    tiles_raw: np.ndarray,
    tiles_cor: np.ndarray,
    overlap_fraction: float = 0.2,
    orientation: str = "horizontal",
    title: str | None = None,
) -> Figure:
    """Explain the seam-consistency metric: illumination field + mosaic row before/after BaSiC.

    Three-panel figure that shows *why* adjacent tiles have a seam and how
    BaSiC removes it:

    * **left** (3-D surface): the BaSiC-estimated flat-field — the spatially
      non-uniform illumination field whose curvature causes the brightness
      mismatch between neighbouring tiles;
    * **middle** (2-D image): a row of raw tiles stitched side-by-side — the
      vignette repeats across tiles, producing visible seams at boundaries
      (dashed red lines);
    * **right** (2-D image): the same row after BaSiC correction — the
      illumination variation is removed and the row is seamless (dashed green
      lines).

    The seam-L1 metric (mean absolute normalised difference over the overlap
    strip) is averaged across all adjacent pairs and annotated on both image
    panels.

    Parameters
    ----------
    flatfield : numpy.ndarray, shape (th, tw)
        BaSiC-estimated illumination flat-field.
    tiles_raw : numpy.ndarray, shape (n_tiles, th, tw)
        Row of raw (uncorrected) tiles in display order.
    tiles_cor : numpy.ndarray, shape (n_tiles, th, tw)
        Row of BaSiC-corrected tiles (same ordering).
    overlap_fraction : float, optional
        Fraction of the tile size used as the overlap strip for the seam-L1
        metric (default 0.2).
    orientation : {"horizontal", "vertical"}, optional
        Tile adjacency direction (default ``"horizontal"``).
    title : str, optional
        Figure suptitle.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled seam-metric figure.
    """
    set_theme()
    ff = np.asarray(flatfield, dtype=np.float32)
    raw = np.asarray(tiles_raw, dtype=np.float32)
    cor = np.asarray(tiles_cor, dtype=np.float32)
    n_tiles, th, tw = raw.shape

    ovl_a, ovl_b, *_ = _overlap_slices((th, tw), orientation, overlap_fraction)
    seam_raw_mean = float(np.mean([_seam_l1(raw[i], raw[i + 1], ovl_a, ovl_b) for i in range(n_tiles - 1)]))
    seam_cor_mean = float(np.mean([_seam_l1(cor[i], cor[i + 1], ovl_a, ovl_b) for i in range(n_tiles - 1)]))
    improvement = _improvement_pct(seam_raw_mean, seam_cor_mean)

    if orientation == "horizontal":
        row_raw = np.concatenate(list(raw), axis=1)  # (th, n_tiles * tw)
        row_cor = np.concatenate(list(cor), axis=1)
        boundaries = [tw * (i + 1) for i in range(n_tiles - 1)]
        draw_vertical = True
        xlabel, ylabel = "x-pixel", "y-pixel"
    else:
        row_raw = np.concatenate(list(raw), axis=0)  # (n_tiles * th, tw)
        row_cor = np.concatenate(list(cor), axis=0)
        boundaries = [th * (i + 1) for i in range(n_tiles - 1)]
        draw_vertical = False
        xlabel, ylabel = "y-pixel", "x-pixel"

    vmin = 0.0
    vmax = float(np.percentile(row_raw, 99))

    fig = plt.figure(figsize=(18, 5))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax_raw = fig.add_subplot(1, 3, 2)
    ax_cor = fig.add_subplot(1, 3, 3)

    # --- Panel 1: 3-D flat-field surface ---
    yy, xx = np.mgrid[0 : ff.shape[0], 0 : ff.shape[1]]
    surf = ax3d.plot_surface(xx, yy, ff, cmap=FLATFIELD_CMAP, rcount=60, ccount=60, linewidth=0, antialiased=True)
    ax3d.set_xlabel("x-pixel", fontsize=7, labelpad=2)
    ax3d.set_ylabel("y-pixel", fontsize=7, labelpad=2)
    ax3d.set_zlabel("gain", fontsize=7, labelpad=2)
    ax3d.set_title("BaSiC flat-field\n(illumination curvature)", fontsize=9)
    ax3d.view_init(elev=32, azim=-58)
    ax3d.tick_params(labelsize=6)
    fig.colorbar(surf, ax=ax3d, fraction=0.025, pad=0.1, shrink=0.65)

    # --- Panels 2 & 3: mosaic row before/after ---
    def _show_row(ax: plt.Axes, img: np.ndarray, label: str, color: str) -> None:
        ax.imshow(img, cmap=INTENSITY_CMAP, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        for b in boundaries:
            if draw_vertical:
                ax.axvline(b - 0.5, color=color, lw=1.2, ls="--", alpha=0.85)
            else:
                ax.axhline(b - 0.5, color=color, lw=1.2, ls="--", alpha=0.85)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)

    _show_row(ax_raw, row_raw, f"Raw mosaic row  (mean seam L1 = {seam_raw_mean:.3f})", "tab:red")
    _show_row(ax_cor, row_cor, f"Corrected  (mean seam L1 = {seam_cor_mean:.3f}, {improvement:+.0f}%)", "tab:green")

    head = f"seam L1  {seam_raw_mean:.3f} \u2192 {seam_cor_mean:.3f}  ({improvement:+.0f}%)"
    fig.suptitle(f"{title}\n{head}" if title else head, fontsize=10)
    fig.tight_layout()
    return fig


def figure_focal_volume(
    *,
    raw_side: np.ndarray,
    est_side: np.ndarray,
    corrected_side: np.ndarray,
    seam_before: np.ndarray,
    seam_after: np.ndarray,
    focal_z: int | None = None,
    title: str | None = None,
) -> Figure:
    """Visualise how BaSiC corrects a depth-varying illumination focal curve.

    In volumetric fluorescence microscopy the illumination field changes with
    depth: at the focal plane the field is nearly flat; away from focus the
    vignette deepens and the mean intensity drops following the Gaussian beam
    envelope.  BaSiC is applied independently at each z-level.

    The figure shows the volume *from the side* — a lateral x depth cross-
    section — so the lens-shaped focal curve is visible before correction and
    absent after.

    Layout (2x2):

    * [0,0]: raw illumination side-view — the focal curve (bright/flat at
      focus, dim/curved away from focus);
    * [0,1]: BaSiC-estimated flat-field side-view — what the algorithm learnt;
    * [1,0]: corrected illumination side-view — spatial non-uniformity removed;
    * [1,1]: seam-L1 per z before and after correction.

    Parameters
    ----------
    raw_side : numpy.ndarray, shape (n_z, n_x)
        Central row of the raw illumination profile at each z-level
        (field x z-dependent mean scale).
    est_side : numpy.ndarray, shape (n_z, n_x)
        Central row of the BaSiC-estimated flat-field at each z-level.
    corrected_side : numpy.ndarray, shape (n_z, n_x)
        Central row of the residual illumination after correction at each z.
    seam_before : numpy.ndarray, shape (n_z,)
        Seam discrepancy before correction at each z-level.
    seam_after : numpy.ndarray, shape (n_z,)
        Seam discrepancy after correction at each z-level.
    focal_z : int, optional
        Index of the focal z-level; marked with a dashed line on all panels.
    title : str, optional
        Overall figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled focal-volume figure.
    """
    set_theme()
    raw_s = np.asarray(raw_side, dtype=np.float32)
    est_s = np.asarray(est_side, dtype=np.float32)
    cor_s = np.asarray(corrected_side, dtype=np.float32)
    sb = np.asarray(seam_before, dtype=np.float64)
    sa = np.asarray(seam_after, dtype=np.float64)
    n_z, n_x = raw_s.shape

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if title:
        fig.suptitle(title, fontsize=11)

    # Shared colour limits for the raw and estimated panels so they are
    # directly comparable; corrected uses its own (narrower) range.
    vmin = float(min(raw_s.min(), est_s.min()))
    vmax = float(max(raw_s.max(), est_s.max()))

    def _side(ax: plt.Axes, data: np.ndarray, label: str, vn: float, vx: float) -> None:
        im = ax.imshow(
            data,
            aspect="auto",
            cmap=FLATFIELD_CMAP,
            vmin=vn,
            vmax=vx,
            interpolation="nearest",
        )
        ax.set_title(label)
        ax.set_xlabel("lateral position (px)")
        ax.set_ylabel("z-level (depth)")
        if focal_z is not None:
            ax.axhline(focal_z, color="white", lw=1.2, ls="--", alpha=0.85)
            ax.text(
                n_x * 0.98,
                focal_z - 0.4,
                "focal plane",
                color="white",
                fontsize=7,
                ha="right",
                va="bottom",
            )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="illumination gain")

    _side(axes[0, 0], raw_s, "Raw — illumination focal curve", vmin, vmax)
    _side(axes[0, 1], est_s, "BaSiC estimated flat-field per z", vmin, vmax)
    _side(
        axes[1, 0],
        cor_s,
        "Corrected — spatial non-uniformity removed",
        float(cor_s.min()),
        float(cor_s.max()),
    )

    # Seam metric per z — before vs after.
    z_idx = np.arange(n_z)
    ax = axes[1, 1]
    ax.plot(z_idx, sb, color="tab:red", lw=1.8, marker="o", ms=4, label="raw")
    ax.plot(z_idx, sa, color="tab:green", lw=1.8, marker="o", ms=4, label="corrected")
    if focal_z is not None:
        ax.axvline(focal_z, color="0.5", lw=1.0, ls="--", alpha=0.7)
    improvement = 100.0 * (sb.mean() - sa.mean()) / (sb.mean() + 1e-12)
    ax.set_title(f"Seam-consistency per z  ({improvement:+.0f}% avg. improvement)")
    ax.set_xlabel("z-level (depth)")
    ax.set_ylabel("seam discrepancy (a.u.)")
    ax.legend(fontsize=9)
    ax.margins(x=0.02)
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    return fig


def figure_tuning_history(
    *,
    trial_values: np.ndarray,
    flatfield: np.ndarray,
    seam_raw: float,
    seam_tuned: float,
) -> Figure:
    """Build the hyperparameter-tuning story figure.

    A 1x3 layout: the Optuna optimisation history (per-trial seam-L1 and the
    best-so-far envelope), the tuned flat-field, and a raw-vs-tuned
    seam-consistency bar chart.

    Parameters
    ----------
    trial_values : numpy.ndarray
        Seam-L1 objective value for each completed trial, in order.
    flatfield : numpy.ndarray
        Tuned flat-field.
    seam_raw, seam_tuned : float
        Seam-consistency metric before tuning and with the tuned parameters.

    Returns
    -------
    matplotlib.figure.Figure
        The tuning figure.
    """
    set_theme()
    values = np.asarray(trial_values, dtype=float)
    improvement = _improvement_pct(seam_raw, seam_tuned)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    if values.size:
        idx = np.arange(values.size)
        ax.scatter(idx, values, c="tab:blue", s=30, alpha=0.8, label="trial")
        ax.step(idx, np.minimum.accumulate(values), where="post", color="tab:red", lw=2, label="best so far")
        ax.legend(fontsize=9)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Seam L1 (lower = better)")
    ax.set_title("Optuna optimisation history")

    show_field(axes[1], flatfield, title="Tuned flat-field")

    ax = axes[2]
    bars = ax.bar(["raw", "tuned"], [seam_raw, seam_tuned], color=["tab:gray", "tab:green"], width=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylabel("Seam L1")
    ax.set_title(f"Seam consistency ({improvement:+.0f}%)")

    fig.suptitle("BaSiC hyperparameter tuning — minimising seam mismatch")
    fig.tight_layout()
    return fig
