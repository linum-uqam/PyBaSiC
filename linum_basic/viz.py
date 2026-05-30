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
    "figure_field_flatten",
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


def _normalised_profile(arr: np.ndarray, axis: int) -> np.ndarray:
    """Mean-collapse *arr* along *axis* and divide by its mean (mean -> 1)."""
    profile = np.asarray(arr, dtype=np.float32).mean(axis=axis)
    return profile / (profile.mean() + 1e-12)


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


def _draw_overlap_outline(ax: plt.Axes, ovl: tuple[slice, slice], shape: tuple[int, int]) -> None:
    """Outline an overlap region (given as array slices) on an image axes."""
    th, tw = shape
    rs, cs = ovl
    r0 = rs.start or 0
    r1 = rs.stop if rs.stop is not None else th
    c0 = cs.start or 0
    c1 = cs.stop if cs.stop is not None else tw
    ax.add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False, edgecolor="tab:orange", lw=1.8))


def _plot_seam_overlap(
    ax: plt.Axes, prof_a: np.ndarray, prof_b: np.ndarray, *, mismatch_color: str, title: str, xlabel: str
) -> None:
    """Plot tile-A vs tile-B seam profiles with the mismatch area shaded."""
    pos = np.arange(prof_a.size)
    ax.plot(pos, prof_a, color="tab:blue", lw=1.4, label="tile A overlap")
    ax.plot(pos, prof_b, color="tab:purple", lw=1.4, label="tile B overlap")
    ax.fill_between(pos, prof_a, prof_b, color=mismatch_color, alpha=0.25, label="mismatch")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean intensity")
    ax.legend(fontsize=8)
    ax.margins(x=0)


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
    raw_tile_a: np.ndarray,
    raw_tile_b: np.ndarray,
    flatfield: np.ndarray,
    darkfield: np.ndarray | None = None,
    orientation: str = "horizontal",
    overlap_fraction: float = 0.2,
    epsilon: float = 1e-6,
    pixel_size_mm: float | None = None,
    title: str | None = None,
) -> Figure:
    """Illustrate the seam-consistency metric on two adjacent tiles.

    Two neighbouring mosaic tiles physically overlap (by ``overlap_fraction``
    of the tile size) and therefore image the *same* tissue in that strip.
    Both tiles are dimmed by the *same* multiplicative illumination field, so
    before correction the overlapping pixels disagree wherever the field is
    not flat.  Dividing each tile by the shared flat-field removes the vignette
    and the overlap regions come into agreement — exactly what the ``seam_l1``
    metric measures (mean relative disagreement over the overlap; 0 = perfect).

    The figure is a 2x4 grid that tells this story end to end:

    * column 0: the shared flat-field (top) and the per-column tile profile
      (bottom), showing the raw vignette flattening after correction;
    * columns 1-2: tiles A and B, raw (top) and corrected (bottom), with the
      overlap strip outlined;
    * column 3: the intensity profile *along the seam* for A vs B, raw (top)
      and corrected (bottom).  The shaded gap between the two curves is the
      disagreement the metric penalises; it collapses after correction.

    Parameters
    ----------
    raw_tile_a, raw_tile_b : numpy.ndarray, shape (th, tw)
        The two raw (uncorrected) adjacent tiles, dimmed by ``flatfield``.
    flatfield : numpy.ndarray, shape (th, tw)
        The shared multiplicative illumination field (mean ~ 1).
    darkfield : numpy.ndarray, optional
        Shared additive dark-field.  Subtracted before dividing by the
        flat-field when provided.
    orientation : {"horizontal", "vertical"}, optional
        Tile adjacency.  ``"horizontal"`` (default) = left/right neighbours
        overlapping in columns; ``"vertical"`` = top/bottom neighbours
        overlapping in rows.
    overlap_fraction : float, optional
        Fraction of the tile size shared by the two tiles (default ``0.2``).
    epsilon : float, optional
        Stabiliser added to the flat-field before division (default ``1e-6``).
    pixel_size_mm : float, optional
        Physical pixel size for scale bars on the tile panels.
    title : str, optional
        Title prefix; the seam-L1 values are appended automatically.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled seam-metric demonstration figure.
    """
    set_theme()
    a_raw = np.asarray(raw_tile_a, dtype=np.float32)
    b_raw = np.asarray(raw_tile_b, dtype=np.float32)
    flat = np.asarray(flatfield, dtype=np.float32)
    dark = np.zeros_like(flat) if darkfield is None else np.asarray(darkfield, dtype=np.float32)

    a_cor = (a_raw - dark) / (flat + epsilon)
    b_cor = (b_raw - dark) / (flat + epsilon)

    th, tw = a_raw.shape
    ovl_a, ovl_b, reduce_axis, seam_axis_label = _overlap_slices((th, tw), orientation, overlap_fraction)

    seam_raw = _seam_l1(a_raw, b_raw, ovl_a, ovl_b)
    seam_cor = _seam_l1(a_cor, b_cor, ovl_a, ovl_b)
    improvement = _improvement_pct(seam_raw, seam_cor)

    prof_a_raw = a_raw[ovl_a].mean(axis=reduce_axis)
    prof_b_raw = b_raw[ovl_b].mean(axis=reduce_axis)
    prof_a_cor = a_cor[ovl_a].mean(axis=reduce_axis)
    prof_b_cor = b_cor[ovl_b].mean(axis=reduce_axis)

    tile_vmax = float(np.percentile(a_raw, 99))
    cor_vmax = float(np.percentile(np.concatenate([a_cor.ravel(), b_cor.ravel()]), 99))

    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    head = f"seam L1: {seam_raw:.3f} -> {seam_cor:.3f}  ({improvement:+.1f}%)"
    fig.suptitle(f"{title}\n{head}" if title else head)

    # Column 0 — shared field + flatness demonstration.
    show_field(axes[0, 0], flat, title="Shared illumination field")

    # Per-position tile profile perpendicular to the seam (the vignette axis):
    # raw is curved by the field, corrected is flat.  Normalise each curve by
    # its own mean so their shapes are directly comparable.
    vig_axis = 0 if orientation == "horizontal" else 1  # average along the seam
    raw_profile = _normalised_profile(a_raw, vig_axis)
    cor_profile = _normalised_profile(a_cor, vig_axis)
    pos = np.arange(raw_profile.size)
    ax = axes[1, 0]
    ax.plot(pos, raw_profile, color="tab:red", lw=1.4, label="raw (vignetted)")
    ax.plot(pos, cor_profile, color="tab:green", lw=1.4, label="corrected (flat)")
    ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax.set_title("Tile profile across the field")
    ax.set_xlabel("x-pixel" if orientation == "horizontal" else "y-pixel")
    ax.set_ylabel("normalised intensity")
    ax.legend(fontsize=8)
    ax.margins(x=0)

    # Columns 1-2 — the two tiles, raw and corrected, overlap outlined.
    show_image(axes[0, 1], a_raw, title="Tile A: raw", vmin=0, vmax=tile_vmax, pixel_size_mm=pixel_size_mm, scalebar=True)
    _draw_overlap_outline(axes[0, 1], ovl_a, (th, tw))
    show_image(axes[0, 2], b_raw, title="Tile B: raw", vmin=0, vmax=tile_vmax, pixel_size_mm=pixel_size_mm, scalebar=True)
    _draw_overlap_outline(axes[0, 2], ovl_b, (th, tw))
    show_image(axes[1, 1], a_cor, title="Tile A: corrected", vmin=0, vmax=cor_vmax)
    _draw_overlap_outline(axes[1, 1], ovl_a, (th, tw))
    show_image(axes[1, 2], b_cor, title="Tile B: corrected", vmin=0, vmax=cor_vmax)
    _draw_overlap_outline(axes[1, 2], ovl_b, (th, tw))

    # Column 3 — seam overlap agreement, raw vs corrected.
    _plot_seam_overlap(
        axes[0, 3],
        prof_a_raw,
        prof_b_raw,
        mismatch_color="tab:red",
        title=f"Seam overlap: raw (L1={seam_raw:.3f})",
        xlabel=seam_axis_label,
    )
    _plot_seam_overlap(
        axes[1, 3],
        prof_a_cor,
        prof_b_cor,
        mismatch_color="tab:green",
        title=f"Seam overlap: corrected (L1={seam_cor:.3f})",
        xlabel=seam_axis_label,
    )

    fig.tight_layout()
    return fig


def figure_field_flatten(
    *,
    raw_tile: np.ndarray,
    flatfield: np.ndarray,
    darkfield: np.ndarray | None = None,
    epsilon: float = 1e-6,
    pixel_size_mm: float | None = None,
    title: str | None = None,
) -> Figure:
    r"""Visualise the illumination field going from curved (raw) to flat (corrected).

    The raw tile is dimmed by a spatially varying ``flatfield`` $S$, so the
    effective illumination across the tile is *curved* — bright in the centre,
    dim at the edges.  Dividing by $S$ flattens that field to a uniform gain of
    one.  This figure shows the real tile before/after correction alongside the
    field's 3-D surface and cross-sections so the curved-to-flat change is
    explicit.

    Layout (2x3):

    * column 0: the raw tile (top) and corrected tile (bottom) as 2-D images;
    * column 1: the illumination field $S$ as a curved 3-D surface (top) and the
      uniform corrected field $S / S \\equiv 1$ as a flat surface (bottom),
      drawn on a shared vertical scale;
    * column 2: horizontal (top) and vertical (bottom) cross-sections through
      the field centre, curved field vs the flat corrected field.

    Parameters
    ----------
    raw_tile : numpy.ndarray, shape (h, w)
        The raw (uncorrected) tile, dimmed by ``flatfield``.
    flatfield : numpy.ndarray, shape (h, w)
        The multiplicative illumination field (mean ~ 1).
    darkfield : numpy.ndarray, optional
        Additive dark-field, subtracted before dividing by the flat-field.
    epsilon : float, optional
        Stabiliser added to the flat-field before division (default ``1e-6``).
    pixel_size_mm : float, optional
        Physical pixel size for a scale bar on the raw tile panel.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled field-flattening figure.
    """
    set_theme()
    raw = np.asarray(raw_tile, dtype=np.float32)
    flat = np.asarray(flatfield, dtype=np.float32)
    dark = np.zeros_like(flat) if darkfield is None else np.asarray(darkfield, dtype=np.float32)
    cor = (raw - dark) / (flat + epsilon)

    h, w = raw.shape
    # The illumination field itself: curved (raw) and the flat unit field that
    # remains after dividing the tile by it.
    raw_field = flat / (flat.mean() + 1e-12)
    flat_field = np.ones_like(flat)
    zmin = float(min(raw_field.min(), 1.0)) - 0.05
    zmax = float(max(raw_field.max(), 1.0)) + 0.05

    raw_vmax = float(np.percentile(raw, 99))
    cor_vmax = float(np.percentile(cor, 99))

    fig = plt.figure(figsize=(16, 9))
    head = "illumination field: curved (raw) to flat (corrected)"
    fig.suptitle(f"{title}\n{head}" if title else head)

    # Column 0 — 2-D tiles, before and after correction.
    ax_raw = fig.add_subplot(2, 3, 1)
    show_image(ax_raw, raw, title="Raw tile", vmin=0, vmax=raw_vmax, pixel_size_mm=pixel_size_mm, scalebar=True)
    ax_cor = fig.add_subplot(2, 3, 4)
    show_image(ax_cor, cor, title="Corrected tile", vmin=0, vmax=cor_vmax)

    # Column 1 — the illumination field surface, curved then flat.
    yy, xx = np.mgrid[0:h, 0:w]
    ax_s_raw = fig.add_subplot(2, 3, 2, projection="3d")
    ax_s_raw.plot_surface(xx, yy, raw_field, cmap=FLATFIELD_CMAP, rcount=60, ccount=60, linewidth=0, antialiased=True)
    ax_s_raw.set_zlim(zmin, zmax)
    ax_s_raw.set_title("Illumination field (curved)")
    ax_s_raw.set_xlabel("x")
    ax_s_raw.set_ylabel("y")
    ax_s_raw.set_zlabel("gain")
    ax_s_raw.view_init(elev=32, azim=-58)

    ax_s_cor = fig.add_subplot(2, 3, 5, projection="3d")
    ax_s_cor.plot_surface(xx, yy, flat_field, cmap=FLATFIELD_CMAP, rcount=60, ccount=60, linewidth=0, antialiased=True)
    ax_s_cor.set_zlim(zmin, zmax)
    ax_s_cor.set_title("Corrected field (flat)")
    ax_s_cor.set_xlabel("x")
    ax_s_cor.set_ylabel("y")
    ax_s_cor.set_zlabel("gain")
    ax_s_cor.view_init(elev=32, azim=-58)

    # Column 2 — central field cross-sections, curved vs flat.
    mid_row = raw_field[h // 2, :]
    mid_col = raw_field[:, w // 2]
    ax_h = fig.add_subplot(2, 3, 3)
    ax_h.plot(mid_row, color="tab:red", lw=1.6, label="field (curved)")
    ax_h.axhline(1.0, color="tab:green", lw=1.6, label="corrected (flat)")
    ax_h.set_title("Horizontal field profile")
    ax_h.set_xlabel("x-pixel")
    ax_h.set_ylabel("gain")
    ax_h.legend(fontsize=8)
    ax_h.margins(x=0)

    ax_v = fig.add_subplot(2, 3, 6)
    ax_v.plot(mid_col, color="tab:red", lw=1.6, label="field (curved)")
    ax_v.axhline(1.0, color="tab:green", lw=1.6, label="corrected (flat)")
    ax_v.set_title("Vertical field profile")
    ax_v.set_xlabel("y-pixel")
    ax_v.set_ylabel("gain")
    ax_v.legend(fontsize=8)
    ax_v.margins(x=0)

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
