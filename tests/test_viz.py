"""Tests for linum_basic.viz — plotting helpers.

All tests use the non-interactive Agg backend so no display is required.
Assertions are smoke-level: figures are created without exceptions and the
returned objects have the expected types.  Pixel-level accuracy is not tested.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # must come before any pyplot import

import matplotlib.pyplot as plt

from linum_basic.viz import (
    Panel,
    add_scalebar,
    aip_preview,
    field_surface_3d,
    figure_apply_correction,
    figure_focal_volume,
    figure_panels,
    figure_seam_metric,
    figure_tuning_history,
    save_figure,
    set_theme,
    show_field,
    show_image,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def flat_field() -> np.ndarray:
    """Small smooth flat-field."""
    y, x = np.mgrid[-1:1:16j, -1:1:16j]  # type: ignore[misc]
    ff = (0.7 + 0.3 * np.exp(-(x**2 + y**2))).astype(np.float32)
    return ff / float(ff.mean())


@pytest.fixture()
def dark_field() -> np.ndarray:
    return np.full((16, 16), 0.05, dtype=np.float32)


@pytest.fixture()
def tile_stack(flat_field: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(42)
    n = 6
    tiles = rng.random((n, 16, 16)).astype(np.float32) * 0.5 + 0.3
    return tiles * flat_field[np.newaxis]


@pytest.fixture()
def volume() -> np.ndarray:
    """A 3-D volume (Z, H, W)."""
    return np.random.default_rng(7).random((4, 32, 48)).astype(np.float32)


def _new_ax() -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(1, 1)


# ---------------------------------------------------------------------------
# set_theme
# ---------------------------------------------------------------------------


class TestSetTheme:
    def test_runs_without_error(self) -> None:
        set_theme()


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------


class TestSaveFigure:
    def test_creates_file(self, tmp_path) -> None:
        fig, ax = plt.subplots()
        ax.plot([1, 2])
        out = save_figure(fig, tmp_path / "fig.png")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path) -> None:
        fig, ax = plt.subplots()
        ax.plot([1])
        out = save_figure(fig, tmp_path / "sub" / "dir" / "fig.png")
        assert out.exists()

    def test_returns_path(self, tmp_path) -> None:
        fig, _ax = plt.subplots()
        result = save_figure(fig, tmp_path / "r.png")
        assert isinstance(result, type(tmp_path / "r.png"))


# ---------------------------------------------------------------------------
# add_scalebar
# ---------------------------------------------------------------------------


class TestAddScalebar:
    def test_none_pixel_size_returns_none(self) -> None:
        fig, ax = _new_ax()
        assert add_scalebar(ax, None) is None
        plt.close(fig)

    def test_zero_pixel_size_returns_none(self) -> None:
        fig, ax = _new_ax()
        assert add_scalebar(ax, 0.0) is None
        plt.close(fig)

    def test_positive_pixel_size_returns_something_or_none(self) -> None:
        """Returns a ScaleBar or None depending on matplotlib_scalebar availability."""
        fig, ax = _new_ax()
        add_scalebar(ax, 0.01)
        # acceptable: either a scalebar object or None (if library not installed)
        plt.close(fig)


# ---------------------------------------------------------------------------
# show_field
# ---------------------------------------------------------------------------


class TestShowField:
    def test_returns_axes_image(self, flat_field) -> None:
        fig, ax = _new_ax()
        im = show_field(ax, flat_field)
        assert im is not None
        plt.close(fig)

    def test_darkfield_mode(self, dark_field) -> None:
        fig, ax = _new_ax()
        im = show_field(ax, dark_field, darkfield=True)
        assert im is not None
        plt.close(fig)

    def test_no_contours(self, flat_field) -> None:
        fig, ax = _new_ax()
        show_field(ax, flat_field, contours=False)
        plt.close(fig)

    def test_no_colorbar(self, flat_field) -> None:
        fig, ax = _new_ax()
        show_field(ax, flat_field, colorbar=False)
        plt.close(fig)

    def test_with_title(self, flat_field) -> None:
        fig, ax = _new_ax()
        show_field(ax, flat_field, title="Test flat-field")
        assert ax.get_title() == "Test flat-field"
        plt.close(fig)

    def test_vmin_vmax(self, flat_field) -> None:
        fig, ax = _new_ax()
        show_field(ax, flat_field, vmin=0.8, vmax=1.2)
        plt.close(fig)


# ---------------------------------------------------------------------------
# show_image
# ---------------------------------------------------------------------------


class TestShowImage:
    def test_returns_axes_image(self, tile_stack) -> None:
        fig, ax = _new_ax()
        im = show_image(ax, tile_stack[0])
        assert im is not None
        plt.close(fig)

    def test_with_title_and_colorbar(self, tile_stack) -> None:
        fig, ax = _new_ax()
        show_image(ax, tile_stack[0], title="Tile 0", colorbar=True)
        assert ax.get_title() == "Tile 0"
        plt.close(fig)

    def test_scalebar_no_pixel_size(self, tile_stack) -> None:
        """scalebar=True with pixel_size_mm=None silently does nothing."""
        fig, ax = _new_ax()
        show_image(ax, tile_stack[0], scalebar=True, pixel_size_mm=None)
        plt.close(fig)

    def test_scalebar_with_pixel_size(self, tile_stack) -> None:
        fig, ax = _new_ax()
        show_image(ax, tile_stack[0], scalebar=True, pixel_size_mm=0.01)
        plt.close(fig)


# ---------------------------------------------------------------------------
# field_surface_3d
# ---------------------------------------------------------------------------


class TestFieldSurface3d:
    def test_returns_fig_and_ax(self, flat_field) -> None:
        fig, ax = field_surface_3d(flat_field)
        assert hasattr(ax, "plot_surface")
        plt.close(fig)

    def test_darkfield_mode(self, dark_field) -> None:
        fig, _ax = field_surface_3d(dark_field, darkfield=True)
        plt.close(fig)

    def test_with_title_and_zlabel(self, flat_field) -> None:
        fig, _ax = field_surface_3d(flat_field, title="Surface", zlabel="intensity")
        plt.close(fig)


# ---------------------------------------------------------------------------
# aip_preview
# ---------------------------------------------------------------------------


class TestAipPreview:
    def test_returns_figure(self, volume) -> None:
        fig = aip_preview(volume)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_axis(self, volume) -> None:
        fig = aip_preview(volume, axis=1)
        plt.close(fig)

    def test_with_title(self, volume) -> None:
        fig = aip_preview(volume, title="AIP test")
        plt.close(fig)

    def test_with_pixel_size(self, volume) -> None:
        fig = aip_preview(volume, pixel_size_mm=0.01)
        plt.close(fig)

    def test_raises_on_2d(self) -> None:
        with pytest.raises(ValueError, match="3-D"):
            aip_preview(np.zeros((4, 4)))

    def test_uniform_volume_no_crash(self) -> None:
        """Edge case: uniform volume where vmin == vmax."""
        fig = aip_preview(np.ones((3, 8, 8), dtype=np.float32))
        plt.close(fig)


# ---------------------------------------------------------------------------
# figure_panels
# ---------------------------------------------------------------------------


class TestFigurePanels:
    def test_image_panel(self, tile_stack) -> None:
        panels = [Panel(data=tile_stack[0], title="Raw", kind="image")]
        fig = figure_panels(panels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_flatfield_panel(self, flat_field) -> None:
        panels = [Panel(data=flat_field, title="Flat", kind="flatfield")]
        fig = figure_panels(panels)
        plt.close(fig)

    def test_darkfield_panel(self, dark_field) -> None:
        panels = [Panel(data=dark_field, title="Dark", kind="darkfield")]
        fig = figure_panels(panels)
        plt.close(fig)

    def test_multi_panel_with_ncols(self, tile_stack, flat_field) -> None:
        panels = [Panel(data=tile_stack[i], title=f"t{i}", kind="image") for i in range(4)]
        fig = figure_panels(panels, ncols=2, suptitle="Grid")
        plt.close(fig)

    def test_image_with_scalebar(self, tile_stack) -> None:
        panels = [Panel(data=tile_stack[0], kind="image", scalebar=True, pixel_size_mm=0.01)]
        fig = figure_panels(panels)
        plt.close(fig)


# ---------------------------------------------------------------------------
# figure_apply_correction
# ---------------------------------------------------------------------------


class TestFigureApplyCorrection:
    def _make_mosaic(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(5)
        raw = rng.random((32, 32)).astype(np.float32) + 0.3
        corrected = raw * 1.1
        return raw, corrected

    def test_returns_figure_no_darkfield(self) -> None:
        y, x = np.mgrid[-1:1:16j, -1:1:16j]  # type: ignore[misc]
        ff = (0.7 + 0.3 * np.exp(-(x**2 + y**2))).astype(np.float32)
        raw, cor = self._make_mosaic()
        fig = figure_apply_correction(
            flatfield=ff,
            raw_mosaic=raw,
            corrected_mosaic=cor,
            raw_tile=raw[:16, :16],
            corrected_tile=cor[:16, :16],
            seam_raw=0.05,
            seam_corrected=0.02,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_figure_with_darkfield(self) -> None:
        y, x = np.mgrid[-1:1:16j, -1:1:16j]  # type: ignore[misc]
        ff = (0.7 + 0.3 * np.exp(-(x**2 + y**2))).astype(np.float32)
        df = np.full((16, 16), 0.05, dtype=np.float32)
        raw, cor = self._make_mosaic()
        fig = figure_apply_correction(
            flatfield=ff,
            raw_mosaic=raw,
            corrected_mosaic=cor,
            raw_tile=raw[:16, :16],
            corrected_tile=cor[:16, :16],
            seam_raw=0.05,
            seam_corrected=0.02,
            darkfield=df,
            title="Test",
        )
        plt.close(fig)

    def test_with_pixel_size(self) -> None:
        y, x = np.mgrid[-1:1:16j, -1:1:16j]  # type: ignore[misc]
        ff = (0.7 + 0.3 * np.exp(-(x**2 + y**2))).astype(np.float32)
        raw, cor = self._make_mosaic()
        fig = figure_apply_correction(
            flatfield=ff,
            raw_mosaic=raw,
            corrected_mosaic=cor,
            raw_tile=raw[:16, :16],
            corrected_tile=cor[:16, :16],
            seam_raw=0.05,
            seam_corrected=0.02,
            pixel_size_mm=0.01,
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# figure_seam_metric
# ---------------------------------------------------------------------------


class TestFigureSeamMetric:
    def _make_tiles(self, n: int = 4, tile: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(9)
        y, x = np.mgrid[-1 : 1 : tile * 1j, -1 : 1 : tile * 1j]  # type: ignore[misc]
        ff = (0.7 + 0.3 * np.exp(-(x**2 + y**2))).astype(np.float32)
        ff /= float(ff.mean())
        raw = rng.random((n, tile, tile)).astype(np.float32) + 0.3
        raw_shaded = raw * ff[np.newaxis]
        corrected = raw_shaded / (ff[np.newaxis] + 1e-6)
        return ff, raw_shaded, corrected

    def test_horizontal_orientation(self) -> None:
        ff, raw, cor = self._make_tiles()
        fig = figure_seam_metric(
            flatfield=ff,
            tiles_raw=raw,
            tiles_cor=cor,
            orientation="horizontal",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_vertical_orientation(self) -> None:
        ff, raw, cor = self._make_tiles()
        fig = figure_seam_metric(
            flatfield=ff,
            tiles_raw=raw,
            tiles_cor=cor,
            orientation="vertical",
        )
        plt.close(fig)

    def test_with_title(self) -> None:
        ff, raw, cor = self._make_tiles()
        fig = figure_seam_metric(
            flatfield=ff,
            tiles_raw=raw,
            tiles_cor=cor,
            title="Seam test",
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# figure_focal_volume
# ---------------------------------------------------------------------------


class TestFigureFocalVolume:
    def _make_data(self, n_z: int = 6, n_x: int = 16) -> dict:
        rng = np.random.default_rng(11)
        return {
            "raw_side": rng.random((n_z, n_x)).astype(np.float32) + 0.3,
            "est_side": rng.random((n_z, n_x)).astype(np.float32) + 0.5,
            "corrected_side": rng.random((n_z, n_x)).astype(np.float32) + 0.2,
            "seam_before": rng.random(n_z).astype(np.float32) * 0.1 + 0.05,
            "seam_after": rng.random(n_z).astype(np.float32) * 0.03,
        }

    def test_returns_figure(self) -> None:
        d = self._make_data()
        fig = figure_focal_volume(**d)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_focal_z(self) -> None:
        d = self._make_data()
        fig = figure_focal_volume(**d, focal_z=3)
        plt.close(fig)

    def test_with_title(self) -> None:
        d = self._make_data()
        fig = figure_focal_volume(**d, title="Focal curve")
        plt.close(fig)


# ---------------------------------------------------------------------------
# figure_tuning_history
# ---------------------------------------------------------------------------


class TestFigureTuningHistory:
    def test_returns_figure(self, flat_field) -> None:
        values = np.array([0.08, 0.07, 0.06, 0.05, 0.05])
        fig = figure_tuning_history(
            trial_values=values,
            flatfield=flat_field,
            seam_raw=0.1,
            seam_tuned=0.05,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_trials(self, flat_field) -> None:
        """Empty trial array does not crash."""
        fig = figure_tuning_history(
            trial_values=np.array([]),
            flatfield=flat_field,
            seam_raw=0.1,
            seam_tuned=0.05,
        )
        plt.close(fig)
