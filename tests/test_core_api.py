"""Tests for BaSiC public API surface — inputs, field accessors, and guards.

Covers input types (files_list, images_list, TypeError), validate_fields()
warnings, write_images() error guards, set/get flatfield/darkfield, and the
warm_start_reweighting path.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

from linum_basic.core import BaSiC

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tiff_stack(directory: Path, n: int = 6, size: int = 32) -> list[Path]:
    """Write *n* synthetic TIFF images to *directory* and return their paths."""
    rng = np.random.default_rng(99)
    paths = []
    for i in range(n):
        img = (rng.random((size, size)) * 60000).astype(np.uint16)
        p = directory / f"tile_{i:04d}.tif"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def _small_stack(n: int = 10, size: int = 32) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.random((n, size, size)).astype(np.float32)


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


class TestInputTypes:
    """BaSiC accepts directories, file lists, ndarray lists, and stacks."""

    def test_files_list_input(self, tmp_path: Path) -> None:
        """A list of Path objects is accepted and triggers 'files_list' input_type."""
        paths = _write_tiff_stack(tmp_path, n=6, size=32)
        model = BaSiC(paths)
        assert model.input_type == "files_list"
        assert len(model.files) == 6

    def test_files_list_string_paths(self, tmp_path: Path) -> None:
        """A list of string paths is converted to Path objects."""
        paths = _write_tiff_stack(tmp_path, n=4, size=32)
        model = BaSiC([str(p) for p in paths])
        assert model.input_type == "files_list"

    def test_images_list_input(self) -> None:
        """A list of ndarrays is stacked and triggers 'images_list' input_type."""
        frames = [np.random.default_rng(i).random((32, 32)).astype(np.float32) for i in range(8)]
        model = BaSiC(frames)
        assert model.input_type == "images_list"
        assert model.img_stack.shape == (8, 32, 32)

    def test_invalid_input_raises_type_error(self) -> None:
        """Unsupported input type raises TypeError with descriptive message."""
        with pytest.raises(TypeError, match="input must be"):
            BaSiC(42)  # ty: ignore[invalid-argument-type]

    def test_invalid_empty_list_raises_type_error(self) -> None:
        """Empty list raises TypeError (no elements to infer type from)."""
        with pytest.raises(TypeError, match="input must be"):
            BaSiC([])

    def test_files_list_runs_end_to_end(self, tmp_path: Path) -> None:
        """BaSiC initialised from a file list can be trained and returns fields."""
        paths = _write_tiff_stack(tmp_path, n=8, size=32)
        model = BaSiC(paths)
        model.working_size = 32
        model.run()
        ff = model.get_flatfield()
        assert ff.shape == (32, 32)
        assert np.isfinite(ff).all()

    def test_images_list_runs_end_to_end(self) -> None:
        """BaSiC initialised from a list of ndarrays trains without error."""
        frames = [np.random.default_rng(i).random((32, 32)).astype(np.float32) + 0.2 for i in range(10)]
        model = BaSiC(frames)
        model.working_size = 32
        model.run()
        ff = model.get_flatfield()
        assert ff.shape == (32, 32)

    def test_load_images_verbose(self, tmp_path: Path, capsys) -> None:
        """verbose=True produces tqdm output during image loading."""
        paths = _write_tiff_stack(tmp_path, n=4, size=32)
        model = BaSiC(paths, verbose=True)
        model.working_size = 32
        # Just prepare() — we don't need the full run
        model.prepare()
        # After prepare(), n_images should be correct regardless of tqdm output
        assert model.n_images == 4


# ---------------------------------------------------------------------------
# validate_fields()
# ---------------------------------------------------------------------------


class TestValidateFields:
    """validate_fields() issues UserWarnings for problematic field values."""

    def _model_with_fields(
        self,
        flatfield: np.ndarray,
        darkfield: np.ndarray,
    ) -> BaSiC:
        """Create a BaSiC with pre-set full-size fields."""
        h, w = flatfield.shape
        stack = np.ones((2, h, w), dtype=np.float32)
        model = BaSiC(stack)
        model.image_shape = (h, w)
        model.flatfield_fullsize = flatfield.copy()
        model.darkfield_fullsize = darkfield.copy()
        return model

    def test_nan_flatfield_warns(self) -> None:
        """NaN in flat-field triggers UserWarning."""
        flat = np.ones((8, 8), dtype=np.float32)
        flat[3, 3] = float("nan")
        dark = np.zeros((8, 8), dtype=np.float32)
        model = self._model_with_fields(flat, dark)

        with pytest.warns(UserWarning, match="non-finite"):
            model.validate_fields()

    def test_non_positive_flatfield_warns(self) -> None:
        """Non-positive values in flat-field trigger UserWarning."""
        flat = np.ones((8, 8), dtype=np.float32)
        flat[0, 0] = -0.1
        dark = np.zeros((8, 8), dtype=np.float32)
        model = self._model_with_fields(flat, dark)

        with pytest.warns(UserWarning, match="non-positive"):
            model.validate_fields()

    def test_nan_darkfield_warns(self) -> None:
        """NaN in dark-field triggers UserWarning."""
        flat = np.ones((8, 8), dtype=np.float32)
        dark = np.zeros((8, 8), dtype=np.float32)
        dark[1, 1] = float("nan")
        model = self._model_with_fields(flat, dark)

        with pytest.warns(UserWarning, match="dark-field.*non-finite"):
            model.validate_fields()

    def test_valid_fields_no_warning(self) -> None:
        """Valid flat/dark-fields produce no UserWarning."""
        flat = np.ones((8, 8), dtype=np.float32) * 0.9
        dark = np.zeros((8, 8), dtype=np.float32)
        model = self._model_with_fields(flat, dark)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            model.validate_fields()  # must not raise


# ---------------------------------------------------------------------------
# write_images() error guards
# ---------------------------------------------------------------------------


class TestWriteImagesGuards:
    """write_images() raises informative errors for unsupported configurations."""

    def test_no_files_attribute_raises(self, tmp_path: Path) -> None:
        """write_images() raises RuntimeError when BaSiC was built from a stack."""
        stack = _small_stack()
        model = BaSiC(stack)
        model.working_size = 32
        model.run()

        with pytest.raises(RuntimeError, match="write_images\\(\\) is only available"):
            model.write_images(tmp_path / "out")

    def test_write_images_works_from_directory(self, tmp_path: Path) -> None:
        """write_images() succeeds when input was a directory."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        _write_tiff_stack(in_dir, n=6, size=32)

        model = BaSiC(in_dir, extension=".tif")
        model.working_size = 32
        model.run()
        model.write_images(out_dir)

        output_files = list(out_dir.glob("*.tif"))
        assert len(output_files) == 6

    def test_write_images_creates_output_directory(self, tmp_path: Path) -> None:
        """write_images() creates the output directory if it does not exist."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_tiff_stack(in_dir, n=4, size=32)

        model = BaSiC(in_dir, extension=".tif")
        model.working_size = 32
        model.run()

        nested_out = tmp_path / "a" / "b" / "c"
        model.write_images(nested_out)
        assert nested_out.is_dir()
        assert len(list(nested_out.glob("*.tif"))) == 4


# ---------------------------------------------------------------------------
# set_flatfield / set_darkfield / get_flatfield / get_darkfield
# ---------------------------------------------------------------------------


class TestFieldAccessors:
    """set/get flatfield and darkfield resize and copy correctly."""

    def _prepared_model(self, size: int = 32) -> BaSiC:
        """Return a prepared (not run) model with known image_shape."""
        stack = _small_stack(n=8, size=size)
        model = BaSiC(stack)
        model.working_size = size
        model.prepare()
        return model

    def test_set_flatfield_accepts_different_size(self) -> None:
        """set_flatfield() resizes the supplied array to image_shape."""
        model = self._prepared_model(size=32)
        small_ff = np.ones((16, 16), dtype=np.float32) * 1.2
        model.set_flatfield(small_ff)
        ff = model.get_flatfield()
        assert ff.shape == (32, 32)

    def test_set_flatfield_preserves_values(self) -> None:
        """set_flatfield() with matching size preserves pixel values (approx)."""
        model = self._prepared_model(size=32)
        custom_ff = np.ones((32, 32), dtype=np.float32) * 1.5
        model.set_flatfield(custom_ff)
        ff = model.get_flatfield()
        np.testing.assert_allclose(ff, 1.5, atol=1e-4)

    def test_set_darkfield_accepts_different_size(self) -> None:
        """set_darkfield() resizes the supplied array to image_shape."""
        model = self._prepared_model(size=32)
        small_df = np.ones((16, 16), dtype=np.float32) * 0.05
        model.set_darkfield(small_df)
        df = model.get_darkfield()
        assert df.shape == (32, 32)

    def test_set_darkfield_preserves_values(self) -> None:
        """set_darkfield() with matching size preserves pixel values (approx)."""
        model = self._prepared_model(size=32)
        custom_df = np.ones((32, 32), dtype=np.float32) * 0.03
        model.set_darkfield(custom_df)
        df = model.get_darkfield()
        np.testing.assert_allclose(df, 0.03, atol=1e-4)

    def test_get_flatfield_returns_copy(self) -> None:
        """get_flatfield() returns a copy; mutating it does not affect model state."""
        model = self._prepared_model(size=32)
        model.run()
        ff1 = model.get_flatfield()
        ff1[:] = 99.0
        ff2 = model.get_flatfield()
        assert not np.any(ff2 == 99.0)

    def test_get_darkfield_returns_copy(self) -> None:
        """get_darkfield() returns a copy; mutating it does not affect model state."""
        model = self._prepared_model(size=32)
        model.run()
        df1 = model.get_darkfield()
        original_mean = df1.mean()
        df1[:] = 99.0
        df2 = model.get_darkfield()
        assert abs(df2.mean() - original_mean) < 1.0


# ---------------------------------------------------------------------------
# _apply_correction() — static method
# ---------------------------------------------------------------------------


class TestApplyCorrection:
    """_apply_correction() static method edge cases."""

    def test_float_input_no_clip(self) -> None:
        """Float32 inputs are not clipped when clip=True (no iinfo for floats)."""
        img = np.array([[[2.0, 3.0]]], dtype=np.float32)
        flat = np.ones((1, 2), dtype=np.float32)
        dark = np.zeros((1, 2), dtype=np.float32)
        result = BaSiC._apply_correction(img, flat, dark, 0.0, np.dtype("float32"), clip=True)
        np.testing.assert_allclose(result, img)

    def test_integer_input_is_clipped(self) -> None:
        """Integer dtypes are clipped to valid range when clip=True."""
        img = np.array([[[50000.0, 70000.0]]], dtype=np.float32)
        flat = np.ones((1, 2), dtype=np.float32) * 0.5  # divide by 0.5 → doubles values
        dark = np.zeros((1, 2), dtype=np.float32)
        result = BaSiC._apply_correction(img, flat, dark, 0.0, np.dtype("uint16"), clip=True)
        assert result.max() <= np.iinfo(np.uint16).max

    def test_clip_and_no_clip_differ_for_overflow(self) -> None:
        """clip=True clamps overflow values; clip=False lets them wrap/differ."""
        img = np.array([[[40000.0]]], dtype=np.float32)
        flat = np.ones((1, 1), dtype=np.float32) * 0.5  # 40000 / 0.5 = 80000 > 65535
        dark = np.zeros((1, 1), dtype=np.float32)
        clipped = BaSiC._apply_correction(img, flat, dark, 0.0, np.dtype("uint16"), clip=True)
        unclipped = BaSiC._apply_correction(img, flat, dark, 0.0, np.dtype("uint16"), clip=False)
        # clip=True must saturate at uint16 max
        assert float(clipped[0, 0, 0]) == np.iinfo(np.uint16).max
        # clip=False wraps, so the result differs from the saturated value
        assert float(unclipped[0, 0, 0]) != float(clipped[0, 0, 0])


# ---------------------------------------------------------------------------
# warm_start_reweighting
# ---------------------------------------------------------------------------


class TestWarmStartReweighting:
    """warm_start_reweighting=True stores and reuses ALM state."""

    def test_warm_start_completes(self) -> None:
        """Model with warm_start_reweighting=True converges without error."""
        stack = _small_stack(n=10, size=32)
        model = BaSiC(stack)
        model.working_size = 32
        model.warm_start_reweighting = True
        model.run()
        ff = model.get_flatfield()
        assert ff.shape == (32, 32)
        assert np.isfinite(ff).all()


# ---------------------------------------------------------------------------
# update() guard
# ---------------------------------------------------------------------------


class TestUpdateGuard:
    """update() raises RuntimeError when called before prepare()."""

    def test_update_without_prepare_raises(self) -> None:
        """Calling update() before prepare() raises RuntimeError for l_s/l_d."""
        stack = _small_stack(n=4, size=32)
        model = BaSiC(stack)
        model.working_size = 32
        # Set required attributes that update() needs besides l_s/l_d
        model.l_s = None
        model.l_d = None
        # We need img_sort to exist for update() to reach the guard
        model.prepare()
        model.l_s = None  # reset after prepare to trigger the guard
        with pytest.raises(RuntimeError, match="l_s and l_d must be set"):
            model.update()


# ---------------------------------------------------------------------------
# update() dark-field zero convergence guard (core invariant)
# ---------------------------------------------------------------------------


class TestDarkfieldZeroConvergenceGuard:
    """Pin core.py's dark-field zero guard in ``BaSiC.update()``.

    The invariant (``linum_basic/core.py``, ``BaSiC.update``)::

        if mad_dark_abs < 1e-7:
            mad_dark = 0.0
        elif last_dark_sum < 1e-7:
            mad_dark = 1.0  # previous estimate was zero; relative change undefined
        else:
            mad_dark = mad_dark_abs / last_dark_sum

    prevents the solver from falsely declaring convergence on the very first
    reweighting iterate, when the previous dark-field estimate is still
    all-zeros (as initialised by ``prepare()``).  Without the guard, the
    pre-fix formula ``mad_dark_abs / max(last_dark_sum, 1e-6)`` would divide
    a tiny dark-field change by the 1e-6 floor and report a small ratio,
    falsely satisfying ``reweighting_tolerance`` and stopping after a single
    iterate.

    These tests use a mocked ``inexact_alm_l1`` returning a near-zero
    dark-field on the first pass with a *loose* tolerance, constructing the
    only regime where old and new behaviour diverge (for the default
    ``reweighting_tolerance=1e-3`` the discriminating window is empty, so a
    loose tolerance is required).
    """

    # Discriminating dark-field change.  Must satisfy
    #   1e-7 <= mad_dark_abs <= tolerance * 1e-6
    # so the first (< 1e-7) branch is skipped, the zero-guard branch is taken
    # under the current invariant (mad_dark = 1.0 -> not converged), but the
    # pre-fix clamped ratio (mad_dark_abs / 1e-6) falls at or below tolerance
    # (false convergence).  3e-7 sits inside the [1e-7, 5e-7] window for the
    # 0.5 tolerance used below.
    _TARGET_MAD_DARK_ABS = 3e-7
    _LOOSE_TOLERANCE = 0.5

    def _make_prepared_model(self, ws: int = 32, n: int = 4) -> BaSiC:
        """Return a prepared BaSiC whose initial dark-field is all-zeros."""
        rng = np.random.default_rng(7)
        stack = rng.random((n, ws, ws)).astype(np.float32) + 0.2
        model = BaSiC(stack, estimate_darkfield=True)
        model.working_size = ws
        model.prepare()
        # Precondition for the zero-guard branch: prepare() initialises the
        # dark-field to zeros.
        assert model.darkfield.shape == (ws, ws)
        assert float(np.abs(model.darkfield).sum()) == 0.0
        return model

    def _fake_alm_factory(self, ws: int, n: int):
        """Build a mock ``inexact_alm_l1`` returning a near-zero dark-field.

        The returned dark-field has ``|D|.sum() == _TARGET_MAD_DARK_ABS`` so
        that ``mad_dark_abs`` (the change from the zero initial dark-field)
        equals the target and exercises the zero-guard branch.  ``Ib`` is all
        ones so ``Ib.mean(axis=0) - D_2d`` normalises back to ones, making
        ``mad_flat == 0`` and leaving ``mad_dark`` as the sole convergence
        determinant.
        """
        c = np.float32(self._TARGET_MAD_DARK_ABS / (ws * ws))
        Ib = np.ones((n, ws, ws), dtype=np.float32)
        Ir = np.zeros((n, ws, ws), dtype=np.float32)
        D = np.full(ws * ws, c, dtype=np.float32)
        alm_state = {"alm_iterations": 5}

        def _fake_alm(*args, **kwargs):
            return Ib, Ir, D, alm_state

        return _fake_alm

    def test_zero_darkfield_guard_prevents_false_convergence(self, monkeypatch) -> None:
        """Under the discriminating fixture the guard keeps reweighting alive.

        With a near-zero dark-field change (3e-7) and a loose tolerance (0.5),
        the current invariant sets ``mad_dark = 1.0`` (zero-guard branch), so
        ``max(mad_flat, 1.0) <= 0.5`` is False and ``_flag_reweighting`` stays
        True.  A regression to the pre-fix clamped ratio would falsely
        converge here (see ``test_fixture_discriminates_from_prefix_formula``).
        """
        ws, n = 32, 4
        model = self._make_prepared_model(ws=ws, n=n)
        model.reweighting_tolerance = self._LOOSE_TOLERANCE
        monkeypatch.setattr("linum_basic.core.inexact_alm_l1", self._fake_alm_factory(ws, n))
        model.update()
        assert model._flag_reweighting is True

    def test_fixture_discriminates_from_prefix_formula(self) -> None:
        """The fixture exercises the zero-guard and would converge pre-fix.

        Asserts the branch preconditions (``mad_dark_abs >= 1e-7`` so the
        first branch is skipped; ``last_dark_sum < 1e-7`` so the zero-guard is
        taken) and that the pre-fix clamped ratio falls at or below the loose
        tolerance -- i.e. the scenario genuinely distinguishes old vs new
        behaviour, so the main test cannot pass under the old formula.
        """
        mad_dark_abs = self._TARGET_MAD_DARK_ABS
        last_dark_sum = 0.0  # prepare() initialises the dark-field to zeros
        # Pre-fix formula: divide by the 1e-6 floor instead of the zero guard.
        prefix_mad_dark = mad_dark_abs / max(last_dark_sum, 1e-6)
        assert mad_dark_abs >= 1e-7, "fixture must skip the < 1e-7 first branch"
        assert last_dark_sum < 1e-7, "fixture must enter the zero-guard branch"
        assert prefix_mad_dark <= self._LOOSE_TOLERANCE, (
            "pre-fix formula must falsely converge on this fixture; "
            "otherwise the main test does not discriminate old vs new behaviour"
        )
