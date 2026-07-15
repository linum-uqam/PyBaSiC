"""CLI integration tests for :mod:`linum_basic.cli`.

Runs the ``basic`` entry-point as a subprocess against
a temporary directory of synthetic TIFF images.  Tests are isolated and
produce no side effects on the permanent filesystem.
"""

from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tiff_stack(directory: Path, n: int = 8, size: int = 64) -> None:
    """Write *n* synthetic TIFF images of shape *(size, size)* to *directory*.

    Parameters
    ----------
    directory : Path
        Target directory (must already exist).
    n : int
        Number of images to create.
    size : int
        Spatial side length in pixels.
    """
    rng = np.random.default_rng(7)
    for i in range(n):
        img = rng.integers(100, 60000, (size, size), dtype=np.uint16)
        cv2.imwrite(str(directory / f"img_{i:04d}.tif"), img)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCliHelp:
    """Smoke tests — verify the CLI is importable and shows help text."""

    def test_help_exits_zero(self) -> None:
        """``--help`` exits with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "linum_basic.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "basic" in result.stdout.lower() or "BaSiC" in result.stdout

    def test_missing_input_exits_nonzero(self, tmp_path: Path) -> None:
        """Missing subcommand or ``--input`` should cause a non-zero exit code."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "linum_basic.cli",
                "correct",
                "--input",
                str(tmp_path / "does_not_exist"),
                "--output",
                str(tmp_path / "out"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    @pytest.mark.parametrize("sub", ["correct", "fit", "tune"])
    def test_device_mps_rejected(self, tmp_path: Path, sub: str) -> None:
        """``--device mps`` is rejected early with a clear stderr message."""
        cmd = [
            sys.executable,
            "-m",
            "linum_basic.cli",
            sub,
            "--device",
            "mps",
        ]
        if sub == "correct":
            cmd.extend(["--input", str(tmp_path), "--output", str(tmp_path / "out")])
        elif sub == "fit":
            cmd.extend(["--input", str(tmp_path / "in.ome.zarr"), "--output", str(tmp_path / "out.ome.zarr")])
        else:  # tune
            cmd.extend(["--input", str(tmp_path / "in.ome.zarr")])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert "mps" in result.stderr.lower()


class TestCliEndToEnd:
    """End-to-end CLI tests against a temporary TIFF stack."""

    def test_output_images_created(self, tmp_path: Path) -> None:
        """Corrected images are written for every input image."""
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        _write_tiff_stack(in_dir, n=8, size=64)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "linum_basic.cli",
                "correct",
                "--input",
                str(in_dir),
                "--output",
                str(out_dir),
                "--extension",
                ".tif",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"CLI exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        output_files = list(out_dir.glob("*.tif"))
        assert len(output_files) == 8, f"Expected 8 output files, got {len(output_files)}"

    def test_output_images_are_readable(self, tmp_path: Path) -> None:
        """Every output TIFF is a valid image with the same dimensions as input."""
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        _write_tiff_stack(in_dir, n=4, size=64)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "linum_basic.cli",
                "correct",
                "--input",
                str(in_dir),
                "--output",
                str(out_dir),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )

        for path in sorted(out_dir.glob("*.tif")):
            img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
            assert img is not None, f"Could not read '{path}'"
            assert img.shape == (64, 64)

    def test_darkfield_flag_does_not_crash(self, tmp_path: Path) -> None:
        """CLI with ``--estimate-darkfield`` completes without error."""
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        _write_tiff_stack(in_dir, n=8, size=64)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "linum_basic.cli",
                "correct",
                "--input",
                str(in_dir),
                "--output",
                str(out_dir),
                "--estimate-darkfield",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"CLI with --estimate-darkfield exited {result.returncode}\nstderr:\n{result.stderr}"

    @pytest.mark.parametrize("backend", ["numpy"])
    def test_backend_flag(self, tmp_path: Path, backend: str) -> None:
        """``--backend <backend>`` flag does not raise."""
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        _write_tiff_stack(in_dir, n=4, size=32)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "linum_basic.cli",
                "correct",
                "--input",
                str(in_dir),
                "--output",
                str(out_dir),
                "--backend",
                backend,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"Backend '{backend}' failed.\nstderr:\n{result.stderr}"


class TestPreviewCli:
    """Tests for the ``basic preview`` AIP preview sub-command."""

    def test_preview_creates_png(self, tmp_path: Path) -> None:
        """``basic preview`` writes a non-empty PNG from a small OME-Zarr volume."""
        from linum_basic.cli import main
        from linum_basic.io.zarr import write_ome_zarr

        rng = np.random.default_rng(3)
        volume = rng.random((5, 32, 48), dtype=np.float32)
        zarr_path = tmp_path / "vol.ome.zarr"
        write_ome_zarr(zarr_path, volume, axes=["z", "y", "x"], scale=[2.0, 0.01, 0.01], overwrite=True)

        out_png = tmp_path / "preview.png"
        rc = main(["preview", "--input", str(zarr_path), "--output", str(out_png), "--dpi", "60"])

        assert rc == 0
        assert out_png.is_file()
        assert out_png.stat().st_size > 0

    def test_preview_help_exits_zero(self) -> None:
        """``basic preview --help`` exits with code 0."""
        from linum_basic.cli import main

        with pytest.raises(SystemExit) as excinfo:
            main(["preview", "--help"])
        assert excinfo.value.code == 0


class TestSubcommandHelp:
    """``--help`` for every sub-command exits with code 0."""

    @pytest.mark.parametrize("sub", ["correct", "fit", "tune", "preview"])
    def test_subcommand_help_exits_zero(self, sub: str) -> None:
        """``basic <sub> --help`` exits with code 0."""
        from linum_basic.cli import main

        with pytest.raises(SystemExit) as excinfo:
            main([sub, "--help"])
        assert excinfo.value.code == 0

    def test_correct_subcommand_in_help(self) -> None:
        """The top-level ``--help`` lists all four sub-commands."""
        result = subprocess.run(
            [sys.executable, "-m", "linum_basic.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        for sub in ("correct", "fit", "tune", "preview"):
            assert sub in result.stdout, f"Sub-command '{sub}' missing from help"

    def test_fit_help_shows_strategy(self) -> None:
        """``basic fit --help`` documents ``--strategy`` and its choices."""
        from linum_basic.cli import main

        buf = io.StringIO()
        with pytest.raises(SystemExit) as excinfo, redirect_stdout(buf):
            main(["fit", "--help"])
        assert excinfo.value.code == 0
        help_text = buf.getvalue()
        assert "--strategy" in help_text
        for choice in ("auto", "sequential", "multi", "batched"):
            assert choice in help_text


# ---------------------------------------------------------------------------
# Helpers for OME-Zarr-based tests (fit / tune)
# ---------------------------------------------------------------------------


def _write_synthetic_mosaic_zarr(zarr_path: Path, n_z: int = 3, n_rows: int = 3, n_cols: int = 3, tile: int = 16) -> None:
    """Write a synthetic 3-D mosaic volume as OME-Zarr.

    The volume is (n_z, n_rows*tile, n_cols*tile).  Tests are intentionally
    tiny so they run quickly even in CI.
    """
    from linum_basic.io.zarr import write_ome_zarr

    rng = np.random.default_rng(42)
    h, w = n_rows * tile, n_cols * tile
    vol = rng.random((n_z, h, w), dtype=np.float32) + 0.3
    write_ome_zarr(zarr_path, vol, axes=["z", "y", "x"], scale=[1.0, 0.01, 0.01], overwrite=True)


class TestFitSubcommand:
    """Tests for the ``basic fit`` sub-command (OME-Zarr → corrected OME-Zarr)."""

    @pytest.mark.parametrize("field_mode", ["per-z", "global"])
    def test_fit_creates_output(self, tmp_path: Path, field_mode: str) -> None:
        """``basic fit`` writes a corrected OME-Zarr without error."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=3, n_cols=3, tile=16)

        rc = main(
            [
                "fit",
                "--input",
                str(zarr_in),
                "--output",
                str(zarr_out),
                "--field-mode",
                field_mode,
                "--overlap",
                "0.2",
            ]
        )
        assert rc == 0
        assert zarr_out.exists()

    def test_fit_save_fields(self, tmp_path: Path) -> None:
        """``--save-fields`` writes flatfields.npy and darkfields.npy."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        fields_dir = tmp_path / "fields"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=3, n_cols=3, tile=16)

        rc = main(
            [
                "fit",
                "--input",
                str(zarr_in),
                "--output",
                str(zarr_out),
                "--save-fields",
                str(fields_dir),
                "--verbose",
            ]
        )
        assert rc == 0
        assert (fields_dir / "flatfields.npy").exists()
        assert (fields_dir / "darkfields.npy").exists()

    def test_fit_strategy_auto_smoke(self, tmp_path: Path) -> None:
        """``--strategy auto`` completes on numpy CI (no CUDA)."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=3, n_cols=3, tile=16)

        rc = main(
            [
                "fit",
                "--input",
                str(zarr_in),
                "--output",
                str(zarr_out),
                "--strategy",
                "auto",
            ]
        )
        assert rc == 0
        assert zarr_out.exists()

    def test_fit_strategy_pass_through(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--strategy`` is forwarded to ``fit_mosaic``."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main
        from linum_basic.fit import MosaicFit

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=2, n_cols=2, tile=8)

        captured: dict = {}

        def _stub_fit(mosaic, **kwargs):
            captured.update(kwargs)
            th, tw = mosaic.tile_shape
            return MosaicFit(
                flatfields=np.ones((2, th, tw), dtype=np.float32),
                darkfields=np.zeros((2, th, tw), dtype=np.float32),
                field_mode="per-z",
                z_indices=[0, 1],
                params={},
            )

        monkeypatch.setattr("linum_basic.fit.fit_mosaic", _stub_fit)
        monkeypatch.setattr("linum_basic.fit.save_corrected", lambda *args, **kwargs: None)

        rc = main(
            [
                "fit",
                "--input",
                str(zarr_in),
                "--output",
                str(zarr_out),
                "--strategy",
                "multi",
            ]
        )
        assert rc == 0
        assert captured.get("strategy") == "multi"

    def test_fit_unknown_strategy_rejected(self, tmp_path: Path) -> None:
        """Unknown ``--strategy`` values are rejected by argparse."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in)

        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "fit",
                    "--input",
                    str(zarr_in),
                    "--output",
                    str(zarr_out),
                    "--strategy",
                    "turbo",
                ]
            )
        assert excinfo.value.code == 2

    def test_fit_omits_backend_when_not_explicit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Omitting ``--backend`` leaves backend out of ``basic_kwargs`` for the resolver."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main
        from linum_basic.fit import MosaicFit

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=1, n_rows=2, n_cols=2, tile=8)

        captured: dict = {}

        def _stub_fit(mosaic, **kwargs):
            captured.update(kwargs)
            th, tw = mosaic.tile_shape
            return MosaicFit(
                flatfields=np.ones((1, th, tw), dtype=np.float32),
                darkfields=np.zeros((1, th, tw), dtype=np.float32),
                field_mode="per-z",
                z_indices=[0],
                params={},
            )

        monkeypatch.setattr("linum_basic.fit.fit_mosaic", _stub_fit)
        monkeypatch.setattr("linum_basic.fit.save_corrected", lambda *args, **kwargs: None)

        rc = main(["fit", "--input", str(zarr_in), "--output", str(zarr_out)])
        assert rc == 0
        basic_kwargs = captured.get("basic_kwargs") or {}
        assert "backend" not in basic_kwargs

    def test_fit_explicit_backend_passed_through(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit ``--backend numpy`` is forwarded in ``basic_kwargs``."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main
        from linum_basic.fit import MosaicFit

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=1, n_rows=2, n_cols=2, tile=8)

        captured: dict = {}

        def _stub_fit(mosaic, **kwargs):
            captured.update(kwargs)
            th, tw = mosaic.tile_shape
            return MosaicFit(
                flatfields=np.ones((1, th, tw), dtype=np.float32),
                darkfields=np.zeros((1, th, tw), dtype=np.float32),
                field_mode="per-z",
                z_indices=[0],
                params={},
            )

        monkeypatch.setattr("linum_basic.fit.fit_mosaic", _stub_fit)
        monkeypatch.setattr("linum_basic.fit.save_corrected", lambda *args, **kwargs: None)

        rc = main(
            [
                "fit",
                "--input",
                str(zarr_in),
                "--output",
                str(zarr_out),
                "--backend",
                "numpy",
            ]
        )
        assert rc == 0
        basic_kwargs = captured.get("basic_kwargs") or {}
        assert basic_kwargs.get("backend") == "numpy"

    def test_fit_verbose_prints_strategy_summary(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        """``--verbose`` prints resolved execution path and reason summary."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main
        from linum_basic.fit import MosaicFit

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=1, n_rows=2, n_cols=2, tile=8)

        def _stub_fit(mosaic, **kwargs):
            th, tw = mosaic.tile_shape
            return MosaicFit(
                flatfields=np.ones((1, th, tw), dtype=np.float32),
                darkfields=np.zeros((1, th, tw), dtype=np.float32),
                field_mode="per-z",
                z_indices=[0],
                params={
                    "_strategy": {
                        "execution_path": "sequential",
                        "reason_summary": "CPU fallback: no CUDA devices visible.",
                    }
                },
            )

        monkeypatch.setattr("linum_basic.fit.fit_mosaic", _stub_fit)
        monkeypatch.setattr("linum_basic.fit.save_corrected", lambda *args, **kwargs: None)

        rc = main(
            [
                "fit",
                "--input",
                str(zarr_in),
                "--output",
                str(zarr_out),
                "--verbose",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "sequential" in out
        assert "CPU fallback: no CUDA devices visible." in out


class TestFitStreamingLazyFlags:
    """Tests for ``--streaming`` and ``--lazy`` on ``basic fit`` (S02/T04)."""

    def test_fit_help_shows_streaming_and_lazy(self) -> None:
        """``basic fit --help`` documents ``--streaming`` and ``--lazy``."""
        from linum_basic.cli import main

        buf = io.StringIO()
        with pytest.raises(SystemExit) as excinfo, redirect_stdout(buf):
            main(["fit", "--help"])
        assert excinfo.value.code == 0
        help_text = buf.getvalue()
        assert "--streaming" in help_text
        assert "--lazy" in help_text

    def test_fit_streaming_flag_forwarded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--streaming`` is forwarded to ``fit_mosaic`` as ``streaming=True``."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main
        from linum_basic.fit import MosaicFit

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=1, n_rows=2, n_cols=2, tile=8)

        captured: dict = {}

        def _stub_fit(mosaic, **kwargs):
            captured.update(kwargs)
            th, tw = mosaic.tile_shape
            return MosaicFit(
                flatfields=np.ones((1, th, tw), dtype=np.float32),
                darkfields=np.zeros((1, th, tw), dtype=np.float32),
                field_mode="per-z",
                z_indices=[0],
                params={},
            )

        monkeypatch.setattr("linum_basic.fit.fit_mosaic", _stub_fit)
        monkeypatch.setattr("linum_basic.fit.save_corrected", lambda *args, **kwargs: None)

        rc = main(
            [
                "fit",
                "--input",
                str(zarr_in),
                "--output",
                str(zarr_out),
                "--streaming",
                "--strategy",
                "sequential",
            ]
        )
        assert rc == 0
        assert captured.get("streaming") is True

    def test_fit_streaming_defaults_false(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Omitting ``--streaming`` forwards ``streaming=False`` (backward compatible)."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main
        from linum_basic.fit import MosaicFit

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=1, n_rows=2, n_cols=2, tile=8)

        captured: dict = {}

        def _stub_fit(mosaic, **kwargs):
            captured.update(kwargs)
            th, tw = mosaic.tile_shape
            return MosaicFit(
                flatfields=np.ones((1, th, tw), dtype=np.float32),
                darkfields=np.zeros((1, th, tw), dtype=np.float32),
                field_mode="per-z",
                z_indices=[0],
                params={},
            )

        monkeypatch.setattr("linum_basic.fit.fit_mosaic", _stub_fit)
        monkeypatch.setattr("linum_basic.fit.save_corrected", lambda *args, **kwargs: None)

        rc = main(["fit", "--input", str(zarr_in), "--output", str(zarr_out)])
        assert rc == 0
        assert captured.get("streaming") is False

    def test_fit_lazy_flag_forwarded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--lazy`` is forwarded to ``MosaicGrid.from_ome_zarr`` as ``lazy=True``."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main
        from linum_basic.mosaic import MosaicGrid

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=1, n_rows=2, n_cols=2, tile=8)

        captured: dict = {}
        original_from = MosaicGrid.from_ome_zarr

        def _spy_from(path, **kwargs):
            captured.update(kwargs)
            return original_from(path, **kwargs)

        monkeypatch.setattr(MosaicGrid, "from_ome_zarr", _spy_from)
        monkeypatch.setattr("linum_basic.fit.save_corrected", lambda *args, **kwargs: None)

        rc = main(
            [
                "fit",
                "--input",
                str(zarr_in),
                "--output",
                str(zarr_out),
                "--lazy",
                "--streaming",
                "--strategy",
                "sequential",
                "--backend",
                "numpy",
            ]
        )
        assert rc == 0
        assert captured.get("lazy") is True

    def test_fit_lazy_defaults_false(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Omitting ``--lazy`` forwards ``lazy=False`` (eager load, backward compatible)."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main
        from linum_basic.mosaic import MosaicGrid

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=1, n_rows=2, n_cols=2, tile=8)

        captured: dict = {}
        original_from = MosaicGrid.from_ome_zarr

        def _spy_from(path, **kwargs):
            captured.update(kwargs)
            return original_from(path, **kwargs)

        monkeypatch.setattr(MosaicGrid, "from_ome_zarr", _spy_from)
        monkeypatch.setattr("linum_basic.fit.save_corrected", lambda *args, **kwargs: None)

        rc = main(["fit", "--input", str(zarr_in), "--output", str(zarr_out)])
        assert rc == 0
        assert captured.get("lazy") is False

    def test_fit_streaming_incompatible_with_multi_raises(self, tmp_path: Path) -> None:
        """``--streaming --strategy multi`` bubbles up ``ValueError`` from ``fit_mosaic``."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=1, n_rows=2, n_cols=2, tile=8)

        with pytest.raises(ValueError, match="streaming"):
            main(
                [
                    "fit",
                    "--input",
                    str(zarr_in),
                    "--output",
                    str(zarr_out),
                    "--strategy",
                    "multi",
                    "--streaming",
                ]
            )

    def test_fit_streaming_creates_output(self, tmp_path: Path) -> None:
        """``--streaming`` end-to-end writes a corrected OME-Zarr (numerics unchanged)."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=3, n_cols=3, tile=16)

        rc = main(
            [
                "fit",
                "--input",
                str(zarr_in),
                "--output",
                str(zarr_out),
                "--streaming",
                "--strategy",
                "sequential",
                "--backend",
                "numpy",
            ]
        )
        assert rc == 0
        assert zarr_out.exists()

    def test_fit_lazy_streaming_creates_output(self, tmp_path: Path) -> None:
        """``--lazy --streaming`` end-to-end writes a corrected OME-Zarr (lowest peak memory)."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=3, n_cols=3, tile=16)

        rc = main(
            [
                "fit",
                "--input",
                str(zarr_in),
                "--output",
                str(zarr_out),
                "--lazy",
                "--streaming",
                "--strategy",
                "sequential",
                "--backend",
                "numpy",
            ]
        )
        assert rc == 0
        assert zarr_out.exists()


class TestTuneSubcommand:
    """Tests for the ``basic tune`` sub-command."""

    def test_tune_returns_zero(self, tmp_path: Path) -> None:
        """``basic tune`` exits 0 and optionally writes best-params JSON."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")

        from linum_basic.cli import main

        zarr_in = tmp_path / "in.ome.zarr"
        json_out = tmp_path / "best.json"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=3, n_cols=3, tile=16)

        rc = main(
            [
                "tune",
                "--input",
                str(zarr_in),
                "--n-trials",
                "2",
                "--z-subsample",
                "1",
                "--out-json",
                str(json_out),
                "--seed",
                "0",
            ]
        )
        assert rc == 0
        assert json_out.exists()
        import json

        data = json.loads(json_out.read_text())
        assert "l_s" in data

    def test_tune_with_apply(self, tmp_path: Path) -> None:
        """``--apply`` runs a full fit and writes the corrected mosaic."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")

        from linum_basic.cli import main

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_apply = tmp_path / "applied.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=3, n_cols=3, tile=16)

        rc = main(
            [
                "tune",
                "--input",
                str(zarr_in),
                "--n-trials",
                "2",
                "--z-subsample",
                "1",
                "--apply",
                str(zarr_apply),
                "--seed",
                "1",
                "--verbose",
            ]
        )
        assert rc == 0
        assert zarr_apply.exists()

    def test_tune_bounds_json_writes_payload(self, tmp_path: Path) -> None:
        """``--bounds-json`` writes a search_space-shaped JSON payload."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")
        pytest.importorskip("pandas")

        import json

        from linum_basic.cli import main

        zarr_in = tmp_path / "in.ome.zarr"
        bounds_out = tmp_path / "bounds.json"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=3, n_cols=3, tile=16)

        rc = main(
            [
                "tune",
                "--input",
                str(zarr_in),
                "--n-trials",
                "2",
                "--z-subsample",
                "1",
                "--bounds-json",
                str(bounds_out),
                "--seed",
                "0",
            ]
        )
        assert rc == 0
        assert bounds_out.exists()

        payload = json.loads(bounds_out.read_text())
        assert "search_space" in payload
        search_space = payload["search_space"]
        # The scale-invariant divisor parametrisation keys must be present.
        for key in ("working_size", "l_s_divisor", "l_d_divisor", "epsilon", "estimate_darkfield"):
            assert key in search_space, f"missing key '{key}' in search_space"
        # working_size is a list of observed integer choices.
        assert isinstance(search_space["working_size"], list)
        assert len(search_space["working_size"]) >= 1
        assert all(isinstance(ws, int) for ws in search_space["working_size"])
        # Divisor / epsilon ranges are [low, high] lists.
        for key in ("l_s_divisor", "l_d_divisor", "epsilon"):
            rng = search_space[key]
            assert len(rng) == 2
            assert rng[0] <= rng[1]
        # estimate_darkfield is a single-element boolean list (majority vote
        # over Optuna-sampled flags — the specific value is seed-dependent and
        # not asserted here).
        edf = search_space["estimate_darkfield"]
        assert len(edf) == 1
        assert isinstance(edf[0], bool)
        # Metadata fields.
        assert payload["n_near_optimal"] >= 1
        assert payload["margin"] == 0.10
        assert "best_value" in payload

    def test_tune_bounds_json_verbose_prints_search_space(self, tmp_path: Path, capsys) -> None:
        """``--bounds-json --verbose`` prints the recommended search space to stdout."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")
        pytest.importorskip("pandas")

        from linum_basic.cli import main

        zarr_in = tmp_path / "in.ome.zarr"
        bounds_out = tmp_path / "bounds.json"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=3, n_cols=3, tile=16)

        rc = main(
            [
                "tune",
                "--input",
                str(zarr_in),
                "--n-trials",
                "2",
                "--z-subsample",
                "1",
                "--bounds-json",
                str(bounds_out),
                "--bounds-margin",
                "0.20",
                "--verbose",
                "--seed",
                "0",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Recommended bounds" in out
        assert "l_s_divisor" in out
        assert "bounds.json" in out

        # The --bounds-margin value is forwarded into the written payload.
        import json

        payload = json.loads(bounds_out.read_text())
        assert payload["margin"] == pytest.approx(0.20)

    def test_tune_bounds_margin_forwarded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--bounds-margin`` is forwarded to ``recommend_bounds``."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")
        pytest.importorskip("pandas")

        from linum_basic import cli
        from linum_basic.tuning import BoundsRecommendation

        captured: dict = {}

        # The CLI imports recommend_bounds lazily inside _run_tune; patch the
        # source module so the lazy import resolves to our spy.
        import linum_basic.tuning as tuning_mod

        original = tuning_mod.recommend_bounds

        def _spy_recommend(result, *, margin=0.10):
            captured["margin"] = margin
            return BoundsRecommendation(
                search_space={
                    "working_size": [128],
                    "l_s_divisor": (800.0, 1200.0),
                    "l_d_divisor": (2000.0, 4000.0),
                    "epsilon": (0.1, 0.3),
                    "estimate_darkfield": [False],
                },
                best_value=0.01,
                n_near_optimal=1,
                margin=margin,
            )

        monkeypatch.setattr(tuning_mod, "recommend_bounds", _spy_recommend)
        try:
            zarr_in = tmp_path / "in.ome.zarr"
            bounds_out = tmp_path / "bounds.json"
            _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=3, n_cols=3, tile=16)

            rc = cli.main(
                [
                    "tune",
                    "--input",
                    str(zarr_in),
                    "--n-trials",
                    "2",
                    "--z-subsample",
                    "1",
                    "--bounds-json",
                    str(bounds_out),
                    "--bounds-margin",
                    "0.05",
                    "--seed",
                    "0",
                ]
            )
            assert rc == 0
            assert bounds_out.exists()
            assert captured["margin"] == pytest.approx(0.05)
        finally:
            tuning_mod.recommend_bounds = original

    def test_tune_bounds_json_and_out_json_both_written(self, tmp_path: Path) -> None:
        """``--bounds-json`` and ``--out-json`` can be combined (two distinct artifacts)."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")
        pytest.importorskip("pandas")

        import json

        from linum_basic.cli import main

        zarr_in = tmp_path / "in.ome.zarr"
        best_out = tmp_path / "best.json"
        bounds_out = tmp_path / "bounds.json"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=3, n_cols=3, tile=16)

        rc = main(
            [
                "tune",
                "--input",
                str(zarr_in),
                "--n-trials",
                "2",
                "--z-subsample",
                "1",
                "--out-json",
                str(best_out),
                "--bounds-json",
                str(bounds_out),
                "--seed",
                "0",
            ]
        )
        assert rc == 0
        assert best_out.exists()
        assert bounds_out.exists()
        # best.json carries best_params (l_s key); bounds.json carries search_space.
        best_data = json.loads(best_out.read_text())
        assert "l_s" in best_data
        bounds_data = json.loads(bounds_out.read_text())
        assert "search_space" in bounds_data

    def test_tune_bounds_failure_exits_nonzero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ``ValueError`` from ``recommend_bounds`` is reported and exits 1."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")
        pytest.importorskip("pandas")

        import linum_basic.tuning as tuning_mod
        from linum_basic import cli

        original = tuning_mod.recommend_bounds

        def _failing_recommend(result, *, margin=0.10):
            msg = "recommend_bounds requires at least one COMPLETE trial"
            raise ValueError(msg)

        monkeypatch.setattr(tuning_mod, "recommend_bounds", _failing_recommend)
        try:
            zarr_in = tmp_path / "in.ome.zarr"
            bounds_out = tmp_path / "bounds.json"
            _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=3, n_cols=3, tile=16)

            rc = cli.main(
                [
                    "tune",
                    "--input",
                    str(zarr_in),
                    "--n-trials",
                    "2",
                    "--z-subsample",
                    "1",
                    "--bounds-json",
                    str(bounds_out),
                    "--seed",
                    "0",
                ]
            )
            assert rc == 1
            assert not bounds_out.exists()
        finally:
            tuning_mod.recommend_bounds = original

    def test_tune_help_documents_bounds_flags(self) -> None:
        """``basic tune --help`` lists ``--bounds-json`` and ``--bounds-margin``."""
        pytest.importorskip("optuna")

        from linum_basic.cli import main

        buf = io.StringIO()
        with pytest.raises(SystemExit) as excinfo, redirect_stdout(buf):
            main(["tune", "--help"])
        assert excinfo.value.code == 0
        help_text = buf.getvalue()
        assert "--bounds-json" in help_text
        assert "--bounds-margin" in help_text


class TestTuneAutoApply:
    """Tests for ``basic tune --auto-apply`` (M008/S02/T03, D023 auto-apply gate).

    The ``--auto-apply`` flag runs the full D023 pipeline via
    :func:`linum_basic.tuning.auto_tune` and composes with
    ``--out-json`` / ``--bounds-json`` / ``--apply`` / ``--verbose``. These
    tests patch ``auto_tune`` (the lazy-imported entry point) to assert the
    flag is forwarded, the winning fit is written, output flags compose, the
    fallback path writes the baseline, and an ``AutoApplyError`` exits 1.
    """

    @staticmethod
    def _make_fit(mosaic) -> object:
        """Build a minimal :class:`MosaicFit` matching *mosaic* tile shape."""
        from linum_basic.fit import MosaicFit

        th, tw = mosaic.tile_shape
        nz = getattr(mosaic, "n_z", 1)
        return MosaicFit(
            flatfields=np.ones((nz, th, tw), dtype=np.float32),
            darkfields=np.zeros((nz, th, tw), dtype=np.float32),
            field_mode="per-z",
            z_indices=list(range(nz)),
            params={},
        )

    @staticmethod
    def _make_auto_result(
        fit,
        *,
        gate_verdict: str = "pass",
        applied: str = "candidate",
        failing_metrics: list[str] | None = None,
        fallback_reason: str | None = None,
    ) -> object:
        """Build a synthetic :class:`AutoTuneResult` with a full gate sub-dict."""
        from linum_basic.tuning import AutoTuneResult, BoundsRecommendation

        best_params = {
            "working_size": 128,
            "l_s": 1000.0,
            "l_d": 3000.0,
            "epsilon": 0.1,
            "estimate_darkfield": True,
        }
        rec = BoundsRecommendation(
            search_space={
                "working_size": [128],
                "l_s_divisor": (800.0, 1200.0),
                "l_d_divisor": (2000.0, 4000.0),
                "epsilon": (0.1, 0.3),
                "estimate_darkfield": [True],
            },
            best_value=0.49,
            n_near_optimal=2,
            margin=0.10,
        )
        gate = {
            "gate_verdict": gate_verdict,
            "applied": applied,
            "no_regression_margin": 0.0,
            "failing_metrics": failing_metrics or [],
            "fallback_reason": fallback_reason,
            "deltas": {
                "seam_l1": {"abs_delta": -0.012, "rel_delta": -0.021},
                "seam_curvature": {"abs_delta": -0.0004, "rel_delta": -0.012},
            },
            "baseline": {
                "bounds": "default",
                "aggregates": {"seam_l1": 0.5717, "seam_curvature": 0.0344},
            },
            "candidate": {
                "bounds": "tune-best-params",
                "best_params": best_params,
                "aggregates": {"seam_l1": 0.5597, "seam_curvature": 0.0340},
            },
            "recommendation": {
                "search_space": rec.search_space,
                "n_near_optimal": rec.n_near_optimal,
                "margin": rec.margin,
                "best_value": rec.best_value,
            },
        }
        return AutoTuneResult(
            fit,
            applied=applied,
            recommendation=rec,
            gate=gate,
        )

    def test_tune_auto_apply_help_documented(self) -> None:
        """``basic tune --help`` lists ``--auto-apply``."""
        pytest.importorskip("optuna")

        from linum_basic.cli import main

        buf = io.StringIO()
        with pytest.raises(SystemExit) as excinfo, redirect_stdout(buf):
            main(["tune", "--help"])
        assert excinfo.value.code == 0
        help_text = buf.getvalue()
        assert "--auto-apply" in help_text
        # The help should explain the safety-gate semantics.
        assert "non-regression" in help_text.lower()

    def test_tune_auto_apply_flag_forwarded_to_auto_tune(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--auto-apply`` delegates to :func:`auto_tune` with forwarded kwargs."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")

        import linum_basic.tuning as tuning_mod
        from linum_basic import cli

        captured: dict = {}
        zarr_in = tmp_path / "in.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=2, n_cols=2, tile=8)

        def _spy_auto_tune(mosaic, **kwargs):
            captured.update(kwargs)
            captured["mosaic"] = mosaic
            return self._make_auto_result(self._make_fit(mosaic))

        monkeypatch.setattr(tuning_mod, "auto_tune", _spy_auto_tune)
        monkeypatch.setattr("linum_basic.fit.save_corrected", lambda *a, **k: None)

        rc = cli.main(
            [
                "tune",
                "--input",
                str(zarr_in),
                "--auto-apply",
                "--n-trials",
                "3",
                "--z-subsample",
                "1",
                "--bounds-margin",
                "0.15",
                "--seed",
                "7",
            ]
        )
        assert rc == 0
        # auto_tune was called (not the manual tune path).
        assert "mosaic" in captured
        assert captured["n_trials"] == 3
        assert captured["z_subsample"] == 1
        # --bounds-margin maps to auto_tune's margin (feeds recommend_bounds).
        assert captured["margin"] == pytest.approx(0.15)
        assert captured["seed"] == 7

    def test_tune_auto_apply_writes_winning_fit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--auto-apply --apply`` writes the winning fit via ``save_corrected``."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")

        import linum_basic.tuning as tuning_mod
        from linum_basic import cli

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "applied.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=2, n_cols=2, tile=8)

        saved: dict = {}

        def _spy_auto_tune(mosaic, **kwargs):
            fit = self._make_fit(mosaic)
            return self._make_auto_result(fit, gate_verdict="pass", applied="candidate")

        def _spy_save(mosaic, fit, out_path, **kwargs):
            saved["fit"] = fit
            saved["out_path"] = out_path

        monkeypatch.setattr(tuning_mod, "auto_tune", _spy_auto_tune)
        monkeypatch.setattr("linum_basic.fit.save_corrected", _spy_save)

        rc = cli.main(
            [
                "tune",
                "--input",
                str(zarr_in),
                "--auto-apply",
                "--apply",
                str(zarr_out),
                "--n-trials",
                "2",
                "--z-subsample",
                "1",
            ]
        )
        assert rc == 0
        # The winning candidate fit was passed to save_corrected.
        assert saved.get("out_path") == zarr_out
        assert saved.get("fit") is not None

    def test_tune_auto_apply_compose_out_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--auto-apply --out-json`` writes the candidate best params."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")

        import linum_basic.tuning as tuning_mod
        from linum_basic import cli

        zarr_in = tmp_path / "in.ome.zarr"
        best_out = tmp_path / "best.json"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=2, n_cols=2, tile=8)

        def _spy_auto_tune(mosaic, **kwargs):
            return self._make_auto_result(self._make_fit(mosaic))

        monkeypatch.setattr(tuning_mod, "auto_tune", _spy_auto_tune)
        monkeypatch.setattr("linum_basic.fit.save_corrected", lambda *a, **k: None)

        rc = cli.main(
            [
                "tune",
                "--input",
                str(zarr_in),
                "--auto-apply",
                "--out-json",
                str(best_out),
                "--n-trials",
                "2",
                "--z-subsample",
                "1",
            ]
        )
        assert rc == 0
        assert best_out.exists()
        import json

        data = json.loads(best_out.read_text())
        # The candidate best_params (from the gate sub-dict) are written.
        assert "l_s" in data
        assert data["working_size"] == 128

    def test_tune_auto_apply_compose_bounds_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--auto-apply --bounds-json`` writes the narrowed recommendation."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")

        import linum_basic.tuning as tuning_mod
        from linum_basic import cli

        zarr_in = tmp_path / "in.ome.zarr"
        bounds_out = tmp_path / "bounds.json"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=2, n_cols=2, tile=8)

        def _spy_auto_tune(mosaic, **kwargs):
            return self._make_auto_result(self._make_fit(mosaic))

        monkeypatch.setattr(tuning_mod, "auto_tune", _spy_auto_tune)
        monkeypatch.setattr("linum_basic.fit.save_corrected", lambda *a, **k: None)

        rc = cli.main(
            [
                "tune",
                "--input",
                str(zarr_in),
                "--auto-apply",
                "--bounds-json",
                str(bounds_out),
                "--n-trials",
                "2",
                "--z-subsample",
                "1",
            ]
        )
        assert rc == 0
        assert bounds_out.exists()
        import json

        payload = json.loads(bounds_out.read_text())
        assert "search_space" in payload
        assert "l_s_divisor" in payload["search_space"]
        assert payload["n_near_optimal"] == 2

    def test_tune_auto_apply_fallback_writes_baseline(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """On a regression-detected fallback, the baseline fit is written and rc is 0."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")

        import linum_basic.tuning as tuning_mod
        from linum_basic import cli

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "applied.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=2, n_cols=2, tile=8)

        saved: dict = {}

        def _spy_auto_tune(mosaic, **kwargs):
            # Regression on seam_l1 -> gate fails, baseline returned.
            return self._make_auto_result(
                self._make_fit(mosaic),
                gate_verdict="fail",
                applied="baseline-default",
                failing_metrics=["seam_l1"],
                fallback_reason="regression-detected",
            )

        def _spy_save(mosaic, fit, out_path, **kwargs):
            saved["fit"] = fit

        monkeypatch.setattr(tuning_mod, "auto_tune", _spy_auto_tune)
        monkeypatch.setattr("linum_basic.fit.save_corrected", _spy_save)

        rc = cli.main(
            [
                "tune",
                "--input",
                str(zarr_in),
                "--auto-apply",
                "--apply",
                str(zarr_out),
                "--n-trials",
                "2",
                "--z-subsample",
                "1",
            ]
        )
        # A fallback is not a CLI error: the safe baseline fit is written, rc 0.
        assert rc == 0
        assert saved.get("fit") is not None

    def test_tune_auto_apply_verbose_prints_verdict(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        """``--auto-apply --verbose`` prints the gate verdict, applied, and deltas."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")

        import linum_basic.tuning as tuning_mod
        from linum_basic import cli

        zarr_in = tmp_path / "in.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=2, n_cols=2, tile=8)

        def _spy_auto_tune(mosaic, **kwargs):
            return self._make_auto_result(
                self._make_fit(mosaic),
                gate_verdict="fail",
                applied="baseline-default",
                failing_metrics=["seam_curvature"],
                fallback_reason="regression-detected",
            )

        monkeypatch.setattr(tuning_mod, "auto_tune", _spy_auto_tune)
        monkeypatch.setattr("linum_basic.fit.save_corrected", lambda *a, **k: None)

        rc = cli.main(
            [
                "tune",
                "--input",
                str(zarr_in),
                "--auto-apply",
                "--verbose",
                "--n-trials",
                "2",
                "--z-subsample",
                "1",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Auto-apply gate verdict: fail" in out
        assert "baseline-default" in out
        assert "regression-detected" in out
        assert "seam_curvature" in out
        # Per-metric deltas are printed for both first-class metrics.
        assert "abs_delta=" in out
        assert "rel_delta=" in out

    def test_tune_auto_apply_baseline_failure_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """An ``AutoApplyError`` (baseline fit failed) exits 1 with a stderr message."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")

        import linum_basic.tuning as tuning_mod
        from linum_basic import cli
        from linum_basic.tuning import AutoApplyError

        zarr_in = tmp_path / "in.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=2, n_cols=2, tile=8)

        def _raising_auto_tune(mosaic, **kwargs):
            raise AutoApplyError("baseline (default-bounds) fit failed: OOM")

        monkeypatch.setattr(tuning_mod, "auto_tune", _raising_auto_tune)
        monkeypatch.setattr("linum_basic.fit.save_corrected", lambda *a, **k: None)

        rc = cli.main(
            [
                "tune",
                "--input",
                str(zarr_in),
                "--auto-apply",
                "--n-trials",
                "2",
                "--z-subsample",
                "1",
            ]
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "auto-apply failed" in err.lower()

    def test_tune_auto_apply_without_apply_exits_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--auto-apply`` without ``--apply`` runs the gate and exits 0 (no fit write)."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")

        import linum_basic.tuning as tuning_mod
        from linum_basic import cli

        zarr_in = tmp_path / "in.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=2, n_cols=2, tile=8)

        save_called = {"n": 0}

        def _spy_auto_tune(mosaic, **kwargs):
            return self._make_auto_result(self._make_fit(mosaic))

        def _fail_save(*a, **k):
            save_called["n"] += 1

        monkeypatch.setattr(tuning_mod, "auto_tune", _spy_auto_tune)
        monkeypatch.setattr("linum_basic.fit.save_corrected", _fail_save)

        rc = cli.main(
            [
                "tune",
                "--input",
                str(zarr_in),
                "--auto-apply",
                "--n-trials",
                "2",
                "--z-subsample",
                "1",
            ]
        )
        assert rc == 0
        # No --apply -> save_corrected must not be called.
        assert save_called["n"] == 0


class TestWorkingSizeFlag:
    """Tests for ``--working-size`` on ``basic fit`` and ``basic tune`` (M006/S02/T03)."""

    def test_fit_help_shows_working_size(self) -> None:
        """``basic fit --help`` documents ``--working-size``."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main

        buf = io.StringIO()
        with pytest.raises(SystemExit) as excinfo, redirect_stdout(buf):
            main(["fit", "--help"])
        assert excinfo.value.code == 0
        assert "--working-size" in buf.getvalue()

    def test_fit_working_size_auto_forwarded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--working-size auto`` is forwarded into ``basic_kwargs["working_size"]``."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main
        from linum_basic.fit import MosaicFit

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=1, n_rows=2, n_cols=2, tile=8)

        captured: dict = {}

        def _stub_fit(mosaic, **kwargs):
            captured.update(kwargs)
            th, tw = mosaic.tile_shape
            return MosaicFit(
                flatfields=np.ones((1, th, tw), dtype=np.float32),
                darkfields=np.zeros((1, th, tw), dtype=np.float32),
                field_mode="per-z",
                z_indices=[0],
                params={},
            )

        monkeypatch.setattr("linum_basic.fit.fit_mosaic", _stub_fit)
        monkeypatch.setattr("linum_basic.fit.save_corrected", lambda *args, **kwargs: None)

        rc = main(
            [
                "fit",
                "--input",
                str(zarr_in),
                "--output",
                str(zarr_out),
                "--working-size",
                "auto",
            ]
        )
        assert rc == 0
        basic_kwargs = captured.get("basic_kwargs") or {}
        assert basic_kwargs.get("working_size") == "auto"

    def test_fit_working_size_int_forwarded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit ``--working-size 64`` is forwarded as an integer."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main
        from linum_basic.fit import MosaicFit

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=1, n_rows=2, n_cols=2, tile=8)

        captured: dict = {}

        def _stub_fit(mosaic, **kwargs):
            captured.update(kwargs)
            th, tw = mosaic.tile_shape
            return MosaicFit(
                flatfields=np.ones((1, th, tw), dtype=np.float32),
                darkfields=np.zeros((1, th, tw), dtype=np.float32),
                field_mode="per-z",
                z_indices=[0],
                params={},
            )

        monkeypatch.setattr("linum_basic.fit.fit_mosaic", _stub_fit)
        monkeypatch.setattr("linum_basic.fit.save_corrected", lambda *args, **kwargs: None)

        rc = main(
            [
                "fit",
                "--input",
                str(zarr_in),
                "--output",
                str(zarr_out),
                "--working-size",
                "64",
            ]
        )
        assert rc == 0
        basic_kwargs = captured.get("basic_kwargs") or {}
        assert basic_kwargs.get("working_size") == 64

    def test_fit_working_size_default_omitted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Omitting ``--working-size`` leaves it out of ``basic_kwargs`` (backward compatible)."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main
        from linum_basic.fit import MosaicFit

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=1, n_rows=2, n_cols=2, tile=8)

        captured: dict = {}

        def _stub_fit(mosaic, **kwargs):
            captured.update(kwargs)
            th, tw = mosaic.tile_shape
            return MosaicFit(
                flatfields=np.ones((1, th, tw), dtype=np.float32),
                darkfields=np.zeros((1, th, tw), dtype=np.float32),
                field_mode="per-z",
                z_indices=[0],
                params={},
            )

        monkeypatch.setattr("linum_basic.fit.fit_mosaic", _stub_fit)
        monkeypatch.setattr("linum_basic.fit.save_corrected", lambda *args, **kwargs: None)

        rc = main(["fit", "--input", str(zarr_in), "--output", str(zarr_out)])
        assert rc == 0
        basic_kwargs = captured.get("basic_kwargs") or {}
        assert "working_size" not in basic_kwargs

    def test_fit_working_size_invalid_rejected(self, tmp_path: Path) -> None:
        """Non-numeric, non-auto ``--working-size`` values are rejected by argparse."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=1, n_rows=2, n_cols=2, tile=8)

        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "fit",
                    "--input",
                    str(zarr_in),
                    "--output",
                    str(zarr_out),
                    "--working-size",
                    "turbo",
                ]
            )
        assert excinfo.value.code == 2

    def test_fit_working_size_negative_rejected(self, tmp_path: Path) -> None:
        """Negative ``--working-size`` values are rejected by the type parser."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")

        from linum_basic.cli import main

        zarr_in = tmp_path / "in.ome.zarr"
        zarr_out = tmp_path / "out.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_in, n_z=1, n_rows=2, n_cols=2, tile=8)

        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "fit",
                    "--input",
                    str(zarr_in),
                    "--output",
                    str(zarr_out),
                    "--working-size",
                    "-1",
                ]
            )
        assert excinfo.value.code == 2

    def test_tune_help_shows_working_size(self) -> None:
        """``basic tune --help`` documents ``--working-size``."""
        pytest.importorskip("optuna")

        from linum_basic.cli import main

        buf = io.StringIO()
        with pytest.raises(SystemExit) as excinfo, redirect_stdout(buf):
            main(["tune", "--help"])
        assert excinfo.value.code == 0
        assert "--working-size" in buf.getvalue()

    def test_tune_working_size_auto_forwarded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``tune --working-size auto`` forwards ``working_size='auto'`` to :func:`tune`."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")

        import linum_basic.tuning as tuning_mod
        from linum_basic import cli

        captured: dict = {}
        original = tuning_mod.tune

        def _spy_tune(mosaic, **kwargs):
            captured.update(kwargs)
            return original(mosaic, **kwargs)

        monkeypatch.setattr(tuning_mod, "tune", _spy_tune)
        try:
            zarr_in = tmp_path / "in.ome.zarr"
            _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=3, n_cols=3, tile=16)

            rc = cli.main(
                [
                    "tune",
                    "--input",
                    str(zarr_in),
                    "--n-trials",
                    "2",
                    "--z-subsample",
                    "1",
                    "--working-size",
                    "auto",
                    "--seed",
                    "0",
                ]
            )
            assert rc == 0
            assert captured.get("working_size") == "auto"
        finally:
            tuning_mod.tune = original

    def test_tune_working_size_int_forwarded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``tune --working-size 64`` forwards an integer to :func:`tune`."""
        pytest.importorskip("zarr")
        pytest.importorskip("ome_zarr")
        pytest.importorskip("optuna")

        import linum_basic.tuning as tuning_mod
        from linum_basic import cli

        captured: dict = {}
        original = tuning_mod.tune

        def _spy_tune(mosaic, **kwargs):
            captured.update(kwargs)
            return original(mosaic, **kwargs)

        monkeypatch.setattr(tuning_mod, "tune", _spy_tune)
        try:
            zarr_in = tmp_path / "in.ome.zarr"
            _write_synthetic_mosaic_zarr(zarr_in, n_z=2, n_rows=3, n_cols=3, tile=16)

            rc = cli.main(
                [
                    "tune",
                    "--input",
                    str(zarr_in),
                    "--n-trials",
                    "2",
                    "--z-subsample",
                    "1",
                    "--working-size",
                    "64",
                    "--seed",
                    "0",
                ]
            )
            assert rc == 0
            assert captured.get("working_size") == 64
        finally:
            tuning_mod.tune = original
