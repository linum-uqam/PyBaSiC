"""CLI integration tests for :mod:`linum_basic.cli`.

Runs the ``basic`` entry-point as a subprocess against
a temporary directory of synthetic TIFF images.  Tests are isolated and
produce no side effects on the permanent filesystem.
"""

from __future__ import annotations

import subprocess
import sys
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
