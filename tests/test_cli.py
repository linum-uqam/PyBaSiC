"""CLI integration tests for :mod:`linum_basic.cli`.

Runs the ``basic_shading_correction`` entry-point as a subprocess against
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
        assert "BaSiC" in result.stdout or "shading" in result.stdout.lower()

    def test_missing_input_exits_nonzero(self, tmp_path: Path) -> None:
        """Missing ``--input`` should cause a non-zero exit code."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "linum_basic.cli",
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
    """Tests for the ``basic_preview`` AIP preview entry-point."""

    def test_preview_creates_png(self, tmp_path: Path) -> None:
        """``preview_main`` writes a non-empty PNG from a small OME-Zarr volume."""
        from linum_basic.cli import preview_main
        from linum_basic.io.zarr import write_ome_zarr

        rng = np.random.default_rng(3)
        volume = rng.random((5, 32, 48), dtype=np.float32)
        zarr_path = tmp_path / "vol.ome.zarr"
        write_ome_zarr(zarr_path, volume, axes=["z", "y", "x"], scale=[2.0, 0.01, 0.01], overwrite=True)

        out_png = tmp_path / "preview.png"
        rc = preview_main(["--input", str(zarr_path), "--output", str(out_png), "--dpi", "60"])

        assert rc == 0
        assert out_png.is_file()
        assert out_png.stat().st_size > 0

    def test_preview_help_exits_zero(self) -> None:
        """``basic_preview --help`` exits with code 0."""
        from linum_basic.cli import _build_preview_parser

        with pytest.raises(SystemExit) as excinfo:
            _build_preview_parser().parse_args(["--help"])
        assert excinfo.value.code == 0
