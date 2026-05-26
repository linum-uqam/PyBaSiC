# SPDX-License-Identifier: MIT
"""Integration test: sbh-simulator vignette -> PyBaSiC flatfield recovery.

Pipeline (single source of truth for both the test suite and the
visualisation script):

1. Generate a 128x128 vignette via ``sbh-vignette`` (Gaussian / Zernike).
2. Tile the bundled source image into 128x128 non-overlapping patches.
3. Multiply each patch by the vignette to build a corrupted stack.
4. Run BaSiC and compare the recovered flat-field to ground truth.

When the environment variable ``PYBASIC_VIGNETTE_ARTIFACT_DIR`` is set, each
test also renders a 6-panel PNG into that directory.  That is how the CI
visualisation step (see ``scripts/visualize_vignette_correction.py``) produces
its figures -- there is no separate pipeline.

Skipped when the ``sbh-vignette`` CLI cannot be located.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pybasic.core import BaSiC

_TILE = 128
_CORRELATION_THRESHOLD = 0.85
_SOURCE_IMAGE = Path(__file__).parent / "data" / "source_image.jpg"
_ARTIFACT_DIR_ENV = "PYBASIC_VIGNETTE_ARTIFACT_DIR"


def _find_sbh_vignette() -> str | None:
    found = shutil.which("sbh-vignette")
    if found:
        return found
    home = os.environ.get("SBH_SIMULATOR_HOME")
    if home:
        candidate = Path(home) / ".venv" / "bin" / "sbh-vignette"
        if candidate.is_file():
            return str(candidate)
    return None


_SBH_VIGNETTE = _find_sbh_vignette()

pytestmark = pytest.mark.skipif(
    _SBH_VIGNETTE is None,
    reason="sbh-vignette not found. Install linum-uqam/sbh_simulator or set SBH_SIMULATOR_HOME.",
)


def _generate_vignette(kind: str, tmp_path: Path, *, order: int = 4) -> np.ndarray:
    """Run ``sbh-vignette`` once and return a float32 (_TILE, _TILE) array."""
    assert _SBH_VIGNETTE is not None
    cmd = [
        _SBH_VIGNETTE,
        kind,
        "--n",
        "1",
        "--size",
        str(_TILE),
        str(_TILE),
        "--seed",
        "42",
        "--output",
        str(tmp_path),
        "--format",
        "npy",
    ]
    if kind == "zernike":
        cmd += ["--order", str(order)]
    subprocess.run(cmd, check=True)
    return np.load(tmp_path / f"{kind}_0.npy")


def _tile_source_image() -> np.ndarray:
    """Tile the bundled image into a stack of (_TILE, _TILE) patches."""
    import cv2

    img = cv2.imread(str(_SOURCE_IMAGE), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Failed to read {_SOURCE_IMAGE}"
    h, w = img.shape
    nh, nw = h // _TILE, w // _TILE
    cropped = img[: nh * _TILE, : nw * _TILE].astype(np.float32) / 255.0
    return cropped.reshape(nh, _TILE, nw, _TILE).transpose(0, 2, 1, 3).reshape(nh * nw, _TILE, _TILE)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def _save_figure(kind: str, vignette, stack, model, corr: float, out_dir: Path) -> None:
    """Write a 6-panel diagnostic figure for *kind* into *out_dir*."""
    try:
        import matplotlib
    except ImportError:
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flatfield = model.flatfield_fullsize
    # Normalize GT to mean=1 so it is on the same scale as flatfield_fullsize.
    vignette_norm = vignette / (vignette.mean() + 1e-9)
    corrected = np.stack([model.normalize(stack[i]) for i in range(len(stack))])
    err = np.abs(flatfield - vignette_norm)
    # Pick the most content-rich tile so the sample panels actually show
    # recognisable image structure (tile 0 of the Landsat source is dark water).
    best = int(np.argmax(stack.std(axis=(1, 2))))
    # Shared color range for GT and estimated flat-field so they are directly comparable.
    ff_vmin = min(float(vignette_norm.min()), float(flatfield.min()))
    ff_vmax = max(float(vignette_norm.max()), float(flatfield.max()))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle(
        f"{kind.capitalize()} vignette  -  Pearson r = {corr:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    panels = [
        (axes[0, 0], vignette_norm, "viridis", True, "Ground-truth vignette (mean=1)", ff_vmin, ff_vmax),
        (axes[0, 1], stack[best], "gray", False, "Sample tile: corrupted", None, None),
        (axes[0, 2], corrected[best], "gray", False, "Sample tile: BaSiC-corrected", None, None),
        (axes[1, 0], stack.mean(axis=0), "viridis", True, "Mean of corrupted stack", None, None),
        (axes[1, 1], flatfield, "viridis", True, "Estimated flat-field", ff_vmin, ff_vmax),
        (axes[1, 2], err, "viridis", True, "Abs error |FF - GT| (mean-norm)", None, None),
    ]
    for ax, data, cmap, contours, title, vmin, vmax in panels:
        imshow_kw: dict = {"vmin": vmin, "vmax": vmax} if vmin is not None else {}
        im = ax.imshow(data, cmap=cmap, interpolation="nearest", **imshow_kw)
        if contours:
            ax.contour(data, levels=10, colors="w", linewidths=0.6, alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vignette_correction_{kind}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}  (Pearson r = {corr:.3f})")


@pytest.mark.parametrize("kind", ["gaussian", "zernike"])
def test_vignette_recovery(kind: str, tmp_path: Path) -> None:
    """BaSiC recovers the sbh-generated vignette from the tiled source image."""
    vignette = _generate_vignette(kind, tmp_path)
    stack = _tile_source_image() * vignette[np.newaxis]

    model = BaSiC(stack, estimate_darkfield=False)
    model.prepare()
    model.run()

    corr = _pearson(model.flatfield_fullsize, vignette)

    artifact_dir = os.environ.get(_ARTIFACT_DIR_ENV)
    if artifact_dir:
        _save_figure(kind, vignette, stack, model, corr, Path(artifact_dir))

    assert corr > _CORRELATION_THRESHOLD, f"[{kind}] recovered flat-field correlation {corr:.3f} <= {_CORRELATION_THRESHOLD}"
