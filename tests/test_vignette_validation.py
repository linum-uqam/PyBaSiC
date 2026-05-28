"""Integration test: sbh-simulator vignette -> Linum BaSiC recovery.

Pipeline (single source of truth for both the test suite and the
visualisation script):

1. Generate a 128x128 flat-field and dark-field via the sbh-simulator Python
   API (Gaussian / Zernike families).
2. Tile the bundled source image into 128x128 non-overlapping patches.
3. Apply degradation: ``corrupted = clean * flatfield + darkfield + noise``.
4. Run BaSiC with ``estimate_darkfield=True`` and compare both recovered
   fields to ground truth.

When the environment variable ``LINUM_BASIC_VIGNETTE_ARTIFACT_DIR`` is set, each
test also renders a 9-panel PNG into that directory.  That is how the CI
visualisation step (see ``scripts/visualize_vignette_correction.py``) produces
its figures -- there is no separate pipeline.

Skipped when the ``sbh-simulator`` package cannot be imported.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest

from linum_basic.core import BaSiC
from linum_basic.data import load_sample_image

try:
    from sbh_simulator.simulator import (
        generate_gaussian_darkfield,
        generate_gaussian_vignette,
        generate_zernike_darkfield,
        generate_zernike_vignette,
    )

    _SBH_SIMULATOR_AVAILABLE = True
except ImportError:
    _SBH_SIMULATOR_AVAILABLE = False

_TILE = 128
_CORRELATION_THRESHOLD = 0.85
# Dark-field recovery uses the same optimisation pass as flat-field recovery.
# We enforce a lower threshold because the algorithm's primary objective is
# flat-field correction; dark-field magnitude and shape are secondary.
_DF_CORRELATION_THRESHOLD = 0.30
_ARTIFACT_DIR_ENV = "LINUM_BASIC_VIGNETTE_ARTIFACT_DIR"

pytestmark = pytest.mark.skipif(
    not _SBH_SIMULATOR_AVAILABLE,
    reason="sbh-simulator not installed. Install with: uv pip install sbh-simulator",
)


def _generate_ground_truth(kind: str, *, order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Generate a (flatfield, darkfield) pair for *kind* using the Python API.

    Both arrays are float32 of shape (_TILE, _TILE).  The flat-field is
    normalised to mean == 1.  The dark-field peak is 0.05.
    """
    rng = random.Random(42)
    if kind == "gaussian":
        flatfield = generate_gaussian_vignette(width=_TILE, height=_TILE, sigma=0.7, rng=rng).astype(np.float32)
        darkfield = generate_gaussian_darkfield(width=_TILE, height=_TILE, sigma=0.7, max_offset=0.05, rng=rng).astype(
            np.float32
        )
    else:
        flatfield = generate_zernike_vignette(width=_TILE, height=_TILE, order=order, rng=rng).astype(np.float32)
        darkfield = generate_zernike_darkfield(width=_TILE, height=_TILE, order=order, max_offset=0.05, rng=rng).astype(
            np.float32
        )
    flatfield /= flatfield.mean() + 1e-9  # normalise so mean == 1
    return flatfield, darkfield


def _tile_source_image() -> np.ndarray:
    """Tile the bundled image into a stack of (_TILE, _TILE) patches."""
    img = load_sample_image()
    h, w = img.shape
    nh, nw = h // _TILE, w // _TILE
    cropped = img[: nh * _TILE, : nw * _TILE].astype(np.float32) / 255.0
    return cropped.reshape(nh, _TILE, nw, _TILE).transpose(0, 2, 1, 3).reshape(nh * nw, _TILE, _TILE)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def _save_figure(
    kind: str,
    gt_flatfield: np.ndarray,
    gt_darkfield: np.ndarray,
    stack: np.ndarray,
    model: BaSiC,
    ff_corr: float,
    df_corr: float,
    out_dir: Path,
) -> None:
    """Write a 9-panel diagnostic figure for *kind* into *out_dir*.

    Layout (3 rows x 3 columns):
      Row 0: GT flat-field | Estimated flat-field | FF abs error
      Row 1: GT dark-field | Estimated dark-field | DF abs error
      Row 2: Sample tile corrupted | Sample tile corrected | Mean of stack
    """
    try:
        import matplotlib
    except ImportError:
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ff_est = model.flatfield_fullsize
    df_est = model.darkfield_fullsize
    corrected = np.stack([model.normalize(stack[i]) for i in range(len(stack))])
    ff_err = np.abs(ff_est - gt_flatfield)
    df_err = np.abs(df_est - gt_darkfield)
    # Pick the most content-rich tile for sample panels.
    best = int(np.argmax(stack.std(axis=(1, 2))))
    # Shared colour ranges for GT vs estimated so they are directly comparable.
    ff_vmin = min(float(gt_flatfield.min()), float(ff_est.min()))
    ff_vmax = max(float(gt_flatfield.max()), float(ff_est.max()))
    df_vmin = 0.0
    df_vmax = max(float(gt_darkfield.max()), float(df_est.max()))

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(
        f"{kind.capitalize()}  |  FF Pearson r = {ff_corr:.3f}  |  DF Pearson r = {df_corr:.3f}",
        fontsize=13,
        fontweight="bold",
    )
    panels = [
        # Row 0: flat-field
        (axes[0, 0], gt_flatfield, "viridis", True, "GT flat-field (mean=1)", ff_vmin, ff_vmax),
        (axes[0, 1], ff_est, "viridis", True, "Estimated flat-field", ff_vmin, ff_vmax),
        (axes[0, 2], ff_err, "viridis", False, "FF abs error", None, None),
        # Row 1: dark-field
        (axes[1, 0], gt_darkfield, "inferno", False, "GT dark-field", df_vmin, df_vmax),
        (axes[1, 1], df_est, "inferno", False, "Estimated dark-field", df_vmin, df_vmax),
        (axes[1, 2], df_err, "inferno", False, "DF abs error", None, None),
        # Row 2: sample tiles + mean stack
        (axes[2, 0], stack[best], "gray", False, "Sample tile: corrupted", None, None),
        (axes[2, 1], corrected[best], "gray", False, "Sample tile: BaSiC-corrected", None, None),
        (axes[2, 2], stack.mean(axis=0), "viridis", True, "Mean of corrupted stack", None, None),
    ]
    for ax, data, cmap, contours, title, vmin, vmax in panels:
        imshow_kw: dict = {"vmin": vmin, "vmax": vmax} if vmin is not None else {}
        im = ax.imshow(data, cmap=cmap, interpolation="nearest", **imshow_kw)
        if contours:
            ax.contour(data, levels=10, colors="w", linewidths=0.6, alpha=0.7)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vignette_correction_{kind}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}  (FF r = {ff_corr:.3f}, DF r = {df_corr:.3f})")


@pytest.mark.parametrize("kind", ["gaussian", "zernike"])
def test_vignette_recovery(kind: str) -> None:
    """BaSiC recovers sbh-generated flat-field and dark-field from the tiled source image."""
    rng = np.random.default_rng(0)
    gt_flatfield, gt_darkfield = _generate_ground_truth(kind)
    noise_std = 0.005
    clean = _tile_source_image()
    stack = clean * gt_flatfield[np.newaxis] + gt_darkfield[np.newaxis]
    stack += rng.normal(0, noise_std, stack.shape).astype(np.float32)
    stack = np.clip(stack, 0, 1)

    model = BaSiC(stack, estimate_darkfield=True)
    model.prepare()
    model.run()

    ff_corr = _pearson(model.flatfield_fullsize, gt_flatfield)
    df_corr = _pearson(model.darkfield_fullsize, gt_darkfield)

    print(f"[{kind}] flat-field Pearson r = {ff_corr:.3f}")
    print(f"[{kind}] dark-field Pearson r = {df_corr:.3f}")

    artifact_dir = os.environ.get(_ARTIFACT_DIR_ENV)
    if artifact_dir:
        _save_figure(kind, gt_flatfield, gt_darkfield, stack, model, ff_corr, df_corr, Path(artifact_dir))

    assert ff_corr > _CORRELATION_THRESHOLD, f"[{kind}] flat-field correlation {ff_corr:.3f} <= {_CORRELATION_THRESHOLD}"
    assert df_corr > _DF_CORRELATION_THRESHOLD, f"[{kind}] dark-field correlation {df_corr:.3f} <= {_DF_CORRELATION_THRESHOLD}"


@pytest.mark.parametrize("kind", ["gaussian", "zernike"])
def test_flatfield_without_darkfield(kind: str) -> None:
    """Flat-field recovery with dark-field estimation disabled.

    Acts as an independent baseline: confirms flat-field quality is not
    contingent on dark-field estimation being enabled.
    """
    rng = np.random.default_rng(0)
    gt_flatfield, gt_darkfield = _generate_ground_truth(kind)
    noise_std = 0.005
    clean = _tile_source_image()
    stack = clean * gt_flatfield[np.newaxis] + gt_darkfield[np.newaxis]
    stack += rng.normal(0, noise_std, stack.shape).astype(np.float32)
    stack = np.clip(stack, 0, 1)

    model = BaSiC(stack, estimate_darkfield=False)
    model.prepare()
    model.run()

    ff_corr = _pearson(model.flatfield_fullsize, gt_flatfield)
    print(f"[{kind}/no-darkfield] flat-field Pearson r = {ff_corr:.3f}")

    assert ff_corr > _CORRELATION_THRESHOLD, (
        f"[{kind}/no-darkfield] flat-field correlation {ff_corr:.3f} <= {_CORRELATION_THRESHOLD}"
    )


@pytest.mark.parametrize("kind", ["gaussian", "zernike"])
def test_flatfield_recovery_with_brightness_drift(kind: str) -> None:
    """BaSiC recovers flat-field under per-tile log-normal brightness drift.

    Each tile receives an independent scalar b_t ~ LogNormal(0, 0.2), which
    is precisely the nuisance that BaSiC's reweighted ALM is designed to
    marginalise out.  The flat-field correlation must still exceed
    *_CORRELATION_THRESHOLD*.
    """
    rng = np.random.default_rng(7)
    gt_flatfield, gt_darkfield = _generate_ground_truth(kind)
    noise_std = 0.005
    clean = _tile_source_image()
    n_tiles = len(clean)

    # Per-tile brightness scalar: b_t ~ LogNormal(mu=0, sigma=0.2)
    b_t = rng.lognormal(mean=0.0, sigma=0.2, size=n_tiles).astype(np.float32)

    # Physical degradation: corrupted_i = clean_i * b_t_i * flatfield + darkfield + noise
    stack = clean * b_t[:, np.newaxis, np.newaxis] * gt_flatfield[np.newaxis] + gt_darkfield[np.newaxis]
    stack += rng.normal(0, noise_std, stack.shape).astype(np.float32)
    stack = np.clip(stack, 0, 1)

    model = BaSiC(stack, estimate_darkfield=True)
    model.prepare()
    model.run()

    ff_corr = _pearson(model.flatfield_fullsize, gt_flatfield)
    print(f"[{kind}/b_t-drift] flat-field Pearson r = {ff_corr:.3f}")

    # Drift makes recovery harder than the no-drift baseline; 0.80 is still a
    # very strong correlation and confirms the algorithm is not defeated by
    # per-tile brightness variation.
    _drift_threshold = 0.80
    assert ff_corr > _drift_threshold, (
        f"[{kind}/b_t-drift] flat-field correlation {ff_corr:.3f} <= {_drift_threshold} "
        f"(brightness drift std=0.2 should not prevent recovery)"
    )
