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

import json
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
_DRIFT_THRESHOLD = 0.80
# Dark-field recovery uses the same optimisation pass as flat-field recovery.
# We enforce a lower threshold because the algorithm's primary objective is
# flat-field correction; dark-field magnitude and shape are secondary.
_DF_CORRELATION_THRESHOLD = 0.30
_ARTIFACT_DIR_ENV = "LINUM_BASIC_VIGNETTE_ARTIFACT_DIR"

pytestmark = pytest.mark.skipif(
    not _SBH_SIMULATOR_AVAILABLE,
    reason="sbh-simulator not installed. Install with: uv pip install sbh-simulator",
)

# Accumulates Pearson-r metrics during a test session so that the session
# fixture below can write a summary into the artifact directory.
_METRICS: dict[str, dict[str, dict[str, object]]] = {}


@pytest.fixture(scope="module", autouse=True)
def _write_metrics_artifacts() -> object:
    """Write metrics.json and metrics_table.md after all module tests run."""
    yield
    artifact_dir = os.environ.get(_ARTIFACT_DIR_ENV)
    if not artifact_dir or not _METRICS:
        return
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(_METRICS, indent=2))

    rows = [
        "| Test | Kind | FF Pearson *r* | DF Pearson *r* | Pass |",
        "|------|------|:---:|:---:|:---:|",
    ]
    label_map = {
        "vignette_recovery": "Vignette recovery (with DF)",
        "no_darkfield": "Vignette recovery (no DF)",
        "brightness_drift": "Brightness drift robustness",
    }
    for test_key in ("vignette_recovery", "no_darkfield", "brightness_drift"):
        if test_key not in _METRICS:
            continue
        label = label_map.get(test_key, test_key)
        for kind in ("gaussian", "zernike"):
            if kind not in _METRICS[test_key]:
                continue
            vals = _METRICS[test_key][kind]
            ff_r = f"**{vals['ff_r']:.3f}**" if "ff_r" in vals else "\u2014"
            df_r = f"{vals['df_r']:.3f}" if "df_r" in vals else "\u2014"
            passed = "\u2705" if vals.get("passed", True) else "\u274c"
            rows.append(f"| {label} | {kind.capitalize()} | {ff_r} | {df_r} | {passed} |")
    (out / "metrics_table.md").write_text("\n".join(rows))


def _generate_ground_truth(kind: str, *, order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Generate a (flatfield, darkfield) pair for *kind* using the Python API.

    Both arrays are float32 of shape (_TILE, _TILE).  The flat-field is
    normalised to max == 1 (the simulator's native convention): the center of
    the field has the highest transmittance (1.0) and all other pixels are
    strictly below 1.0, so the corruption model only ever darkens pixels —
    never brightens them.  The dark-field peak is 0.05.
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
    # The simulator already normalises to max == 1; no further rescaling is
    # needed.  Renormalising to mean == 1 (as BaSiC does internally) would push
    # the centre above 1.0 and artificially brighten those pixels — which is
    # physically wrong for a vignette.  Pearson correlation (used in the
    # assertions) is scale-invariant, so BaSiC's mean == 1 estimate and this
    # max == 1 ground truth are directly comparable.
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
      Row 2: Sample tile clean (GT) | Sample tile corrupted | Sample tile BaSiC-corrected
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
    # Recover the clean (pre-corruption) tiles from ground-truth fields so we can
    # show the full clean → corrupted → corrected story in row 2.
    clean = np.clip((stack - gt_darkfield[np.newaxis]) / (gt_flatfield[np.newaxis] + 1e-9), 0.0, 1.0)
    # Pick the tile where the vignette edge-darkening effect is most visible:
    # the one with the most content in the peripheral region (where F < flat-field mean,
    # so the vignette multiplier < 1 and the tile is clearly darkened by the corruption).
    edge_mask = gt_flatfield < gt_flatfield.mean()
    best = int(np.argmax(stack[:, edge_mask].mean(axis=-1)))
    # Shared colour ranges for GT vs estimated so they are directly comparable.
    ff_vmin = min(float(gt_flatfield.min()), float(ff_est.min()))
    ff_vmax = max(float(gt_flatfield.max()), float(ff_est.max()))
    df_vmin = 0.0
    df_vmax = max(float(gt_darkfield.max()), float(df_est.max()))
    # All three tile panels share the clean-tile range so any deviation is clearly visible.
    tile_vmin = float(clean[best].min())
    tile_vmax = float(clean[best].max())

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
        # Row 2: clean → corrupted → corrected, all on the same scale.
        # The vignette darkens edges (F < 1) so the corrupted tile should appear darker
        # at the periphery; BaSiC-corrected should match the clean reference.
        (axes[2, 0], clean[best], "gray", False, "Sample tile: clean (GT)", tile_vmin, tile_vmax),
        (axes[2, 1], stack[best], "gray", False, "Sample tile: corrupted", tile_vmin, tile_vmax),
        (axes[2, 2], corrected[best], "gray", False, "Sample tile: BaSiC-corrected", tile_vmin, tile_vmax),
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

    _METRICS.setdefault("vignette_recovery", {})[kind] = {
        "ff_r": round(ff_corr, 3),
        "df_r": round(df_corr, 3),
        "passed": ff_corr > _CORRELATION_THRESHOLD and df_corr > _DF_CORRELATION_THRESHOLD,
    }

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

    _METRICS.setdefault("no_darkfield", {})[kind] = {
        "ff_r": round(ff_corr, 3),
        "passed": ff_corr > _CORRELATION_THRESHOLD,
    }

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

    _METRICS.setdefault("brightness_drift", {})[kind] = {
        "ff_r": round(ff_corr, 3),
        "passed": ff_corr > _DRIFT_THRESHOLD,
    }

    # Drift makes recovery harder than the no-drift baseline; 0.80 is still a
    # very strong correlation and confirms the algorithm is not defeated by
    # per-tile brightness variation.
    assert ff_corr > _DRIFT_THRESHOLD, (
        f"[{kind}/b_t-drift] flat-field correlation {ff_corr:.3f} <= {_DRIFT_THRESHOLD} "
        f"(brightness drift std=0.2 should not prevent recovery)"
    )
