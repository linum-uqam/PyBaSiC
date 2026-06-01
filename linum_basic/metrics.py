"""Seam-consistency metrics for mosaic-grid shading correction.

The core idea: adjacent tiles in an OCT mosaic physically overlap by
~20 % of their width/height.  After ideal shading correction the
overlapping pixels of two neighbours should be identical (they image
the same tissue).  Measuring the disagreement at every seam gives a
self-supervised quality metric that requires **no ground truth**.

Two intensity metrics are provided:

``seam_l1``
    Mean *per-seam relative* absolute intensity difference.  For every
    seam the disagreement ``mean|a - b|`` is divided by the local mean
    brightness of that seam ``mean((|a| + |b|) / 2)``, then averaged over
    all seams.  Normalising each seam by its own local intensity makes
    the metric scale-invariant (a global gain leaves it unchanged) and
    physical (a 5-count step on a 10-count background scores worse than
    on a 1000-count background), and it cannot be gamed by brightening
    tile interiors away from the seams.  Lower is better; 0 = perfect
    agreement.

``seam_pearson``
    Mean Pearson correlation between the paired overlap regions.  Returns
    a value in ``[-1, 1]``; higher is better.  1 - ``seam_pearson`` is
    used as a loss.

``evaluate_correction`` applies ``(tile - darkfield) / flatfield`` then
computes both metrics and returns them as a dict.

For per-z (depth-resolved) evaluation of a full mosaic fitting run see
:func:`evaluate_correction_volume`, which combines the intensity seam
metrics with the flatfield focal-curvature metric from
:mod:`linum_basic.curvature`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from linum_basic.curvature import seam_curvature
from linum_basic.mosaic import MosaicGrid, SeamPair

if TYPE_CHECKING:
    from linum_basic.fit import MosaicFit

__all__ = [
    "evaluate_correction",
    "evaluate_correction_volume",
    "seam_l1",
    "seam_pearson",
]


def seam_l1(tiles: np.ndarray, seam_pairs: list[SeamPair]) -> float:
    """Mean per-seam relative absolute error.

    Parameters
    ----------
    tiles : numpy.ndarray, shape (N, th, tw)
        Tile stack (already corrected).
    seam_pairs : list of SeamPair
        Seam descriptors from :meth:`~linum_basic.mosaic.MosaicGrid.seam_pairs`.

    Returns
    -------
    float
        Mean over seams of ``mean(|a - b|) / (mean((|a| + |b|) / 2) + epsilon)``.
        Scale-invariant and physical; 0 = perfect agreement.
    """
    if not seam_pairs:
        return 0.0

    rels: list[float] = []
    for sp in seam_pairs:
        a = tiles[sp.idx_a][sp.slice_a].ravel()
        b = tiles[sp.idx_b][sp.slice_b].ravel()
        local = float((np.abs(a) + np.abs(b)).mean()) / 2.0
        rels.append(float(np.abs(a - b).mean()) / (local + 1e-9))

    return float(np.mean(rels))


def seam_pearson(tiles: np.ndarray, seam_pairs: list[SeamPair]) -> float:
    """Mean Pearson correlation across all seam pairs.

    Parameters
    ----------
    tiles : numpy.ndarray, shape (N, th, tw)
        Tile stack (already corrected).
    seam_pairs : list of SeamPair
        Seam descriptors.

    Returns
    -------
    float
        Mean Pearson r in ``[-1, 1]``.  Pairs where either side has zero
        variance are skipped.  Returns 1.0 if no valid pair exists.
    """
    if not seam_pairs:
        return 1.0

    corrs: list[float] = []
    for sp in seam_pairs:
        a = tiles[sp.idx_a][sp.slice_a].ravel().astype(np.float64)
        b = tiles[sp.idx_b][sp.slice_b].ravel().astype(np.float64)
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(r):
            corrs.append(r)

    return float(np.mean(corrs)) if corrs else 1.0


def evaluate_correction(
    tiles: np.ndarray,
    flatfield: np.ndarray,
    darkfield: np.ndarray,
    seam_pairs: list[SeamPair],
) -> dict[str, float]:
    """Apply shading correction and return both seam metrics.

    Parameters
    ----------
    tiles : numpy.ndarray, shape (N, th, tw)
        Raw (uncorrected) tile stack.
    flatfield : numpy.ndarray, shape (th, tw)
        Flat-field estimate (normalised to mean ≈ 1).
    darkfield : numpy.ndarray, shape (th, tw)
        Dark-field estimate.
    seam_pairs : list of SeamPair
        Seam descriptors.

    Returns
    -------
    dict
        ``{"seam_l1": float, "seam_1minus_pearson": float}``.
    """
    corrected = (tiles.astype(np.float32) - darkfield[np.newaxis]) / (flatfield[np.newaxis] + 1e-6)
    return {
        "seam_l1": seam_l1(corrected, seam_pairs),
        "seam_1minus_pearson": 1.0 - seam_pearson(corrected, seam_pairs),
    }


def evaluate_correction_volume(
    mosaic: MosaicGrid,
    fit: MosaicFit,
    *,
    metrics: Sequence[str] = ("seam", "curvature"),
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Evaluate shading-correction quality over all z-levels in a mosaic fit.

    Combines the intensity seam metrics (``seam_l1``, ``seam_pearson``)
    with the flatfield focal-curvature metric
    (:func:`~linum_basic.curvature.seam_curvature`), selectable via the
    *metrics* parameter.  All scalar results are averages over the fitted
    z-levels.

    Parameters
    ----------
    mosaic : MosaicGrid
        The source mosaic volume; used to extract per-z tile stacks.
    fit : MosaicFit
        Fitted flat/dark-fields from :func:`~linum_basic.fit.fit_mosaic`.
        Supports both ``field_mode="per-z"`` (recommended — one field per z)
        and ``field_mode="global"`` (single shared field).
    metrics : sequence of str
        Subset of ``{"seam", "curvature"}`` to compute.  Default: both.
    epsilon : float
        Divisor stabilisation constant for the flat-field correction.

    Returns
    -------
    dict
        Subset of keys from
        ``{"seam_l1", "seam_1minus_pearson", "seam_curvature"}``,
        depending on *metrics*.

    Examples
    --------
    >>> from linum_basic import MosaicGrid, fit_mosaic
    >>> from linum_basic.metrics import evaluate_correction_volume
    >>> mosaic = MosaicGrid.from_ome_zarr("mosaic.ome.zarr")
    >>> fit = fit_mosaic(mosaic, field_mode="per-z")
    >>> scores = evaluate_correction_volume(mosaic, fit)
    >>> print(scores["seam_l1"], scores["seam_curvature"])
    """
    seam_pairs = mosaic.seam_pairs()
    result: dict[str, float] = {}

    # --- intensity seam metrics (averaged over fitted z-levels) ----------
    if "seam" in metrics:
        l1_vals: list[float] = []
        pearson_vals: list[float] = []
        for z_pos, z in enumerate(fit.z_indices):
            tiles = mosaic.iter_tiles(z)
            if fit.field_mode == "global":
                ff: np.ndarray = fit.flatfields
                df: np.ndarray = fit.darkfields
            else:
                ff = fit.flatfields[z_pos]
                df = fit.darkfields[z_pos]
            corrected = (tiles.astype(np.float32) - df[np.newaxis]) / (ff[np.newaxis] + epsilon)
            l1_vals.append(seam_l1(corrected, seam_pairs))
            pearson_vals.append(seam_pearson(corrected, seam_pairs))
        result["seam_l1"] = float(np.mean(l1_vals)) if l1_vals else 0.0
        result["seam_1minus_pearson"] = float(1.0 - np.mean(pearson_vals)) if pearson_vals else 0.0

    # --- flatfield curvature metric --------------------------------------
    if "curvature" in metrics:
        ffs = fit.flatfields[np.newaxis] if fit.field_mode == "global" else fit.flatfields
        result["seam_curvature"] = seam_curvature(ffs, seam_pairs)

    return result
