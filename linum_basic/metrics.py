"""Seam-consistency metrics for mosaic-grid shading correction.

The core idea: adjacent tiles in an OCT mosaic physically overlap by
~20 % of their width/height.  After ideal shading correction the
overlapping pixels of two neighbours should be identical (they image
the same tissue).  Measuring the disagreement at every seam gives a
self-supervised quality metric that requires **no ground truth**.

Two metrics are provided:

``seam_l1``
    Mean absolute intensity difference across all seams, normalised by
    the mean tile intensity so the value is scale-invariant.  Lower is
    better; zero means perfect agreement.

``seam_pearson``
    Mean Pearson correlation between the paired overlap regions.  Returns
    a value in ``[-1, 1]``; higher is better.  1 - ``seam_pearson`` is
    used as a loss.

``evaluate_correction`` applies ``(tile - darkfield) / flatfield`` then
computes both metrics and returns them as a dict.
"""

from __future__ import annotations

import numpy as np

from linum_basic.mosaic import SeamPair

__all__ = ["evaluate_correction", "seam_l1", "seam_pearson"]


def seam_l1(tiles: np.ndarray, seam_pairs: list[SeamPair]) -> float:
    """Mean absolute seam error normalised by mean tile intensity.

    Parameters
    ----------
    tiles : numpy.ndarray, shape (N, th, tw)
        Tile stack (already corrected).
    seam_pairs : list of SeamPair
        Seam descriptors from :meth:`~linum_basic.mosaic.MosaicGrid.seam_pairs`.

    Returns
    -------
    float
        ``mean(|overlap_a - overlap_b|) / (mean(tiles) + epsilon)``.
        Scale-invariant; 0 = perfect agreement.
    """
    if not seam_pairs:
        return 0.0

    diffs: list[float] = []
    for sp in seam_pairs:
        a = tiles[sp.idx_a][sp.slice_a].ravel()
        b = tiles[sp.idx_b][sp.slice_b].ravel()
        diffs.append(float(np.abs(a - b).mean()))

    norm = float(np.mean(np.abs(tiles))) + 1e-9
    return float(np.mean(diffs)) / norm


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
