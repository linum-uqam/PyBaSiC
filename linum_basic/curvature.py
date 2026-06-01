"""Flatfield focal-curvature analysis for per-z mosaic seam metrics.

When BaSiC is fit per z-level (``field_mode="per-z"``,
:func:`~linum_basic.fit.fit_mosaic`), each flatfield slice captures the
Gaussian illumination profile at that depth — the focal curvature.  In OCT
data this Gaussian narrows to a minimum at the focal plane, then broadens
above and below it.

This module provides three entry points:

``fit_focal_gaussian``
    Fit a 1-D Gaussian + constant baseline to a flatfield profile.  Uses
    moment-based initial guesses so it is robust to off-centre focal spots.

``curvature_depth_profile``
    For every z-level, extract the seam-relevant mean profile of the flatfield
    and fit a Gaussian.  Returns a list of :class:`GaussianParams` that can be
    plotted to visualise the focal curve (sigma(z), amplitude(z), centre(z)).

``seam_curvature``
    A scalar quality metric (lower = better).  For each z-level and each seam
    orientation present in *seam_pairs*, extract the mean flatfield profile,
    fit a Gaussian, then compute the normalised RMS residual of the fit
    *only in the overlap regions*.  A flatfield that faithfully follows the
    Gaussian optics model scores near zero; artefacts or poorly estimated
    fields score higher.

``seam_curvature_per_z``
    Same as :func:`seam_curvature` but returns one value per z-level,
    enabling depth-profile plots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from linum_basic.mosaic import SeamPair

__all__ = [
    "GaussianParams",
    "curvature_depth_profile",
    "fit_focal_gaussian",
    "focal_profile",
    "seam_curvature",
    "seam_curvature_per_z",
]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, init=False)
class GaussianParams:
    """Result of fitting a 1-D Gaussian to a flatfield profile.

    Attributes
    ----------
    amplitude : float
        Peak amplitude above the baseline offset.
    center : float
        Pixel position of the Gaussian peak.  May lie outside the profile
        range for an off-centre focal spot.
    sigma : float
        Standard deviation of the Gaussian in pixels.  ``nan`` when the
        fit did not converge.
    offset : float
        Constant baseline added to the Gaussian.
    rms_residual : float
        Normalised root-mean-square residual of the fit over the *full*
        profile — ``rms(profile - fit) / mean(|profile|)``.  ``nan`` on
        fit failure.
    """

    amplitude: float
    center: float
    sigma: float
    offset: float
    rms_residual: float

    def __init__(
        self,
        amplitude: float,
        center: float,
        sigma: float,
        offset: float,
        rms_residual: float,
    ) -> None:
        """Construct GaussianParams from individual fit results."""
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "rms_residual", rms_residual)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gaussian_1d(
    x: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    offset: float,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset


def _overlap_slices_1d(seam: SeamPair) -> tuple[slice, slice]:
    """Return the 1-D overlap slices within the mean profile for *seam*.

    For a horizontal seam the profile axis is columns (x); for a vertical
    seam it is rows (y).

    Returns
    -------
    sl_a, sl_b : slice, slice
        Profile slice for the A-side (right / bottom edge) and B-side
        (left / top edge) of the seam.
    """
    if seam.orientation == "horizontal":
        return seam.slice_a[1], seam.slice_b[1]
    return seam.slice_a[0], seam.slice_b[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_focal_gaussian(profile: np.ndarray) -> GaussianParams:
    """Fit a 1-D Gaussian + constant baseline to *profile*.

    Uses moment-based initial guesses so the fit is robust even when the
    focal spot is significantly off-centre.  Falls back to a
    :class:`GaussianParams` of all ``nan`` when the optimiser does not
    converge (e.g. flat profile with fewer than 4 points).

    Parameters
    ----------
    profile : numpy.ndarray, shape (N,)
        1-D array of positive values (e.g. a mean flatfield row or column).

    Returns
    -------
    GaussianParams
        Fitted parameters.  Check ``np.isnan(result.sigma)`` to detect fit
        failure.
    """
    n = len(profile)
    _nan = float("nan")
    _fail = GaussianParams(amplitude=_nan, center=_nan, sigma=_nan, offset=_nan, rms_residual=_nan)
    if n < 4:
        return _fail

    x = np.arange(n, dtype=np.float64)
    p = profile.astype(np.float64)

    # Moment-based initial guesses
    offset0 = float(p.min())
    amp = p - offset0
    amp_sum = float(amp.sum()) or 1.0
    center0 = float((x * amp).sum() / amp_sum)
    var = float((amp * (x - center0) ** 2).sum() / amp_sum)
    sigma0 = float(max(np.sqrt(var) if var > 0 else 1.0, 1.0))
    amplitude0 = float(amp.max()) or 1.0

    try:
        popt, _ = curve_fit(
            _gaussian_1d,
            x,
            p,
            p0=[amplitude0, center0, sigma0, offset0],
            bounds=([-np.inf, -np.inf, 1e-3, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=2000,
        )
        fitted = _gaussian_1d(x, *popt)
        local = float(np.abs(p).mean()) + 1e-9
        rms = float(np.sqrt(np.mean((p - fitted) ** 2))) / local
        return GaussianParams(
            amplitude=float(popt[0]),
            center=float(popt[1]),
            sigma=float(popt[2]),
            offset=float(popt[3]),
            rms_residual=rms,
        )
    except RuntimeError, ValueError:
        return _fail


def focal_profile(field_2d: np.ndarray, orientation: str) -> np.ndarray:
    """Extract the seam-relevant 1-D mean profile from a 2-D flatfield.

    Parameters
    ----------
    field_2d : numpy.ndarray, shape (th, tw)
        A single z-slice of the per-z flatfield.
    orientation : {"horizontal", "vertical"}
        Seam orientation.  ``"horizontal"`` seams separate left-right
        neighbours — the relevant profile spans columns (mean over rows).
        ``"vertical"`` seams separate top-bottom neighbours — the profile
        spans rows (mean over columns).

    Returns
    -------
    numpy.ndarray, shape (tw,) or (th,)
        1-D mean profile along the relevant axis.
    """
    if orientation == "horizontal":
        return field_2d.mean(axis=0)  # average over rows → (tw,)
    return field_2d.mean(axis=1)  # average over cols → (th,)


def seam_curvature(
    flatfields: np.ndarray,
    seam_pairs: list[SeamPair],
) -> float:
    """Gaussian focal-curvature consistency metric over all z-levels.

    For each z-level and each seam orientation present in *seam_pairs*,
    extract the 1-D mean profile of the flatfield in the seam direction, fit
    a Gaussian, and measure the normalised RMS residual of the fit in the
    overlap regions specifically.  A flatfield that faithfully captures the
    Gaussian optics model will have a small residual; artefact-corrupted or
    non-Gaussian fields will score higher.

    Because the per-z flatfield is shared across all tiles (one BaSiC model
    per depth), the overlap residual is the same for every seam of the same
    orientation at a given z.  The metric therefore averages over
    ``(z, orientation)`` pairs rather than individual seams.

    Lower is better; 0 = perfect Gaussian fit in every overlap region.

    Parameters
    ----------
    flatfields : numpy.ndarray, shape (Z, th, tw)
        Per-z flatfield stack from :attr:`~linum_basic.fit.MosaicFit.flatfields`.
    seam_pairs : list of SeamPair
        Seam descriptors from :meth:`~linum_basic.mosaic.MosaicGrid.seam_pairs`.

    Returns
    -------
    float
        Mean normalised Gaussian-fit residual in the overlap regions.
    """
    if not seam_pairs or flatfields.ndim != 3:
        return 0.0

    n_z = flatfields.shape[0]

    # One representative seam per orientation — overlap slices are identical
    # for all seams of the same orientation (tiles are equal-sized).
    orient_to_rep: dict[str, SeamPair] = {}
    for sp in seam_pairs:
        if sp.orientation not in orient_to_rep:
            orient_to_rep[sp.orientation] = sp

    residuals: list[float] = []

    for z in range(n_z):
        field = flatfields[z]  # (th, tw)
        for orient, rep_seam in orient_to_rep.items():
            profile = focal_profile(field, orient)
            n = len(profile)

            params = fit_focal_gaussian(profile)
            if np.isnan(params.sigma):
                continue

            x = np.arange(n, dtype=np.float64)
            fitted = _gaussian_1d(x, params.amplitude, params.center, params.sigma, params.offset)

            sl_a, sl_b = _overlap_slices_1d(rep_seam)
            ovl_a = profile[sl_a]
            ovl_b = profile[sl_b]
            fit_a = fitted[sl_a]
            fit_b = fitted[sl_b]

            combined_err = np.concatenate([(ovl_a - fit_a) ** 2, (ovl_b - fit_b) ** 2])
            local_mean = float(np.abs(np.concatenate([ovl_a, ovl_b])).mean()) + 1e-9
            rms = float(np.sqrt(combined_err.mean()))
            residuals.append(rms / local_mean)

    return float(np.mean(residuals)) if residuals else 0.0


def seam_curvature_per_z(
    flatfields: np.ndarray,
    seam_pairs: list[SeamPair],
) -> np.ndarray:
    """Per-z :func:`seam_curvature` values for depth-profile visualisation.

    Parameters
    ----------
    flatfields : numpy.ndarray, shape (Z, th, tw)
        Per-z flatfield stack.
    seam_pairs : list of SeamPair
        Seam descriptors.

    Returns
    -------
    numpy.ndarray, shape (Z,)
        Seam curvature residual for each z-level.
    """
    n_z = flatfields.shape[0]
    return np.array([seam_curvature(flatfields[z : z + 1], seam_pairs) for z in range(n_z)])


def curvature_depth_profile(
    flatfields: np.ndarray,
    seam_pairs: list[SeamPair],
    *,
    orientation: str = "horizontal",
) -> list[GaussianParams]:
    """Fit a Gaussian to the flatfield profile at each z-level.

    Enables tracking of focal-curve parameters — sigma(z), amplitude(z),
    centre(z) — as a function of depth, which reveals the focal plane
    location and the rate at which the Gaussian broadens away from focus.

    Parameters
    ----------
    flatfields : numpy.ndarray, shape (Z, th, tw)
        Per-z flatfield stack.
    seam_pairs : list of SeamPair
        Used to determine the profile axis.  The orientation of the first
        seam in the list is used; if *seam_pairs* is empty, *orientation*
        is used as the fallback.
    orientation : {"horizontal", "vertical"}
        Fallback axis when *seam_pairs* is empty.

    Returns
    -------
    list of GaussianParams
        One entry per z-level (length Z).
    """
    if flatfields.ndim != 3:
        return []

    orient = orientation
    for sp in seam_pairs:
        orient = sp.orientation
        break

    return [fit_focal_gaussian(focal_profile(flatfields[z], orient)) for z in range(flatfields.shape[0])]
