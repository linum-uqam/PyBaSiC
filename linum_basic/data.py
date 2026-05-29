"""Sample data bundled with the linum-basic package.

The ``linum_basic/data/`` directory ships a single greyscale photograph
(a Landsat scene) that is used as test input throughout the documentation
and example notebooks.
"""

from __future__ import annotations

import math
from importlib.resources import files

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = ["load_sample_image", "zernike_flatfield"]


def load_sample_image() -> NDArray[np.uint8]:
    """Load the bundled greyscale sample image.

    The image is a Landsat satellite scene distributed with the package
    for demonstration and testing purposes.  It is read in greyscale and
    returned as a 2-D ``uint8`` array.

    Returns
    -------
    numpy.ndarray
        2-D array of shape *(H, W)* and dtype ``uint8``.

    Examples
    --------
    >>> from linum_basic.data import load_sample_image
    >>> img = load_sample_image()
    >>> img.ndim
    2
    >>> img.dtype
    dtype('uint8')
    """
    raw = (files("linum_basic") / "data" / "source_image.jpg").read_bytes()
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:  # pragma: no cover
        msg = "Failed to decode bundled sample image."
        raise RuntimeError(msg)
    return img.astype(np.uint8)


def _zernike_radial(n: int, m: int, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the Zernike radial polynomial ``R_n^m`` on ``rho``.

    Parameters
    ----------
    n : int
        Radial order (non-negative).
    m : int
        Azimuthal frequency; ``abs(m) <= n`` and ``n - abs(m)`` must be even.
    rho : numpy.ndarray
        Radial coordinate on the unit disk.

    Returns
    -------
    numpy.ndarray
        The radial polynomial evaluated element-wise.
    """
    m = abs(m)
    out = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        coef = (
            (-1) ** k
            * math.factorial(n - k)
            / (math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k))
        )
        out += coef * rho ** (n - 2 * k)
    return out


def _zernike(n: int, m: int, rho: NDArray[np.float64], theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the full Zernike polynomial ``Z_n^m`` (radial x angular)."""
    radial = _zernike_radial(n, m, rho)
    if m >= 0:
        return radial * np.cos(m * theta)
    return radial * np.sin(-m * theta)


def zernike_flatfield(
    size: int,
    *,
    n_max: int = 4,
    coeffs: dict[tuple[int, int], float] | None = None,
    contrast: float = 0.4,
    seed: int = 0,
) -> NDArray[np.float32]:
    """Generate a smooth synthetic flat-field from low-order Zernike modes.

    The field is built as a weighted sum of Zernike polynomials up to radial
    order ``n_max`` defined on the disk inscribing the tile, normalised so the
    perturbation has unit RMS and then scaled around a mean gain of 1.  Odd
    modes (tilt, coma, astigmatism) introduce the asymmetric shading that makes
    inter-tile seams visible, which is useful for demonstrating correction.

    Parameters
    ----------
    size : int
        Side length of the square field in pixels.
    n_max : int, optional
        Maximum Zernike radial order to include (default 4).  Orders of 4-5
        produce a pronounced yet smooth shading.
    coeffs : dict of {(n, m): float}, optional
        Explicit mode coefficients keyed by ``(radial_order, azimuthal_freq)``.
        When omitted, reproducible pseudo-random coefficients are drawn and
        weighted by ``1 / n`` so lower orders dominate.
    contrast : float, optional
        Peak-to-mean modulation strength of the gain (default 0.4).
    seed : int, optional
        Seed for the default random coefficients (default 0).

    Returns
    -------
    numpy.ndarray
        2-D ``float32`` flat-field of shape *(size, size)* with mean 1 and
        strictly positive values.

    Examples
    --------
    >>> from linum_basic.data import zernike_flatfield
    >>> field = zernike_flatfield(64, n_max=4, seed=0)
    >>> field.shape
    (64, 64)
    >>> bool(field.min() > 0)
    True
    >>> bool(abs(field.mean() - 1.0) < 1e-5)
    True
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    center = (size - 1) / 2.0
    half_diag = math.hypot(center, center)
    rho = np.hypot(xx - center, yy - center) / half_diag
    theta = np.arctan2(yy - center, xx - center)

    modes = [(n, m) for n in range(1, n_max + 1) for m in range(-n, n + 1, 2)]
    if coeffs is None:
        coeffs = {(n, m): float(rng.normal()) / n for (n, m) in modes}

    field = np.zeros_like(rho)
    for (n, m), amp in coeffs.items():
        field += amp * _zernike(n, m, rho, theta)

    field -= field.mean()
    rms = float(np.sqrt(np.mean(field**2)))
    if rms > 0:
        field /= rms
    flat = 1.0 + contrast * field
    flat = np.clip(flat, 0.05, None)
    flat /= flat.mean()
    return flat.astype(np.float32)
