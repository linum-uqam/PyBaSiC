"""Sample data bundled with the linum-basic package.

The ``linum_basic/data/`` directory ships a single greyscale photograph
(a Landsat scene) that is used as test input throughout the documentation
and example notebooks.
"""

from __future__ import annotations

from importlib.resources import files

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = ["load_sample_image"]


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
