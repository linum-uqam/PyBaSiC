"""linum-basic — illumination correction for optical microscopy images.

Public API
----------
BaSiC
    Main estimator class.  See :class:`linum_basic.core.BaSiC` for full
    documentation.
correct_images
    One-call convenience function: accepts any input format accepted by
    :class:`BaSiC` and returns the corrected image stack.
algorithms
    Low-level ALM solver and soft-threshold operator exposed via
    :mod:`linum_basic.algorithms` for users who want direct access to the
    numerical core (e.g. custom reweighting schemes).

Quick start
-----------
See the :ref:`Getting Started <getting_started>` page in the online
documentation for install instructions, CLI reference, and worked examples.
See the :ref:`Parameter Tuning <parameters>` page for guidance on every
tuning knob exposed by :class:`BaSiC`.

Examples
--------
>>> import numpy as np
>>> from linum_basic import BaSiC, correct_images
>>> stack = np.random.rand(30, 64, 64).astype("float32")
>>> model = BaSiC(stack)
>>> model.run()  # prepare() is called automatically
>>> flatfield = model.get_flatfield()

One-call convenience:

>>> corrected = correct_images(stack)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from linum_basic.algorithms import inexact_alm_l1, shrink
from linum_basic.core import BaSiC
from linum_basic.curvature import (
    GaussianParams,
    curvature_depth_profile,
    fit_focal_gaussian,
    focal_profile,
    seam_curvature,
    seam_curvature_per_z,
)
from linum_basic.data import load_sample_image
from linum_basic.fit import MosaicFit, apply_fit, fit_mosaic
from linum_basic.metrics import evaluate_correction, evaluate_correction_volume
from linum_basic.mosaic import MosaicGrid, SeamPair
from linum_basic.tuning import (
    AutoApplyError,
    AutoTuneResult,
    BoundsRecommendation,
    TuneResult,
    auto_tune,
    recommend_bounds,
    tune,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def correct_images(
    input: str | Path | list[str | Path] | list[NDArray] | NDArray,
    *,
    estimate_darkfield: bool = False,
    backend: Literal["numpy", "torch", "auto"] = "numpy",
    device: str | None = None,
    verbose: bool = False,
    **knobs: Any,
) -> NDArray:
    """Estimate and apply BaSiC shading correction in a single call.

    Convenience wrapper around :class:`BaSiC` that runs the full pipeline
    (load → prepare → optimise → correct) and returns the corrected stack.

    Parameters
    ----------
    input : str, Path, list of str/Path, list of numpy.ndarray, or numpy.ndarray
        Input images — any format accepted by :class:`BaSiC`.
    estimate_darkfield : bool
        When ``True``, estimate a dark-field in addition to the flat-field.
    backend : {"numpy", "torch", "auto"}
        Compute backend for the ALM optimisation loop.
    device : str or None
        PyTorch device string (e.g. ``"cuda:0"``).  Ignored for NumPy.
    verbose : bool
        Show progress bars.
    **knobs
        Extra keyword arguments set as attributes on the :class:`BaSiC`
        instance before fitting (e.g. ``working_size=256``, ``l_s=0.5``).

    Returns
    -------
    numpy.ndarray, shape (N, H, W)
        Corrected image stack with the same dtype as the input.

    Examples
    --------
    >>> import numpy as np
    >>> from linum_basic import correct_images
    >>> stack = np.random.rand(30, 512, 512).astype(np.float32)
    >>> corrected = correct_images(stack)

    Enable dark-field estimation and use a finer working resolution:

    >>> corrected = correct_images(stack, estimate_darkfield=True, working_size=256)
    """
    model = BaSiC(
        input,
        estimate_darkfield=estimate_darkfield,
        backend=backend,
        device=device,
        verbose=verbose,
    )
    for key, val in knobs.items():
        setattr(model, key, val)
    model.run()  # auto-calls prepare()
    return BaSiC._apply_correction(
        model.img_stack.astype("float32"),
        model.flatfield_fullsize,
        model.darkfield_fullsize,
        1e-6,
        model.img_stack.dtype,
    )


__version__ = "2.0.0"
__all__ = [
    "AutoApplyError",
    "AutoTuneResult",
    "BaSiC",
    "BoundsRecommendation",
    "GaussianParams",
    "MosaicFit",
    "MosaicGrid",
    "SeamPair",
    "TuneResult",
    "__version__",
    "apply_fit",
    "auto_tune",
    "correct_images",
    "curvature_depth_profile",
    "evaluate_correction",
    "evaluate_correction_volume",
    "fit_focal_gaussian",
    "fit_mosaic",
    "focal_profile",
    "inexact_alm_l1",
    "load_sample_image",
    "recommend_bounds",
    "seam_curvature",
    "seam_curvature_per_z",
    "shrink",
    "tune",
]
