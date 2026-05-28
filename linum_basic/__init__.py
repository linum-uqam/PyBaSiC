"""linum-basic — illumination correction for optical microscopy images.

Public API
----------
BaSiC
    Main estimator class.  See :class:`linum_basic.core.BaSiC` for full
    documentation.
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
>>> from linum_basic import BaSiC
>>> stack = np.random.rand(30, 64, 64).astype("float32")
>>> model = BaSiC(stack)
>>> model.prepare()
>>> model.run()
>>> flatfield = model.get_flatfield()
"""

from linum_basic.algorithms import inexact_alm_l1, shrink
from linum_basic.core import BaSiC

__version__ = "0.2.0"
__all__ = ["BaSiC", "__version__", "inexact_alm_l1", "shrink"]
