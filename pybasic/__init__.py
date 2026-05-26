"""PyBaSiC — illumination correction for optical microscopy images.

Public API
----------
BaSiC
    Main estimator class.  See :class:`pybasic.core.BaSiC` for full
    documentation.

Examples
--------
>>> import numpy as np
>>> from pybasic import BaSiC
>>> stack = np.random.rand(30, 64, 64).astype("float32")
>>> model = BaSiC(stack)
>>> model.prepare()
>>> model.run()
>>> flatfield = model.get_flatfield()
"""

from pybasic.core import BaSiC

__version__ = "0.2.0"
__all__ = ["BaSiC", "__version__"]
