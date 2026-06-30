"""Public algorithm interface for linum-basic.

Re-exports the core numerical routines from the internal ``linum_basic._alm``
module under a stable public namespace.  Import from here rather than from
``linum_basic._alm`` to ensure forwards-compatibility.

Public symbols
--------------
inexact_alm_l1
    L1-penalised matrix factorisation solver used by :class:`~linum_basic.core.BaSiC`.
shrink
    Element-wise soft-threshold (shrinkage) operator.

Examples
--------
Run the ALM solver directly on a pre-loaded image stack:

>>> import numpy as np
>>> from linum_basic.algorithms import inexact_alm_l1
>>> rng = np.random.default_rng(0)
>>> imgs = rng.standard_normal((20, 128, 128)).astype(np.float32)
>>> imgs_sorted = np.sort(imgs, axis=0)
>>> Ib, Ir, D, _ = inexact_alm_l1(imgs_sorted, l_s=0.5, l_d=0.2)

See Also
--------
linum_basic.core.BaSiC : High-level estimator that drives ``inexact_alm_l1``.
"""

from linum_basic._alm import inexact_alm_l1, inexact_alm_l1_batched, shrink

__all__ = ["inexact_alm_l1", "inexact_alm_l1_batched", "shrink"]
