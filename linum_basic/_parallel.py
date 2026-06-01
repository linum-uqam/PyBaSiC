"""Shared parallelism helpers for the BaSiC pipelines.

The heavy compute in this library — the per-z-level ALM solves in
:func:`linum_basic.fit.fit_mosaic` — is **GIL-bound** on the NumPy/SciPy
backend (SciPy FFT and NumPy matrix ops hold the GIL).  Threads therefore
give no speed-up; the work must be spread across **processes**.  These
units (one z-level each) are fully independent and coarse-grained, which
makes them ideal for process-based parallelism with low overhead.

Two important caveats are handled here:

* **BLAS oversubscription.**  Each worker process runs its own BLAS/FFT
  threads.  With ``cpu_count() - 2`` worker processes each spawning a full
  BLAS thread pool the machine thrashes.  :func:`parallel_map` pins inner
  threads to 1 per worker via :func:`joblib.parallel_config`.
* **Single-GPU contention.**  On the PyTorch CUDA/MPS backend a single
  device is the bottleneck; fanning out processes would serialise on the
  device and multiply VRAM use.  :func:`resolve_workers` collapses the
  worker count to 1 for accelerator backends.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Sequence

__all__ = ["default_workers", "is_gpu_backend", "parallel_map", "resolve_workers"]


def default_workers() -> int:
    """Return the default worker count: ``cpu_count() - 2`` (at least 1).

    Leaving two cores free keeps the machine responsive and avoids
    starving the OS / I/O threads during long fits.

    Returns
    -------
    int
        ``max(1, os.cpu_count() - 2)``.
    """
    return max(1, (os.cpu_count() or 4) - 2)


def is_gpu_backend(backend: str | None, device: str | None) -> bool:
    """Return whether *backend* / *device* target a single accelerator.

    Process-based fan-out is counter-productive on a single GPU, so callers
    use this to collapse the worker count to 1.

    Parameters
    ----------
    backend : str or None
        BaSiC backend string (``"numpy"``, ``"torch"`` or ``"auto"``).
    device : str or None
        PyTorch device string (e.g. ``"cuda"``, ``"cuda:0"``, ``"mps"``).

    Returns
    -------
    bool
        ``True`` when the backend may resolve to a CUDA/MPS accelerator.
    """
    if backend not in {"torch", "auto"}:
        return False
    return (device or "").lower().startswith(("cuda", "mps"))


def resolve_workers(n_workers: int | None, backend: str | None = None, device: str | None = None) -> int:
    """Resolve a concrete worker count, applying defaults and the GPU guard.

    Parameters
    ----------
    n_workers : int or None
        Requested worker count.  ``None`` uses :func:`default_workers`.
    backend : str or None
        BaSiC backend string; used to detect accelerator targets.
    device : str or None
        PyTorch device string; used to detect accelerator targets.

    Returns
    -------
    int
        The effective worker count (always ``>= 1``).  Forced to 1 for
        accelerator backends.
    """
    requested = default_workers() if n_workers is None else max(1, int(n_workers))
    if is_gpu_backend(backend, device):
        if requested > 1:
            warnings.warn(
                f"Process-based parallelism is disabled on the '{backend}' backend with device "
                f"'{device}' (single-accelerator contention); running z-levels sequentially.",
                stacklevel=2,
            )
        return 1
    return requested


def parallel_map[T, R](
    fn: Callable[[T], R],
    items: Sequence[T],
    n_workers: int,
    *,
    desc: str | None = None,
    verbose: bool = False,
) -> list[R]:
    """Map *fn* over *items*, in parallel processes when ``n_workers > 1``.

    Order is preserved: ``result[i] == fn(items[i])``.  When
    ``n_workers <= 1`` the work runs sequentially in the current process
    (no joblib overhead and no extra processes), which also preserves any
    caller-side early-stopping logic.

    Parameters
    ----------
    fn : callable
        A **picklable**, module-level function applied to each item.
    items : sequence
        Items to process.
    n_workers : int
        Number of worker processes.  ``<= 1`` runs sequentially.
    desc : str or None
        Progress-bar label (used only when *verbose*).
    verbose : bool
        Show a :mod:`tqdm` progress bar.

    Returns
    -------
    list
        ``[fn(x) for x in items]`` in input order.
    """
    items = list(items)

    if n_workers <= 1:
        iterator = items
        if verbose:
            from tqdm.auto import tqdm

            iterator = tqdm(items, desc=desc, total=len(items), leave=False)
        return [fn(x) for x in iterator]

    from joblib import Parallel, delayed, parallel_config

    # Pin inner BLAS/FFT threads to 1 per worker to avoid oversubscription.
    with parallel_config(backend="loky", inner_max_num_threads=1, n_jobs=n_workers):
        results: list[R] = Parallel(verbose=10 if verbose else 0)(delayed(fn)(x) for x in items)
    return results
