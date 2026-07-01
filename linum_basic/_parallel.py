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
* **Single-GPU contention.**  On the PyTorch CUDA backend a single
  device is the bottleneck; fanning out processes would serialise on the
  device and multiply VRAM use.  :func:`resolve_workers` collapses the
  worker count to 1 for accelerator backends.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Sequence

from linum_basic._torch_cache import _cuda_joblib_worker_init, collect_cuda_worker_env

__all__ = [
    "default_workers",
    "is_gpu_backend",
    "list_cuda_devices",
    "parallel_map",
    "parallel_map_cuda_devices",
    "resolve_workers",
]


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
        PyTorch device string (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``).

    Returns
    -------
    bool
        ``True`` when the backend may resolve to a CUDA accelerator.
    """
    if backend not in {"torch", "auto"}:
        return False
    # Explicit device string wins
    if (device or "").lower().startswith("cuda"):
        return True
    # "auto" with no explicit device: inspect the runtime to decide
    if backend == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return True
        except ImportError:
            pass
    return False


def list_cuda_devices(device: str | None) -> list[str]:
    """Return CUDA device strings available for z-level parallelism.

    When *device* is ``"cuda"`` (or ``None`` on a CUDA host), all visible
    GPUs are returned.  A specific index such as ``"cuda:0"`` yields a
    single-device list.

    Parameters
    ----------
    device : str or None
        PyTorch device string from BaSiC kwargs.

    Returns
    -------
    list of str
        Device strings, e.g. ``["cuda:0", "cuda:1"]``.  Empty when CUDA is
        unavailable or *device* targets a non-CUDA backend.
    """
    dev = (device or "").lower()
    if dev.startswith("mps"):
        return []
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    dev = (device or "cuda").lower()
    if dev in {"", "cuda"}:
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if dev.startswith("cuda:"):
        return [dev]
    return []


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
        cuda_devices = list_cuda_devices(device)
        if len(cuda_devices) > 1:
            return min(requested, len(cuda_devices))
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


def parallel_map_cuda_devices[T, R](
    fn: Callable[[T, str], R],
    items: Sequence[T],
    devices: Sequence[str],
    *,
    desc: str | None = None,
    verbose: bool = False,
) -> list[R]:
    """Map *fn(item, device)* over *items*, round-robin across *devices*.

    Each item runs in a **separate process** with ``CUDA_VISIBLE_DEVICES`` set
    to the assigned GPU index.  This avoids ``torch.compile`` thread-safety
    issues and keeps VRAM isolated per device.

    Order is preserved: ``result[i]`` corresponds to ``items[i]``.

    Parameters
    ----------
    fn : callable
        ``fn(item, device) -> result``.  *device* is always ``"cuda:0"`` inside
        the worker (the only visible GPU in that process).
    items : sequence
        Work items (one per z-level tile stack).
    devices : sequence of str
        CUDA device strings, e.g. ``["cuda:0", "cuda:1"]``.
    desc : str or None
        Progress-bar label when *verbose*.
    verbose : bool
        Show a :mod:`tqdm` progress bar.

    Returns
    -------
    list
        Results in input order.
    """
    import os

    items = list(items)
    devs = list(devices)
    if not items:
        return []
    if len(devs) == 1:
        dev = devs[0]
        iterator = items
        if verbose:
            from tqdm.auto import tqdm

            iterator = tqdm(items, desc=desc, total=len(items), leave=False)
        return [fn(item, dev) for item in iterator]

    def _task(index: int, item: T, device: str) -> tuple[int, R]:
        gpu_ix = device.rsplit(":", 1)[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ix
        return index, fn(item, "cuda:0")

    from joblib import Parallel, delayed, parallel_config

    n_jobs = min(len(devs), len(items))
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    env_items = collect_cuda_worker_env()
    with parallel_config(
        backend="loky",
        inner_max_num_threads=1,
        n_jobs=n_jobs,
        initializer=_cuda_joblib_worker_init,
        initargs=(cache_dir, env_items),
    ):
        pairs: list[tuple[int, R]] = Parallel(verbose=10 if verbose else 0)(
            delayed(_task)(i, item, devs[i % len(devs)]) for i, item in enumerate(items)
        )
    pairs.sort(key=lambda pair: pair[0])
    return [value for _, value in pairs]
