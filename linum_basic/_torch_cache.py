"""Persistent ``torch.compile`` / Inductor cache configuration.

Call :func:`configure_torch_inductor_cache` **before** importing ``torch`` so
that Triton kernel artifacts survive across Nextflow tasks and ``-resume``
reruns.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "_cuda_joblib_worker_init",
    "collect_cuda_worker_env",
    "configure_torch_inductor_cache",
    "enable_fx_graph_cache",
    "warm_policy_passes",
]


def _cuda_joblib_worker_init(
    cache_dir: str | None,
    env_items: tuple[tuple[str, str], ...],
) -> None:
    """Configure shared Inductor cache and fast-path env in a loky worker.

    Must run **before** the first ``torch`` import in the worker process.

    Notes
    -----
    Fresh loky workers otherwise pay minutes of Inductor compile time per
    z-plane while GPUs sit idle.  Default to eager GPU in workers unless the
    parent process already opted into a non-default compile mode.
    """
    for key, value in env_items:
        os.environ[key] = value
    compile_mode = os.environ.get("LINUM_BASIC_ALM_COMPILE_MODE", "default").strip().lower()
    if compile_mode in ("", "default"):
        os.environ["LINUM_BASIC_ALM_COMPILE_MODE"] = "off"
    configure_torch_inductor_cache(cache_dir)


def collect_cuda_worker_env() -> tuple[tuple[str, str], ...]:
    """Snapshot fast-path env keys from the parent for loky worker initargs.

    Returns
    -------
    tuple[tuple[str, str], ...]
        Sorted ``(key, value)`` pairs for ``LINUM_BASIC_*`` and Inductor cache env vars.
    """
    keys = sorted(
        key
        for key in os.environ
        if key == "LINUM_BASIC_DCT_KERNEL" or key.startswith("LINUM_BASIC_") or key == "TORCHINDUCTOR_FX_GRAPH_CACHE"
    )
    return tuple((key, os.environ[key]) for key in keys)


def configure_torch_inductor_cache(cache_dir: str | Path | None = None) -> Path:
    """Set environment variables for a persistent Inductor on-disk cache.

    Parameters
    ----------
    cache_dir : str or Path or None
        Directory for compiled kernels.  When ``None``, uses
        ``TORCHINDUCTOR_CACHE_DIR`` if already set, otherwise
        ``~/.cache/linum-basic/inductor``.

    Returns
    -------
    Path
        Resolved cache directory (created if missing).
    """
    if cache_dir is None:
        cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "linum-basic" / "inductor"
    path = Path(cache_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(path))
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    return path


def warm_policy_passes() -> int:
    """Return extra untimed warm fits before measured benchmark repeats.

    Reads ``LINUM_BASIC_INDUCTOR_WARM_PASSES`` once per call. Unset or invalid
    values default to ``0`` (current production behavior).

    Returns
    -------
    int
        Number of extra warm fits to run before timed repeats.
    """
    raw = os.environ.get("LINUM_BASIC_INDUCTOR_WARM_PASSES", "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        return 0
    return max(0, value)


def enable_fx_graph_cache() -> None:
    """Enable the FX graph cache after ``torch`` has been imported."""
    try:
        import torch._inductor.config as inductor_config
    except ImportError:
        return
    inductor_config.fx_graph_cache = True
