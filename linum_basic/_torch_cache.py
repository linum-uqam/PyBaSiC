"""Persistent ``torch.compile`` / Inductor cache configuration.

Call :func:`configure_torch_inductor_cache` **before** importing ``torch`` so
that Triton kernel artifacts survive across Nextflow tasks and ``-resume``
reruns.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["configure_torch_inductor_cache", "enable_fx_graph_cache"]


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


def enable_fx_graph_cache() -> None:
    """Enable the FX graph cache after ``torch`` has been imported."""
    try:
        import torch._inductor.config as inductor_config
    except ImportError:
        return
    inductor_config.fx_graph_cache = True
