"""Supplementary BaSiCPy comparison metrics for the A/B harness (informational only)."""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

SIDECAR_FILENAME = "basicpy-supplementary.json"
_BASICPY_INSTALL_HINT = "uv sync --extra basicpy"

__all__ = [
    "SIDECAR_FILENAME",
    "compute_basicpy_comparison",
    "run_basicpy_reference_fit",
    "write_basicpy_sidecar",
]


def _flatfield_pearson_r(
    linum_flatfield: np.ndarray,
    reference_flatfield: np.ndarray,
) -> float:
    """Pearson correlation between flattened flat-field arrays."""
    a = np.asarray(linum_flatfield, dtype=np.float64).ravel()
    b = np.asarray(reference_flatfield, dtype=np.float64).ravel()
    if a.shape != b.shape:
        msg = f"flatfield shapes must match: {a.shape} vs {b.shape}"
        raise ValueError(msg)
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = float(np.sqrt(np.dot(a_centered, a_centered) * np.dot(b_centered, b_centered)))
    if denom == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a_centered, b_centered) / denom)


def compute_basicpy_comparison(
    linum_flatfield: np.ndarray,
    reference_flatfield: np.ndarray,
    *,
    linum_ms: float,
    reference_ms: float,
    gate_applicable: bool = False,
) -> dict[str, Any]:
    """Compare linum-basic and BaSiCPy flat-fields as supplementary metrics.

    Parameters
    ----------
    linum_flatfield : numpy.ndarray
        Flat-field estimate from linum-basic.
    reference_flatfield : numpy.ndarray
        Flat-field estimate from BaSiCPy (``basicpy``).
    linum_ms : float
        Wall time for the linum-basic fit in milliseconds.
    reference_ms : float
        Wall time for the BaSiCPy reference fit in milliseconds.
    gate_applicable : bool
        When ``False`` (default), metrics are informational only and never
        block harness promotion unless an operator elevates them after review.

    Returns
    -------
    dict
        Sidecar payload with ``flatfield_pearson_r``, ``wall_time_ratio``
        (reference_ms / linum_ms), ``supplementary=True``, and
        ``gate_applicable``.
    """
    if linum_ms <= 0.0:
        msg = "linum_ms must be positive"
        raise ValueError(msg)
    return {
        "flatfield_pearson_r": _flatfield_pearson_r(linum_flatfield, reference_flatfield),
        "wall_time_ratio": reference_ms / linum_ms,
        "supplementary": True,
        "gate_applicable": gate_applicable,
    }


def run_basicpy_reference_fit(
    stack: np.ndarray,
    *,
    working_size: int = 128,
    estimate_darkfield: bool = False,
) -> dict[str, Any]:
    """Run a thin BaSiCPy reference fit on a tile stack.

    Parameters
    ----------
    stack : numpy.ndarray
        Tile stack with shape ``(n_tiles, height, width)``.
    working_size : int
        BaSiCPy working resolution (default 128, matching linum-basic production).
    estimate_darkfield : bool
        When ``True``, enable BaSiCPy dark-field estimation.

    Returns
    -------
    dict
        ``flatfield`` array and ``elapsed_ms`` wall time for the reference fit.

    Raises
    ------
    ImportError
        When the optional ``basicpy`` package is not installed.
    """
    try:
        basicpy = importlib.import_module("basicpy")
        basic_class = basicpy.BaSiC
    except ImportError as exc:
        msg = f"BaSiCPy (basicpy) is not installed. Install the supplementary optional extra: {_BASICPY_INSTALL_HINT}"
        raise ImportError(msg) from exc

    images = np.asarray(stack, dtype=np.float64)
    if images.ndim != 3:
        msg = f"stack must be 3-D (n_tiles, height, width); got shape {images.shape!r}"
        raise ValueError(msg)

    model = basic_class(working_size=working_size, get_darkfield=estimate_darkfield)
    started = time.perf_counter()
    model.fit(images)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    flatfield = np.asarray(model.flatfield, dtype=np.float64)
    return {"flatfield": flatfield, "elapsed_ms": elapsed_ms}


def write_basicpy_sidecar(output_dir: Path | str, comparison: dict[str, Any]) -> Path:
    """Write supplementary BaSiCPy comparison metrics beside harness artifacts.

    Parameters
    ----------
    output_dir : Path or str
        Candidate or baseline bundle directory.
    comparison : dict
        Payload from :func:`compute_basicpy_comparison`.

    Returns
    -------
    Path
        Path to ``basicpy-supplementary.json``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SIDECAR_FILENAME
    text = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return path
