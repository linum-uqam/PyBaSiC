"""Strategy resolution for the A/B benchmark harness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BUILTIN_STRATEGIES: tuple[str, ...] = ("baseline", "sequential", "multi", "batched")

ALLOWED_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "working_size",
        "estimate_darkfield",
        "max_reweighting_iterations",
        "l_s",
        "l_d",
        "epsilon",
        "batched_z_chunk_size",
        "warm_start_reweighting",
        "device",
        "backend",
        "force_batched_cuda",
        "reweighting_tolerance",
        "convergence_check_every",
        "tile_subsample_ratio",
    }
)

__all__ = [
    "ALLOWED_OVERRIDE_KEYS",
    "BUILTIN_STRATEGIES",
    "StrategyResult",
    "load_overrides",
    "resolve_strategy",
]


@dataclass(frozen=True, slots=True, init=False)
class StrategyResult:
    """Resolved fit strategy for a benchmark run.

    Attributes
    ----------
    name : str
        Strategy name (one of ``BUILTIN_STRATEGIES``).
    basic_kwargs : dict
        Keyword arguments passed to :func:`linum_basic.fit.fit_mosaic`.
    batched_z_chunk_size : int or None
        Optional z chunk size for batched CUDA mode.
    force_batched : bool
        Whether to force the batched CUDA path.
    release_gate : bool
        ``True`` for real-subject runs eligible for release decisions.
    label : str
        Human-readable strategy label.
    """

    name: str
    basic_kwargs: dict[str, Any]
    batched_z_chunk_size: int | None
    force_batched: bool
    release_gate: bool
    label: str

    def __init__(
        self,
        name: str,
        basic_kwargs: dict[str, Any],
        batched_z_chunk_size: int | None,
        force_batched: bool,
        release_gate: bool,
        label: str,
    ) -> None:
        """Initialise a resolved strategy result.

        Parameters
        ----------
        name : str
            Strategy name (one of ``BUILTIN_STRATEGIES``).
        basic_kwargs : dict
            Keyword arguments passed to :func:`linum_basic.fit.fit_mosaic`.
        batched_z_chunk_size : int or None
            Optional z chunk size for batched CUDA mode.
        force_batched : bool
            Whether to force the batched CUDA path.
        release_gate : bool
            ``True`` for real-subject runs eligible for release decisions.
        label : str
            Human-readable strategy label.
        """
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "basic_kwargs", basic_kwargs)
        object.__setattr__(self, "batched_z_chunk_size", batched_z_chunk_size)
        object.__setattr__(self, "force_batched", force_batched)
        object.__setattr__(self, "release_gate", release_gate)
        object.__setattr__(self, "label", label)


def resolve_strategy(
    name: str,
    *,
    working_size: int,
    estimate_darkfield: bool,
    max_reweighting_iterations: int,
    batched_z_chunk_size: int | None = None,
    overrides: dict[str, Any] | None = None,
    is_synthetic: bool = False,
) -> StrategyResult:
    """Resolve a built-in strategy name to validated fit kwargs.

    Parameters
    ----------
    name : str
        One of ``BUILTIN_STRATEGIES``.
    working_size : int
        BaSiC working resolution.
    estimate_darkfield : bool
        Whether to estimate a dark-field.
    max_reweighting_iterations : int
        ALM reweighting iteration cap.
    batched_z_chunk_size : int or None, optional
        Optional z chunk size for batched mode.
    overrides : dict or None, optional
        Allowlisted override keys merged onto base kwargs.
    is_synthetic : bool, optional
        When ``True``, marks the run as smoke-only (``release_gate=False``).

    Returns
    -------
    StrategyResult
        Resolved strategy with fit kwargs and release-gate flag.

    Raises
    ------
    ValueError
        If *name* is not a built-in strategy.
    """
    if name not in BUILTIN_STRATEGIES:
        msg = f"Unknown strategy: {name}; choose from {BUILTIN_STRATEGIES}"
        raise ValueError(msg)

    basic_kwargs: dict[str, Any] = {
        "backend": "torch",
        "working_size": working_size,
        "estimate_darkfield": estimate_darkfield,
        "max_reweighting_iterations": max_reweighting_iterations,
    }

    force_batched = False
    chunk_size = batched_z_chunk_size

    if name in ("baseline", "sequential"):
        basic_kwargs["device"] = "cuda:0"
        basic_kwargs["warm_start_reweighting"] = False
    elif name == "multi":
        basic_kwargs["device"] = "cuda"
        basic_kwargs["warm_start_reweighting"] = True
    elif name == "batched":
        basic_kwargs["device"] = "cuda"
        basic_kwargs["warm_start_reweighting"] = True
        basic_kwargs["force_batched_cuda"] = True
        force_batched = True
        if chunk_size is not None:
            basic_kwargs["batched_z_chunk_size"] = chunk_size

    if overrides:
        basic_kwargs.update(overrides)
        if "batched_z_chunk_size" in overrides:
            chunk_size = overrides["batched_z_chunk_size"]

    release_gate = not is_synthetic
    return StrategyResult(
        name=name,
        basic_kwargs=basic_kwargs,
        batched_z_chunk_size=chunk_size,
        force_batched=force_batched,
        release_gate=release_gate,
        label=name,
    )


def load_overrides(path: str | Path) -> dict[str, Any]:
    """Load and validate JSON or YAML override file.

    Parameters
    ----------
    path : str or Path
        Path to a ``.json``, ``.yaml``, or ``.yml`` file.

    Returns
    -------
    dict
        Allowlisted override key-value pairs.

    Raises
    ------
    ValueError
        If the file format is unsupported, content is not a mapping,
        unknown keys are present, or PyYAML is missing for YAML files.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            msg = "PyYAML is required to load YAML override files"
            raise ValueError(msg) from exc
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    else:
        msg = f"Unsupported override file extension: {suffix}"
        raise ValueError(msg)

    if not isinstance(raw, dict):
        msg = "Override file must contain a mapping at the top level"
        raise ValueError(msg)

    unknown = set(raw) - ALLOWED_OVERRIDE_KEYS
    if unknown:
        msg = f"Unknown override keys: {sorted(unknown)}"
        raise ValueError(msg)

    return dict(raw)
