"""Versioned JSON artifacts for baseline and candidate benchmark runs."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1"

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

__all__ = [
    "SCHEMA_VERSION",
    "BaselineBundle",
    "CandidateArtifact",
    "ToleranceSidecar",
    "assert_comparable",
    "make_baseline_id",
    "read_artifact",
    "slugify_label",
    "write_artifact",
    "write_summary_table",
]


@dataclass(frozen=True, slots=True, init=False)
class BaselineBundle:
    """Persisted production baseline metrics and reproducibility metadata.

    Attributes
    ----------
    baseline_id : str
        Human-readable baseline identifier.
    uuid : str
        Stable UUID distinct from ``baseline_id``.
    schema_version : str
        Artifact schema version tag.
    metric_definition_version : str
        Metric computation version for compatibility checks.
    subject_id : str
        Human-readable subject identifier.
    run_label : str
        Operator-provided run label.
    input_fingerprint : str
        Hash or fingerprint of the input volume.
    z_indices : list of int
        Selected z-planes included in the run.
    array_shape : list of int
        Full volume shape ``(n_z, height, width)``.
    tile_shape : list of int
        Mosaic tile shape ``(tile_h, tile_w)``.
    strategy_params : dict
        Resolved fit strategy parameters.
    metrics_rows : list of dict
        Per-z metric rows (no full corrected volumes).
    metrics_aggregates : dict[str, float]
        Volume-level aggregate metrics.
    repeats : int
        Number of measured repeats (excluding warmup).
    metadata : dict
        Git commit, host, and other run metadata.
    timestamp : str
        ISO-8601 run timestamp.
    """

    baseline_id: str
    uuid: str
    schema_version: str
    metric_definition_version: str
    subject_id: str
    run_label: str
    input_fingerprint: str
    z_indices: list[int]
    array_shape: list[int]
    tile_shape: list[int]
    strategy_params: dict[str, Any]
    metrics_rows: list[dict[str, Any]]
    metrics_aggregates: dict[str, float]
    repeats: int
    metadata: dict[str, Any]
    timestamp: str

    def __init__(
        self,
        baseline_id: str,
        uuid: str,
        schema_version: str,
        metric_definition_version: str,
        subject_id: str,
        run_label: str,
        input_fingerprint: str,
        z_indices: list[int],
        array_shape: list[int],
        tile_shape: list[int],
        strategy_params: dict[str, Any],
        metrics_rows: list[dict[str, Any]],
        metrics_aggregates: dict[str, float],
        repeats: int,
        metadata: dict[str, Any],
        timestamp: str,
    ) -> None:
        """Initialise a baseline artifact bundle."""
        object.__setattr__(self, "baseline_id", baseline_id)
        object.__setattr__(self, "uuid", uuid)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "metric_definition_version", metric_definition_version)
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "run_label", run_label)
        object.__setattr__(self, "input_fingerprint", input_fingerprint)
        object.__setattr__(self, "z_indices", list(z_indices))
        object.__setattr__(self, "array_shape", list(array_shape))
        object.__setattr__(self, "tile_shape", list(tile_shape))
        object.__setattr__(self, "strategy_params", dict(strategy_params))
        object.__setattr__(self, "metrics_rows", [dict(row) for row in metrics_rows])
        object.__setattr__(self, "metrics_aggregates", dict(metrics_aggregates))
        object.__setattr__(self, "repeats", repeats)
        object.__setattr__(self, "metadata", dict(metadata))
        object.__setattr__(self, "timestamp", timestamp)


@dataclass(frozen=True, slots=True, init=False)
class CandidateArtifact:
    """Persisted candidate run referencing a saved baseline bundle.

    Attributes
    ----------
    candidate_id : str
        Human-readable candidate run identifier.
    baseline_id : str
        Referenced baseline bundle id.
    schema_version : str
        Artifact schema version tag.
    metric_definition_version : str
        Metric computation version for compatibility checks.
    subject_id : str
        Human-readable subject identifier.
    run_label : str
        Operator-provided run label.
    input_fingerprint : str
        Hash or fingerprint of the input volume.
    z_indices : list of int
        Selected z-planes included in the run.
    array_shape : list of int
        Full volume shape ``(n_z, height, width)``.
    tile_shape : list of int
        Mosaic tile shape ``(tile_h, tile_w)``.
    strategy_params : dict
        Resolved fit strategy parameters.
    metrics_rows : list of dict
        Per-z metric rows (no full corrected volumes).
    metrics_aggregates : dict[str, float]
        Volume-level aggregate metrics.
    repeats : int
        Number of measured repeats (excluding warmup).
    metadata : dict
        Git commit, host, and other run metadata.
    environment : dict
        CUDA/PyTorch environment metadata for soft mismatch warnings.
    timestamp : str
        ISO-8601 run timestamp.
    """

    candidate_id: str
    baseline_id: str
    schema_version: str
    metric_definition_version: str
    subject_id: str
    run_label: str
    input_fingerprint: str
    z_indices: list[int]
    array_shape: list[int]
    tile_shape: list[int]
    strategy_params: dict[str, Any]
    metrics_rows: list[dict[str, Any]]
    metrics_aggregates: dict[str, float]
    repeats: int
    metadata: dict[str, Any]
    environment: dict[str, Any]
    timestamp: str

    def __init__(
        self,
        candidate_id: str,
        baseline_id: str,
        schema_version: str,
        metric_definition_version: str,
        subject_id: str,
        run_label: str,
        input_fingerprint: str,
        z_indices: list[int],
        array_shape: list[int],
        tile_shape: list[int],
        strategy_params: dict[str, Any],
        metrics_rows: list[dict[str, Any]],
        metrics_aggregates: dict[str, float],
        repeats: int,
        metadata: dict[str, Any],
        environment: dict[str, Any],
        timestamp: str,
    ) -> None:
        """Initialise a candidate artifact."""
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "baseline_id", baseline_id)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "metric_definition_version", metric_definition_version)
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "run_label", run_label)
        object.__setattr__(self, "input_fingerprint", input_fingerprint)
        object.__setattr__(self, "z_indices", list(z_indices))
        object.__setattr__(self, "array_shape", list(array_shape))
        object.__setattr__(self, "tile_shape", list(tile_shape))
        object.__setattr__(self, "strategy_params", dict(strategy_params))
        object.__setattr__(self, "metrics_rows", [dict(row) for row in metrics_rows])
        object.__setattr__(self, "metrics_aggregates", dict(metrics_aggregates))
        object.__setattr__(self, "repeats", repeats)
        object.__setattr__(self, "metadata", dict(metadata))
        object.__setattr__(self, "environment", dict(environment))
        object.__setattr__(self, "timestamp", timestamp)


@dataclass(frozen=True, slots=True, init=False)
class ToleranceSidecar:
    """Calibrated quality tolerances stored beside a baseline bundle.

    Attributes
    ----------
    baseline_id : str
        Referenced baseline bundle id.
    schema_version : str
        Artifact schema version tag.
    metric_definition_version : str
        Metric computation version for compatibility checks.
    calibration_policy : str
        Named calibration policy (e.g. ``mean+3std``).
    sigma : float
        Standard-deviation multiplier used during calibration.
    min_abs : float
        Minimum absolute tolerance floor.
    tolerances : dict
        Per-metric ``mean``, ``std``, ``abs_tol``, and ``rel_tol`` values.
    """

    baseline_id: str
    schema_version: str
    metric_definition_version: str
    calibration_policy: str
    sigma: float
    min_abs: float
    tolerances: dict[str, dict[str, float]]

    def __init__(
        self,
        baseline_id: str,
        schema_version: str,
        metric_definition_version: str,
        calibration_policy: str,
        sigma: float,
        min_abs: float,
        tolerances: dict[str, dict[str, float]],
    ) -> None:
        """Initialise a tolerance sidecar."""
        object.__setattr__(self, "baseline_id", baseline_id)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "metric_definition_version", metric_definition_version)
        object.__setattr__(self, "calibration_policy", calibration_policy)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "min_abs", min_abs)
        object.__setattr__(self, "tolerances", {k: dict(v) for k, v in tolerances.items()})


def slugify_label(label: str) -> str:
    """Sanitize a label for safe use in filenames and baseline ids.

    Parameters
    ----------
    label : str
        Raw operator-provided label or subject slug source.

    Returns
    -------
    str
        Filename-safe string using ``[A-Za-z0-9._-]`` only.
    """
    cleaned = label.lstrip("./\\")
    while cleaned.startswith(".."):
        cleaned = cleaned[2:].lstrip("./\\")
    sanitized = _SAFE_CHARS.sub("-", cleaned)
    result = sanitized.strip("-")
    return result or "unknown"


def make_baseline_id(*, commit: str, subject_id: str, timestamp: str) -> str:
    """Build a human-readable baseline identifier.

    Parameters
    ----------
    commit : str
        Full git commit hash.
    subject_id : str
        Subject identifier slugged for the id suffix.
    timestamp : str
        Compact timestamp string embedded in the id.

    Returns
    -------
    str
        ``baseline-{timestamp}-{short_commit}-{subject_slug}``.
    """
    short_commit = commit[:7]
    subject_slug = slugify_label(subject_id)
    return f"baseline-{timestamp}-{short_commit}-{subject_slug}"


def write_artifact(path: Path | str, payload: Any, *, overwrite: bool = False) -> None:
    """Write a dataclass payload to JSON with stable key ordering.

    Parameters
    ----------
    path : Path or str
        Destination JSON file path.
    payload : dataclass
        Artifact dataclass instance to serialize.
    overwrite : bool
        When ``False`` (default), raise if *path* already exists.

    Raises
    ------
    FileExistsError
        When the target exists and *overwrite* is ``False``.
    """
    out = Path(path)
    if out.exists() and not overwrite:
        msg = f"Output path already exists: {path}. Set overwrite=True to overwrite."
        raise FileExistsError(msg)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(asdict(payload), indent=2, sort_keys=True) + "\n"
    out.write_text(text, encoding="utf-8")


def read_artifact(path: Path | str, cls: type[Any]) -> Any:
    """Load a JSON artifact and reconstruct a dataclass instance.

    Parameters
    ----------
    path : Path or str
        Source JSON file path.
    cls : type
        Target dataclass type (``BaselineBundle``, ``CandidateArtifact``, etc.).

    Returns
    -------
    dataclass
        Reconstructed artifact instance.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return cls(**data)


_HARD_COMPARE_KEYS = (
    "subject_id",
    "input_fingerprint",
    "z_indices",
    "array_shape",
    "tile_shape",
    "metric_definition_version",
)


def _identity_dict(artifact: BaselineBundle | CandidateArtifact) -> dict[str, Any]:
    return {
        "subject_id": artifact.subject_id,
        "input_fingerprint": artifact.input_fingerprint,
        "z_indices": artifact.z_indices,
        "array_shape": artifact.array_shape,
        "tile_shape": artifact.tile_shape,
        "metric_definition_version": artifact.metric_definition_version,
    }


def assert_comparable(
    candidate: CandidateArtifact,
    baseline: BaselineBundle,
) -> list[str]:
    """Verify a candidate can be compared to a baseline bundle.

    Hard mismatches on subject, input fingerprint, z selection, shapes, or metric
    definition raise :class:`ValueError`. Soft mismatches on git commit or environment
    return warning strings without raising.

    Parameters
    ----------
    candidate : CandidateArtifact
        Candidate run artifact.
    baseline : BaselineBundle
        Saved baseline bundle.

    Returns
    -------
    list of str
        Soft-warning messages (may be empty).

    Raises
    ------
    ValueError
        When any hard compatibility key differs.
    """
    cand = _identity_dict(candidate)
    base = _identity_dict(baseline)
    mismatches = [key for key in _HARD_COMPARE_KEYS if cand[key] != base[key]]
    if mismatches:
        joined = ", ".join(mismatches)
        msg = f"Candidate cannot be compared to baseline; mismatched: {joined}"
        raise ValueError(msg)

    warnings: list[str] = []
    cand_commit = candidate.metadata.get("git_commit")
    base_commit = baseline.metadata.get("git_commit")
    if cand_commit and base_commit and cand_commit != base_commit:
        warnings.append(f"git commit differs: candidate={cand_commit}, baseline={base_commit}")

    cand_env = candidate.environment
    base_env = baseline.metadata.get("environment", {})
    if isinstance(base_env, dict):
        warnings.extend(
            f"environment.{key} differs: candidate={cand_env.get(key)!r}, baseline={base_env.get(key)!r}"
            for key in sorted(set(cand_env) | set(base_env))
            if cand_env.get(key) != base_env.get(key)
        )

    return warnings


def write_summary_table(path: Path | str, rows: list[dict[str, Any]], *, fmt: str = "markdown") -> None:
    """Write a human-readable summary table beside machine-readable JSON.

    Parameters
    ----------
    path : Path or str
        Destination file path.
    rows : list of dict
        Table rows with uniform keys.
    fmt : str
        ``"markdown"`` for a pipe table or ``"csv"`` for comma-separated values.

    Raises
    ------
    ValueError
        When *fmt* is not ``markdown`` or ``csv``.
    """
    if not rows:
        msg = "rows must not be empty"
        raise ValueError(msg)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        fieldnames = list(rows[0].keys())
        with out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return

    if fmt == "markdown":
        headers = list(rows[0].keys())
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(row[h]) for h in headers) + " |" for row in rows),
        ]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    msg = f"Unknown summary format: {fmt!r}; expected 'markdown' or 'csv'"
    raise ValueError(msg)
