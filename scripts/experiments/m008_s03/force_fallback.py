#!/usr/bin/env python3
"""Deliberately trigger a D023 auto-apply fallback and persist the ``.gate`` dict.

Operator-run probe for the M008/S03 real-subject *fallback* validation (T05,
R057). It runs :func:`linum_basic.tuning.auto_tune` on a real OME-Zarr mosaic
under an **adversarially over-regularised** search space that forces the tuned
candidate's flat-field to be too flat to correct a real vignette. The
non-regression gate must then DETECT the seam_l1 regression, REFUSE to apply
the candidate, and RETURN the safe default-bounds baseline fit — proving the
safety gate can refuse on purpose, not just pass on easy data (design contract
item 2 in docs/auto_apply_safety_gate.md).

Why an adversarial search space (not a monkeypatch)?
----------------------------------------------------
The unit tests already prove every fallback row fires by monkeypatching
``compute_deltas`` (tests/test_tuning.py, ``TestAutoTuneRegression`` /
``TestAutoTuneFallbackContract``). An *operator* cannot monkeypatch production
code, so this probe triggers the fallback through the **real** code path:

    auto_tune → tune → recommend_bounds → TWO fit_mosaic calls →
    compute_quality_report → compute_deltas → _evaluate_regression

The mechanism is backend-independent (the regression rule is a plain float
comparison). On a *vignetted* synthetic mosaic the adversarial space makes
the candidate under-correct ``seam_l1`` and the gate refuses — sensitivity
proof. NOTE: on real ``sub-22`` tissue the gate does NOT refuse, because
``seam_l1`` on that volume has a hard content floor (≈ 0.371) already below
the default-bounds baseline (≈ 0.501), so no flat-field regression is
available to detect (the 2026-07-16 real-subject run instead proves the
gate's *specificity* — it withholds a false refusal; see the Validation Log
in docs/tuning.md). The refusal is unit-tested for all six fallback reasons
in tests/test_tuning.py.

* Baseline fit uses the BaSiC defaults (``working_size=128``,
  ``estimate_darkfield=True``, auto-tuned ``l_s``/``l_d``) and corrects the
  vignette.
* The candidate is constrained to ``working_size=16`` with a TINY
  ``l_s_divisor`` (5..15) → ``l_s ≈ dct_sum/10``, an over-large regularisation
  weight that over-smooths the flat-field toward a constant. On data with a
  real (non-flat) vignette the candidate then under-corrects ``seam_l1`` and
  the gate fires ``regression-detected``, returning the baseline. (On a
  *flat* ground-truth mosaic this is a no-op — a real vignette is required.
  Equally, a real subject whose ``seam_l1`` is content-floored below the
  baseline (like ``sub-22``) cannot regress on this axis, so the real-subject
  run proves the gate's specificity rather than its sensitivity.)

Local smoke proof (no CUDA; proves the gate genuinely refuses on a vignetted
synthetic mosaic — the same ``_vignette`` field shape the test suite uses)::

    uv run python scripts/experiments/m008_s03/force_fallback.py --smoke \\
        --output /tmp/force-fallback-smoke.json --verbose

Real-subject run on the A6000 (sub-22)::

    CUDA_VISIBLE_DEVICES=0 uv run python \\
        scripts/experiments/m008_s03/force_fallback.py \\
        --input /scratch/workspace/sub-22/output/27/resample_mosaic_grid \\
        --output scripts/experiments/m008_s03/auto-apply-fallback.json \\
        --n-trials 50 --z-subsample 4 --backend torch --device cuda:0 \\
        --max-tiles 64 --verbose --label "sub-22 slice 27 fallback"

The probe adds no solver code and touches no BaSiC invariant (K02).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from linum_basic.benchmark import collect_git_commit, collect_host_info
from linum_basic.mosaic import MosaicGrid
from linum_basic.tuning import AutoApplyError, auto_tune

SCHEMA_VERSION = 1

# Adversarial over-regularisation search space (see module docstring). Forcing
# the candidate into this region makes its flat-field too flat to correct a
# real vignette, so ``seam_l1`` regresses against the well-corrected baseline
# and the gate fires ``regression-detected``. This is NOT a default the
# operator should ship; it exists solely to exercise the refusal path.
ADVERSARIAL_SEARCH_SPACE: dict[str, list | tuple] = {
    "working_size": [16],
    "l_s_divisor": (5.0, 15.0),
    "l_d_divisor": (20.0, 60.0),
    "epsilon": (0.8, 1.0),
    "estimate_darkfield": [False],
}


def _collect_software_versions() -> dict[str, Any]:
    """Collect torch / CUDA versions (best-effort) for provenance."""
    info: dict[str, Any] = {"python_version": platform.python_version()}
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
    except ImportError, RuntimeError:  # pragma: no cover - torch optional
        info["torch_version"] = "unavailable"
        info["cuda_available"] = False
        info["cuda_version"] = None
    return info


def _coerce(value: Any) -> Any:
    """Recursively coerce numpy scalars to plain Python for JSON."""
    if isinstance(value, dict):
        return {k: _coerce(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    try:
        return float(value)
    except TypeError, ValueError:
        return str(value)


def _gate_to_jsonable(gate: dict[str, Any]) -> dict[str, Any]:
    """Make the ``.gate`` dict JSON-serialisable (numpy floats → plain floats)."""
    return _coerce(gate)


def _build_artifact(
    gate: dict[str, Any],
    elapsed: float,
    backend: str,
    device: str | None,
    *,
    smoke: bool,
    provenance_extra: dict[str, Any] | None = None,
    error: bool = False,
) -> dict[str, Any]:
    """Assemble the full JSON artifact: provenance + gate + timing + outcome."""
    host = collect_host_info()
    verdict = gate.get("gate_verdict", "unknown")
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "probe": "force_fallback",
        "intent": ("deliberately trigger a regression-detected fallback via an adversarially over-regularised search space"),
        "outcome": _classify_outcome(verdict, error),
        "smoke": bool(smoke),
        "provenance": {
            "git_commit": collect_git_commit(),
            "host_node": host["host_node"],
            "platform": host["platform"],
            "software": _collect_software_versions(),
            "backend": backend,
            "device": device,
            "search_space": "adversarial-over-regularisation (NOT a ship default)",
            **(provenance_extra or {}),
        },
        "gate": gate,
        "wall_clock_seconds": round(elapsed, 3),
    }
    if error:
        artifact["outcome"] = "auto-apply-error"
    return artifact


def _classify_outcome(verdict: str, error: bool) -> str:
    """Classify the probe outcome for the operator (gate-refused vs. unexpected)."""
    if error:
        return "auto-apply-error"
    if verdict == "fail":
        return "gate-refused-regression-detected"
    if verdict == "fallback":
        return "gate-refused-upstream-fallback"
    if verdict == "pass":
        # On the vignetted *smoke* mosaic a pass is unexpected (the adversarial
        # space is designed to regress seam_l1 there). On *real* tissue a pass
        # is correct specificity: if the subject's seam_l1 is content-floored
        # below the baseline (as on sub-22), no regression exists to detect.
        return "gate-did-not-refuse (smoke=unexpected; real=correct-specificity)"
    return verdict


def _build_vignetted_mosaic() -> MosaicGrid:
    """Tiny vignetted mosaic for the ``--smoke`` plumbing proof (no CUDA).

    Unlike the pass-path smoke fixture (flat field), this carries a real
    Gaussian vignette so the adversarial over-regularisation genuinely
    under-corrects ``seam_l1`` and the gate refuses — mirroring the
    ``_vignette`` field shape in ``tests/test_tuning.py``.
    """
    import numpy as np

    yy, xx = np.mgrid[0:12, 0:12]
    cy, cx = 5.5, 5.5
    r2 = ((yy - cy) / 12.0) ** 2 + ((xx - cx) / 12.0) ** 2
    ff = np.exp(-r2 / (2 * 0.35**2)).astype(np.float32)
    flatfield = ff / float(ff.mean())

    rng = np.random.default_rng(7)
    tile_h, tile_w = 12, 12
    n_rows, n_cols, n_z = 4, 5, 4
    overlap_x = round(0.2 * tile_w)
    overlap_y = round(0.2 * tile_h)
    n_tiles = n_rows * n_cols
    raw = np.zeros((n_z, n_rows * tile_h, n_cols * tile_w), dtype=np.float32)
    for z in range(n_z):
        tiles = rng.random((n_tiles, tile_h, tile_w)).astype(np.float32) + 0.5
        for r in range(n_rows):
            for c in range(n_cols - 1):
                shared = rng.random((tile_h, overlap_x)).astype(np.float32) + 0.5
                tiles[r * n_cols + c, :, -overlap_x:] = shared
                tiles[r * n_cols + c + 1, :, :overlap_x] = shared
        for r in range(n_rows - 1):
            for c in range(n_cols):
                shared = rng.random((overlap_y, tile_w)).astype(np.float32) + 0.5
                tiles[r * n_cols + c, -overlap_y:, :] = shared
                tiles[(r + 1) * n_cols + c, :overlap_y, :] = shared
        for r in range(n_rows):
            for c in range(n_cols):
                idx = r * n_cols + c
                raw[z, r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tiles[idx] * flatfield
    return MosaicGrid(raw, tile_shape=(tile_h, tile_w), overlap_fraction=0.2)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Deliberately trigger a D023 auto-apply fallback (regression-detected) "
            "and persist the .gate dict (M008/S03 T05, R057)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", metavar="ZARR", type=Path, help="Path to the input OME-Zarr mosaic grid (required unless --smoke)."
    )
    p.add_argument(
        "--output", metavar="JSON", required=True, type=Path, help="Write the full gate dict + provenance JSON here."
    )
    p.add_argument(
        "--smoke", action="store_true", help="Run on a vignetted synthetic mosaic (NumPy) to prove the gate refuses locally."
    )
    p.add_argument("--n-trials", metavar="N", type=int, default=50, help="Optuna trials (forwarded to auto_tune).")
    p.add_argument("--z-subsample", metavar="N", type=int, default=4, help="Z-levels per trial (forwarded to auto_tune).")
    p.add_argument("--seed", metavar="N", type=int, default=0, help="Random seed.")
    p.add_argument("--backend", default="torch", help="Array backend (torch | numpy).")
    p.add_argument("--device", default=None, help="Device (e.g. cuda:0); omit for CPU/NumPy.")
    p.add_argument("--max-tiles", metavar="N", type=int, default=64, help="Tiles per trial seam metric; 0 = all.")
    p.add_argument("--n-extra-rows", metavar="N", type=int, default=0, help="Leading rows to drop per tile.")
    p.add_argument("--overlap", metavar="FRAC", type=float, default=0.2, help="Physical tile-overlap fraction.")
    p.add_argument("--verbose", action="store_true", help="Forwarded to auto_tune.")
    p.add_argument("--label", default=None, help="Free-form provenance label (e.g. 'sub-22 slice 27 fallback').")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.smoke:
        mosaic = _build_vignetted_mosaic()
        n_trials, backend, device, max_tiles = 4, "numpy", None, 8
        z_subsample = 2
    else:
        if args.input is None:
            print("error: --input ZARR is required (or pass --smoke)", file=sys.stderr)
            return 2
        mosaic = MosaicGrid.from_ome_zarr(str(args.input), overlap_fraction=args.overlap)
        n_trials = args.n_trials
        z_subsample = args.z_subsample
        backend = args.backend
        device = args.device
        max_tiles = args.max_tiles if args.max_tiles > 0 else None

    provenance_extra = {"label": args.label, "n_trials": n_trials, "z_subsample": z_subsample}

    t0 = time.perf_counter()
    try:
        result = auto_tune(
            mosaic,
            n_trials=n_trials,
            z_subsample=z_subsample,
            search_space=ADVERSARIAL_SEARCH_SPACE,
            seed=args.seed,
            backend=backend,
            device=device,
            max_tiles=max_tiles,
            n_extra_rows=args.n_extra_rows,
            verbose=args.verbose,
        )
        gate = _gate_to_jsonable(result.gate)
        error = False
    except AutoApplyError as exc:
        elapsed = time.perf_counter() - t0
        gate = {
            "gate_verdict": "error",
            "applied": None,
            "fallback_reason": "auto-apply-error",
            "error": str(exc),
        }
        artifact = _build_artifact(
            gate, elapsed, backend, device, smoke=args.smoke, provenance_extra=provenance_extra, error=True
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            json.dump(artifact, fh, indent=2)
        _print_summary(gate, artifact)
        return 0

    elapsed = time.perf_counter() - t0
    artifact = _build_artifact(
        gate, elapsed, backend, device, smoke=args.smoke, provenance_extra=provenance_extra, error=error
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(artifact, fh, indent=2)

    _print_summary(gate, artifact)
    return 0


def _print_summary(gate: dict[str, Any], artifact: dict[str, Any]) -> None:
    """Print the operator-facing one-glance summary to stdout."""
    verdict = gate.get("gate_verdict", "unknown")
    applied = gate.get("applied")
    fallback_reason = gate.get("fallback_reason")
    failing = gate.get("failing_metrics")
    print(f"gate_verdict: {verdict}")
    print(f"applied: {applied}")
    if fallback_reason is not None:
        print(f"fallback_reason: {fallback_reason}")
    if failing:
        print(f"failing_metrics: {failing}")
    deltas = gate.get("deltas") or {}
    for metric in ("seam_l1", "seam_curvature"):
        if metric in deltas:
            d = deltas[metric]
            print(f"  {metric}: abs_delta={d.get('abs_delta'):+.6f} rel_delta={d.get('rel_delta'):+.6f}")
    print(f"outcome: {artifact.get('outcome')}")
    print(f"wall_clock: {artifact['wall_clock_seconds']}s")


if __name__ == "__main__":
    raise SystemExit(main())
