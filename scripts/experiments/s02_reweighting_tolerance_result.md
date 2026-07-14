# S02 — Reweighting-tolerance candidate A/B (K01-gated)

**Task:** M005 / S02 / T02
**Subject:** sub-22 (real microscopy mosaic, z-slice 27, OME-Zarr)
**Hardware:** A6000 server `sn4622125853` (132.207.157.41), single GPU `cuda:0`, PyTorch 2.12.1+cu130, CUDA 13.0, Python 3.14.3
**Commit:** `4e924e0`
**Date:** 2026-07-14

## Lever under test

`reweighting_tolerance` loosened from the BaSiC default `1e-3` → `5e-3`, combined
with a hard `max_reweighting_iterations` cap of `6` (vs the harness default `15`).
Both keys are on the allowlist (`ALLOWED_OVERRIDE_KEYS`) and were supplied via
`scripts/experiments/s02_configs/reweighting_tolerance.json` (no production code
changed). The intent is to speed convergence by accepting a looser stopping rule
and bounding the reweighting loop.

```json
{ "reweighting_tolerance": 5e-3, "max_reweighting_iterations": 6 }
```

## Baseline (precondition)

The S02 baseline bundle had not been produced (T01 recovery left a placeholder),
so a fresh production-shaped baseline was generated on the A6000 as a required
data precondition for this candidate A/B (per `gpu-server.json` `auto_execute`).
It is reusable by the T03 sync-cadence candidate.

- **baseline_id:** `baseline-20260714T123812-4e924e0-sub-22`
- z_indices `[0, 13, 27, 40, 54]` (z-sample 5), array_shape `[55, 2325, 1200]`, tile_shape `[75, 75]`
- strategy: `baseline` / sequential scalar, `cuda:0`, ws=128, darkfield on, `max_reweighting_iterations=15`, `reweighting_tolerance=1e-3`
- aggregates: `seam_l1=0.571734`, `seam_curvature=0.034412`; `release_gate=True`
- artifacts: `scripts/experiments/s02_artifacts/baseline/{baseline-bundle.json, tolerance-sidecar.json, summary.md}`

## Candidate run

- **candidate_id:** `candidate-20260714T124042-4e924e0-sub-22`
- Same strategy (`baseline`/sequential/cuda:0/ws=128/darkfield) so the A/B
  isolates the reweighting lever; only `reweighting_tolerance` and
  `max_reweighting_iterations` differ.
- aggregates: `seam_l1=0.576561`, `seam_curvature=0.036232`

## Verdict: REJECT  ✘ (fail)

| metric | abs_delta | rel_delta | gate |
| --- | --- | --- | --- |
| seam_l1 | 0.004828 | 0.8444% | ✘ fail |
| seam_curvature | 0.001820 | 5.2903% | ✘ fail |
| speed | 1.9264x | faster | ✓ pass (≥1.30x) |
| **overall** | | | **reject / fail** |
| worst_z | 27 | | |

### Speed — passes the 1.30x gate

`baseline_ms=14621.995` → `candidate_ms=7590.293`, **ratio 1.9264x** (cold/warm
repeats all ≈7.5–7.6s). This clears the D-18/MEM007 1.30x speed threshold by a
wide margin. The lever is genuinely fast.

### Quality — fails the K01 gate

The K01 release gate compares aggregate and per-z seam deltas against tolerances
calibrated from the baseline (`mean+3std`, `sigma=3`). **Both metrics fail.**

Per-z failures (largest contributors):

| z | metric | abs_delta | rel_delta |
| --- | --- | --- | --- |
| 13 | seam_curvature | 0.024038 | **56.56%** |
| 27 | seam_l1 | 0.076855 | **11.89%** |
| 0  | seam_l1 | 0.000208 | 0.064% |
| 27 | seam_curvature | 0.000080 | 0.18% |
| 40 | seam_curvature | 0.000127 | 0.55% |
| 54 | seam_curvature | 0.000041 | 0.17% |

worst_z = 27. The damage is concentrated at z=13 (curvature) and z=27 (seam_l1).

### Root cause: iteration cap, not convergence

Convergence telemetry shows **every plane hit the `max_reweighting_iterations=6`
ceiling** — `reweight_iterations_per_z = {0:6, 13:6, 27:6, 40:6, 54:6}`, median
6.0. The looser `5e-3` tolerance was never reached within 6 iterations on any
plane, so the solver was *truncated* rather than *converged*. The 1.93x speed
win is therefore the cost of under-fitting: fewer reweighting passes leave the
flat/dark estimates rougher, which surfaces as elevated seam error at the
mid-stack planes.

### Gate-threshold caveat (for S03)

The baseline was calibrated with `repeats=3`; inter-repeat variance on the
identical-strategy repeats was ~0, so the `mean+3std` tolerances collapsed to the
`1e-6` floor (`seam_l1` rel_tol ≈1.7e-6, `seam_curvature` rel_tol ≈2.9e-5). This
makes the gate extremely sensitive to any real algorithmic change. However, the
z=13 curvature (+56.6%) and z=27 seam_l1 (+11.9%) deltas are far above any
plausible noise band, so the REJECT is robust — not an artifact of the tight
floor.

## Conclusion for S03

The reweighting-tolerance lever **cannot be promoted as specified** (5e-3 +
6-iter cap): it delivers a large, real speed win (≈1.9x) but at the cost of
material seam-quality regression because the hard iteration cap truncates the
solver before it converges. The quality failure is not noise (worst-plane deltas
are 12–57%).

**Reassessment signal:** This is strong enough to warrant down-ranking the
"loosen tolerance + cap iterations" formulation in S01's shortlist. If S03 wants
to rescue the lever, the evidence points to a specific variant to test in a
follow-on execution milestone: **loosen `reweighting_tolerance` alone (drop the
`max_reweighting_iterations` cap, or raise it)** — i.e. let the looser stopping
rule end the loop early only when it is actually met, rather than force-truncating
at 6. That preserves the speed mechanism (early stop on tolerance) without the
under-fit. This variant was not part of T02's contract and is flagged for S03,
not executed here.

## Artifacts

- `scripts/experiments/s02_configs/reweighting_tolerance.json` — lever override config
- `scripts/experiments/s02_artifacts/reweighting_tolerance/candidate-artifact.json` — full candidate artifact (verdict, deltas, convergence, telemetry)
- `scripts/experiments/s02_artifacts/reweighting_tolerance/summary.md` — harness verdict table
- `scripts/experiments/s02_artifacts/baseline/{baseline-bundle.json, tolerance-sidecar.json, summary.md}` — baseline (generated as precondition; reusable by T03)

Server-side originals (A6000): `runs/s02/baseline-20260714T123812-4e924e0-sub-22/`
and `runs/s02/candidate-20260714T124042-4e924e0-sub-22/`.
