# S02 — Sync-cadence candidate A/B (K01-gated)

**Task:** M005 / S02 / T03
**Subject:** sub-22 (real microscopy mosaic, z-slice 27, OME-Zarr)
**Hardware:** A6000 server `sn4622125853` (132.207.157.41), single GPU `cuda:0`, PyTorch 2.12.1+cu130, CUDA 13.0, Python 3.14.3
**Commit:** `4e924e0`
**Date:** 2026-07-14

## Lever under test

`convergence_check_every` raised from the GPU default `10` → `20`. The key is on
the allowlist (`ALLOWED_OVERRIDE_KEYS`, `linum_basic/benchmark/strategies.py:60`)
and was supplied via `scripts/experiments/s02_configs/sync_cadence.json` (no
production code changed). The S01 shortlist rationale for this lever was that the
inner ALM solver's convergence check calls `xp.norm_fro(dY)`, which forces a
GPU→CPU synchronisation (`linum_basic/_alm.py:441-500`); checking half as often
should amortise that sync cost and speed up the run.

```json
{ "convergence_check_every": 20 }
```

## Baseline (reused from T02 precondition)

The S02 baseline bundle generated for T02 is reused unchanged (same commit, same
input, same z-selection), per the slice Must-Have that the baseline is
"reusable by both candidate runs".

- **baseline_id:** `baseline-20260714T123812-4e924e0-sub-22`
- z_indices `[0, 13, 27, 40, 54]` (z-sample 5), array_shape `[55, 2325, 1200]`, tile_shape `[75, 75]`
- strategy: `baseline` / sequential scalar, `cuda:0`, ws=128, darkfield on, `max_reweighting_iterations=15`
- aggregates: `seam_l1=0.571734`, `seam_curvature=0.034412`; `release_gate=True`
- artifacts: `scripts/experiments/s02_artifacts/baseline/{baseline-bundle.json, tolerance-sidecar.json, summary.md}`

## Candidate run

- **candidate_id:** `candidate-20260714T125229-4e924e0-sub-22`
- Same strategy (`baseline`/sequential/cuda:0/ws=128/darkfield) so the A/B
  isolates the sync-cadence lever; only `convergence_check_every` differs (20
  vs the GPU default 10).
- aggregates: `seam_l1=0.511713`, `seam_curvature=0.036182`

## Verdict: REJECT  ✘ (fail) — double failure

| metric | abs_delta | rel_delta | gate |
| --- | --- | --- | --- |
| seam_l1 | -0.060021 | -10.498% | ✓ pass (improved) |
| seam_curvature | +0.001770 | +5.144% | ✘ fail |
| speed | 0.9359x | **slower** | ✘ fail (regression, far below 1.30x) |
| **overall** | | | **reject / fail** |
| worst_z | 13 | | |

The lever fails on **both** the speed gate and the K01 quality gate — the
strongest possible refutation. There is no rescue variant worth testing: a lever
that makes the run slower *and* degrades curvature has no operating point worth
exploring.

### Speed — fails the 1.30x gate (actually a regression)

`baseline_ms=14621.995` → `candidate_ms=15623.209`, **ratio 0.9359x (slower)**.
This is below 1.0, i.e. the candidate is measurably slower than the baseline, not
just "not fast enough". The 3 measured repeats were tight (steady_state 15623ms,
cold 15623ms, warm 15646ms), so this is a real effect, not timing noise. The S01
hypothesis — that the `norm_fro` GPU→CPU sync is the dominant cost worth
amortising at ws=128 — is **empirically refuted**: halving the sync frequency
made the run slower, not faster.

### Quality — fails the K01 curvature gate

The K01 release gate compares aggregate and per-z seam deltas against tolerances
calibrated from the baseline (`mean+3std`, `sigma=3`). **seam_curvature fails**
(+5.14% aggregate). Per-z failures (largest contributors):

| z | metric | abs_delta | rel_delta |
| --- | --- | --- | --- |
| 13 | seam_curvature | 0.023890 | **56.21%** |
| 40 | seam_curvature | 0.000226 | 0.98% |
| 40 | seam_l1 | 0.006794 | 0.89% |
| 54 | seam_l1 | 0.000059 | 0.018% |
| 54 | seam_curvature | 0.000045 | 0.18% |

worst_z = 13. The curvature damage is concentrated at z=13 (+56%), the same
mid-stack plane that dominated the T02 reweighting-tolerance failure — this plane
appears intrinsically sensitive to solver-trajectory perturbation.

Note `seam_l1` actually **improved** at the aggregate (-10.5%) and dramatically at
z=27 (-44.9%). This is not a clean quality win: it is a *different* (not a
*better*) solver trajectory — the inner ALM loop, checked half as often, runs
extra iterations that drive seam_l1 down while pushing seam_curvature up. A lever
that trades one K01 metric for another is still a quality-gate failure.

### Root cause: inner-loop overshoot, not sync savings

Convergence telemetry: `reweight_iterations_per_z = {0:15, 13:15, 27:15, 40:15,
54:15}`, median 15.0 — the **outer** reweighting loop hit its 15-iteration cap on
every plane (identical to the baseline path; the outer loop is not converging at
ws=128). `convergence_check_every` only governs the **inner** ALM solver
(`_alm.py`), so the lever's sole effect is on the inner solver's stopping point.

With `check_every=20` (vs default 10), the inner stop criterion (`norm_fro(dY) /
(d_norm + 1e-9) < tol`) is evaluated half as often, so on each reweighting step
the inner solver runs up to ~10 extra ALM iterations before detecting
convergence. At ws=128 / sequential-scalar, those extra iterations' compute cost
**exceeds** the GPU→CPU sync cost the lever was designed to amortise — hence the
net slowdown. The trajectory shift (more inner iterations per reweighting step)
is also what moves seam_l1/seam_curvature. The speed mechanism the lever was
predicated on (sync amortisation) does not materialise at this workload shape.

### Gate-threshold caveat (for S03)

As with T02, the baseline was calibrated with `repeats=3` and identical-strategy
inter-repeat variance ~0, so the `mean+3std` tolerances collapsed to the `1e-6`
floor (`seam_l1` rel_tol ≈1.7e-6, `seam_curvature` rel_tol ≈2.9e-5). This makes
the curvature gate extremely sensitive. However, the z=13 curvature delta (+56%)
is orders of magnitude above any plausible noise band, and the speed result is a
clear regression independent of any tolerance floor — so the REJECT is doubly
robust.

## Conclusion for S03

The sync-cadence lever **cannot be promoted**: it makes the run **slower**
(0.9359x, a regression below the 1.0 break-even, let alone the 1.30x gate) **and**
fails the K01 seam-curvature gate (+5.14%, worst z=13 at +56%). Both S01
shortlisted speed levers have now been tested on real subjects (T02 + T03) and
**both are REJECTED**: reweighting-tolerance failed quality (truncation
under-fit) while delivering speed, and sync-cadence failed both speed and
quality (sync amortisation refuted).

**Reassessment signal for S03:** the aspirational 30% ws=128 speed target (R059)
is not reachable through either top-2 S01 shortlisted lever. The sync-cadence
result is the more decisive negative: it directly refutes the premise (sync is
the bottleneck) that underpinned several S01 shortlist entries. S03 should treat
the GPU→CPU sync-amortisation lever class as a dead end at ws=128/sequential and
re-rank the shortlist accordingly — the next speed levers to test (if any) should
target the actual dominant cost (the ALM compute itself, or the outer-loop
iteration count), not sync frequency.

## Artifacts

- `scripts/experiments/s02_configs/sync_cadence.json` — lever override config
- `scripts/experiments/s02_artifacts/sync_cadence/candidate-artifact.json` — full candidate artifact (verdict, deltas, convergence, telemetry)
- `scripts/experiments/s02_artifacts/sync_cadence/summary.md` — harness verdict table
- `scripts/experiments/s02_artifacts/baseline/{baseline-bundle.json, tolerance-sidecar.json, summary.md}` — baseline (reused from T02 precondition)

Server-side originals (A6000): `runs/s02/baseline-20260714T123812-4e924e0-sub-22/`
(reused) and `runs/s02/candidate-20260714T125229-4e924e0-sub-22/` (this task).
