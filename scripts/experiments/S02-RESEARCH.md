# S02 Research: Real-Subject A/B of the Top-2 S01 Speed Levers

> **Milestone M005 / Slice S02 — canonical research deliverable (T04).**
> Synthesizes the K01-quality-gated real-subject A/B evidence from T02
> (`s02_reweighting_tolerance_result.md`) and T03
> (`s02_sync_cadence_result.md`), both run against the single fresh S02
> baseline (T01) on the A6000. This document is standalone: an S03 agent
> can make the go/no-go call on the aspirational 30% ws=128 target (R059)
> without reading T01/T02/T03.
>
> **Read-only research.** No production source file under `linum_basic/`
> was modified. All numbers are measured on real subject `sub-22` and are
> cross-referenced to the version-controlled JSON artifacts under
> `scripts/experiments/s02_artifacts/`.

---

## Executive summary (TL;DR)

- **Both top-2 S01-shortlisted speed levers are REJECTED on real
  subjects.** Neither `reweighting-tolerance` (T02) nor `sync-cadence`
  (T03) can be promoted: both fail the K01 seam-quality gate, and
  `sync-cadence` additionally fails the 1.30x speed gate.
- **The aspirational 30% ws=128 speed target (R059) is not reachable
  through either shortlisted lever.** `reweighting-tolerance` delivered a
  large, real speed win (1.93x) but only by truncating the solver, which
  broke seam quality; `sync-cadence` made the run *slower* (0.936x) and
  broke curvature.
- **S01's shortlist rank order warrants reassessment (yes).** The
  `sync-cadence` result is the decisive negative: it empirically refutes
  the premise (GPU→CPU sync is the dominant cost worth amortising) that
  underpinned several S01 shortlist entries. The next speed levers — if
  any — must target the actual dominant cost (ALM compute / outer-loop
  iteration count), not sync frequency.
- **Net for S03:** a defensible **no-go** on reaching 30% via these
  levers, with one concrete rescue variant flagged (loosen
  `reweighting_tolerance` alone, drop the hard iteration cap) and the
  `amp-selective-fp16-dct` ambitious swing still open if appetite exists.

---

## 1. Baseline reference (T01 precondition)

A single fresh production-shaped baseline bundle was generated on the
A6000 and reused unchanged by both candidate runs (same commit, same
input, same z-selection), per the slice Must-Have that the baseline is
"reusable by both candidate runs".

| field | value |
|-------|-------|
| **baseline_id** | `baseline-20260714T123812-4e924e0-sub-22` |
| subject | `sub-22` (real microscopy mosaic, OME-Zarr) |
| input_fingerprint | `sha256:41ba18e223ab497f` |
| commit | `4e924e0` |
| hardware | A6000 `sn4622125853`, single GPU `cuda:0`, PyTorch 2.12.1+cu130, CUDA 13.0, Python 3.14.3 |
| z_indices | `[0, 13, 27, 40, 54]` (z-sample 5) |
| array_shape / tile_shape | `[55, 2325, 1200]` / `[75, 75]` (n_tiles=496) |
| strategy | `baseline` / `sequential_scalar`, `cuda:0`, ws=128, darkfield on, `max_reweighting_iterations=15`, `reweighting_tolerance=1e-3` |
| **baseline_ms** | **14621.995** (steady state; cold 14753.9, warm 14587.5) |
| seam_l1 (aggregate) | **0.571734** |
| seam_curvature (aggregate) | **0.034412** |
| release_gate | `True` |

Calibration: tolerances via `mean+3std`, `sigma=3`, `repeats=3`,
`min_abs=1e-6`. Identical-strategy inter-repeat variance was ~0, so the
tolerances collapsed to the `1e-6` floor
(`seam_l1` rel_tol ≈1.75e-6, `seam_curvature` rel_tol ≈2.91e-5) — see the
gate-floor caveat in §5.

Artifacts: `scripts/experiments/s02_artifacts/baseline/{baseline-bundle.json,
tolerance-sidecar.json, summary.md}`.

---

## 2. Candidate 1 — `reweighting-tolerance` (T02): REJECT ✘

| field | value |
|-------|-------|
| **candidate_id** | `candidate-20260714T124042-4e924e0-sub-22` |
| lever override | `reweighting_tolerance` `1e-3 → 5e-3`; `max_reweighting_iterations` `15 → 6` |
| config | `scripts/experiments/s02_configs/reweighting_tolerance.json` |
| run label | `s02-reweighting-tolerance` |

### Verdict table

| metric | abs_delta | rel_delta | gate |
|--------|-----------|-----------|------|
| seam_l1 | +0.004828 | +0.8444% | ✘ fail |
| seam_curvature | +0.001820 | +5.2903% | ✘ fail |
| speed | — | **1.9264x faster** | ✓ pass (≥1.30x) |
| **overall** | | | **reject / fail** |
| worst_z | 27 | | |

### Speed — passes the 1.30x gate

`14621.995 ms → 7590.293 ms` (**1.9264x**). Cold/warm repeats all ≈7.5–7.6s.
This clears the D-18/MEM007 1.30x speed threshold by a wide margin. **The
lever is genuinely fast.**

### Quality — fails the K01 gate (both metrics)

Per-z failures (largest contributors):

| z | metric | abs_delta | rel_delta |
|---|--------|-----------|-----------|
| 13 | seam_curvature | 0.024038 | **56.56%** |
| 27 | seam_l1 | 0.076855 | **11.89%** |
| 0 | seam_l1 | 0.000208 | 0.064% |
| 27 | seam_curvature | 0.000080 | 0.18% |
| 40 | seam_curvature | 0.000127 | 0.55% |
| 54 | seam_curvature | 0.000041 | 0.17% |

worst_z = 27. Damage concentrated at z=13 (curvature) and z=27 (seam_l1).

### Root cause — iteration cap truncated the solver, not genuine convergence

Convergence telemetry: every plane hit the `max_reweighting_iterations=6`
ceiling — `reweight_iterations_per_z = {0:6, 13:6, 27:6, 40:6, 54:6}`,
median 6.0. The looser `5e-3` tolerance was never reached within 6
iterations on any plane, so the solver was **truncated**, not **converged**.
The 1.93x speed win is therefore the cost of under-fitting: fewer
reweighting passes leave the flat/dark estimates rougher, surfacing as
elevated seam error at the mid-stack planes.

---

## 3. Candidate 2 — `sync-cadence` (T03): REJECT ✘ (double failure)

| field | value |
|-------|-------|
| **candidate_id** | `candidate-20260714T125229-4e924e0-sub-22` |
| lever override | `convergence_check_every` `10 → 20` (scalar path) |
| config | `scripts/experiments/s02_configs/sync_cadence.json` |
| run label | `s02-sync-cadence` |

### Verdict table

| metric | abs_delta | rel_delta | gate |
|--------|-----------|-----------|------|
| seam_l1 | -0.060021 | -10.498% | ✓ pass (improved) |
| seam_curvature | +0.001770 | +5.144% | ✘ fail |
| speed | — | **0.9359x (slower)** | ✘ fail (regression, <1.0) |
| **overall** | | | **reject / fail** |
| worst_z | 13 | | |

Fails on **both** the speed gate and the K01 quality gate — the strongest
possible refutation. There is no rescue variant worth testing: a lever that
makes the run slower *and* degrades curvature has no operating point worth
exploring.

### Speed — fails the 1.30x gate (an actual regression)

`14621.995 ms → 15623.209 ms` (**0.9359x — slower**), below the 1.0
break-even, let alone the 1.30x gate. The 3 measured repeats were tight
(steady 15623ms, cold 15623ms, warm 15646ms), so this is a real effect.
The S01 hypothesis — that the `norm_fro` GPU→CPU sync is the dominant cost
worth amortising at ws=128 — is **empirically refuted**: halving the sync
frequency made the run slower, not faster.

### Quality — fails the K01 curvature gate

Per-z failures (largest contributors):

| z | metric | abs_delta | rel_delta |
|---|--------|-----------|-----------|
| 13 | seam_curvature | 0.023890 | **56.21%** |
| 40 | seam_curvature | 0.000226 | 0.98% |
| 40 | seam_l1 | 0.006794 | 0.89% |
| 54 | seam_l1 | 0.000059 | 0.018% |
| 54 | seam_curvature | 0.000045 | 0.18% |

worst_z = 13. Curvature damage concentrated at z=13 (+56%) — the same
mid-stack plane that dominated the T02 curvature failure.

Note `seam_l1` actually **improved** at the aggregate (-10.5%) and
dramatically at z=27 (-44.9%). This is **not** a clean quality win: it is
a *different* (not a *better*) solver trajectory — the inner ALM loop,
checked half as often, runs extra iterations that drive seam_l1 down while
pushing seam_curvature up. A lever that trades one K01 metric for another
is still a quality-gate failure.

### Root cause — inner-loop overshoot, not sync savings

Convergence telemetry: `reweight_iterations_per_z = {0:15, 13:15, 27:15,
40:15, 54:15}`, median 15.0 — the **outer** reweighting loop hit its
15-iteration cap on every plane (identical to the baseline path; the outer
loop is not converging at ws=128). `convergence_check_every` only governs
the **inner** ALM solver (`_alm.py`), so the lever's sole effect is on the
inner solver's stopping point.

With `check_every=20` (vs default 10), the inner stop criterion
(`norm_fro(dY) / (d_norm + 1e-9) < tol`) is evaluated half as often, so on
each reweighting step the inner solver runs up to ~10 extra ALM iterations
before detecting convergence. At ws=128 / sequential-scalar, those extra
iterations' compute cost **exceeds** the GPU→CPU sync cost the lever was
designed to amortise — hence the net slowdown. The trajectory shift (more
inner iterations per reweighting step) is also what moves
seam_l1/seam_curvature. **The speed mechanism the lever was predicated on
(sync amortisation) does not materialise at this workload shape.**

---

## 4. Cross-cutting findings

### 4.1 z=13 is intrinsically sensitive to solver-trajectory perturbation

Both candidates' worst curvature damage lands on **z=13** (+56.6% in T02,
+56.2% in T03), despite the two levers perturbing the solver through
entirely different mechanisms (outer-loop truncation vs inner-loop
overshoot). This plane appears intrinsically sensitive to any change in
the solver trajectory — a useful canary for any future candidate A/B.

### 4.2 The outer reweighting loop does not converge at ws=128

Baseline and sync-cadence both run the full 15-iteration outer cap on
every plane (`reweight_iterations_median = 15.0`). The reweighting
tolerance is never met within the cap on the production path. This means
the dominant iteration budget is the **outer loop**, not the inner ALM
solve — which is why the one lever that bounded the outer loop
(reweighting-tolerance) was the only one that produced speed, and why
inner-loop tuning (sync-cadence) could not.

### 4.3 Gate-floor caveat (does not change any verdict)

The baseline was calibrated with `repeats=3`; identical-strategy
inter-repeat variance was ~0, so the `mean+3std` tolerances collapsed to
the `1e-6` floor. This makes the gate extremely sensitive to any real
algorithmic change. However, the worst-plane deltas are 12–57% — orders of
magnitude above any plausible noise band — and the sync-cadence speed
result is a clear regression independent of any tolerance floor. **Both
REJECT verdicts are robust**, not artifacts of the tight floor.

---

## 5. Reassessment of S01's shortlist rank order — YES, warranted

The slice Must-Have asks whether either result is surprising enough to
warrant reassessing S01's shortlist rank order. **The answer is yes —
primarily because of the sync-cadence result.**

S01's shortlist (`s01_synthesis_notes.md` §5) ranked `reweighting-tolerance`
#1 and `sync-cadence` #2 as the two `attempt-now` levers. The S01 analysis
predicated `sync-cadence`'s value on the premise that the inner-loop
`norm_fro` GPU→CPU sync (`_alm.py:441-500`) is a cost worth amortising at
ws=128. **T03 measured the opposite**: halving the sync frequency made the
run measurably slower (0.936x) because the extra inner-iteration compute
dominates the saved sync latency. This directly falsifies the premise and
has implications beyond the single lever:

- **The sync-amortisation lever class is a dead end at ws=128/sequential.**
  S01 §4 row 4 (`sync-cadence`) and any other shortlist entry resting on
  "sync is the bottleneck" should be down-ranked. S01 §2 already noted
  sync is "a minor cost ... real but small vs bandwidth-bound kernel
  time"; T03 elevates that from an estimate to a measured refutation.
- **The dominant cost is the outer reweighting loop / ALM compute, not
  sync.** Any next speed lever should reduce the number of ALM solves or
  bytes-per-iteration, not tune sync cadence.

`reweighting-tolerance`'s result (T02) is **directionally as S01 predicted**
(S01 flagged it as Medium-High seam risk and bimodal: "~15-35% if
convergence allows ... ~0% if gate trips"). The surprise is only in the
specific failure mode — the lever was truncated by the iteration cap
rather than converging early — which down-ranks the "loosen tolerance +
cap iterations" *formulation* (the cap did the damage), not the lever
class itself. A rescue variant exists (§6).

**Reassessed signal for S03:** demote `sync-cadence` and the
sync-amortisation class; treat `reweighting-tolerance` as still
potentially viable only via the cap-free rescue variant; keep the
precision-gated `amp-selective-fp16-dct` ambitious swing as the only
remaining lever with a theoretical path to a large fraction of 30%.

---

## 6. Conclusion & handoff to S03

### Go/no-go on R059 (aspirational 30% ws=128 target)

**No-go via the top-2 S01 shortlisted levers.** Both are REJECTED with
K01-gated real-subject evidence:
- `reweighting-tolerance`: 1.93x faster but K01 seam quality FAIL (truncation
  under-fit; worst z=27).
- `sync-cadence`: 0.936x slower AND K01 curvature FAIL (+5.14%, worst z=13);
  sync-amortisation premise refuted.

This corroborates S01 §6's honest assessment that the 30% target is "not
credibly reachable from any single remaining lever on the production eager
path." S03 should record a defensible **no-go** on the 30% target through
these levers.

### Concrete next steps for S03 (if appetite remains)

1. **One rescue variant worth testing** (flagged, not executed here): loosen
   `reweighting_tolerance` **alone** — drop the `max_reweighting_iterations`
   cap (or raise it well above 6) so the looser stopping rule ends the outer
   loop early *only when actually met*, rather than force-truncating at 6.
   This preserves the speed mechanism (early stop on tolerance) without the
   under-fit. Given the outer loop runs the full 15-iter cap on every plane
   (§4.2), a genuine early-stop has real headroom — but it is bimodal and
   must pass the full K01 gate.
2. **The only remaining path to a large fraction of 30%** is
   `amp-selective-fp16-dct` (S01 §5 ambitious swing), the sole lever that
   attacks the confirmed memory-bandwidth bottleneck at its root. It is
   hard-gated by K02/K06 (precision); if attempted, it must ship with the
   full seam gate + a fixed-iteration numerical parity check (K08), scoped
   to DCT-matmul-in-fp16 with full fp32 accumulate/reduce. Its real
   precision-safe ceiling is **unknown without its own real-subject A/B**.
3. **Do not pursue** (measured/rationale-closed): `sync-cadence` and the
   sync-amortisation class (T03 refuted), plus all S01 §7 rejects
   (`svd-power-iter-n-iter` K02 invariant, `warm-start-state-device-resident`
   ~0% ceiling, `batched-cuda-cache-fit-ws64` MEM002 closed,
   `tile-subsampling` tuning-only, compile-class levers inert under eager).

### What S02 proves (slice proof level: operational)

This slice operationally exercises the existing K01-gated evidence harness
on real A6000 hardware for the top-2 shortlisted levers and produces
defensible, quality-gated promote/reject verdicts. It does **not** change
any production code. The remaining step before the milestone is usable
end-to-end is S03 reading these verdicts and producing the go/no-go
decision artifact (R059); if go, a follow-on execution milestone scopes
the actual code change.

---

## Verification

- **No production source files modified** — this is a read-only
  synthesis of existing version-controlled artifacts. No file under
  `linum_basic/` was touched in T04.
- Every numeric claim is cross-referenced to a version-controlled JSON
  artifact: baseline (`s02_artifacts/baseline/baseline-bundle.json`,
  `tolerance-sidecar.json`), reweighting-tolerance
  (`s02_artifacts/reweighting_tolerance/candidate-artifact.json`),
  sync-cadence (`s02_artifacts/sync_cadence/candidate-artifact.json`).
- T04 task `Verify` gate:
  `test -s scripts/experiments/s02_reweighting_tolerance_result.md &&
   test -s scripts/experiments/s02_sync_cadence_result.md` — both source
  documents exist (T02/T03 deliverables).
- Slice Must-Have satisfied: a single research artifact (this file)
  records the baseline id, both candidate ids, per-candidate speed ratio +
  quality verdict + promote/reject decision with rationale, and flags
  whether either result warrants reassessing S01's shortlist rank order
  (§5: yes).

## Failure Modes

This task has **no external dependencies** — it is a read-only synthesis
of already-produced version-controlled artifacts (the T01 baseline bundle
and the T02/T03 candidate artifacts and result markdown), which are static
numeric evidence captured during the earlier A6000 runs. No network,
filesystem mutation, subprocess, or API is invoked in T04. The only
failure mode is a source artifact being missing or internally
inconsistent; this was guarded by verifying all three candidate JSON files
and both result markdown files exist and that their reported deltas match
the raw aggregate metrics recomputed from the per-candidate
`metrics_aggregates` vs baseline. The candidate artifacts' own
`warnings` field records benign environment-differs notices (cuda/platform/
python/torch version strings present on candidate but `None` on the
baseline's nested environment block) that do not affect the verdicts — the
quality and speed deltas are computed from the shared baseline reference,
not from environment comparison.

## Load Profile

Not applicable — T04 is a one-shot static synthesis with no runtime load
dimension. There is no request volume to saturate; the artifact is a
fixed-size markdown document consumed by S03 (go/no-go decision). No
compute, GPU, or concurrency is exercised by the synthesis itself.

## Negative Tests

Not applicable — T04 produces a research synthesis, not executable code,
so there is no malformed-input surface. Factual claims are individually
cross-referenced to source JSON artifacts so a reviewer can falsify any
row. The two underlying A/B runs (T02/T03) *are* the negative-evidence
generators: each is a real-subject test whose K01-gated verdict is a
measured REJECT, and the candidate artifacts persist the per-z failure
rows, worst_z, convergence telemetry, and speed ratio that constitute the
objective negative result. The harness's own failure-visibility
(ValueError + exit 1 on subject/fingerprint/z-selection mismatch; per-z
quality/speed verdicts in artifact metadata) was exercised by the T02/T03
runs that produced this synthesis's inputs.
