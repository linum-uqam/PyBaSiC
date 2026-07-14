# S03 Decision: Go / No-Go on the Aspirational 30% ws=128 Speed Target (R059)

> **Milestone M005 / Slice S03 — terminal decision artifact.**
> This document closes the M005 research spike with a defensible go/no-go
> verdict on R059. It is the synthesis of S01 (lever survey + bottleneck
> re-profiling) and S02 (K01-gated real-subject A/B of the top-2 levers),
> both of which are standalone and cited below. An S04/next-milestone
> agent can read this file alone to understand the ceiling without
> re-running the A6000 experiments.
>
> **Read-only synthesis.** No production source file under `linum_basic/`
> is modified. No roadmap structure is changed. The verdict is persisted
> into durable project state (R059 requirement + DECISIONS.md) in T02.

---

## Verdict (TL;DR)

### **NO-GO: the aspirational 30% ws=128 speed target (R059) is NOT achievable through any currently-tested lever.**

- **Realistic confirmed ceiling today:** the already-shipped
  **~2.2%** steady-state reduction from the `dct-kernel-tuning` /
  `worker-compile-off` + `auto-l_s` stack promoted in M002 (D003/D005,
  MEM003). **0% additional** is confirmed from the two S02-tested levers.
- **Why:** S01 established analytically that 30% is "not credibly reachable
  from any single remaining lever on the production eager path" (§1).
  S02 then measured the top-2 shortlisted levers on real subject `sub-22`
  against a single fresh A6000 baseline — **both were REJECTED** by the K01
  seam-quality gate, and one was additionally a *speed regression*
  (S02 §2–§3).
- **R059 disposition:** **re-deferred.** R059 remains `deferred` with this
  spike's ceiling and evidence trail cited, so a future agent re-discovering
  the requirement does not re-run the same rejected experiments.

### Two unproven avenues remain open (flagged, NOT committed to a milestone)

1. **`reweighting-tolerance` rescue variant** — loosen `reweighting_tolerance`
   *alone*, dropping the `max_reweighting_iterations` hard cap that
   force-truncated S02's run. Bimodal; must pass the full K01 gate.
2. **`amp-selective-fp16-dct`** — the only lever that attacks the confirmed
   memory-bandwidth bottleneck at its root (DCT matmul bytes/element). Hard
   precision-gated by K02/K06; real precision-safe ceiling unknown without
   its own A/B.

Neither is scoped into an execution milestone here. M005 is a
plan-and-research spike; if appetite remains, a **follow-on milestone** (to
be scoped separately) would pick up either or both.

---

## 1. The decision question and what "go" would require

M005 exists to answer one question (per its roadmap vision and success
criteria):

> *Is the aspirational 30% ws=128 speed target (R059) achievable, now that
> M002 shipped the performance forensics and v1.0 achieved only ~2.2%?*

v1.0's ~2.2% came from the `worker-compile-off` + `auto-l_s` +
`dct-kernel-tuning` stack (D003/D005, MEM003). The 30% target is
**aspirational**, not a release gate. "Go" would mean: a lever (or stack)
demonstrably closes a credible fraction of the gap to 30% while passing the
**K01 real-subject seam gate** (`seam_l1` + `seam_curvature`) and the
**1.30x speed gate** (D-18/D007). "No-go" means documenting the realistic
ceiling with rationale and re-deferring R059.

The bar for *go* is high because ws=128 is **memory-bandwidth-bound**, not
compute-bound (MEM001/MEM002, K03) — the obvious fix (re-enabling
`torch.compile` for kernel fusion) is **structurally closed** because
`worker-compile-off` (D005) is load-bearing: it killed an Inductor CPU
compile storm that had pushed per-z fits to ~130 s+.

---

## 2. Evidence — S01: the analytic ceiling (not credibly reachable)

Source: `scripts/experiments/s01_synthesis_notes.md` (T03 canonical
synthesis of the 13-lever catalog from T01/T02).

**Bottleneck, confirmed and sharpened (S01 §2):** the eager ws=128 inner
loop is a chain of unfused full-matrix passes (77 `np.float32` casts + 6
`xp.clone` in `_alm.py:187-246`; dominant op is the eager fp32 2D-DCT
matmul `_Ap @ x.contiguous() @ _Aq.T` at `_alm.py:155`). The working set
(~6.5 MB for n≈100) exceeds the A6000's 4.8 MB L2 and spills to HBM, which
is *why* batched CUDA at ws=128 equals or loses to sequential (K03/D009).
Compile/fusion — the natural bandwidth fix — is closed by D005.

**S01 §6 honest ceiling assessment:**

1. The bandwidth-attacking levers are **precision-blocked**
   (`amp-selective-fp16-dct`, gated by K02/K06) or **data-rejected**
   (`batched-cuda-cache-fit-ws64`, closed by MEM002's real-subject sweep,
   ratio <1.30 at quality parity).
2. The only high-mechanism-ceiling lever (`reweighting-tolerance`,
   ~15-35%) is **quality-gated and bimodal** — ~0% if the seam gate trips.
3. Everything else is single-digit or zero (`sync-cadence` ~1-3%,
   `fp32-y-accumulator` ~3-6%, `cuda-stream-prefetch` low-moderate);
   stacked optimistically these sum to well under 30%.

> **S01 conclusion:** "not credibly reachable from any single remaining
> lever on the production eager path. … S02 should plan for a ~5-10%
> realistic stack, not 30%."

S01 ranked `reweighting-tolerance` #1 and `sync-cadence` #2 as the two
`attempt-now` levers to try first on real subjects.

---

## 3. Evidence — S02: both top-2 levers REJECTED on real subjects

Source: `scripts/experiments/S02-RESEARCH.md` (T04 canonical synthesis).
Both candidates were run against a **single fresh production-shaped
baseline** on the A6000 (reused unchanged by both runs — same commit,
input, z-selection):

| baseline field | value |
|---|---|
| baseline_id | `baseline-20260714T123812-4e924e0-sub-22` |
| subject | `sub-22` (real microscopy mosaic, OME-Zarr) |
| commit | `4e924e0` |
| hardware | A6000 `sn4622125853`, single GPU `cuda:0`, PyTorch 2.12.1+cu130, CUDA 13.0, Python 3.14.3 |
| baseline_ms | **14621.995** (steady state) |
| baseline seam_l1 / seam_curvature | 0.571734 / 0.034412 |

### 3.1 Candidate 1 — `reweighting-tolerance`: REJECT ✘ (quality)

| metric | result | gate |
|---|---|---|
| speed | **1.9264x faster** (14621.995 → 7590.293 ms) | ✓ pass (≥1.30x) |
| seam_l1 | +0.8444% | ✘ **fail** |
| seam_curvature | +5.2903% | ✘ **fail** |
| worst_z | 27 | |

**Evidence ID:** `candidate-20260714T124042-4e924e0-sub-22`
(`s02_artifacts/reweighting_tolerance/candidate-artifact.json`).

The lever is genuinely fast (clears the speed gate by a wide margin), but
the speed came from **under-fitting, not convergence**: every plane hit
the `max_reweighting_iterations=6` cap (`reweight_iterations_per_z` = 6 on
all planes), so the looser `5e-3` tolerance was never reached. The flatter
flat/dark estimates surfaced as elevated seam error at the mid-stack
planes (z=13 curvature +56.6%, z=27 seam_l1 +11.9%).

### 3.2 Candidate 2 — `sync-cadence`: REJECT ✘ (double failure)

| metric | result | gate |
|---|---|---|
| speed | **0.9359x — SLOWER** (14621.995 → 15623.209 ms) | ✘ **fail** (regression) |
| seam_l1 | -10.498% | ✓ improved (but see note) |
| seam_curvature | +5.144% | ✘ **fail** |
| worst_z | 13 | |

**Evidence ID:** `candidate-20260714T125229-4e924e0-sub-22`
(`s02_artifacts/sync_cadence/candidate-artifact.json`).

This is the **strongest possible refutation**: a lever that makes the run
slower *and* degrades curvature. There is no rescue variant worth testing.
The S01 premise it rested on — that the inner-loop `norm_fro` GPU→CPU sync
is a cost worth amortising at ws=128 — is **empirically refuted**: halving
the sync frequency (`check_every` 10→20) made the run slower because the
extra inner-iteration **compute** exceeds the saved sync latency. The
`seam_l1` "improvement" is not a quality win but a *different* solver
trajectory (more inner iterations per reweighting step move seam_l1 down
and curvature up); trading one K01 metric for another is still a quality
failure.

### 3.3 Cross-cutting S02 findings that shape the verdict

- **z=13 is an intrinsic canary.** Both candidates' worst curvature damage
  lands on z=13 (+56.6% / +56.2%) despite perturbing the solver through
  entirely different mechanisms (outer-loop truncation vs inner-loop
  overshoot). Any future candidate A/B should watch z=13.
- **The outer reweighting loop does not converge at ws=128** — baseline
  and sync-cadence both run the full 15-iteration outer cap on every plane.
  The dominant iteration budget is the **outer loop / ALM compute, not
  sync**, which is exactly why the one lever that bounded the outer loop
  produced speed and the inner-loop lever could not.
- **Sync-amortisation is a dead-end lever class at ws=128/sequential** —
  down-ranked decisively by T03's measurement.
- **Gate-floor caveat (does not change the verdict):** baseline calibration
  with `repeats=3` collapsed `mean+3std` tolerances to the `1e-6` floor
  (inter-repeat variance ~0), making the gate very sensitive. But the
  worst-plane deltas are 12–57% — orders of magnitude above any plausible
  noise band — and the sync-cadence speed regression is independent of any
  tolerance floor. **Both REJECT verdicts are robust.**

---

## 4. Synthesis — why this is a defensible NO-GO

| Question | Answer |
|---|---|
| Did any single lever reach 30%? | **No.** S01's highest-mechanism-ceiling lever (`reweighting-tolerance`, ~15-35% theoretical) produced 1.93x speed *but* failed the K01 quality gate; its speed was truncation under-fit, not genuine convergence. |
| Did any lever reach the 1.30x speed gate *and* the K01 quality gate? | **No.** `reweighting-tolerance` passed speed, failed quality. `sync-cadence` failed both (a regression). |
| Is the bottleneck itself attackable? | **Not without re-opening closed paths.** The bandwidth bottleneck (S01 §2) can only be cut by fusion (closed by D005) or by halving dominant-tensor bytes (`amp-selective-fp16-dct`, precision-blocked by K02/K06). |
| Is the analytic ceiling corroborated by measurement? | **Yes.** S01 predicted "~5-10% realistic stack, not 30%"; S02 measured **0% confirmed** from the top-2 levers, exactly the pessimistic tail of S01's bimodal estimate. |
| Is any confirmed speed retained? | **Only the already-shipped ~2.2%** from M002's `dct-kernel-tuning` stack. No new speed is confirmed promotable from M005. |

**The 30% aspirational target is not achievable through the levers M005
investigated, and the evidence shows why it is unlikely to be reachable
from the *next* levers either:** the dominant cost is the outer
reweighting loop / ALM compute on a bandwidth-bound eager path, and the
two structural escape hatches (fusion, half-precision) are respectively
load-bearing-closed (D005) and precision-gated (K02/K06).

---

## 5. Realistic ceiling and R059 disposition

**Documented realistic ceiling:** **~2.2%** steady-state fit-time reduction,
fully shipped (M002 / D003 / MEM003). **0% additional** confirmed
promotable from the M005 spike. S01's optimistic *stack* ceiling was
~5-10%; S02's real-subject measurements landed at the pessimistic tail
(0% from the tested levers).

**R059 disposition: re-deferred.** R059 remains `deferred`. T02 updates
R059's requirement notes/validation to cite this spike's evidence trail
(`s01_synthesis_notes.md`, `S02-RESEARCH.md`, this decision) and the
documented ceiling, so the requirement carries the *why* and not just the
*what*. T02 also records the structural go/no-go call in `DECISIONS.md`
(scope `M005/R059`) for cross-milestone visibility.

---

## 6. Open avenues (flagged, not committed)

These are named so a future milestone can evaluate them; **M005 does not
scope an execution milestone for either.**

### 6.1 `reweighting-tolerance` rescue variant

Loosen `reweighting_tolerance` **alone** — drop (or raise well above 6)
the `max_reweighting_iterations` cap that force-truncated S02's run. With
the cap gone, the looser stopping rule ends the outer loop early *only when
actually met* rather than truncating. Because the outer loop runs the full
15-iter cap on every plane (S02 §4.2), a genuine early-stop has real
headroom — but the lever remains **bimodal** (it could still trip the K01
gate) and must pass the full seam gate. This was S01's #1 lever and S02's
#1 by speed; the cap-free formulation is the untested variant.

### 6.2 `amp-selective-fp16-dct` (ambitious swing)

The sole lever that attacks the confirmed memory-bandwidth bottleneck at
its root — halving the dominant DCT-matmul tensor's bytes roughly doubles
effective memory throughput on a bandwidth-bound kernel (~30-50%
*theoretical* ceiling). Hard-gated by K02/K06: only a **selective** shape
(DCT matmul in fp16, full fp32 accumulate/reduce) is viable, and its real
precision-safe ceiling is **unknown without its own real-subject A/B**. If
attempted, it must ship with the full K01 seam gate **and** a
fixed-iteration numerical parity check (K08).

### 6.3 Do not pursue (measured or rationale-closed)

- `sync-cadence` and the sync-amortisation class — refuted by S02 §3.
- `svd-power-iter-n-iter` — K02-protected invariant (~0.001% σ₁ drift
  breaks darkfield tests).
- `warm-start-state-device-resident` — ~0% ceiling (~2 ms vs multi-second
  fit).
- `batched-cuda-cache-fit-ws64` — closed by MEM002 (ratio <1.30 at quality
  parity).
- `tile-subsampling` — tuning-search only, does not affect fit time.
- Compile-class levers (`compile-surfacing`, `compile-shape-stability`,
  `inductor-cache-warm-policy`) — inert under the production eager path
  (compile off per D005).

---

## 7. What this milestone proves and what it does not

**Proves (proof level: operational):** M005 operationally answers its
framing question with quality-gated real-subject evidence. The go/no-go
decision is defensible and reproducible from the version-controlled
artifacts under `scripts/experiments/s02_artifacts/`.

**Does not prove / does not do:**
- It does **not** change any production code (no file under `linum_basic/`
  is modified in any M005 slice).
- It does **not** exhaustively rule out 30% forever — two avenues (§6.1,
  §6.2) remain genuinely open and would require a separate execution
  milestone to evaluate.
- It does **not** re-open the closed compile/fusion path (D005) or custom
  Triton kernels (out of scope for this spike).

---

## Verification

- **No production source files modified** — this is a read-only decision
  synthesis. No file under `linum_basic/` is touched.
- Every numeric claim is cross-referenced to a version-controlled source:
  S01 catalog/ceiling → `s01_synthesis_notes.md`; S02 measured deltas →
  `S02-RESEARCH.md` and the JSON artifacts under
  `scripts/experiments/s02_artifacts/`; the ~2.2% shipped stack → D003 /
  MEM003.
- T01 task `Verify` gate:
  `test -s scripts/experiments/S03-DECISION.md &&
   grep -q "NO-GO" scripts/experiments/S03-DECISION.md` — satisfied by
  this file (the verdict appears in the §Verdict heading and is the
  document's central finding).
- T02 (separate task) persists this verdict into R059
  (`gsd_requirement_update`) and `DECISIONS.md` (`gsd_decision_save`),
  closing the slice's Must-Haves for durable cross-milestone state.

## Failure Modes

This task has **no external dependencies** — it is a one-shot read-only
synthesis of two version-controlled research artifacts
(`s01_synthesis_notes.md`, `S02-RESEARCH.md`), which themselves are static
analyses of version-controlled source and measured A6000 artifacts. No
network, filesystem mutation, subprocess, or API is invoked. The only
failure mode is a source artifact being missing or a claim drifting from
its cited evidence; this is guarded by cross-referencing every numeric
claim to its source file and evidence ID, so a reviewer can falsify any
row.

## Load Profile

Not applicable — this is a one-shot decision synthesis with no runtime
load dimension. There is no request volume to saturate; the artifact is a
fixed-size markdown document consumed by R059's requirement update (T02)
and by any future milestone that re-evaluates the target. No compute, GPU,
or concurrency is exercised by the synthesis itself.

## Negative Tests

Not applicable — T01 produces a decision artifact, not executable code, so
there is no malformed-input surface. The decision's falsification surface
is its evidence: every REJECT verdict traces to a measured K01-gated
real-subject A/B (S02 §2–§3), and the analytic ceiling (S01 §6) is
explicitly labelled as an estimate with stated assumptions (eager
production path, real-subject gate per K01). A reviewer who believes the
no-go is wrong must produce a lever that passes both the 1.30x speed gate
and the `seam_l1`/`seam_curvature` quality gate on real subjects — the
two open avenues in §6 are exactly the candidates for such a challenge.
