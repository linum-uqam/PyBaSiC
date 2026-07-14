# S01 Research: ws=128 Speed-Lever Survey & Bottleneck Re-profiling

> **Milestone M005 / Slice S01 — canonical research deliverable.**
> Synthesized by T03 from T01 (`s01_lever_inventory.md` — existing 7-lever
> taxonomy with line-cited code evidence) and T02 (`s01_new_levers.md` — 6
> new levers + ceiling estimates for all 13). This document is standalone:
> an S02/S03 agent can act on it without reading T01/T02.
>
> **Read-only research.** No production source file under `linum_basic/`
> was modified. All claims are line-cited to the current tree; ceiling
> figures are explicitly *estimates* anchored to the eager production path
> and the real-subject seam gate (K01), not measured results.

---

## Executive summary (TL;DR)

- **Bottleneck:** ws=128 is **memory-bandwidth-bound**, confirmed and
  sharpened (§2). The obvious fix — re-enabling `torch.compile` for kernel
  fusion — is **structurally closed** because `worker-compile-off` (D005)
  is load-bearing (it killed an Inductor CPU compile storm that had pushed
  per-z fits to ~130 s+).
- **30% aspirational target:** **not credibly reachable from any single
  remaining lever** on the production eager path. The two levers that
  attack the confirmed bottleneck at its root are precision-blocked
  (`amp-selective-fp16-dct`) or already data-rejected
  (`batched-cuda-cache-fit-ws64`). A realistic *stack* of the open levers
  is ~5-10%, not 30% (§6).
- **Ranked shortlist for S02** (§5):
  1. **`reweighting-tolerance`** — `attempt-now`. Highest mechanism ceiling
     (~15-35%), **zero code change** (public `BaSiC` attributes), quality-gated.
  2. **`sync-cadence`** — `attempt-now`. ~1-3%, **near-zero risk**, easy A/B.
  - *(ambitious swing)* **`amp-selective-fp16-dct`** — `needs-more-research`.
     Only lever with a theoretical path to a large fraction of 30%, but
     precision-gated by K02/K06; may not pay out after precision-safe scoping.

---

## 1. Scope & production baseline

This artifact answers: *beyond the already-promoted stack
(`worker-compile-off` + `auto-l-s` + `dct-kernel-tuning` ≈ **2.2%**
reported steady-state reduction, D005/FORE-04/D003), what levers remain
for cutting ws=128 fit time, what is each one's ceiling, and which 1-2
should S02 try first on real subjects?*

Target path: the **production eager multi-GPU fan-out** (D-14), one
z-plane per joblib worker, `LINUM_BASIC_ALM_COMPILE_MODE=off`. The
already-promoted `dct-kernel-tuning` matmul-DCT path
(`A_p @ X @ A_q.T`) is active but runs eager. Adoption bar for any new
lever is the **1.30x speed gate + seam_l1/seam_curvature quality gate on
real subjects** (D-18/D007, K01).

---

## 2. Bottleneck reassessment — memory-bandwidth-bound, confirmed (MH2)

The MEM001/MEM002/K03 finding (ws=128 is memory-bandwidth-bound, not
compute-bound) is **confirmed and sharpened** using current codebase
evidence, not just prior memory entries:

| Evidence | Site | Why it shows bandwidth-bound |
|----------|------|------------------------------|
| Eager inner loop = chain of unfused full-matrix passes | 77 `np.float32` casts + 6 `xp.clone` in `_alm.py` inner region (`:187-246`) | Each cast/clone/matmul is a separate kernel launch touching `(n, 16384)` tensors — no fusion because compile is off (D005) |
| Dominant op runs eager fp32 matmul | `_alm.py:155` `_Ap @ x.contiguous() @ _Aq.T` | The 2D DCT matmul touches the full sorted-image matrix every inner iteration |
| Working set exceeds A6000 L2 | ws=128 sorted matrix ≈ `n × 16384 × 4 B` ≈ **6.5 MB** for n≈100 > A6000 L2 (4.8 MB) | Spills to HBM → bandwidth, not compute, limits throughput. This is the physical reason batched CUDA at ws=128 equals/loses to sequential (K03/D009) |
| Sync cadence is a minor cost | `_alm.py:441-442,500` (scalar `convergence_check_every=10`), `:789,834` (batched hardcoded 10) | Convergence `norm_fro` forces host sync every 10 iters — real but small vs bandwidth-bound kernel time |
| Compile-shape guards exist but are inert | `_alm.py:477-480` (mu 0-dim tensor), `clone()` dispatch-key strip | These prevent recompile storms, but with compile off they don't affect eager fit time |

**Why compile/fusion (the obvious bandwidth fix) is closed:** D005 shipped
because the compiled path caused an Inductor CPU compile storm in
short-lived joblib workers (per-z fit dropped from ~130 s+ to ~3-6 s/z).
Re-enabling compile to recover fusion would regress far more than any
fusion gain. **The bandwidth-bound eager path is the intentional steady
state, not an accident.** This is why the realistic ceiling across all
remaining levers is modest — the system is already near its achievable
bandwidth efficiency, and the levers that could change the byte budget
are either precision-blocked or data-rejected.

---

## 3. The decisive reframing: production runs EAGER

`worker-compile-off` (D005) disables `torch.compile` in CUDA joblib
workers. This single fact collapses the ceiling of every compile-class
lever:

| Implication | Evidence | Effect |
|-------------|----------|--------|
| No Inductor → no Triton fusion | `_alm.py:53` `_read_alm_compile_mode`; D005 `511c88c` | `compile-shape-stability`, `inductor-cache-warm-policy` inert — ceiling ~0% |
| TF32 only set inside compiled step | `torch.set_float32_matmul_precision("high")` gated behind compile branch (`_alm.py:255` scalar, `:662` batched) | Under eager, DCT matmul runs **default float32** — no TF32 to "widen"; the TF32 half of `compile-surfacing` has no speed surface (independent of K06) |

**Triage:** on the eager path, only levers that (a) reduce the *number* of
ALM solves/reweighting passes, (b) reduce *bytes per inner iteration*, or
(c) overlap transfer with compute can move fit time.

---

## 4. Complete lever catalog — 13 levers (MH1)

Master table. Every lever from `profile.py:_LEVER_DEFINITIONS` (7) plus the
6 newly identified. **Ceiling** = realistic *additional* steady-state
fit-time reduction beyond the already-promoted stack, under the eager
production path. **Seam-quality risk** = expected risk to the
seam_l1/seam_curvature gate (K01).

| # | Lever | Bottleneck class | Status | Ceiling (+ rationale) | Seam-quality risk | Recommendation |
|---|-------|------------------|--------|------------------------|-------------------|----------------|
| 1 | `reweighting-tolerance` (`core.py:208,209,451`) | compute (fewer solves) | open, **zero code change** | **~15-35%** if convergence allows fewer full ALM solves; ~0% if gate trips | **Medium-High** — fewer passes can let flat-field drift; K01 gate is exactly the test; prior phases found convergence cuts frequently trip quality | **attempt-now** |
| 2 | `amp-selective-fp16-dct` (`_alm.py:155`, `backend.py`) | memory-bandwidth | NEW, not attempted | **~30-50% theoretical** (halve dominant-tensor bytes ≈ double effective BW), precision-gated; real ceiling unknown | **High** — fp16/bf16 drift shifts ALM soft-threshold boundaries (K02); must scope DCT-only + fp32 accumulate | **needs-more-research** |
| 3 | `fp32-y-accumulator` (`_alm.py:496`) | memory-bandwidth | NEW, not attempted | **~3-6%** (Y is ~⅙-⅕ of per-iter tensors; halving one) | **Medium** — fp64 Y was deliberate for long-run stability; precision-adjacent (K06) but milder than #2 | **needs-more-research** |
| 4 | `sync-cadence` (`_alm.py:441,500,789,834`) | synchronization | open, kwarg exists | **~1-3%** (sync latency small vs bandwidth-bound kernel time) | **Low** — checking convergence less often cannot change the converged result; worst case runs N extra iters (slower, not lower quality) | **attempt-now** |
| 5 | `cuda-stream-prefetch-z-overlap` (`fit.py:394`) | sync + bandwidth | NEW, not attempted | **low-moderate**, needs profile; capped by multi-process model + possible BW saturation | **Low** — pure transfer/compute overlap, no numerical change | **needs-more-research** |
| 6 | `dct-kernel-tuning` (`backend.py`, `_alm.py:193-205`) | compute | **promoted (~2.2%, D003)** | **~0-1%** more (contiguity marginal under eager) | n/a — **already passed** seam gate | done |
| 7 | `compile-surfacing` (`_alm.py:44,53`) | compile-shape | promoted (observability, K07) | **~0%** (TF32 not active under eager; K06 forbids widening) | n/a — observability only | done |
| 8 | `compile-shape-stability` (`_alm.py:477-480`) | compile-shape | satisfied in code | **~0%** (compile off in production) | n/a — moot under eager | moot |
| 9 | `inductor-cache-warm-policy` (`_torch_cache.py:60,86`) | compile-shape | deferred (Phase 5 backlog) | **~0%** (benchmark-harness infra, not production fit) | n/a — harness only | reject |
| 10 | `tile-subsampling` (`tuning.py:72,329,338`) | compute | exists | **0%** for fit-time target (tuning-search only) | n/a — affects Optuna trials, not fit | reject |
| 11 | `svd-power-iter-n-iter` (`backend.py:478,501`) | compute | NEW | **~0%** (once per solve, tiny `bmm`) | **High if touched** — K02 protected invariant (~0.001% σ₁ drift breaks darkfield tests) | **reject-with-rationale** |
| 12 | `warm-start-state-device-resident` (`_alm.py:421,525`) | memory-bandwidth | NEW | **~0%** (~2 ms host↔device vs multi-second fit) | **None** — correctness-preserving cleanup | **reject-with-rationale** (as speed lever) |
| 13 | `batched-cuda-cache-fit-ws64` (`fit.py`/`strategies.py`) | chunking + bandwidth | NEW | N/A — **MEM002 already rejected** ws<128 (ratio <1.30, quality parity) | n/a — closed on real subjects | **reject-with-rationale** |

---

## 5. Ranked shortlist for S02 (MH3)

### #1 — `reweighting-tolerance` (attempt-now)

**Why it ranks first:** highest *mechanism* ceiling of any open lever
(~15-35%), and the only one that could plausibly approach a meaningful
fraction of the target. Each removed full reweighting pass eliminates a
complete ALM solve.

**Why chosen over the rest:** it requires **zero code change** —
`reweighting_tolerance` (1e-3) and `max_reweighting_iterations` (10) are
public `BaSiC` attributes (`core.py:208-209`), so S02 can A/B by
construction. By contrast, the precision-levers (#2/#3) need code changes
and carry K02/K06 risk, and `sync-cadence` (#4) has a far smaller ceiling.

**Supporting evidence:** the stop condition (`core.py:451-452`) and the
dark-field zero-guard (`core.py:443-446`) are already wired in both the
scalar `BaSiC` and batched (`_batched_fit.py`) paths, so the A/B is a
clean parameter sweep. D-08 framed this as a convergence-knob but never
ran it as a *speed* lever.

**Caveat (the wide variance):** its expected value is bimodal — ~15-35%
if the flat-field stops drifting within tolerance at the lower pass count,
~0% if the seam gate trips. K01 exists precisely to resolve this. **S02
must run the full seam_l1 + seam_curvature gate; synthetic parity alone is
never sufficient.**

**Suggested first cut:** raise `reweighting_tolerance` to `5e-3` and/or
cap `max_reweighting_iterations=6`, measure pass-count reduction and seam
metrics on a real subject.

### #2 — `sync-cadence` (attempt-now)

**Why it ranks second:** near-zero risk and the easiest possible A/B. The
scalar `convergence_check_every` kwarg already exists
(`inexact_alm_l1`, `_alm.py:290`); tuning it 10→20 cuts host syncs in half
with no numerical change to the converged result (worst case: up to N
extra iterations — slower, not lower quality).

**Why chosen over #3/#5:** same low effort as `fp32-y-accumulator` but
strictly lower seam risk (no precision change at all), and far lower
effort than the CUDA-stream prefetch lever (#5) which needs a profiling
pass. Its ceiling is small (~1-3%) but it is essentially free evidence to
collect alongside #1.

**Caveat:** the *batched* path hardcodes `check_every=10` (`_alm.py:789`)
— raising it there needs a one-line code change and risks running past
convergence by up to N iterations. The scalar kwarg path needs no change.

### Ambitious swing — `amp-selective-fp16-dct` (needs-more-research)

**Why it's the only path to a large fraction of 30%:** it is the sole
lever that attacks the *confirmed* memory-bandwidth bottleneck at its root
(bytes-per-element on the dominant DCT matmul). Halving the dominant
tensor's footprint roughly doubles effective memory throughput on a
bandwidth-bound kernel — a ~30-50% *theoretical* ceiling.

**Why it's not #1:** hard-gated by precision. fp16 (~3.3 decimals) and
bf16 (~2.4) do not survive ALM soft-threshold sensitivity (K02: "~0.001%
σ₁ drift shifts soft-threshold boundaries"). Only a **selective** shape
(DCT matmul in fp16, full fp32 accumulate/reduce) is viable, and its real
ceiling after precision-safe scoping is **unknown without a real-subject
A/B**. If S02 has appetite for one ambitious candidate, this is it — but
it must ship with the full seam gate + a fixed-iteration numerical parity
check (K08), and may not pay out.

---

## 6. Can any lever approach the 30% aspirational target?

**Honest answer: not credibly, as a single lever on the production eager
path.**

1. **The bandwidth-attacking levers are precision-blocked or
   data-rejected.** `amp-selective-fp16-dct` is hard-gated by K02/K06
   (real precision-safe ceiling unknown); `batched-cuda-cache-fit-ws64` is
   already closed by MEM002's real-subject sweep (ratio <1.30 at quality
   parity).
2. **The only high-mechanism-ceiling lever is quality-gated.**
   `reweighting-tolerance` can in principle reach ~15-35%, but *only if*
   the flat-field stops drifting at the lower pass count — wide-variance,
   not a reliable 30%.
3. **Everything else is single-digit or zero.** `sync-cadence` (~1-3%),
   `fp32-y-accumulator` (~3-6%), `cuda-stream-prefetch` (low-moderate) are
   incremental and, stacked optimistically, sum to well under 30%.

**Implication:** S02 should plan for a **~5-10% realistic stack**, not
30%. The 30% target is most likely unreachable from any single remaining
lever; closing the gap would require either re-opening the compile/fusion
path (contradicts D005) or custom Triton kernels (high effort, out of
scope for this spike).

---

## 7. Handoff to S02 — concrete next steps

1. **Run `reweighting-tolerance` A/B first** (zero code change, highest
   ceiling). Vary `reweighting_tolerance` ∈ {2e-3, 5e-3, 1e-2} and
   `max_reweighting_iterations` ∈ {6, 8}. Gate on seam_l1 + seam_curvature
   (K01) + 1.30x speed (D-18). Record pass-count reduction.
2. **Add `sync-cadence` as a free parallel A/B** (scalar kwarg 10→20).
   Near-zero risk; collect the small win alongside #1.
3. **Only if #1/#2 underperform and S02 wants an ambitious swing:**
   prototype `amp-selective-fp16-dct` scoped to the DCT matmul only, with
   fixed-iteration numerical parity (K08) before any real-subject run.
4. **Do not pursue** (rejected with rationale): `svd-power-iter-n-iter`
   (K02 invariant), `warm-start-state-device-resident` (~0% ceiling),
   `batched-cuda-cache-fit-ws64` (MEM002 closed), `tile-subsampling`
   (tuning-search only), and the compile-class levers (inert under eager).

---

## Verification

- **No production source files modified** — this is read-only research
  synthesis. No file under `linum_basic/` was touched.
- All claims are line-cited to the current tree via T01/T02 `gsd_exec`
  grep digests and targeted reads.
- T03 deliverable existence check (task `Verify`):
  `test -s scripts/experiments/s01_synthesis_notes.md` (this file).
- This standalone document IS the slice's research deliverable, satisfying
  all four slice Must-Haves (complete lever catalog with status/ceiling/
  class/seam-risk; codebase-grounded bottleneck reassessment; ranked top
  1-2 shortlist with rationale; no `linum_basic/` source modified).

## Failure Modes

This task has **no external dependencies** — it is a read-only synthesis
of two version-controlled scratch documents (`s01_lever_inventory.md`,
`s01_new_levers.md`) which themselves are static analyses of
version-controlled source. No network, filesystem mutation, subprocess,
or API is involved. The only failure mode is ceiling estimates or line
numbers drifting after future code changes; every estimate in §4 is
anchored to a specific line site so a reviewer can re-evaluate.

## Load Profile

Not applicable — this is a one-shot static synthesis with no runtime load
dimension. There is no request volume to saturate; the artifact is a
fixed-size markdown document consumed by S02 (execution) and S03.

## Negative Tests

Not applicable — T03 produces a research synthesis, not executable code.
There is no malformed-input surface. Factual claims are individually
line-cited so a reviewer can falsify any row; ceiling figures are
explicitly labelled as estimates with stated assumptions (eager production
path, real-subject gate per K01), not as measured results. The one
data-grounded claim (ws=64 rejection, §4 row 13) cites the existing MEM002
real-subject sweep rather than asserting a new measurement.
