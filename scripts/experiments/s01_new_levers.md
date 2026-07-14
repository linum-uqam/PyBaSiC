# S01/T02 — New Levers Survey + Ceiling Estimates (All Levers)

> **Scope:** T02 only. Two deliverables: (1) **new** speed levers visible in
> the current codebase that are **not** in the 7-lever taxonomy inventoried
> in `s01_lever_inventory.md` (T01); (2) an estimated **speed ceiling** for
> every lever — old and new — against the ws=128 steady-state fit-time
> target, with an updated bottleneck classification. No production code is
> modified; this is read-only analysis with line-cited evidence.
>
> **Sister docs:** `s01_lever_inventory.md` (T01 — existing taxonomy),
> `s01_synthesis_notes.md` (T03 — ranked shortlist + bottleneck
> reassessment → feeds `S01-RESEARCH.md`).

---

## 0. The decisive reframing: the production path runs EAGER

T01 catalogued the taxonomy but did not draw out its most important
consequence for this milestone. **`worker-compile-off` (D005) disables
`torch.compile` in CUDA joblib workers** — and that is the production path
(D-14 multi-GPU fan-out). This single fact collapses the ceiling of every
compile-class lever and changes which levers can *plausibly* move fit time:

| Implication | Evidence | Effect on levers |
|-------------|----------|------------------|
| No Inductor → no Triton kernel fusion | `_alm.py:53` `_read_alm_compile_mode`; D005 commit `511c88c` | `compile-shape-stability` (T01 §2.5) and `inductor-cache-warm-policy` (T01 §2.6) are **inert on the production path** — ceiling ~0% |
| TF32 precision only set inside the compiled step | `torch.set_float32_matmul_precision("high")` is gated behind the compile branch (`_alm.py:255` scalar, `:662` batched) | With compile off, the DCT matmul `A_p @ X @ A_q.T` runs in **default float32** — there is no TF32 to "widen". The TF32 half of `compile-surfacing` (T01 §2.2) has **no remaining speed surface**, independent of K06 |
| Eager = one kernel launch per `astype`/`clone`/matmul | 77 `np.float32` casts + 6 `xp.clone` in `_alm.py`; per-iter cast chain at `:187-246` | The inner loop is a **stream of unfused memory passes**. This *is* the memory-bandwidth bottleneck made concrete — but fusing it requires re-enabling compile (contradicts D005) or writing custom Triton kernels (high effort) |

**Takeaway for S02:** any lever whose mechanism depends on Inductor
(compile-shape, inductor-cache, TF32-via-compile, contiguous-layout Triton
selection) has a near-zero ceiling on the production eager path. The only
levers that can move eager fit time are those that (a) reduce the *number*
of ALM solves / reweighting passes, (b) reduce *bytes transferred per
inner iteration*, or (c) overlap transfer with compute. The surveys below
are organised around this triage.

---

## 1. NEW levers not in the taxonomy

Six candidate levers surfaced from the codebase survey. Each is classified
against the `BottleneckClass` taxonomy (T01 §5) and given a ceiling.

### 1.1 `reweighting-tolerance` (open, highest mechanism ceiling) — `core.py`

> Already in the taxonomy (T01 §2.3) but **not** in `_LEVER_DEFINITIONS` as
> a *speed* lever — D-08 framed it as a convergence-knob. Repeated here
> because T02's ceiling analysis elevates it to the top candidate. See §2
> for the ceiling.

### 1.2 `amp-selective-fp16-dct` — NEW — `backend.py` / `_alm.py`

**Mechanism.** Attack the memory-bandwidth bottleneck directly by halving
bytes transferred on the dominant inner-loop op (the 2D DCT matmul
`A_p @ X @ A_q.T`), using a true AMP pattern: fp16 storage/matmul with
fp32 accumulation. The Y/Lagrange accumulator and the σ₁-derived scalars
stay fp32/fp64.

| Element | Evidence |
|---------|----------|
| Dominant op | `_alm.py:155` `_Ap @ x.contiguous() @ _Aq.T` — eager, fp32 |
| Accumulator already split | `_alm.py:496` `Y = Y + xp.astype(mu, np.float64) * xp.astype(dY, np.float64)` — Y is deliberately fp64 |
| Cast density (the bandwidth cost) | 77 fp32 + 17 fp64 `astype` in `_alm.py` inner region (`:187-246`) |

**Bottleneck class:** `memory-bandwidth` (the confirmed dominant class,
MEM001/MEM002/K03). This is the *only* surveyed lever that attacks the
confirmed bottleneck at its root (bytes-per-element).

**Ceiling estimate:** **potentially large in principle (≈30-50% of
inner-loop bandwidth)** because halving the dominant tensor's footprint
roughly doubles effective memory throughput on a bandwidth-bound kernel.
**BUT** this is a *theoretical* ceiling, gated hard by precision:

- K06 forbids globally widening tolerances; K02 marks the σ₁
  power-iteration and Eq.6 dual shrink as invariants where "~0.001% drift
  shifts soft-threshold boundaries."
- fp16 has ~3.3 significant decimals; bf16 ~2.4. Neither survives the
  ALM soft-threshold sensitivity. **Selective** application (DCT matmul
  only, full fp32 accumulate/reduce) is the only viable shape, and its
  real ceiling after precision-safe scoping is unknown without a
  real-subject A/B (K01).

**Recommendation:** **needs-more-research.** Highest upside, highest risk.
If S02 has appetite for one ambitious candidate, this is it — but it must
ship with the full seam_l1 + seam_curvature gate and a fixed-iteration
numerical parity check (K08), and even then the precision-safe scoping may
eat most of the theoretical ceiling.

### 1.3 `fp32-y-accumulator` — NEW — `_alm.py:496`

**Mechanism.** Demote the Lagrange-multiplier accumulator `Y` from fp64 to
fp32, halving its per-iteration memory traffic.

| Element | Evidence |
|---------|----------|
| Current fp64 update | `_alm.py:496` `Y = Y + xp.astype(mu, np.float64) * xp.astype(dY, np.float64)` |
| Footprint | Y is `(n, p*q)` — same shape as `Ir`; at ws=128, p*q = 16384, so for n≈100 images Y is ~6.5 MB in fp64 vs ~3.3 MB in fp32 |

**Bottleneck class:** `memory-bandwidth`.

**Ceiling estimate:** **low-to-moderate (~3-6% of inner-loop bandwidth).**
Y is one of roughly 5-6 full-matrix tensors touched per inner iteration
(D, W, Y, S_spatial, B, plus the DCT intermediates), so halving one is
~⅙ to ~⅕ of the per-iteration memory traffic — and only if accumulation
error over the (typically hundreds of) inner iterations stays within
tolerance. The fp64 promotion was deliberate (stability over long ALM
runs), so this is precision-adjacent to K06 but milder than §1.2.

**Recommendation:** **needs-more-research.** A clean A/B candidate: flip
the two `astype(..., float64)` to float32 at `:496`, run fixed-iteration
parity + real-subject seam gate. Low effort, bounded upside.

### 1.4 `svd-power-iter-n-iter` — NEW (candidate) — `backend.py` — **REJECT**

**Mechanism.** Reduce `n_iter=30` in `_svd_leading_singular_torch_batched`
(`backend.py:537`), since the docstring itself states the uniform initial
vector "converges quickly."

| Element | Evidence |
|---------|----------|
| Default | `backend.py:478,501` `n_iter: int = 30` |
| Call frequency | `_alm.py:418` / `:730` — called **once per ALM solve**, *before* the inner loop |

**Bottleneck class:** `compute` (bmm).

**Ceiling estimate:** **~0%.** The SVD runs once per solve (not per inner
iteration), and each power iteration is a tiny `bmm` on `(n, 16384) @ (16384, 1)`
— ~1.6 Mflop per image. Reducing 30→5 saves ~83% of a negligible,
once-per-solve cost.

**Recommendation:** **reject-with-rationale.** Independently of the
negligible ceiling, K02 lists the power-iteration σ₁ as a **protected
invariant** ("~0.001% σ₁ drift shifts soft-threshold boundaries and breaks
darkfield regression tests"). Do not touch.

### 1.5 `warm-start-state-device-resident` — NEW (candidate) — `_alm.py` — **REJECT (low ceiling)**

**Mechanism.** When `warm_start_reweighting=True` (now the fit default,
`fit.py:578-579`), each reweighting pass round-trips ALM state
GPU→CPU→GPU: `xp.to_numpy(Sf)...` at pack (`:525-528`) then
`xp.asarray(warm_start[...])` at unpack (`:421-424`).

| Element | Evidence |
|---------|----------|
| Pack (device→host) | `_alm.py:525-528` (scalar), `:862-865` (batched) |
| Unpack (host→device) | `_alm.py:421-424` (scalar), `:747-750` (batched) |

**Bottleneck class:** `memory-bandwidth` (host↔device, the slowest kind).

**Ceiling estimate:** **~0%.** The largest packed tensor is `Ir` at
`(n, p*q)` ≈ 6.5 MB for n≈100. At PCIe Gen4 (~32 GB/s shared), a full
round-trip across ~5 reweighting passes is ≈ 65 MB ≈ 2 ms — negligible
against a multi-second-per-z fit. Keeping state device-resident would be a
clean correctness-preserving micro-cleanup but not a speed lever.

**Recommendation:** **reject-with-rationale** as a *speed* lever (low
ceiling). Fine to note as a future code-hygiene cleanup.

### 1.6 `cuda-stream-prefetch-z-overlap` — NEW — `fit.py` / `_parallel.py`

**Mechanism.** Overlap the next z-plane's host→device upload with the
current z-plane's compute via a second CUDA stream, so the
bandwidth-bound compute kernel and the PCIe transfer run concurrently.

| Element | Evidence |
|---------|----------|
| Current z-fanout | `fit.py:394` joblib `Parallel` — one z per worker process, no intra-worker overlap |
| No stream plumbing | `_parallel.py` / `_batched_fit.py` contain no explicit `torch.cuda.Stream` / prefetch |

**Bottleneck class:** `synchronization` + `memory-bandwidth` (overlap).

**Ceiling estimate:** **unknown, bounded low-to-moderate.** This is the
textbook win for bandwidth-bound kernels *if* transfer and compute can
truly overlap. Two factors cap it: (a) the production model is
**multi-process** (one z per joblib worker), so cross-process stream
sharing is impractical — this lever only applies to the
within-process sequential path; (b) if the compute kernel already
saturates device memory bandwidth, concurrent upload competes for the
same bus and yields little. Needs a profile to confirm headroom.

**Recommendation:** **needs-more-research.** Requires a CUDA-stream
profiling pass (K05: always sync when timing) before S02 commits to it.

### 1.7 `batched-cuda-cache-fit-ws64` — NEW (candidate) — `fit.py`/`strategies.py` — **REJECT (data-evaluated)**

**Mechanism.** Revisit batched CUDA at a *smaller* working size where the
working set fits in GPU L2 cache, flipping the kernel from
bandwidth-bound to compute-bound so batching finally wins. Physical
motivation: at ws=128, the sorted-image matrix is `n × 16384 × 4 B` ≈
6.5 MB for n≈100 — **larger than A6000 L2 (4.8 MB)**, which is precisely
why ws=128 is bandwidth-bound (K03). At ws=64, the same matrix is ≈ 1.6 MB
— **fits L2** — potentially making batched CUDA win.

| Element | Evidence |
|---------|----------|
| Production guard | K03 / D009: `strategy="auto"` never selects batched CUDA at ws≥128 |
| The physical threshold | A6000 L2 = 4.8 MB; ws=128 working set ≈ 6.5 MB > L2 |

**Bottleneck class:** `chunking` + `memory-bandwidth`.

**Ceiling estimate:** **N/A — already evaluated and rejected.** MEM002
recorded the A6000 working-size sweep: "ws<128 is not materially faster
(ratio < 1.30) at quality parity on real subject data." The speed *and*
quality answer already exists.

**Recommendation:** **reject-with-rationale.** MEM002 closed this question
on real subjects. Naming it here only so S02 does not re-derive the
cache-fit argument from first principles and mistake it for novel.

---

## 2. Ceiling estimates for ALL levers (old + new)

Unified table. "Ceiling" = realistic *additional* steady-state fit-time
reduction achievable beyond the already-promoted stack
(worker-compile-off + auto-l_s + dct-kernel-tuning ≈ 2.2% reported), under
the production eager multi-GPU path. Ordered by ceiling then by effort.

| # | Lever | Class | In taxonomy? | Status | Estimated ceiling | Effort | Recommendation |
|---|-------|-------|:---:|--------|:---:|:---:|---|
| 1 | `reweighting-tolerance` (`core.py`) | compute (fewer solves) | yes (T01 §2.3) | open, zero code change | **~15-35%** if convergence allows fewer full ALM solves; **~0%** if quality gate trips | low | **attempt-now** (A/B with seam gate) |
| 2 | `amp-selective-fp16-dct` (`backend.py`/`_alm.py`) | memory-bandwidth | **no (NEW §1.2)** | not attempted | **~30-50% theoretical**, precision-gated; real ceiling unknown | high | **needs-more-research** |
| 3 | `fp32-y-accumulator` (`_alm.py:496`) | memory-bandwidth | **no (NEW §1.3)** | not attempted | **~3-6%** | low | **needs-more-research** |
| 4 | `sync-cadence` (`_alm.py`) | synchronization | yes (T01 §2.1) | open, kwarg exists | **~1-3%** (sync latency small vs bandwidth-bound kernel time) | low | **attempt-now** (easy A/B) |
| 5 | `cuda-stream-prefetch-z-overlap` (`fit.py`) | sync + bandwidth | **no (NEW §1.6)** | not attempted | **low-moderate**, needs profile | high | **needs-more-research** |
| 6 | `dct-kernel-tuning` (`backend.py`/`_alm.py`) | compute | yes (T01 §2.4) | **promoted (~2.2%)** | **~0-1%** more (contiguity marginal under eager) | — | done |
| 7 | `compile-surfacing` (`_alm.py`) | compile-shape | yes (T01 §2.2) | promoted (observability) | **~0%** (TF32 not active under eager; K06) | — | done |
| 8 | `compile-shape-stability` (`_alm.py`) | compile-shape | yes (T01 §2.5) | satisfied in code | **~0%** (compile off in production) | — | moot |
| 9 | `inductor-cache-warm-policy` (`_torch_cache.py`) | compile-shape | yes (T01 §2.6) | deferred | **~0%** (harness-only, not production fit) | — | reject |
| 10 | `tile-subsampling` (`tuning.py`) | compute | yes (T01 §2.7) | exists | **0%** for fit-time target (tuning-search only) | — | reject |
| 11 | `svd-power-iter-n-iter` (`backend.py`) | compute | **no (NEW §1.4)** | — | **~0%** (once per solve, tiny) + K02 invariant | — | **reject-with-rationale** |
| 12 | `warm-start-state-device-resident` (`_alm.py`) | memory-bandwidth | **no (NEW §1.5)** | — | **~0%** (~2 ms vs multi-second fit) | — | **reject-with-rationale** |
| 13 | `batched-cuda-cache-fit-ws64` (`fit.py`) | chunking | **no (NEW §1.7)** | — | N/A — **MEM002 already rejected** ws<128 | — | **reject-with-rationale** |

---

## 3. Can any lever approach the 30% aspirational target?

**Honest answer: not credibly, as a single lever on the production eager
path.** The reasoning, which T03 will formalise:

1. **The bandwidth-attacking levers are precision-blocked or data-rejected.**
   The two levers that attack the *confirmed* memory-bandwidth bottleneck
   at its root are `amp-selective-fp16-dct` (§1.2) and
   `batched-cuda-cache-fit-ws64` (§1.7). The former is hard-gated by K06/K02
   precision sensitivity (its real, precision-safe ceiling is unknown and
   may be small); the latter is already closed by MEM002's real-subject
   sweep.

2. **The only high-mechanism-ceiling lever is quality-gated.**
   `reweighting-tolerance` (§1.1/§2 row 1) can in principle cut fit time by
   a full reweighting pass per pass removed — potentially ~15-35% — but
   *only if* the flat-field stops drifting within tolerance at the lower
   pass count. That is exactly what K01's real-subject seam gate exists to
   test, and prior phases found that convergence-count reductions
   frequently trip quality. Its expected value is therefore wide-variance,
   not a reliable 30%.

3. **Everything else is single-digit or zero.** The remaining open levers
   (sync-cadence ~1-3%, fp32-Y ~3-6%, prefetch ~low-moderate) are
   incremental and, even stacked optimistically, sum to well under 30%.

**Implication for S02:** the realistic top-1-to-2 shortlist is
`reweighting-tolerance` (attempt-now, quality-gated, highest mechanism
ceiling) and `sync-cadence` (attempt-now, near-zero risk, easy A/B). If
S02 wants one ambitious swing, `amp-selective-fp16-dct` is the only lever
with a theoretical path to a large fraction of 30%, but it must be scoped
precision-safe and may not pay out. The 30% target is most likely
**unreachable** from any single remaining lever; S02 should plan for a
~5-10% realistic stack, not 30%.

---

## 4. Updated bottleneck classification

The MEM001/MEM002/K03 finding (ws=128 is memory-bandwidth-bound, not
compute-bound) is **confirmed and sharpened** by this survey:

- **Why bandwidth-bound, made concrete:** the eager inner loop is a chain
  of unfused full-matrix passes — 77 fp32 `astype` + 6 `clone` + the DCT
  matmul, each a separate kernel touching `(n, 16384)` tensors larger than
  A6000 L2 (§1.7). There is no fusion because compile is off (D005).
- **Why compile-off is load-bearing anyway:** D005 shipped because the
  compiled path caused an Inductor CPU compile storm in short-lived joblib
  workers (per-z fit dropped from ~130 s+ to ~3-6 s/z). Re-enabling
  compile to get fusion back would regress far more than any fusion gain
  — so the bandwidth-bound eager path is the *intentional* steady state,
  not an accident.
- **Net:** the bottleneck is `memory-bandwidth`, *and* the obvious fix
  (compile/fusion) is structurally closed. This is why the realistic
  ceiling across all remaining levers is modest (§3) — the system is
  near its achievable bandwidth efficiency already, and the levers that
  could change the byte budget are either precision-blocked or
  data-rejected.

---

## 5. Verification

- **No production source files modified** (read-only research). All
  evidence is line-cited to the current tree via `gsd_exec` grep digests
  (`.gsd/exec/3e674540-...stdout`, `.gsd/exec/4f5e5470-...stdout`) and the
  T01 inventory.
- T02 deliverable existence check is the task's `Verify` command:
  `test -s scripts/experiments/s01_new_levers.md` (this file).

## Failure Modes

This task has **no external dependencies** — it is a static read-only
analysis of version-controlled source files (`_alm.py`, `backend.py`,
`_batched_fit.py`, `benchmark/strategies.py`, `core.py`, `fit.py`,
`_parallel.py`, `_torch_cache.py`, `tuning.py`) cross-referenced against
the tracked T01 inventory and `.gsd/KNOWLEDGE.md` invariants. No network,
filesystem mutation, subprocess, or API is involved. The only failure mode
is ceiling estimates drifting after future code changes; §0 and the §2
"Evidence" columns anchor each estimate to specific line sites so a
reviewer can re-evaluate.

## Load Profile

Not applicable — this is a one-shot static analysis with no runtime load
dimension. There is no request volume to saturate; the artifact is a
fixed-size markdown file consumed by T03 (synthesis) and S02 (execution).

## Negative Tests

Not applicable — T02 produces a lever survey with ceiling estimates, not
executable code. There is no malformed-input surface. The analytical
claims are individually line-cited so a reviewer can falsify any row;
the ceiling estimates are explicitly labelled as estimates with stated
assumptions (eager production path, real-subject gate per K01), not as
measured results. The one claim that *is* data-grounded rather than
estimated — the ws=64 rejection (§1.7) — cites the existing MEM002
real-subject sweep rather than asserting a new measurement.
