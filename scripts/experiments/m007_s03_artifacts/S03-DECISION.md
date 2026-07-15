# S03 Decision: Go / No-Go on Building a Terabyte-scale Distributed/Cluster Streaming Redesign

> **Milestone M007 / Slice S03 — terminal decision artifact.**
> This document closes M007/S03 with a defensible go/no-go verdict on the
> slice's single question: *does production-scale evidence justify building
> a terabyte-scale distributed/cluster streaming redesign (coordination
> layer, multi-host worker model, R056) now, or should it be re-deferred?*
> It does **not** produce new measurements — the measurement was S01's job
> and is already on record. It synthesises the existing, version-controlled
> evidence chain into an independently auditable closure for the
> distributed-redesign question specifically, so a future agent or operator
> can audit *why no coordination/cluster layer was built* from this one file
> without re-deriving the decision from S01, S02, and D022.
>
> **Evidence chain.** This decision is a read-only synthesis of:
> - **D022** (`.gsd/DECISIONS.md`) — the cross-milestone decision record that
>   closed the R051 operator follow-up and re-deferred S02 and S03.
> - **`scripts/experiments/s01_artifacts/S01-DECISION.md`** — the terminal
>   S01 decision that measured production-scale peak memory and first recorded
>   the S03 re-deferral (its §5).
> - **`scripts/experiments/s01_artifacts/{eager,streaming}-probe.json`** — the
>   raw OS-level peak-memory artifacts the verdict is computed from.
> - **`scripts/experiments/m007_s02_artifacts/S02-DECISION.md`** — the sibling
>   terminal decision (write-path) whose citation pattern this file follows.
> - **R056** (`.gsd/REQUIREMENTS.md`) — the terabyte-scale streaming redesign
>   requirement, currently `deferred` with note *"v2.0 delivers MVP only."*
> - **`linum_basic/_parallel.py`** and **`linum_basic/fit.py`** — read-only
>   reference confirming today's parallelism model is single-host only
>   (joblib process pool + CUDA multi-GPU fan-out); no coordination, cluster,
>   scheduler, or multi-host worker layer exists.
>
> **Pattern.** This follows the terminal-slice decision-artifact pattern
> established by `scripts/experiments/s03_artifacts/S03-DECISION.md`
> (M006/R058), `scripts/experiments/s01_artifacts/S01-DECISION.md`
> (M007/S01), and `scripts/experiments/m007_s02_artifacts/S02-DECISION.md`
> (M007/S02): a read-only synthesis that records an explicit go/no-go and the
> exact, falsifiable condition under which to revisit it.
>
> **Read-only synthesis of evidence.** This task changes **no**
> `linum_basic/` production code, no BaSiC invariant (K02), no default, and
> no algorithm. Its only write is this document plus the `docs/streaming.md`
> cross-reference (T02). No coordination/cluster/distributed code is
> introduced, by design: the evidence does not earn it.

---

## Verdict (TL;DR)

### **NOT JUSTIFIED — the terabyte-scale distributed/cluster streaming redesign is re-deferred. No coordination layer or multi-host worker model is built in M007. The codebase stays single-host.**

The distributed redesign would add a coordination layer and a multi-host
worker model so a cluster of hosts could jointly process a volume too large
for any single host. M007's vision gates this slice on S02 completing and
on real evidence showing a cluster-scale target exists. S01 measured the
largest available real per-z mosaic volume and the gate is **not met**: no
workload approaches — let alone exceeds — single-host memory budget, so a
cluster layer would unlock work that does not otherwise fail.

### Why the gate is not met (the four independent reasons)

1. **No terabyte-scale target exists.** The largest available real per-z
   mosaic volume peaks at **2.40 GB** host RSS under the non-streaming
   (eager) path — **~416x below 1 terabyte** (1e12 B) and **~4.66%** of a
   single A6000's 48 GiB host-RAM budget (**~21.5x headroom**). There is no
   production volume in the evidence base that remotely approaches, let alone
   exceeds, single-host capacity. The streaming read path drops that further
   to 1.79 GB (**~3.46%** of budget, **~28.9x headroom**).
2. **The codebase is single-host today, by design.** `linum_basic/_parallel.py`
   exports only `parallel_map` (a joblib process pool) and
   `parallel_map_cuda_devices` (a single-host multi-GPU fan-out setting
   `CUDA_VISIBLE_DEVICES` per joblib process). `linum_basic/fit.py`'s
   `strategy="multi"` is exactly this single-host multi-GPU fan-out — it is
   **not** multi-host. There is no coordinator, scheduler, message broker, or
   multi-host worker abstraction anywhere in `linum_basic/`. A distributed
   layer is a large scope/dependency expansion (coordination, scheduling,
   fault-tolerance, deployment) that the single-host codebase does not carry.
3. **The streaming MVP already removes the single-host read-path ceiling.**
   Because streaming bounds host RSS to ~one z-plane's tile stack, a single
   A6000 can scale to far larger `n_z` than measured before approaching host
   memory limits — further reducing the near-term need for distribution. The
   measured single-host peak (2.40 GB eager, 1.79 GB streaming) is far below
   the 48 GB A6000 budget, so no single-host workload fails today.
4. **The roadmap gate is explicit.** M007's vision: *"the codebase is
   single-host today and a cluster layer is a large scope/dependency
   expansion that must be earned by evidence."* S01 measured the largest
   available real per-z mosaic volume (`sub-22` slice 27, `n_z=55`) and found
   it fits comfortably in single-host memory. The gate is not met.

### Disposition

- **S03 stays a `[sketch]`.** No distributed/cluster implementation in M007.
- **The single-host parallelism model stands:** `linum_basic/_parallel.py`
  remains joblib processes (NumPy backend) + single-host CUDA multi-GPU
  fan-out (Torch backend); `fit.py`'s `strategy="multi"` remains single-host
  multi-GPU, unchanged.
- **R056 stays `deferred`** (re-deferred, not invalidated): *"v2.0 delivers
  MVP only"* remains its note. This evaluation corroborates that note at
  production scale and records the falsifiable revisit condition (§6).
- **The documented MVP boundary stands:** `docs/streaming.md` records "Scale |
  RAM-bounded, not TB-grade", now cross-referenced to this evaluation (T02)
  so a fresh reader sees it was *evaluated and deferred*, not overlooked.
- **Milestone closure:** S03 is the milestone's last planned slice; its
  completion *is* this documented re-defer, exactly as the milestone's
  evidence-gated structure intends.

---

## 1. The decision question and what "go" would require

M007/S03 exists to answer one question (per the roadmap and its must-haves):

> *Does the production-scale evidence justify building a terabyte-scale
> distributed/cluster streaming redesign (coordination layer, multi-host
> worker model, R056) now, or should it be re-deferred?*

"Go" would mean: evidence that a real production workload exceeds the memory
(or wall-time) budget of the project's largest single host *even with
streaming enabled*, so that a distributed/cluster layer would actually
unlock work that otherwise fails. "No-go" means keeping the single-host
parallelism model and documenting why, per the milestone's evidence-gate
design.

This is the broadest of M007's three questions. S01 had to *measure*; S02
had to *close the write-path* (a single-process memory problem); S03 has to
*close the cluster-scaling* question (a multi-host distribution problem).
S01 already produced the decisive measurement (the two OS-process
peak-memory probes); D022 already recorded the cross-milestone verdict;
S01's §5 and S02's §5 both recorded S03 as NOT JUSTIFIED in passing. S03's
contribution is the terminal, distributed-redesign-specific closure artifact
this file is — it makes the "not justified" verdict for THIS slice's
question independently auditable rather than buried inside S01's §5.

---

## 2. The evidence chain (cited, not re-measured)

The verdict rests on S01's production-scale measurement, recorded in
`S01-DECISION.md` and durable across milestones in D022. The numbers below
are the cited figures; every one was recomputed directly from the raw JSON
artifacts in this task (internal-consistency check, §Verification).

| metric | eager (non-streaming) | streaming | delta | relevance to the cluster question |
|---|---|---|---|---|
| **peak host RSS** | **2,402,631,680 B (2.40 GB)** | **1,785,761,792 B (1.79 GB)** | **-616,869,888 B (-25.7%)** | the per-host peak a cluster layer would have to exceed |
| peak GPU VRAM | 408,166,912 B (389 MiB) | 408,166,912 B (389 MiB) | **0 - identical** | per-host GPU working set (unchanged by streaming or distribution axis) |
| wall time | 17,733 ms | 19,529 ms | +1,796 ms (+10.1%) | single-host wall time; no SLA pressure measured |
| **headroom vs 48 GiB budget** | **4.66% used (21.5x)** | **3.46% used (28.9x)** | — | the headroom a cluster would need to relieve |
| **gap to 1 terabyte** | **416x below 1 TB** | — | — | the scale gap a cluster redesign is supposed to bridge |

Measurement provenance (identical across the two probes, so the comparison is
apples-to-apples): subject `sub-22` slice 27, `n_z=55`, z-sample
`[0, 13, 27, 40, 54]`, `working_size=128`, `backend=torch`, `device=cuda:0`,
`strategy=sequential`, `estimate_darkfield=true`, `overlap=0.2`,
`max_reweighting_iterations=15`, `git_commit=4e924e0`, A6000
`sn4622125853`. OS-level peak RSS via `resource.getrusage(RUSAGE_SELF)
.ru_maxrss` (sees the native buffers `tracemalloc` cannot). See
`S01-DECISION.md` §2-§3 for the full method.

**The decisive figure for the cluster question is the gap to single-host
capacity, not the streaming delta.** A distributed/cluster layer earns its
large scope/dependency cost only when a workload genuinely cannot run on the
project's largest single host. The measured single-host peak (2.40 GB eager,
1.79 GB streaming) is ~21-29x below the 48 GiB A6000 budget and ~416x below
terabyte scale. The gap a cluster would bridge does not exist in the evidence
base.

---

## 3. Today's parallelism model is single-host (read-only confirmation)

A cluster layer would be an addition on top of today's model. Today's model,
confirmed by read-only inspection of `linum_basic/_parallel.py` and
`linum_basic/fit.py`, is single-host only:

| Surface | What it is | Scope |
|---|---|---|
| `parallel_map()` | joblib `loky` process pool, `inner_max_num_threads=1` to avoid BLAS oversubscription | single host, CPU processes |
| `parallel_map_cuda_devices()` | joblib processes, each with `CUDA_VISIBLE_DEVICES` set to one GPU index, round-robin | **single host**, multi-GPU |
| `resolve_workers()` / `is_gpu_backend()` | collapse worker count for single-accelerator contention; CUDA-only (MPS excluded per K04) | single host |
| `fit.py` `strategy="multi"` | calls `parallel_map_cuda_devices` — single-host multi-GPU fan-out (the production concurrency fallback per MEM001/D-14) | **single host**, multi-GPU |
| `fit.py` `strategy="batched"` | single-process batched CUDA solve over z-planes | single host, single GPU |

There is **no** coordinator, scheduler, message broker, RPC, shared/filesystem
job queue, or multi-host worker abstraction in `linum_basic/`. The two
multi-GPU paths are explicitly single-host (they set `CUDA_VISIBLE_DEVICES`
per process on one machine). This is the correct design for the measured
workload — and it is exactly why a cluster layer is a scope/dependency
expansion that must be *earned by evidence*, per the M007 vision.

---

## 4. Why this is a defensible NOT JUSTIFIED

| Question | Answer |
|---|---|
| Is there evidence a real workload exceeds the largest single host's memory budget? | **No.** The largest real volume peaks at 2.40 GB — 21.5x below the 48 GiB A6000 budget. |
| Is there evidence a real workload approaches terabyte scale? | **No.** 2.40 GB is ~416x below 1 terabyte. No production volume in the evidence base is remotely cluster-scale. |
| Does the streaming MVP change the cluster verdict? | **It strengthens no-go.** Streaming bounds host RSS to ~one z-plane, so a single host scales to far larger `n_z` before approaching limits — pushing the cluster breakpoint further out. |
| Does the roadmap gate allow building now? | **No.** The gate requires evidence a cluster layer is earned; the evidence points the other way. |
| Would a cluster layer unlock work that otherwise fails? | **No.** No measured or target production workload fails to run single-host. |
| Is the scope/dependency cost proportionate? | **No.** A distributed layer adds coordination, scheduling, fault-tolerance, and deployment dependencies the single-host codebase does not carry — disproportionate to a non-existent need. |

This is a legitimate "measurement closes the question" outcome. S03 earns
its keep by preventing premature over-engineering — by formally recording
that the evidence points away from a build — not by unlocking new code. The
alternative (silently skipping S03) would leave the distributed-redesign
question implicitly unanswered and force a future agent to re-derive the
verdict from S01's §5 and D022; this file makes the answer explicit and
auditable.

---

## 5. What this slice proves and what it does not

**Proves (proof level: operational):** M007/S03 operationally closes the
distributed-redesign evaluation with a terminal, auditable decision. The
NOT-JUSTIFIED verdict is reproducible from the version-controlled evidence
chain (`S01-DECISION.md` + `{eager,streaming}-probe.json` + D022) and the
read-only confirmation that today's parallelism model is single-host, without
re-running the A6000 job. A future agent can audit why no coordination/cluster
layer was built from this file plus `docs/streaming.md`'s cross-reference
(T02).

**Does not prove / does not do:**
- It does **not** produce new measurements — S01 owns the measurement; this
  slice only synthesises it for the distributed-redesign question.
- It does **not** change any production code path, default, invariant (K02),
  or the streaming read MVP (which shipped in M003/S02 and was measured at
  production scale in M007/S01).
- It does **not** rule out a distributed/cluster redesign forever — it rules
  it out *on the current evidence*. The revisit condition is stated in §6.
- It does **not** address the sibling write-path question (S02), which has
  its own terminal decision (`S02-DECISION.md`) with a narrower revisit
  condition.
- It does **not** exercise the streaming MVP at true terabyte scale (no such
  volume exists in the project); `n_z=55` with `z-sample=5` is the largest
  real per-z mosaic volume available. Extrapolation to terabyte scale is
  qualitative, not measured.

---

## 6. The exact, falsifiable revisit condition

> **Build the terabyte-scale distributed/cluster streaming redesign when:** a
> real production workload genuinely exceeds the memory (or wall-time) budget
> of the project's largest single host *even with streaming enabled* — e.g. a
> multi-terabyte corpus that must be processed under a fixed wall-time SLA,
> or a host-RAM ceiling that streaming cannot keep a single z-plane's tile
> stack under — **or** an operator states a requirement to process volumes
> across multiple hosts. Until then, the single-host parallelism model
> (joblib processes + CUDA multi-GPU fan-out) is correct and sufficient.

This condition is falsifiable and forward-looking: it names the concrete
signal (a workload that does not fit on the largest single host even with
streaming, or an explicit multi-host operator requirement) that would flip
the verdict, not a vague "when volumes get bigger." It is deliberately
broader than S02's revisit condition (an output write that does not fit in
host RAM), because the cluster question is a multi-host scaling problem, not
a single-process memory problem.

---

## Verification

- **Internal consistency (gsd_exec e125af7d, exit 0, 377 ms):** every numeric
  claim in §2 was recomputed directly from
  `scripts/experiments/s01_artifacts/eager-probe.json` and
  `streaming-probe.json`: eager RSS `2,402,631,680 B` (2.40 GB / 2.2376 GiB),
  streaming RSS `1,785,761,792 B` (1.79 GB / 1.6631 GiB), VRAM `identical`
  (408,166,912 B both modes), wall delta `+1,796.266 ms (+10.129%)`, eager at
  **4.66% of the 48 GiB budget** (21.5x headroom), streaming at **3.46%**
  (28.9x headroom), and **1 TB / eager peak = 416x** — the headline scale-gap
  figure. The config-equality check reported the two JSONs differ only in
  `peak_rss_bytes`/`wall_ms` (the measured outputs) plus the intended
  `mode`/`streaming`/`lazy_load` fields — confirming the comparison is clean.
- **Single-host parallelism model confirmed (read-only inspection):**
  `linum_basic/_parallel.py` exports only `default_workers`, `is_gpu_backend`,
  `list_cuda_devices`, `parallel_map`, `parallel_map_cuda_devices`,
  `resolve_workers` — all single-host (joblib process pool + single-host CUDA
  multi-GPU fan-out via `CUDA_VISIBLE_DEVICES`). A grep for
  `cluster|coordinator|distributed|worker model|ray|dask|mpi|celery|scheduler`
  across `linum_basic/` found **no** coordination/cluster/distributed
  abstraction (matches were docstring references to joblib's "scheduler" and
  unrelated `torch.compile`/`re.compile` substrings). `fit.py`'s
  `strategy="multi"` is `parallel_map_cuda_devices` — single-host multi-GPU,
  not multi-host.
- **Evidence-chain presence:** all cited artifacts exist and are non-empty —
  `S01-DECISION.md` (380 lines), `S02-DECISION.md` (sibling precedent, 16 KB),
  `eager-probe.json` (1003 B), `streaming-probe.json` (1005 B); D022 is
  present in `.gsd/DECISIONS.md`; R056 is present in `.gsd/REQUIREMENTS.md`
  (status `deferred`, note *"v2.0 delivers MVP only"*).
- **No code change to verify:** this task writes only this document (the
  `docs/streaming.md` cross-reference is T02, a separate task); no test
  suite, lint, or typecheck is affected by T01.

---

## Failure Modes

This task has **no external runtime dependencies** — it is a read-only
synthesis of version-controlled evidence artifacts (`S01-DECISION.md`,
`S02-DECISION.md`, `eager-probe.json`, `streaming-probe.json`, D022, R056)
plus a read-only confirmation that the parallelism model in
`linum_basic/_parallel.py` and `linum_basic/fit.py` is single-host. The only
failure mode is a claim drifting from its cited evidence; this is guarded by
the §Verification internal-consistency recompute (`gsd_exec e125af7d`), which
recomputes every figure directly from the raw JSONs (including the 416x
scale-gap and the 4.66%/3.46% budget-utilisation figures), and by the
single-host-model inspection that confirms no coordination/cluster layer
exists. The decision document invokes no API, network, subprocess, or GPU at
runtime — it is a static markdown artifact consumed by
`docs/streaming.md`'s cross-reference (T02) and by any future milestone that
re-evaluates the distributed redesign.

## Load Profile

Not applicable — this is a one-shot decision synthesis with no runtime load
dimension. There is no request volume to saturate; the artifact is a
fixed-size markdown document, produced once and consumed by the `docs/`
cross-reference and the M007 roadmap. No compute, GPU, concurrency, or I/O
is exercised by the synthesis itself. (The probes that *generated* the
underlying evidence were one-shot operator runs on the A6000, already
completed in M007/S01; they are inputs to this task, not a load it imposes.)

## Negative Tests

The decision's falsification surface is its evidence chain, locked by the
existing measurement and streaming test suites. This task adds no executable
code, so it adds no new negative test; its negative surface is the
measurement contract that the verdict rests on, which is already covered:
- `tests/test_streaming_memory_probe.py` — the probe's synthetic contract
  test proves the JSON schema is well-formed and peak values are
  non-negative for both modes; it would fail if the probe regressed in a way
  that produced malformed or negative-peak artifacts. Re-runnable locally
  (no CUDA) via `uv run pytest tests/test_streaming_memory_probe.py -v`.
- `tests/test_streaming_fit.py::TestStreamingNumericsParity` (M003/S02) —
  locks the bit-identical-numerics premise the evidence chain relies on
  (streaming changes memory, not math).
- The §Verification internal-consistency recompute (re-runnable via the
  `gsd_exec` recompute against the two raw JSONs) — would surface any future
  drift that made the cited figures stale (e.g. the 416x scale-gap or the
  budget-utilisation percentages), which would invalidate the NOT-JUSTIFIED
  verdict and trigger the §6 revisit condition.
