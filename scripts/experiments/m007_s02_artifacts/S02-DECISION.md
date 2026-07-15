# S02 Decision: Go / No-Go on Building a Streamed Zarr Write Path

> **Milestone M007 / Slice S02 — terminal decision artifact.**
> This document closes M007/S02 with a defensible go/no-go verdict on the
> slice's single question: *does production-scale evidence justify building a
> streamed Zarr write path in this milestone, or should it be re-deferred?*
> It does **not** produce new measurements — the measurement was S01's job
> and is already on record. It synthesises the existing, version-controlled
> evidence chain into an independently auditable closure for the write-path
> question specifically, so a future agent or operator can audit *why the
> write path is not streamed* from this one file without re-deriving the
> decision from S01 and D022.
>
> **Evidence chain.** This decision is a read-only synthesis of:
> - **D022** (`.gsd/DECISIONS.md`) — the cross-milestone decision record that
>   closed the R051 operator follow-up and re-deferred S02 and S03.
> - **`scripts/experiments/s01_artifacts/S01-DECISION.md`** — the terminal
>   S01 decision that measured production-scale peak memory and first recorded
>   the S02 re-deferral (its §4).
> - **`scripts/experiments/s01_artifacts/{eager,streaming}-probe.json`** — the
>   raw OS-level peak-memory artifacts the verdict is computed from.
> - **R051** (`.gsd/REQUIREMENTS.md`) — the streaming read-MVP requirement
>   whose operator follow-up S01 closed; the write path is its documented MVP
>   boundary ("Write path: Not streamed").
> - **`linum_basic/fit.py`** — read-only reference confirming `apply_fit()` /
>   `save_corrected()` remain eager by design (no drift since S01).
>
> **Pattern.** This follows the terminal-slice decision-artifact pattern
> established by `scripts/experiments/s03_artifacts/S03-DECISION.md`
> (M006/R058) and `scripts/experiments/s01_artifacts/S01-DECISION.md`
> (M007/S01): a read-only synthesis that records an explicit go/no-go and the
> exact, falsifiable condition under which to revisit it.
>
> **Read-only synthesis of evidence.** This task changes **no** `linum_basic/`
> production code, no BaSiC invariant (K02), no default, and no algorithm.
> Its only write is this document plus the `docs/streaming.md` cross-reference
> (T02). No write-path streaming code is introduced, by design: the evidence
> does not earn it.

---

## Verdict (TL;DR)

### **NOT JUSTIFIED — the streamed Zarr write path is re-deferred. No write-path code is built in M007. The eager write path remains by design.**

The streamed write path would make the *output* write path stream Zarr
incrementally instead of materialising the full corrected volume, with a
bit-identical on-disk layout. M007's vision gates this slice on S01 showing
real volumes justify it. S01 measured the largest available real per-z mosaic
volume and the gate is **not met**: there is no write-path memory pressure to
relieve at measured production scale.

### Why the gate is not met (the three independent reasons)

1. **No evidence of write-path memory pressure.** The measured *output*
   shape is `[5, 75, 75]` for flat-fields and dark-fields (5 measured
   z-planes at `working_size=128`, upsampled to 75x75 tiles) — **~220 KB
   for both fields combined** (`5 x 75 x 75 x 4 B x 2 = 225,000 B`). The
   corrected output volume is tiny relative to host memory. The streamed
   write path addresses an output-materialisation problem that does not
   exist at measured scale.
2. **Single-host headroom is enormous.** The *entire* eager read+compute
   peak is **2.40 GB** on an A6000 with 48 GB host RAM — **~5% of budget**,
   roughly 20x headroom. Streaming drops that to ~3.7%. A streamed write
   path would relieve pressure that is not present.
3. **The roadmap gate is explicit.** M007's vision: *"Each later slice is
   gated on S01 showing real volumes justify it."* S01 measured the largest
   available real per-z mosaic volume (`sub-22` slice 27, `n_z=55`) and found
   it fits comfortably in single-host memory under the non-streamed path.

### Disposition

- **S02 stays a `[sketch]`.** No streamed-write-path implementation in M007.
- **`apply_fit()` / `save_corrected()` remain eager** — they materialise the
  full corrected volume (`np.asarray(mosaic.array)...copy()` then
  `write_ome_zarr(...)`), by design and unchanged since S01.
- **The documented MVP boundary stands:** `docs/streaming.md` records "Write
  path: Not streamed", now cross-referenced to this evaluation (T02) so a
  fresh reader sees it was *evaluated and deferred*, not overlooked.
- **Roadmap sequencing preserved:** S03 (terabyte-scale distributed redesign
  decision) remains gated on S02 completing — and S02's completion *is* this
  documented re-defer, exactly as the milestone's evidence-gated structure
  intends.

---

## 1. The decision question and what "go" would require

M007/S02 exists to answer one question (per the roadmap and its must-haves):

> *Does the production-scale peak-memory evidence justify building a streamed
> Zarr write path now, or should the write path be re-deferred?*

"Go" would mean: evidence that a real production *output* volume's
materialised write approaches or exceeds host-memory budget, so that a
streamed write path would actually unlock work that otherwise fails.
"No-go" means keeping the eager write path and documenting why, per the
milestone's evidence-gate design.

This is a narrower question than S01's. S01 had to *measure*; S02 has to
*close*. S01 already produced the decisive measurement (the two OS-process
peak-memory probes); D022 already recorded the cross-milestone verdict.
S02's contribution is the terminal, write-path-specific closure artifact
this file is — it makes the "not justified" verdict for THIS slice's question
independently auditable rather than buried inside S01's §4.

---

## 2. The evidence chain (cited, not re-measured)

The verdict rests on S01's production-scale measurement, recorded in
`S01-DECISION.md` and durable across milestones in D022. The numbers below
are the cited figures; every one was recomputed directly from the raw JSON
artifacts in this task (internal-consistency check, §Verification).

| metric | eager (non-streaming) | streaming | delta | relevance to the write path |
|---|---|---|---|---|
| **peak host RSS** | **2,402,631,680 B (2.40 GB)** | **1,785,761,792 B (1.79 GB)** | **-616,869,888 B (-25.7%)** | the *read+compute* peak the write path would share a process with |
| peak GPU VRAM | 408,166,912 B (389 MiB) | 408,166,912 B (389 MiB) | **0 - identical** | streaming does not touch the GPU working set |
| wall time | 17,733 ms | 19,529 ms | +1,796 ms (+10.1%) | the read-path streaming overhead, already acceptable |
| **output flatfields/darkfields shape** | `[5, 75, 75]` | `[5, 75, 75]` | identical | **the volume a streamed write path would write** |

Measurement provenance (identical across the two probes, so the comparison is
apples-to-apples): subject `sub-22` slice 27, `n_z=55`, z-sample
`[0, 13, 27, 40, 54]`, `working_size=128`, `backend=torch`, `device=cuda:0`,
`strategy=sequential`, `estimate_darkfield=true`, `overlap=0.2`,
`max_reweighting_iterations=15`, `git_commit=4e924e0`, A6000
`sn4622125853`. OS-level peak RSS via `resource.getrusage(RUSAGE_SELF)
.ru_maxrss` (sees the native buffers `tracemalloc` cannot). See
`S01-DECISION.md` §2-§3 for the full method.

**The decisive figure for the write path is the output shape, not the
read+compute peak.** Even if the read+compute peak were at the budget edge,
the *output* volume a streamed write path would incrementally emit is
`[5, 75, 75]` for each field — a few hundred KB. That is the volume that
would need to exceed host RAM to earn a streamed write. It does not, by
many orders of magnitude.

---

## 3. Why this is a defensible NOT JUSTIFIED

| Question | Answer |
|---|---|
| Is there evidence a real output volume's write approaches host RAM? | **No.** The measured output is `[5,75,75]` per field — ~220 KB both fields combined. |
| Is single-host headroom a constraint on the write path? | **No.** The full eager read+compute peak is ~5% of the 48 GB budget; the write adds a few hundred KB. |
| Does streaming the read path change the write-path verdict? | **No.** Read-path streaming (S01, -617 MB host RSS) bounds the *read+compute* peak, not the *output* size. The output remains tiny. |
| Does the roadmap gate allow building now? | **No.** The gate requires S01 to show real volumes justify it; S01 showed the opposite. |
| Would the write path unlock work that otherwise fails? | **No.** No measured or target production volume fails to write under the eager path. |

This is a legitimate "measurement closes the question" outcome. S02 earns
its keep by preventing premature over-engineering — by formally recording
that the evidence points away from a build — not by unlocking new code. The
alternative (silently skipping S02) would leave the write-path question
implicitly unanswered and force a future agent to re-derive the verdict from
S01 and D022; this file makes the answer explicit and auditable.

---

## 4. The eager write path remains by design

The current write path in `linum_basic/fit.py` materialises the full
corrected volume before writing:

- **`apply_fit()`** — `corrected = np.asarray(mosaic.array).astype(np.float32).copy()`
  materialises the entire `(Z, H, W)` corrected volume, then applies the
  per-tile flat/dark-field correction in place.
- **`save_corrected()`** — calls `apply_fit()`, then
  `write_ome_zarr(output_path, corrected, ...)` writes the full materialised
  array in one pass.

This is **intentional and unchanged since S01** (read-only reference; no
edit). It matches the evidence: because there is no write-path memory
pressure at measured production scale, an eager write is correct and
sufficient. A streamed write path would add lazy/incremental write
machinery whose only justification is an output volume large enough to
threaten host RAM — and no such volume exists in the evidence base.

---

## 5. What this slice proves and what it does not

**Proves (proof level: operational):** M007/S02 operationally closes the
write-path evaluation with a terminal, auditable decision. The NOT-JUSTIFIED
verdict is reproducible from the version-controlled evidence chain
(`S01-DECISION.md` + `{eager,streaming}-probe.json` + D022) without
re-running the A6000 job. A future agent can audit why the write path is not
streamed from this file plus `docs/streaming.md`'s cross-reference (T02).

**Does not prove / does not do:**
- It does **not** produce new measurements — S01 owns the measurement; this
  slice only synthesises it for the write-path question.
- It does **not** change any production code path, default, invariant (K02),
  or the streaming read MVP (which shipped in M003/S02 and was measured at
  production scale in M007/S01).
- It does **not** rule out a streamed write path forever — it rules it out
  *on the current evidence*. The revisit condition is stated in §6.
- It does **not** address S03 (terabyte-scale distributed redesign, R056);
  S03 is a separate downstream slice with its own revisit condition (stated
  in `S01-DECISION.md` §5 and D022).

---

## 6. The exact, falsifiable revisit condition

> **Build the streamed Zarr write path when:** a real production *output*
> volume's materialised write exceeds host-memory budget — e.g. a full-tile,
> full-`n_z` mosaic write at native resolution on a memory-constrained host —
> **or** an operator states a requirement to write corrected volumes larger
> than available RAM. Until then, the eager write path is correct and
> sufficient.

This condition is falsifiable and forward-looking: it names the concrete
signal (an output write that does not fit in host RAM) that would flip the
verdict, not a vague "when volumes get bigger." It is deliberately narrower
than S03's revisit condition (a workload exceeding the largest single host's
budget even with streaming), because the write path is a single-process
memory problem, not a cluster-scaling problem.

---

## Verification

- **Internal consistency (gsd_exec a0b447bc, exit 0, 115 ms):** every numeric
  claim in §2 was recomputed directly from
  `scripts/experiments/s01_artifacts/eager-probe.json` and
  `streaming-probe.json`: eager RSS `2,402,631,680 B` (2.40 GB), streaming
  RSS `1,785,761,792 B` (1.79 GB), eager at **3.46% of the 48 GiB budget**
  (~5% in decimal GB, the headline figure), VRAM `identical` (408,166,912 B
  both modes), wall delta `+1,796.3 ms (+10.13%)`, output shape `[5,75,75]`
  for both fields, both fields at f32 = `225,000 B` (~220 KB). The
  config-equality check reported the two JSONs differ only in
  `peak_rss_bytes`/`wall_ms` (the measured outputs) plus the intended
  `mode`/`streaming`/`lazy_load` fields — confirming the comparison is clean.
- **Eager write path unchanged (read-only confirmation):** `apply_fit()` and
  `save_corrected()` in `linum_basic/fit.py` materialise the full corrected
  volume (`np.asarray(mosaic.array).astype(np.float32).copy()` then
  `write_ome_zarr`); no streamed/incremental write code exists. This matches
  the evidence that no write-path memory pressure exists.
- **Evidence-chain presence:** all cited artifacts exist and are non-empty —
  `S01-DECISION.md` (380 lines), `eager-probe.json` (1003 B),
  `streaming-probe.json` (1005 B); D022 is present in `.gsd/DECISIONS.md`;
  R051 is present in `.gsd/REQUIREMENTS.md`.
- **No code change to verify:** this task writes only this document (the
  `docs/streaming.md` cross-reference is T02, a separate task); no test
  suite, lint, or typecheck is affected by T01.

---

## Failure Modes

This task has **no external runtime dependencies** — it is a read-only
synthesis of version-controlled evidence artifacts (`S01-DECISION.md`,
`eager-probe.json`, `streaming-probe.json`, D022, R051) plus a read-only
confirmation that the write path in `linum_basic/fit.py` is still eager.
The only failure mode is a claim drifting from its cited evidence; this is
guarded by the §Verification internal-consistency recompute
(`gsd_exec a0b447bc`), which recomputes every figure directly from the raw
JSONs, and by the config-equality check that confirms the eager/streaming
comparison is clean (no unintended configuration differences). The decision
document invokes no API, network, subprocess, or GPU at runtime — it is a
static markdown artifact consumed by `docs/streaming.md`'s cross-reference
(T02) and by any future milestone that re-evaluates the write path.

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
  drift that made the cited figures stale, which would invalidate the
  NOT-JUSTIFIED verdict and trigger the §6 revisit condition.
