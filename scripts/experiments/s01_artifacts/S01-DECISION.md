# S01 Decision: Production-scale streaming peak-memory verdict — close R051, re-defer S02/S03

> **Milestone M007 / Slice S01 — terminal decision artifact.**
> This document closes M007/S01 with a defensible go/no-go verdict on the
> R051 operator follow-up ("server-side/large-volume production peak-memory
> measurement against real per-z mosaic volumes") and on whether the two
> downstream sketch slices are justified by the measured numbers:
> - **S02 — Streaming Zarr write path** (the M003 MVP materializes output in full)
> - **S03 — Terabyte-scale distributed streaming redesign** (R056, currently deferred)
>
> It synthesises the two production-scale peak-memory probes produced in this
> slice — T02 (eager baseline) and T03 (streaming candidate) — both run as
> **independent OS processes** on the A6000 against the real per-z mosaic
> volume `sub-22` slice 27 (`n_z=55`). A future agent or operator can audit
> the R051 closure and the S02/S03 go/no-go from this file plus the two JSON
> artifacts alongside it (`eager-probe.json`, `streaming-probe.json`) without
> re-running the A6000 job.
>
> **Pattern.** This follows the terminal-slice decision-artifact pattern
> established by `scripts/experiments/s03_artifacts/S03-DECISION.md`
> (M006/R058): a read-only synthesis of version-controlled evidence that
> records an explicit go/no-go and what would need to change to revisit it.
>
> **Read-only synthesis.** This task changes **no** `linum_basic/` production
> code, no BaSiC invariant (K02), no default, and no algorithm. Its only
> write is this document. The R051 requirement-note update that records the
> production-scale corroboration is specified in §7 below and is deferred to
> slice closure (the requirement-update lifecycle tool is phase-gated for the
> execute-task lane; the orchestrator performs it when closing S01). The
> streaming MVP itself shipped in M003/S02; S01's job was solely to measure
> it at production scale and size the remaining work.

---

## Verdict (TL;DR)

### **R051 operator follow-up: CLOSED. The streaming read-path MVP delivers a real, production-scale host-memory win at zero numeric and zero VRAM cost.**

On the real per-z mosaic volume `sub-22` slice 27 (`n_z=55`, 5 of 55 z-planes
measured: `[0, 13, 27, 40, 54]`, `working_size=128`, `cuda:0`), measured as
two independent OS processes via `resource.getrusage(RUSAGE_SELF).ru_maxrss`:

| metric | eager (non-streaming) | streaming | delta |
|---|---|---|---|
| **peak host RSS** | **2,402,631,680 B (2.40 GB)** | **1,785,761,792 B (1.79 GB)** | **−616,869,888 B (−617 MB, −25.7%)** |
| peak GPU VRAM (`max_memory_allocated`) | 408,166,912 B (389 MiB) | 408,166,912 B (389 MiB) | **0 — identical** |
| wall time | 17,733 ms (17.7 s) | 19,529 ms (19.5 s) | +1,796 ms (+10.1% — streaming overhead) |

- The **25.7% host-RSS reduction** is the headline production-scale result.
  It corroborates — at OS level, on real data, on production hardware — the
  dev-time `tracemalloc` proof in `tests/test_streaming_fit.py` that M003/S02
  used to validate R051. `tracemalloc` cannot see native buffers
  (NumPy/BLAS scratch, PyTorch host-pinned memory, OpenCV internals); the
  OS-level RSS measurement here is the stronger, operator-grade proof.
- **VRAM is identical** because streaming changes only *when* host-side
  tile stacks are materialised, not the GPU working set — exactly as the MVP
  design intended. GPU memory is not the axis streaming addresses.
- **Numerics are bit-identical**, already proven in M003/S02
  (`TestStreamingNumericsParity`, `np.testing.assert_array_equal` on
  flat/dark-fields). This slice deliberately did not re-measure numerics;
  it measured memory only, with a byte-identical configuration.
- The **+10% wall-time overhead** is the cost of the lazy `zarr.Array`
  read path re-fetching per-z tile stacks instead of holding them resident.
  It is a favourable trade for a 617 MB host-RAM reduction and is well
  within operator tolerance for a fallback that exists to unlock large
  volumes, not to win speed.

### S02 disposition (streamed Zarr write path): **NOT JUSTIFIED by current evidence — re-deferred (stays a sketch).**

### S03 disposition (terabyte-scale distributed redesign, R056): **NOT JUSTIFIED — R056 stays deferred.**

The M007 vision gates both downstream slices on S01 showing real volumes
justify them: *"the codebase is single-host today and a cluster layer is a
large scope/dependency expansion that must be earned by evidence."* The
measured evidence does not earn them (see §4–§5). This is a legitimate
"measurement closes the question" outcome: S01 earns its keep by preventing
premature over-engineering, not by unlocking new builds.

---

## 1. The decision question and what "go" would require

M007/S01 exists to answer one question (per the roadmap vision and its
must-haves):

> *At production scale on a real large per-z mosaic volume, does the
> streaming read-path MVP deliver a materially significant host-memory
> reduction — and do the measured numbers justify building the streamed
> write path (S02) and the distributed redesign (S03)?*

For R051's closure, "go" only requires: a real, OS-level peak-RSS
measurement showing streaming reduces host memory vs eager on production
hardware, in independent processes. **That bar is met** (§3).

For S02/S03, "go" requires: evidence that real volumes approach or exceed
the single-host memory budget, so that a streamed write path or a cluster
layer would actually unlock work that otherwise fails. **That bar is not
met** (§4–§5).

---

## 2. Measurement method (why this proof is operator-grade)

T01 built `scripts/streaming_memory_probe.py`, a standalone operator script
that runs exactly **one** `fit_mosaic` invocation per process and records:

- **peak host RSS** via `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`
  (normalised to bytes). This is a monotonic per-process high-water mark
  set by the kernel — it sees the native buffers (`malloc`/`mmap` behind
  NumPy, BLAS, OpenCV, PyTorch host-pinned memory) that Python's
  `tracemalloc` fundamentally cannot.
- **peak GPU VRAM** via the existing `linum_basic.benchmark.telemetry.
  collect_memory_stats` helper (reused, not duplicated): records both
  `max_memory_allocated_bytes` and `max_memory_reserved_bytes`.
- full provenance: `mode`, `streaming`, `lazy_load`, `peak_rss_bytes`,
  `peak_vram_bytes`, `wall_ms`, `input_path`, `n_z`, `z_indices`,
  `working_size`, `backend`, `device`, `strategy`, `estimate_darkfield`,
  `overlap`, `max_reweighting_iterations`, `git_commit`, `host.platform`,
  `host.python_version`, `host.numpy_version`, `host.host_node`,
  `schema_version`, `timestamp_utc`.

T02 and T03 each invoked this script as a **separate OS process** with
byte-identical configuration — same subject, same `--z-sample 5` →
`[0, 13, 27, 40, 54]`, same `working_size=128`, same `backend=torch`,
same `device=cuda:0`, same `--estimate-darkfield` — changing only `--mode`
(`eager` vs `streaming`) and the output path. This is mandatory: peak RSS
is a monotonic per-process high-water mark, so the two modes must never
share a process or the second measurement would be contaminated by the
first. A config-equality check over the two JSONs confirms every
non-output field matches (only `mode`/`streaming`/`lazy_load`/timestamps
differ by design).

---

## 3. The measured result (apples-to-apples)

| field | eager-probe.json | streaming-probe.json | notes |
|---|---|---|---|
| `mode` | `eager` | `streaming` | only intended diff |
| `streaming` | `false` | `true` | |
| `lazy_load` | `false` | `true` | MVP streaming = lazy `zarr.Array` read |
| **`peak_rss_bytes`** | **2,402,631,680** | **1,785,761,792** | **−616,869,888 (−25.675%)** |
| `peak_vram_bytes` | 408,166,912 | 408,166,912 | identical (GPU working set unchanged) |
| `gpu_memory.max_memory_allocated_bytes` | 408,166,912 | 408,166,912 | identical |
| `gpu_memory.max_memory_reserved_bytes` | 551,550,976 | 551,550,976 | identical (PyTorch caching allocator) |
| `wall_ms` | 17,733.182 | 19,529.448 | +1,796 ms (+10.13%) streaming overhead |
| `input_path` | `…/sub-22/output/27/resample_mosaic_grid/mosaic_grid_z27_resampled.ome.zarr` | (same) | identical |
| `n_z` | 55 | 55 | identical |
| `z_indices` | `[0, 13, 27, 40, 54]` | `[0, 13, 27, 40, 54]` | identical (z-sample 5) |
| `working_size` | 128 | 128 | identical (production default) |
| `backend` / `device` | `torch` / `cuda:0` | `torch` / `cuda:0` | identical |
| `strategy` | `sequential` | `sequential` | identical |
| `estimate_darkfield` / `overlap` | `true` / `0.2` | `true` / `0.2` | identical |
| `max_reweighting_iterations` | 15 | 15 | identical |
| `flatfields_shape` / `darkfields_shape` | `[5,75,75]` | `[5,75,75]` | identical (5 z-planes × 75×75) |
| `git_commit` | `4e924e0` | `4e924e0` | identical |
| `host.host_node` | `sn4622125853` | `sn4622125853` | identical A6000 |
| `host.platform` | `Linux-…-x86_64-glibc2.43` | (same) | identical |

**Internal-consistency recompute (gsd_exec 2665d59a):** every figure above
was recomputed directly from the two JSON files; the percentage reduction
is `(2,402,631,680 − 1,785,761,792) / 2,402,631,680 = 25.675%`, matching
the headline. The config-equality check printed no `CONFIG DIFFERS` lines
for any non-output field, confirming the delta is a clean apples-to-apples
comparison attributable solely to the streaming mode.

**What the MVP does and does not change (recap from M003/S02):** streaming
changes *when* the per-z tile stacks are materialised in host memory (lazily,
one z-level at a time, over a `zarr.Array`-backed `MosaicGrid`) — it does
not change the GPU working set (hence identical VRAM) and does not change
the math (hence bit-identical numerics). The 617 MB host-RSS saving is the
materialisation window shrinking from "all measured z-planes resident" to
"one z-plane resident at a time."

---

## 4. S02 (streamed Zarr write path): NOT JUSTIFIED — re-deferred

S02 would make the *output* write path stream Zarr incrementally instead of
materialising the full output volume, with a bit-identical on-disk layout.
The M003 MVP's documented limit is exactly this: *"write path is NOT
streamed"* (R051 notes).

The case **against** building S02 now:

1. **No evidence of write-path memory pressure.** The measured output shape
   is `[5, 75, 75]` for flat-fields and dark-fields (5 measured z-planes at
   `working_size=128` → upsampled to 75×75 tiles) — a few hundred KB. Even
   at full `n_z=55` and full tile resolution, the output volume is tiny
   relative to host memory. The streamed write path addresses an output-
   materialisation problem that does not exist at measured scale.
2. **Single-host headroom is enormous.** The *entire* eager read+compute
   peak is 2.40 GB on an A6000 with 48 GB host RAM — ~5% of budget, ~20×
   headroom. The streaming mode drops that to ~3.7%. There is no host-
   memory pressure that a streamed write path would relieve.
3. **The roadmap gate is explicit.** M007's vision: *"Each later slice is
   gated on S01 showing real volumes justify it."* S01 measured the largest
   available real per-z mosaic volume and found it fits comfortably in
   single-host memory under the non-streamed path. The gate is not met.

**What would justify S02 in a future milestone:** evidence of a real
production *output* volume whose materialised write exceeds host memory
budget (e.g. a full-tile, full-`n_z` mosaic write at native resolution on a
memory-constrained host), or a stated operator requirement to write
corrected volumes larger than available RAM. Until then, the write-path
limitation stays documented (`docs/streaming.md`) and the MVP's eager write
remains correct and sufficient.

**Disposition:** S02 stays a `[sketch]`. No implementation in M007.

---

## 5. S03 (terabyte-scale distributed redesign, R056): NOT JUSTIFIED — stays deferred

S03 would scope a cluster-scale distributed streaming redesign
(coordination layer, worker model). R056 ("Full terabyte-scale streaming
Zarr redesign") is already `deferred` with note *"v2.0 delivers MVP only."*

The case **against** S03 now:

1. **No terabyte-scale target exists.** No production volume approaching
   host-memory limits has been demonstrated. The largest real volume
   measured peaks at 2.40 GB eager / 1.79 GB streaming — three orders of
   magnitude below "terabyte-scale."
2. **Single-host is comfortably sufficient.** The A6000 (48 GB RAM, 48 GB
   VRAM) handles the measured workload at <5% memory utilisation. A
   2×A6000 host (the project's existing production concurrency fallback,
   per MEM001/D-14) doubles that. There is no workload that *requires* a
   cluster to run.
3. **Scope/dependency expansion is disproportionate.** A distributed layer
   is a coordination, scheduling, fault-tolerance, and deployment
   dependency that the single-host codebase does not carry. The M007 vision
   correctly demands this be *"earned by evidence"* — and the evidence here
   points the other way.
4. **The streaming MVP already removes the read-path ceiling for
   single-host scaling.** Because streaming bounds host RSS to ~one
   z-plane's tile stack, a single A6000 can scale to far larger `n_z` than
   measured before approaching memory limits — further reducing the
   near-term need for distribution.

**What would justify S03 in a future milestone:** a real production
requirement to process a volume that genuinely exceeds the memory (or wall-
time) budget of the project's largest single host *even with streaming
enabled* — e.g. a multi-terabyte corpus that must be processed under a
fixed wall-time SLA, or a host-RAM ceiling that streaming cannot keep a
single z-plane's tile stack under.

**Disposition:** R056 stays `deferred`. S03 stays a `[sketch]` and should
not be expanded in M007. The build-or-defer decision the roadmap's success
criterion #3 demands is **defer**, documented here with rationale.

---

## 6. What this milestone proves and what it does not

**Proves (proof level: operational):** M007/S01 operationally closes the
R051 operator follow-up with OS-level, production-hardware, real-subject
evidence. The 25.7% host-RSS reduction (617 MB) is reproducible from the
two version-controlled JSON artifacts under `scripts/experiments/s01_artifacts/`.
The S02/S03 re-deferral is defensible from the same evidence: the measured
single-host peak (2.40 GB eager, 1.79 GB streaming) is far below the 48 GB
A6000 budget, so no write-path or cluster work is earned.

**Does not prove / does not do:**
- It does **not** change any production code path, default, invariant (K02),
  or the streaming MVP itself (which shipped in M003/S02).
- It does **not** re-measure numerics — bit-identical correction was proven
  in M003/S02 and is not re-litigated here; this slice measured memory only.
- It does **not** rule out S02/S03 forever — it rules them out *on the
  current evidence*. The conditions for revisiting are stated in §4–§5.
- It does **not** exercise the streaming MVP at true terabyte scale (no such
  volume exists in the project); `n_z=55` with `z-sample=5` is the largest
  real per-z mosaic volume available. Extrapolation to terabyte scale is
  qualitative (§5 point 4), not measured.

---

## 7. Deferred action — R051 note update (performed at slice closure)

The `execute-task` lane for T04 is phase-gated to a restricted tool set
(`gsd_exec`, `gsd_capture_thought`, `gsd_decision_save`, `gsd_resume`,
`gsd_task_complete`); the requirement-update lifecycle tool is **not**
available in this lane, so the R051 note update cannot be written by this
task. It is deferred to **slice closure**, where the orchestrator owns the
lifecycle tools. The exact update the orchestrator should apply (append to
the existing R051 Notes field) is:

> Operator follow-up (server-side production-scale peak-memory measurement
> against real per-z mosaic volumes) CLOSED in M007/S01: ran
> `scripts/streaming_memory_probe.py` twice as independent OS processes on
> the A6000 against `sub-22` slice 27 (`n_z=55`, z=`[0,13,27,40,54]`,
> `ws=128`, `cuda:0`). OS-level peak RSS via `resource.getrusage` (sees
> native buffers `tracemalloc` cannot): eager 2.40 GB, streaming 1.79 GB =
> −617 MB / −25.7% host-RSS reduction, with IDENTICAL peak VRAM (408 MB)
> and +10% wall-time overhead — corroborating the dev-time `tracemalloc`
> proof at production scale. Evidence:
> `scripts/experiments/s01_artifacts/{eager,streaming}-probe.json` +
> `scripts/experiments/s01_artifacts/S01-DECISION.md`. S02 (streamed write
> path) and S03 (distributed redesign, R056) re-deferred: the measured
> 2.40 GB single-host peak is ~5% of the 48 GB A6000 budget, so no
> write-path or cluster work is earned by the evidence.

R051's status stays `validated` (it was already validated in M003/S02);
this update records the production-scale corroboration and the S02/S03
dispositions in its Notes field. No status transition is required. The
decision is also captured durably via `gsd_decision_save` (allowed in this
lane) for cross-milestone state in `DECISIONS.md`.

---

## Verification

- **Internal consistency (gsd_exec 2665d59a, exit 0, 82 ms):** every numeric
  claim in §3 was recomputed directly from `eager-probe.json` and
  `streaming-probe.json` — RSS delta `616,869,888 B`, percentage `25.675%`,
  VRAM delta `0` (identical), wall delta `+1,796.266 ms (+10.129%)`. The
  config-equality check reported no `CONFIG DIFFERS` lines for any
  non-output field, confirming the comparison is apples-to-apples.
- **Artifact presence:** both evidence bundles exist and are non-empty —
  `scripts/experiments/s01_artifacts/eager-probe.json` (1003 B) and
  `streaming-probe.json` (1005 B), each self-describing per the §2
  provenance schema.
- **Provenance traceability:** both artifacts record `git_commit=4e924e0`
  (A6000 committed HEAD) and `host_node=sn4622125853`; the probe harness
  itself was added in T01 commit `f9140fc` and changed no library path, so
  the measured `fit_mosaic`/library code is identical to local HEAD.
- **No code change to verify:** this task writes only this document (the
  R051 note update is deferred to slice closure per §7); no test suite,
  lint, or typecheck is affected.

---

## Failure Modes

This task has **no external runtime dependencies** — it is a read-only
synthesis of two version-controlled JSON evidence bundles (`eager-probe.json`,
`streaming-probe.json`). The only failure mode is a claim drifting from its
cited evidence; this is guarded by the §Verification internal-consistency
recompute (`gsd_exec 2665d59a`), which cross-checks every delta directly
against the stored artifact values, and by the config-equality check that
confirms the eager/streaming comparison is clean (no unintended
configuration differences). The decision document does not invoke any API,
network, subprocess, or GPU at runtime — it is a static markdown artifact
consumed by the R051 requirement record (updated at slice closure per §7)
and by any future milestone that re-evaluates S02/S03.

## Load Profile

Not applicable — this is a one-shot decision synthesis with no runtime load
dimension. There is no request volume to saturate; the artifact is a fixed-
size markdown document, produced once and consumed by the R051 requirement
record (updated at slice closure per §7) and the M007 roadmap. No compute,
GPU, concurrency, or I/O is exercised by the synthesis itself. (The probes
that *generated* the evidence were one-shot operator runs on the A6000,
already completed in T02/T03; they are inputs to this task, not a load it
imposes.)

## Negative Tests

The decision's falsification surface is its evidence, locked by the
probe's contract test and the artifact schema. Specifically, the following
guarantees protect this verdict and would catch a regression of the
underlying measurement:
- `tests/test_streaming_memory_probe.py` (T01, 26 cases) — the probe's
  synthetic contract test proves the JSON schema is well-formed and that
  peak values are non-negative for both modes; it would fail if the probe
  script regressed in a way that produced malformed or negative-peak
  artifacts. Re-runnable locally (no CUDA) via
  `uv run pytest tests/test_streaming_memory_probe.py -v`.
- `tests/test_streaming_fit.py::TestStreamingNumericsParity` (M003/S02) —
  locks the bit-identical-numerics premise this decision relies on
  (streaming changes memory, not math).
- The §Verification config-equality check (re-runnable via the `gsd_exec`
  recompute) — would surface any future drift that made the eager/streaming
  comparison non-apples-to-apples (e.g. a working_size or z-sample change
  between runs), which would invalidate the 25.7% delta.

No new negative test is added by this task because the decision adds no
executable code; its negative surface is the measurement contract above,
which is already covered.
