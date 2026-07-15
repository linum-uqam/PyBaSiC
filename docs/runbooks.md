(runbooks)=
# Operator Runbooks

This page is the **single entry point for operators** running linum-basic in
production. Each runbook below covers one operational scenario with
reproducible commands, failure-mode interpretation, and a link to the
underlying detail. If you are unsure which runbook applies, start with the
[decision table](#which-runbook-when).

```{note}
These runbooks describe **operational** workflows — what to run, when, and
how to read the result — not the library API or the algorithm internals.
For the mathematical model see {doc}`algorithm`; for per-parameter meaning
see {doc}`parameters`.
```

---

## Which runbook when

| Situation | Use | Why |
|---|---|---|
| About to release or merge a CUDA/ALM change | {doc}`gpu_smoke` | Lightweight, manual CUDA regression check on the A6000 — the only place CUDA regressions are caught before release (no automated GPU CI exists yet). |
| A mosaic volume is too large for RAM | {doc}`streaming` | Plane-by-plane fitting bounds peak memory to one focal plane; bit-identical output to the eager path. |
| Default BaSiC params underperform on a dataset | {doc}`tuning` | Optuna search over `l_s`/`l_d`/`working_size`, then `recommend_bounds` narrows the next run. |
| Want tuning applied automatically with a safety net | {ref}`runbook-auto-apply` | `basic tune --auto-apply` tunes once, gates the candidate on full-volume quality, and falls back to defaults on any regression. |
| Need the fastest fit on a multi-GPU box | {doc}`parallelism` | Process / GPU fan-out and strategy selection — for wall time, not memory. |
| Selecting or debugging the compute backend | {doc}`gpu` | Backend selection (`numpy`/`torch`), CUDA setup, MPS exclusion rationale. |

The four runbooks in {ref}`runbooks-coverage` below — GPU smoke, streaming,
tuning, and auto-apply — are the ones this index consolidates. Parallelism
and GPU are linked for completeness because they answer the natural
follow-up questions ("how do I go faster?" and "which backend?").

---

(runbooks-coverage)=
## Runbooks at a glance

| Runbook | Goal | Cost | Where it runs | Output |
|---|---|---|---|---|
| {doc}`gpu_smoke` | Catch a CUDA regression before release | < 5 min, 1 GPU | A6000 server only | pytest pass/fail log |
| {doc}`streaming` | Fit a volume that exceeds RAM | Wall-time penalty, 1 plane in memory | Any backend (NumPy on Mac) | corrected OME-Zarr |
| {doc}`tuning` → `recommend_bounds` | Improve correction quality, then narrow the search | Minutes to hours (Optuna) | Any backend | best params + narrowed bounds |
| `basic tune --auto-apply` | Tune once and apply a vetted result with a safety gate | Tune budget + two full-volume fits | Any backend | corrected OME-Zarr + gate verdict |

---

(runbook-gpu-smoke)=
## GPU smoke test

**Full page:** {doc}`gpu_smoke`

### When

Run it on the A6000 server when **any** of these are true:

- Preparing a release tag.
- About to merge a change that edits `linum_basic/_alm.py`,
  `linum_basic/_batched_fit.py`, `linum_basic/backend.py`,
  `linum_basic/_torch_cache.py`, or anything else in the CUDA/ALM hot path.
- After a PyTorch, CUDA driver, or `torch.compile` upgrade.
- After changing the ALM algorithm invariants (see the invariants table in
  `AGENTS.md`).

### Why it exists

There is **no automated GitHub Actions GPU job**. CUDA build/numerics
regressions in the core ALM solver are caught *only* when an operator runs
this. It is intentionally a *smoke* test, not the full quality gate: the
real-subject A/B seam-quality gate (`scripts/benchmark_speedup.py`) remains
the release authority for any speed/strategy change.

### Reproducible command

On the A6000 server:

```bash
ssh frans@132.207.157.41
cd /home/frans/code/linum-basic
git pull
uv sync --extra dev --extra gpu
make test-gpu-smoke
```

`make test-gpu-smoke` pins a single GPU, sets `LINUM_BASIC_ALM_COMPILE_MODE=off`
(the production worker path), and wraps `pytest tests/test_alm_batched.py`
in a 5-minute (300s) `timeout`.

### What to look for

| Output | Meaning | Action |
|---|---|---|
| `PASS: GPU smoke passed in Ns (budget 300s).` | CUDA ALM parity passed within budget | Safe to proceed |
| `FAIL: GPU smoke failed (pytest exit N) after Ns.` | A real CUDA regression | Read the `--tb=short` trace; do not merge until fixed |
| `FAIL: GPU smoke exceeded the 300s wall-time budget (Ns).` | Hang or perf regression | Treat as failure; investigate |
| `skip: CUDA not available (...)` | Not on the server / `gpu` extra missing | Re-run on the A6000 after `uv sync --extra dev --extra gpu` |

### Gotchas

- Check `nvidia-smi` first — the smoke takes no GPU lock, so contention
  with a running Nextflow/benchmark job can spuriously blow the 5-minute
  budget. Switch devices with
  `LINUM_BASIC_GPU_SMOKE_DEVICE=1 make test-gpu-smoke`.
- On a laptop without CUDA the same command prints `skip: ...` and exits
  0 — it does not fail the local gate.

---

(runbook-streaming)=
## Streaming: fit a volume that exceeds RAM

**Full page:** {doc}`streaming`

### When

Reach for streaming when the full mosaic volume is the binding constraint:

- Peak memory of the eager fit exceeds available RAM.
- You are on a memory-limited host (laptop, shared login node, tight
  container cgroup) and only need the corrected output on disk.

You do **not** need streaming for small volumes, single-tile stacks, or the
tuning/GPU paths — those already fit comfortably, and streaming adds no
speed benefit (it is often *slower*).

### Why it exists

Streaming is a read-path optimisation that bounds peak memory to a single
focal plane. It does **not** change the correction math: the estimated
flat-fields and dark-fields are *bit-identical* to the eager path, because
it reuses the same solver routine, only changing *when* tiles are read.

### Reproducible command

```bash
basic fit --input subject.ome.zarr --output corrected.ome.zarr \
    --streaming --lazy --verbose
```

Both `--lazy` (keep the backing volume on disk) and `--streaming` (fit one
plane at a time) must be set for the memory win — without `--lazy` the grid
still loads the whole volume eagerly before the streaming fit begins.

### What to look for

| Symptom | Cause | Fix |
|---|---|---|
| No memory reduction | `--lazy` missing | Add `--lazy` so the backing volume stays on disk |
| `ValueError: streaming=True is incompatible with strategy='multi'` | Streaming is sequential-only | Drop `--strategy multi`/`batched`, or stop using `--streaming` |
| Slower than expected | Expected — streaming is sequential and trades wall time for memory | If memory is not the constraint, use {doc}`parallelism` or {doc}`gpu` instead |

### Gotchas

- The **write path stays eager**: applying the fit and writing the corrected
  volume still needs the full volume. Streaming makes large *fits* feasible
  on modest hosts; it does not by itself make large *correction write-outs*
  feasible.
- On Apple Silicon, streaming runs on the NumPy solver plane-by-plane (the
  MPS backend is unsupported for BaSiC fitting in any case).

---

(runbook-tuning)=
## Tuning: improve correction quality and narrow the search

**Full page:** {doc}`tuning` (the `recommend_bounds` workflow is the
"Recommending narrowed bounds" section)

### When

When the auto-tuned default `l_s`/`l_d` underperform on a dataset — visible
as poor seam consistency (high `seam_l1`) or a non-Gaussian flat-field
profile (high `seam_curvature`). Typically a first dataset of a new modality
or microscope.

### Why it exists

`tune()` runs Optuna (TPE sampler) over a log-uniform search space of
divisors ($l_s = \text{dct\_sum} / l_s\text{\_divisor}$) against the
seam/curvature/composite objective. The near-optimal trials usually cluster
in a small region; `recommend_bounds` collapses that region into a narrowed
`search_space` so a follow-up `tune` call concentrates where it pays off.

### Reproducible command

Library workflow — tune once, recommend narrowed bounds, then focus a second
run:

```python
from linum_basic.tuning import tune, recommend_bounds

result = tune(mosaic, n_trials=50, seed=42)

rec = recommend_bounds(result, margin=0.10)
print(f"Best value: {rec.best_value:.4f}")
print(f"Near-optimal trials: {rec.n_near_optimal} / {len(result.trials_df)}")
print(rec.search_space)

# Focus the next run on the empirically productive region
focused = tune(mosaic, n_trials=50, search_space=rec.search_space)
```

CLI equivalent — write the recommended bounds to JSON and print them with
`--verbose`:

```bash
basic tune --input mosaic.ome.zarr \
           --n-trials 50 \
           --bounds-json bounds.json \
           --bounds-margin 0.10 \
           --verbose
```

### What to look for

| Signal | Meaning |
|---|---|
| `n_near_optimal == 1` | Only the single best trial fell in the band; the recommended ranges are degenerate (`low == high`). Valid, not an error — widen `margin` to pull in more trials. |
| `n_near_optimal` grows as `margin` widens | Expected — a larger band admits more near-optimal trials and widens the ranges. |
| Bounds `search_space` transfers across datasets | Intended: bounds stay in the scale-invariant divisor parametrisation, so they do not carry a per-subject `dct_sum`. |

### Gotchas

- `recommend_bounds` is an **advisory** workflow heuristic, distinct from
  the release-gate calibration in `linum_basic.benchmark.quality` (which
  applies a `mean+3std` policy to repeated A/B baselines). It **never**
  gates a production fit — it only narrows the search space of a subsequent
  `tune` call.
- `best_params` (from the `TuneResult`) already holds resolved BaSiC
  hyperparameters and can be passed directly to `fit_mosaic`; the
  `BoundsRecommendation.search_space` is for re-feeding `tune`.

---

(runbook-auto-apply)=
## Auto-apply (fully automated)

**Full page:** {doc}`tuning` (the auto-apply workflow is the "Auto-apply
safety gate" section)

### When

Use `--auto-apply` when you want the tuned result applied **without** a
manual loop — i.e. you do not want to read best params, hand them to
`fit_mosaic`, and judge whether they are safe yourself. The gate guarantees
the applied result is never worse than the default-bounds fit on the full
volume.

Reach for it instead of the manual `tune` + `recommend_bounds` runbook
(above) when:

- You are correcting a *new* dataset and want one command that both tunes
  and applies a vetted result.
- You want the safety net: if tuning produces a worse full-volume fit, the
  default-bounds fit is applied automatically and the reason is recorded.

Prefer the **manual** tuning runbook when you want to *inspect* the
near-optimal region, narrow the search, and run a second focused search —
`--auto-apply` runs one tune pass and makes one apply decision per
invocation.

### Why it exists

`tune()` optimises a *subsampled* objective (a handful of z-levels, ≤
`max_tiles` tiles) on a single metric. A subsampled, single-metric score
cannot prove the best trial holds on the *full* volume against *both*
first-class metrics (`seam_l1` **and** `seam_curvature`). The auto-apply
gate closes that gap: it runs one extra full-volume fit with the tuned best
params, compares both metrics against the default-bounds full-volume fit,
and applies the candidate only if it does not regress either metric (fixed
`0.0` non-regression margin, D023). It adds no solver code and touches no
BaSiC invariant. Full design contract: {doc}`auto_apply_safety_gate`.

### Reproducible command

```bash
basic tune --input mosaic.ome.zarr \
    --auto-apply \
    --apply corrected.ome.zarr \
    --out-json best_params.json \
    --bounds-json narrowed_bounds.json \
    --n-trials 50 \
    --verbose
```

- `--auto-apply` switches on the guarded pipeline (baseline fit → tune →
  recommend_bounds → candidate fit → non-regression gate).
- `--apply corrected.ome.zarr` writes the **winning** fit: the tuned
  candidate on a pass, the default-bounds baseline on any fallback. The
  gate guarantees this is never worse than the default-bounds fit.
- `--out-json` writes the candidate best params (skipped on a fallback that
  fired before a candidate existed).
- `--bounds-json` writes the narrowed `search_space` recommendation.
- The non-regression margin is fixed at `0.0` (D023); it is not a CLI flag.

### What to look for

With `--verbose`, the verdict prints to stdout (deltas appear only when
computed — on a pass or a fail, not a pre-candidate fallback):

```
Auto-apply gate verdict: pass
  applied: candidate
  seam_l1: abs_delta=-0.012000 rel_delta=-0.021000
  seam_curvature: abs_delta=-0.000400 rel_delta=-0.012000
```

| Verdict / output | Meaning | Action |
|---|---|---|
| `verdict: pass`, `applied: candidate` | Candidate held both metrics ≤ baseline within the 0.0 margin | The `--apply` output is the tuned correction; safe to use |
| `verdict: fail`, `applied: baseline-default`, `fallback_reason: regression-detected` | Candidate regressed ≥ 1 metric on the full volume | The `--apply` output is the default-bounds fit; inspect `failing_metrics` + deltas to understand the regression |
| `verdict: fallback`, `applied: baseline-default` | An upstream step raised before a clean gate result | The `--apply` output is the default-bounds fit; check the `fallback_reason` |
| `error: auto-apply failed: ...` (stderr, exit 1) | The *baseline* fit itself failed — there is no safe floor | This is **not** a fallback. Fix the input / backend, then re-run |

`fallback_reason` enumeration (on a `fail` or `fallback`):

| Reason | Meaning |
|---|---|
| `regression-detected` | Candidate failed the non-regression gate on ≥ 1 metric |
| `tune-failed` | `tune()` raised (Optuna missing, all trials pruned, OOM in a trial) |
| `degenerate-trial-history` | `recommend_bounds()` saw zero completed trials |
| `invalid-margin` | `--bounds-margin` was outside `[0, 1]` |
| `candidate-fit-failed` | The candidate full-volume fit raised (OOM, divergence) |
| `quality-report-failed` | Quality report / deltas computation raised |

### Gotchas

- **A fallback is a success, not an error.** `--auto-apply` returns exit 0
  and writes a valid (default-bounds) corrected volume on any fallback;
  only the *baseline-fit* failure (`error: auto-apply failed: ...`, exit 1)
  is a true error. Do not treat a `regression-detected` fallback as a crash.
- **The applied fit is always safe.** Whether `applied` is `candidate` or
  `baseline-default`, `--apply` writes a correction the gate has proven is
  not worse than the default-bounds fit — you never receive a silently
  degraded volume.
- **One tune pass, one decision.** `--auto-apply` does not iterate the
  tune → narrow → retune loop; for multi-pass narrowing use the manual
  tuning runbook above.
- **Do not conflate** the auto-apply gate (per-dataset, fixed margin, two
  fits) with the `mean+3std` release gate in `linum_basic.benchmark.quality`
  (repeated A/B calibration). They reuse the same quality primitives but
  cannot disagree on what a metric means.

---

## Cross-cutting notes

- **No automated GPU CI.** The GPU smoke runbook is the only pre-release
  CUDA gate and it is manual; see its "Deferred automation" section for the
  self-hosted-runner plan.
- **Numerics are never changed by these workflows.** Streaming is
  bit-identical to the eager path; tuning, bounds recommendation, and
  auto-apply only change the *parameter values* fed to the same solver. The
  ALM invariants documented in `AGENTS.md` are untouched by all four
  runbooks.
- **Apple Silicon / MPS.** None of the GPU paths support MPS (float64 SVD
  is unavailable on Metal). Use `backend="numpy"` (or `backend="auto"`,
  which falls back to NumPy without CUDA) locally; run the GPU smoke on the
  A6000 server.
