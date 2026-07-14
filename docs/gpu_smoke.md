# GPU Smoke Test (Operator Runbook)

A short, manual CUDA regression check to run **before release or before merging
changes that touch CUDA/ALM code paths**. It runs the synthetic batched-ALM
parity test (`tests/test_alm_batched.py`) on a single A6000 GPU under the
production-relevant compile-off environment, with a sub-5-minute wall-time
budget. It is intentionally lightweight — one test file, one GPU — so it stays
cheap enough to run by hand on every CUDA-touching change.

> **Scope.** This is a *smoke* test, not the full quality gate. It catches CUDA
> build/numerics regressions in the core ALM solver. The real-subject A/B
> seam-quality gate (`scripts/benchmark_speedup.py`) remains the release
> authority for any speed/strategy change.

---

## When to run

Run the GPU smoke manually, on the A6000 server, when any of these are true:

- Preparing a release tag.
- About to merge a change that edits `linum_basic/_alm.py`,
  `linum_basic/_batched_fit.py`, `linum_basic/backend.py`,
  `linum_basic/_torch_cache.py`, or anything else in the CUDA/ALM hot path.
- After a PyTorch, CUDA driver, or `torch.compile` upgrade on the server.
- After changing the ALM algorithm invariants (see the invariants table in
  `AGENTS.md`).

There is **no automated GitHub Actions GPU job**. CUDA regressions are caught
only when an operator runs this. (A future blocking PR GPU CI check is noted
under [Deferred automation](#deferred-automation) below.)

---

## Prerequisites

- SSH access to the A6000 server (`frans@132.207.157.41`).
- The server checkout of this repo at `/home/frans/code/linum-basic`.
- The `gpu` extra installed (done by the `uv sync` step below).

Apple Silicon and MPS are **not** supported by this smoke — it is CUDA-only.
On a machine without CUDA the script skips cleanly with an explanatory message
rather than failing (see [Local development behavior](#local-development-behavior)).

---

## Procedure

Run the following on the A6000 server:

```bash
ssh frans@132.207.157.41
cd /home/frans/code/linum-basic
git pull
uv sync --extra dev --extra gpu
make test-gpu-smoke
```

That is the entire operator invocation. `make test-gpu-smoke` delegates to
`scripts/gpu_smoke.sh`, which:

1. Sets `LINUM_BASIC_ALM_COMPILE_MODE=off` (the production worker-compile-off
   path).
2. Pins execution to a single GPU via `CUDA_VISIBLE_DEVICES=0`, so the smoke
   targets `cuda:0` only and never fans out to the second A6000.
3. Runs `pytest tests/test_alm_batched.py -v --tb=short`.
4. Enforces a 5-minute (300s) wall-time budget via a `timeout` wrapper.

The only output is the pytest terminal log — there is no `smoke-result.json` or
other artifact to collect.

### Interpreting the result

| Output | Meaning | Action |
|---|---|---|
| `PASS: GPU smoke passed in Ns (budget 300s).` | The CUDA ALM parity test passed within budget. | Safe to proceed with the release/merge. |
| `FAIL: GPU smoke failed (pytest exit N) after Ns.` | A test failed — a real CUDA regression in the ALM solver. | Read the `--tb=short` traceback above the line; do **not** merge until fixed. |
| `FAIL: GPU smoke exceeded the 300s wall-time budget (Ns).` | The smoke hung or regressed past the budget (timeout exit 124). | Treat as a failure. Investigate a hang or perf regression before merging. |
| `skip: CUDA not available (...)` | No CUDA was detected at run time. | You are not on the server, or the `gpu` extra/`torch` is missing. Re-run on the A6000 after `uv sync --extra dev --extra gpu`. |

---

## Avoiding resource conflicts

The smoke does **not** place a lock on the GPU and does not run an
`nvidia-smi` pre-flight guard. Before running, check that no other heavy GPU
job is using `cuda:0`:

```bash
nvidia-smi
```

In particular, **avoid running the smoke during active Nextflow or benchmark
harness jobs**. Concurrent GPU contention can inflate the wall time past the
5-minute budget (producing a spurious timeout failure) or starve memory. If the
server is busy, wait for the other job to finish, or run the smoke on the other
A6000 by overriding the device:

```bash
LINUM_BASIC_GPU_SMOKE_DEVICE=1 make test-gpu-smoke
```

This remaps physical GPU 1 to the `cuda:0` the tests target. The default is
device 0.

---

## Wall-time budget

The smoke must finish in **under 5 minutes (300s)**. The script wraps pytest in
a `timeout` and treats a timeout (exit code 124) as a failure. The budget can
be overridden for diagnosis only — do **not** raise it to make a regression
disappear:

```bash
# Diagnostic only: allow 10 minutes to capture a full traceback on a slow run
LINUM_BASIC_GPU_SMOKE_TIMEOUT=600 make test-gpu-smoke
```

A healthy run completes well under budget. A run that consistently nears or
exceeds 5 minutes signals a performance regression that should be investigated,
not absorbed.

---

## Local development behavior

On a developer laptop without CUDA (e.g. Apple Silicon), `make test-gpu-smoke`
prints a clear skip message and exits 0 — it does not fail the local gate:

```
skip: CUDA not available (probe='no-torch').
      GPU smoke runs on the A6000 server; see docs/gpu_smoke.md.
```

This keeps the same entrypoint usable everywhere. The real CUDA verification
only happens on the server.

---

## Deferred automation

This slice delivers the manual operator path only. **Automated, blocking
GPU-on-PR CI** (a self-hosted GitHub Actions runner on the A6000 that gates
merges) is deferred until an accessible runner exists. When that runner is
added, the same `make test-gpu-smoke` entrypoint is intended to be invoked
unchanged by the CI job — no new script will be needed.

---

## Troubleshooting

### `skip: CUDA not available (probe='no-torch')` on the server

The `uv`-managed venv does not have `torch`. Re-run:

```bash
uv sync --extra dev --extra gpu
```

The smoke probes CUDA through `uv run python` so it sees exactly the venv the
server uses.

### `skip: CUDA not available (probe='no-cuda')` on the server

`torch` is installed but `torch.cuda.is_available()` returned `False`. Check the
driver and CUDA build:

```python
import torch
print(torch.cuda.is_available(), torch.version.cuda)
```

### Spurious timeout under contention

If `nvidia-smi` shows another process on `cuda:0`, see
[Avoiding resource conflicts](#avoiding-resource-conflicts) and either wait or
switch devices with `LINUM_BASIC_GPU_SMOKE_DEVICE=1`.

---

## Validation Log

Dated evidence that the smoke procedure above was executed end-to-end on real
server-side CUDA hardware. Each entry records a successful operator run; append
a new dated entry on each future validation run. This is the in-repo proof for
requirement R050 (server-side CUDA smoke).

### 2026-07-14 — A6000 (`sn4622125853`), commit `155a0dd`

- **Result:** PASS — `PASS: GPU smoke passed in 9s (budget 300s).`
- **Server host:** `sn4622125853` (`132.207.157.41`)
- **GPU:** NVIDIA RTX A6000, smoke on `cuda:0` (`CUDA_VISIBLE_DEVICES=0`)
- **Git commit:** `155a0dd660d271663bd564bc929705d50773e63d` (branch `modernisation`)
- **Software:** torch `2.12.1+cu130`, CUDA `13.0`, Python `3.14.3`, pytest `9.1.1`
- **Test file:** `tests/test_alm_batched.py` — 3 passed, 0 failed
- **Compile mode:** `LINUM_BASIC_ALM_COMPILE_MODE=off` (eager / production worker path)
- **Wall-clock:** 9s (budget 300s)

This run closes the live server-CUDA follow-up noted in R050's validation field.
The initial run failed due to `warm_start_reweighting=True` numerical
instability under compile-off; root-caused and fixed in commit `155a0dd` before
this successful re-run (see the test comment in `tests/test_alm_batched.py`).
