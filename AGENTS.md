# linum-basic — Agent Context Guide

This file provides a structured overview of the linum-basic repository for AI
agents and automated tools. It is intentionally comprehensive to reduce
cold-start exploration time. Follow the agent instruction below closely.

---

## Project purpose

linum-basic is a Python implementation of the **BaSiC** (Background and Shading
Correction) algorithm for optical microscopy images. It retrospectively
estimates a spatially non-uniform flat-field and an optional dark-field from
a stack of images, then applies the correction.

- Reference paper: Peng et al., *Nat. Commun.* 8, 14836 (2017).
- Package name on PyPI: `linum-basic`
- Version: 2.0.0
- Python requirement: ≥ 3.14
- License: GPL-3.0-or-later

---

## Agent Instructions

- Always use uv to manage dependencies and virtual environments, and follow the existing `Makefile` commands for consistency.
- Always install all optional dependencies when working on a new feature or fixing a bug, to ensure you have all the tools needed for testing and development. Use `uv sync --extra gpu --extra viz --extra docs --extra dev --extra validation --extra notebooks` (or `--all-extras` when available) to install everything.
- Use todos as much as possible to break up complex tasks into smaller steps to prevent context and reasoning overload. Always check open todos before starting a new task, and try to complete existing todos before creating new ones. Check if they are not stale or already resolved before acting on them.
- Write short and concise commit messages in the imperative mood, e.g. "Add auto-tuning of regularisation parameters" or "Fix GPU backend DCT implementation".
- For code changes, follow the existing style and conventions as closely as possible.
- For documentation changes, ensure that all public methods and parameters are documented with clear descriptions and usage examples where appropriate.
- Always run code quality checks (lint, format, typecheck) and tests locally before committing.
- Don't add license headers to files, the repo wide GPL 3.0 license file is sufficient.
- Always try to fix issues instead of ignoring them with `# noqa` or similar, unless the issue is a false positive
- Use standard unicode symbols unless they are mathematical variables (e.g. `μ` for the Lagrange multiplier) or part of code snippets. Use regular x and - for multiplication and subtraction in prose.
- Be careful when truncating command output with commands or parameters like `--tb=short` in pytest or `tail`, as it can hide important context for debugging test failures. For tests and other things with a progress bar, don't truncate the output as the user might want to track the progress.
- When asked to check something or add something make sure to check/add it in all relevant places, e.g. if adding a new parameter to the BaSiC class, make sure to add it to the CLI, documentation, and any relevant tests.
- When asked to check documentation, do actually check it, be it online or local files.
- When adding test coverage, make sure to add conceptually meaningful tests that cover edge cases and potential failure modes, not just superficial coverage. For example, if adding a new parameter to the BaSiC class, add tests that check its effect on the output and how it interacts with other parameters, not just that it can be set without error. When working with mathematics or physics concepts, make sure to check the relevant literature and ensure that the implementation matches the theoretical description, and add tests that validate this against known results or properties so it makes physical sense and is not just a code change that passes tests but is actually correct in the context of the problem domain.
- Don't add license headers to files, the repo wide GPL 3.0 license file is sufficient.
- When getting close to a context limit, prioritize completing existing todos and fixing issues over starting new tasks, and try to break up complex tasks into smaller steps to avoid hitting the context limit. Also consider summarizing or archiving old issues and todos that are no longer relevant to free up context space. If the context limit is still an issue, save the current findings into your memory so you can refer back to them later, and then clear the context to free up space for new information.
- Prefer repo memory over session memory, as findings could be useful to other agents or future sessions, and the repo memory is more persistent and less likely to be lost due to session timeouts or context limits. However, use your judgment to decide when to use session memory for temporary notes or findings that are only relevant to the current session and don't need to be saved long-term.

---

## Repository layout

```
linum-basic/ (repo: linum-basic)
├── linum_basic/              # Main Python package
│   ├── __init__.py           # Public API (BaSiC, fit_mosaic, tune, correct_images, …)
│   ├── algorithms.py         # Thin public re-export of ALM solver
│   ├── _alm.py               # Core ALM solver (inexact_alm_l1, shrink)
│   ├── backend.py            # ArrayNamespace: NumPy/PyTorch abstraction
│   ├── cli.py                # `basic` CLI entry point (correct|fit|tune|preview)
│   ├── core.py               # BaSiC estimator class (single-stack workflow)
│   ├── fit.py                # fit_mosaic, strategy resolver, batched CUDA path
│   ├── mosaic.py             # MosaicGrid, seam-pair enumeration
│   ├── tuning.py             # Optuna hyperparameter search
│   ├── metrics.py            # seam_l1 self-supervised metric
│   ├── curvature.py          # seam_curvature metric
│   ├── _parallel.py          # joblib process pools, CUDA device fan-out
│   ├── _batched_fit.py       # Batched CUDA ALM over z-planes
│   ├── _torch_cache.py       # Torch inductor / compile cache warm-up
│   ├── viz.py                # Optional matplotlib helpers
│   ├── io/zarr.py            # OME-Zarr read/write
│   └── benchmark/            # A/B harness library (artifacts, quality, strategies)
├── tests/
│   ├── test_alm_parity.py        # NumPy ↔ Torch ALM parity
│   ├── test_backend_parity.py    # DCT parity across backends
│   ├── test_cli.py               # CLI smoke tests
│   ├── test_docstrings.py        # Doctest runner for all public modules
│   ├── test_vignette_validation.py  # Integration test (needs validation extra)
│   └── test_benchmark_cli.py     # Benchmark harness CLI tests
├── docs/                     # Sphinx documentation source
│   ├── conf.py               # Sphinx configuration (nbsphinx executes notebooks)
│   ├── index.rst             # Landing page
│   ├── algorithm.md          # BaSiC math derivation + ALM flowchart
│   ├── parameters.md         # Every tuning knob with physical meaning
│   ├── getting_started.md
│   ├── gpu.md                # Backend selection and PyTorch GPU guide
│   ├── contributing.md
│   ├── validation.md         # Test suite + vignette integration test
│   ├── cli.rst               # sphinxarg-rendered CLI reference
│   ├── reference.rst         # Narrative toctree
│   ├── notebooks/            # Executable Jupyter tutorials
│   └── api/index.rst         # autoapi stub
├── scripts/
│   ├── benchmark_speedup.py  # fit_mosaic A/B benchmark harness
│   └── visualize_vignette_correction.py
├── pyproject.toml            # Build config, deps, Ruff, ty, pytest
├── Makefile                  # install / lint / format / typecheck / test / docs
├── .readthedocs.yaml         # Read the Docs build config (uv + Python 3.14)
└── CITATION.cff
```

---

## Key class: `BaSiC` (`linum_basic.core`)

The main estimator for flat tile stacks. Typical usage:

```python
from linum_basic import BaSiC
model = BaSiC(stack_or_dir, estimate_darkfield=True)
model.prepare()   # load + auto-tune l_s, l_d
model.run()       # reweighted ALM loop
flatfield = model.get_flatfield()
darkfield = model.get_darkfield()
corrected = model.normalize(img)
```

Post-init tuning knobs (set before `prepare()`):
- `working_size` (default 128) — resize resolution
- `epsilon` (default 0.1) — reweighting stability constant
- `l_s` / `l_d` (default None → auto-tuned) — regularisation weights
- `reweighting_tolerance` (default 1e-3)
- `max_reweighting_iterations` (default 10)

For OME-Zarr mosaic volumes, use `fit_mosaic()` in `linum_basic/fit.py` instead.

---

## Module map

| Module | Exports | Purpose |
|---|---|---|
| `linum_basic` | `BaSiC`, `correct_images`, `fit_mosaic`, `tune`, `inexact_alm_l1`, `shrink`, `__version__` | Public API |
| `linum_basic.core` | `BaSiC` | Single-stack shading estimator |
| `linum_basic.fit` | `fit_mosaic`, `MosaicFit`, `apply_fit` | Per-z mosaic fitting with strategy resolution |
| `linum_basic.mosaic` | `MosaicGrid`, `SeamPair` | Tile extraction and seam enumeration |
| `linum_basic.tuning` | `tune`, `TuneResult` | Optuna hyperparameter search |
| `linum_basic.algorithms` | `inexact_alm_l1`, `shrink` | Public ALM solver access |
| `linum_basic._alm` | `inexact_alm_l1`, `inexact_alm_l1_batched`, `shrink` | Core numerical solver (private) |
| `linum_basic.backend` | `ArrayNamespace`, `Backend`, `get_xp` | NumPy/PyTorch abstraction |
| `linum_basic._parallel` | `parallel_map`, `parallel_map_cuda_devices`, `is_gpu_backend` | Process/GPU parallelism |
| `linum_basic._batched_fit` | `fit_stacks_batched` | Batched CUDA ALM over z-chunks |
| `linum_basic._torch_cache` | `warm_policy_passes` | Torch compile cache warm-up |
| `linum_basic.benchmark` | artifacts, quality, strategies, profile | A/B harness library |
| `linum_basic.cli` | `main`, `_build_arg_parser` | CLI entry point |
| `linum_basic.io.zarr` | `load_ome_zarr`, `write_ome_zarr` | OME-Zarr I/O |

---

## CLI entry point

```
basic correct --input DIR --output DIR [--estimate-darkfield]
              [--extension .tif] [--backend auto|numpy|torch]
              [--device cuda:0] [--verbose]

basic fit --input subject.ome.zarr --output corrected.ome.zarr ...
basic tune --input subject.ome.zarr ...
basic preview --input subject.ome.zarr ...
```

Registered in `pyproject.toml` as:

```
basic = "linum_basic.cli:main"
```

MPS devices are rejected at argparse validation; use `--backend numpy` on Apple Silicon or `--device cuda:0` on CUDA servers.

---

## Benchmark harness

`scripts/benchmark_speedup.py` compares `fit_mosaic` wall time across CUDA execution strategies and supports a versioned A/B artifact workflow.

**Legacy `--mode` flags** (still supported without deprecation warnings):

| Mode | Description |
|---|---|
| `sequential` | Single GPU, one z-level at a time |
| `multi` | Multi-GPU process fan-out across z-levels |
| `batched` | Single batched CUDA solve over all z-planes |
| `all` | Run every mode and print a comparison table |

**Harness subcommands:**

| Subcommand | Purpose |
|---|---|
| `baseline` | Run a production-shaped baseline fit, calibrate tolerances, write a versioned artifact bundle |
| `candidate` | Run a candidate strategy against a saved baseline; emit speed, quality, and promote/reject verdicts |
| `compare` | Recompute promote/reject summary from saved baseline and candidate artifacts |
| `concurrency` | Multi vs batched A/B aggregator (canonical concurrency verdict) |

**Quality gates:** The harness calibrates per-metric tolerances (`mean+3std` policy on `seam_l1` and `seam_curvature`) from a baseline bundle, then evaluates candidate deltas via `evaluate_quality_gate()`. Quality failure always rejects promotion regardless of speed ratio.

**Handoff artifacts:** Profiling and optimization workflows emit JSON bundles including `phase5-fast-path.json` (recommended fast-path configuration) and concurrency verdict summaries consumed by downstream compare/integration steps. Pass `--frozen-fast-path` to candidate runs for git-commit drift checks.

Example:

```bash
uv run python scripts/benchmark_speedup.py baseline \
    --input subject.ome.zarr --subject-id sub-22 --output-dir ./runs \
    --z-sample 5 --strategy baseline --synthetic
```

---

## Dependencies

**Core:** `numpy>=2.2`, `scipy>=1.15`, `opencv-python>=4.11`, `tqdm>=4.67`, `zarr>=3`, `ome-zarr>=0.10`, `optuna>=3.6`, `joblib>=1.4`

**Optional extras** (install with `uv sync --extra <name>`):

| Extra | Purpose |
|---|---|
| `gpu` | PyTorch CUDA backend (`torch>=2.7`) |
| `viz` | Matplotlib visualisation helpers |
| `docs` | Sphinx documentation build stack |
| `dev` | pytest, ruff, ty, pre-commit, pandas, pyyaml |
| `validation` | `sbh-simulator` for vignette integration tests |
| `notebooks` | Jupyter notebook runtime for local tutorials |

Full dev install:

```bash
uv sync --extra dev --extra gpu --extra docs --extra validation --extra notebooks
```

---

## Build / dev commands

```bash
make install        # uv sync --extra dev
make test           # pytest -q
make lint           # ruff check
make format         # ruff format
make typecheck      # ty check linum_basic
make docs           # sphinx-build -W --keep-going -n -b html docs docs/_build/html
make docs-live      # sphinx-autobuild docs docs/_build/html
make all            # lint + format-check + typecheck + test
```

---

## Code conventions

- **Docstrings:** NumPy style (configured in `pyproject.toml` via
  `ruff.lint.pydocstyle.convention = "numpy"`)
- **Linter / formatter:** Ruff (`ruff check`, `ruff format`)
- **Type checker:** `ty` (not mypy)
- **Package manager:** `uv`
- **Math variable names:** uppercase (`Ib`, `Ir`, `D`, `W`) are intentional
  and suppressed via `N806` ignore in `pyproject.toml`

---

## Documentation hosting

Published to Read the Docs via webhook. Build config: `.readthedocs.yaml`.
Docs URL: https://linum-basic.readthedocs.io/

---

## Key algorithms / implementation notes

- Images are resized to `working_size x working_size` (default 128) before
  optimisation; the estimated flat-field is up-sampled back to the original
  resolution after `run()`.
- `prepare()` auto-tunes `l_s = dct_sum / 800` and `l_d = dct_sum / 2000`
  from the DCT of the normalised mean image.
- The `set_flatfield` / `set_darkfield` methods apply an implicit `.T`
  (transpose) before resizing — this is intentional to match OpenCV's
  column-major convention.
- `linum_basic.algorithms` is a thin public re-export of `linum_basic._alm` to avoid
  users having to import from a private module.
- The dark-field is initialised to all-zeros in `prepare()`. The convergence
  guard in `update()` returns `mad_dark = 1.0` when the previous dark-field
  estimate is zero (instead of dividing by a clamped epsilon), so the
  solver does not incorrectly declare convergence on the very first iterate.
- `fit_mosaic()` resolves `strategy="auto"` via `resolve_strategy()` in
  `linum_basic/benchmark/strategies.py`, choosing sequential, multi-GPU, or
  batched CUDA paths from workload shape and hardware discovery.
- ALM convergence notices use `_warn_convergence_once()` with `_CONVERGENCE_WARNED`;
  compile-fallback notices use `_warn_compile_fallback_once()` with `_COMPILE_WARNED`
  — not stdout `print()`.

## Invariants — do not change without careful regression testing

The following implementation details look like bugs or inefficiencies but
are **intentional**. Changing them silently breaks the darkfield regression
tests or contradicts the published algorithm.

| Location | Code | Why it must stay this way |
|---|---|---|
| `_alm.py` — B1 monotonicity guard | Only update `B1` when the clamped candidate `> 0`; otherwise keep the previous positive estimate. | B1 oscillates between ~0.09 and 0 on alternate iterations because the formula gives negative values that would be clipped to 0. On Linux, convergence fires when B1=0, yielding D_field ≈ 0. The guard ensures B1 is never reset to 0 after a valid estimate has been found. Do NOT replace with `B1 = max(0.0, min(...))` which silently drops the estimate. |
| `_alm.py` — second shrink | `Ir = shrink(xp.idctn(Ib), ...)` | Correct per paper Eq. 6 — dual penalty `\|F(DR)\|_1 + \|DR\|_1`. Removing the second shrink changes the estimated flat-field. |
| `backend.py` — `svd_leading_singular` | NumPy: full `numpy.linalg.svd(compute_uv=False)`; GPU/Torch: batched power iteration (`n_iter=30`) in `_svd_leading_singular_torch_batched` | Avoids full GPU SVD and device sync. Do NOT switch the Torch path back to `torch.linalg.svd` — even ~0.001% σ₁ drift shifts soft-threshold boundaries and breaks darkfield regression tests. |
| `core.py` — `cv2.resize` calls | `cv2.resize(img.T, new_shape, ...).T` | The double transpose is required for OpenCV's column-major convention. Removing it introduces a 1e-7 float difference that cascades through the iterative solver. |
| `fit.py` / `strategies.py` — ws≥128 batched guard | At `working_size >= 128`, `strategy="auto"` never selects batched CUDA; explicit `force_batched_cuda` or env override required. | UAT evidence shows batched ws=128 is memory-bandwidth bound and regresses quality vs sequential/multi-GPU paths at production resolution. |
| `fit.py` / `strategies.py` — auto strategy resolver | `resolve_strategy()` precedence: explicit user kwargs → env vars → auto heuristics from `WorkloadContext`. | Overrides must win predictably; changing precedence silently changes production fit paths. |
| `backend.py` / `_parallel.py` — CUDA-only MPS policy | MPS raises `NotImplementedError`; `is_gpu_backend()` is CUDA-only; `backend="auto"` on Apple Silicon (no CUDA) falls back to NumPy. | PyTorch MPS lacks float64 SVD support required by BaSiC; false GPU detection on Mac would trigger harmful multi-process fan-out. |
