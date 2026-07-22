# Changelog

All notable user-facing changes to linum-basic are documented here.

## 2.0.0 — Performance and maturity release

This release delivers faster per-z mosaic fitting, a clearer GPU backend policy, structured library warnings, and stronger documentation and CI coverage. Version remains **2.0.0**; no PyPI publish is included in this milestone summary.

### Mosaic fitting

- **`fit_mosaic`** runs BaSiC independently on each z-level of an OME-Zarr mosaic, with optional global or per-z field aggregation.
- **Auto strategy selection** (`strategy="auto"`) picks sequential single-GPU, multi-GPU fan-out, or batched CUDA execution from workload shape, visible hardware, and explicit overrides.
- Seam-quality metrics (`seam_l1`, `seam_curvature`) support self-supervised evaluation without ground-truth references.
- **Optuna tuning** (`tune`) searches hyperparameters against combined seam metrics on subsampled z-levels.

### Performance

- **Batched CUDA path** fuses multiple z-planes into a single GPU ALM solve when resolution and hardware favour it; z-chunk sizing limits peak memory on large volumes.
- **Multi-GPU concurrency** fans out per-z fits across visible CUDA devices when batched execution is not selected.
- At **working size ≥ 128**, auto strategy never selects batched CUDA without an explicit override — production UAT showed sequential or multi-GPU paths preserve quality at full resolution.
- **`scripts/benchmark_speedup.py`** provides baseline/candidate/compare/concurrency subcommands, quality gates on seam metrics, and JSON artifact bundles (including fast-path configuration files) for reproducible speed/quality A/B workflows.

### GPU and CUDA backend

- **CUDA-only GPU policy:** PyTorch MPS (Apple Silicon Metal) is **not supported** for BaSiC fitting — float64 SVD requirements are unmet on MPS.
- **`backend="auto"`** on machines without CUDA resolves to **NumPy**; use `--backend numpy` on Apple Silicon or `--device cuda:0` on CUDA servers.
- **`is_gpu_backend()`** detects CUDA only, preventing false multi-process fan-out on Mac.
- The CLI rejects `--device mps` early with a clear error message.
- **`torch.compile`** is used when available; compile failures emit a **filterable `UserWarning`** and fall back to eager mode instead of failing silently.

### CLI

- Entry point renamed to **`basic`** with subcommands: `correct`, `fit`, `tune`, `preview`.
- OME-Zarr mosaic workflows are first-class via `basic fit` and `basic tune`.

### Warnings and diagnostics

- ALM **max-iteration convergence notices** now emit **`UserWarning`** (once per process for scalar fits, once per z-batch size for batched CUDA) instead of printing to stdout — consistent with compile-fallback warnings and filterable in notebooks and pipelines.
- Verbose progress remains available via `tqdm` when `verbose=True`.

### CI and documentation

- **`sbh-simulator`** moved from core dependencies to the optional **`validation`** extra; vignette integration tests install via `uv sync --extra validation`.
- **Docs CI** runs the same strict Sphinx build as `make docs` (`-W --keep-going -n`) on every pull request, executing all notebooks during the build.
- Documentation sweep aligned GPU/MPS guidance across README, parameters, CLI help, and API docstrings with the canonical policy in `docs/gpu.md`.
- **`AGENTS.md`** reconciled with `pyproject.toml` (version 2.0.0, GPL-3.0-or-later, `basic` CLI, all optional extras, benchmark harness, expanded invariants).

### Dependencies

- Python **≥ 3.14** required.
- Core stack: NumPy, SciPy, OpenCV, tqdm, zarr, ome-zarr, Optuna, joblib.
- Optional extras: `gpu`, `viz`, `docs`, `dev`, `validation`, `notebooks`.
