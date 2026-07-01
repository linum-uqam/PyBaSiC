(validation)=
# Validation

Linum BaSiC's test suite is divided into three layers: unit/parity tests for the
numerical core, docstring tests, and an integration test against a simulated
vignette.

---

## Test files

| File | Purpose |
|---|---|
| `tests/test_alm_parity.py` | NumPy ↔ Torch parity for the ALM solver |
| `tests/test_alm_compile_warning.py` | `torch.compile` fallback and ALM convergence `UserWarning` deduplication |
| `tests/test_backend_parity.py` | NumPy ↔ Torch DCT/IDCT parity |
| `tests/test_cli.py` | CLI smoke tests (including MPS rejection on all GPU subcommands) |
| `tests/test_docstrings.py` | NumPy-style docstring quality checks via `numpydoc.validate` |
| `tests/test_parallel.py` | Worker resolution, multi-GPU fan-out, and vectorised `apply_fit` paths |
| `tests/test_vignette_validation.py` | Integration test: vignette recovery |
| `tests/test_demo_fitting.py` | Demo fit with synthetic vignette (skipped without `validation` extra; excluded from default `make test`) |

---

## Vignette integration test

The integration test in `tests/test_vignette_validation.py` provides the
primary end-to-end validation of the BaSiC algorithm:

1. **Generate a ground-truth vignette** using the
   [sbh-simulator](https://github.com/linum-uqam/sbh_simulator) Python API
   (`sbh_simulator.simulator`) — Gaussian or Zernike radial profile on a
   128 × 128 grid.
2. **Tile a source image** (`linum_basic/data/source_image.jpg`) into 128 × 128
   non-overlapping patches.
3. **Corrupt each patch** by multiplying with the vignette to simulate
   non-uniform illumination.
4. **Run BaSiC** and compare the recovered flat-field to the ground truth
   using Pearson correlation.

The test passes when the correlation exceeds **0.85**.

### Prerequisites

The vignette and demo fitting tests require the optional **`validation`**
extra, which installs
[sbh-simulator](https://github.com/linum-uqam/sbh_simulator). This dependency
is **not** part of the default install — only the dedicated CI vignette job,
the docs build (which executes `docs/notebooks/basic_usage.ipynb`), and local
validation workflows need it.

```bash
uv sync --extra dev --extra validation
uv run pytest tests/test_vignette_validation.py -v
```

To run the demo fitting test as well (as in CI):

```bash
uv run pytest tests/test_vignette_validation.py tests/test_demo_fitting.py -v
```

### Visualisation artefacts

When the environment variable `LINUM_BASIC_VIGNETTE_ARTIFACT_DIR` is set, each
test sub-case renders a 9-panel PNG into that directory:

```bash
mkdir -p /tmp/vignette_artefacts
LINUM_BASIC_VIGNETTE_ARTIFACT_DIR=/tmp/vignette_artefacts \
    uv run pytest tests/test_vignette_validation.py -v
open /tmp/vignette_artefacts/*.png
```

This is the same pipeline used by `scripts/visualize_vignette_correction.py`.

---

## Mosaic and curvature tests

The `tests/test_mosaic.py`, `tests/test_metrics.py`, and `tests/test_curvature.py`
suites cover the full mosaic pipeline:

| File | What it tests |
|---|---|
| `test_mosaic.py` | `MosaicGrid` tile extraction, seam-pair enumeration, overlap geometry |
| `test_metrics.py` | `seam_l1`, `seam_pearson`, `evaluate_correction`, `evaluate_correction_volume` |
| `test_curvature.py` | `fit_focal_gaussian`, `focal_profile`, `seam_curvature`, `seam_curvature_per_z`, `curvature_depth_profile` |

All three suites run without optional dependencies.

---

## ALM / backend parity tests

`test_alm_parity.py` and `test_backend_parity.py` verify that the NumPy and
Torch backends produce numerically identical results (within floating-point
tolerance) for the DCT operations and the full ALM loop.  These tests run
unconditionally — no GPU required.

---

## Running the full test suite

The default test suite (via `make test` or CI core jobs) excludes
`tests/test_demo_fitting.py`, which requires the validation extra. Core tests
run without sbh-simulator:

```bash
make test
# or:
uv run pytest -q --ignore=tests/test_demo_fitting.py
```

For verbose output:

```bash
uv run pytest -v --tb=short
```
