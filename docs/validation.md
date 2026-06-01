(validation)=
# Validation

Linum BaSiC's test suite is divided into four layers: unit/parity tests for the
numerical core, docstring tests, and an integration test against a simulated
vignette.

---

## Test files

| File | Purpose |
|---|---|
| `tests/test_alm_parity.py` | NumPy ↔ Torch parity for the ALM solver |
| `tests/test_backend_parity.py` | NumPy ↔ Torch DCT/IDCT parity |
| `tests/test_cli.py` | CLI smoke tests |
| `tests/test_docstrings.py` | Doctest extraction from all public modules |
| `tests/test_vignette_validation.py` | Integration test: vignette recovery |

---

## Vignette integration test

The integration test in `tests/test_vignette_validation.py` provides the
primary end-to-end validation of the BaSiC algorithm:

1. **Generate a ground-truth vignette** using
   [sbh-simulator](https://github.com/linum-uqam/sbh_simulator)'s
   `sbh-vignette` CLI (Gaussian or Zernike radial profile on a 128 × 128
   grid).
2. **Tile a source image** (`linum_basic/data/source_image.jpg`) into 128 × 128
   non-overlapping patches.
3. **Corrupt each patch** by multiplying with the vignette to simulate
   non-uniform illumination.
4. **Run BaSiC** and compare the recovered flat-field to the ground truth
   using Pearson correlation.

The test passes when the correlation exceeds **0.85**.

### Prerequisites

The test is automatically **skipped** when `sbh-vignette` cannot be found.
To run it:

```bash
# Install sbh_simulator into the same environment
pip install git+https://github.com/linum-uqam/sbh_simulator.git
```

Then:

```bash
uv run pytest tests/test_vignette_validation.py -v
```

### Visualisation artefacts

When the environment variable `LINUM_BASIC_VIGNETTE_ARTIFACT_DIR` is set, each
test sub-case renders a 6-panel PNG (input tiles, ground-truth vignette,
recovered flat-field, residual map, correlation scatter, and correction
result) into that directory:

```bash
mkdir -p /tmp/vignette_artefacts
LINUM_BASIC_VIGNETTE_ARTIFACT_DIR=/tmp/vignette_artefacts \
    uv run pytest tests/test_vignette_validation.py -v
open /tmp/vignette_artefacts/*.png
```

This is the same pipeline used by `scripts/visualize_vignette_correction.py`.

---

## ALM / backend parity tests

`test_alm_parity.py` and `test_backend_parity.py` verify that the NumPy and
Torch backends produce numerically identical results (within floating-point
tolerance) for the DCT operations and the full ALM loop.  These tests run
unconditionally — no GPU required.

---

## Running the full test suite

```bash
make test
# or:
uv run pytest -q
```

For verbose output:

```bash
uv run pytest -v --tb=short
```
