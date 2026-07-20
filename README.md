# Linum BaSiC

*Python implementation of the BaSiC shading correction method — Python 3.14+, GPU-ready via PyTorch.*

[![CI](https://github.com/linum-uqam/Linum-BaSiC/actions/workflows/ci.yml/badge.svg)](https://github.com/linum-uqam/Linum-BaSiC/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/linum-uqam/Linum-BaSiC/branch/master/graph/badge.svg)](https://codecov.io/gh/linum-uqam/Linum-BaSiC)
[![Documentation](https://readthedocs.org/projects/linum-basic/badge/?version=latest)](https://linum-basic.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.14%2B-blue)](https://www.python.org/)
[![License: GPLv3+](https://img.shields.io/badge/License-GPLv3%2B-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![DOI](https://zenodo.org/badge/219489337.svg)](https://zenodo.org/badge/latestdoi/219489337)

* **Documentation**: https://linum-basic.readthedocs.io
* **Original paper**: T. Peng *et al.*, "A BaSiC tool for background and shading correction of optical microscopy images," *Nat. Commun.*, vol. 8, p. 14836, Jun. 2017. [DOI:10.1038/ncomms14836](https://doi.org/10.1038/ncomms14836)
* **Nature Supplementary Materials**: [PDF](https://static-content.springer.com/esm/art%3A10.1038%2Fncomms14836/MediaObjects/41467_2017_BFncomms14836_MOESM560_ESM.pdf)
* **MATLAB implementation**: https://github.com/QSCD/BaSiC
* **Fiji plugin**: [URL](https://www.helmholtz-muenchen.de/icb/research/groups/quantitative-single-cell-dynamics/software/basic/index.html)
* **Demo data**: [Dropbox](https://www.dropbox.com/s/plznvzdjglrse3h/Demoexamples.zip?dl=0)

---

## Getting Started

### 1. Install uv

linum-basic uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.
Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and install

```bash
git clone https://github.com/linum-uqam/Linum-BaSiC.git
cd Linum-BaSiC
uv sync
```

This creates an isolated virtual environment and installs all required dependencies.

### 3. Run on your data

```bash
basic correct --input /path/to/tiles --output /path/to/corrected
```

Or from Python:

```python
from linum_basic import BaSiC

model = BaSiC("/path/to/tiles")
model.prepare()
model.run()
corrected_img = model.normalize(my_image)
```

---

## Requirements

| Tool | Version |
|------|---------|
| Python | >= 3.14 |
| uv | >= 0.5 |

---

## Installation

### CPU-only (default)

```bash
git clone https://github.com/linum-uqam/Linum-BaSiC.git
cd Linum-BaSiC
uv sync
```

### GPU acceleration (PyTorch)

```bash
uv sync --extra gpu
```

This installs PyTorch alongside the core dependencies. On a machine with
CUDA, the `--backend auto` flag selects the Torch+CUDA backend when the
stack is large enough; otherwise it falls back to NumPy. Apple Silicon is
not supported for GPU acceleration — use `--backend numpy` or `--backend auto`
(NumPy fallback). See the [GPU docs](docs/gpu.md) for details.

### Development environment

```bash
uv sync --extra dev               # linting, type-checking, testing, pre-commit
uv sync --extra dev --extra gpu   # all extras
uv run pre-commit install         # install git hooks
```

---

## Usage

### Command-line

```bash
basic correct \
    --input  /path/to/tiles \
    --output /path/to/corrected \
    --extension .tif \
    --estimate-darkfield \
    --backend auto \
    --verbose
```

Full argument reference:

```
usage: basic correct [-h] --input DIR --output DIR
                     [--extension EXT]
                     [--estimate-darkfield]
                     [--backend {numpy,torch,auto}]
                     [--device DEVICE]
                     [--verbose]

options:
  --input DIR           Directory containing the input image stack.
  --output DIR          Directory to write corrected images.
  --extension EXT       File extension filter (default: .tif)
  --estimate-darkfield  Estimate the dark-field in addition to the flat-field.
  --backend {numpy,torch,auto}
                        Compute backend. 'auto' picks GPU when available.
  --device DEVICE       PyTorch device string, e.g. 'cuda:0'.
  --verbose             Show progress bars.
```

### Python API

```python
import numpy as np
from linum_basic import BaSiC

# From a directory
model = BaSiC("/path/to/tiles", estimate_darkfield=True, backend="auto")
model.prepare()
model.run()

flatfield = model.flatfield_fullsize   # numpy array (H, W)
darkfield  = model.darkfield_fullsize  # numpy array (H, W)

# Apply correction to a single image
corrected = model.normalize(my_image)

# Save the corrected stack to disk
model.write_images("/path/to/corrected")
```

```python
# From an existing NumPy stack
stack = np.load("stack.npy")   # shape (N, H, W)
model = BaSiC(stack, backend="torch", device="cuda")
model.prepare()
model.run()
```

---

## Development

```bash
make install       # uv sync --extra dev
make lint          # ruff check
make format        # auto-format with ruff
make typecheck     # ty check
make test          # pytest
make all           # lint + format check + typecheck + test
```

---

## Citation

If you use linum-basic in your work, please cite the original paper:

```
T. Peng, K. Thorn, T. Schroeder, L. Wang, F. J. Theis, C. Marr, N. Navab,
"A BaSiC tool for background and shading correction of optical microscopy images,"
Nature Communications, 8:14836, 2017.
https://doi.org/10.1038/ncomms14836
```
