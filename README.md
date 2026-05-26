# PyBaSiC

*Python implementation of the BaSiC shading correction method — Python 3.14+, GPU-ready via PyTorch.*

[![DOI](https://zenodo.org/badge/219489337.svg)](https://zenodo.org/badge/latestdoi/219489337)
[![CI](https://github.com/YOUR_ORG/PyBaSiC/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/PyBaSiC/actions/workflows/ci.yml)

* **Original paper**: T. Peng *et al.*, "A BaSiC tool for background and shading correction of optical microscopy images," *Nat. Commun.*, vol. 8, p. 14836, Jun. 2017. [DOI:10.1038/ncomms14836](https://doi.org/10.1038/ncomms14836)
* **Nature Supplementary Materials**: [PDF](https://static-content.springer.com/esm/art%3A10.1038%2Fncomms14836/MediaObjects/41467_2017_BFncomms14836_MOESM560_ESM.pdf)
* **MATLAB implementation**: https://github.com/QSCD/BaSiC
* **Fiji plugin**: [URL](https://www.helmholtz-muenchen.de/icb/research/groups/quantitative-single-cell-dynamics/software/basic/index.html)
* **Demo data**: [Dropbox](https://www.dropbox.com/s/plznvzdjglrse3h/Demoexamples.zip?dl=0)

---

## Requirements

| Tool | Version |
|------|---------|
| Python | >= 3.14 |
| uv | >= 0.5 |

---

## Installation

PyBaSiC uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

### CPU-only (default)

```bash
git clone https://github.com/YOUR_ORG/PyBaSiC.git
cd PyBaSiC
uv sync
```

### GPU acceleration (PyTorch)

```bash
uv sync --extra gpu
```

This installs PyTorch alongside the core dependencies. On a machine with
CUDA or Apple Silicon (MPS), the `--backend auto` flag will automatically
select the accelerator.

### Development environment

```bash
uv sync --extra dev               # linting, type-checking, testing
uv sync --extra dev --extra gpu   # all extras
```

---

## Usage

### Command-line

```bash
basic_shading_correction \
    --input  /path/to/tiles \
    --output /path/to/corrected \
    --extension .tif \
    --estimate-darkfield \
    --backend auto \
    --verbose
```

Full argument reference:

```
usage: basic_shading_correction [-h] --input DIR --output DIR
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
  --device DEVICE       PyTorch device string, e.g. 'cuda:0' or 'mps'.
  --verbose             Show progress bars.
```

### Python API

```python
import numpy as np
from pybasic import BaSiC

# From a directory
model = BaSiC("/path/to/tiles", estimate_darkfield=True, backend="auto")
model.prepare()
model.run()

flatfield = model.get_flatfield()   # numpy array (H, W)
darkfield  = model.get_darkfield()  # numpy array (H, W)

# Apply to a new image
corrected = model.normalize(my_image)

# Save corrected stack
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
make lint          # ruff check + format check
make format        # auto-format with ruff
make typecheck     # ty check
make test          # pytest with coverage
make all           # lint + typecheck + test
```

---

## Citation

If you use PyBaSiC in your work, please cite the original paper:

```
T. Peng, K. Thorn, T. Schroeder, L. Wang, F. J. Theis, C. Marr, N. Navab,
"A BaSiC tool for background and shading correction of optical microscopy images,"
Nature Communications, 8:14836, 2017.
https://doi.org/10.1038/ncomms14836
```
