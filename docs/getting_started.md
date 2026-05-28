(getting_started)=
# Getting Started

## Installation

PyBaSiC requires Python 3.14+ and is available on PyPI:

```bash
pip install pybasic
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add pybasic
```

For GPU acceleration (PyTorch backend), install the optional `gpu` extra:

```bash
pip install "pybasic[gpu]"
# or
uv add "pybasic[gpu]"
```

---

## Minimal working example

```python
import numpy as np
from pybasic import BaSiC

# Simulate a fluorescence stack: 50 tiles, each 512 × 512
stack = np.random.rand(50, 512, 512).astype(np.float32)

model = BaSiC(stack, estimate_darkfield=True)
model.prepare()
model.run()

flatfield = model.get_flatfield()   # shape (512, 512)
darkfield = model.get_darkfield()   # shape (512, 512)

# Apply correction to each tile
corrected = np.stack([model.normalize(tile) for tile in stack])
```

---

## Reading images from disk

Pass a directory path (or a list of file paths) and PyBaSiC will load and
resize the images automatically:

```python
model = BaSiC("/path/to/tiles", extension=".tif", estimate_darkfield=True)
model.prepare()
model.run()

# Write corrected images to a new directory
model.write_images("/path/to/corrected")
```

---

## Command-line interface

PyBaSiC ships a `basic_shading_correction` command installed into the
Python environment's `bin/`:

```bash
basic_shading_correction --input /path/to/tiles \
                         --output /path/to/corrected \
                         --estimate-darkfield
```

See the [CLI reference](cli.rst) for a full list of flags.

---

## Quick-start checklist

1. Gather at least 20–30 representative images (more is better; 100+ is
   ideal for fluorescence).
2. Call {meth}`~pybasic.core.BaSiC.prepare` to load data and auto-tune
   regularisation.
3. Call {meth}`~pybasic.core.BaSiC.run` to fit the model.
4. Inspect `flatfield` and `darkfield` visually before applying correction.
5. See {ref}`Parameter Tuning <parameters>` if the result looks over- or
   under-smoothed.

---

## Next steps

- {ref}`Algorithm <algorithm>` — understand what BaSiC is solving.
- {ref}`Parameter Tuning <parameters>` — tune `l_s`, `l_d`, and friends.
- {ref}`GPU Acceleration <gpu>` — speed up large datasets with PyTorch.
