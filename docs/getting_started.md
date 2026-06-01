(getting_started)=
# Getting Started

## Installation

linum-basic requires Python 3.14+ and is available on PyPI:

```bash
pip install linum-basic
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add linum-basic
```

For GPU acceleration (PyTorch backend), install the optional `gpu` extra:

```bash
pip install "linum-basic[gpu]"
# or
uv add "linum-basic[gpu]"
```

---

## Minimal working example

```python
import numpy as np
from linum_basic import BaSiC

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

Pass a directory path (or a list of file paths) and linum-basic will load and
resize the images automatically:

```python
model = BaSiC("/path/to/tiles", extension=".tif", estimate_darkfield=True)
model.prepare()
model.run()

# Write corrected images to a new directory
model.write_images("/path/to/corrected")
```

Three factory classmethods provide equivalent, more explicit alternatives:

```python
# From a directory
model = BaSiC.from_directory("/path/to/tiles", estimate_darkfield=True)

# From an explicit list of file paths
model = BaSiC.from_files(sorted(Path("/path/to/tiles").glob("*.tif")))

# From a pre-loaded NumPy array
model = BaSiC.from_array(stack, estimate_darkfield=True)
```

---

## Dark-field estimation

Enable joint dark-field estimation with `estimate_darkfield=True`. This is
recommended for fluorescence microscopy where auto-fluorescence from the
objective, coverslip, or mounting medium produces a spatially varying
additive background:

```python
import numpy as np
from linum_basic import BaSiC

# Synthetic example: Gaussian vignette + constant dark-field offset
rng = np.random.default_rng(0)
stack = rng.uniform(0.3, 1.0, (100, 512, 512)).astype(np.float32)

# Enable dark-field estimation
model = BaSiC(stack, estimate_darkfield=True)
model.prepare()
model.run()

flatfield = model.get_flatfield()   # shape (512, 512), mean ≈ 1
darkfield = model.get_darkfield()   # shape (512, 512), small non-negative values

print(f"Flat-field mean : {flatfield.mean():.4f}")
print(f"Dark-field mean : {darkfield.mean():.4f}")

# Correct a single image: (image - darkfield) / flatfield
corrected = model.normalize(stack[0])
```

> **Note:** Dark-field estimation works best when at least some images in the
> stack have noticeably different per-image brightnesses (i.e. varying
> acquisition exposure or scene content).  If all images have nearly identical
> mean intensity, set `estimate_darkfield=False`.

```{figure} _static/demo/darkfield_demo_comparison.png
:alt: Darkfield correction comparison
:align: center

**Darkfield correction demo.**  Top row: one input tile with simulated
vignette + dark offset, ground-truth flat-field, ground-truth dark-field.
Bottom row: BaSiC-corrected tile, estimated flat-field, estimated dark-field.
```

---

## Command-line interface

linum-basic ships a `basic` command installed into the
Python environment's `bin/`:

```bash
basic correct --input /path/to/tiles \
              --output /path/to/corrected \
              --estimate-darkfield
```

See the [CLI reference](cli.rst) for a full list of flags.

---

## Quick-start checklist

1. Gather at least 20–30 representative images (more is better; 100+ is
   ideal for fluorescence).
2. Call {meth}`~linum_basic.core.BaSiC.prepare` to load data and auto-tune
   regularisation.
3. Call {meth}`~linum_basic.core.BaSiC.run` to fit the model.
4. Inspect `flatfield` and `darkfield` visually before applying correction.
5. See {ref}`Parameter Tuning <parameters>` if the result looks over- or
   under-smoothed.

---

## Next steps

- {ref}`Algorithm <algorithm>` — understand what BaSiC is solving.
- {ref}`Parameter Tuning <parameters>` — tune `l_s`, `l_d`, and friends.
- {ref}`GPU Acceleration <gpu>` — speed up large datasets with PyTorch.
