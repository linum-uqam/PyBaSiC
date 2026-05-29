# Mosaic Grid Correction and Hyperparameter Tuning

This guide covers the OME-Zarr mosaic pipeline introduced in linum-basic 0.2 —
tiling, seam-consistency metrics, the per-z fit pipeline, and Optuna-driven
hyperparameter search.

---

## Seam Consistency

When a microscopy mosaic is imaged with a spatially non-uniform flat-field
(vignette), adjacent tiles look different at their shared boundaries even
though they image the same tissue.  The **seam-consistency L1 metric**
quantifies this mismatch as a *per-seam relative* error:

$$
\text{seam\_l1} = \frac{1}{|P|} \sum_{(i,j) \in P}
    \frac{\operatorname{mean}\left| a_{ij} - b_{ij} \right|}
         {\operatorname{mean}\!\left(\tfrac{|a_{ij}| + |b_{ij}|}{2}\right) + \varepsilon}
$$

where $P$ is the set of seam pairs and $a_{ij}$, $b_{ij}$ are the pixel
values in the shared overlap strip of tiles $i$ and $j$.  Dividing each
seam's absolute mismatch by its own local brightness makes the metric
**scale-invariant** (a global gain leaves it unchanged), **physical** (a
5-count step on a 10-count background scores worse than on a 1000-count
background), and impossible to game by brightening tile interiors away
from the seams.  Lower is better; a value near zero means adjacent tiles
are seamless.

A complementary metric is the **seam Pearson correlation** — the average
Pearson $r$ over all seam pairs.  Higher is better; values close to 1
indicate that adjacent tiles show the same content at their boundaries.

```python
from linum_basic.metrics import seam_l1, seam_pearson, evaluate_correction
from linum_basic.mosaic import MosaicGrid

mosaic = MosaicGrid.from_ome_zarr("my_mosaic.ome.zarr", overlap_fraction=0.2)

# Extract all tiles for z=0
tiles = mosaic.iter_tiles(z=0)
pairs = mosaic.seam_pairs()

print("Raw seam L1:", seam_l1(tiles, pairs))
print("Raw Pearson:", seam_pearson(tiles, pairs))
```

```{figure} _static/demo/seam_metric_demo.png
:alt: BaSiC flat-field 3-D surface and mosaic row before and after correction
:align: center
:width: 100%

**Left:** the BaSiC-estimated flat-field rendered as a 3-D surface — the dome-shaped
illumination curvature is the root cause of all tile-boundary artefacts.
**Middle:** a row of four raw tiles stitched side-by-side; the vignette repeats on every tile,
producing visible brightness jumps at each boundary (red dashed lines, mean seam L1 ≈ 0.67).
**Right:** the same row after BaSiC correction — the illumination curvature is removed and the
row is seamless (mean seam L1 ≈ 0.02, −96%).
```

---

## Loading an OME-Zarr Mosaic

linum-basic can load and write OME-Zarr v0.5 stores directly:

```python
from linum_basic.io import load_ome_zarr, write_ome_zarr

array, axes, scale = load_ome_zarr("my_mosaic.ome.zarr")
# array: numpy float32 (Z, H, W) or (H, W)
# axes:  e.g. ['z', 'y', 'x']
# scale: physical scale per axis in the same order
```

A `MosaicGrid` can be constructed directly from a Zarr store:

```python
from linum_basic.mosaic import MosaicGrid

mosaic = MosaicGrid.from_ome_zarr(
    "my_mosaic.ome.zarr",
    overlap_fraction=0.2,   # fraction of tile width/height that overlaps
)

print(mosaic.n_rows, mosaic.n_cols)  # grid dimensions
print(mosaic.tile_shape)             # (tile_h, tile_w)
```

Or from a plain numpy array:

```python
import numpy as np
from linum_basic.mosaic import MosaicGrid

array = np.load("mosaic.npy")   # shape (Z, H, W)
mosaic = MosaicGrid(array, tile_shape=(512, 512), overlap_fraction=0.2)
```

---

## Fitting the Flat-Field with `fit_mosaic`

`fit_mosaic` runs BaSiC independently on each z-level (or once globally) and
returns a `MosaicFit` object containing the estimated fields:

```python
from linum_basic.fit import fit_mosaic, apply_fit

fit = fit_mosaic(
    mosaic,
    field_mode="per-z",           # "per-z" or "global"
    basic_kwargs={"estimate_darkfield": True, "backend": "auto"},
    verbose=True,
)

# Inspect the estimated fields
print(fit.flatfields.shape)   # (n_z, tile_h, tile_w)
print(fit.darkfields.shape)   # (n_z, tile_h, tile_w)

# Apply correction — returns a (Z, H, W) float32 array
corrected = apply_fit(mosaic, fit)
```

To save the corrected mosaic and the estimated fields:

```python
from linum_basic.fit import save_corrected

save_corrected(
    mosaic,
    fit,
    output_path="corrected.ome.zarr",
    input_path="my_mosaic.ome.zarr",  # copies OME metadata
    overwrite=True,
)
```

### Field modes

| Mode | Description | Best for |
|---|---|---|
| `"per-z"` | Fit one flat-field / dark-field per z-level | 3-D stacks where illumination varies with depth |
| `"global"` | Average all per-z estimates into a single field | Large stacks where memory is limited |

---

## Hyperparameter Tuning with `tune`

BaSiC has two primary regularisation weights, `l_s` and `l_d`, that control
the trade-off between spatial smoothness and data fidelity.  The auto-tuned
defaults work well for typical microscopy data, but seam consistency can often
be improved by searching the parameter space.

`tune` uses **Optuna** (TPE sampler) to minimise `seam_l1` over a
log-uniform search space of divisors:

$$
l_s = \frac{\text{dct\_sum}}{l_s\text{\_divisor}}, \quad
l_d = \frac{\text{dct\_sum}}{l_d\text{\_divisor}}
$$

```{figure} _static/demo/tuning_demo.png
:alt: Optuna optimisation history, tuned flat-field, and seam-consistency improvement
:align: center
:width: 100%

A short 25-trial tuning run on a synthetic overlapping mosaic.  **Left:** each
Optuna trial's seam-L1 score with the running best (red).  **Centre:** the
flat-field recovered by the best trial.  **Right:** seam consistency before and
after correction — the optimiser drives the mismatch down by ~97 %.
```

```python
from linum_basic.mosaic import MosaicGrid
from linum_basic.tuning import tune

mosaic = MosaicGrid.from_ome_zarr("my_mosaic.ome.zarr", overlap_fraction=0.2)

result = tune(
    mosaic,
    n_trials=50,         # Optuna trials
    z_subsample=4,       # z-levels evaluated per trial (for speed)
    max_tiles=64,        # tiles used for the seam metric (subsampled)
    seed=42,
    verbose=True,
)

print("Best seam L1:", result.best_value)
print("Best params:", result.best_params)
# e.g. {'working_size': 128, 'l_s': 0.41, 'l_d': 0.16,
#       'epsilon': 0.1, 'estimate_darkfield': True}
```

`best_params` already holds resolved BaSiC hyperparameters (the searched
`l_s_divisor`/`l_d_divisor` are converted to absolute `l_s`/`l_d`), so the
dict can be passed directly to `fit_mosaic`:

### Search space

By default `tune` searches five parameters; pass a `search_space` dict to
override any of them (categorical entries take a list of choices, continuous
entries take a `(low, high)` tuple sampled log-uniformly):

| Parameter | Default range | Effect |
|---|---|---|
| `working_size` | `[64, 96, 128, 160, 192]` | Internal processing resolution |
| `l_s_divisor` | `(100.0, 5000.0)` | Flat-field smoothness ($l_s = \text{dct\_sum} / \text{divisor}$) |
| `l_d_divisor` | `(500.0, 10000.0)` | Dark-field smoothness ($l_d = \text{dct\_sum} / \text{divisor}$) |
| `epsilon` | `(0.01, 1.0)` | Reweighting stability constant |
| `estimate_darkfield` | `[True, False]` | Whether to estimate a dark-field |

The remaining BaSiC knobs (`reweighting_tolerance`, `max_reweighting_iterations`,
`warm_start_reweighting`) are convergence/speed controls rather than
quality-shaping parameters, so they are not part of the search space — set
them directly on the model when needed (see {doc}`parameters`).

### Speed and accuracy knobs

| Argument | Default | Purpose |
|---|---|---|
| `z_subsample` | `4` | Number of z-levels evaluated per trial.  More = more reliable, slower. |
| `max_tiles` | `64` | Tiles (evenly spaced) used for the seam metric per trial.  `None` uses all tiles. |
| `n_workers` | `1` | Worker processes used to fit z-levels in parallel within each trial (see {doc}`parallelism`). |
| `n_extra_rows` | `0` | Leading rows per tile to drop before fitting (galvo fly-back artefact). |
| `run_full_fit` | `False` | After tuning, run a full-z fit with the best params into `result.best_fit`. |

The `best_params` dict can be passed directly to `fit_mosaic`:

```python
fit = fit_mosaic(
    mosaic,
    basic_kwargs=result.best_params,
)
```

### Persistent studies with SQLite

For large datasets or multi-machine tuning, use a persistent Optuna storage:

```python
result = tune(
    mosaic,
    n_trials=100,
    storage="sqlite:///my_study.db",
    study_name="my-mosaic-tuning",
    n_workers=4,
)
```

Multiple workers can attach to the same database and tune in parallel:

```bash
# In four separate terminals
basic_tune --input my_mosaic.ome.zarr \
           --n-trials 25 \
           --storage sqlite:///tune.db \
           --study-name my-mosaic-tuning
```

---

## API Reference

| Symbol | Module | Description |
|---|---|---|
| `MosaicGrid` | `linum_basic.mosaic` | Tile extraction + seam-pair enumeration |
| `SeamPair` | `linum_basic.mosaic` | Descriptor for one pair of adjacent tiles |
| `seam_l1` | `linum_basic.metrics` | Mean per-seam relative mismatch (lower is better) |
| `seam_pearson` | `linum_basic.metrics` | Mean seam Pearson correlation (higher is better) |
| `evaluate_correction` | `linum_basic.metrics` | Combined metrics before and after correction |
| `MosaicFit` | `linum_basic.fit` | Container for estimated flat/dark-fields |
| `fit_mosaic` | `linum_basic.fit` | Run BaSiC on every z-level of a mosaic |
| `apply_fit` | `linum_basic.fit` | Apply a `MosaicFit` to produce a corrected array |
| `save_corrected` | `linum_basic.fit` | Save corrected mosaic as OME-Zarr |
| `TuneResult` | `linum_basic.tuning` | Best trial params, value, and optional fit |
| `tune` | `linum_basic.tuning` | Optuna-based hyperparameter search |

See the {doc}`API reference <api/index>` for full parameter documentation.
