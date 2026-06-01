(parallelism)=
# Parallelism

linum-basic parallelises the expensive per-z-level ALM solves across CPU
cores using process-based parallelism (joblib / loky backend).  This page
explains when and how parallelism is used, how to tune the worker count,
and what to expect on CUDA/MPS hardware.

---

## Why processes, not threads

The core compute in {func}`~linum_basic.fit.fit_mosaic` and
{func}`~linum_basic.tuning.tune` is the **ALM solver** — a sequence of DCT
transforms and BLAS matrix operations provided by SciPy and NumPy.  On
CPython these operations hold the GIL, so a thread pool offers no
parallelism benefit.

Each z-level in a mosaic is an **independent** BaSiC solve: the flat-field
at depth *z* does not depend on depth *z ± 1*.  This makes them ideal for
process-based parallelism — coarse-grained, embarrassingly parallel, zero
inter-process communication.

linum-basic uses **joblib** with the loky process backend.  Each worker
runs its own Python interpreter and NumPy/SciPy environment; the main
process passes tile arrays and parameters over the joblib serialisation
channel and collects flat/dark-fields back.

---

## BLAS oversubscription guard

With *k* worker processes each spawning a full BLAS thread pool (typically
one thread per core), a machine with *n* logical cores would run
*k × n* threads simultaneously — thrashing the L3 cache and reducing
throughput.

linum-basic pins **inner BLAS/FFT threads to 1** per worker process via
{func}`joblib.parallel_config`.  This keeps the total hardware thread count
at *k*, matching the physical parallelism available.

---

## Default worker count

The default is

$$w = \max(1,\ \text{cpu\_count} - 2)$$

On a 12-core machine this gives **10 workers**.  Two cores are left free to
keep the operating system and I/O threads responsive during long fits.

You can override this via:

- the `n_workers` argument to {func}`~linum_basic.fit.fit_mosaic` and
  {func}`~linum_basic.tuning.tune`
- the `--n-jobs` CLI flag on `basic_fit` and `basic_tune`
- setting `n_workers=1` to run sequentially (useful for debugging)

```python
from linum_basic.fit import fit_mosaic
from linum_basic.mosaic import MosaicGrid

mosaic = MosaicGrid.from_ome_zarr("mosaic.ome.zarr")

# Default: cpu_count - 2
fit = fit_mosaic(mosaic)

# Explicit
fit = fit_mosaic(mosaic, n_workers=4)

# Sequential (for debugging or memory-limited machines)
fit = fit_mosaic(mosaic, n_workers=1)
```

---

## GPU / accelerator behaviour

On the PyTorch CUDA or MPS backend a single device is the bottleneck;
spawning multiple worker processes would serialise access to the device
and multiply VRAM consumption without benefit.

linum-basic **automatically collapses the worker count to 1** when
`backend="torch"` or `backend="auto"` is combined with a CUDA or MPS
device, and emits a {class}`UserWarning`:

```python
# This will run with n_workers=1 (single GPU) regardless of n_workers=8
fit = fit_mosaic(mosaic, n_workers=8, basic_kwargs={"backend": "torch", "device": "cuda:0"})
```

To use both a GPU backend and multiple devices, pass a different `device`
per z-batch manually — this is not yet automated.

---

## Expected scaling

The speedup is roughly linear up to the number of physical cores because
each z-level occupies one core.  The upper limit comes from:

- **Amdahl's law** — the data-loading phase (zarr read + tile extraction)
  is serial.
- **Memory bandwidth** — each worker allocates working arrays; on machines
  with limited DRAM bandwidth the curve flattens beyond 6–8 workers.
- **Tile stack size** — with very small mosaics (few z-levels) there is
  not enough work to saturate all workers.

### Benchmark results

All numbers below were recorded on real OCT data
(55 z-levels, 31 × 16 tiles per z, tile 75 × 75, `working_size=128`,
2 repeats, mean wall-clock time).

#### macOS — Apple M-series, 12 cores (numpy backend)

| workers | mean (s) | speedup |
|--------:|---------:|--------:|
|       1 |          |         |
|       2 |          |         |
|       4 |          |         |
|       6 |          |         |
|      10 |          |         |

#### Linux — AMD Threadripper PRO 5965WX, 48 cores (numpy backend)

| workers | mean (s) | speedup |
|--------:|---------:|--------:|
|       1 |          |         |
|       2 |          |         |
|       4 |          |         |
|       6 |          |         |
|      10 |          |         |

#### Linux — NVIDIA RTX A6000 (torch + cuda:0 backend)

GPU backend automatically collapses to `n_workers=1`.

| workers | mean (s) | vs. CPU w=1 |
|--------:|---------:|------------:|
|       1 |    70.92 |       ~5.0× |

Run the benchmark yourself with:

```bash
# Real data
uv run python scripts/benchmark_parallel.py --input /path/to/mosaic.ome.zarr

# Synthetic (no data required — useful for CI or remote testing)
uv run python scripts/benchmark_parallel.py --synthetic --syn-z 40

# GPU (remote machine with CUDA)
uv run python scripts/benchmark_parallel.py \\
    --input /path/to/mosaic.ome.zarr \\
    --backend torch --device cuda:0
```

Results are saved to `benchmark_results.json`.

---

## Vectorised correction

{func}`~linum_basic.fit.apply_fit` applies the estimated flat/dark-fields
to each tile.  Rather than looping over tiles one at a time, the z-plane is
reshaped to a ``(nrows, th, ncols, tw)`` view and the correction is applied
with a single NumPy broadcast:

```python
view = corrected[z].reshape(nrows, th, ncols, tw)
view[...] = (view - df[None, :, None, :]) / (ff[None, :, None, :] + epsilon)
```

This removes all Python-level tile iteration and lets NumPy use optimised
BLAS/LAPACK routines internally — beneficial even without multiple workers.

---

## Internal API reference

The helper functions live in {mod}`linum_basic._parallel`:

```{eval-rst}
.. autofunction:: linum_basic._parallel.default_workers
.. autofunction:: linum_basic._parallel.resolve_workers
.. autofunction:: linum_basic._parallel.is_gpu_backend
.. autofunction:: linum_basic._parallel.parallel_map
```
