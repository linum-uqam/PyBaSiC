(gpu)=
# GPU Acceleration

linum-basic ships a backend abstraction layer
({mod}`linum_basic.backend`) that lets the ALM solver run on any device
supported by PyTorch: NVIDIA CUDA or CPU via the Torch compute graph.

```{note}
**Apple MPS is not supported.**  The ALM solver requires float64 for the
Lagrange multiplier and uses `torch.linalg.svdvals`, neither of which is
implemented on Apple Metal (MPS).  Passing `device="mps"` raises a
{exc}`NotImplementedError` with a clear explanation.  Use `backend="auto"`
(selects CUDA if available, otherwise NumPy) or `backend="numpy"` on
Apple Silicon.
```

---

## Prerequisites

Install the optional `gpu` extra:

```bash
pip install "linum-basic[gpu]"
# or
uv add "linum-basic[gpu]"
```

This pulls `torch>=2.7`.  For CUDA support, follow the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/)
to install the correct CUDA toolkit build for your driver version.

---

## Selecting the backend

Pass `backend=` to {class}`~linum_basic.core.BaSiC`:

```python
from linum_basic import BaSiC

# Auto-select: Torch+CUDA if available, otherwise NumPy
model = BaSiC(stack, backend="auto")

# Explicit CUDA device
model = BaSiC(stack, backend="torch", device="cuda:0")

# Explicit NumPy (default)
model = BaSiC(stack, backend="numpy")
```

The `"auto"` mode selects CUDA if `torch.cuda.is_available()` returns
`True`, and falls back to NumPy otherwise.

When `"auto"` selects CUDA, it also applies a **size threshold**: if the
total number of image elements (`N × working_size²`) is below
{data}`~linum_basic.core.GPU_MIN_ELEMENTS` (default 1 000 000), the job is
routed back to NumPy because JIT compilation and kernel-launch overhead
outweigh the arithmetic gain on small stacks.  You can override this per
instance:

```python
# Always use the GPU, regardless of stack size
model = BaSiC(stack, backend="auto", gpu_min_elements=0)

# Raise the threshold (force NumPy for anything under 10 M elements)
model = BaSiC(stack, backend="auto", gpu_min_elements=10_000_000)
```

Explicit `backend="torch"` always uses the GPU regardless of stack size.

---

## Backend internals

The {class}`~linum_basic.backend.ArrayNamespace` wraps NumPy or Torch into a
uniform interface so the ALM loop in {func}`~linum_basic.algorithms.inexact_alm_l1`
can be backend-agnostic:

| Operation | NumPy | Torch |
|---|---|---|
| DCT / IDCT | `scipy.fft.dctn` | custom `_torch_dctn` (FFT-based DCT-II/III) |
| Sign | `numpy.sign` | `torch.sign` |
| Absolute value | `numpy.abs` | `torch.abs` |
| Maximum | `numpy.maximum` | `torch.maximum` |
| Array move to/from device | no-op | `tensor.to(device)` / `tensor.cpu().numpy()` |
| Minimum (global) | `numpy.min` | `torch.min` |
| Pass-through asarray | identity if already ndarray | identity if already correct-device tensor |

---

## Performance notes

- The ALM loop itself is compute-bound only for large `working_size`
  (≥256) and large stacks (≥500 images).  For typical `working_size=128`
  the GPU overhead of data transfer can outweigh the kernel benefit.
- Image loading and resizing (OpenCV) always run on CPU regardless of
  backend.
- When a CUDA device is selected, the per-iteration tensor work (steps 1–4
  of the ALM loop) is compiled with {func}`torch.compile` on the first call.
  The compiled function is **cached** at module scope, keyed by
  `(backend, device, image_size, l_s)`.  Subsequent calls with the same
  problem shape pay no compile cost — the same artifact is reused across all
  reweighting iterations and across different {class}`~linum_basic.core.BaSiC`
  instances.  The one-time warmup is amortised over the entire process
  lifetime.
- The precomputed DCT-II matrices and forward/inverse FFT twiddle factors
  are cached per ``(length, dtype, device)`` at module scope and bounded by
  an LRU policy (default 64 entries per cache).  Long-running batch jobs
  that fit many distinct image sizes therefore cannot leak memory across
  fits.  To force a full release between independent runs, call
  {func}`~linum_basic.backend.clear_dct_caches`.
- The darkfield estimation step (when `estimate_darkfield=True`) runs
  **entirely on-device** with no host-device synchronisation points during
  the iteration.  The only sync per iteration is the convergence check.
- Use `backend="auto"` for a transparent performance check — if CUDA is
  available and the stack is large enough it will be used.

Use `scripts/benchmark_gpu.py` to measure throughput on your hardware:

```bash
uv run python scripts/benchmark_gpu.py --n 8 32 128 --size 64 128 --iters 50
```

---

## Troubleshooting

### `RuntimeError: CUDA out of memory`

Reduce `working_size` (e.g. from 256 to 128) or process the stack in
smaller batches.

### `backend="torch"` but CPU Torch selected

Check that your PyTorch build includes CUDA support:

```python
import torch
print(torch.cuda.is_available())  # should be True for CUDA
```

If `False`, reinstall `torch` with the appropriate CUDA index URL from
<https://pytorch.org/get-started/locally/>.

### MPS (Apple Metal) not supported

`device="mps"` raises a {exc}`NotImplementedError` because Apple Metal
does not implement float64 arithmetic or `svdvals`, both of which the ALM
solver requires.  On Apple Silicon, use `backend="numpy"` or
`backend="auto"` (falls back to NumPy when no CUDA device is present).
