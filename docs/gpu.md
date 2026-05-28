(gpu)=
# GPU Acceleration

linum-basic ships a backend abstraction layer
({mod}`linum_basic.backend`) that lets the ALM solver run on any device
supported by PyTorch: NVIDIA CUDA, Apple MPS (Metal), or CPU via the
Torch compute graph.

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

# Apple Silicon MPS
model = BaSiC(stack, backend="torch", device="mps")

# Explicit NumPy (default)
model = BaSiC(stack, backend="numpy")
```

The `"auto"` mode probes for CUDA first, then MPS, then falls back to
NumPy.

---

## Backend internals

The {class}`~linum_basic.backend.ArrayNamespace` wraps NumPy or Torch into a
uniform interface so the ALM loop in {func}`~linum_basic.algorithms.inexact_alm_l1`
can be backend-agnostic:

| Operation | NumPy | Torch |
|---|---|---|
| DCT / IDCT | `scipy.fft.dctn` | custom `_torch_dctn` (Walsh-Hadamard) |
| Sign | `numpy.sign` | `torch.sign` |
| Absolute value | `numpy.abs` | `torch.abs` |
| Maximum | `numpy.maximum` | `torch.maximum` |
| Array move to/from device | no-op | `tensor.to(device)` / `tensor.cpu().numpy()` |

---

## Performance notes

- The ALM loop itself is compute-bound only for large `working_size`
  (≥256) and large stacks (≥500 images).  For typical `working_size=128`
  the GPU overhead of data transfer can outweigh the kernel benefit.
- Image loading and resizing (OpenCV) always run on CPU regardless of
  backend.
- Use `backend="auto"` for a transparent performance check — if GPU is
  available it will be used.

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
print(torch.backends.mps.is_available())  # should be True on M-series Mac
```

If `False`, reinstall `torch` with the appropriate CUDA index URL from
<https://pytorch.org/get-started/locally/>.

### MPS instability (NaN/Inf in flat-field)

Apple MPS support in PyTorch is still maturing.  If you encounter numerical
issues, fall back to `device="cpu"` with `backend="torch"`, or use
`backend="numpy"`.
