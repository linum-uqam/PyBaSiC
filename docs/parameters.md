(parameters)=
# Parameter Tuning

This page describes every tuning knob exposed by
{class}`~pybasic.core.BaSiC` — what each parameter controls physically,
its default value, the heuristic used to auto-set it, and common
troubleshooting recipes.

---

## Constructor parameters

These are set at object construction time and control data loading and
backend selection.

### `estimate_darkfield`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `False` |

Whether to estimate the dark-field $B(p)$ in addition to the flat-field
$S(p)$.

**When to enable:** fluorescence microscopy with autofluorescence from
objective, coverslip, or mounting medium; any acquisition where zero-photon
pixels are not truly zero.

**When to leave off:** bright-field, phase contrast, or any modality where
the background offset is constant and already subtracted by the camera.

**Symptom of wrong value:** leaving it `False` in a fluorescence experiment
causes the estimated flat-field to absorb the dark-field signal, producing
a flat-field with elevated edges or a ring artefact.

---

### `extension`

| | |
|---|---|
| **Type** | `str` |
| **Default** | `".tif"` |

File extension glob used when *input* is a directory.  Change to `".png"`,
`".jpg"`, etc. as needed.

---

### `backend`

| | |
|---|---|
| **Type** | `"numpy"` \| `"torch"` \| `"auto"` |
| **Default** | `"numpy"` |

Compute backend for the ALM solver.  See {ref}`GPU Acceleration <gpu>` for
full details.

---

### `device`

| | |
|---|---|
| **Type** | `str \| None` |
| **Default** | `None` |

PyTorch device string, e.g. `"cuda:0"` or `"mps"`.  Ignored when
`backend="numpy"`.

---

## Post-init tuning knobs

These attributes can be set **after** construction and **before** calling
{meth}`~pybasic.core.BaSiC.prepare`:

```python
model = BaSiC(stack)
model.working_size = 256          # higher resolution
model.epsilon = 0.05              # sharper reweighting
model.max_reweighting_iterations = 20
model.prepare()
model.run()
```

---

### `working_size`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `128` |

All images are resized to `working_size × working_size` before
optimisation.  Larger values capture more spatial detail in the flat-field
but increase memory and computation roughly as $O(P^2)$ where
$P = \text{working\_size}^2$.

**Guidance:**
- `128` — suitable for most cameras; very fast.
- `256` — better for sensors with strong radial gradients or >4 Mpx
  tiles.
- `64` — useful for rapid iteration or very large stacks.

**Symptom of wrong value:** if the estimated flat-field looks blocky or
misses a known gradient pattern, increase `working_size`.

---

### `epsilon`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.1` |

Regularisation constant $\varepsilon$ used in the reweighted-L1 weight
update:

$$
W_{i,p} = \frac{1}{|E_{i,p} / (\bar{B} + \varepsilon)| + \varepsilon}
$$

A smaller `epsilon` concentrates weights on the largest residual entries
(sharper sparsity); a larger value provides more uniform weighting (closer
to plain L1).

**Symptom of wrong value:** if the corrected images show ring or halo
artefacts, try increasing `epsilon` to `0.5`.  If the flat-field looks
under-corrected, try decreasing to `0.01`.

---

### `l_s` (flat-field regularisation)

| | |
|---|---|
| **Type** | `float \| None` |
| **Default** | `None` (auto-tuned) |

Regularisation weight $\lambda_s$ for the DCT-domain sparsity of the
flat-field.  Auto-tuned by {meth}`~pybasic.core.BaSiC.prepare` as:

$$
\lambda_s = \frac{\|\tilde{\bar{D}}\|_1}{800}
$$

where $\tilde{\bar{D}}$ are the DCT-II coefficients of the normalised mean
image.

**Physical meaning:** larger $\lambda_s$ forces the flat-field to be
smoother (fewer DCT terms survive thresholding); smaller $\lambda_s$ allows
more spatial variation.

**When to tune manually:**
- Flat-field looks **over-smooth** (misses real gradient) → decrease `l_s`
  (e.g. halve it).
- Flat-field shows **noise or texture** from the sample → increase `l_s`
  (e.g. double it).

**Usage:**

```python
model = BaSiC(stack)
model.prepare()          # auto-tunes l_s
print(model.l_s)         # inspect the auto-tuned value
model.l_s = model.l_s * 2   # manual override before run()
model.run()
```

---

### `l_d` (dark-field regularisation)

| | |
|---|---|
| **Type** | `float \| None` |
| **Default** | `None` (auto-tuned) |

Regularisation weight $\lambda_d$ for the dark-field.  Auto-tuned as:

$$
\lambda_d = \frac{\|\tilde{\bar{D}}\|_1}{2000}
$$

Only effective when `estimate_darkfield=True`.

**Physical meaning:** larger $\lambda_d$ forces the dark-field toward zero
(useful when genuine dark-field is minimal); smaller $\lambda_d$ allows a
richer dark-field profile.

**When to tune manually:** if the dark-field absorbs low-frequency scene
content, increase `l_d` (e.g. `model.l_d = model.l_d * 3`).

---

### `reweighting_tolerance`

| | |
|---|---|
| **Type** | `float` |
| **Default** | `1e-3` |

Convergence threshold for the outer reweighting loop.  The loop stops when:

$$
\max\!\left(\frac{\|S^{(k)} - S^{(k-1)}\|_1}{\|S^{(k-1)}\|_1},\;
            \frac{\|B^{(k)} - B^{(k-1)}\|_1}{\|B^{(k-1)}\|_1}\right)
\leq \text{reweighting\_tolerance}
$$

Tighter tolerance (e.g. `1e-4`) gives more stable results at the cost of
more iterations.  Looser tolerance (e.g. `5e-3`) speeds up fitting for
exploratory work.

---

### `max_reweighting_iterations`

| | |
|---|---|
| **Type** | `int` |
| **Default** | `10` |

Hard cap on the number of outer reweighting iterations regardless of
convergence.  Increase to `20` if the flat-field is still changing
noticeably at iteration 10 (check with `verbose=True`).

---

## ALM solver parameters

These are passed through to {func}`~pybasic.algorithms.inexact_alm_l1` via
{meth}`~pybasic.core.BaSiC.update`.  To override them, call `update()`
directly or subclass {class}`~pybasic.core.BaSiC`.

### `tol` (ALM convergence tolerance)

| | |
|---|---|
| **Default** | `1e-6` |

Relative Frobenius-norm residual threshold for the inner ALM loop.
Rarely needs changing.

---

### `max_iter` (ALM iterations)

| | |
|---|---|
| **Default** | `500` |

Maximum number of ALM iterations per reweighting step.

---

### `rho` (multiplier growth factor)

| | |
|---|---|
| **Default** | `1.5` |

Controls how quickly the Lagrange multiplier $\mu$ grows between ALM
iterations: $\mu^{(k+1)} = \rho \cdot \mu^{(k)}$.

- **Larger $\rho$** (e.g. `2.0`) — faster convergence per reweighting step
  but may overshoot and oscillate.
- **Smaller $\rho$** (e.g. `1.1`) — slower but more numerically stable for
  ill-conditioned stacks.

---

## Worked example: troubleshooting a bad flat-field

```python
import numpy as np
from pybasic import BaSiC

stack = load_my_stack()          # shape (N, H, W)
model = BaSiC(stack, estimate_darkfield=True, verbose=True)

# Step 1: inspect auto-tuned parameters
model.prepare()
print(f"l_s = {model.l_s:.4f},  l_d = {model.l_d:.4f}")

# Step 2: if flat-field looks noisy, increase l_s
model.l_s *= 2.0
model._flag_reweighting = True   # reset convergence flag
model.reweighting_iteration = 0

# Step 3: re-run from current state
model.run()
```

---

## Quick-reference table

| Parameter | Where set | Default | Effect of increasing |
|---|---|---|---|
| `estimate_darkfield` | `__init__` | `False` | Enables dark-field estimation |
| `working_size` | post-init | `128` | More spatial resolution, slower |
| `epsilon` | post-init | `0.1` | More uniform L1 weights |
| `l_s` | post-init / auto | `dct_sum/800` | Smoother flat-field |
| `l_d` | post-init / auto | `dct_sum/2000` | Sparser dark-field |
| `reweighting_tolerance` | post-init | `1e-3` | Fewer outer iterations |
| `max_reweighting_iterations` | post-init | `10` | More outer iterations allowed |
| `tol` (ALM) | `inexact_alm_l1` | `1e-6` | Tighter inner convergence |
| `rho` (ALM) | `inexact_alm_l1` | `1.5` | Faster but less stable |
