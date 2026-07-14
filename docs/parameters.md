(parameters)=
# Parameter Tuning

This page describes every tuning knob exposed by
{class}`~linum_basic.core.BaSiC` — what each parameter controls physically,
its default value, the heuristic used to auto-set it, and common
troubleshooting recipes.

---

## Quick reference

| Parameter | Where set | Type | Default | One-liner |
|---|---|---|---|---|
| `estimate_darkfield` | constructor | `bool` | `False` | Estimate the dark-field $B(p)$ alongside the flat-field |
| `extension` | constructor | `str` | `".tif"` | File-extension glob when input is a directory |
| `backend` | constructor | `str` | `"numpy"` | Compute backend (`"numpy"`, `"torch"`, `"auto"`) |
| `device` | constructor | `str\|None` | `None` | PyTorch device string (e.g. `"cuda:0"`) |
| `working_size` | post-init | `int` | `128` | Resize resolution; higher = more spatial detail |
| `epsilon` | post-init | `float` | `0.1` | Reweighting stability constant |
| `l_s` | post-init | `float\|None` | auto | Flat-field DCT regularisation weight ($\lambda_s$) |
| `l_d` | post-init | `float\|None` | auto | Dark-field regularisation weight ($\lambda_d$) |
| `reweighting_tolerance` | post-init | `float` | `1e-3` | Outer-loop convergence threshold |
| `max_reweighting_iterations` | post-init | `int` | `10` | Hard cap on outer iterations |
| `convergence_check_every` | post-init | `int` or `None` | `None` | Inner ALM sync cadence (`None` = backend default) |
| `warm_start_reweighting` | post-init | `bool` | `False` | Warm-start ALM primal variables between outer iterations |

---



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

PyTorch device string, e.g. `"cuda:0"`.  Apple MPS is not supported; on
Apple Silicon use `backend="numpy"` or `backend="auto"`.  See
{ref}`GPU Acceleration <gpu>`.  Ignored when `backend="numpy"`.

---

## Post-init tuning knobs

These attributes can be set **after** construction and **before** calling
{meth}`~linum_basic.core.BaSiC.prepare`:

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

#### `working_size="auto"` (adaptive selection)

| | |
|---|---|
| **Type** | `"auto"` sentinel (accepted by {func}`~linum_basic.fit.fit_mosaic`, {func}`~linum_basic.tuning.tune`, and the `basic fit` / `basic tune` CLI subcommands) |
| **Default** | `128` (the BaSiC class still takes an `int`; `"auto"` is opt-in at the fit/tune layer) |

{class}`~linum_basic.core.BaSiC` itself always receives a concrete
integer — the literal `"auto"` **never** reaches the solver.  When a
caller passes `working_size="auto"`, the fit/tune layer resolves the
sentinel to a concrete integer drawn from the validated grid
`{64, 96, 128, 160, 192}` *before* {class}`~linum_basic.core.BaSiC` is
constructed.  This makes the adaptive rule purely additive: callers that
never pass `"auto"` are byte-for-byte unaffected, and the production
default remains `128`.

The resolver is a **pure function** of cheap shape/memory signals — it
never runs a probe fit at any candidate size (the *cost bound*).  It
implements two deliberately asymmetric branches:

1. **Memory-ceiling shrink** (always safe).  When the memory budget is
   known and `128` does not fit, it returns the largest feasible size
   `≤ 128` so constrained hardware (laptops, shared GPUs, large per-z
   volumes) degrades gracefully instead of out-of-memory-ing.  If even
   `64` does not fit, it fails safe to `128`.
2. **Quality-floor raise** (opt-in, gated).  When `128` fits, the preview
   DCT signal indicates high-frequency flat-field structure a `128²` grid
   cannot represent, and memory permits enlarging, it returns the largest
   feasible size `> 128` (i.e. `160` or `192`).

Whenever a required signal is unavailable or ambiguous (unknown memory
budget, unformable preview, out-of-range signal), the resolver **fails
safe to `128`** and records a human-readable `fallback_reason`.

**Where it is accepted:**

```python
from linum_basic import fit_mosaic, tune

# Python API — pass the sentinel through basic_kwargs / working_size
fit = fit_mosaic(mosaic, basic_kwargs={"working_size": "auto"})
result = tune(mosaic, working_size="auto")

# CLI — the --working-size flag accepts an integer or the literal 'auto'
# basic fit  --input subject.ome.zarr --output corrected.ome.zarr --working-size auto
# basic tune --input subject.ome.zarr --working-size auto
```

**Opt-in status.** `"auto"` is opt-in only.  Milestone M006 S03 evaluated
the quality-floor *raise* branch on real subjects (sub-22, A6000) against a
fresh `ws=128` baseline: both raise targets (`160`, `192`) **FAILED the
K01 `seam_l1` / `seam_curvature` gate** (and were 3.7x / 9.6x slower), so
the raise branch is **not promoted** and R058 (auto as default) is not
validated.  The production default stays the integer `128`.  Passing an
explicit integer always wins over `"auto"`.  See
`docs/adaptive_working_size.md` (Validation plan) and
`scripts/experiments/s03_artifacts/S03-DECISION.md` for the full verdict.

**Reproducibility / observability.**  Every resolution records full
explainability metadata, mirroring the existing `params["_strategy"]`
pattern.  After a fit it lives on
`fit.params["_working_size_selector"]`; after a tune it lives on
`result.working_size_selector`.  Both carry `resolved_working_size`,
`requested`, `candidate_grid`, `rule_path`
(`"baseline-default"` / `"memory-ceiling-shrink"` / `"quality-floor-raise"`
/ `"fallback-safe-default"`), `fallback_reason`, `gate_status`, `signals`,
and `peak_memory_estimate_bytes` per candidate — purely additive metadata
that never affects fit numerics.  See `docs/adaptive_working_size.md` for
the full design.

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
flat-field.  Auto-tuned by {meth}`~linum_basic.core.BaSiC.prepare` as:

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

### `convergence_check_every`

| | |
|---|---|
| **Type** | `int` or `None` |
| **Default** | `None` (backend default: every iteration on NumPy, every 10 on GPU) |

Controls how often the inner ALM loop evaluates the primal residual norm.
Higher values reduce GPU synchronisation overhead at the cost of slower
early exit inside a single reweighting pass.  Leave at the default unless
you are tuning performance with the benchmark harness.

---

### `warm_start_reweighting`

| | |
|---|---|
| **Type** | `bool` |
| **Default** | `False` |

When `True`, the primal variables from the previous inner ALM solve
($S_f$, $I_r$, $B$, $D$) are passed as a warm start to the next outer
reweighting iteration instead of re-initialising from zeros.  The
Lagrange multiplier $Y$ and step size $\mu$ are always reset so that
the changed weight matrix $\mathbf{W}$ does not cause divergence.

**When to enable:** only when running many reweighting iterations
(`max_reweighting_iterations` ≥ 5) on large stacks where each inner
ALM solve is expensive.  The warm start can reduce the total number of
inner iterations by providing a better initial point.

**When to leave off (default):** for typical usage the default cold
start gives identical results at negligible extra cost because the
inner ALM loop converges quickly from zero.  Warm-starting with
changed weights occasionally affects the dark-field estimation
sensitivity.

**Usage:**

```python
model = BaSiC(stack, estimate_darkfield=True)
model.max_reweighting_iterations = 15
model.warm_start_reweighting = True   # enable warm start
model.prepare()
model.run()
```

---

## ALM solver parameters

These are passed through to {func}`~linum_basic.algorithms.inexact_alm_l1` via
{meth}`~linum_basic.core.BaSiC.update`.  To override them, call `update()`
directly or subclass {class}`~linum_basic.core.BaSiC`.

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
from linum_basic import BaSiC

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
