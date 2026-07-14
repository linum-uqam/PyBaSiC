# Adaptive `working_size` selection — design

> **Status:** S01 (design) and S02 (implementation) are shipped. S03 has now
> validated the resolver against the real-subject K01 seam gate (M006, July
> 2026): the quality-floor **raise branch was NOT promoted** — both raise
> targets (`160`, `192`) failed `seam_l1`/`seam_curvature` parity against the
> `ws=128` baseline on `sub-22`, so `"auto"` stays **opt-in** (requirement
> R058 not validated). See
> `scripts/experiments/s03_artifacts/S03-DECISION.md` for the full verdict.

This document specifies how `fit_mosaic` and `tune` should resolve
`working_size="auto"` to a concrete integer. It defines (1) the cheap
**signal inputs** the decision consumes, (2) the **selection rule** that
maps those signals to a value drawn from the already-validated Optuna grid,
(3) a strict **cost bound** that forbids any probe fit, (4) a **quality-safety
clause** anchored to the `ws=128` production baseline, (5) the **S02
integration contract**, and (6) the **observability contract** that makes
every resolution inspectable.

---

## Background and scope

`working_size` is the resolution BaSiC resizes every image to before
optimisation (`linum_basic.core.BaSiC`, default `128`). The estimated
flat-field is up-sampled back to the native tile resolution after `run()`.
It therefore controls *how much spatial detail the flat-field can represent*,
and — secondarily — the per-solve working-set memory and the per-iteration
ALM cost.

Three established decisions bound this design:

- **D002 / MEM002 — `ws=128` is the production default.** The A6000 sweep
  showed that `ws < 128` is *not materially faster* (ratio `< 1.30`, the
  D007 / MEM007 "materially faster" threshold) at quality parity on real
  subject data. `ws=64` and `ws=96` were therefore **not** adopted. Any
  adaptive rule must treat `128` as the safe default and never silently
  regress a caller to a smaller size for a speed reason that the evidence has
  already refuted.

- **D017 / MEM067 — the speed door at `ws=128` is closed.** The M005 research
  spike established that the aspirational 30% steady-state fit-time reduction
  is not credibly reachable from any single remaining lever on the production
  eager path (the dominant cost is the outer reweighting loop on a
  memory-bandwidth-bound eager path; `torch.compile` fusion is
  load-bearing-off per D005; half-precision is precision-gated by K02/K06).
  Consequently **adaptive `working_size` is not a speed feature.** Its value
  proposition is *robustness and quality headroom*, not wall-clock reduction:

  1. **Memory safety (robustness).** Automatically shrink the working size to
     fit constrained hardware (laptops, shared GPUs, large per-z mosaic
     volumes) instead of OOM-ing, with graceful, explained degradation.
  2. **Flat-field resolution headroom (quality).** On large-tile datasets
     where the illumination pattern has resolvable fine structure that a
     `128 × 128` grid undersamples, use a larger size to represent it; on
     smooth datasets, stay at `128` and spend no extra compute.

- **K01 — the real-subject seam gate is the release gate.** Any change that
  can affect reconstruction quality (including promoting a non-128 size) must
  pass `seam_l1` / `seam_curvature` parity against the `ws=128` baseline on
  real subject data (`linum_basic.benchmark.quality`). Synthetic parity alone
  is never sufficient.

The already-validated candidate set is the Optuna grid
`linum_basic.tuning._DEFAULT_SEARCH_SPACE["working_size"] == [64, 96, 128, 160, 192]`.
The selector **only ever returns values from this grid** — it never invents a
size outside it.

---

## Signal inputs

All inputs are *cheap*: a handful of shape/memory queries plus at most one
small DCT. None requires running BaSiC or the ALM solver. The resolver
consumes a frozen context dataclass, analogous to
`linum_basic.benchmark.strategies.WorkloadContext`.

| Signal | Source | Role in the rule |
|---|---|---|
| `n_z` | mosaic workload (`MosaicGrid.n_z`) | Cost/memory scaling; preview subsample bound. |
| `n_tiles` | mosaic workload | Per-solve working-set scales with tile count. |
| `tile_shape` | representative tile `(H, W)` | Quality-floor signal: large tiles can carry flat-field structure a `128²` grid cannot represent. |
| `field_mode` | `"per-z"` or `"global"` | `global` averages fields across z, lowering per-plane memory pressure. |
| `memory_budget_bytes` | explicit `memory_budget_bytes` kwarg, else `torch.cuda.mem_get_info(device)[0]` on CUDA, else host virtual memory | Hard memory ceiling — the *only* legitimate reason to select a size **below** 128. |
| `preview_quality` | derived from `linum_basic.core.dct_energy` on a fixed-resolution mean image (see below) | Quality-floor signal: high-frequency energy fraction indicates the flat-field has structure beyond what `128²` resolves. |

### Preview-quality signal (DCT-energy based)

`preview_quality` reuses the existing `linum_basic.core.dct_energy` primitive
(the same scale-invariant DCT sum BaSiC uses to auto-tune `l_s` / `l_d` in
`BaSiC.prepare`). To make the signal **comparable across datasets and
independent of the candidate `working_size`**, S02 computes it at a *fixed*
preview resolution `PREVIEW_RESOLUTION` (default `256`), not at any candidate
size:

1. Form the mean image from a **bounded subsample** — at most
   `min(n_z, PREVIEW_Z_SAMPLE)` z-planes (default `PREVIEW_Z_SAMPLE = 8`,
   matching the `min(n_z, 8)` chunk convention already in
   `build_workload_context`), meaned across tiles.
2. Resize to `PREVIEW_RESOLUTION × PREVIEW_RESOLUTION` (one `cv2.resize`).
3. Compute the 2-D DCT and a **high-frequency energy fraction**: the share of
   total `|DCT|` energy lying in spatial frequencies above the Nyquist limit
   of a `128 × 128` grid. Call this `hf_fraction ∈ [0, 1]`.

`hf_fraction` near `0` means the flat-field is smooth and `128` resolves it
fully (no reason to enlarge). A large `hf_fraction` means the `128²`
representation throws away real structure (candidate reason to enlarge, gated
by the quality-safety clause below).

---

## Selection rule

The resolver is a **pure function** `resolve_working_size(context) -> int`
(analogous to `resolve_auto_strategy` / `WorkloadContext` in
`linum_basic.benchmark.strategies`). It always returns a member of the grid
`{64, 96, 128, 160, 192}`, and always `>= 64`.

```python
GRID = (64, 96, 128, 160, 192)              # _DEFAULT_SEARCH_SPACE["working_size"]
SAFE_DEFAULT = 128                           # D002 / MEM002

def resolve_working_size(ctx) -> int:
    candidate = SAFE_DEFAULT                 # (1) start from the safe baseline

    # (2) MEMORY CEILING — the only reason to go BELOW 128.
    feasible = [s for s in GRID if peak_memory(s, ctx) <= ctx.memory_budget_bytes]
    if feasible:
        at_or_below = [s for s in feasible if s <= SAFE_DEFAULT]
        candidate = max(at_or_below) if at_or_below else min(feasible)
        # Prefer 128 if it fits; otherwise the largest grid size that fits.
    # If NO size fits (even 64) OR memory_budget is unavailable/ambiguous,
    # candidate stays 128 (safe default) — see quality-safety clause.

    # (3) QUALITY FLOOR — opt-in reason to go ABOVE 128.
    if (candidate == SAFE_DEFAULT
            and ctx.preview_quality is not None
            and ctx.preview_quality > WORKING_SIZE_QUALITY_RAISE_THRESHOLD
            and 160 in feasible):            # memory must permit enlarging
        candidate = max(s for s in feasible if s > SAFE_DEFAULT)  # at most 192

    return candidate
```

The two branches are deliberately asymmetric, and that asymmetry is the whole
point of the evidence above:

- **Shrink (memory ceiling)** is a *corrective* branch: it only fires when
  `128` genuinely does not fit the memory budget, and it shrinks the *least*
  amount that fits (largest feasible `≤ 128`, never smaller than necessary).
- **Enlarge (quality floor)** is a *quality-opportunity* branch: it only
  enlarges when (a) `128` already fits, (b) the preview signal indicates real
  unrepresentable structure, (c) memory permits `≥ 160`, and (d) the
  quality-safety clause (below) is satisfied. Until S03 validates it, this
  branch is effectively inert.

`peak_memory(ws, ctx)` reuses the existing
`linum_basic.benchmark.strategies.estimate_strategy_vram_bytes` shape — a
conservative `activation_factor × n_tiles × ws² × bytes_per_elem` estimate
(`activation_factor = 4`, `bytes_per_elem = 4` for float32, matching the
existing estimator). For `per-z` mode the per-solve chunk is one z-plane, so
`peak_memory(ws, ctx) = estimate_strategy_vram_bytes(1, n_tiles, ws)`.

---

## Cost bound: no probe fits

Selection **must not run a BaSiC or ALM solve at any candidate `working_size`.**
Concretely, the resolver's total cost is bounded by:

1. A handful of shape queries (`n_z`, `n_tiles`, `tile_shape`, `field_mode`)
   — O(1) metadata reads already available on `MosaicGrid`.
2. One memory query (`torch.cuda.mem_get_info` or host memory) — O(1).
3. At most **one small DCT**: forming the preview mean from
   `min(n_z, 8)` planes, resizing to `256 × 256`, and one `dctn` — bounded,
   predictable, and many orders of magnitude cheaper than a single ALM solve.

The rule is explicitly forbidden from:

- Running `BaSiC.prepare()` / `BaSiC.run()` at any candidate size.
- Calling `inexact_alm_l1` / `shrink` / any solver primitive.
- Executing a full or partial `fit_mosaic` / `tune` at a candidate size to
  "measure" quality or speed (no probe fit, no measure-then-pick).

This bound is **checkable**: S02's tests assert that calling the resolver
does not instantiate `BaSiC` or touch any `_alm` / `fit` solve path (e.g. via
a test that monkeypatches the solver entry points to raise, and confirms the
resolver still returns).

---

## Quality-safety clause

S03's real-subject K01 evaluation has been completed, with a **no-go**
verdict on promotion. Adaptive selection is therefore **opt-in and
conservative**:

1. **Opt-in only.** The default value of `working_size` everywhere
   (`fit_mosaic`, `tune`, the `basic` CLI, `BaSiC`) remains the integer `128`.
   Adaptive selection activates *only* when a caller explicitly passes
   `working_size="auto"`. Existing callers are byte-for-byte unaffected.
   **S03 did not change this:** R058 (auto as default) was not validated.

2. **Fail-safe to `128`.** Whenever a required signal is **unavailable or
   ambiguous**, `"auto"` MUST resolve to `128`. This covers:
   - `memory_budget_bytes` cannot be determined (no CUDA, no host query, no
     explicit kwarg);
   - the preview mean cannot be formed (e.g. unreadable tiles);
   - `preview_quality` is `None` or outside `[0, 1]`;
   - `n_tiles` or `tile_shape` are unknown (non-mosaic inputs).

   The resolution records `fallback_reason` in the observability metadata so
   the operator can see *why* `128` was chosen.

3. **Raise branch is gated — and NOT promoted.** S03 (M006) ran the raise
   branch on the production-shaped subject `sub-22` on the A6000 and it
   **FAILED the K01 gate**: requesting `working_size="auto"` fired the raise
   branch (`auto -> 192`) and failed both `seam_l1` (+5.77%) and
   `seam_curvature` (+13.53%) while running 9.62x slower; an explicit
   `working_size=160` failed `seam_l1` (+15.29%, 3.73x slower) too. A `> 128`
   size **cannot** promote toward the default until a future milestone
   demonstrates, on real subject data, that the larger size holds `seam_l1` /
   `seam_curvature` parity with the `ws=128` baseline (K01) **and** that the
   larger size does not regress the D009 / K03 strategy invariants. Promotion
   of `"auto"` to a default is owned by R058 and is **not validated** by S03;
   the raise branch stays opt-in and quality-risky. See
   `scripts/experiments/s03_artifacts/S03-DECISION.md`.

4. **Shrink branch is always safe.** The memory-ceiling (shrink) branch may
   fire immediately, because selecting the largest feasible size `≤ 128` is
   strictly safer than OOM-ing and the evidence (D002) already establishes
   quality parity at `≤ 128`.

`WORKING_SIZE_QUALITY_RAISE_THRESHOLD` is a named, conservative constant.
S03 found the default value (`0.15`) is low enough to fire on real per-z
illumination data (it fired on `sub-22`, resolving `auto -> 192`), but the
raise targets it selects to (`160`, `192`) then failed the K01 seam gate on
that same subject. The value is **kept at `0.15`** — no evidence-based
replacement number is available (the exact `sub-22` preview signal value was
not persisted into the harness artifact JSON), so an upward recalibration is
deferred to a future milestone that closes that observability gap. The
constant's docstring in `linum_basic/_working_size.py` records this outcome.

---

## S02 integration contract

Slice S02 implements this design as follows (S01 specifies, does not build):

- **New pure resolver.** A new function
  `resolve_working_size(context: WorkingSizeContext) -> WorkingSizeResolution`
  in a new module (e.g. `linum_basic/_working_size.py`), together with a
  frozen `WorkingSizeContext` dataclass and a `WorkingSizeResolution`
  result dataclass carrying the chosen `int` plus the explainability
  metadata. This mirrors the `WorkloadContext` / `AutoStrategyResult` /
  `resolve_auto_strategy` precedent in
  `linum_basic/benchmark/strategies.py`.

- **`fit_mosaic` wiring.**
  `fit_mosaic(..., basic_kwargs={"working_size": "auto"})` detects the
  `"auto"` sentinel before the existing `int(user_kwargs.get("working_size",
  128))` coercion, builds a `WorkingSizeContext`, calls the resolver, and
  replaces the sentinel with the resolved `int` in the params that flow into
  `BaSiC`. A literal `"auto"` must never reach `BaSiC.__init__`.

- **`tune` wiring.** `tune(..., working_size="auto")` resolves the sentinel
  to a concrete `int` (or, alternatively, to the *subset* of the grid the
  rule deems feasible) before constructing the Optuna
  `_DEFAULT_SEARCH_SPACE`. The resolved value / feasible subset is recorded
  so the search is reproducible.

- **`basic` CLI.** The CLI accepts `--working-size auto` in the `fit`,
  `tune`, and `preview` subcommands, forwarding the sentinel to the
  library functions above. (The CLI already validates device/backends at
  argparse time; the `"auto"` acceptance follows the same pattern.)

- **Reuse, not duplication.** The resolver reuses
  `estimate_strategy_vram_bytes`, `dct_energy`, and the
  `WorkloadContext`-style construction. It does **not** duplicate strategy
  resolution: it runs *before* `resolve_auto_strategy` and only fixes the
  `working_size` field; the existing strategy resolver then sees a concrete
  `int`, preserving the K03 / D009 `ws ≥ 128` batched-CUDA guard verbatim.

---

## Observability contract

Every resolution is recorded for reproducibility, mirroring the existing
`params["_strategy"]` explainability pattern (D-19) in `fit.py`. The resolved
`working_size` and the reasoning behind it are stored in a nested
`params["_working_size_selector"]` sub-dict on `MosaicFit.params`:

```python
params["_working_size_selector"] = {
    "resolved_working_size": 128,                 # the concrete int chosen
    "requested": "auto",                          # "auto" or an explicit int
    "candidate_grid": [64, 96, 128, 160, 192],    # the grid used
    "rule_path": "baseline-default",              # see paths below
    "fallback_reason": None,                      # why 128 was chosen, if it was
    "gate_status": "opt-in (raise branch not promoted)", # promotion state (S03 no-go)
    "signals": {
        "n_z": 41,
        "n_tiles": 72,
        "tile_shape": [2048, 2048],
        "field_mode": "per-z",
        "memory_budget_bytes": 42949672960,
        "preview_quality": 0.03,                  # hf_fraction, or None
    },
    "peak_memory_estimate_bytes": {               # per candidate, for audit
        "64":  ...,
        "96":  ...,
        "128": ...,
        "160": ...,
        "192": ...,
    },
}
```

`rule_path` is one of:

- `"baseline-default"` — no branch fired; `128` is feasible and sufficient.
- `"memory-ceiling-shrink"` — `128` did not fit; returned the largest feasible
  size `≤ 128` (or `64` if even `96` did not fit).
- `"quality-floor-raise"` — `128` fit but the preview signal indicated real
  unrepresentable structure and memory permitted; returned `160` or `192`.
- `"fallback-safe-default"` — a required signal was unavailable/ambiguous;
  returned `128` with a populated `fallback_reason`.

This lets a future agent (or operator) inspect **why a given size was chosen
without re-deriving it**, satisfying the slice's observability requirement.
The metadata is purely additive; it never affects the numerics of the fit.

---

## Validation plan (S03) — outcome: NO-GO on promotion

S03 has been executed (M006, July 2026) with a **no-go** verdict: the
quality-floor raise branch was **not promoted**. The runs and their outcomes:

1. **K01 real-subject A/B (DONE — both raise candidates REJECTED).** Ran the
   resolver and explicit raise-target sizes on the production-shaped subject
   `sub-22` on the A6000, comparing `seam_l1` / `seam_curvature` against a
   fresh `ws=128` baseline via `linum_basic.benchmark.quality`'s mean+3std
   gate:
   - `working_size="auto"` fired the raise branch (`auto -> 192`): REJECT,
     `seam_l1` +5.77%, `seam_curvature` +13.53%, 9.62x slower.
   - explicit `working_size=160` (smallest raise target): REJECT, `seam_l1`
     +15.29% (curvature improved -9.57%, but K01 needs both), 3.73x slower.
   - The baseline-default path (`auto -> 128`) was *not* exercised on this
     subject because the raise branch fired first.
2. **Cost-bound verification (DONE in S02).** A test proves the resolver
   never invokes a solver (`tests/test_working_size_resolver.py::TestCostBound`).
3. **Fail-safe verification (DONE in S02).** Tests assert each ambiguous /
   missing-signal path resolves to `128` with the correct `fallback_reason`
   (`tests/test_working_size_resolver.py::TestFailSafe`).
4. **Threshold calibration (RECORDED — value kept at 0.15).** S03 found the
   threshold fires on real per-z data but the raise *targets* fail K01; the
   value is kept and the outcome is recorded in the constant's docstring and
   in `scripts/experiments/s03_artifacts/S03-DECISION.md`. A principled
   upward recalibration is deferred.

Because (1) did not pass, `"auto"` remains opt-in and the raise branch
stays conservative. Promotion of the raise branch (or of `"auto"` as a
default) would require, at minimum, a real-subject dataset where `160`/`192`
holds K01 parity, closure of the `_working_size_selector` explainability gap
in the harness artifact JSON, and multi-subject evidence. Full synthesis:
`scripts/experiments/s03_artifacts/S03-DECISION.md`.

---

## Non-goals

- **No speed optimisation.** Per D017 / MEM067, the `ws=128` speed ceiling is
  ~2.2% and closed; this feature does not pursue wall-clock reduction.
- **No new grid values.** Sizes outside `{64, 96, 128, 160, 192}` are out of
  scope.
- **No change to the ALM solver or BaSiC invariants** (K02). The resolver
  only picks an integer; it touches no numerics.
- **No change to strategy resolution.** The K03 / D009 `ws ≥ 128` batched-CUDA
  guard and the `resolve_strategy` precedence are preserved unchanged; the
  resolver runs upstream of them and supplies a concrete `int`.
- **No silent default change.** `"auto"` is opt-in until S03 + R058.
