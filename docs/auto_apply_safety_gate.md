# Auto-apply safety gate for `recommend_bounds` — design

> **Status:** S01 (this design) is the only M008 deliverable that lands here.
> S02 implements the guarded auto-apply loop against this contract; S03
> documents it in `docs/tuning.md` / `docs/runbooks.md` and runs an
> end-to-end validation. No production code in `linum_basic/` changes in
> this slice — this document is the spec both later slices build on.

This document specifies the **M008 auto-apply safety gate**: the mechanism
that lets the M003 `recommend_bounds` heuristic *auto-apply* its narrowed
bounds without ever shipping a worse correction than the default-bounds fit.
It defines (1) the **gate mechanism** — a fixed non-regression margin
comparison of full-volume `QualityReport` deltas between exactly one
default-bounds baseline fit and exactly one narrowed-bounds candidate fit
(D023); (2) a strict **cost bound** — exactly the `tune()` trial budget plus
two full-volume fits, never a repeated multi-run `mean+3std` calibration;
(3) the **fallback-to-defaults contract** for any quality regression or
degenerate trial history; (4) the explicit **distinction from the D015
advisory scope and from the `quality.py` `mean+3std` release gate**; (5) the
**S02 integration contract** (a new guarded `linum_basic.tuning.auto_tune`
entry point and a `basic tune --auto-apply` CLI flag); and (6) the
**observability contract** that records the gate verdict, deltas, and
fallback reason so a future agent or operator can inspect *why* auto-apply
did or did not apply the narrowed bounds.

---

## Background and scope

`linum_basic.tuning.recommend_bounds` (M003, D015 / MEM047–MEM048) collapses
the near-optimal region of a completed Optuna `tune()` study into a narrowed
`search_space` (in the scale-invariant divisor parametrisation). By design it
is **advisory only**: it is an operator input to a *subsequent*
`tune()` / `fit_mosaic()` call, never a release gate. Requirement **R057**
asks to close the loop to full automation — have the tuning workflow narrow
bounds *and apply them itself*, with no operator in the loop.

The blocker on full automation is safety. `tune()` minimises a *subsampled*
objective (a handful of z-levels and ≤ `max_tiles` tiles) on a single
objective (`seam_l1`, `curvature`, or `composite`). A subsampled, single
objective cannot prove that the best trial holds quality on the *full* volume
against *both* first-class metrics (`seam_l1` **and** `seam_curvature`). It
is especially unsafe because `tune()` may explore `working_size > 128`, and
the M006/S03 evidence (D020 / MEM081) shows `ws > 128` regresses real-subject
seam quality — exactly the failure an automated apply must never ship.

Three established decisions bound this design:

- **D015 — `recommend_bounds` is advisory, distinct from the release gate.**
  It collapses one Optuna study's trial history into a narrowed search space
  and never gates a production fit. Auto-apply adds a *new*, distinct
  mechanism on top of it; it does not redefine `recommend_bounds` and does
  not touch the benchmark A/B release-gate machinery.

- **D023 — the gate is a fixed non-regression margin on full-volume
  `QualityReport` deltas.** It reuses `linum_basic.benchmark.quality`'s
  `QualityReport` / `compute_quality_report` / `compute_deltas` primitives on
  full-volume `seam_l1` + `seam_curvature`, comparing exactly one
  default-bounds baseline fit against exactly one narrowed-bounds candidate
  fit. It is deliberately **not** the `calibrate_tolerances` /
  `evaluate_quality_gate` `mean+3std` release gate.

- **K01 — the real-subject seam gate is the release gate.** `seam_l1` /
  `seam_curvature` on full real volumes is the bar; a subsampled objective
  alone is never sufficient. The auto-apply gate enforces K01 *per dataset,
  at apply time*, which is the only place an automated workflow can enforce
  it without an operator.

This gate answers one question, once, per dataset: **"Is applying the tuned
best params at least as good as the default-bounds fit on the full volume?"**
If yes, apply; if no — for any reason — fall back to the default-bounds fit
and say why.

---

## Gate mechanism: fixed non-regression margin (D023)

The gate reuses two proven primitives from `linum_basic.benchmark.quality`:

- `compute_quality_report(mosaic, fit)` → a `QualityReport` whose
  `aggregates` dict holds full-volume `seam_l1` and `seam_curvature` scalars.
- `compute_deltas(candidate, baseline)` → an `aggregate` dict mapping each
  first-class metric to a `MetricDelta(abs_delta, rel_delta)`, where
  `abs_delta = candidate − baseline` and
  `rel_delta = abs_delta / max(|baseline|, rel_floor)`.

Both `seam_l1` and `seam_curvature` are **lower is better**, so a positive
delta (candidate worse than baseline) directly indicates a regression.

### Inputs

Exactly two full-volume `MosaicFit` objects, both produced by the existing
`linum_basic.fit.fit_mosaic` on the *same* mosaic:

1. **Baseline (default-bounds) fit.** `fit_mosaic(mosaic, basic_kwargs=<BaSiC
   defaults>)` — `working_size=128`, `estimate_darkfield=True`, and `l_s` /
   `l_d` left at `None` so `BaSiC.prepare()` auto-tunes them (the production
   default an operator gets without tuning). This is the floor the candidate
   must beat.
2. **Candidate (narrowed-bounds) fit.** `fit_mosaic(mosaic,
   basic_kwargs=tune_result.best_params)` — the single best parameter set
   `tune()` found, which by construction sits at the argmin of the
   `recommend_bounds` near-optimal region. (`tune_result.best_params` already
   carries resolved absolute `l_s` / `l_d`; see `tuning._basic_params`.)

### Non-regression rule

For each metric `m ∈ FIRST_CLASS_METRICS == ("seam_l1", "seam_curvature")`:

```python
NO_REGRESSION_MARGIN = 0.0   # D023 default: candidate must not be measurably worse
FLOAT_NOISE_EPS = 1e-9       # absorb pure float jitter in the aggregate scalars

def _regresses(delta: MetricDelta, *, margin: float) -> bool:
    # "lower is better" → candidate worse than baseline iff rel_delta > margin
    return delta.rel_delta > margin + FLOAT_NOISE_EPS

deltas = compute_deltas(candidate_report, baseline_report)["aggregate"]
failing = [m for m in FIRST_CLASS_METRICS if _regresses(deltas[m], margin=NO_REGRESSION_MARGIN)]
gate_verdict = "pass" if not failing else "fail"   # failing ⊆ {"seam_l1", "seam_curvature"}
```

The gate **passes only if neither first-class metric regresses** beyond
`NO_REGRESSION_MARGIN` (default `0.0`, i.e. the candidate must be ≤ the
baseline within float noise on both metrics). A regression on *either*
metric fails the gate, because K01 needs both. The margin is a single named,
overridable constant; the default `0.0` realises the milestone's literal
contract — *"never ships a worse correction than the default-bounds fit."*

### What this deliberately is *not*

- **Not `evaluate_quality_gate` / `calibrate_tolerances`.** That release-gate
  machinery calibrates an absolute + relative tolerance from a *distribution*
  of baseline repeats (`CALIBRATION_POLICY == "mean+3std"`,
  `abs_tol = max(min_abs, sigma * std)`). Calibrated against a *single*
  baseline report it degenerates: `std = 0`, so `abs_tol = min_abs = 1e-6` —
  an impossibly strict floor that rejects nearly every candidate on pure float
  jitter. Auto-apply is single-shot per dataset and can afford exactly one
  baseline fit, so the variance-estimating release gate is the wrong tool.
  The fixed-margin comparison is the right tool. See *Distinction from the
  advisory and release gates* below.
- **Not a comparison on the subsampled tuning objective.** The candidate's
  quality is measured on the *full* volume against *both* metrics, not on the
  `z_subsample` / `max_tiles` slice `tune()` optimised. This is what backs the
  "held-out full-volume" guarantee D023 requires.

---

## Cost bound: two full-volume fits, no N-repeat calibration

Auto-apply's total cost is bounded, predictable, and independent of the
quality check itself:

| Phase | Cost | Fits? |
|---|---|---|
| `tune(mosaic, n_trials=…)` | One Optuna study of subsampled per-z fits (`z_subsample`, `max_tiles` bounded) | subsampled trial fits (already what `tune` costs) |
| `recommend_bounds(result, margin=…)` | O(completed trials) DataFrame reduction | **no fits** |
| Baseline full-volume fit | exactly **1** `fit_mosaic` (default bounds) | 1 full fit |
| Candidate full-volume fit | exactly **1** `fit_mosaic` (`best_params`) | 1 full fit |
| `compute_quality_report` x2 + `compute_deltas` + margin check | metric computation on two `MosaicFit`s | **no fits** |

**Total: the `tune()` trial budget plus exactly two full-volume fits.** The
gate adds a constant — two fits — on top of tuning, never a repeated multi-run
`mean+3std` calibration (which would multiply the baseline fit by `N` repeats
to estimate variance). The two full-volume fits reuse `fit_mosaic` exactly as
`tune(run_full_fit=True)` already does for `TuneResult.best_fit`; no new solver
code, no ALM / backend changes (K02 invariants untouched).

This bound is **checkable**: S02's tests assert that, given a fixed
`tune_result`, the auto-apply path issues exactly two `fit_mosaic` calls
(counting them) and zero `calibrate_tolerances` / repeat-baseline calls.

---

## Load profile

The runtime load dimension of auto-apply is the **two full-volume fits**, and
both scale with dataset size (z-planes × tiles × tile resolution).

- **What saturates first at 10x load.** At 10x z-planes (or 10x tiles) the
  binding resource is the **per-z working-set memory of the full-volume
  `fit_mosaic`** — GPU VRAM on the CUDA path
  (`activation_factor × n_tiles × working_size² × bytes_per_elem`, the same
  shape `estimate_strategy_vram_bytes` uses) and host RSS for the
  materialised mosaic on the NumPy path. The two fits are sequential and
  independent, so their peak is one fit's peak, not two.
- **Protection applied.**
  1. The fixed cost bound caps the *count* of full fits at two regardless of
     dataset size — there is no `N`-repeat calibration whose cost grows with
     the desired confidence.
  2. The candidate and baseline fits inherit every existing memory guard:
     `fit_mosaic(streaming=True)` (M003/S02, MEM042) bounds host RSS to
     ~one z-plane's tile stack; the strategy resolver preserves the K03 / D009
     `ws ≥ 128` batched-CUDA guard; `strategy="auto"` never selects the
     memory-bandwidth-bound batched path at production resolution.
  3. The `tune()` phase cost is decoupled from full volume by `z_subsample`
     and `max_tiles`, so tuning never sees the 10x load directly.
- **No new saturation risk is introduced.** Auto-apply issues the same two
  `fit_mosaic` shapes an operator would issue by hand; it adds no
  multiprocessing fan-out, no in-memory caching of full volumes between the
  two fits, and no network/filesystem fan-out beyond what `fit_mosaic`
  already does.

---

## Failure modes and fallback-to-defaults contract

Auto-apply's central invariant is the **fallback-to-defaults contract**: on
*any* path where the narrowed bounds cannot be proven safe, auto-apply
returns the default-bounds baseline fit and records why. The contract covers
every external dependency's failure path.

| Dependency | Failure mode | Handling (→ fallback) |
|---|---|---|
| `tune()` (Optuna) | `optuna` not installed (`ImportError`); every trial pruned → no `study.best_trial` (`ValueError`); OOM in a trial fit | catch → fallback, `fallback_reason="tune-failed"` |
| `recommend_bounds()` | `trials_df is None` (pandas unavailable) or empty; **zero `COMPLETE` trials**; `margin ∉ [0, 1]` — all raise `ValueError` | catch → fallback, `fallback_reason="degenerate-trial-history"` (zero `COMPLETE`) or `"invalid-margin"` |
| Baseline full fit (`fit_mosaic`) | OOM, unreadable tiles, ALM divergence / convergence failure | catch → the whole auto-apply is unsafe; raise a single `AutoApplyError` carrying the partial state (the default fit itself failed, so there is no safe floor to return) |
| Candidate full fit (`fit_mosaic`) | OOM, unreadable tiles, convergence failure | catch → fallback to the (already-computed) baseline fit, `fallback_reason="candidate-fit-failed"` |
| `compute_quality_report` / `compute_deltas` | candidate and baseline z-index sets differ → `ValueError`; non-finite aggregate | catch → fallback, `fallback_reason="quality-report-failed"` |
| Non-regression gate | candidate worse than baseline on `seam_l1` **or** `seam_curvature` beyond `NO_REGRESSION_MARGIN` | deterministic → fallback, `gate_verdict="fail"`, `fallback_reason="regression-detected"` (+ the failing metrics) |

Two clarifications on the degenerate-history cases:

1. **Zero `COMPLETE` trials.** If every Optuna trial was pruned or failed,
   `recommend_bounds` raises `ValueError("…at least one COMPLETE trial…")`.
   Auto-apply treats this as `fallback_reason="degenerate-trial-history"` and
   returns the baseline fit. (See `recommend_bounds`'s `COMPLETE`-state
   guard in `tuning.py`.)
2. **Single-trial near-optimal set is *valid*, not a failure.**
   `recommend_bounds` collapses a one-trial near-optimal subset into point
   ranges (`low == high`), which are explicitly valid. That is a narrow — not
   degenerate — recommendation; it still flows through the full-volume gate.
   The fallback fires only when safety genuinely cannot be established
   (regression, exceptions, zero `COMPLETE`), never merely because the region
   is narrow.

The fallback fit is always the **already-computed baseline (default-bounds)
fit**, so fallback adds no extra full-volume solve. The only path that does
not fall back gracefully is a *baseline-fit* failure: if the default-bounds
fit itself cannot be produced, there is no safe floor to return, and auto-apply
raises a single `AutoApplyError` (a typed exception) carrying the partial
state and the underlying cause rather than silently returning an undefined
correction.

---

## Distinction from the advisory and release gates (D015)

There are now **three distinct mechanisms** in this area. Conflating them is
the primary design error this section exists to prevent.

| Mechanism | Lives in | What it decides | How many fits | Repeats / calibration |
|---|---|---|---|---|
| **`recommend_bounds`** (advisory) | `linum_basic.tuning` | Narrows a *search space* for a follow-up `tune()` | 0 (reads a `TuneResult`) | none — collapses one study's trial history |
| **Auto-apply non-regression gate** (this design, D023) | `linum_basic.tuning.auto_tune` | Per-dataset, single-shot: *apply* the tuned params only if not worse than default on the full volume | exactly 2 full-volume fits (baseline + candidate) | none — fixed margin, no variance estimate |
| **`mean+3std` release gate** | `linum_basic.benchmark.quality` (`calibrate_tolerances` / `evaluate_quality_gate`) | A/B-harness promotion: is a candidate *strategy* as good as a frozen baseline? | N repeat baseline fits + candidate | yes — `CALIBRATION_POLICY == "mean+3std"` estimates std from repeats |

- The **advisory** mechanism (`recommend_bounds`) answers "what region should
  the next search focus on?" — it never gates a production fit (D015).
- The **auto-apply gate** (this design) answers "is it safe to *use* the tuned
  params on *this* dataset right now?" — a single-shot, per-dataset decision
  with a fixed margin and no variance estimation.
- The **release gate** (`quality.py`) answers "is this candidate strategy
  releasable against a frozen baseline across repeats?" — it estimates
  variance from `N` repeats and is the A/B-harness promotion bar (K01 at
  release time).

The auto-apply gate **reuses the release gate's *primitives***
(`QualityReport`, `compute_quality_report`, `compute_deltas`, the
`FIRST_CLASS_METRICS` tuple, the `MetricDelta` shape) but **not its *policy***
(`calibrate_tolerances` / `evaluate_quality_gate` / `CALIBRATION_POLICY`).
This keeps the two mechanisms architecturally separate per D015 while sharing
the proven, K01-aligned metric definitions — so the per-dataset gate and the
release gate can never disagree on *what* a metric *means*, only on *how
strictly* it is applied.

---

## S02 integration contract

Slice S02 implements this design (S01 specifies, does not build). No
`linum_basic/` numerics change; the implementation composes existing surfaces.

- **New guarded entry point.** A new function
  `linum_basic.tuning.auto_tune(mosaic, *, n_trials=…, margin=…,
  no_regression_margin=0.0, …) -> AutoTuneResult` that runs the full pipeline:

  1. `result = tune(mosaic, n_trials=…, …)` (one Optuna study; reuses
     `tune()` verbatim, including its `working_size` / objective handling).
  2. `rec = recommend_bounds(result, margin=margin)` (narrowed `search_space`
     + degeneracy guard).
  3. `baseline_fit = fit_mosaic(mosaic, basic_kwargs=<BaSiC defaults>)`.
  4. `candidate_fit = fit_mosaic(mosaic, basic_kwargs=result.best_params)`.
  5. `baseline_report = compute_quality_report(mosaic, baseline_fit)`;
     `candidate_report = compute_quality_report(mosaic, candidate_fit)`.
  6. `deltas = compute_deltas(candidate_report, baseline_report)`; apply the
     non-regression margin rule.
  7. Return an `AutoTuneResult` whose `.fit` is the candidate on `"pass"` and
     the baseline on `"fail"` / fallback, plus the explainability sub-dict
     (below).

  Every exception path in step 1–6 maps to the fallback contract above. A
  baseline-fit failure raises `AutoApplyError`; all other failures fall back
  to the baseline fit with a recorded reason.

- **Result dataclass.** `AutoTuneResult` is a frozen dataclass carrying:
  `.fit` (the winning `MosaicFit`), `.applied` (`"candidate"` or
  `"baseline-default"`), `.recommendation` (the `BoundsRecommendation`), and
  `.gate` (the explainability sub-dict below). It must not re-derive any
  field the sub-dict already records.

- **CLI.** The `basic tune` subcommand gains an `--auto-apply` flag (see
  {doc}`cli`). With `--auto-apply`, after the normal tuning + bounds output,
  the command runs the gate and writes the winning corrected volume (or
  reports the fallback). The flag is independent of `--bounds-json` /
  `--bounds-margin` / `--out-json` / `--apply`, mirroring how those existing
  flags compose.

- **Reuse, not duplication.** The implementation calls `tune`,
  `recommend_bounds`, `fit_mosaic`, `compute_quality_report`, and
  `compute_deltas` as-is. It adds **no** solver code, touches no `_alm.py` /
  `backend.py` / BaSiC invariant (K02), and does not call
  `calibrate_tolerances` / `evaluate_quality_gate`.

---

## Observability contract

Every auto-apply decision is recorded for reproducibility, mirroring the
existing explainability pattern used by `params["_strategy"]` (D-19, in
`fit.py`) and `params["_working_size_selector"]` / `TuneResult.working_size_selector`
(M006, in `tuning.py` / `fit.py`). The gate metadata is a nested sub-dict on
`AutoTuneResult.gate`:

```python
AutoTuneResult.gate = {
    "gate_verdict": "pass",            # "pass" | "fail" | "fallback"
    "applied": "candidate",            # "candidate" | "baseline-default"
    "no_regression_margin": 0.0,       # the margin the gate used
    "failing_metrics": [],             # ⊆ ["seam_l1", "seam_curvature"]; [] on pass
    "fallback_reason": None,           # None, or one of the reasons in the failure table
    "deltas": {                        # compute_deltas()["aggregate"], per first-class metric
        "seam_l1": {"abs_delta": -0.012, "rel_delta": -0.021},
        "seam_curvature": {"abs_delta": -0.0004, "rel_delta": -0.012},
    },
    "baseline": {                      # the default-bounds floor
        "bounds": "default",           # working_size=128, l_s/l_d auto-tuned, estimate_darkfield=True
        "aggregates": {"seam_l1": 0.5717, "seam_curvature": 0.0344},
    },
    "candidate": {                     # the narrowed-bounds contender
        "bounds": "tune-best-params",
        "best_params": result.best_params,
        "aggregates": {"seam_l1": 0.5597, "seam_curvature": 0.0340},
    },
    "recommendation": {                # the BoundsRecommendation, for traceability
        "search_space": rec.search_space,
        "n_near_optimal": rec.n_near_optimal,
        "margin": rec.margin,
        "best_value": rec.best_value,
    },
}
```

`gate_verdict` is one of:

- `"pass"` — candidate held both first-class metrics ≤ baseline within the
  margin; `applied == "candidate"`.
- `"fail"` — candidate regressed on ≥ 1 metric; `applied == "baseline-default"`,
  `failing_metrics` populated, `fallback_reason == "regression-detected"`.
- `"fallback"` — an upstream step raised before a clean gate result
  (`tune-failed`, `degenerate-trial-history`, `invalid-margin`,
  `candidate-fit-failed`, `quality-report-failed`); `applied ==
  "baseline-default"`.

This lets a future agent (or operator) inspect **why auto-apply did or did
not apply the narrowed bounds without re-deriving it**, satisfying the slice's
observability requirement. The metadata is purely additive: it never affects
the numerics of either fit.

---

## Negative tests

Two layers of negative protection apply.

### Structural guard on this design (T03, this slice)

`tests/test_auto_apply_safety_gate_doc.py` is a plain pytest module that
asserts the required headings of *this* doc exist and cross-checks the symbols
it cites against the real code, mirroring
`tests/test_adaptive_working_size_doc.py`. Its drift-checks assert that
`FIRST_CLASS_METRICS` / `CALIBRATION_POLICY` exist in
`linum_basic.benchmark.quality`, that `recommend_bounds` / `BoundsRecommendation`
exist in `linum_basic.tuning`, and that the doc's stated `FIRST_CLASS_METRICS`
tuple matches the code tuple — so a future edit cannot silently narrow the
gate's scope or cite a renamed primitive.

### Implementation negative tests (S02 contract)

S02's `tests/test_auto_apply_*` must cover these negative scenarios (each
asserts the contract, not just that the call returns):

| Negative scenario | Expected outcome |
|---|---|
| Candidate regresses `seam_l1` beyond the margin | `gate_verdict="fail"`, `applied="baseline-default"`, `failing_metrics=["seam_l1"]`, baseline fit returned |
| Candidate regresses `seam_curvature` (but not `seam_l1`) | same, `failing_metrics=["seam_curvature"]` (K01 needs both) |
| `tune()` produces zero `COMPLETE` trials (all pruned) | `gate_verdict="fallback"`, `fallback_reason="degenerate-trial-history"` |
| `margin` out of `[0, 1]` → `recommend_bounds` raises | `gate_verdict="fallback"`, `fallback_reason="invalid-margin"` |
| Candidate full fit raises (e.g. injected OOM) | `gate_verdict="fallback"`, `fallback_reason="candidate-fit-failed"`, baseline fit returned |
| `compute_deltas` z-index mismatch → raises | `gate_verdict="fallback"`, `fallback_reason="quality-report-failed"` |
| Candidate within float noise of baseline (equal quality) | `gate_verdict="pass"` (margin `0.0` tolerates `FLOAT_NOISE_EPS`) |
| Baseline fit itself raises | `AutoApplyError` raised (no silent undefined correction) |

These are conceptually meaningful cases on the safety boundary, not
superficial coverage — each corresponds to a row of the failure-modes table or
the non-regression rule, and validates that the fallback never ships a worse
correction.

---

## Validation plan (S03)

S03 (operator docs + end-to-end validation) will demonstrate the contract on a
real subject:

1. **End-to-end run.** Run `basic tune --auto-apply` (once S02 ships it) on
   the production-shaped subject `sub-22`, recording `gate_verdict`,
   `applied`, `deltas`, and `fallback_reason` in a dated *Validation Log*
   entry per MEM059.
2. **Fallback exercised.** At least one validation case must hit the
   `regression-detected` (or `degenerate-trial-history`) fallback on purpose,
   proving the gate can refuse — not just that it passes on an easy dataset.
3. **Docs.** Document the workflow, the safety gate, and the fallback
   behaviour in `docs/tuning.md` and `docs/runbooks.md`; advance R057.

Because S01 ships only this document, no runtime validation runs in this
slice. The observability contract above is the spec S02 must implement and S03
must exercise.

---

## Non-goals

- **No change to `recommend_bounds`.** It stays advisory (D015); auto-apply is
  a new, separate consumer of its output.
- **No new release-gate policy.** `calibrate_tolerances` /
  `evaluate_quality_gate` / `CALIBRATION_POLICY` are untouched; the auto-apply
  gate is a distinct, fixed-margin mechanism.
- **No ALM / backend / BaSiC-invariant changes** (K02). The gate composes
  existing `fit_mosaic` and quality primitives; it touches no numerics.
- **No variance estimation / repeated fits.** Auto-apply costs exactly two
  full-volume fits; it never runs an `N`-repeat `mean+3std` calibration.
- **No promotion of `working_size > 128`.** The gate *protects against* a
  tuned `ws > 128` regressing quality (per D020 / MEM081) by falling back; it
  does not itself promote any size.
- **No operator-removal beyond this workflow.** `--auto-apply` is opt-in;
  plain `tune` / `recommend_bounds` / `fit_mosaic` behaviour is unchanged.
