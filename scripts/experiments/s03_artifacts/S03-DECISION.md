# S03 Decision: Go / No-Go on Promoting Adaptive `working_size` to a Default (R058)

> **Milestone M006 / Slice S03 — terminal decision artifact.**
> This document closes M006 with a defensible go/no-go verdict on R058
> ("adaptive `working_size` as default"). It synthesises the three K01-gated
> real-subject A/B runs produced in this slice — the fresh `ws=128` baseline
> (T02), the `working_size="auto"` sentinel candidate (T03), and the explicit
> `ws=160` quality-floor-raise candidate (T04) — all on the production-shaped
> subject `sub-22` on the A6000. A future agent can audit the R058 decision
> from this file plus the JSON bundles alongside it without re-running the
> A6000 harness.
>
> **Note on location.** The plan's Files list named
> `scripts/experiments/S03-DECISION.md`, but that path already holds M005's
> terminal R059 decision (the 30% speed target). Overwriting it would destroy
> durable cross-milestone state, so this M006/R058 decision is co-located with
> its evidence bundles under `scripts/experiments/s03_artifacts/` instead.
>
> **Read-only synthesis of evidence.** The only `linum_basic/` source change
> in this task is updating the resolver's `gate_status` constant and a
> docstring to reflect the outcome; no numerics, no algorithm, and no default
> value of `working_size` change. Docs (`docs/adaptive_working_size.md`,
> `docs/parameters.md`) are updated to record the verdict.

---

## Verdict (TL;DR)

### **NO-GO: adaptive `working_size` is NOT promoted toward a default. `"auto"` stays opt-in with documented guidance.**

- **Both quality-floor-raise candidates were REJECTED** by the K01
  real-subject seam gate on `sub-22`:
  - `working_size="auto"` resolved the quality-floor-raise branch to **192**
    and REJECTED: `seam_l1` **+5.77%** (fail), `seam_curvature` **+13.53%**
    (fail), **9.62x slower**.
  - explicit `working_size=160` (the smallest raise target) REJECTED:
    `seam_l1` **+15.29%** (fail), `seam_curvature` **-9.57%** (improve — the
    only favourable signal), **3.73x slower**.
- **The raise branch is the blocker.** On a real per-z-mode subject where
  `ws=128` fits the memory budget, requesting `"auto"` made the resolver's
  quality-floor-raise branch fire (preview DCT signal >
  `WORKING_SIZE_QUALITY_RAISE_THRESHOLD=0.15`), and **both raise targets
  (160, 192) failed K01**. Promoting `"auto"` to a default would therefore
  regress sub-22-like subjects whenever the raise branch fires.
- **The baseline-default path (`auto -> 128`) was never exercised** on this
  subject, because the raise branch fired first. There is consequently
  **no real-subject evidence that the default path is *better* than the
  status quo** (an explicit `128`), so promotion is unsupported on two
  independent grounds: the raise branch fails, and the no-op path offers no
  benefit to justify changing the default.

### R058 disposition

**R058 stays `active` (not validated).** `"auto"` remains **opt-in only**;
the production default of `working_size` stays the integer `128` everywhere
(`fit_mosaic`, `tune`, the `basic` CLI, `BaSiC`). The raise branch stays
gated/conservative. The `gate_status` string emitted on every resolution is
updated from `"opt-in (K01 not yet passed)"` to
`"opt-in (raise branch not promoted: K01 failed at 160/192 on real
subjects)"` so the persisted metadata reflects the evaluated outcome rather
than a pending state.

### What would need to change to promote in a future milestone

1. **Make the raise branch quality-safe, or disable it.** Either find a
   dataset where `160`/`192` holds K01 parity (none found on sub-22), or
   recalibrate `WORKING_SIZE_QUALITY_RAISE_THRESHOLD` upward so the raise
   branch does not fire on real per-z data, then re-run the K01 gate.
2. **Close the observability gap.** The `_working_size_selector`
   explainability block (`rule_path`, `resolved_working_size`, `signals`,
   `peak_memory_estimate_bytes`) is **not propagated into the harness
   artifact JSON** — T03/T04 reconstructed `rule_path` by deterministic
   inference. Future promotion evidence must record the actual preview signal
   value and resolved size from the persisted metadata, not inferred values.
3. **Multi-subject evidence.** A single subject (sub-22) is insufficient to
   promote a default; at minimum a second per-z-mode subject is needed.

---

## 1. The decision question and what "go" would require

M006/S03 exists to answer one question (per its roadmap vision):

> *Is adaptive `working_size` ready to become the default (R058), now that
> S02 shipped the resolver and S03 has produced the first real-subject K01
> evidence?*

"Go" would mean: on the production-shaped real subject, requesting
`working_size="auto"` holds `seam_l1` / `seam_curvature` parity with the
`ws=128` baseline (K01, `linum_basic.benchmark.quality` mean+3std gate)
**and** the explicit raise-target size (`160`) does the same — confirming
both branches the resolver can take are quality-safe. "No-go" means keeping
`"auto"` opt-in and documenting why, per the design's quality-safety clause
(`docs/adaptive_working_size.md`).

The design itself set this bar: the raise branch *"cannot promote a `> 128`
size to the default until S03 demonstrates, on real subject data, that the
larger size holds `seam_l1` / `seam_curvature` parity"* (Quality-safety
clause §3). S03's job was to produce that demonstration. It produced the
opposite.

---

## 2. Baseline (T02)

Fresh, release-gated `ws=128` baseline on the A6000, matching the M005/S02
baseline pattern exactly.

| baseline field | value |
|---|---|
| baseline_id | `baseline-20260714T190137-4e924e0-sub-22` |
| subject | `sub-22` (real microscopy mosaic, OME-Zarr) |
| commit | `4e924e0` |
| hardware | A6000 `sn4622125853`, single GPU `cuda:0`, PyTorch 2.12.1+cu130, CUDA 13.0, Python 3.14.3 |
| working_size | `128` (strategy `sequential`, release_gate=True) |
| end_to_end_ms | **14606.648** |
| seam_l1 / seam_curvature | **0.571734 / 0.034412** |
| source | `baseline/baseline-bundle.json` |

---

## 3. Candidate A — `working_size="auto"` (T03): REJECT ✘ (quality + speed)

The `"auto"` sentinel flowed through `fit_mosaic`'s mosaic-aware resolver,
which fired the **quality-floor-raise branch** (preview DCT signal exceeded
the `0.15` threshold) and resolved `auto -> 192`. `rule_path =
"quality-floor-raise"` (reconstructed by deterministic inference from the
resolved size and resolver branch logic — see §observability gap).

| metric | result | gate |
|---|---|---|
| resolved working_size | **192** (auto -> raise branch fired) | — |
| seam_l1 | **+5.77%** (0.571734 -> 0.604736) | ✘ **fail** |
| seam_curvature | **+13.53%** (0.034412 -> 0.039067) | ✘ **fail** |
| per-z failures | 6 (z=0 seam_l1 +79.96%; z=13/40/54 curvature; z=54 seam_l1 +83.90%) | ✘ **fail** |
| speed | **0.1040x — 9.62x SLOWER** (14606.6 -> 140485.3 ms) | ✘ **fail** |
| overall | **reject** | |

**Evidence ID:** `candidate-20260714T192818-4e924e0-sub-22`
(`auto_candidate/candidate-artifact.json`).

This is the strongest possible signal against promoting `"auto"`: the exact
mechanism the default would enable (the raise branch) fires on a real
production-shaped subject and degrades **both** seam metrics while making
the run nearly an order of magnitude slower. Every z-plane ran the full
15-iteration reweighting cap; the larger size did not aid convergence — it
just cost more compute for worse seams.

---

## 4. Candidate B — explicit `working_size=160` (T04): REJECT ✘ (quality + speed)

Independently measures the raise branch's smallest target grid size
(`160`, the next step above `128`) against the same baseline, regardless of
what the resolver chose. `rule_path = "explicit-integer (resolver bypassed)"`.

| metric | result | gate |
|---|---|---|
| working_size | **160** (explicit) | — |
| seam_l1 | **+15.29%** (0.571734 -> 0.659125) | ✘ **fail** |
| seam_curvature | **-9.57%** (0.034412 -> 0.031117) | ✓ **improve** (only favourable signal) |
| per-z failures | 5 (z=0 seam_l1 +90.24%; z=27/40 curvature; z=54 seam_l1 +84.31%) | ✘ **fail** |
| speed | **0.2681x — 3.73x SLOWER** (14606.6 -> 54478.1 ms) | ✘ **fail** |
| overall | **reject** | |

**Evidence ID:** `candidate-20260714T221456-4e924e0-sub-22`
(`ws160_candidate/candidate-artifact.json`).

The `seam_curvature` improvement is real but does not rescue the candidate:
`seam_l1` fails by 15% at the aggregate and by 84-90% on the worst z-planes,
and the run is 3.7x slower. Trading one K01 metric for another (lower
curvature, higher seam_l1) is still a quality failure — K01 requires parity
on **both** metrics.

---

## 5. Synthesis — why this is a defensible NO-GO

| Question | Answer |
|---|---|
| Did `"auto"` hold K01 parity with `ws=128`? | **No.** It raised to 192 and failed both seam metrics (+5.77% / +13.53%) and was 9.62x slower. |
| Did the smallest raise target (160) hold K01 parity? | **No.** `seam_l1` failed +15.29% (curvature improved -9.57%, but K01 needs both). 3.73x slower. |
| Was the baseline-default path (`auto -> 128`) demonstrated? | **No.** The raise branch fired on sub-22, so the no-op path was never exercised on a real subject. |
| Is the raise branch attackable to quality-safety? | **Not with current evidence.** Both raise targets (160, 192) fail K01 on this subject; no real-subject dataset where enlarging holds parity has been found. |
| Does the speed dimension change the verdict? | **No — it reinforces it.** Both candidates are much slower; even if a future raise candidate passed quality, it would need to clear the speed dimension too. |

**The raise branch is the decisive blocker.** Promoting `"auto"` to a
default turns on a branch that, on real per-z-mode data, degrades quality
and speed. The design's own quality-safety clause gates raise-branch
promotion on exactly the real-subject parity S03 was meant to demonstrate;
S03 demonstrated its absence.

---

## 6. Calibration outcome for `WORKING_SIZE_QUALITY_RAISE_THRESHOLD`

The design doc lists threshold calibration as an S03 deliverable. S03's
finding:

- `WORKING_SIZE_QUALITY_RAISE_THRESHOLD = 0.15` is **low enough to fire on
  real per-z-mode data** — it fired on sub-22, resolving `auto -> 192`.
- The raise **targets** (`160`, `192`) then **failed K01**. The problem is
  therefore not solely the threshold firing; the sizes it raises **to** are
  not quality-safe on this subject.
- **Disposition:** the threshold value is **kept at `0.15`** (changing it
  would alter validated resolver behaviour, and no evidence-based
  replacement number is available — the exact sub-22 preview signal value
  was not persisted into the artifact JSON). The constant's docstring and
  this decision now record that, when the raise branch fires on real per-z
  data, the resulting sizes fail K01, so the branch must remain opt-in and
  quality-risky. A principled upward recalibration is deferred to a future
  milestone that closes the observability gap (§"what would need to change")
  and records exact preview signals.

---

## 7. What this milestone proves and what it does not

**Proves (proof level: operational):** M006 operationally answers its
framing question with quality-gated real-subject evidence. The no-go
decision is defensible and reproducible from the version-controlled
artifacts under `scripts/experiments/s03_artifacts/`. All deltas in §3-§4
were recomputed from raw `metrics_aggregates` and match the stored
`aggregate_deltas` to 1e-9.

**Does not prove / does not do:**
- It does **not** change the production default of `working_size` (stays
  `128`), the ALM solver, or any BaSiC invariant (K02).
- It does **not** rule out adaptive `working_size` forever — it rules out
  promotion *on the current single-subject evidence*. The shrink branch
  (always safe per D002) and the baseline-default path remain valid opt-in
  behaviour; only the raise branch is evidence-against-promotion.
- It does **not** exercise the memory-ceiling shrink branch on a real
  subject (sub-22 fits `128` in budget). The shrink branch's safety rests
  on D002 (quality parity at `<= 128`), not on this slice's runs.

---

## Verification

- **Internal consistency:** every numeric claim in §3-§4 was recomputed from
  raw `metrics_aggregates` in the three JSON bundles and matched the stored
  `aggregate_deltas` (`abs_delta` / `rel_delta`) and `speed_verdict.ratio`
  to within 1e-9 (recompute script recorded as UAT/`gsd_exec` evidence).
- **Code/doc alignment:** the resolver's `gate_status` constant
  (`linum_basic/_working_size.py`) and the design/parameters docs are
  updated to the evaluated outcome; the full resolver test suite
  (`tests/test_working_size_resolver.py`, `tests/test_fit_working_size_auto.py`,
  `tests/test_tuning.py`, `tests/test_cli.py`,
  `tests/test_adaptive_working_size_doc.py`) passes.
- T05 persists this verdict into R058 (`gsd_requirement_update`) and
  `DECISIONS.md` (`gsd_decision_save`) for durable cross-milestone state.

## Failure Modes

This task has **no external runtime dependencies** — it is a read-only
synthesis of three version-controlled JSON evidence bundles plus targeted
doc/code edits. The only failure mode is a claim drifting from its cited
evidence; this is guarded by the §Verification internal-consistency
recompute (every delta cross-checked against the stored artifact value) and
by the existing doc-structure regression test
(`tests/test_adaptive_working_size_doc.py`), which mechanically locks the
required design-doc sections and the candidate-grid drift check.

## Load Profile

Not applicable — this is a one-shot decision synthesis with no runtime load
dimension. There is no request volume to saturate; the artifact is a
fixed-size markdown document plus constant/docstring updates, consumed by
R058's requirement update and by any future milestone that re-evaluates
promotion. No compute, GPU, or concurrency is exercised by the synthesis
itself.

## Negative Tests

The decision's falsification surface is its evidence, locked by the existing
resolver test suite. Specifically, the following negative guarantees survive
this change and would catch a regression of the no-go decision:
- `tests/test_working_size_resolver.py::TestFailSafe` — every ambiguous /
  missing-signal path still resolves to `128` (the opt-in safety net is
  intact).
- `tests/test_working_size_resolver.py::TestCostBound` — the resolver still
  never invokes a solver (no probe fit).
- `tests/test_fit_working_size_auto.py::TestOptInIsolation` — callers that
  omit `"auto"` are byte-for-byte unaffected (no silent default change).
- A new assertion (`tests/test_working_size_resolver.py`) locks
  `GATE_STATUS_OPT_IN` to the evaluated, not-promoted state, so the stale
  `"not yet passed"` phrasing cannot silently return.
