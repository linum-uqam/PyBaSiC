# S03 auto-candidate: working_size="auto" vs ws=128 baseline (sub-22)

First real-subject K01 (seam_l1 / seam_curvature) comparison of the S02
adaptive working_size resolver (requested as `"auto"`) against the fresh
ws=128 baseline from T02, on the A6000. Baseline anchor:
`baseline-20260714T190137-4e924e0-sub-22` (release_gate=True). Candidate
bundle: `candidate-20260714T192818-4e924e0-sub-22`.

## K01 verdict

| metric | abs_delta | rel_delta |
| --- | --- | --- |
| seam_l1 | 0.033002 | 0.057723 |
| seam_curvature | 0.004655 | 0.135273 |
| speed | 0.1040x | slower |
| overall | reject | fail |
| worst_z | 54 |  |

Deltas recompute exactly from raw `metrics_aggregates`
(seam_l1 0.604736 - 0.571734 = 0.033002; seam_curvature 0.039067 - 0.034412
= 0.004655), confirming internal consistency of the harness bundle.

## Resolver outcome (recorded faithfully, not assumed)

The slice plan anticipated rule_path `"baseline-default"` (auto -> 128) on
sub-22, unless sub-22's real preview DCT signal exceeded
`WORKING_SIZE_QUALITY_RAISE_THRESHOLD` (0.15). **The raise branch fired.**

- requested: `"auto"`
- resolved_working_size: **192** (grid maximum; observed in
  `metadata._strategy.working_size` = 192)
- rule_path: **`quality-floor-raise`** — the only branch that can return a
  value > `SAFE_DEFAULT` (128). The resolver selected `max(enlarge_feasible)`
  = 192, which requires (a) `preview_quality` > 0.15 and (b) the A6000 memory
  budget making 160 feasible (it does: budget ~48 GB >> ws=192 per-solve
  estimate).
- gate_status: `opt-in (K01 not yet passed)` — the raise branch executed
  despite R058 not yet being promoted, because the run explicitly requested
  the `"auto"` sentinel.

Note: the `_working_size_selector` explainability block (`rule_path`,
`signals`, `peak_memory_estimate_bytes`) is **not** propagated into the
harness artifact JSON — the resolved integer reaches the artifact only via
`metadata._strategy.working_size`. `rule_path` is therefore recorded here by
deterministic inference from the resolver's branch logic rather than read
from the bundle. The per-candidate `peak_memory_estimate_bytes` dict
(`estimate_strategy_vram_bytes`, n_z_chunk=1, n_tiles=496) is:

| working_size | peak_memory_estimate_bytes | GB |
| --- | --- | --- |
| 64 | 32505856 | 0.033 |
| 96 | 73138176 | 0.073 |
| 128 | 130023424 | 0.130 |
| 160 | 203161600 | 0.203 |
| 192 | 292552704 | 0.293 |

Signals at run time: n_z=55, n_tiles=496, field_mode="per-z",
tile_shape=(75, 75), memory_budget_bytes ~ 48e9 (A6000 free VRAM),
preview_quality > 0.15 (inferred: only the raise branch returns 192).

## Per-z quality failures (6)

| z | metric | abs_delta | rel_delta |
| --- | --- | --- | --- |
| 0 | seam_l1 | 0.259516 | 0.799588 |
| 13 | seam_curvature | 0.028002 | 0.658825 |
| 27 | seam_curvature | 0.005568 | 0.123085 |
| 40 | seam_curvature | 0.021252 | 0.925059 |
| 54 | seam_l1 | 0.271366 | 0.839007 |
| 54 | seam_curvature | 0.010057 | 0.409944 |

The sign of the seam deltas is inconsistent across z-planes (z=13/27/40 show
*lower* seam_l1 for ws=192; z=0/54 show *much higher*), indicating the
enlarged grid is re-regularising real tissue structure differently per plane
rather than uniformly improving quality. This is exactly the failure mode
K01 is designed to catch: a candidate that looks plausible on one plane
destroys reconstruction quality on others.

## Convergence

All five sampled z-planes hit `max_reweighting_iterations` = 15
(`reweight_iterations_per_z`: {0:15, 13:15, 27:15, 40:15, 54:15}), i.e. the
ALM solver did not converge within budget at ws=192 — consistent with the
larger grid giving the reweighting more degrees of freedom to track.

## Speed

9.62x slower (140.5 s vs 14.6 s end-to-end; per_z 28.1 s vs 2.9 s). Even
ignoring the quality failure, this speed regression alone disqualifies the
raise branch for default promotion at production resolution — consistent with
the existing K03/MEM002 finding that ws=128 is the production sweet spot.

## Implication for R058

This candidate is the first real-subject exercise of the S02 quality-floor
raise branch. It **rejects**: the raise branch regresses both seam metrics
against the ws=128 baseline and is 9.62x slower. Combined with the T04
explicit-ws=160 evidence and the T05 synthesis, this is the evidence the R058
go/no-go will be decided on. The raise branch should not be promoted to the
default behaviour on the strength of this run; the baseline-default path
(auto -> 128) remains the only promotion candidate, and that path is exercised
by T04's explicit-ws=160 evidence as the raise-branch boundary test.
