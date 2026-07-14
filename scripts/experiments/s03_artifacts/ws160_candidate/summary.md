# S03 ws160-candidate: explicit working_size=160 vs ws=128 baseline (sub-22)

First real-subject K01 (seam_l1 / seam_curvature) comparison of the S02
quality-floor-raise branch's TARGET grid size (160 — the next grid step above
128 in `WORKING_SIZE_GRID`) against the fresh ws=128 baseline from T02, on
the A6000. Baseline anchor:
`baseline-20260714T190137-4e924e0-sub-22` (release_gate=True). Candidate
bundle: `candidate-20260714T221456-4e924e0-sub-22`.

Unlike T03's `"auto"` sentinel (which invoked the resolver and resolved to
192), this run requested an **explicit integer** working_size=160, so the
adaptive resolver is **not** invoked — `rule_path` is `explicit-integer
(resolver bypassed)`. This is the independent, direct measurement of the
raise branch's TARGET size regardless of what T03's resolver decided, per the
S02 validation plan.

## K01 verdict

| metric | abs_delta | rel_delta |
| --- | --- | --- |
| seam_l1 | +0.087391 | +0.152853 |
| seam_curvature | -0.003295 | -0.095740 |
| speed | 0.2681x | slower (3.73x) |
| overall | reject | fail (seam_l1) |
| worst_z | 0 | |

Deltas recompute exactly from raw `metrics_aggregates`
(seam_l1 0.659125 - 0.571734 = +0.087391; seam_curvature 0.031117 - 0.034412
= -0.003295), confirming internal consistency of the harness bundle. All 10
per-z delta values (5 z-planes x 2 metrics) also match the stored
`deltas.per_z` bit-for-bit.

## Working_size / strategy (recorded faithfully)

- requested: **160** (explicit integer; not `"auto"`)
- resolved_working_size: **160** (observed in `metadata._strategy.working_size`
  = 160; identity passthrough — the resolver is bypassed for explicit ints)
- rule_path: **`explicit-integer (resolver bypassed)`** — because the request
  was an int, `_resolve_working_size_auto` is never called. This run isolates
  the raise branch's TARGET grid size directly, independent of the resolver's
  preview-signal gating.
- execution_path: `sequential_scalar`, strategy `sequential` (strategy_lock);
  backend torch, device cuda:0
- peak VRAM observed: 619857920 bytes (0.620 GB); strategy memory_estimate
  1015808000 bytes (1.016 GB) — both comfortably within the A6000's ~48 GB
  budget, so memory was never the constraint; the constraint is quality.
- git_commit: 4e924e0 (valid for explicit-int runs per the T02 server-commit
  decision; T01's parser change has zero numerical impact on int requests)

Note: the `_working_size_selector` explainability block (`rule_path`,
`signals`, `peak_memory_estimate_bytes`) is **not** propagated into the
harness artifact JSON — same observability gap recorded in T03. Since the
resolver is bypassed here, the only field that would apply (`rule_path`) is
recorded above by deterministic inference (explicit int -> resolver bypassed).

## Per-z quality failures (5)

| z | metric | abs_delta | rel_delta |
| --- | --- | --- | --- |
| 0 | seam_l1 | +0.292883 | +0.902394 |
| 27 | seam_l1 | +0.057255 | +0.088588 |
| 27 | seam_curvature | +0.000351 | +0.007765 |
| 40 | seam_curvature | +0.001772 | +0.077137 |
| 54 | seam_l1 | +0.272698 | +0.843124 |

The seam_l1 per-z sign is inconsistent across planes: z=0 (+90%) and z=54
(+84%) show large regressions, while z=13 (-11%) and z=40 (-13%) actually
*improve*. This is the same z-plane-dependent re-regularisation pattern seen
at ws=192 in T03 — a candidate that helps some planes and destroys others,
which is exactly the failure mode K01 is designed to catch (aggregate
averaging masks the per-plane damage).

Notably, seam_curvature *improves* in aggregate (-9.57% rel) — the only
metric/metric-direction favorable to the raise branch across both candidates.
But seam_l1's aggregate regression (+15.29%) is decisive: the K01 quality
gate requires both metrics to pass, and seam_l1 fails.

## Convergence

All five sampled z-planes hit `max_reweighting_iterations` = 15
(`reweight_iterations_per_z`: {0:15, 13:15, 27:15, 40:15, 54:15}), i.e. the
ALM solver did not converge within budget at ws=160 — the same non-convergence
observed at ws=192 in T03. The larger grid gives the reweighting more degrees
of freedom to track, preventing the stopping criterion from firing at the
15-iteration budget.

## Speed

3.73x slower (54.5 s vs 14.6 s end-to-end; per_z 10.9 s vs 2.9 s). Slower than
ws=192's 9.62x only because 160 < 192; both regress speed materially. Even
ignoring the quality failure, this speed regression alone disqualifies the
raise branch for default promotion at production resolution — consistent with
the existing K03/MEM002 finding that ws=128 is the production sweet spot.

## Cross-candidate comparison (ws=160 vs ws=192 auto)

| candidate | seam_l1 rel | seam_curvature rel | speed | overall |
| --- | --- | --- | --- | --- |
| ws=160 (this run) | +15.29% | -9.57% | 3.73x slower | reject |
| ws=192 (T03 auto) | +5.77% | +13.53% | 9.62x slower | reject |

Both grid steps above 128 reject K01. ws=160 is faster and has a favorable
curvature signal, but a worse seam_l1 regression; ws=192 is slower and fails
both metrics. Neither outcome supports promoting a working_size > 128.

## Implication for R058

This candidate is the independent, direct measurement of the S02
quality-floor-raise branch's TARGET grid size (160). It **rejects**: even the
smallest grid step above 128 regresses seam_l1 against the ws=128 baseline
and is 3.73x slower, with all z-planes non-converged. Combined with T03's
ws=192 auto-sentinel reject, **both grid steps above 128 fail K01 on sub-22**.

The raise branch (resolver-selected enlargement, or explicit enlargement to
the raise TARGET) therefore should not be promoted to default behaviour on
this evidence. The baseline-default path (auto -> 128) remains the only
promotion candidate for R058, and T05 will synthesise both rejects into the
go/no-go with the explicit recommendation that "auto" default-promotion must
gate on the resolver resolving to 128 (baseline-default), not on the raise
branch firing.
