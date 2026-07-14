# S01/T01 — Existing Lever Taxonomy & Prior Attempt Outcomes (Code Evidence)

> **Scope:** T01 only. This is a read-only inventory of the lever taxonomy
> already encoded in `linum_basic/benchmark/profile.py` and where each
> lever's mechanism actually lives in the **current** codebase. It
> establishes ground truth on what was already attempted, promoted, or
> rejected *before* T02 surveys for new levers. No production code is
> modified.
>
> **Sister docs:** `s01_new_levers.md` (T02 — missing levers + ceilings),
> `s01_synthesis_notes.md` (T03 — ranked shortlist + bottleneck
> reassessment → feeds `S01-RESEARCH.md`).

---

## 1. Canonical taxonomy source

The single source of truth for the lever taxonomy is
`linum_basic/benchmark/profile.py`:

| Symbol | Location | Role |
|--------|----------|------|
| `_LEVER_DEFINITIONS` | `profile.py:63` | Tuple of `(lever_id, target_file, risk, gate_notes)` — the **7 named levers** |
| `BottleneckClass` (Literal) | `profile.py:13` | 5 classes: `memory-bandwidth`, `compute`, `compile-shape`, `synchronization`, `chunking` |
| `_PRIMARY_LEVER_ORDER` | `profile.py:138` | Per-bottleneck-class priority ranking of the 7 levers |
| `_classify_op` | `profile.py:338` | Op-name → bottleneck-class heuristic (matmul→compute, memcpy/cat→memory-bandwidth, synchronize/item→synchronization, inductor/compile→compile-shape) |
| `RankedLever` | `profile.py:205` | Dataclass: `lever_id, target_file, priority, expected_risk, gate_notes` |
| `LeverAttemptRow` | `profile.py:440` | Per-attempt outcome: `overall` (promote/reject), `quality_passed`, `speed_passed`, `speed_ratio`, `quality_failures` |
| `build_phase5_backlog` | `profile.py:727` | Emits rejected/blocked/deferred rows → `phase5-backlog.json` |
| `build_phase5_fast_path` | `profile.py:940` | Emits promoted-override stack → `phase5-fast-path.json` |
| `build_forensics_recovery_levers` | `profile.py:1456` | **Non-taxonomy** levers from A6000 forensics (FORE-04) |
| `SPEED_RATIO_THRESHOLD` | `benchmark/sweep.py:11` | `1.30` — the adoption bar (D-18/D007) |

> **Important:** `profile.py` is over the 50 KB whole-file observation
> threshold, so line numbers above come from targeted `gsd_exec` grep
> digests rather than a full read. They were verified against the read
> windows at offsets 1–200, 205–424, 440–529, 727–816, 1456–1515.

---

## 2. The 7 taxonomy levers — status against current code

For each lever: mechanism location, what the code actually does today,
and prior attempt outcome (promoted / attempted-rejected / never-attempted
in production A/B).

### 2.1 `sync-cadence` — medium risk — `_alm.py`

**Mechanism.** The ALM convergence check calls `xp.norm_fro(dY)`, which on
GPU forces a host-device synchronisation (`float()` of a device tensor).
The cadence controls how often this sync fires.

| Path | Location | Current value | Knob |
|------|----------|---------------|------|
| Scalar ALM | `_alm.py:441-442` | `convergence_check_every = 10` (GPU) / `1` (CPU) when caller passes `None` | `convergence_check_every` param on `inexact_alm_l1` (`:290`) |
| Scalar check site | `_alm.py:500` | `if iteration % convergence_check_every == 0:` then `xp.norm_fro(dY)` | — |
| Batched ALM | `_alm.py:789` | `check_every = 10` **hardcoded** | none exposed |
| Batched sync site | `_alm.py:834,838,846` | `xp.to_numpy(xp.sum(converged)).item()` every 10 iters | — |

**Status:** **NEVER ATTEMPTED as a standalone speed lever.** The mechanism
exists (and was clearly designed to be tunable — there is a public
`convergence_check_every` kwarg), but no harness override / env var tunes
it, and no `phase5-fast-path.json` / backlog row records a promotion or
rejection. Tuning it higher (e.g. 20) is *structurally* open. **Note:**
raising the batched `check_every` above 10 would require a code change
(it's a module literal, not env-driven) and risks running past
convergence by up to N extra iterations.

### 2.2 `compile-surfacing` — low risk — `_alm.py`

**Mechanism.** Two sub-parts: (a) make `torch.compile` fallback audible;
(b) TF32 precision policy.

| Sub-part | Location | Current behavior |
|----------|----------|------------------|
| Fallback warning (dedup) | `_alm.py:44` `_warn_compile_fallback_once`; `:267` (scalar), `:666` (batched) | Emits one `warnings.warn` per `cache_key` on compile failure; records `last_compile_fallback` |
| Compile mode env | `_alm.py:53` `_read_alm_compile_mode` | `LINUM_BASIC_ALM_COMPILE_MODE` — `off`/`disabled` → eager; `default` → compiled |
| TF32 precision | `_alm.py:255` (scalar), `:662`-area (batched) | `torch.set_float32_matmul_precision("high")` always set when compiling |

**Status:** **PROMOTED (compile-fallback surfacing).** The warning dedup
shipped via commits `ca28234` (WR-03) and `8b85797` (tests), documented
in K07. TF32 `high` is active whenever compile is on. **However:** D-15
"precision metadata surfacing" beyond the warning string is not evident
in code — the warning carries the reason but not a structured precision
tag. The *observability* lever is essentially done; there is no
remaining *speed* surface here unless TF32 is widened (which K06
forbids globally).

### 2.3 `reweighting-tolerance` — medium risk — `core.py`

**Mechanism.** The *outer* reweighting loop early-exits when the
flat/dark-field stops changing.

| Element | Location | Current value |
|---------|----------|---------------|
| Tolerance | `core.py:208` | `reweighting_tolerance = 1e-3` |
| Max iters | `core.py:209` | `max_reweighting_iterations = 10` |
| Stop condition | `core.py:451-452` | `max(mad_flat, mad_dark) <= reweighting_tolerance or reweighting_iteration >= max_reweighting_iterations` |
| Dark-field guard | `core.py:443-446` | `mad_dark = 1.0` when previous estimate was zero (do-not-falsely-converge invariant) |

**Status:** **NEVER ATTEMPTED as a speed lever.** D-08 is referenced in
the lever's `gate_notes` but no env var or harness override tunes the
tolerance or iteration cap from production defaults. The mechanism is
fully wired (both scalar `BaSiC` and batched `_batched_fit.py` implement
the same `mad` freeze logic), so a candidate A/B could raise tolerance
to e.g. `5e-3` or cap at `max_reweighting_iterations=6` with **zero**
code change (these are public `BaSiC` attributes). Risk: fewer
reweighting passes may shift the flat-field and trip seam quality
gates — needs real-subject A/B (K01).

### 2.4 `dct-kernel-tuning` — medium risk — `backend.py` + `_alm.py`

**Mechanism.** Replace FFT-based DCT (which produces complex tensors
Torchinductor cannot compile to Triton) with a real matmul DCT
`Y = A_p @ X @ A_q.T` inside the compiled step.

| Element | Location | Current behavior |
|---------|----------|------------------|
| Mode env | `backend.py` `read_dct_kernel_mode()` (~710) | `LINUM_BASIC_DCT_KERNEL` — only `"tuned"` enables contiguous layout; else `default` |
| Matrix cache | `backend.py` `_DCT_MATRIX_CACHE`, `_get_dct_matrix` (~671) | Per-`(n, device)` orthonormal DCT-II matrix, LRU-bounded (`_DCT_CACHE_MAXSIZE=64`) |
| Tuned path | `_alm.py:193-205` | `_Ap/_Aq.contiguous()` + `x.contiguous()` for Inductor kernel selection |
| Default path | `_alm.py:206-212` | `A_p @ x @ A_q.T` without explicit contiguity |

**Status:** **PROMOTED — D003.** The only taxonomy lever that passed
**both** the seam quality gate and the 1.30× speed gate on real
subjects, yielding ~2.2% steady-state reduction. Recorded in DECISIONS.md
D003. The 30% aspirational target was explicitly *not* met and deferred
to post-forensics (this milestone, M005 / PERF-F01). This is one third
of the already-promoted stack cited in the S01 goal.

### 2.5 `compile-shape-stability` — medium risk — `_alm.py`

**Mechanism.** Prevent `torch._dynamo` recompile storms by keeping
tensor guards stable across iterations.

| Element | Location | Current behavior |
|---------|----------|------------------|
| mu as 0-dim InferenceMode tensor | `_alm.py:477-480` | `mu` grows by `rho=1.5` each iter but stays a device tensor so the guard keys on dispatch, not float value — prevents `recompile_limit=8` exhaustion after 9 steps |
| `xp.clone()` of fed-back tensors | `_alm.py` loop body; `backend.py` `clone()` docstring | Strips `ADInplaceOrView` dispatch key that `reshape`/`.to()` attach, preventing spurious guard failures |
| `fullgraph=False` | `_alm.py:258` (scalar), `:662` (batched) | Relaxed graph mode |

**Status:** **LARGELY SATISFIED by existing code.** The mu-tensor and
clone() fixes are already present (they encode the lever's intent). What
remains is *measure* whether further shape stability (e.g. padding to a
fixed shape, or `dynamic=True`) buys more — but since `worker-compile-off`
(§3.1) now disables compile in production workers entirely, this lever's
ceiling is mostly moot for the production path. It only matters for the
non-default compiled path.

### 2.6 `inductor-cache-warm-policy` — low risk — `_torch_cache.py`

**Mechanism.** Steady-state timing fidelity + persistent on-disk Inductor
cache.

| Element | Location | Current behavior |
|---------|----------|------------------|
| Warm passes | `_torch_cache.py:86` `warm_policy_passes()` | Returns extra *untimed* warm fits before measured benchmark repeats |
| Persistent cache | `_torch_cache.py:60` `configure_torch_inductor_cache` | Sets `TORCHINDUCTOR_CACHE_DIR` → `~/.cache/linum-basic/inductor`; called in loky worker init |
| FX graph cache | `_torch_cache.py:107` `enable_fx_graph_cache` | Enables `inductor_config.fx_graph_cache` post-import |

**Status:** **NEVER ATTEMPTED — explicitly deferred.** The
`build_phase5_backlog` docstring (`profile.py:733`) names
`inductor-cache-warm-policy` by example as a lever "not attempted in
Phase 3." The mechanism is **benchmark-harness infrastructure, not a
production speed lever** — warm passes improve measurement fidelity, not
fit time. Like §2.5, its relevance collapses once compile is off in
production workers. Low ceiling for a production speed target.

### 2.7 `tile-subsampling` — high risk — `tuning.py`

**Mechanism.** Reduce the tile count used for Optuna seam-metric
evaluation.

| Element | Location | Current behavior |
|---------|----------|------------------|
| Subsample fn | `tuning.py:72` `_subsample_tiles` | Evenly spaced subset via `np.linspace`; `None` or `>= n` returns all |
| Default cap | `tuning.py:338` | `max_tiles = 64` |
| z subsample | `tuning.py:329` | `z_subsample = 4` |

**Status:** **EXISTS but ORTHOGONAL to the speed target.** This is a
*tuning-search* speedup (makes Optuna trials cheaper), not a
*production-fit* speedup. The S01 goal targets ws=128 steady-state fit
time, which is unaffected by how many tiles Optuna samples during
hyperparameter search. D-07 referenced as "last-resort." **Reject for
this milestone's purpose** unless the goal were to reframe to include
`tune()` wall time.

---

## 3. Non-taxonomy promoted levers (forensics, FORE-04)

`build_forensics_recovery_levers` (`profile.py:1456`) records two levers
that are **not** in `_LEVER_DEFINITIONS` but are part of the
"already-promoted stack" the S01 goal cites:

### 3.1 `worker-compile-off` — low risk — `_parallel.py` — **PROMOTED (D005)**

| Element | Evidence |
|---------|----------|
| Decision | DECISIONS.md D005: `LINUM_BASIC_ALM_COMPILE_MODE=off` default for CUDA joblib workers |
| Commit | `511c88c` "Fix GPU fit slowness by disabling compile in CUDA workers." |
| Mechanism | `_alm.py:53` `_read_alm_compile_mode` returns `None` for `off` → eager step (no Inductor CPU compile storm in short-lived workers) |
| Impact | Per-z fit ~229s (numpy) / ~130s+ (compile cold) → ~3-6s/z; no quality regression |
| Tests | `18dcf63` asserts phase5 fast-path carries `compile_mode=off` stack |

This is the dominant promoted lever. It interacts with §2.5 and §2.6:
with compile off in workers, compile-shape-stability and
inductor-cache-warm-policy are largely inert on the production path.

### 3.2 `auto-l-s-config` — low risk — subject config — **PROMOTED**

| Element | Evidence |
|---------|----------|
| Mechanism | `build_forensics_recovery_levers` priority 2: set `fix_illum_smoothness_flatfield=null` → auto `l_s` (vs fixed `l_s=0.05`) |
| Code site | `core.py` auto-tune: `l_s = dct_sum / 800` (`DEFAULT_L_S_DIVISOR`); batched equivalent in `_batched_fit.py:104-107` |

Part of the promoted stack. Speed impact is via fewer/better-conditioned
reweighting iterations rather than per-iteration speed.

---

## 4. Prior attempt-outcome summary table

| Lever | In taxonomy? | Outcome | Evidence | Production speed relevance |
|-------|:---:|---------|----------|----------------------------|
| `worker-compile-off` | no (FORE-04) | **PROMOTED** | D005, `511c88c`, `build_forensics_recovery_levers` | **Dominant** (~3-6s/z from 130s+) |
| `auto-l-s-config` | no (FORE-04) | **PROMOTED** | `build_forensics_recovery_levers` priority 2 | Indirect (iteration count) |
| `dct-kernel-tuning` | yes | **PROMOTED** | D003; `_alm.py:193-205`; `read_dct_kernel_mode` | **~2.2%** steady-state |
| `compile-surfacing` | yes | **PROMOTED (observability)** | `ca28234`, `8b85797`; K07 | None remaining (TF32 forbidden by K06) |
| `compile-shape-stability` | yes | **Largely satisfied in code** | `_alm.py:477-480`, `clone()` | Moot under `worker-compile-off` |
| `inductor-cache-warm-policy` | yes | **Deferred (Phase 5 backlog)** | `build_phase5_backlog` docstring | Harness-only, not production fit |
| `sync-cadence` | yes | **Never attempted** | `_alm.py:441-442,500,789,834` | **Open** — kwarg exists, no override |
| `reweighting-tolerance` | yes | **Never attempted** | `core.py:208,209,451`; D-08 | **Open** — public attrs, zero code change |
| `tile-subsampling` | yes | Exists / orthogonal | `tuning.py:72,329,338`; D-07 | Tuning-search only, not fit time |

---

## 5. Bottleneck-class map (from `_PRIMARY_LEVER_ORDER` + `_classify_op`)

The classifier (`_classify_op`, `profile.py:338`) maps profiler op names:

| Op signal | Bottleneck class |
|-----------|------------------|
| `::mm`, `::bmm`, `matmul`, `addmm` | `compute` |
| `memcpy`, `copy_`, `cat`, `clone`, `contiguous` | `memory-bandwidth` |
| `cudadevicesynchronize`, `synchronize`, `item`, `::norm` | `synchronization` |
| `inductor`, `compile`, `triton_compile` | `compile-shape` |
| `chunk` + (`cat`/`batch`) | `chunking` |
| (default) | `compute` |

The dominant production finding (MEM001/MEM002/K03) is that ws=128 is
**memory-bandwidth-bound**, not compute-bound — which is why batched
CUDA at ws=128 equals or loses to sequential despite high GPU activity
(D009/K03). T03 will restate/confirm this against the current matmul-DCT
path (`A_p @ X @ A_q.T`, four `clone()`/`.contiguous()` per iter) and
the sync cadence sites above.

---

## 6. Verification

- **No production source files modified** (read-only research). All
  evidence above is line-cited to the current tree via `gsd_exec` grep
  digests and targeted `read` windows.
- T01 deliverable existence check is the task's `Verify` command:
  `test -s scripts/experiments/s01_lever_inventory.md` (this file).

## Failure Modes

This task has **no external dependencies** — it is a static read-only
inventory of version-controlled source files (`profile.py`, `_alm.py`,
`backend.py`, `_torch_cache.py`, `_parallel.py`, `core.py`, `tuning.py`,
`_batched_fit.py`, `fit.py`, `strategies.py`) plus the tracked
`.gsd/DECISIONS.md`. No network, filesystem mutation, subprocess, or API
is involved. The only failure mode is stale line numbers after future
edits, which the header note in §1 explicitly flags.

## Load Profile

Not applicable — this is a one-shot static analysis with no runtime load
dimension. There is no request volume to saturate; the artifact is a
fixed-size markdown file read by S02/S03 agents.

## Negative Tests

Not applicable — T01 produces a research inventory, not executable code.
There is no malformed-input surface. The factual claims are individually
line-cited to source so a reviewer can falsify any row; correctness is
enforced by citation, not by test assertions.
