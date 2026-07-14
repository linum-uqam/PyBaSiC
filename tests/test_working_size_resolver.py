"""Cost-bound and fail-safe tests for the adaptive working_size resolver.

Covers the design contract in ``docs/adaptive_working_size.md`` (M006/S02):

* **Cost bound** — the resolver never instantiates ``BaSiC``, never calls the
  ALM solver, and never runs a probe fit (verified by monkeypatching solver
  entry points to raise).
* **Fail-safe to 128** — every ambiguous / missing-signal path resolves to
  ``SAFE_DEFAULT`` (128) with a populated ``fallback_reason``.
* **Memory-ceiling shrink** (always safe) — returns the largest feasible size
  ``<= 128`` when 128 does not fit.
* **Quality-floor raise** (opt-in, gated) — enlarges only when 128 fits, the
  preview signal exceeds the threshold, and memory permits 160.
* **Grid drift** — ``WORKING_SIZE_GRID`` mirrors
  ``linum_basic.tuning._DEFAULT_SEARCH_SPACE["working_size"]``.
* **Observability** — ``metadata()`` carries every required key.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from linum_basic import _working_size as wsm
from linum_basic._working_size import (
    GATE_STATUS_OPT_IN,
    PREVIEW_RESOLUTION,
    SAFE_DEFAULT,
    WORKING_SIZE_GRID,
    WORKING_SIZE_QUALITY_RAISE_THRESHOLD,
    WorkingSizeContext,
    compute_preview_quality,
    peak_memory_estimate,
    resolve_working_size,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_METADATA_KEYS = frozenset(
    {
        "resolved_working_size",
        "requested",
        "candidate_grid",
        "rule_path",
        "fallback_reason",
        "gate_status",
        "signals",
        "peak_memory_estimate_bytes",
    }
)

REQUIRED_SIGNAL_KEYS = frozenset({"n_z", "n_tiles", "tile_shape", "field_mode", "memory_budget_bytes", "preview_quality"})

VALID_RULE_PATHS = frozenset({"baseline-default", "memory-ceiling-shrink", "quality-floor-raise", "fallback-safe-default"})


def _ctx(
    *,
    n_z: int = 41,
    n_tiles: int = 72,
    field_mode: str = "per-z",
    tile_shape: tuple[int, int] | None = (2048, 2048),
    memory_budget_bytes: int | None = None,
    preview_quality: float | None = None,
    requested: str | int = "auto",
) -> WorkingSizeContext:
    """Build a context with explicit signals (no env/torch probing)."""
    return WorkingSizeContext(
        n_z=n_z,
        n_tiles=n_tiles,
        field_mode=field_mode,
        tile_shape=tile_shape,
        memory_budget_bytes=memory_budget_bytes,
        preview_quality=preview_quality,
        requested=requested,
    )


# ---------------------------------------------------------------------------
# Tracer: proves the resolver never touches the solver / fit path.
# ---------------------------------------------------------------------------


class _BoomError(RuntimeError):
    """Raised when a forbidden function is invoked during resolution."""


@pytest.fixture()
def solver_boom(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch every solver / fit entry point to raise.

    If :func:`resolve_working_size` ever calls into ``BaSiC``, the ALM solver,
    or a fit path, the test invoking it under this fixture fails loudly.
    """
    import linum_basic.core as core

    monkeypatch.setattr(core, "BaSiC", _make_boom("BaSiC.__init__"))
    monkeypatch.setattr(core, "inexact_alm_l1", _make_boom("inexact_alm_l1"))

    import linum_basic._alm as alm

    monkeypatch.setattr(alm, "inexact_alm_l1", _make_boom("inexact_alm_l1"))
    monkeypatch.setattr(alm, "inexact_alm_l1_batched", _make_boom("inexact_alm_l1_batched"))
    monkeypatch.setattr(alm, "shrink", _make_boom("shrink"))

    import linum_basic.fit as fit

    monkeypatch.setattr(fit, "fit_mosaic", _make_boom("fit_mosaic"))
    monkeypatch.setattr(fit, "_fit_one_z", _make_boom("_fit_one_z"))
    monkeypatch.setattr(fit, "make_model", _make_boom("make_model"))


def _make_boom(name: str) -> Any:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise _BoomError(f"resolver must not call {name}")

    _boom.__name__ = name
    return _boom


# ---------------------------------------------------------------------------
# Cost bound
# ---------------------------------------------------------------------------


class TestCostBound:
    """The resolver must never run a BaSiC/ALM solve at any candidate size."""

    def test_baseline_resolution_does_not_touch_solver(self, solver_boom: None) -> None:
        ctx = _ctx(memory_budget_bytes=40 * 1024**3, preview_quality=0.0)
        res = resolve_working_size(ctx)
        assert res.resolved_working_size == SAFE_DEFAULT

    def test_shrink_resolution_does_not_touch_solver(self, solver_boom: None) -> None:
        # Budget only fits 64.
        budget = peak_memory_estimate(64, _ctx(n_tiles=72))
        ctx = _ctx(memory_budget_bytes=budget, preview_quality=None)
        res = resolve_working_size(ctx)
        assert res.resolved_working_size == 64

    def test_quality_raise_does_not_touch_solver(self, solver_boom: None) -> None:
        ctx = _ctx(memory_budget_bytes=128 * 1024**3, preview_quality=0.9)
        res = resolve_working_size(ctx)
        assert res.resolved_working_size in (160, 192)


# ---------------------------------------------------------------------------
# Fail-safe to 128
# ---------------------------------------------------------------------------


class TestFailSafe:
    """Every ambiguous / missing-signal case resolves to 128."""

    @pytest.mark.parametrize(
        "label,ctx",
        [
            (
                "no_memory_budget",
                _ctx(memory_budget_bytes=None, preview_quality=None),
            ),
            (
                "non_int_memory_budget",
                _ctx(memory_budget_bytes="not-a-number", preview_quality=None),
            ),
            (
                "preview_quality_none",
                _ctx(memory_budget_bytes=128 * 1024**3, preview_quality=None),
            ),
            (
                "preview_quality_out_of_range_high",
                _ctx(memory_budget_bytes=128 * 1024**3, preview_quality=1.5),
            ),
            (
                "preview_quality_out_of_range_low",
                _ctx(memory_budget_bytes=128 * 1024**3, preview_quality=-0.2),
            ),
            (
                "preview_quality_nan",
                _ctx(memory_budget_bytes=128 * 1024**3, preview_quality=float("nan")),
            ),
            (
                "tile_shape_none",
                _ctx(tile_shape=None, memory_budget_bytes=128 * 1024**3, preview_quality=None),
            ),
        ],
    )
    def test_resolves_to_safe_default_with_reason(self, label: str, ctx: WorkingSizeContext) -> None:
        res = resolve_working_size(ctx)
        assert res.resolved_working_size == SAFE_DEFAULT, label
        # When the memory budget is the missing signal, fallback_reason must be populated.
        if label in {"no_memory_budget", "non_int_memory_budget"}:
            assert res.rule_path == "fallback-safe-default"
            assert res.fallback_reason is not None and "128" in res.fallback_reason

    def test_budget_too_small_for_any_candidate_fails_safe(self) -> None:
        # Budget smaller than even the 64-candidate estimate.
        budget = peak_memory_estimate(64, _ctx(n_tiles=72)) - 1
        ctx = _ctx(memory_budget_bytes=budget, preview_quality=0.9)
        res = resolve_working_size(ctx)
        assert res.resolved_working_size == SAFE_DEFAULT
        assert res.rule_path == "fallback-safe-default"
        assert res.fallback_reason is not None

    def test_quality_signal_below_threshold_stays_at_128(self) -> None:
        ctx = _ctx(memory_budget_bytes=128 * 1024**3, preview_quality=0.01)
        res = resolve_working_size(ctx)
        assert res.resolved_working_size == SAFE_DEFAULT
        assert res.rule_path == "baseline-default"
        assert res.fallback_reason is None


# ---------------------------------------------------------------------------
# Memory-ceiling shrink branch (always safe)
# ---------------------------------------------------------------------------


class TestMemoryCeilingShrink:
    """When 128 does not fit, return the largest feasible size <= 128."""

    @pytest.mark.parametrize("n_tiles", [72, 200, 1])
    def test_shrink_to_largest_feasible_at_or_below_128(self, n_tiles: int) -> None:
        # Budget fits 96 but not 128.
        base_ctx = _ctx(n_tiles=n_tiles)
        budget = peak_memory_estimate(96, base_ctx)  # fits 96 exactly
        ctx = _ctx(n_tiles=n_tiles, memory_budget_bytes=budget, preview_quality=0.0)
        res = resolve_working_size(ctx)
        assert res.resolved_working_size == 96
        assert res.rule_path == "memory-ceiling-shrink"
        assert res.fallback_reason is None

    def test_shrink_to_64_when_only_64_fits(self) -> None:
        base_ctx = _ctx(n_tiles=72)
        budget = peak_memory_estimate(64, base_ctx)
        ctx = _ctx(memory_budget_bytes=budget, preview_quality=0.9)
        res = resolve_working_size(ctx)
        assert res.resolved_working_size == 64
        assert res.rule_path == "memory-ceiling-shrink"

    def test_huge_budget_stays_at_128(self) -> None:
        ctx = _ctx(memory_budget_bytes=10**15, preview_quality=0.0)
        res = resolve_working_size(ctx)
        assert res.resolved_working_size == SAFE_DEFAULT
        assert res.rule_path == "baseline-default"

    def test_shrink_does_not_raise_even_with_strong_quality_signal(self) -> None:
        # Shrink branch wins over raise: once below 128, do not enlarge.
        base_ctx = _ctx(n_tiles=72)
        budget = peak_memory_estimate(96, base_ctx)
        ctx = _ctx(memory_budget_bytes=budget, preview_quality=0.99)
        res = resolve_working_size(ctx)
        assert res.resolved_working_size == 96
        assert res.rule_path == "memory-ceiling-shrink"


# ---------------------------------------------------------------------------
# Quality-floor raise branch (opt-in, gated)
# ---------------------------------------------------------------------------


class TestQualityFloorRaise:
    """Enlarges only when 128 fits, preview > threshold, and 160 is feasible."""

    def test_raises_to_largest_feasible_above_128(self) -> None:
        ctx = _ctx(memory_budget_bytes=128 * 1024**3, preview_quality=0.9)
        res = resolve_working_size(ctx)
        assert res.resolved_working_size == 192  # everything > 128 fits
        assert res.rule_path == "quality-floor-raise"

    def test_raises_to_160_when_192_does_not_fit(self) -> None:
        base_ctx = _ctx(n_tiles=72)
        # Budget fits 128 and 160 but not 192.
        budget = peak_memory_estimate(160, base_ctx)
        ctx = _ctx(memory_budget_bytes=budget, preview_quality=0.9)
        res = resolve_working_size(ctx)
        assert res.resolved_working_size == 160
        assert res.rule_path == "quality-floor-raise"

    def test_does_not_raise_when_160_does_not_fit(self) -> None:
        base_ctx = _ctx(n_tiles=72)
        # Budget fits exactly 128 but not 160.
        budget = peak_memory_estimate(128, base_ctx)
        ctx = _ctx(memory_budget_bytes=budget, preview_quality=0.9)
        res = resolve_working_size(ctx)
        assert res.resolved_working_size == SAFE_DEFAULT
        assert res.rule_path == "baseline-default"

    def test_does_not_raise_at_threshold_boundary(self) -> None:
        # Preview exactly at threshold: strictly greater required.
        ctx = _ctx(
            memory_budget_bytes=128 * 1024**3,
            preview_quality=WORKING_SIZE_QUALITY_RAISE_THRESHOLD,
        )
        res = resolve_working_size(ctx)
        assert res.resolved_working_size == SAFE_DEFAULT
        assert res.rule_path == "baseline-default"


# ---------------------------------------------------------------------------
# Purity & determinism
# ---------------------------------------------------------------------------


class TestPurity:
    def test_resolver_is_pure_and_deterministic(self) -> None:
        ctx = _ctx(memory_budget_bytes=40 * 1024**3, preview_quality=0.05)
        r1 = resolve_working_size(ctx)
        r2 = resolve_working_size(ctx)
        assert r1 == r2

    def test_context_is_frozen(self) -> None:
        ctx = _ctx()
        with pytest.raises((AttributeError, Exception)):
            ctx.n_z = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Grid invariants
# ---------------------------------------------------------------------------


class TestGridInvariants:
    def test_grid_contains_safe_default(self) -> None:
        assert SAFE_DEFAULT in WORKING_SIZE_GRID

    def test_grid_is_sorted_asc(self) -> None:
        assert list(WORKING_SIZE_GRID) == sorted(WORKING_SIZE_GRID)

    def test_grid_min_is_64(self) -> None:
        assert min(WORKING_SIZE_GRID) == 64

    def test_resolved_value_always_in_grid(self) -> None:
        for budget in [None, 0, peak_memory_estimate(64, _ctx()), 128 * 1024**3, 10**15]:
            for pq in [None, 0.0, 0.01, 0.5, 0.9, 1.0, 1.5]:
                res = resolve_working_size(_ctx(memory_budget_bytes=budget, preview_quality=pq))
                assert res.resolved_working_size in WORKING_SIZE_GRID

    def test_grid_mirrors_tuning_default_search_space(self) -> None:
        """The resolver grid must not drift from the validated Optuna grid."""
        from linum_basic.tuning import _DEFAULT_SEARCH_SPACE

        tuning_grid = list(_DEFAULT_SEARCH_SPACE["working_size"])
        assert tuning_grid == list(WORKING_SIZE_GRID), (
            "WORKING_SIZE_GRID has drifted from _DEFAULT_SEARCH_SPACE['working_size']; update the resolver grid to match."
        )


# ---------------------------------------------------------------------------
# Observability metadata
# ---------------------------------------------------------------------------


class TestObservability:
    def test_metadata_has_required_keys(self) -> None:
        res = resolve_working_size(_ctx(memory_budget_bytes=40 * 1024**3, preview_quality=0.05))
        md = res.metadata()
        assert set(md.keys()) >= REQUIRED_METADATA_KEYS

    def test_signals_snapshot_has_required_keys(self) -> None:
        ctx = _ctx(
            memory_budget_bytes=40 * 1024**3,
            preview_quality=0.05,
            tile_shape=(2048, 2048),
        )
        res = resolve_working_size(ctx)
        assert set(res.signals.keys()) >= REQUIRED_SIGNAL_KEYS
        assert res.signals["n_z"] == ctx.n_z
        assert res.signals["n_tiles"] == ctx.n_tiles
        assert res.signals["field_mode"] == ctx.field_mode
        assert res.signals["tile_shape"] == [2048, 2048]
        assert res.signals["memory_budget_bytes"] == 40 * 1024**3
        assert res.signals["preview_quality"] == pytest.approx(0.05)

    def test_peak_memory_per_candidate(self) -> None:
        ctx = _ctx(n_tiles=72)
        res = resolve_working_size(_ctx(memory_budget_bytes=128 * 1024**3))
        assert set(res.peak_memory_estimate_bytes.keys()) == {str(s) for s in WORKING_SIZE_GRID}
        for s in WORKING_SIZE_GRID:
            assert res.peak_memory_estimate_bytes[str(s)] == peak_memory_estimate(s, ctx)

    def test_candidate_grid_in_metadata(self) -> None:
        res = resolve_working_size(_ctx(memory_budget_bytes=128 * 1024**3))
        assert res.metadata()["candidate_grid"] == list(WORKING_SIZE_GRID)

    def test_rule_path_is_one_of_valid_set(self) -> None:
        for budget in [None, 0, peak_memory_estimate(96, _ctx()), 128 * 1024**3, 10**15]:
            for pq in [None, 0.0, 0.9]:
                res = resolve_working_size(_ctx(memory_budget_bytes=budget, preview_quality=pq))
                assert res.rule_path in VALID_RULE_PATHS

    def test_gate_status_is_opt_in(self) -> None:
        res = resolve_working_size(_ctx(memory_budget_bytes=128 * 1024**3))
        assert res.gate_status == GATE_STATUS_OPT_IN

    def test_requested_is_echoed(self) -> None:
        ctx = _ctx(memory_budget_bytes=128 * 1024**3, requested="auto")
        assert resolve_working_size(ctx).requested == "auto"
        ctx2 = _ctx(memory_budget_bytes=128 * 1024**3, requested=96)
        assert resolve_working_size(ctx2).requested == 96


# ---------------------------------------------------------------------------
# compute_preview_quality signal
# ---------------------------------------------------------------------------


class TestPreviewQuality:
    def test_smooth_image_low_hf_fraction(self) -> None:
        # A smooth gradient concentrates energy in low frequencies.
        yy, xx = np.mgrid[0:512, 0:512]
        smooth = 100.0 + 0.01 * xx + 0.01 * yy
        frac = compute_preview_quality(smooth)
        assert frac is not None
        assert 0.0 <= frac <= WORKING_SIZE_QUALITY_RAISE_THRESHOLD

    def test_high_frequency_image_has_higher_hf_fraction(self) -> None:
        rng = np.random.default_rng(0)
        noisy = 100.0 + 50.0 * rng.standard_normal((512, 512))
        frac = compute_preview_quality(noisy)
        assert frac is not None
        assert frac > 0.2

    def test_none_for_non_finite(self) -> None:
        bad = np.full((64, 64), np.inf)
        assert compute_preview_quality(bad) is None

    def test_none_for_non_positive_mean(self) -> None:
        assert compute_preview_quality(np.zeros((64, 64))) is None

    def test_none_for_wrong_shape(self) -> None:
        assert compute_preview_quality(np.zeros((64, 64, 3))) is None
        assert compute_preview_quality(np.zeros((0,))) is None

    def test_none_for_none_input(self) -> None:
        assert compute_preview_quality(None) is None  # type: ignore[arg-type]

    def test_resolution_is_256(self) -> None:
        assert PREVIEW_RESOLUTION == 256

    def test_smooth_vs_noisy_ordering(self) -> None:
        yy, xx = np.mgrid[0:512, 0:512]
        smooth = 100.0 + 0.01 * xx + 0.01 * yy
        rng = np.random.default_rng(1)
        noisy = 100.0 + 50.0 * rng.standard_normal((512, 512))
        fs = compute_preview_quality(smooth)
        fn = compute_preview_quality(noisy)
        assert fs is not None and fn is not None
        assert fn > fs


# ---------------------------------------------------------------------------
# build_working_size_context (wiring helper, no solver)
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_explicit_budget_is_used(self) -> None:
        ctx = wsm.build_working_size_context(
            n_z=10,
            n_tiles=20,
            field_mode="per-z",
            tile_shape=(512, 512),
            memory_budget_bytes=10**12,
            requested="auto",
        )
        assert ctx.memory_budget_bytes == 10**12

    def test_mean_image_drives_preview(self) -> None:
        yy, xx = np.mgrid[0:256, 0:256]
        mean_img = 100.0 + 0.01 * xx + 0.01 * yy
        ctx = wsm.build_working_size_context(
            n_z=10,
            n_tiles=20,
            field_mode="per-z",
            tile_shape=(256, 256),
            memory_budget_bytes=10**12,
            mean_image=mean_img,
        )
        assert ctx.preview_quality is not None
        assert 0.0 <= ctx.preview_quality <= 1.0

    def test_no_mean_image_yields_none_preview(self) -> None:
        ctx = wsm.build_working_size_context(
            n_z=10,
            n_tiles=20,
            field_mode="per-z",
            tile_shape=(256, 256),
            memory_budget_bytes=10**12,
        )
        assert ctx.preview_quality is None


# ---------------------------------------------------------------------------
# Resolution result dataclass basics
# ---------------------------------------------------------------------------


class TestResolutionDataclass:
    def test_is_frozen(self) -> None:
        res = resolve_working_size(_ctx(memory_budget_bytes=128 * 1024**3))
        with pytest.raises((AttributeError, Exception)):
            res.resolved_working_size = 64  # type: ignore[misc]
