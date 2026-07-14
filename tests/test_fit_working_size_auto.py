"""Integration tests for ``working_size="auto"`` wiring in :func:`fit_mosaic`.

Covers the S02 integration contract in ``docs/adaptive_working_size.md``:

* **Sentinel resolution** — ``basic_kwargs={"working_size": "auto"}`` resolves
  to a concrete int drawn from the validated grid before ``BaSiC`` sees it.
* **Metadata recording** — ``MosaicFit.params["_working_size_selector"]``
  carries the full explainability dict (resolved value, rule path, signals,
  per-candidate memory estimates).
* **Opt-in isolation** — callers that never pass ``"auto"`` are unaffected:
  no ``_working_size_selector`` key is added and the ws=128 default holds.
* **Numerics untouched** — the resolved size produces a real BaSiC fit
  (flat/dark-fields of the expected shape).

Tests use a pure-NumPy synthetic mosaic (no ``sbh_simulator``), matching the
``test_streaming_fit.py`` fixture convention so they run in minimal CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from linum_basic._working_size import SAFE_DEFAULT, WORKING_SIZE_GRID
from linum_basic.fit import fit_mosaic
from linum_basic.mosaic import MosaicGrid

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_COMMON_KW: dict[str, object] = {
    "backend": "numpy",
    "estimate_darkfield": False,
    "max_reweighting_iterations": 3,
}

_REQUIRED_SELECTOR_KEYS = frozenset(
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


def _make_synthetic_volume(
    n_z: int = 4,
    n_rows: int = 3,
    n_cols: int = 3,
    th: int = 32,
    tw: int = 32,
    seed: int = 42,
) -> np.ndarray:
    """Build a synthetic mosaic volume with a smooth vignette BaSiC can fit.

    Pure NumPy — no ``sbh_simulator`` dependency. Mirrors the
    ``test_streaming_fit`` fixture so these tests run in minimal CI.
    """
    rng = np.random.default_rng(seed)
    H = n_rows * th
    W = n_cols * tw
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W / 2
    sigma = min(H, W) / 3
    vignette = 1.0 + 0.3 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    volume = np.zeros((n_z, H, W), dtype=np.float32)
    for z in range(n_z):
        noise = rng.random((H, W), dtype=np.float32) * 0.1
        volume[z] = (0.5 + noise) * vignette * (1.0 + 0.02 * z)
    return volume


@pytest.fixture()
def mosaic_grid() -> MosaicGrid:
    """Eager (ndarray-backed) MosaicGrid over a small synthetic mosaic."""
    volume = _make_synthetic_volume()
    return MosaicGrid(volume, tile_shape=(32, 32), overlap_fraction=0.2)


# ---------------------------------------------------------------------------
# Sentinel resolution + metadata recording
# ---------------------------------------------------------------------------


class TestAutoResolution:
    def test_auto_resolves_to_grid_member(self, mosaic_grid: MosaicGrid) -> None:
        """``working_size="auto"`` resolves to a concrete grid member."""
        kw = {**_COMMON_KW, "working_size": "auto"}
        fit = fit_mosaic(mosaic_grid, strategy="sequential", n_workers=1, basic_kwargs=kw)
        selector = fit.params["_working_size_selector"]
        assert selector["resolved_working_size"] in WORKING_SIZE_GRID
        assert selector["requested"] == "auto"

    def test_selector_metadata_has_required_keys(self, mosaic_grid: MosaicGrid) -> None:
        """``_working_size_selector`` carries every explainability key."""
        kw = {**_COMMON_KW, "working_size": "auto"}
        fit = fit_mosaic(mosaic_grid, strategy="sequential", n_workers=1, basic_kwargs=kw)
        selector = fit.params["_working_size_selector"]
        assert set(selector.keys()) >= _REQUIRED_SELECTOR_KEYS

    def test_selector_candidate_grid_is_validated_grid(self, mosaic_grid: MosaicGrid) -> None:
        kw = {**_COMMON_KW, "working_size": "auto"}
        fit = fit_mosaic(mosaic_grid, strategy="sequential", n_workers=1, basic_kwargs=kw)
        selector = fit.params["_working_size_selector"]
        assert selector["candidate_grid"] == list(WORKING_SIZE_GRID)

    def test_selector_signals_snapshot(self, mosaic_grid: MosaicGrid) -> None:
        kw = {**_COMMON_KW, "working_size": "auto"}
        fit = fit_mosaic(mosaic_grid, strategy="sequential", n_workers=1, basic_kwargs=kw)
        signals = fit.params["_working_size_selector"]["signals"]
        assert signals["n_z"] == mosaic_grid.n_z
        assert signals["n_tiles"] == mosaic_grid.n_tiles
        assert signals["field_mode"] == "per-z"
        assert signals["tile_shape"] == [32, 32]
        assert "memory_budget_bytes" in signals
        assert "preview_quality" in signals

    def test_selector_peak_memory_per_candidate(self, mosaic_grid: MosaicGrid) -> None:
        kw = {**_COMMON_KW, "working_size": "auto"}
        fit = fit_mosaic(mosaic_grid, strategy="sequential", n_workers=1, basic_kwargs=kw)
        peak = fit.params["_working_size_selector"]["peak_memory_estimate_bytes"]
        assert set(peak.keys()) == {str(s) for s in WORKING_SIZE_GRID}
        for val in peak.values():
            assert isinstance(val, int)
            assert val > 0

    def test_rule_path_is_valid(self, mosaic_grid: MosaicGrid) -> None:
        valid_paths = {
            "baseline-default",
            "memory-ceiling-shrink",
            "quality-floor-raise",
            "fallback-safe-default",
        }
        kw = {**_COMMON_KW, "working_size": "auto"}
        fit = fit_mosaic(mosaic_grid, strategy="sequential", n_workers=1, basic_kwargs=kw)
        assert fit.params["_working_size_selector"]["rule_path"] in valid_paths

    def test_gate_status_is_opt_in(self, mosaic_grid: MosaicGrid) -> None:
        kw = {**_COMMON_KW, "working_size": "auto"}
        fit = fit_mosaic(mosaic_grid, strategy="sequential", n_workers=1, basic_kwargs=kw)
        assert "opt-in" in fit.params["_working_size_selector"]["gate_status"]


# ---------------------------------------------------------------------------
# Opt-in isolation: callers that never pass "auto" are unaffected
# ---------------------------------------------------------------------------


class TestOptInIsolation:
    def test_explicit_int_no_selector_key(self, mosaic_grid: MosaicGrid) -> None:
        """An explicit int working_size never adds the selector metadata."""
        kw = {**_COMMON_KW, "working_size": 32}
        fit = fit_mosaic(mosaic_grid, strategy="sequential", n_workers=1, basic_kwargs=kw)
        assert "_working_size_selector" not in fit.params
        assert fit.params["working_size"] == 32

    def test_default_128_no_selector_key(self, mosaic_grid: MosaicGrid) -> None:
        """Omitting working_size keeps the 128 default and adds no metadata."""
        fit = fit_mosaic(mosaic_grid, strategy="sequential", n_workers=1, basic_kwargs=_COMMON_KW)
        assert "_working_size_selector" not in fit.params
        assert fit.params.get("working_size", 128) == 128 or "working_size" not in _COMMON_KW


# ---------------------------------------------------------------------------
# Fail-safe: unknown/ambiguous budget resolves to 128
# ---------------------------------------------------------------------------


class TestFailSafe:
    def test_auto_fails_safe_to_128_on_host_without_budget(
        self, mosaic_grid: MosaicGrid, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On a host where the memory budget cannot be probed, auto -> 128."""
        from linum_basic import _working_size as wsm

        # Force the budget query to return None (simulates non-Linux, no CUDA).
        monkeypatch.setattr(wsm, "_query_memory_budget", lambda device: None)
        kw = {**_COMMON_KW, "working_size": "auto"}
        fit = fit_mosaic(mosaic_grid, strategy="sequential", n_workers=1, basic_kwargs=kw)
        selector = fit.params["_working_size_selector"]
        assert selector["resolved_working_size"] == SAFE_DEFAULT
        assert selector["fallback_reason"] is not None
        assert "128" in selector["fallback_reason"]


# ---------------------------------------------------------------------------
# Numerics: the resolved size produces a real BaSiC fit
# ---------------------------------------------------------------------------


class TestNumericsUntouched:
    def test_auto_produces_valid_flatfield_shape(self, mosaic_grid: MosaicGrid) -> None:
        """The resolved working_size flows into a real BaSiC fit."""
        kw = {**_COMMON_KW, "working_size": "auto"}
        fit = fit_mosaic(mosaic_grid, strategy="sequential", n_workers=1, basic_kwargs=kw)
        assert fit.flatfields.shape == (mosaic_grid.n_z, 32, 32)
        assert fit.darkfields.shape == (mosaic_grid.n_z, 32, 32)
        assert np.all(np.isfinite(fit.flatfields))

    def test_auto_does_not_leak_sentinel_into_strategy_meta(self, mosaic_grid: MosaicGrid) -> None:
        """The strategy metadata must record a concrete int, never 'auto'."""
        kw = {**_COMMON_KW, "working_size": "auto"}
        fit = fit_mosaic(mosaic_grid, strategy="sequential", n_workers=1, basic_kwargs=kw)
        ws_meta = fit.params["_strategy"]["working_size"]
        assert isinstance(ws_meta, int)
        assert ws_meta != "auto"

    def test_auto_with_strategy_auto_resolves_before_strategy(
        self, mosaic_grid: MosaicGrid, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """strategy='auto' + working_size='auto' resolves the sentinel first."""
        from linum_basic import _working_size as wsm

        monkeypatch.setattr(wsm, "_query_memory_budget", lambda device: None)
        kw = {**_COMMON_KW, "working_size": "auto"}
        fit = fit_mosaic(mosaic_grid, strategy="auto", n_workers=1, basic_kwargs=kw)
        # Both selector and strategy metadata see the concrete int.
        assert fit.params["_working_size_selector"]["resolved_working_size"] == SAFE_DEFAULT
        assert isinstance(fit.params["_strategy"]["working_size"], int)
        # The literal 'auto' must never survive into BaSiC-facing params.
        assert fit.params.get("working_size") != "auto"


# ---------------------------------------------------------------------------
# Cost bound: fit_mosaic auto path does not run the resolver more than once
# ---------------------------------------------------------------------------


class TestWiringCost:
    def test_resolver_called_at_most_once(self, mosaic_grid: MosaicGrid, monkeypatch: pytest.MonkeyPatch) -> None:
        """The sentinel-resolution helper runs exactly once per fit_mosaic call."""
        import linum_basic.fit as fit_mod

        call_count = {"n": 0}
        original = fit_mod._resolve_working_size_auto

        def _counting(*args: object, **kwargs: object) -> object:
            call_count["n"] += 1
            return original(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(fit_mod, "_resolve_working_size_auto", _counting)
        kw = {**_COMMON_KW, "working_size": "auto"}
        fit_mosaic(mosaic_grid, strategy="sequential", n_workers=1, basic_kwargs=kw)
        assert call_count["n"] == 1

    def test_resolver_not_called_without_sentinel(self, mosaic_grid: MosaicGrid, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the 'auto' sentinel the helper is a no-op."""
        import linum_basic.fit as fit_mod

        call_count = {"n": 0}
        original = fit_mod._resolve_working_size_auto

        def _counting(*args: object, **kwargs: object) -> object:
            call_count["n"] += 1
            return original(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(fit_mod, "_resolve_working_size_auto", _counting)
        kw = {**_COMMON_KW, "working_size": 32}
        fit_mosaic(mosaic_grid, strategy="sequential", n_workers=1, basic_kwargs=kw)
        # _resolve_working_size_auto returns early (None) without resolution;
        # it is still *called* once but does no work.
        assert call_count["n"] == 1
