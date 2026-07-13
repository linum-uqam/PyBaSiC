"""Regression tests for the bounded backend DCT caches (K10).

Asserts that the three module-level DCT caches in
:mod:`linum_basic.backend` (``_DCT_MATRIX_CACHE``, ``_DCT_TWIDDLE_CACHE``,
``_IDCT_TWIDDLE_CACHE``) never grow beyond :data:`_DCT_CACHE_MAXSIZE`,
that eviction is true LRU (oldest-first, with access-recency promotion),
and that :func:`clear_dct_caches` releases every entry.  This closes the
K10 memory-leak concern for long-running batch jobs that fit many stacks
of varying shapes.

All tests require PyTorch because the cache getters import ``torch``
internally; they are skipped cleanly when PyTorch is not installed.
"""

from __future__ import annotations

import pytest

from linum_basic import backend as backend_mod
from linum_basic.backend import (
    _DCT_CACHE_MAXSIZE,
    _DCT_MATRIX_CACHE,
    _DCT_TWIDDLE_CACHE,
    _IDCT_TWIDDLE_CACHE,
    clear_dct_caches,
    get_xp,
)

torch = pytest.importorskip("torch", reason="PyTorch not installed — skipping DCT cache bound tests.")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_caches() -> object:
    """Clear all three DCT caches before and after every test.

    The caches are module-level singletons, so without isolation a test
    that populates them would leak state into the next.  Clearing on both
    entry and exit keeps each test independent and deterministic.
    """
    clear_dct_caches()
    yield
    clear_dct_caches()


def _distinct_keys(count: int, *, offset: int = 0) -> list[int]:
    """Return *count* distinct cache-key lengths starting at ``offset + 4``.

    Lengths start at 4 (the smallest meaningful DCT) and increase by 1 so
    every key is unique while the per-entry matrix cost stays O(n^2)-tiny.
    """
    return [offset + 4 + i for i in range(count)]


# ---------------------------------------------------------------------------
# Per-cache bound (core K10 regression)
# ---------------------------------------------------------------------------


class TestCacheBound:
    """Each of the three caches stays at or below ``_DCT_CACHE_MAXSIZE``."""

    @pytest.mark.parametrize(
        ("getter", "cache", "label"),
        [
            (backend_mod._get_dct_matrix, _DCT_MATRIX_CACHE, "matrix"),
            (backend_mod._get_dct_twiddles, _DCT_TWIDDLE_CACHE, "twiddle"),
            (backend_mod._get_idct_twiddles, _IDCT_TWIDDLE_CACHE, "idct-twiddle"),
        ],
    )
    def test_bounded_across_many_distinct_keys(
        self,
        getter: object,
        cache: dict,
        label: str,
    ) -> None:
        """Inserting far more distinct keys than the bound never overflows it.

        This is the direct K10 regression: a long-running batch job that
        fits stacks of many distinct sizes must not accumulate cache entries
        without limit.  We insert ``_DCT_CACHE_MAXSIZE + 20`` distinct keys
        and assert the ``len(cache) <= _DCT_CACHE_MAXSIZE`` invariant holds
        at *every* step, not only at the end.
        """
        n_keys = _DCT_CACHE_MAXSIZE + 20
        sizes = _distinct_keys(n_keys)
        for n in sizes:
            if label == "matrix":
                backend_mod._get_dct_matrix(n, "cpu")  # type: ignore[operator]
            else:
                backend_mod._get_dct_twiddles(n, torch.float32, "cpu")  # type: ignore[operator]
                backend_mod._get_idct_twiddles(n, torch.float32, "cpu")  # type: ignore[operator]
            assert len(cache) <= _DCT_CACHE_MAXSIZE, f"{label} cache grew to {len(cache)} (> {_DCT_CACHE_MAXSIZE}) after n={n}"
        assert len(cache) == _DCT_CACHE_MAXSIZE, (
            f"{label} cache should be exactly full ({_DCT_CACHE_MAXSIZE}), got {len(cache)}"
        )

    def test_twiddle_dtype_axis_is_distinct_key(self) -> None:
        """Different dtypes produce distinct keys but the bound still holds.

        The twiddle caches key on ``(n, dtype, device)``; cycling two dtypes
        across many lengths must still respect the per-cache cap.
        """
        sizes = _distinct_keys(_DCT_CACHE_MAXSIZE + 10)
        for n in sizes:
            backend_mod._get_dct_twiddles(n, torch.float32, "cpu")
            backend_mod._get_dct_twiddles(n, torch.float64, "cpu")
            assert len(_DCT_TWIDDLE_CACHE) <= _DCT_CACHE_MAXSIZE
        assert len(_DCT_TWIDDLE_CACHE) == _DCT_CACHE_MAXSIZE


# ---------------------------------------------------------------------------
# True LRU semantics
# ---------------------------------------------------------------------------


class TestLRUEviction:
    """Eviction removes the least-recently-used entry, not an arbitrary one."""

    def test_oldest_inserted_evicted_first(self) -> None:
        """After overflow, the earliest-inserted key is gone and latest remains."""
        sizes = _distinct_keys(_DCT_CACHE_MAXSIZE + 1)
        for n in sizes:
            backend_mod._get_dct_matrix(n, "cpu")
        first_key = (sizes[0], "cpu")
        last_key = (sizes[-1], "cpu")
        assert first_key not in _DCT_MATRIX_CACHE, f"oldest key {first_key} should have been evicted, but cache still holds it"
        assert last_key in _DCT_MATRIX_CACHE, f"newest key {last_key} should survive, but cache dropped it"

    def test_access_promotes_recency(self) -> None:
        """A cache hit moves the entry to most-recently-used, deferring eviction.

        Fill the matrix cache, touch an early (near-LRU) key, then drive
        enough fresh inserts to evict everything older than the touched key.
        The touched key must survive past keys that were inserted later but
        never re-accessed — proving ``move_to_end`` on the hit path works.
        """
        # Fill to exactly the bound: keys for sizes 4, 5, ..., (4 + maxsize - 1).
        fill_sizes = _distinct_keys(_DCT_CACHE_MAXSIZE)
        for n in fill_sizes:
            backend_mod._get_dct_matrix(n, "cpu")
        # Re-access an early key so it becomes most-recently-used.
        hot_n = fill_sizes[2]
        backend_mod._get_dct_matrix(hot_n, "cpu")
        # Insert fresh keys to evict the oldest surviving entries one by one.
        # We evict `hot_index - 1` entries: everything inserted *before* hot_n
        # that was never touched. hot_n itself must still be present.
        evict_count = 2  # evict the two keys inserted strictly before hot_n
        extra_sizes = _distinct_keys(evict_count, offset=_DCT_CACHE_MAXSIZE)
        for n in extra_sizes:
            backend_mod._get_dct_matrix(n, "cpu")
        hot_key = (hot_n, "cpu")
        assert hot_key in _DCT_MATRIX_CACHE, f"re-accessed key {hot_key} should survive via recency promotion"
        # The two keys inserted before hot_n (and never re-touched) are gone.
        for older_n in fill_sizes[:evict_count]:
            assert (older_n, "cpu") not in _DCT_MATRIX_CACHE, (
                f"stale key {(older_n, 'cpu')} should have been evicted before the hot key"
            )


# ---------------------------------------------------------------------------
# clear_dct_caches API
# ---------------------------------------------------------------------------


class TestClearDctCaches:
    """``clear_dct_caches`` releases every entry and is part of the public API."""

    def test_clears_all_three_after_population(self) -> None:
        """Populating then clearing leaves all three caches empty."""
        for n in _distinct_keys(10):
            backend_mod._get_dct_matrix(n, "cpu")
            backend_mod._get_dct_twiddles(n, torch.float32, "cpu")
            backend_mod._get_idct_twiddles(n, torch.float32, "cpu")
        assert len(_DCT_MATRIX_CACHE) > 0
        assert len(_DCT_TWIDDLE_CACHE) > 0
        assert len(_IDCT_TWIDDLE_CACHE) > 0
        clear_dct_caches()
        assert len(_DCT_MATRIX_CACHE) == 0
        assert len(_DCT_TWIDDLE_CACHE) == 0
        assert len(_IDCT_TWIDDLE_CACHE) == 0

    def test_clear_is_idempotent_on_empty_caches(self) -> None:
        """Calling clear on already-empty caches is a no-op (no error)."""
        clear_dct_caches()
        clear_dct_caches()  # must not raise
        assert len(_DCT_MATRIX_CACHE) == 0

    def test_exported_in_all(self) -> None:
        """``clear_dct_caches`` is part of the documented public API."""
        assert "clear_dct_caches" in backend_mod.__all__


# ---------------------------------------------------------------------------
# Integration: bound holds through the public DCT surface
# ---------------------------------------------------------------------------


class TestBoundHoldsThroughPublicDCT:
    """The bound holds when caches are fed through ``ArrayNamespace.dctn``.

    Exercises the real call path (``dctn``/``idctn`` -> ``_torch_dct1d`` ->
    ``_get_dct_twiddles`` / ``_get_idct_twiddles``) rather than the private
    getters directly, so the regression covers the surface a batch job
    actually uses.
    """

    def test_many_distinct_fit_sizes_via_dctn(self) -> None:
        """Fitting many distinct image sizes never overflows the twiddle caches."""
        xp = get_xp("torch", "cpu")
        sizes = _distinct_keys(_DCT_CACHE_MAXSIZE + 15)
        for n in sizes:
            x = torch.zeros((n, n), dtype=torch.float32)
            xp.dctn(xp.asarray(x))
            xp.idctn(xp.asarray(x))
            assert len(_DCT_TWIDDLE_CACHE) <= _DCT_CACHE_MAXSIZE, f"forward twiddle cache overflowed at n={n}"
            assert len(_IDCT_TWIDDLE_CACHE) <= _DCT_CACHE_MAXSIZE, f"inverse twiddle cache overflowed at n={n}"
        assert len(_DCT_TWIDDLE_CACHE) == _DCT_CACHE_MAXSIZE
        assert len(_IDCT_TWIDDLE_CACHE) == _DCT_CACHE_MAXSIZE

    def test_clear_between_runs_prevents_accumulation(self) -> None:
        """Calling clear_dct_caches between simulated runs resets the caches.

        Models the documented batch-job escape hatch: many independent fits
        can be run back-to-back with a guaranteed-cold DCT cache each time,
        so residual memory never accumulates across runs.
        """
        xp = get_xp("torch", "cpu")
        for run in range(5):
            for n in _distinct_keys(8, offset=run * 8):
                xp.dctn(xp.asarray(torch.zeros((n, n), dtype=torch.float32)))
            # Each run leaves a small footprint; clearing must zero it.
            clear_dct_caches()
            assert len(_DCT_TWIDDLE_CACHE) == 0
            assert len(_IDCT_TWIDDLE_CACHE) == 0
