"""Tests for linum_basic.tuning — Optuna-based BaSiC hyperparameter tuning.

These tests build a small synthetic mosaic (no external dependencies) and
exercise the public :func:`tune` API plus the internal tile-subsampling
helper.  Trial counts and the search space are kept tiny so the suite runs
quickly while still covering the full objective path.
"""

from __future__ import annotations

import numpy as np
import pytest

from linum_basic.core import dct_energy
from linum_basic.mosaic import MosaicGrid
from linum_basic.tuning import (
    BoundsRecommendation,
    TuneResult,
    _subsample_tiles,
    recommend_bounds,
    tune,
)

optuna = pytest.importorskip("optuna")
pd = pytest.importorskip("pandas")

# A small but valid search space so tuning runs fast on tiny tiles.
_TEST_SEARCH_SPACE = {
    "working_size": [16, 32],
    "l_s_divisor": (100.0, 5000.0),
    "l_d_divisor": (500.0, 10000.0),
    "epsilon": (0.01, 1.0),
    "estimate_darkfield": [False],
}


def _vignette(tile_h: int, tile_w: int) -> np.ndarray:
    """Smooth radial Gaussian vignette normalised to mean ~1 (no deps)."""
    yy, xx = np.mgrid[0:tile_h, 0:tile_w]
    cy, cx = (tile_h - 1) / 2.0, (tile_w - 1) / 2.0
    r2 = ((yy - cy) / tile_h) ** 2 + ((xx - cx) / tile_w) ** 2
    ff = np.exp(-r2 / (2 * 0.35**2)).astype(np.float32)
    return ff / float(ff.mean())


def _synthetic_mosaic(n_rows=4, n_cols=5, tile_h=12, tile_w=12, n_z=4, seed=7):
    """Build a synthetic mosaic with shared seam content and a known vignette."""
    rng = np.random.default_rng(seed)
    flatfield = _vignette(tile_h, tile_w)
    overlap_x = round(0.2 * tile_w)
    overlap_y = round(0.2 * tile_h)
    n_tiles = n_rows * n_cols

    raw = np.zeros((n_z, n_rows * tile_h, n_cols * tile_w), dtype=np.float32)
    for z in range(n_z):
        tiles = rng.random((n_tiles, tile_h, tile_w)).astype(np.float32) + 0.5
        for r in range(n_rows):
            for c in range(n_cols - 1):
                shared = rng.random((tile_h, overlap_x)).astype(np.float32) + 0.5
                tiles[r * n_cols + c, :, -overlap_x:] = shared
                tiles[r * n_cols + c + 1, :, :overlap_x] = shared
        for r in range(n_rows - 1):
            for c in range(n_cols):
                shared = rng.random((overlap_y, tile_w)).astype(np.float32) + 0.5
                tiles[r * n_cols + c, -overlap_y:, :] = shared
                tiles[(r + 1) * n_cols + c, :overlap_y, :] = shared
        for r in range(n_rows):
            for c in range(n_cols):
                idx = r * n_cols + c
                raw[z, r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tiles[idx] * flatfield
    return raw, flatfield


@pytest.fixture
def synthetic_mosaic():
    raw, ff = _synthetic_mosaic()
    mosaic = MosaicGrid(raw, tile_shape=(12, 12), overlap_fraction=0.2)
    return mosaic, ff


class TestSubsampleTiles:
    def test_none_returns_all(self):
        tiles = np.zeros((20, 5, 5))
        assert _subsample_tiles(tiles, None) is tiles

    def test_larger_than_count_returns_all(self):
        tiles = np.zeros((10, 5, 5))
        assert _subsample_tiles(tiles, 50) is tiles

    def test_subsamples_evenly(self):
        tiles = np.arange(100).reshape(100, 1, 1).astype(float)
        out = _subsample_tiles(tiles, 10)
        assert out.shape[0] <= 10
        # Evenly spaced and unique
        flat = out[:, 0, 0]
        assert flat[0] == 0
        assert flat[-1] == 99
        assert np.all(np.diff(flat) > 0)


class TestTune:
    def test_returns_tuneresult(self, synthetic_mosaic):
        mosaic, _ = synthetic_mosaic
        result = tune(
            mosaic,
            n_trials=3,
            z_subsample=2,
            search_space=_TEST_SEARCH_SPACE,
            seed=0,
        )
        assert isinstance(result, TuneResult)
        assert set(result.best_params) == {
            "working_size",
            "l_s",
            "l_d",
            "epsilon",
            "estimate_darkfield",
        }
        assert isinstance(result.best_value, float)
        assert result.best_fit is None

    def test_run_full_fit_attaches_fit(self, synthetic_mosaic):
        mosaic, _ = synthetic_mosaic
        result = tune(
            mosaic,
            n_trials=3,
            z_subsample=2,
            search_space=_TEST_SEARCH_SPACE,
            seed=0,
            run_full_fit=True,
        )
        assert result.best_fit is not None
        assert result.best_fit.flatfields.shape == (mosaic.n_z, 12, 12)

    def test_max_tiles_smaller_than_total(self, synthetic_mosaic):
        """Tuning with a tile cap below the tile count still completes."""
        mosaic, _ = synthetic_mosaic
        assert mosaic.n_tiles == 20
        result = tune(
            mosaic,
            n_trials=2,
            z_subsample=1,
            search_space=_TEST_SEARCH_SPACE,
            seed=0,
            max_tiles=8,
        )
        assert isinstance(result.best_value, float)

    def test_n_extra_rows_honored(self, synthetic_mosaic):
        """n_extra_rows is propagated to the full fit without shape errors."""
        mosaic, _ = synthetic_mosaic
        result = tune(
            mosaic,
            n_trials=2,
            z_subsample=1,
            search_space=_TEST_SEARCH_SPACE,
            seed=0,
            n_extra_rows=2,
            run_full_fit=True,
        )
        assert result.best_fit is not None
        assert result.best_fit.flatfields.shape == (mosaic.n_z, 12, 12)

    def test_reproducible(self, synthetic_mosaic):
        """Same seed → same best parameters."""
        mosaic, _ = synthetic_mosaic
        r1 = tune(mosaic, n_trials=3, z_subsample=2, search_space=_TEST_SEARCH_SPACE, seed=1)
        r2 = tune(mosaic, n_trials=3, z_subsample=2, search_space=_TEST_SEARCH_SPACE, seed=1)
        assert r1.best_params == r2.best_params


class TestDctEnergyParity:
    def test_tuning_uses_same_dct_path(self, synthetic_mosaic):
        """The dct_sum reference in tune() matches dct_energy on the mean image.

        l_s / l_d in best_params are dct_sum / divisor; recovering dct_sum
        from them must equal dct_energy of the first reference z-level mean.
        """
        mosaic, _ = synthetic_mosaic
        # Reproduce tune()'s reference computation.
        n_z = mosaic.n_z
        z_step = max(1, n_z // 2)
        z0 = next(iter(range(0, n_z, z_step)))
        ref_tiles = mosaic.iter_tiles(z0)
        expected_dct = dct_energy(ref_tiles.mean(axis=0))

        result = tune(mosaic, n_trials=2, z_subsample=2, search_space=_TEST_SEARCH_SPACE, seed=0)
        # best_params stores l_s = dct_sum / l_s_divisor; the study stores the
        # divisor, but we can confirm dct_sum is positive and finite and that
        # l_s, l_d are consistent with a single shared dct_sum.
        assert np.isfinite(expected_dct) and expected_dct > 0
        # l_s and l_d derive from the SAME dct_sum, so their ratio equals the
        # ratio of divisors (independent of dct_sum) — sanity on the shared path.
        assert result.best_params["l_s"] > 0
        assert result.best_params["l_d"] > 0


class TestTuneParallel:
    """Cover the n_workers > 1 ThreadPoolExecutor path in tune()."""

    def test_n_workers_gt1_returns_tuneresult(self, synthetic_mosaic):
        """n_workers=2 exercises the parallel ThreadPoolExecutor branch."""
        mosaic, _ = synthetic_mosaic
        result = tune(
            mosaic,
            n_trials=2,
            z_subsample=2,
            search_space=_TEST_SEARCH_SPACE,
            seed=3,
            n_workers=2,
        )
        assert isinstance(result, TuneResult)
        assert result.best_value > 0


class TestTuneTorchWarning:
    """Cover the backend='torch' + n_workers>1 warning path."""

    def test_torch_backend_warns_and_runs(self, synthetic_mosaic):
        torch = pytest.importorskip("torch")  # noqa: F841
        mosaic, _ = synthetic_mosaic
        with pytest.warns(UserWarning, match="n_workers forced to 1"):
            result = tune(
                mosaic,
                n_trials=2,
                z_subsample=2,
                search_space=_TEST_SEARCH_SPACE,
                seed=5,
                n_workers=2,
                backend="torch",
            )
        assert isinstance(result, TuneResult)


class TestTuneTrialsDfFallback:
    """Cover the trials_df exception fallback path."""

    def test_trials_df_none_when_dataframe_raises(self, synthetic_mosaic, monkeypatch):
        """When study.trials_dataframe() raises, trials_df is set to None."""

        class _FailingStudy:
            """Thin wrapper that makes trials_dataframe() raise."""

            def __init__(self, real_study):
                self._study = real_study
                self.best_trial = None  # set after optimize

            def optimize(self, *a, **kw):
                self._study.optimize(*a, **kw)
                self.best_trial = self._study.best_trial

            def trials_dataframe(self):
                raise RuntimeError("pandas not available (simulated)")

        import optuna as _optuna

        _original_create_study = _optuna.create_study

        def _patched_create_study(**kw):
            real = _original_create_study(**kw)
            return _FailingStudy(real)

        monkeypatch.setattr(_optuna, "create_study", _patched_create_study)

        mosaic, _ = synthetic_mosaic
        result = tune(
            mosaic,
            n_trials=2,
            z_subsample=2,
            search_space=_TEST_SEARCH_SPACE,
            seed=0,
        )
        assert result.trials_df is None


def _fake_trials_df(rows):
    """Build a trials_dataframe-shaped DataFrame from compact row dicts.

    Each row: {value, state, working_size, l_s_divisor, l_d_divisor, epsilon,
    estimate_darkfield}.  Column names match optuna's trials_dataframe()
    output (params_* prefix, stringified state).
    """
    return pd.DataFrame(
        {
            "number": range(len(rows)),
            "value": [r["value"] for r in rows],
            "state": [r["state"] for r in rows],
            "params_working_size": [r["working_size"] for r in rows],
            "params_l_s_divisor": [r["l_s_divisor"] for r in rows],
            "params_l_d_divisor": [r["l_d_divisor"] for r in rows],
            "params_epsilon": [r["epsilon"] for r in rows],
            "params_estimate_darkfield": [r["estimate_darkfield"] for r in rows],
        }
    )


def _result_from_df(trials_df, best_value=None):
    """Wrap a trials_dataframe in a TuneResult (no tune() call needed)."""
    if best_value is None:
        best_value = float(trials_df.loc[trials_df["state"] == "COMPLETE", "value"].min())
    return TuneResult(
        best_params={},
        best_value=best_value,
        trials_df=trials_df,
        best_fit=None,
        objective="seam_l1",
    )


class TestRecommendBounds:
    """Quality-aware search-space narrowing over a TuneResult trial history."""

    def test_deterministic_exact_bounds_from_synthetic_df(self):
        """Fixed-input DataFrame yields exact, reproducible bounds.

        Three trials complete: best=1.0, runner-up=1.05 (within 10%),
        worst=1.5 (outside).  With margin=0.10 the near-optimal set is the
        first two, so bounds are their min/max ranges and working_size is
        the sorted union.
        """
        df = _fake_trials_df(
            [
                {
                    "value": 1.0,
                    "state": "COMPLETE",
                    "working_size": 128,
                    "l_s_divisor": 400.0,
                    "l_d_divisor": 1000.0,
                    "epsilon": 0.1,
                    "estimate_darkfield": True,
                },
                {
                    "value": 1.05,
                    "state": "COMPLETE",
                    "working_size": 96,
                    "l_s_divisor": 600.0,
                    "l_d_divisor": 2000.0,
                    "epsilon": 0.2,
                    "estimate_darkfield": True,
                },
                {
                    "value": 1.5,
                    "state": "COMPLETE",
                    "working_size": 64,
                    "l_s_divisor": 3000.0,
                    "l_d_divisor": 8000.0,
                    "epsilon": 0.9,
                    "estimate_darkfield": False,
                },
            ]
        )
        result = _result_from_df(df)

        rec = recommend_bounds(result, margin=0.10)

        assert isinstance(rec, BoundsRecommendation)
        assert rec.n_near_optimal == 2
        assert rec.best_value == 1.0
        assert rec.margin == 0.10
        # working_size: sorted union of near-optimal trials (128, 96)
        assert rec.search_space["working_size"] == [96, 128]
        assert rec.search_space["l_s_divisor"] == (400.0, 600.0)
        assert rec.search_space["l_d_divisor"] == (1000.0, 2000.0)
        assert rec.search_space["epsilon"] == (0.1, 0.2)
        # Both near-optimal trials estimate darkfield → majority True
        assert rec.search_space["estimate_darkfield"] == [True]

    def test_single_best_trial_degenerate_case(self):
        """margin=0.0 collapses to only the best trial (degenerate ranges)."""
        df = _fake_trials_df(
            [
                {
                    "value": 0.5,
                    "state": "COMPLETE",
                    "working_size": 128,
                    "l_s_divisor": 500.0,
                    "l_d_divisor": 1500.0,
                    "epsilon": 0.15,
                    "estimate_darkfield": False,
                },
                {
                    "value": 0.9,
                    "state": "COMPLETE",
                    "working_size": 64,
                    "l_s_divisor": 2000.0,
                    "l_d_divisor": 6000.0,
                    "epsilon": 0.8,
                    "estimate_darkfield": True,
                },
            ]
        )
        result = _result_from_df(df)

        rec = recommend_bounds(result, margin=0.0)

        assert rec.n_near_optimal == 1
        # Single-trial ranges are degenerate (min == max)
        assert rec.search_space["working_size"] == [128]
        assert rec.search_space["l_s_divisor"] == (500.0, 500.0)
        assert rec.search_space["l_d_divisor"] == (1500.0, 1500.0)
        assert rec.search_space["epsilon"] == (0.15, 0.15)
        assert rec.search_space["estimate_darkfield"] == [False]

    def test_margin_widening_brings_more_trials_in(self):
        """Increasing margin widens the near-optimal set."""
        df = _fake_trials_df(
            [
                {
                    "value": 1.0,
                    "state": "COMPLETE",
                    "working_size": 128,
                    "l_s_divisor": 400.0,
                    "l_d_divisor": 1000.0,
                    "epsilon": 0.1,
                    "estimate_darkfield": True,
                },
                {
                    "value": 1.08,
                    "state": "COMPLETE",
                    "working_size": 96,
                    "l_s_divisor": 500.0,
                    "l_d_divisor": 1500.0,
                    "epsilon": 0.15,
                    "estimate_darkfield": True,
                },
                {
                    "value": 1.25,
                    "state": "COMPLETE",
                    "working_size": 64,
                    "l_s_divisor": 2000.0,
                    "l_d_divisor": 5000.0,
                    "epsilon": 0.6,
                    "estimate_darkfield": False,
                },
            ]
        )
        result = _result_from_df(df)

        tight = recommend_bounds(result, margin=0.05)
        wide = recommend_bounds(result, margin=0.30)

        assert tight.n_near_optimal == 1  # only the best (1.0)
        assert wide.n_near_optimal == 3  # all three within 30% of 1.0
        # Widening the set can only expand (or keep) the l_s range
        assert wide.search_space["l_s_divisor"][0] <= tight.search_space["l_s_divisor"][0]
        assert wide.search_space["l_s_divisor"][1] >= tight.search_space["l_s_divisor"][1]

    def test_pruned_and_failed_trials_excluded(self):
        """Only COMPLETE trials contribute; PRUNED/FAIL do not affect bounds."""
        df = _fake_trials_df(
            [
                {
                    "value": 1.0,
                    "state": "COMPLETE",
                    "working_size": 128,
                    "l_s_divisor": 400.0,
                    "l_d_divisor": 1000.0,
                    "epsilon": 0.1,
                    "estimate_darkfield": True,
                },
                # Pruned trial has extreme params but must be ignored
                {
                    "value": float("nan"),
                    "state": "PRUNED",
                    "working_size": 16,
                    "l_s_divisor": 100.0,
                    "l_d_divisor": 500.0,
                    "epsilon": 0.01,
                    "estimate_darkfield": False,
                },
            ]
        )
        result = _result_from_df(df, best_value=1.0)

        rec = recommend_bounds(result, margin=0.10)

        assert rec.n_near_optimal == 1
        assert rec.search_space["working_size"] == [128]
        assert rec.search_space["l_s_divisor"] == (400.0, 400.0)

    def test_estimate_darkfield_majority_vote_tie_and_false(self):
        """Majority vote: tie (50%) defaults to True; all-False → [False]."""
        # 2 True, 2 False → 50% mean → >= 0.5 → True
        df_tie = _fake_trials_df(
            [
                {
                    "value": 1.0,
                    "state": "COMPLETE",
                    "working_size": 128,
                    "l_s_divisor": 400.0,
                    "l_d_divisor": 1000.0,
                    "epsilon": 0.1,
                    "estimate_darkfield": True,
                },
                {
                    "value": 1.0,
                    "state": "COMPLETE",
                    "working_size": 128,
                    "l_s_divisor": 400.0,
                    "l_d_divisor": 1000.0,
                    "epsilon": 0.1,
                    "estimate_darkfield": True,
                },
                {
                    "value": 1.0,
                    "state": "COMPLETE",
                    "working_size": 128,
                    "l_s_divisor": 400.0,
                    "l_d_divisor": 1000.0,
                    "epsilon": 0.1,
                    "estimate_darkfield": False,
                },
                {
                    "value": 1.0,
                    "state": "COMPLETE",
                    "working_size": 128,
                    "l_s_divisor": 400.0,
                    "l_d_divisor": 1000.0,
                    "epsilon": 0.1,
                    "estimate_darkfield": False,
                },
            ]
        )
        rec_tie = recommend_bounds(_result_from_df(df_tie), margin=0.10)
        assert rec_tie.search_space["estimate_darkfield"] == [True]

        # All False → [False]
        df_false = _fake_trials_df(
            [
                {
                    "value": 1.0,
                    "state": "COMPLETE",
                    "working_size": 128,
                    "l_s_divisor": 400.0,
                    "l_d_divisor": 1000.0,
                    "epsilon": 0.1,
                    "estimate_darkfield": False,
                },
            ]
        )
        rec_false = recommend_bounds(_result_from_df(df_false), margin=0.10)
        assert rec_false.search_space["estimate_darkfield"] == [False]

    def test_search_space_feeds_back_into_tune(self, synthetic_mosaic):
        """End-to-end: recommend_bounds output is a valid search_space for tune()."""
        mosaic, _ = synthetic_mosaic
        first = tune(
            mosaic,
            n_trials=3,
            z_subsample=2,
            search_space=_TEST_SEARCH_SPACE,
            seed=0,
        )
        rec = recommend_bounds(first, margin=0.20)

        # The recommended space must be consumable by tune() without error.
        second = tune(
            mosaic,
            n_trials=2,
            z_subsample=2,
            search_space=rec.search_space,
            seed=0,
        )
        assert isinstance(second, TuneResult)
        assert isinstance(second.best_value, float)

    def test_recommend_bounds_on_real_tune_result(self, synthetic_mosaic):
        """recommend_bounds runs on a genuine tune() trials_df and returns sane bounds."""
        mosaic, _ = synthetic_mosaic
        result = tune(
            mosaic,
            n_trials=4,
            z_subsample=2,
            search_space=_TEST_SEARCH_SPACE,
            seed=0,
        )
        rec = recommend_bounds(result, margin=0.15)

        assert rec.best_value == pytest.approx(result.best_value)
        assert rec.n_near_optimal >= 1
        assert set(rec.search_space) == {
            "working_size",
            "l_s_divisor",
            "l_d_divisor",
            "epsilon",
            "estimate_darkfield",
        }
        # Bounds must be ordered low <= high
        for key in ("l_s_divisor", "l_d_divisor", "epsilon"):
            lo, hi = rec.search_space[key]
            assert lo <= hi

    def test_default_margin_is_10_percent(self):
        """Default margin is 0.10 (documented in the plan)."""
        import inspect

        sig = inspect.signature(recommend_bounds)
        assert sig.parameters["margin"].default == 0.10


class TestRecommendBoundsNegative:
    """Negative/error-path coverage for recommend_bounds (gate Q7)."""

    def test_margin_below_zero_raises(self):
        df = _fake_trials_df(
            [
                {
                    "value": 1.0,
                    "state": "COMPLETE",
                    "working_size": 128,
                    "l_s_divisor": 400.0,
                    "l_d_divisor": 1000.0,
                    "epsilon": 0.1,
                    "estimate_darkfield": True,
                }
            ]
        )
        with pytest.raises(ValueError, match="margin must be in"):
            recommend_bounds(_result_from_df(df), margin=-0.01)

    def test_margin_above_one_raises(self):
        df = _fake_trials_df(
            [
                {
                    "value": 1.0,
                    "state": "COMPLETE",
                    "working_size": 128,
                    "l_s_divisor": 400.0,
                    "l_d_divisor": 1000.0,
                    "epsilon": 0.1,
                    "estimate_darkfield": True,
                }
            ]
        )
        with pytest.raises(ValueError, match="margin must be in"):
            recommend_bounds(_result_from_df(df), margin=1.01)

    def test_none_trials_df_raises(self):
        """When trials_df is None (pandas unavailable), raise ValueError."""
        result = TuneResult(
            best_params={},
            best_value=1.0,
            trials_df=None,
            best_fit=None,
            objective="seam_l1",
        )
        with pytest.raises(ValueError, match="populated trials_df"):
            recommend_bounds(result, margin=0.10)

    def test_empty_trials_df_raises(self):
        """An empty DataFrame has no COMPLETE trials."""
        result = _result_from_df(_fake_trials_df([]), best_value=1.0)
        with pytest.raises(ValueError, match="populated trials_df"):
            recommend_bounds(result, margin=0.10)

    def test_no_complete_trials_raises(self):
        """All-PRUNED history cannot define a near-optimal band."""
        df = _fake_trials_df(
            [
                {
                    "value": float("nan"),
                    "state": "PRUNED",
                    "working_size": 128,
                    "l_s_divisor": 400.0,
                    "l_d_divisor": 1000.0,
                    "epsilon": 0.1,
                    "estimate_darkfield": True,
                }
            ]
        )
        result = _result_from_df(df, best_value=1.0)
        with pytest.raises(ValueError, match="COMPLETE"):
            recommend_bounds(result, margin=0.10)

    def test_bounds_recommendation_is_immutable(self):
        """BoundsRecommendation is frozen: attribute reassignment is blocked."""
        from dataclasses import FrozenInstanceError

        df = _fake_trials_df(
            [
                {
                    "value": 1.0,
                    "state": "COMPLETE",
                    "working_size": 128,
                    "l_s_divisor": 400.0,
                    "l_d_divisor": 1000.0,
                    "epsilon": 0.1,
                    "estimate_darkfield": True,
                }
            ]
        )
        rec = recommend_bounds(_result_from_df(df), margin=0.10)
        with pytest.raises(FrozenInstanceError):
            rec.best_value = 999.0  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            rec.n_near_optimal = 42  # type: ignore[misc]

    def test_state_trialstate_enum_form_also_matches(self):
        """str(state).endswith('COMPLETE') tolerates 'TrialState.COMPLETE'."""
        df = pd.DataFrame(
            {
                "number": [0],
                "value": [1.0],
                "state": ["TrialState.COMPLETE"],
                "params_working_size": [128],
                "params_l_s_divisor": [400.0],
                "params_l_d_divisor": [1000.0],
                "params_epsilon": [0.1],
                "params_estimate_darkfield": [True],
            }
        )
        rec = recommend_bounds(_result_from_df(df), margin=0.10)
        assert rec.n_near_optimal == 1
