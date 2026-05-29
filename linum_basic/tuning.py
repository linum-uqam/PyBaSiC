"""Optuna-based hyperparameter tuning for BaSiC on mosaic grids.

Uses the seam-consistency L1 metric (see :mod:`linum_basic.metrics`) as
the objective — lower is better.  Intermediate values are reported after
each z-level subsample step so Optuna's :class:`~optuna.pruners.MedianPruner`
can kill unpromising trials early.

The tunable parameters are:

``working_size``
    Internal resolution for BaSiC processing.  Larger values are slower
    but may capture finer vignette detail.

``l_s_divisor``, ``l_d_divisor``
    Regularisation strengths are set as ``l_s = dct_sum / l_s_divisor``
    and ``l_d = dct_sum / l_d_divisor``, where ``dct_sum`` is the DCT
    energy of the normalised mean image at the first reference z-level.
    This parametrisation is scale-invariant across datasets.

``epsilon``
    Reweighting stability constant.

``estimate_darkfield``
    Whether to estimate a dark-field in addition to the flat-field.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from linum_basic.fit import MosaicFit, _make_model
from linum_basic.metrics import evaluate_correction
from linum_basic.mosaic import MosaicGrid

__all__ = ["TuneResult", "tune"]

# Default Optuna search space.
# Categorical values → suggest_categorical; length-2 tuples → log-uniform float.
_DEFAULT_SEARCH_SPACE: dict[str, list | tuple] = {
    "working_size": [64, 96, 128, 160, 192],
    "l_s_divisor": (100.0, 5000.0),
    "l_d_divisor": (500.0, 10000.0),
    "epsilon": (0.01, 1.0),
    "estimate_darkfield": [True, False],
}


@dataclass(init=False)
class TuneResult:
    """Result of a hyperparameter tuning run.

    Attributes
    ----------
    best_params : dict
        BaSiC hyperparameters that minimised the objective.
    best_value : float
        Mean seam-L1 at *best_params*.
    trials_df : pandas.DataFrame or None
        Full trial history (``None`` if pandas is not installed).
    best_fit : MosaicFit or None
        Full-z fit with *best_params*.  Set only when
        ``run_full_fit=True`` is passed to :func:`tune`.
    """

    best_params: dict[str, Any]
    best_value: float
    trials_df: Any  # pandas.DataFrame when available, else None
    best_fit: MosaicFit | None

    def __init__(
        self,
        best_params: dict[str, Any],
        best_value: float,
        trials_df: Any,
        best_fit: MosaicFit | None = None,
    ) -> None:
        """Construct a :class:`TuneResult`.

        Parameters
        ----------
        best_params : dict
            BaSiC hyperparameters that minimised the objective.
        best_value : float
            Mean seam-L1 at *best_params*.
        trials_df : pandas.DataFrame or None
            Full trial history (``None`` if pandas is not installed).
        best_fit : MosaicFit or None
            Full-z fit with *best_params*.  Set only when
            ``run_full_fit=True`` is passed to :func:`tune`.
        """
        self.best_params = best_params
        self.best_value = best_value
        self.trials_df = trials_df
        self.best_fit = best_fit


def tune(
    mosaic: MosaicGrid,
    *,
    n_trials: int = 50,
    z_subsample: int = 4,
    search_space: dict[str, list | tuple] | None = None,
    seed: int = 0,
    n_jobs: int = 1,
    storage: str | None = None,
    study_name: str = "basic-tune",
    run_full_fit: bool = False,
    verbose: bool = False,
) -> TuneResult:
    """Tune BaSiC hyperparameters using the seam-consistency L1 metric.

    Parameters
    ----------
    mosaic : MosaicGrid
        The mosaic grid to optimise on.
    n_trials : int
        Number of Optuna trials.
    z_subsample : int
        Number of z-levels sampled per trial (evenly spaced).  Increase
        for more reliable estimates; decrease for faster tuning.
    search_space : dict or None
        Override the default search space.  Categorical parameters take a
        list of choices; continuous parameters take a ``(low, high)``
        tuple (sampled log-uniformly).
    seed : int
        Random seed for reproducibility.
    n_jobs : int
        Parallel Optuna workers.  In-memory storage only supports 1 worker;
        a warning is issued if ``n_jobs > 1`` and ``storage`` is ``None``.
    storage : str or None
        Optuna storage URL (e.g. ``"sqlite:///tune.db"``).  ``None`` uses
        in-memory storage (not resumable).
    study_name : str
        Name of the Optuna study.  Used for persistent-storage resumability.
    run_full_fit : bool
        After tuning, run a full-z fit with the best parameters and attach
        the result to :attr:`TuneResult.best_fit`.
    verbose : bool
        Enable Optuna logging and tqdm progress bars.

    Returns
    -------
    TuneResult
        Best parameters, best seam-L1 value, and (optionally) the full fit.
    """
    try:
        import optuna
    except ImportError as exc:
        msg = "optuna is required for tuning. Install with: pip install optuna"
        raise ImportError(msg) from exc

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if n_jobs > 1 and storage is None:
        warnings.warn(
            "n_jobs > 1 requires a persistent storage URL; falling back to n_jobs=1.",
            stacklevel=2,
        )
        n_jobs = 1

    sp = {**_DEFAULT_SEARCH_SPACE, **(search_space or {})}

    # Evenly spaced z-level subsample
    n_z = mosaic.n_z
    z_step = max(1, n_z // z_subsample)
    z_indices = list(range(0, n_z, z_step))[:z_subsample]
    seam_pairs = mosaic.seam_pairs()

    # Compute DCT energy reference on the first z-level.
    # l_s and l_d are parametrised as dct_sum / divisor, matching
    # BaSiC's own auto-tuning convention.
    from scipy.fft import dctn

    ref_tiles = mosaic.iter_tiles(z_indices[0])
    mean_img = ref_tiles.mean(axis=0)
    denom = float(mean_img.mean()) + 1e-9
    dct_sum = float(np.abs(dctn(mean_img / denom, norm="ortho")).sum())

    def _objective(trial: optuna.Trial) -> float:
        # --- suggest hyperparameters ---
        working_size = trial.suggest_categorical("working_size", sp["working_size"])
        l_s_div = trial.suggest_float("l_s_divisor", *sp["l_s_divisor"], log=True)
        l_d_div = trial.suggest_float("l_d_divisor", *sp["l_d_divisor"], log=True)
        epsilon = trial.suggest_float("epsilon", *sp["epsilon"], log=True)
        estimate_darkfield = trial.suggest_categorical("estimate_darkfield", sp["estimate_darkfield"])

        params: dict[str, Any] = {
            "working_size": working_size,
            "l_s": dct_sum / l_s_div,
            "l_d": dct_sum / l_d_div,
            "epsilon": epsilon,
            "estimate_darkfield": estimate_darkfield,
        }

        total_l1 = 0.0
        for step, z in enumerate(z_indices):
            tiles = mosaic.iter_tiles(z)
            model = _make_model(tiles, params)
            model.prepare()
            model.run()

            ff = model.get_flatfield()
            df = model.get_darkfield()
            metrics = evaluate_correction(tiles, ff, df, seam_pairs)
            total_l1 += metrics["seam_l1"]

            # Report intermediate value for MedianPruner
            trial.report(total_l1 / (step + 1), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return total_l1 / len(z_indices)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    study.optimize(_objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=verbose)

    best_trial = study.best_trial
    # Convert divisor params back to actual l_s / l_d
    raw = dict(best_trial.params)
    best_params: dict[str, Any] = {
        "working_size": raw["working_size"],
        "l_s": dct_sum / raw["l_s_divisor"],
        "l_d": dct_sum / raw["l_d_divisor"],
        "epsilon": raw["epsilon"],
        "estimate_darkfield": raw["estimate_darkfield"],
    }

    # Build trials DataFrame if pandas is available
    try:
        trials_df = study.trials_dataframe()
    except Exception:
        trials_df = None

    best_fit: MosaicFit | None = None
    if run_full_fit:
        from linum_basic.fit import fit_mosaic

        best_fit = fit_mosaic(mosaic, basic_kwargs=best_params, verbose=verbose)

    return TuneResult(
        best_params=best_params,
        best_value=float(best_trial.value or 0.0),
        trials_df=trials_df,
        best_fit=best_fit,
    )
