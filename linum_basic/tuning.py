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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from linum_basic._parallel import default_workers
from linum_basic._working_size import (
    PREVIEW_Z_SAMPLE,
    WorkingSizeResolution,
    build_working_size_context,
    resolve_working_size,
)
from linum_basic.core import dct_energy
from linum_basic.curvature import seam_curvature
from linum_basic.fit import MosaicFit, make_model
from linum_basic.metrics import seam_l1
from linum_basic.mosaic import MosaicGrid

__all__ = ["BoundsRecommendation", "TuneResult", "recommend_bounds", "tune"]

# Default Optuna search space.
# Categorical values → suggest_categorical; length-2 tuples → log-uniform float.
_DEFAULT_SEARCH_SPACE: dict[str, list | tuple] = {
    "working_size": [64, 96, 128, 160, 192],
    "l_s_divisor": (100.0, 5000.0),
    "l_d_divisor": (500.0, 10000.0),
    "epsilon": (0.01, 1.0),
    "estimate_darkfield": [True, False],
}


def _resolve_working_size_auto(
    mosaic: MosaicGrid,
    *,
    n_z: int,
    device: str | None,
) -> WorkingSizeResolution | None:
    """Resolve the ``working_size="auto"`` sentinel for :func:`tune`.

    Activates *only* when the caller asked for ``"auto"`` (opt-in). Builds a
    :class:`WorkingSizeContext` from mosaic metadata, a bounded preview mean,
    and a best-effort memory budget, then calls the pure resolver. Mirrors
    :func:`linum_basic.fit._resolve_working_size_auto` but adapts to tune's
    per-z evaluation context (``field_mode="per-z"``).

    Returns the :class:`WorkingSizeResolution` (never ``None`` here since the
    caller only invokes this when ``working_size == "auto"``). Always fails
    safe to :data:`~linum_basic._working_size.SAFE_DEFAULT` (128).
    """
    sample_z = list(range(n_z))[:PREVIEW_Z_SAMPLE]
    mean_image: np.ndarray | None = None
    if sample_z:
        try:
            stacks = [mosaic.iter_tiles(z) for z in sample_z]
            stacked = np.stack(stacks, axis=0)  # (n_sample_z, n_tiles, th, tw)
            mean_image = stacked.mean(axis=(0, 1))
        except ValueError, IndexError, OSError:
            mean_image = None

    ctx = build_working_size_context(
        n_z=n_z,
        n_tiles=mosaic.n_tiles,
        field_mode="per-z",
        tile_shape=tuple(mosaic.tile_shape),
        mean_image=mean_image,
        device=device,
        requested="auto",
    )
    return resolve_working_size(ctx)


def _basic_params(raw: dict[str, Any], dct_sum: float) -> dict[str, Any]:
    """Convert raw search-space values into BaSiC hyperparameters.

    The tunable knobs are divisors (``l_s_divisor``/``l_d_divisor``) which are
    converted to absolute regularisation weights via ``dct_sum / divisor``.
    This is the single conversion path shared by the objective and the
    best-trial extraction so the two can never drift apart.
    """
    return {
        "working_size": raw["working_size"],
        "l_s": dct_sum / raw["l_s_divisor"],
        "l_d": dct_sum / raw["l_d_divisor"],
        "epsilon": raw["epsilon"],
        "estimate_darkfield": raw["estimate_darkfield"],
    }


def _subsample_tiles(tiles: np.ndarray, max_tiles: int | None) -> np.ndarray:
    """Return an evenly spaced subset of *tiles* of size <= *max_tiles*.

    Subsampling the tile stack used for seam-metric evaluation speeds up
    tuning dramatically with negligible loss of accuracy (empirically ~64
    tiles is the sweet spot).  ``None`` or a count >= the number of tiles
    returns the input unchanged.
    """
    n = tiles.shape[0]
    if max_tiles is None or max_tiles >= n:
        return tiles
    idx = np.unique(np.linspace(0, n - 1, max_tiles).round().astype(int))
    return tiles[idx]


@dataclass(init=False)
class TuneResult:
    """Result of a hyperparameter tuning run.

    Attributes
    ----------
    best_params : dict
        BaSiC hyperparameters that minimised the objective.
    best_value : float
        Value of the chosen objective (see :attr:`objective`) at
        *best_params*.
    objective : str
        Name of the minimised objective (``"seam_l1"``,
        ``"curvature"``, or ``"composite"``).
    trials_df : pandas.DataFrame or None
        Full trial history (``None`` if pandas is not installed).
    best_fit : MosaicFit or None
        Full-z fit with *best_params*.  Set only when
        ``run_full_fit=True`` is passed to :func:`tune`.
    working_size_selector : dict or None
        Explainability metadata for the ``working_size="auto"`` resolution
        (M006/S02).  ``None`` unless the caller opted into ``"auto"``.  When
        present, carries ``resolved_working_size``, ``requested``,
        ``candidate_grid``, ``rule_path``, ``fallback_reason``,
        ``gate_status``, ``signals``, and ``peak_memory_estimate_bytes``
        per candidate, mirroring the ``params["_working_size_selector"]``
        pattern on :class:`MosaicFit`.  Purely additive metadata that never
        affects tune numerics.
    """

    best_params: dict[str, Any]
    best_value: float
    objective: str
    trials_df: Any  # pandas.DataFrame when available, else None
    best_fit: MosaicFit | None
    working_size_selector: dict[str, Any] | None

    def __init__(
        self,
        best_params: dict[str, Any],
        best_value: float,
        trials_df: Any,
        best_fit: MosaicFit | None = None,
        objective: str = "seam_l1",
        working_size_selector: dict[str, Any] | None = None,
    ) -> None:
        """Construct a :class:`TuneResult`.

        Parameters
        ----------
        best_params : dict
            BaSiC hyperparameters that minimised the objective.
        best_value : float
            Value of the chosen objective at *best_params*.
        trials_df : pandas.DataFrame or None
            Full trial history (``None`` if pandas is not installed).
        best_fit : MosaicFit or None
            Full-z fit with *best_params*.  Set only when
            ``run_full_fit=True`` is passed to :func:`tune`.
        objective : str
            Name of the minimised objective (e.g. ``"seam_l1"``,
            ``"curvature"``, ``"composite"``).
        working_size_selector : dict or None
            Explainability metadata for the ``working_size="auto"``
            resolution.  ``None`` unless ``"auto"`` was requested.
        """
        self.best_params = best_params
        self.best_value = best_value
        self.objective = objective
        self.trials_df = trials_df
        self.best_fit = best_fit
        self.working_size_selector = working_size_selector


@dataclass(frozen=True, slots=True, init=False)
class BoundsRecommendation:
    """Quality-aware search-space narrowing derived from a :class:`TuneResult`.

    Summarises the near-optimal region of an Optuna trial history into a
    ``search_space``-shaped dict that can be fed straight back into a
    follow-up :func:`tune` call (or serialised to JSON for the CLI).  Bounds
    are expressed in the scale-invariant *divisor* parametrisation
    (``l_s_divisor`` / ``l_d_divisor``), matching
    :data:`_DEFAULT_SEARCH_SPACE` so they transfer across datasets without
    the per-subject ``dct_sum``.

    Attributes
    ----------
    search_space : dict
        Narrowed search space ready for :func:`tune`::

            {
                "working_size": [96, 128],
                "l_s_divisor": (300.0, 1200.0),
                "l_d_divisor": (800.0, 4000.0),
                "epsilon": (0.05, 0.4),
                "estimate_darkfield": [True],
            }

        Degenerate ranges (``low == high``) are valid and arise when the
        near-optimal subset contains a single trial or a constant value.
    best_value : float
        Objective value of the best completed trial.
    n_near_optimal : int
        Number of completed trials within *margin* of ``best_value``.
    margin : float
        Relative margin used to select the near-optimal subset.
    """

    search_space: dict[str, Any]
    best_value: float
    n_near_optimal: int
    margin: float

    def __init__(
        self,
        search_space: dict[str, Any],
        best_value: float,
        n_near_optimal: int,
        margin: float,
    ) -> None:
        """Construct a :class:`BoundsRecommendation`.

        Parameters
        ----------
        search_space : dict
            Narrowed search space (see class docstring for shape).
        best_value : float
            Objective value of the best completed trial.
        n_near_optimal : int
            Number of completed trials within *margin* of ``best_value``.
        margin : float
            Relative margin used to select the near-optimal subset.
        """
        object.__setattr__(self, "search_space", dict(search_space))
        object.__setattr__(self, "best_value", float(best_value))
        object.__setattr__(self, "n_near_optimal", int(n_near_optimal))
        object.__setattr__(self, "margin", float(margin))


def recommend_bounds(
    result: TuneResult,
    *,
    margin: float = 0.10,
) -> BoundsRecommendation:
    """Propose narrowed l_s / l_d / working_size bounds from a tuning run.

    Selects the subset of completed trials whose objective value is within
    *margin* (a relative fraction) of the best observed value, then collapses
    that subset into a ``search_space``-shaped recommendation:

    - ``working_size`` → sorted unique choices observed in the subset.
    - ``l_s_divisor``, ``l_d_divisor``, ``epsilon`` → ``(min, max)`` range
      over the subset.  The regularisation strengths stay in the
      scale-invariant divisor parametrisation so the bounds transfer across
      datasets.
    - ``estimate_darkfield`` → majority vote, returned as a single-element
      list.

    The returned :attr:`BoundsRecommendation.search_space` plugs directly
    into a follow-up :func:`tune` call, focusing the next search on the
    empirically productive region.

    Parameters
    ----------
    result : TuneResult
        Output of :func:`tune`.  Must carry a populated ``trials_df`` with at
        least one ``COMPLETE`` trial.
    margin : float
        Relative width of the near-optimal band.  ``0.0`` keeps only the
        best trial(s); ``0.10`` (default) keeps trials within 10% of the
        best value.  Must be in ``[0, 1]``.

    Returns
    -------
    BoundsRecommendation
        Narrowed search space plus the best value, near-optimal count, and
        margin used.

    Raises
    ------
    ValueError
        If *margin* is outside ``[0, 1]``, *result* has no ``trials_df``
        (e.g. pandas was unavailable), or no trial reached the ``COMPLETE``
        state.

    Notes
    -----
    This is a distinct, advisory heuristic from the release-gate calibration
    in :mod:`linum_basic.benchmark.quality` (which applies a ``mean+3std``
    policy to repeated A/B baselines).  ``recommend_bounds`` instead collapses
    a single Optuna study's trial history into a narrowed search space and
    never gates a production fit.

    Examples
    --------
    >>> rec = recommend_bounds(result, margin=0.10)  # doctest: +SKIP
    >>> rec.search_space["working_size"]  # doctest: +SKIP
    [128]
    >>> rec.n_near_optimal >= 1  # doctest: +SKIP
    True
    """
    if not 0.0 <= margin <= 1.0:
        msg = f"margin must be in [0, 1], got {margin!r}"
        raise ValueError(msg)

    trials_df = result.trials_df
    if trials_df is None or len(trials_df) == 0:
        msg = "recommend_bounds requires a TuneResult with a populated trials_df"
        raise ValueError(msg)

    # Only COMPLETE trials carry a final, comparable objective value.
    # trials_dataframe() stringifies TrialState (e.g. "COMPLETE");
    # str.endswith also tolerates a future "TrialState.COMPLETE" form.
    state = trials_df["state"].astype(str)
    complete = trials_df[state.str.endswith("COMPLETE")]
    if len(complete) == 0:
        msg = "recommend_bounds requires at least one COMPLETE trial"
        raise ValueError(msg)

    values = complete["value"].to_numpy(dtype=float)
    best_value = float(np.nanmin(values))

    # Near-optimal band: values within `margin` of the best (relative).
    threshold = best_value * (1.0 + margin)
    near = complete[values <= threshold]

    working_sizes = sorted({int(v) for v in near["params_working_size"].to_numpy()})

    def _range(col: str) -> tuple[float, float]:
        arr = near[col].to_numpy(dtype=float)
        return (float(np.nanmin(arr)), float(np.nanmax(arr)))

    # Majority vote on the boolean darkfield flag across the near-optimal set.
    df_flags = near["params_estimate_darkfield"].to_numpy().astype(bool)
    estimate_darkfield = bool(np.mean(df_flags) >= 0.5)

    search_space: dict[str, Any] = {
        "working_size": working_sizes,
        "l_s_divisor": _range("params_l_s_divisor"),
        "l_d_divisor": _range("params_l_d_divisor"),
        "epsilon": _range("params_epsilon"),
        "estimate_darkfield": [estimate_darkfield],
    }

    return BoundsRecommendation(
        search_space=search_space,
        best_value=best_value,
        n_near_optimal=len(near),
        margin=margin,
    )


def tune(
    mosaic: MosaicGrid,
    *,
    n_trials: int = 50,
    z_subsample: int = 4,
    search_space: dict[str, list | tuple] | None = None,
    seed: int = 0,
    n_workers: int | None = None,
    backend: str = "numpy",
    device: str | None = None,
    storage: str | None = None,
    study_name: str = "basic-tune",
    run_full_fit: bool = False,
    max_tiles: int | None = 64,
    n_extra_rows: int = 0,
    objective: Literal["seam_l1", "curvature", "composite"] = "seam_l1",
    composite_weights: tuple[float, float] = (1.0, 1.0),
    working_size: int | str = 128,
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
    n_workers : int
        Number of threads used to evaluate z-levels in parallel within each
        trial. ``None`` (default) uses ``cpu_count() - 2``. Set to ``1`` for
        sequential evaluation, which enables Optuna trial pruning (skipping
        unpromising trials early); parallel evaluation trades pruning for
        within-trial z-level parallelism. Threads are used (not processes)
        because the per-trial tile caches are large and shared, and the ALM
        kernels (DCT, SVD) release the GIL.  When *backend* is ``"torch"``
        the worker count is forced to 1 to avoid GPU memory contention.
    backend : str
        Compute backend for BaSiC.  ``"numpy"`` (default) or ``"torch"``.
    device : str or None
        Torch device string (e.g. ``"cuda:0"``).  Ignored when *backend* is
        ``"numpy"``.
    storage : str or None
        Optuna storage URL (e.g. ``"sqlite:///tune.db"``).  ``None`` uses
        in-memory storage (not resumable).
    study_name : str
        Name of the Optuna study.  Used for persistent-storage resumability.
    run_full_fit : bool
        After tuning, run a full-z fit with the best parameters and attach
        the result to :attr:`TuneResult.best_fit`.
    max_tiles : int or None
        Maximum number of tiles (evenly spaced) used for the seam-metric
        objective during tuning.  Subsampling speeds up tuning with little
        accuracy loss (default 64).  ``None`` uses all tiles.  The final
        ``run_full_fit`` always uses every tile.
    n_extra_rows : int
        Number of leading rows per tile to drop before fitting (galvo
        fly-back artefact).  Propagated to the full fit when
        ``run_full_fit=True``.
    objective : {"seam_l1", "curvature", "composite"}
        Objective function to minimise.

        ``"seam_l1"`` (default)
            Mean per-seam relative absolute intensity difference in the
            corrected tile overlaps.  Fast and reliable; the recommended
            starting point.
        ``"curvature"``
            Gaussian focal-curvature consistency of the estimated flatfield
            in the overlap regions (see
            :func:`~linum_basic.curvature.seam_curvature`).  Measures
            whether the per-z flatfield follows the expected Gaussian
            optics model; useful when the illumination pattern is the
            primary unknown.
        ``"composite"``
            Weighted sum of the two metrics, normalised by the sum of
            weights: ``(w1 * seam_l1 + w2 * curvature) / (w1 + w2)``.
            Use *composite_weights* to set ``(w1, w2)``.
    composite_weights : tuple of float
        Weights ``(w_seam, w_curvature)`` for the ``"composite"``
        objective.  Default is ``(1.0, 1.0)`` (equal contribution).  Has
        no effect unless ``objective="composite"``.
    working_size : int or str
        Controls the ``working_size`` search-space dimension.  An explicit
        integer (default ``128``) leaves the dimension open for Optuna to
        explore the full grid (backward compatible with callers that never
        pass this parameter).  The sentinel ``"auto"`` (opt-in, M006/S02)
        resolves to a concrete grid integer before the search space is built
        and pins the dimension to that single value, recording the decision
        in :attr:`TuneResult.working_size_selector`; the literal ``"auto"``
        never reaches :data:`_DEFAULT_SEARCH_SPACE`.  Always fails safe
        to ``128``.
    verbose : bool
        Enable Optuna logging and tqdm progress bars.

    Returns
    -------
    TuneResult
        Best parameters, best objective value, and (optionally) the full fit.
    """
    try:
        import optuna
    except ImportError as exc:
        msg = "optuna is required for tuning. Install with: pip install optuna"
        raise ImportError(msg) from exc

    n_workers = default_workers() if n_workers is None else max(1, int(n_workers))

    # Multiple threads competing for the same GPU causes contention and OOM;
    # force sequential evaluation (which also enables pruning).
    if backend == "torch" and n_workers > 1:
        import warnings

        warnings.warn(
            "backend='torch': n_workers forced to 1 to avoid GPU memory contention.",
            UserWarning,
            stacklevel=2,
        )
        n_workers = 1

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Resolve the working_size="auto" sentinel (opt-in, M006/S02) before the
    # search space is constructed, so the literal "auto" never reaches
    # _DEFAULT_SEARCH_SPACE. The resolved value pins the working_size
    # dimension; an explicit integer (including the default 128) leaves the
    # search-space dimension open for exploration (backward compatible).
    working_size_selector: dict[str, Any] | None = None
    if working_size == "auto":
        ws_resolution = _resolve_working_size_auto(mosaic, n_z=mosaic.n_z, device=device)
        assert ws_resolution is not None  # the tune resolver always returns a value in the "auto" branch
        sp = {**_DEFAULT_SEARCH_SPACE, **(search_space or {}), "working_size": [ws_resolution.resolved_working_size]}
        working_size_selector = ws_resolution.metadata()
    else:
        sp = {**_DEFAULT_SEARCH_SPACE, **(search_space or {})}

    # Evenly spaced z-level subsample
    n_z = mosaic.n_z
    z_step = max(1, n_z // z_subsample)
    z_indices = list(range(0, n_z, z_step))[:z_subsample]
    seam_pairs = mosaic.seam_pairs()

    # Compute DCT energy reference on the first z-level.
    # l_s and l_d are parametrised as dct_sum / divisor, matching
    # BaSiC's own auto-tuning convention.
    ref_tiles = mosaic.iter_tiles(z_indices[0])
    dct_sum = dct_energy(ref_tiles.mean(axis=0))

    # Cache the tile stacks per z-level: tiles are identical across trials,
    # so we extract them only once.  ``full_tiles`` is used for the seam
    # metric (adjacency requires every tile); ``fit_tiles`` is an evenly
    # spaced subset used only to estimate the flat/dark fields (subsampling
    # there is the main tuning speed-up with negligible accuracy loss).
    full_tiles_cache: dict[int, np.ndarray] = {z: mosaic.iter_tiles(z) for z in z_indices}
    fit_tiles_cache: dict[int, np.ndarray] = {z: _subsample_tiles(t, max_tiles) for z, t in full_tiles_cache.items()}

    def _objective(trial: optuna.Trial) -> float:
        # --- suggest hyperparameters ---
        raw = {
            "working_size": trial.suggest_categorical("working_size", sp["working_size"]),
            "l_s_divisor": trial.suggest_float("l_s_divisor", *sp["l_s_divisor"], log=True),
            "l_d_divisor": trial.suggest_float("l_d_divisor", *sp["l_d_divisor"], log=True),
            "epsilon": trial.suggest_float("epsilon", *sp["epsilon"], log=True),
            "estimate_darkfield": trial.suggest_categorical("estimate_darkfield", sp["estimate_darkfield"]),
        }
        params = _basic_params(raw, dct_sum)

        def _eval_z(z: int) -> float:
            full_tiles = full_tiles_cache[z]
            fit_tiles = fit_tiles_cache[z]
            if n_extra_rows > 0:
                fit_tiles = fit_tiles[:, n_extra_rows:, :]
            extra_kw: dict[str, str] = {"backend": backend}
            if device is not None:
                extra_kw["device"] = device
            model = make_model(fit_tiles, {**params, **extra_kw})
            model.prepare()
            model.run()
            ff = model.get_flatfield()
            df = model.get_darkfield()
            if n_extra_rows > 0:
                # Fields were estimated on the cropped tiles; edge-extend the
                # leading rows so they line up with the full-height tiles used
                # for the seam metric (mirrors apply_fit).
                ff = np.concatenate([np.repeat(ff[:1], n_extra_rows, axis=0), ff], axis=0)
                df = np.concatenate([np.repeat(df[:1], n_extra_rows, axis=0), df], axis=0)
            corrected = (full_tiles.astype(np.float32) - df[np.newaxis]) / (ff[np.newaxis] + 1e-6)
            if objective == "curvature":
                val = seam_curvature(ff[np.newaxis], seam_pairs)
                return val if np.isfinite(val) else 1e6
            if objective == "composite":
                w1, w2 = composite_weights
                denom = (w1 + w2) or 1.0
                val = (w1 * seam_l1(corrected, seam_pairs) + w2 * seam_curvature(ff[np.newaxis], seam_pairs)) / denom
                return val if np.isfinite(val) else 1e6
            val = seam_l1(corrected, seam_pairs)
            return val if np.isfinite(val) else 1e6

        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                l1_values = list(pool.map(_eval_z, z_indices))
            return sum(l1_values) / len(l1_values)

        # Sequential mode: use pruning to skip unpromising trials early.
        total_l1 = 0.0
        for step, z in enumerate(z_indices):
            total_l1 += _eval_z(z)
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

    study.optimize(_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=verbose)

    best_trial = study.best_trial
    # Convert divisor params back to actual l_s / l_d via the shared path.
    best_params = _basic_params(dict(best_trial.params), dct_sum)

    # Build trials DataFrame if pandas is available
    try:
        trials_df = study.trials_dataframe()
    except Exception:
        trials_df = None

    best_fit: MosaicFit | None = None
    if run_full_fit:
        from linum_basic.fit import fit_mosaic

        extra_kw: dict[str, str] = {"backend": backend}
        if device is not None:
            extra_kw["device"] = device
        best_fit = fit_mosaic(
            mosaic,
            basic_kwargs={**best_params, **extra_kw},
            n_extra_rows=n_extra_rows,
            n_workers=n_workers,
            verbose=verbose,
        )

    return TuneResult(
        best_params=best_params,
        best_value=float(best_trial.value or 0.0),
        trials_df=trials_df,
        best_fit=best_fit,
        objective=objective,
        working_size_selector=working_size_selector,
    )
