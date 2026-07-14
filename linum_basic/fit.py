"""Per-z mosaic-grid fitting pipeline.

Runs one :class:`~linum_basic.core.BaSiC` model per z-level over all
tiles extracted from a :class:`~linum_basic.mosaic.MosaicGrid`, then
optionally collapses the per-z fields into a single global field.

OCT data typically exhibits a **focal curve** — the illumination profile
shifts slightly with depth because the focal plane moves as z increases.
``field_mode="per-z"`` captures this variation; ``field_mode="global"``
averages it out, which is more robust on noisy data but blurs the depth
dependence.

.. tip::
   For a smooth per-z estimate on noisy data, post-process
   ``MosaicFit.flatfields`` with a 1-D Gaussian kernel along the z-axis
   (e.g. ``scipy.ndimage.gaussian_filter1d(flatfields, sigma=2, axis=0)``)
   before calling :func:`apply_fit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from linum_basic._parallel import list_cuda_devices, parallel_map, parallel_map_cuda_devices, resolve_workers
from linum_basic._working_size import (
    PREVIEW_Z_SAMPLE,
    WorkingSizeResolution,
    build_working_size_context,
    resolve_working_size,
)
from linum_basic.benchmark.strategies import (
    BASELINE_REFERENCE_ID,
    EVIDENCE_POLICY_ID,
    REASON_GUARD_WS128_BLOCKED_BATCHED,
    build_workload_context,
    resolve_auto_strategy,
    resolve_strategy,
)
from linum_basic.core import BaSiC
from linum_basic.mosaic import MosaicGrid

try:
    from linum_basic._batched_fit import (
        fit_stacks_batched,
        prepare_stacks_batched,
        should_use_batched_cuda,
        upsample_fields_batched,
    )

    _BATCHED_FIT_AVAILABLE = True
except ImportError:
    _BATCHED_FIT_AVAILABLE = False

if TYPE_CHECKING:
    pass

__all__ = ["MosaicFit", "apply_fit", "fit_mosaic", "make_model"]

FIT_MOSAIC_STRATEGIES: tuple[str, ...] = ("auto", "sequential", "multi", "batched")

# BaSiC.__init__ keyword-argument names; everything else is a post-init attr.
_BASIC_INIT_PARAMS: frozenset[str] = frozenset({"estimate_darkfield", "extension", "verbose", "backend", "device"})


def make_model(tiles: np.ndarray, params: dict[str, Any]) -> BaSiC:
    """Construct and fully configure a :class:`BaSiC` model from *params*.

    Parameters accepted by ``BaSiC.__init__`` are passed as keyword
    arguments; the rest (``working_size``, ``l_s``, ``l_d``,
    ``epsilon``, …) are set as attributes after construction.

    Parameters
    ----------
    tiles : numpy.ndarray
        Tile stack, shape ``(N, th, tw)``, used to initialise the model.
    params : dict
        BaSiC hyperparameters.  Keys matching ``BaSiC.__init__`` arguments
        are passed to the constructor; remaining keys are set as attributes.

    Returns
    -------
    BaSiC
        The configured model, ready for ``prepare()`` and ``run()``.
    """
    init_kw = {k: v for k, v in params.items() if k in _BASIC_INIT_PARAMS}
    post_kw = {k: v for k, v in params.items() if k not in _BASIC_INIT_PARAMS}
    model = BaSiC(tiles, **init_kw)
    for key, val in post_kw.items():
        setattr(model, key, val)
    return model


def _convergence_from_model(model: Any) -> dict[str, Any]:
    """Extract reweighting iteration count and auto-tuned regularisation."""
    l_s = getattr(model, "l_s", None)
    l_d = getattr(model, "l_d", None)
    return {
        "reweighting_iteration": int(model.reweighting_iteration),
        "alm_iterations_last": int(getattr(model, "last_alm_iterations", 0)),
        "l_s": float(l_s) if l_s is not None else None,
        "l_d": float(l_d) if l_d is not None else None,
    }


def _fit_one_z(
    tiles: np.ndarray,
    params: dict[str, Any],
    n_extra_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a single z-level tile stack and return its flat/dark-fields.

    Module-level (picklable) so it can run inside a worker process. The
    galvo-return rows are stripped before fitting; the caller writes the
    returned fields back into the masked output arrays.

    Parameters
    ----------
    tiles : numpy.ndarray
        Tile stack for one z-level, shape ``(N, th, tw)``.
    params : dict
        BaSiC hyperparameters (see :func:`make_model`).
    n_extra_rows : int
        Number of galvo-return rows to exclude from the top of each tile.

    Returns
    -------
    tuple of numpy.ndarray
        ``(flatfield, darkfield, convergence)`` for the (possibly row-cropped) tiles.
        *convergence* carries ``reweighting_iteration``, ``l_s``, and ``l_d``.
    """
    if n_extra_rows > 0:
        tiles = tiles[:, n_extra_rows:, :]
    model = make_model(tiles, params)
    model.prepare()
    model.run()
    return model.get_flatfield(), model.get_darkfield(), _convergence_from_model(model)


def _fit_one_z_on_device(
    tiles: np.ndarray,
    device: str,
    params: dict[str, Any],
    n_extra_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one z-level on a specific CUDA device (picklable entry point)."""
    dev_params = dict(params)
    dev_params["device"] = device
    return _fit_one_z(tiles, dev_params, n_extra_rows)


def _fit_one_z_cuda_map(
    tiles: np.ndarray,
    device: str,
    *,
    params: dict[str, Any],
    n_extra_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Module-level partial target for :func:`parallel_map_cuda_devices`."""
    return _fit_one_z_on_device(tiles, device, params, n_extra_rows)


def _fit_batched_chunk(
    tile_stacks: list[np.ndarray],
    device: str,
    *,
    params: dict[str, Any],
    n_extra_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a z-chunk with the batched CUDA solver on *device*."""
    from linum_basic.backend import get_xp

    dev_params = dict(params)
    dev_params["device"] = device
    working_size = int(dev_params.get("working_size", 128))
    l_s = dev_params.get("l_s")
    l_d = dev_params.get("l_d")

    img_sort, l_s_arr, l_d_arr, crop_shape = prepare_stacks_batched(
        tile_stacks,
        working_size=working_size,
        n_extra_rows=n_extra_rows,
        l_s=float(l_s) if l_s is not None else None,
        l_d=float(l_d) if l_d is not None else None,
    )
    xp = get_xp("torch", device)
    ff_ws, df_ws = fit_stacks_batched(
        img_sort,
        l_s_arr=l_s_arr,
        l_d_arr=l_d_arr,
        params=dev_params,
        xp=xp,
    )
    return upsample_fields_batched(ff_ws, df_ws, (int(crop_shape[0]), int(crop_shape[1])))


def _resolve_batched_z_chunk_size(params: dict[str, Any], n_z: int, n_devices: int) -> int:
    """Resolve z-chunk size for the batched CUDA path.

    A single ws=128 batch over dozens of z-planes is memory-bandwidth bound.
    Smaller chunks keep per-batch working sets cache-friendlier and, on
    multi-GPU systems, give the scheduler enough chunks to keep both devices
    busy without launching one process per z-plane.
    """
    import os

    raw = params.get("batched_z_chunk_size")
    if raw is None:
        raw = os.environ.get("LINUM_BASIC_BATCHED_Z_CHUNK_SIZE")
    if raw is not None:
        chunk_size = int(raw)
        if chunk_size <= 0:
            return n_z
        return max(1, min(n_z, chunk_size))

    working_size = int(params.get("working_size", 128))
    if working_size >= 128:
        # 8 z-planes is a good default for production ws=128: it limits peak
        # memory while creating enough chunks for two A6000s to share work.
        return min(n_z, 8)
    if working_size >= 96:
        return min(n_z, 12)
    if working_size >= 64:
        return min(n_z, 24 if n_devices <= 1 else 16)
    return n_z


def _allow_batched_cuda_for_params(params: dict[str, Any]) -> bool:
    """Return whether batched CUDA should run for this parameter set."""
    import os

    if bool(params.get("force_batched_cuda", False)):
        return True
    if os.environ.get("LINUM_BASIC_FORCE_BATCHED_CUDA", "0") == "1":
        return True

    # Production linumpy currently uses BaSiC's default working_size=128.
    # Server benchmarks show this shape is memory-bound: chunking helps peak
    # memory, but still loses to the scalar per-z CUDA path. Keep the batched
    # path for lower working sizes where it gives a real throughput win.
    return int(params.get("working_size", 128)) < 128


def _strategy_lock_execution_path(name: str, n_gpus: int) -> str:
    """Map a named strategy lock to the resolver execution-path vocabulary."""
    if name == "sequential":
        return "sequential_scalar"
    if name == "multi":
        return "multi_gpu_fanout"
    if name == "batched":
        return "batched_cuda_multi_gpu" if n_gpus >= 2 else "batched_cuda"
    return "sequential_scalar"


def _build_strategy_lock_metadata(
    *,
    name: str,
    params: dict[str, Any],
    context: Any,
    n_z: int,
) -> dict[str, Any]:
    """Build D-19 ``_strategy`` metadata for a named (non-auto) strategy lock."""
    execution_path = _strategy_lock_execution_path(name, context.n_gpus)
    return {
        "name": name,
        "execution_path": execution_path,
        "chunk_size": None,
        "device": params.get("device"),
        "backend": params.get("backend", "torch"),
        "working_size": int(params.get("working_size", 128)),
        "n_z": n_z,
        "n_tiles": context.n_tiles,
        "n_gpus": context.n_gpus,
        "memory_estimate_bytes": context.memory_estimate_bytes,
        "batched_cuda_guard": None,
        "attempted_execution_path": None,
        "reason_codes": [f"STRATEGY_LOCK_{name.upper()}"],
        "reason_summary": f"Named strategy '{name}' locks execution path (auto heuristics suppressed).",
        "override_source": "strategy_lock",
        "evidence_policy": EVIDENCE_POLICY_ID,
        "baseline_reference_id": BASELINE_REFERENCE_ID,
    }


def _batched_explicitly_desired(params: dict[str, Any], strategy_meta: dict[str, Any]) -> bool:
    """Return whether the caller or resolver intentionally selected batched CUDA."""
    import os

    if bool(params.get("force_batched_cuda", False)):
        return True
    if os.environ.get("LINUM_BASIC_FORCE_BATCHED_CUDA", "0") == "1":
        return True
    if strategy_meta.get("name") == "batched":
        return True
    resolved_path = str(strategy_meta.get("execution_path") or "")
    return "batched" in resolved_path


def _record_batched_guard_downgrade(
    strategy_meta: dict[str, Any],
    *,
    n_cuda_devices: int,
) -> None:
    """Record ws>=128 batched guard block in ``_strategy`` metadata (D-18)."""
    attempted = "batched_cuda_multi_gpu" if n_cuda_devices > 1 else "batched_cuda"
    actual = "multi_gpu_fanout" if n_cuda_devices > 1 else "sequential_scalar"
    strategy_meta["attempted_execution_path"] = attempted
    strategy_meta["batched_cuda_guard"] = "ws>=128_blocked"
    strategy_meta["execution_path"] = actual
    reason_codes = strategy_meta.setdefault("reason_codes", [])
    if REASON_GUARD_WS128_BLOCKED_BATCHED not in reason_codes:
        reason_codes.append(REASON_GUARD_WS128_BLOCKED_BATCHED)


def _finalize_strategy_execution_metadata(
    strategy_meta: dict[str, Any],
    *,
    params: dict[str, Any],
    n_z: int,
    cuda_devices: list[str],
    use_batched_cuda: bool,
    use_cuda_fanout: bool,
) -> None:
    """Set final ``execution_path`` and ``chunk_size`` from the actual execution fork."""
    if use_batched_cuda:
        strategy_meta["execution_path"] = "batched_cuda_multi_gpu" if len(cuda_devices) > 1 else "batched_cuda"
        strategy_meta["batched_cuda_guard"] = None
        strategy_meta["chunk_size"] = _resolve_batched_z_chunk_size(
            params,
            n_z,
            len(cuda_devices) or 1,
        )
        return

    strategy_meta["chunk_size"] = None
    if strategy_meta.get("batched_cuda_guard") is not None:
        return
    if use_cuda_fanout:
        strategy_meta["execution_path"] = "multi_gpu_fanout"
    else:
        strategy_meta["execution_path"] = "sequential_scalar"


def _iter_z_chunks(tile_stacks: list[np.ndarray], chunk_size: int) -> list[tuple[int, list[np.ndarray]]]:
    """Return ``(start_index, chunk)`` pairs preserving z order."""
    return [(start, tile_stacks[start : start + chunk_size]) for start in range(0, len(tile_stacks), chunk_size)]


def _fit_mosaic_batched_cuda(
    tile_stacks: list[np.ndarray],
    *,
    params: dict[str, Any],
    n_extra_rows: int,
    cuda_devices: list[str],
    verbose: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Batched CUDA fit, optionally splitting z-levels into chunks across GPUs."""
    chunk_size = _resolve_batched_z_chunk_size(params, len(tile_stacks), len(cuda_devices))
    chunks = _iter_z_chunks(tile_stacks, chunk_size)

    if len(cuda_devices) <= 1 or len(chunks) <= 1:
        device = cuda_devices[0] if cuda_devices else "cuda:0"
        results: list[tuple[np.ndarray, np.ndarray]] = []
        for _start, chunk in chunks:
            ff, df = _fit_batched_chunk(chunk, device, params=params, n_extra_rows=n_extra_rows)
            results.extend((ff[i], df[i]) for i in range(ff.shape[0]))
        return results

    n_jobs = min(len(cuda_devices), len(chunks))
    chunk_fn = partial(_fit_batched_chunk, params=params, n_extra_rows=n_extra_rows)

    def _run_chunk(item: tuple[int, list[np.ndarray]], device: str) -> tuple[int, list[tuple[np.ndarray, np.ndarray]]]:
        ff, df = chunk_fn(item[1], device)
        return item[0], [(ff[i], df[i]) for i in range(ff.shape[0])]

    import os

    from joblib import Parallel, delayed, parallel_config

    from linum_basic._torch_cache import _cuda_joblib_worker_init, collect_cuda_worker_env

    def _task(index: int, chunk: list[np.ndarray], device: str) -> tuple[int, list[tuple[np.ndarray, np.ndarray]]]:
        gpu_ix = device.rsplit(":", 1)[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ix
        return _run_chunk((index, chunk), "cuda:0")

    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    env_items = collect_cuda_worker_env()
    with parallel_config(
        backend="loky",
        inner_max_num_threads=1,
        n_jobs=n_jobs,
        initializer=_cuda_joblib_worker_init,
        initargs=(cache_dir, env_items),
    ):
        pairs: list[tuple[int, list[tuple[np.ndarray, np.ndarray]]]] = Parallel(verbose=10 if verbose else 0)(
            delayed(_task)(start, chunk, cuda_devices[i % len(cuda_devices)]) for i, (start, chunk) in enumerate(chunks)
        )
    pairs.sort(key=lambda pair: pair[0])
    return [item for _idx, chunk_results in pairs for item in chunk_results]


@dataclass(init=False)
class MosaicFit:
    """Results of a BaSiC fit applied to a mosaic grid.

    Attributes
    ----------
    flatfields : numpy.ndarray
        Shape ``(Z, th, tw)`` for ``field_mode="per-z"``;
        shape ``(th, tw)`` for ``field_mode="global"``.
    darkfields : numpy.ndarray
        Same shape as *flatfields*.
    field_mode : {"per-z", "global"}
        Whether fields are per-z or a single averaged field.
    z_indices : list of int
        Z-levels that were fitted (subset of all z-levels).
    params : dict
        BaSiC hyperparameters used for this fit.
    convergence_per_z : list of dict or None
        Per-z reweighting telemetry from the scalar fit path; ``None`` when
        unavailable (e.g. batched CUDA).
    """

    flatfields: np.ndarray
    darkfields: np.ndarray
    field_mode: str
    z_indices: list[int]
    params: dict[str, Any]
    convergence_per_z: list[dict[str, Any]] | None

    def __init__(
        self,
        flatfields: np.ndarray,
        darkfields: np.ndarray,
        field_mode: str,
        z_indices: list[int],
        params: dict[str, Any] | None = None,
        convergence_per_z: list[dict[str, Any]] | None = None,
    ) -> None:
        """Construct a :class:`MosaicFit` result.

        Parameters
        ----------
        flatfields : numpy.ndarray
            Shape ``(Z, th, tw)`` for ``field_mode="per-z"``;
            shape ``(th, tw)`` for ``field_mode="global"``.
        darkfields : numpy.ndarray
            Same shape as *flatfields*.
        field_mode : {"per-z", "global"}
            Whether fields are per-z or a single averaged field.
        z_indices : list of int
            Z-levels that were fitted (subset of all z-levels).
        params : dict or None
            BaSiC hyperparameters used for this fit.  Defaults to ``{}``.
        convergence_per_z : list of dict or None
            Per-z reweighting telemetry; ``None`` when not recorded.
        """
        self.flatfields = flatfields
        self.darkfields = darkfields
        self.field_mode = field_mode
        self.z_indices = z_indices
        self.params = params if params is not None else {}
        self.convergence_per_z = convergence_per_z


def _resolve_working_size_auto(
    mosaic: MosaicGrid,
    user_kwargs: dict[str, Any],
    *,
    n_z: int,
    field_mode: str,
) -> WorkingSizeResolution | None:
    """Resolve the ``working_size="auto"`` sentinel to a concrete int.

    Activates *only* when the caller asked for ``"auto"`` (opt-in). Builds a
    :class:`WorkingSizeContext` from mosaic metadata, a bounded preview mean,
    and a best-effort memory budget, then calls the pure resolver. The
    resolved int replaces the sentinel in *user_kwargs* so a literal
    ``"auto"`` never reaches :class:`BaSiC` or the strategy resolver.

    Returns ``None`` when no resolution is needed (caller passed an explicit
    int or omitted ``working_size``). Always fails safe to
    :data:`~linum_basic._working_size.SAFE_DEFAULT` (128).
    """
    requested = user_kwargs.get("working_size", 128)
    if requested != "auto":
        return None

    # Form a bounded preview mean across at most PREVIEW_Z_SAMPLE z-planes,
    # averaged across tiles. Cheap (one read per plane + one mean).
    sample_z = list(range(n_z))[:PREVIEW_Z_SAMPLE]
    mean_image: np.ndarray | None = None
    if sample_z:
        try:
            stacks = [mosaic.iter_tiles(z) for z in sample_z]
            stacked = np.stack(stacks, axis=0)  # (n_sample_z, n_tiles, th, tw)
            mean_image = stacked.mean(axis=(0, 1))
        except ValueError, IndexError, OSError:
            mean_image = None

    device = user_kwargs.get("device")
    ctx = build_working_size_context(
        n_z=n_z,
        n_tiles=mosaic.n_tiles,
        field_mode=field_mode,
        tile_shape=tuple(mosaic.tile_shape),
        mean_image=mean_image,
        device=device,
        requested="auto",
    )
    resolution = resolve_working_size(ctx)
    # Replace the sentinel with the concrete int before it flows downstream.
    user_kwargs["working_size"] = resolution.resolved_working_size
    return resolution


def fit_mosaic(
    mosaic: MosaicGrid,
    *,
    z_indices: list[int] | None = None,
    field_mode: Literal["per-z", "global"] = "per-z",
    basic_kwargs: dict[str, Any] | None = None,
    strategy: Literal["auto", "sequential", "multi", "batched"] = "auto",
    streaming: bool = False,
    n_extra_rows: int = 0,
    n_workers: int | None = None,
    verbose: bool = False,
) -> MosaicFit:
    """Fit one BaSiC model per z-level over all tiles of *mosaic*.

    Parameters
    ----------
    mosaic : MosaicGrid
        The mosaic grid to fit.
    z_indices : list of int or None
        Z-levels to fit.  ``None`` fits every z-level.
    field_mode : {"per-z", "global"}
        ``"per-z"`` stores one flat/dark-field per fitted z-level.
        ``"global"`` averages all per-z fields into a single field.
        Use ``"per-z"`` when OCT focal-plane position varies with depth.
        Use ``"global"`` when the dataset is too noisy for reliable per-z
        estimation.
    basic_kwargs : dict or None
        Hyperparameters forwarded to :class:`~linum_basic.core.BaSiC`.
        May include ``working_size``, ``l_s``, ``l_d``, ``epsilon``,
        ``estimate_darkfield``, ``backend``, ``device``, etc.
    strategy : {"auto", "sequential", "multi", "batched"}
        Execution strategy. ``"auto"`` (default) resolves path from workload
        shape, hardware, and Phase 2/3 policy. Named strategies lock the
        execution path; the resolved decision is recorded in
        ``params["_strategy"]``.
    streaming : bool
        When ``True``, fit z-levels one at a time without pre-extracting all
        per-z tile stacks into memory, bounding peak memory to a single plane
        instead of the full z-stack. Sequential only: raises
        :exc:`ValueError` if combined with ``strategy="multi"`` or
        ``"batched"``. Numerics are identical to the non-streaming sequential
        path — the same ``_fit_one_z`` runs on the same tiles in the same
        order; only the extraction timing differs (one-at-a-time vs
        all-up-front). Pair with a lazy ``MosaicGrid``
        (``MosaicGrid.from_ome_zarr(..., lazy=True)``) so the backing volume
        is also not loaded into memory up front. Default ``False``.
    n_extra_rows : int
        Number of rows at the top of each tile to exclude from the BaSiC
        fit (galvo return / flyback signal).  The excluded rows are filled
        with 1.0 (flat-field) and 0.0 (dark-field) in the output so that
        :func:`apply_fit` treats them as uncorrected pass-through pixels.
        Default ``0`` (no masking).
    n_workers : int or None
        Number of worker processes used to fit z-levels in parallel. Each
        z-level is an independent BaSiC solve, so this scales nearly
        linearly on CPU. ``None`` (default) uses ``cpu_count() - 2``.
        ``1`` runs sequentially. On a **single** CUDA device the worker count is
        forced to ``1`` (single-accelerator contention). With multiple CUDA
        devices, z-levels fan out across GPUs up to ``min(n_workers, n_gpus)``.
    verbose : bool
        Show a progress bar over z-levels.

    Returns
    -------
    MosaicFit
        Fitted flat/dark-fields for each requested z-level.
    """
    if strategy not in FIT_MOSAIC_STRATEGIES:
        msg = f"Unknown strategy: {strategy}; choose from {FIT_MOSAIC_STRATEGIES}"
        raise ValueError(msg)

    if streaming and strategy in ("multi", "batched"):
        msg = (
            f"streaming=True is incompatible with strategy={strategy!r}; "
            "streaming is sequential only. Use strategy='auto' or 'sequential'."
        )
        raise ValueError(msg)

    user_kwargs = dict(basic_kwargs or {})
    z_idx = list(z_indices) if z_indices is not None else list(range(mosaic.n_z))

    # Resolve the working_size="auto" sentinel (opt-in, M006/S02) to a concrete
    # int *before* it reaches BaSiC, the workload context, or the strategy
    # resolver. Existing callers that never pass "auto" are unaffected.
    ws_resolution = _resolve_working_size_auto(mosaic, user_kwargs, n_z=len(z_idx), field_mode=field_mode)

    working_size = int(user_kwargs.get("working_size", 128))
    estimate_darkfield = bool(user_kwargs.get("estimate_darkfield", False))
    max_reweighting_iterations = int(user_kwargs.get("max_reweighting_iterations", 10))

    context = build_workload_context(
        n_z=len(z_idx),
        n_tiles=mosaic.n_tiles,
        field_mode=field_mode,
        working_size=working_size,
        device=user_kwargs.get("device"),
    )

    if strategy == "auto":
        resolved = resolve_auto_strategy(context, user_kwargs=user_kwargs)
        strategy_meta = dict(resolved.strategy_metadata)
        params: dict[str, Any] = {**resolved.basic_kwargs, **user_kwargs}
    else:
        strat = resolve_strategy(
            strategy,
            working_size=working_size,
            estimate_darkfield=estimate_darkfield,
            max_reweighting_iterations=max_reweighting_iterations,
        )
        params = {**strat.basic_kwargs, **user_kwargs}
        strategy_meta = _build_strategy_lock_metadata(
            name=strategy,
            params=params,
            context=context,
            n_z=len(z_idx),
        )

    # Warm-start inner ALM across outer reweighting passes when the cap is high.
    if "warm_start_reweighting" not in params and int(params.get("max_reweighting_iterations", 10)) >= 5:
        params["warm_start_reweighting"] = True

    params["_strategy"] = strategy_meta

    # Record the adaptive working_size resolution (purely additive metadata;
    # never affects fit numerics). Mirrors the params["_strategy"] D-19 pattern.
    if ws_resolution is not None:
        params["_working_size_selector"] = ws_resolution.metadata()

    th, tw = mosaic.tile_shape
    flatfields = np.ones((len(z_idx), th, tw), dtype=np.float32)
    darkfields = np.zeros((len(z_idx), th, tw), dtype=np.float32)

    n_eff = resolve_workers(n_workers, params.get("backend"), params.get("device"))
    cuda_devices = list_cuda_devices(params.get("device"))
    use_cuda_fanout = (
        len(cuda_devices) > 1
        and params.get("backend") in {"torch", "auto"}
        and (params.get("device") or "cuda").lower().startswith("cuda")
    )

    convergence_per_z: list[dict[str, Any]] | None = None
    use_batched_cuda = False
    if streaming:
        # Sequential streaming: extract and fit one z-level at a time so peak
        # memory is bounded by a single plane instead of all per-z tile stacks
        # held live simultaneously.  The same ``_fit_one_z`` runs on the same
        # tiles in the same order as the non-streaming sequential path, so
        # numerics are identical — only the extraction timing differs.
        convergence_per_z = []
        fit_fn = partial(_fit_one_z, params=params, n_extra_rows=n_extra_rows)
        iterator: Any = z_idx
        if verbose:
            from tqdm.auto import tqdm

            iterator = tqdm(z_idx, desc="Fitting z-levels (streaming)", total=len(z_idx), leave=False)
        results = [fit_fn(mosaic.iter_tiles(z)) for z in iterator]
        use_cuda_fanout = False
    else:
        # Pre-extract per-z tile stacks so workers receive only the slice they
        # need (cheap views into the in-memory mosaic) instead of the whole grid.
        tile_stacks = [mosaic.iter_tiles(z) for z in z_idx]

        use_batched_cuda = _BATCHED_FIT_AVAILABLE and should_use_batched_cuda(
            field_mode=field_mode,
            n_z=len(z_idx),
            backend=params.get("backend"),
            device=params.get("device"),
        )
        if use_batched_cuda and not _allow_batched_cuda_for_params(params):
            if _batched_explicitly_desired(params, params["_strategy"]):
                _record_batched_guard_downgrade(
                    params["_strategy"],
                    n_cuda_devices=len(cuda_devices) or 1,
                )
            use_batched_cuda = False
        if use_batched_cuda:
            results = _fit_mosaic_batched_cuda(
                tile_stacks,
                params=params,
                n_extra_rows=n_extra_rows,
                cuda_devices=cuda_devices or ["cuda:0"],
                verbose=verbose,
            )
        elif use_cuda_fanout:
            convergence_per_z = []
            cuda_fit = partial(_fit_one_z_cuda_map, params=params, n_extra_rows=n_extra_rows)
            results = parallel_map_cuda_devices(
                cuda_fit,
                tile_stacks,
                cuda_devices,
                desc="Fitting z-levels (multi-GPU)",
                verbose=verbose,
            )
        else:
            convergence_per_z = []
            results = parallel_map(
                partial(_fit_one_z, params=params, n_extra_rows=n_extra_rows),
                tile_stacks,
                n_eff,
                desc="Fitting z-levels",
                verbose=verbose,
            )
    for i, result in enumerate(results):
        flatfield, darkfield = result[0], result[1]
        if convergence_per_z is not None and len(result) == 3:
            convergence_per_z.append(result[2])
        flatfields[i, n_extra_rows:, :] = flatfield
        darkfields[i, n_extra_rows:, :] = darkfield

    _finalize_strategy_execution_metadata(
        params["_strategy"],
        params=params,
        n_z=len(z_idx),
        cuda_devices=cuda_devices,
        use_batched_cuda=use_batched_cuda,
        use_cuda_fanout=use_cuda_fanout,
    )

    if field_mode == "global":
        flatfields_out: np.ndarray = flatfields.mean(axis=0)
        darkfields_out: np.ndarray = darkfields.mean(axis=0)
    else:
        flatfields_out = flatfields
        darkfields_out = darkfields

    return MosaicFit(
        flatfields=flatfields_out,
        darkfields=darkfields_out,
        field_mode=field_mode,
        z_indices=z_idx,
        params=params,
        convergence_per_z=convergence_per_z,
    )


def apply_fit(
    mosaic: MosaicGrid,
    fit: MosaicFit,
    *,
    epsilon: float = 1e-6,
    n_extra_rows: int = 0,
) -> np.ndarray:
    """Apply a :class:`MosaicFit` to produce a corrected mosaic volume.

    Parameters
    ----------
    mosaic : MosaicGrid
        The original (uncorrected) mosaic.
    fit : MosaicFit
        Flat/dark-field estimates from :func:`fit_mosaic`.
    epsilon : float
        Stability constant added to the flat-field denominator.
    n_extra_rows : int
        Number of galvo-return rows at the top of each tile to replace by
        edge-extension from the first valid row.  Must match the value used
        in :func:`fit_mosaic`.  Default ``0`` (no replacement).

    Returns
    -------
    numpy.ndarray, shape (Z, H, W), dtype float32
        Corrected mosaic where every tile has been divided by its flat-field
        and had its dark-field subtracted.  If *n_extra_rows* > 0 the galvo
        rows are replaced with the first valid row value so that they do not
        create a visible dark notch at tile boundaries.
    """
    nz = mosaic.n_z
    th, tw = mosaic.tile_shape
    nrows, ncols = mosaic.n_rows, mosaic.n_cols

    # Materialise to ndarray: no-op for eager MosaicGrid, required read for a
    # lazy (zarr.Array-backed) grid.  apply_fit always needs the full volume
    # since it corrects every pixel.
    corrected = np.asarray(mosaic.array).astype(np.float32).copy()

    for z in range(nz):
        if fit.field_mode == "global":
            ff: np.ndarray = fit.flatfields
            df: np.ndarray = fit.darkfields
        else:
            pos = _nearest_z_pos(z, fit.z_indices)
            ff = fit.flatfields[pos]
            df = fit.darkfields[pos]

        # View the (H, W) plane as a (nrows, th, ncols, tw) tile grid and apply
        # the (th, tw) fields to every tile at once via broadcasting.
        view = corrected[z].reshape(nrows, th, ncols, tw)
        view[...] = (view - df[None, :, None, :]) / (ff[None, :, None, :] + epsilon)
        if n_extra_rows > 0:
            first_valid = view[:, n_extra_rows : n_extra_rows + 1, :, :]
            view[:, :n_extra_rows, :, :] = first_valid
        corrected[z] = view.reshape(nrows * th, ncols * tw)

    return corrected


def save_corrected(
    mosaic: MosaicGrid,
    fit: MosaicFit,
    output_path: str | Path,
    *,
    input_path: str | Path | None = None,
    overwrite: bool = False,
    epsilon: float = 1e-6,
) -> None:
    """Correct *mosaic* with *fit* and save the result as OME-Zarr.

    Parameters
    ----------
    mosaic : MosaicGrid
        Original mosaic grid.
    fit : MosaicFit
        Estimated fields.
    output_path : str or Path
        Destination ``.ome.zarr`` path.
    input_path : str or Path or None
        If provided the axes and scale metadata are copied from the input
        OME-Zarr; otherwise defaults ``(z, y, x)`` / 1.0 are used.
    overwrite : bool
        Overwrite *output_path* if it exists.
    epsilon : float
        Forwarded to :func:`apply_fit`.
    """
    from linum_basic.io.zarr import load_ome_zarr, write_ome_zarr

    corrected = apply_fit(mosaic, fit, epsilon=epsilon)

    axes = ["z", "y", "x"]
    scale = [1.0, 1.0, 1.0]
    if input_path is not None:
        _, axes, scale = load_ome_zarr(input_path)

    write_ome_zarr(output_path, corrected, axes=axes, scale=scale, overwrite=overwrite)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nearest_z_pos(z: int, z_indices: list[int]) -> int:
    """Index into *z_indices* of the element closest to *z*."""
    arr = np.array(z_indices, dtype=np.int64)
    return int(np.argmin(np.abs(arr - z)))
