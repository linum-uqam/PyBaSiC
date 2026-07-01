"""Tests for parallelism helpers and parallel/vectorized pipeline paths."""

from __future__ import annotations

import os

import numpy as np
import pytest

from linum_basic._parallel import (
    default_workers,
    is_gpu_backend,
    list_cuda_devices,
    parallel_map,
    parallel_map_cuda_devices,
    resolve_workers,
)
from linum_basic._torch_cache import _cuda_joblib_worker_init
from linum_basic.fit import MosaicFit, apply_fit, fit_mosaic
from linum_basic.mosaic import MosaicGrid


def _square(x: int) -> int:
    """Module-level worker so :func:`parallel_map` can pickle it."""
    return x * x


def _make_mosaic(n_z: int, n_rows: int, n_cols: int, th: int, tw: int, *, seed: int = 0) -> MosaicGrid:
    """Build a small synthetic mosaic with random tile content."""
    rng = np.random.default_rng(seed)
    array = rng.random((n_z, n_rows * th, n_cols * tw), dtype=np.float64).astype(np.float32)
    return MosaicGrid(array=array, tile_shape=(th, tw), overlap_fraction=0.2)


# ---------------------------------------------------------------------------
# default_workers / resolve_workers / is_gpu_backend
# ---------------------------------------------------------------------------


def test_default_workers_is_cpu_count_minus_two() -> None:
    assert default_workers() == max(1, (os.cpu_count() or 4) - 2)


@pytest.mark.parametrize(
    ("backend", "device", "expected"),
    [
        ("torch", "cuda:0", True),
        ("auto", "mps", False),
        ("torch", "mps", False),
        ("torch", "cuda", True),
        ("numpy", "cuda", False),
        ("torch", "cpu", False),
        ("torch", None, False),
        ("numpy", None, False),
    ],
)
def test_is_gpu_backend(backend: str, device: str | None, expected: bool) -> None:
    assert is_gpu_backend(backend, device) is expected


def test_resolve_workers_defaults_and_clamps() -> None:
    assert resolve_workers(None, "numpy", None) == default_workers()
    assert resolve_workers(4, "numpy", None) == 4
    assert resolve_workers(0, "numpy", None) == 1
    assert resolve_workers(-3, "numpy", None) == 1


def test_resolve_workers_gpu_guard_forces_one() -> None:
    assert resolve_workers(4, "torch", "cpu") == 4
    assert resolve_workers(4, "auto", "mps") == 4
    assert resolve_workers(None, "auto", "mps") == default_workers()
    with pytest.warns(UserWarning, match="sequentially"):
        assert resolve_workers(4, "torch", "cuda:0") == 1


def test_list_cuda_devices_explicit_index() -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    assert list_cuda_devices("cuda:0") == ["cuda:0"]


def test_fit_mosaic_multi_gpu_matches_sequential() -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >= 2 CUDA devices")
    mosaic = _make_mosaic(n_z=4, n_rows=2, n_cols=2, th=8, tw=8, seed=7)
    basic_kwargs = {
        "estimate_darkfield": True,
        "working_size": 8,
        "backend": "torch",
        "device": "cuda",
        "max_reweighting_iterations": 3,
    }
    seq = fit_mosaic(mosaic, basic_kwargs=basic_kwargs, n_workers=1)
    par = fit_mosaic(mosaic, basic_kwargs=basic_kwargs, n_workers=2)
    assert np.allclose(seq.flatfields, par.flatfields, atol=1e-4)
    assert np.allclose(seq.darkfields, par.darkfields, atol=1e-4)


# ---------------------------------------------------------------------------
# parallel_map
# ---------------------------------------------------------------------------


def test_parallel_map_sequential_preserves_order() -> None:
    assert parallel_map(_square, [0, 1, 2, 3, 4], 1) == [0, 1, 4, 9, 16]


def test_parallel_map_parallel_preserves_order() -> None:
    assert parallel_map(_square, [0, 1, 2, 3, 4], 2) == [0, 1, 4, 9, 16]


def test_parallel_map_empty() -> None:
    assert parallel_map(_square, [], 2) == []


# ---------------------------------------------------------------------------
# CUDA joblib worker initializer
# ---------------------------------------------------------------------------


def test_cuda_joblib_worker_init_sets_env_and_cache(tmp_path) -> None:
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TORCHINDUCTOR_FX_GRAPH_CACHE", "LINUM_BASIC_DCT_KERNEL"):
        os.environ.pop(key, None)
    cache_dir = tmp_path / "inductor-cache"
    _cuda_joblib_worker_init(str(cache_dir), (("LINUM_BASIC_DCT_KERNEL", "tuned"),))
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(cache_dir.resolve())
    assert os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"
    assert os.environ["LINUM_BASIC_DCT_KERNEL"] == "tuned"


def test_cuda_joblib_worker_init_none_cache_dir_is_safe() -> None:
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TORCHINDUCTOR_FX_GRAPH_CACHE"):
        os.environ.pop(key, None)
    _cuda_joblib_worker_init(None, ())
    assert "TORCHINDUCTOR_FX_GRAPH_CACHE" in os.environ


def _read_torchinductor_cache_dir(_item: int, _device: str) -> str | None:
    """Module-level worker returning the Inductor cache dir env var."""
    return os.environ.get("TORCHINDUCTOR_CACHE_DIR")


def test_parallel_map_cuda_devices_worker_sees_cache_env(tmp_path, monkeypatch) -> None:
    try:
        from joblib import parallel_config  # noqa: F401
    except ImportError:
        pytest.skip("joblib not installed")
    cache_dir = tmp_path / "shared-inductor"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
    results = parallel_map_cuda_devices(
        _read_torchinductor_cache_dir,
        [0, 1],
        ["cuda:0", "cuda:1"],
    )
    expected = str(cache_dir.resolve())
    assert len(results) == 2
    assert all(value == expected for value in results)


# ---------------------------------------------------------------------------
# fit_mosaic: parallel result must match sequential
# ---------------------------------------------------------------------------


def test_fit_mosaic_parallel_matches_sequential() -> None:
    mosaic = _make_mosaic(n_z=3, n_rows=2, n_cols=2, th=8, tw=8, seed=1)
    basic_kwargs = {"estimate_darkfield": False, "working_size": 8}

    seq = fit_mosaic(mosaic, basic_kwargs=basic_kwargs, n_workers=1)
    par = fit_mosaic(mosaic, basic_kwargs=basic_kwargs, n_workers=2)

    assert np.allclose(seq.flatfields, par.flatfields, atol=1e-5)
    assert np.allclose(seq.darkfields, par.darkfields, atol=1e-5)


# ---------------------------------------------------------------------------
# apply_fit: vectorized output must match the reference nested-loop version
# ---------------------------------------------------------------------------


def _apply_fit_reference(
    mosaic: MosaicGrid,
    fit: MosaicFit,
    *,
    epsilon: float = 1e-6,
    n_extra_rows: int = 0,
) -> np.ndarray:
    """Original triple-nested-loop implementation, kept as a test oracle."""
    from linum_basic.fit import _nearest_z_pos

    nz = mosaic.n_z
    th, tw = mosaic.tile_shape
    nrows, ncols = mosaic.n_rows, mosaic.n_cols
    corrected = mosaic.array.astype(np.float32).copy()
    for z in range(nz):
        if fit.field_mode == "global":
            ff = fit.flatfields
            df = fit.darkfields
        else:
            pos = _nearest_z_pos(z, fit.z_indices)
            ff = fit.flatfields[pos]
            df = fit.darkfields[pos]
        for r in range(nrows):
            for c in range(ncols):
                tile = corrected[z, r * th : (r + 1) * th, c * tw : (c + 1) * tw]
                corrected[z, r * th : (r + 1) * th, c * tw : (c + 1) * tw] = (tile - df) / (ff + epsilon)
                if n_extra_rows > 0:
                    first_valid = corrected[z, r * th + n_extra_rows, c * tw : (c + 1) * tw]
                    corrected[z, r * th : r * th + n_extra_rows, c * tw : (c + 1) * tw] = first_valid[np.newaxis, :]
    return corrected


@pytest.mark.parametrize("n_extra_rows", [0, 2])
def test_apply_fit_per_z_matches_reference(n_extra_rows: int) -> None:
    n_z, n_rows, n_cols, th, tw = 3, 2, 2, 8, 8
    mosaic = _make_mosaic(n_z, n_rows, n_cols, th, tw, seed=2)
    rng = np.random.default_rng(3)
    flatfields = rng.uniform(0.5, 1.5, size=(n_z, th, tw)).astype(np.float32)
    darkfields = rng.uniform(0.0, 0.1, size=(n_z, th, tw)).astype(np.float32)
    fit = MosaicFit(flatfields, darkfields, field_mode="per-z", z_indices=list(range(n_z)))

    out = apply_fit(mosaic, fit, n_extra_rows=n_extra_rows)
    ref = _apply_fit_reference(mosaic, fit, n_extra_rows=n_extra_rows)
    assert np.allclose(out, ref, atol=1e-6)


def test_apply_fit_global_matches_reference() -> None:
    n_z, n_rows, n_cols, th, tw = 2, 2, 3, 8, 8
    mosaic = _make_mosaic(n_z, n_rows, n_cols, th, tw, seed=4)
    rng = np.random.default_rng(5)
    flatfield = rng.uniform(0.5, 1.5, size=(th, tw)).astype(np.float32)
    darkfield = rng.uniform(0.0, 0.1, size=(th, tw)).astype(np.float32)
    fit = MosaicFit(flatfield, darkfield, field_mode="global", z_indices=list(range(n_z)))

    out = apply_fit(mosaic, fit)
    ref = _apply_fit_reference(mosaic, fit)
    assert np.allclose(out, ref, atol=1e-6)
