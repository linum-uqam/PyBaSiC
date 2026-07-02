"""CI guard: scripts/phase11_gpu_run.sh syntax and worker-compile-off env levers."""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path("scripts/phase11_gpu_run.sh")


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_phase11_gpu_run_script_passes_bash_syntax_check() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_phase11_gpu_run_sets_compile_mode_off() -> None:
    content = _script_text()
    assert "LINUM_BASIC_ALM_COMPILE_MODE=off" in content


def test_phase11_gpu_run_sets_dct_kernel_tuned() -> None:
    content = _script_text()
    assert "LINUM_BASIC_DCT_KERNEL=tuned" in content


def test_phase11_gpu_run_unsets_inductor_warm_passes() -> None:
    content = _script_text()
    assert "unset LINUM_BASIC_INDUCTOR_WARM_PASSES" in content


def test_phase11_gpu_run_uses_ws128_and_500_reweight_iters() -> None:
    content = _script_text()
    assert "WS=128" in content
    assert "MAX_REWEIGHT=500" in content
    assert '--working-size "$WS"' in content
    assert '--max-reweighting-iterations "$MAX_REWEIGHT"' in content


def test_phase11_gpu_run_uses_sequential_strategy() -> None:
    content = _script_text()
    assert "--strategy sequential" in content
    assert "--strategy baseline" not in content


def test_phase11_gpu_run_reuses_frozen_sub22_baseline() -> None:
    content = _script_text()
    assert "baseline-20260701T020128-be1e880-sub-22" in content
    assert "preflight_frozen_baseline" in content


def test_phase11_gpu_run_covers_three_subjects() -> None:
    content = _script_text()
    for subject_id in ("sub-18", "sub-21", "sub-22"):
        assert subject_id in content


def test_phase11_gpu_run_runs_cuda_joblib_worker_init_pytest_gate() -> None:
    content = _script_text()
    assert "tests/test_parallel.py" in content
    assert "cuda_joblib_worker_init" in content


def test_phase11_gpu_run_uses_worker_compile_off_run_label() -> None:
    content = _script_text()
    assert 'RUN_LABEL="lever-worker-compile-off"' in content
    assert "bottleneck-report-worker-compile-off.json" in content


def test_phase11_gpu_run_writes_multisubject_summary() -> None:
    content = _script_text()
    assert "phase11-multisubject-summary.json" in content


def test_phase11_gpu_run_uses_shared_z_indices() -> None:
    content = _script_text()
    assert 'Z_INDICES="0,13,27,40,54"' in content
