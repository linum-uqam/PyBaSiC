"""CI guard: scripts/phase10_gpu_run.sh syntax and candidate-id capture."""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path("scripts/phase10_gpu_run.sh")


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_phase10_gpu_run_script_passes_bash_syntax_check() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_phase10_gpu_run_suppresses_candidate_stdout_for_id_capture() -> None:
    """tee must not leak harness stdout into CANDIDATE_ID=$(run_multi_candidate)."""
    content = _script_text()
    assert 'tee "$logfile" >/dev/null' in content


def test_phase10_gpu_run_allows_candidate_id_override() -> None:
    content = _script_text()
    assert "Reusing CANDIDATE_ID=" in content
    assert "${CANDIDATE_ID:-}" in content
