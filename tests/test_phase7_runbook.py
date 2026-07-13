"""CI guard: Phase 7 OPTIMIZATION_RUNBOOK.md documents Nextflow a6000 deployment."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

RUNBOOK = Path(".planning/phases/07-nextflow-pipeline-integration/OPTIMIZATION_RUNBOOK.md")

# The runbook is a gitignored post-migration (.planning/) artifact; skip the
# whole module on clean checkouts / CI where it is absent (convention K15).
pytestmark = pytest.mark.skipif(
    not RUNBOOK.exists(),
    reason=f"{RUNBOOK} not present (gitignored post-migration artifact); runbook guard skipped",
)

_BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def _runbook_bash_blocks() -> str:
    """Return concatenated bash fenced blocks (copy-pasteable commands only)."""
    text = RUNBOOK.read_text(encoding="utf-8")
    blocks = _BASH_BLOCK.findall(text)
    return "\n".join(blocks)


def _runbook_full_text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def test_runbook_documents_both_repo_sync() -> None:
    content = _runbook_bash_blocks()
    assert "/home/frans/code/linum-basic" in content
    assert "/home/frans/code/linumpy" in content


def test_runbook_documents_a6000_profile_launch() -> None:
    content = _runbook_bash_blocks()
    assert "-profile a6000" in content


def test_runbook_documents_fast_path_env() -> None:
    content = _runbook_bash_blocks()
    assert "LINUM_BASIC_DCT_KERNEL=tuned" in content


def test_runbook_documents_shared_inductor_cache() -> None:
    content = _runbook_bash_blocks()
    assert "TORCHINDUCTOR_CACHE_DIR" in content


def test_runbook_uses_frozen_phase5_baseline_id() -> None:
    content = _runbook_full_text()
    assert "baseline-20260701T020128-be1e880-sub-22" in content


def test_runbook_documents_dynamic_max_forks() -> None:
    content = _runbook_full_text()
    assert "gpuPinBlock" in content
    assert "gpuExposeAllBlock" in content


def test_runbook_documents_dual_timing() -> None:
    content = _runbook_full_text()
    assert "nextflow log" in content
    assert "operator_timing.end_to_end_ms" in content


def test_runbook_documents_resume_guidance() -> None:
    content = _runbook_bash_blocks()
    assert "-resume" in content


def test_runbook_omits_production_strategy_override() -> None:
    content = _runbook_bash_blocks()
    assert "--strategy multi" not in content
    assert "--strategy batched" not in content


def test_runbook_documents_integration_check_script() -> None:
    content = _runbook_full_text()
    assert "phase7_integration_check.sh" in content


def test_runbook_documents_regression_triage() -> None:
    content = _runbook_full_text()
    assert "pipeline" in content.lower() or "orchestration" in content.lower()
    assert "algorithm" in content.lower() or "env drift" in content.lower()


def test_runbook_documents_env_snapshot_fields() -> None:
    content = _runbook_full_text()
    assert "TORCHINDUCTOR_CACHE_DIR" in content
    assert "LINUM_BASIC_DCT_KERNEL" in content
    assert "CUDA_VISIBLE_DEVICES" in content


def test_runbook_references_uat_signoff() -> None:
    content = _runbook_full_text()
    assert "07-UAT.md" in content


def test_runbook_documents_post_config_change_workflow() -> None:
    content = _runbook_full_text()
    assert "integration sign-off" in content.lower()
    assert "before merge to dev" in content.lower() or "after every" in content.lower()
