"""CI guard: Phase 6 OPTIMIZATION_RUNBOOK.md documents concurrency A/B workflow."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

RUNBOOK = Path(".planning/phases/06-production-concurrency-decision/OPTIMIZATION_RUNBOOK.md")

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


def test_runbook_documents_concurrency_subcommand() -> None:
    content = _runbook_bash_blocks()
    assert " concurrency " in content or " concurrency \\" in content


def test_runbook_documents_multi_and_batched_strategy_arms() -> None:
    content = _runbook_bash_blocks()
    assert "--strategy multi" in content
    assert "--strategy batched" in content


def test_runbook_documents_phase6_concurrency_verdict() -> None:
    content = _runbook_bash_blocks()
    assert "phase6-concurrency-verdict.json" in content


def test_runbook_documents_phase5_fast_path_dct_kernel() -> None:
    content = _runbook_bash_blocks()
    assert "LINUM_BASIC_DCT_KERNEL=tuned" in content


def test_runbook_uses_frozen_phase5_baseline_id() -> None:
    content = _runbook_full_text()
    assert "baseline-20260701T020128-be1e880-sub-22" in content


def test_runbook_omits_z_selection_overrides() -> None:
    content = _runbook_bash_blocks()
    z_opt = "--" + "z-"
    assert f"{z_opt}indices" not in content
    assert f"{z_opt}sample" not in content


def test_runbook_documents_shared_inductor_cache() -> None:
    content = _runbook_bash_blocks()
    assert "TORCHINDUCTOR_CACHE_DIR" in content
    assert "inductor-cache" in content


def test_runbook_documents_fork_model_mapping() -> None:
    content = _runbook_full_text()
    assert "maxForks" in content
    assert "multi" in content.lower()
    assert "batched" in content.lower()


def test_runbook_documents_pytest_gate_stack() -> None:
    content = _runbook_bash_blocks()
    assert "tests/test_alm_parity.py" in content
    assert "tests/test_backend_parity.py" in content
    assert "tests/test_darkfield.py" in content
    assert "tests/test_vignette_validation.py" in content
    assert "tests/test_benchmark_quality.py" in content
