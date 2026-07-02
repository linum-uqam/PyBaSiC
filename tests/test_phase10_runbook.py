"""CI guard: 10-RUNBOOK.md and 10-UAT.md Phase 10 planning artifacts."""

from __future__ import annotations

import re
from pathlib import Path

RUNBOOK = Path(".planning/phases/10-v1-0-debt-closure/10-RUNBOOK.md")
UAT = Path(".planning/phases/10-v1-0-debt-closure/10-UAT.md")
CLOSURE = Path(".planning/phases/10-v1-0-debt-closure/10-DEBT-CLOSURE.md")

_BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def _runbook_bash_blocks() -> str:
    """Return concatenated bash fenced blocks (copy-pasteable commands only)."""
    text = RUNBOOK.read_text(encoding="utf-8")
    blocks = _BASH_BLOCK.findall(text)
    return "\n".join(blocks)


def _runbook_full_text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def test_runbook_documents_nflo_artifact_filenames() -> None:
    content = _runbook_full_text()
    assert "nflo03-inductor-cache-audit.json" in content
    assert "nflo04-wallclock-reconciliation.json" in content
    assert "scripts/phase10_gpu_run.sh" in content


def test_runbook_documents_shared_inductor_cache_in_bash_blocks() -> None:
    content = _runbook_bash_blocks()
    assert "TORCHINDUCTOR_CACHE_DIR" in content
    assert "inductor-cache" in content


def test_runbook_documents_strategy_auto_in_bash_blocks() -> None:
    content = _runbook_bash_blocks()
    assert "--strategy auto" in content


def test_runbook_documents_canonical_workload() -> None:
    content = _runbook_full_text()
    assert "128" in content
    assert "500" in content
    assert "baseline-20260701T020128-be1e880-sub-22" in content


def test_runbook_documents_nflo04_tolerance_and_operator_timing() -> None:
    content = _runbook_full_text()
    assert "10%" in content
    assert "operator_timing" in content
    assert "end_to_end_ms" in content


def test_runbook_omits_z_selection_with_baseline_id() -> None:
    content = _runbook_bash_blocks()
    z_opt = "--" + "z-"
    assert f"{z_opt}indices" not in content
    assert f"{z_opt}sample" not in content


def test_runbook_references_phase10_gpu_script() -> None:
    content = _runbook_full_text()
    assert "scripts/phase10_gpu_run.sh" in content


def test_uat_cites_phase6_concurrency_verdict() -> None:
    content = UAT.read_text(encoding="utf-8")
    assert "phase6-concurrency-verdict.json" in content
    assert "maxForks" in content


def test_uat_documents_batched_rejection_and_max_forks() -> None:
    content = UAT.read_text(encoding="utf-8")
    assert "batched" in content.lower()
    assert "reject" in content.lower()
    assert "maxForks=2" in content


def test_uat_includes_operator_signoff() -> None:
    content = UAT.read_text(encoding="utf-8")
    assert "Signed by:" in content


def test_uat_closure_summary_references_all_debt_ids() -> None:
    content = CLOSURE.read_text(encoding="utf-8")
    for debt_id in ("DEBT-01", "DEBT-02", "DEBT-03", "DEBT-04", "DEBT-05"):
        assert debt_id in content
