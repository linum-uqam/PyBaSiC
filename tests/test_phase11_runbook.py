"""CI guard: 11-RUNBOOK.md and 11-UAT.md Phase 11 planning artifacts."""

from __future__ import annotations

import re
from pathlib import Path

RUNBOOK = Path(".planning/phases/11-ws-128-speed-push/11-RUNBOOK.md")
UAT = Path(".planning/phases/11-ws-128-speed-push/11-UAT.md")

_BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def _runbook_bash_blocks() -> str:
    """Return concatenated bash fenced blocks (copy-pasteable commands only)."""
    text = RUNBOOK.read_text(encoding="utf-8")
    blocks = _BASH_BLOCK.findall(text)
    return "\n".join(blocks)


def _runbook_full_text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def test_runbook_documents_worker_compile_off_lever() -> None:
    content = _runbook_full_text()
    assert "worker-compile-off" in content
    assert "511c88c" in content
    assert "LINUM_BASIC_ALM_COMPILE_MODE=off" in content


def test_runbook_documents_ws128_500_iter_workload() -> None:
    content = _runbook_full_text()
    assert "working_size" in content
    assert "128" in content
    assert "500" in content
    assert "0,13,27,40,54" in content


def test_runbook_documents_env_levers_in_bash_blocks() -> None:
    content = _runbook_bash_blocks()
    assert "LINUM_BASIC_DCT_KERNEL=tuned" in content
    assert "LINUM_BASIC_ALM_COMPILE_MODE=off" in content


def test_runbook_documents_three_subjects() -> None:
    content = _runbook_full_text()
    for subject_id in ("sub-18", "sub-21", "sub-22"):
        assert subject_id in content


def test_runbook_documents_reject_outcome() -> None:
    content = _runbook_full_text()
    assert "REJECT" in content
    assert "all_subjects_promotion_eligible: false" in content


def test_runbook_documents_optimize_command() -> None:
    content = _runbook_bash_blocks()
    assert " optimize " in content or " optimize \\" in content
    assert "baseline-20260701T020128-be1e880-sub-22" in content
    assert "candidate-artifact.json" in content


def test_runbook_documents_phase5_backlog_rejection() -> None:
    content = _runbook_full_text()
    assert "phase5-backlog.json" in content
    assert '"status": "rejected"' in content or '"status": "rejected"' in content.replace("'", '"')
    assert "worker-compile-off" in content


def test_runbook_documents_steady_state_primary_metric() -> None:
    content = _runbook_full_text()
    assert "steady_state_ms" in content
    assert "end_to_end_ms" in content


def test_runbook_documents_130_speed_threshold() -> None:
    content = _runbook_full_text()
    assert "1.30" in content


def test_runbook_references_phase11_gpu_script() -> None:
    content = _runbook_full_text()
    assert "scripts/phase11_gpu_run.sh" in content


def test_runbook_documents_code_retention_511c88c() -> None:
    content = _runbook_full_text()
    assert "Do not revert" in content or "not revert" in content.lower()
    assert "511c88c" in content


def test_runbook_documents_multisubject_summary_artifact() -> None:
    content = _runbook_full_text()
    assert "phase11-multisubject-summary.json" in content


def test_runbook_documents_bottleneck_fixture() -> None:
    content = _runbook_bash_blocks()
    assert "bottleneck-report-worker-compile-off.json" in content


def test_runbook_references_uat_signoff() -> None:
    content = _runbook_full_text()
    assert "11-UAT.md" in content


def test_uat_includes_per_subject_table() -> None:
    content = UAT.read_text(encoding="utf-8")
    for subject_id in ("sub-18", "sub-21", "sub-22"):
        assert subject_id in content
    assert "steady_state_ms" in content


def test_uat_includes_operator_signoff() -> None:
    content = UAT.read_text(encoding="utf-8")
    assert "Signed by:" in content
    assert "Operator sign-off" in content or "sign-off" in content.lower()


def test_uat_documents_reject_decision() -> None:
    content = UAT.read_text(encoding="utf-8")
    assert "REJECT" in content
    assert "all_subjects_promotion_eligible: false" in content


def test_uat_documents_code_retention() -> None:
    content = UAT.read_text(encoding="utf-8")
    assert "511c88c" in content
    assert "retained" in content.lower() or "None" in content


def test_uat_documents_promotion_eligible_field() -> None:
    content = UAT.read_text(encoding="utf-8")
    assert "Eligible" in content or "promotion_eligible" in content
