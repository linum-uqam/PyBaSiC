"""CI guard: OPTIMIZATION_RUNBOOK.md keeps production-faithful profile flags."""

from __future__ import annotations

import re
from pathlib import Path

RUNBOOK = Path(".planning/phases/03-ws-128-optimization/OPTIMIZATION_RUNBOOK.md")
UAT = Path(".planning/phases/03-ws-128-optimization/03-UAT.md")

_BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def _runbook_bash_blocks() -> str:
    """Return concatenated bash fenced blocks (copy-pasteable commands only)."""
    text = RUNBOOK.read_text(encoding="utf-8")
    blocks = _BASH_BLOCK.findall(text)
    return "\n".join(blocks)


def test_runbook_production_faithful_profile_flags() -> None:
    content = _runbook_bash_blocks()
    assert " profile " in content or " profile \\" in content
    assert "--working-size 128" in content
    assert "--max-reweighting-iterations 500" in content
    assert "baseline-20260630T163351-e7c47a4-sub-22" in content
    assert "sub-22" in content
    assert "--mode sequential" in content


def test_runbook_omits_z_selection_overrides() -> None:
    content = _runbook_bash_blocks()
    z_opt = "--" + "z-"
    assert f"{z_opt}indices" not in content
    assert f"{z_opt}sample" not in content


def test_runbook_default_reweighting_not_15() -> None:
    content = _runbook_bash_blocks()
    assert "--max-reweighting-iterations 15" not in content


def test_runbook_documents_batched_diagnostic_after_sequential() -> None:
    content = _runbook_bash_blocks()
    assert "--mode batched-diagnostic" in content
    sequential_pos = content.index("--mode sequential")
    batched_pos = content.index("--mode batched-diagnostic")
    assert sequential_pos < batched_pos


def test_runbook_documents_lever_candidate_commands() -> None:
    content = _runbook_bash_blocks()
    assert " candidate " in content or " candidate \\" in content
    assert "lever-stack.json" in content
    assert "lever-sync-cadence" in content
    assert "lever-reweighting-tolerance" in content
    assert "lever-tile-subsampling" in content
    assert "convergence_check_every" in content
    assert "reweighting_tolerance" in content
    assert "tile_subsample_ratio" in content


def test_runbook_documents_per_lever_pytest_gate_stack() -> None:
    content = _runbook_bash_blocks()
    assert "tests/test_alm_parity.py" in content
    assert "tests/test_backend_parity.py" in content
    assert "tests/test_darkfield.py" in content
    assert "tests/test_vignette_validation.py" in content
    assert "tests/test_benchmark_quality.py" in content


def test_runbook_documents_convergence_telemetry_prerequisite() -> None:
    content = RUNBOOK.read_text(encoding="utf-8")
    assert "reweight_iterations_median" in content
    assert "telemetry_missing" in content
    assert "outer-loop" in content.lower() or "outer loop" in content.lower()


def test_runbook_documents_inductor_cache_phase5_deferral() -> None:
    content = RUNBOOK.read_text(encoding="utf-8")
    assert "inductor-cache-warm-policy" in content
    assert "Phase 5" in content
    assert "ALGO-01" in content


def test_runbook_documents_optimize_aggregate_step() -> None:
    content = _runbook_bash_blocks()
    assert " optimize " in content or " optimize \\" in content
    assert "lever-attempt-table.json" in content
    assert "phase5-backlog.json" in content
    assert "phase3-handoff-config.json" in content


def test_uat_template_includes_promote_reject_table_and_pytest_column() -> None:
    content = UAT.read_text(encoding="utf-8")
    assert "pytest_gate_passed" in content
    assert "baseline-20260630T163351-e7c47a4-sub-22" in content
    assert "phase5-backlog" in content.lower() or "Phase 5 backlog" in content
    assert "30%" in content or "aspirational" in content.lower()
    assert "inductor-cache-warm-policy" in content
