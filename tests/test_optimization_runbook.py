"""CI guard: OPTIMIZATION_RUNBOOK.md keeps production-faithful profile flags."""

from __future__ import annotations

import re
from pathlib import Path

RUNBOOK = Path(".planning/phases/05-algorithm-runtime-minimization/OPTIMIZATION_RUNBOOK.md")
UAT = Path(".planning/phases/05-algorithm-runtime-minimization/05-UAT.md")

_BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def _runbook_bash_blocks() -> str:
    """Return concatenated bash fenced blocks (copy-pasteable commands only)."""
    text = RUNBOOK.read_text(encoding="utf-8")
    blocks = _BASH_BLOCK.findall(text)
    return "\n".join(blocks)


def _runbook_full_text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def test_runbook_wave0_500_iter_rebaseline() -> None:
    content = _runbook_bash_blocks()
    assert " baseline " in content or " baseline \\" in content
    assert "--max-reweighting-iterations 500" in content
    assert "--working-size 128" in content
    assert "sub-22" in content


def test_runbook_baseline_id_must_use_500_iter_bundle() -> None:
    content = _runbook_full_text()
    assert "attempt_table.baseline_id" in content
    assert "500-iter" in content.lower() or "500 iter" in content.lower()
    assert "baseline-20260630T163351-e7c47a4-sub-22" in content


def test_runbook_omits_z_selection_overrides() -> None:
    content = _runbook_bash_blocks()
    z_opt = "--" + "z-"
    assert f"{z_opt}indices" not in content
    assert f"{z_opt}sample" not in content


def test_runbook_default_reweighting_not_15() -> None:
    content = _runbook_bash_blocks()
    assert "--max-reweighting-iterations 15" not in content


def test_runbook_documents_two_tier_validation_gate() -> None:
    content = _runbook_full_text()
    assert "two-tier" in content.lower() or "Two-tier" in content
    assert "Tier 1" in content or "Tier 2" in content
    assert "--max-reweighting-iterations 25" in _runbook_bash_blocks()
    assert "500-iter" in content.lower() or "500 iter" in content.lower()


def test_runbook_documents_code_path_first_lever_order() -> None:
    content = _runbook_full_text()
    table_start = content.index("## Code-path-first lever order")
    table_section = content[table_start:]
    dct_pos = table_section.index("dct-kernel-tuning")
    inductor_pos = table_section.index("inductor-cache-warm-policy")
    sync_pos = table_section.index("sync-cadence")
    tile_pos = table_section.index("tile-subsampling")
    assert dct_pos < inductor_pos < sync_pos < tile_pos
    assert "code-path" in content.lower() or "Code-path" in content


def test_runbook_documents_profile_bottleneck_refresh() -> None:
    content = _runbook_bash_blocks()
    assert " profile " in content or " profile \\" in content
    assert "--mode sequential" in content
    assert "bottleneck-report.json" in _runbook_full_text()


def test_runbook_documents_per_lever_pytest_gate_stack() -> None:
    content = _runbook_bash_blocks()
    assert "tests/test_alm_parity.py" in content
    assert "tests/test_backend_parity.py" in content
    assert "tests/test_darkfield.py" in content
    assert "tests/test_vignette_validation.py" in content
    assert "tests/test_benchmark_quality.py" in content


def test_runbook_documents_pytest_pass_alone_does_not_promote() -> None:
    content = _runbook_full_text()
    assert "pytest pass alone does not promote" in content.lower() or (
        "pytest" in content.lower() and "seam" in content.lower()
    )


def test_runbook_documents_d08_backlog_inherit_extend() -> None:
    content = _runbook_full_text()
    assert "phase5-backlog.json" in content
    assert "inherit" in content.lower() or "Extend" in content or "extend" in content
    assert "reject" in content.lower() or "blocked" in content.lower()
    assert "prior" in content.lower() or "Phase 3" in content


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
    assert "convergence_check_every" in content


def test_runbook_documents_optimize_aggregate_step() -> None:
    content = _runbook_bash_blocks()
    assert " optimize " in content or " optimize \\" in content
    assert "lever-attempt-table.json" in content
    assert "phase5-backlog.json" in content
    assert "phase5-fast-path.json" in content


def test_runbook_documents_ws128_target_with_phase2_sweep() -> None:
    content = _runbook_full_text()
    assert "working_size=128" in content or "working_size=128" in content.replace(" ", "")
    assert "PERF-05" in content or "Phase 2" in content
    assert "ws64" in content.lower() or "ws 64" in content.lower() or "64" in content
    assert "ws96" in content.lower() or "ws 96" in content.lower() or "96" in content


def test_uat_template_includes_promote_reject_table_and_dual_timing() -> None:
    content = UAT.read_text(encoding="utf-8")
    assert "pytest_gate_passed" in content
    assert "steady_state_ms" in content
    assert "end_to_end_ms" in content or "end-to-end" in content.lower()
    assert "phase5-backlog" in content.lower() or "phase5-backlog.json" in content
    assert "phase5-fast-path" in content.lower() or "phase5-fast-path.json" in content
    assert "30%" in content or "aspirational" in content.lower()
    assert "Signed by:" in content or "Sign-off" in content
