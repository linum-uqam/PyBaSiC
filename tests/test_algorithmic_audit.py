"""Unit tests for Phase 8.1 algorithmic foundation audit (ALGO-01..04)."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pytest

from linum_basic.benchmark import (
    AuditFinding,
    MicroBenchmarkResult,
    TraceabilityRow,
    build_traceability_matrix,
)
from linum_basic.benchmark.audit import (
    _SBH_SIMULATOR_AVAILABLE,
    AUDIT_MAX_REWEIGHTING_ITERATIONS,
    MAX_N,
    MAX_WS,
    MIN_N,
    MIN_WS,
    RECOVERY_CORRELATION_THRESHOLD,
    _fit_with_audit_caps,
    make_handrolled_stack,
    make_sbh_stack,
    run_micro_benchmark,
)
from linum_basic.benchmark.audit import (
    AuditFinding as AuditFindingDirect,
)
from linum_basic.benchmark.audit import (
    MicroBenchmarkResult as MicroBenchmarkResultDirect,
)
from linum_basic.benchmark.audit import (
    TraceabilityRow as TraceabilityRowDirect,
)
from linum_basic.benchmark.audit import (
    build_traceability_matrix as build_traceability_matrix_direct,
)

# ---------------------------------------------------------------------------
# Task 1: dataclass scaffold (import, construction, validation)
# ---------------------------------------------------------------------------


def test_import_from_audit_module() -> None:
    """Direct audit submodule exports are importable."""
    assert TraceabilityRowDirect is TraceabilityRow
    assert AuditFindingDirect is AuditFinding
    assert MicroBenchmarkResultDirect is MicroBenchmarkResult
    assert build_traceability_matrix_direct is build_traceability_matrix


def test_import_from_benchmark_package() -> None:
    """Audit symbols re-export through linum_basic.benchmark."""
    from linum_basic.benchmark import (
        AuditFinding,
        MicroBenchmarkResult,
        TraceabilityRow,
        build_traceability_matrix,
    )

    assert TraceabilityRow is not None
    assert AuditFinding is not None
    assert MicroBenchmarkResult is not None
    assert build_traceability_matrix is not None


def test_traceability_row_construction_and_immutability() -> None:
    row = TraceabilityRow(
        paper_step="x",
        paper_ref="Eq. 6",
        code_refs=("core.py:1",),
        status="MATCH",
        intentional=False,
        notes="",
    )
    assert row.paper_step == "x"
    assert row.status == "MATCH"
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.status = "GAP"  # type: ignore[misc]


def test_audit_finding_construction() -> None:
    finding = AuditFinding(
        finding_id="sorted-stack-full-copy",
        rank=1,
        description="np.sort duplicates entire working stack",
        fix_class="allocation",
        measured_cost_ms=0.0,
        measured_cost_pct=0.0,
        paper_ref="Eq. 4-5",
        code_refs=("linum_basic/core.py:348",),
        phase9_change_class="algorithm",
        phase11_priority="high",
        suggested_action="Investigate in-place views",
    )
    assert finding.finding_id == "sorted-stack-full-copy"
    assert finding.fix_class == "allocation"


def test_micro_benchmark_result_construction() -> None:
    result = MicroBenchmarkResult(
        name="handrolled-ws64",
        working_size=64,
        n_images=12,
        backend="numpy",
        max_reweighting_iterations=5,
        prepare_ms=10.0,
        run_ms=100.0,
        normalize_ms=5.0,
        peak_bytes=1_000_000,
    )
    assert result.working_size == 64
    assert result.backend == "numpy"


def test_traceability_row_invalid_status_raises() -> None:
    with pytest.raises(ValueError, match="status"):
        TraceabilityRow(
            paper_step="x",
            paper_ref="Eq. 6",
            code_refs=("core.py:1",),
            status="UNKNOWN",
            intentional=False,
            notes="",
        )


def test_audit_finding_invalid_fix_class_raises() -> None:
    with pytest.raises(ValueError, match="fix_class"):
        AuditFinding(
            finding_id="x",
            rank=1,
            description="d",
            fix_class="invalid",
            measured_cost_ms=0.0,
            measured_cost_pct=0.0,
            paper_ref="Eq. 6",
            code_refs=("core.py:1",),
            phase9_change_class="algorithm",
            phase11_priority="low",
            suggested_action="none",
        )


# ---------------------------------------------------------------------------
# Task 2: traceability matrix (ALGO-01)
# ---------------------------------------------------------------------------

REQUIRED_CONTEXT_STEPS = [
    "Measurement matrix",
    "Low-rank",
    "Eq. 6",
    "Two-step B",
    "Reweighted L1",
    "Auto",
    "Correction",
]

INVARIANT_KEYWORDS = [
    "B1 monotonicity",
    "dual shrink",
    "OpenCV",
    "power-iteration SVD",
]


def test_traceability_matrix() -> None:
    matrix = build_traceability_matrix()
    assert len(matrix) > 0
    for step in REQUIRED_CONTEXT_STEPS:
        assert any(step in row.paper_step for row in matrix), f"missing CONTEXT step: {step}"
    for row in matrix:
        assert row.code_refs
        assert row.status in {"MATCH", "INTENTIONAL", "GAP", "PARTIAL"}


def test_two_step_b_gap() -> None:
    matrix = build_traceability_matrix()
    eq8_rows = [r for r in matrix if "Two-step B" in r.paper_step]
    assert len(eq8_rows) == 1
    row = eq8_rows[0]
    assert row.status == "GAP"
    assert row.intentional is False


def test_agents_invariants_intentional() -> None:
    matrix = build_traceability_matrix()
    invariant_rows = [r for r in matrix if any(kw in r.paper_step for kw in INVARIANT_KEYWORDS)]
    assert len(invariant_rows) == 4
    for row in invariant_rows:
        assert row.status == "INTENTIONAL"
        assert row.intentional is True


# ---------------------------------------------------------------------------
# Task 1 (Plan 02): synthetic ground-truth generators (ALGO-02)
# ---------------------------------------------------------------------------


def test_handrolled_stack_shape_and_dtype() -> None:
    stack, flatfield, darkfield = make_handrolled_stack(n=12, ws=64)
    assert stack.shape == (12, 64, 64)
    assert stack.dtype == np.float32
    assert flatfield.shape == (64, 64)
    assert flatfield.dtype == np.float32
    assert darkfield is None


def test_handrolled_stack_mean_normalised_flatfield() -> None:
    _, flatfield, _ = make_handrolled_stack(n=12, ws=64)
    assert abs(float(flatfield.mean()) - 1.0) < 1e-5


def test_handrolled_stack_deterministic() -> None:
    stack_a, ff_a, _ = make_handrolled_stack(n=12, ws=64, seed=42)
    stack_b, ff_b, _ = make_handrolled_stack(n=12, ws=64, seed=42)
    np.testing.assert_array_equal(stack_a, stack_b)
    np.testing.assert_array_equal(ff_a, ff_b)


def test_handrolled_stack_different_seeds_differ() -> None:
    stack_a, _, _ = make_handrolled_stack(n=12, ws=64, seed=0)
    stack_b, _, _ = make_handrolled_stack(n=12, ws=64, seed=1)
    assert not np.array_equal(stack_a, stack_b)


def test_handrolled_stack_darkfield_when_requested() -> None:
    _, _, darkfield = make_handrolled_stack(n=12, ws=64, estimate_darkfield=True)
    assert darkfield is not None
    assert darkfield.shape == (64, 64)
    assert darkfield.dtype == np.float32


@pytest.mark.parametrize("n", [MIN_N - 1, MAX_N + 1])
def test_handrolled_stack_rejects_out_of_bound_n(n: int) -> None:
    with pytest.raises(ValueError, match="n"):
        make_handrolled_stack(n=n, ws=64)


@pytest.mark.parametrize("ws", [MIN_WS - 1, MAX_WS + 1])
def test_handrolled_stack_rejects_out_of_bound_ws(ws: int) -> None:
    with pytest.raises(ValueError, match=r"working_size|ws"):
        make_handrolled_stack(n=12, ws=ws)


@pytest.mark.skipif(
    not _SBH_SIMULATOR_AVAILABLE,
    reason="sbh-simulator not installed",
)
def test_sbh_stack_shape_and_dtype() -> None:
    stack, flatfield, darkfield = make_sbh_stack(n=12, ws=64)
    assert stack.shape == (12, 64, 64)
    assert stack.dtype == np.float32
    assert flatfield.shape == (64, 64)
    assert darkfield is not None
    assert darkfield.shape == (64, 64)


# ---------------------------------------------------------------------------
# Task 2 (Plan 02): micro-benchmark runner (ALGO-02)
# ---------------------------------------------------------------------------

_MICRO_BENCHMARK_WALL_LIMIT_S = 30.0


def test_micro_benchmark_result_fields() -> None:
    stack, _, _ = make_handrolled_stack(n=12, ws=64)
    result = run_micro_benchmark(stack, working_size=64, estimate_darkfield=False)
    assert result.backend == "numpy"
    assert result.n_images == 12
    assert result.working_size == 64
    assert result.max_reweighting_iterations == AUDIT_MAX_REWEIGHTING_ITERATIONS
    assert result.prepare_ms >= 0.0
    assert result.run_ms >= 0.0
    assert result.normalize_ms >= 0.0
    assert np.isfinite(result.prepare_ms)
    assert np.isfinite(result.run_ms)
    assert np.isfinite(result.normalize_ms)
    assert result.peak_bytes > 0


def test_micro_benchmark_rejects_out_of_bounds_stack() -> None:
    stack, _, _ = make_handrolled_stack(n=MIN_N, ws=64)
    small_stack = stack[: MIN_N - 1]
    with pytest.raises(ValueError, match="n"):
        run_micro_benchmark(small_stack, working_size=64)


def test_micro_benchmark_runtime() -> None:
    stack, _, _ = make_handrolled_stack(n=12, ws=64)
    t0 = time.perf_counter()
    run_micro_benchmark(stack, working_size=64, estimate_darkfield=False)
    elapsed = time.perf_counter() - t0
    assert elapsed < _MICRO_BENCHMARK_WALL_LIMIT_S


def test_recovery_quality() -> None:
    stack, flatfield_gt, _ = make_handrolled_stack(n=12, ws=64, estimate_darkfield=False)
    model = _fit_with_audit_caps(stack, working_size=64, estimate_darkfield=False)
    estimated = model.get_flatfield()
    correlation = float(np.corrcoef(estimated.ravel(), flatfield_gt.ravel())[0, 1])
    assert correlation >= RECOVERY_CORRELATION_THRESHOLD, f"Flat-field correlation too low: {correlation:.3f}"


# ---------------------------------------------------------------------------
# Task 1 (Plan 03): profile_call wrapper (ALGO-03)
# ---------------------------------------------------------------------------


def test_profile_call() -> None:
    from linum_basic.benchmark.audit import profile_call

    def _sum_squares(arr: np.ndarray) -> float:
        return float(np.sum(arr * arr))

    data = np.arange(1000, dtype=np.float64)

    out = profile_call(_sum_squares, data)
    assert set(out.keys()) == {"wall_ms", "peak_bytes", "pstats_top", "result"}
    assert isinstance(out["wall_ms"], float)
    assert out["wall_ms"] >= 0.0
    assert np.isfinite(out["wall_ms"])
    assert isinstance(out["peak_bytes"], int)
    assert out["peak_bytes"] > 0
    assert isinstance(out["pstats_top"], str)
    assert len(out["pstats_top"]) > 0
    assert "cumtime" in out["pstats_top"].lower() or "ncalls" in out["pstats_top"].lower()
    assert out["result"] == _sum_squares(data)

    # tracemalloc must be stopped between calls
    out2 = profile_call(_sum_squares, data)
    assert out2["result"] == _sum_squares(data)


# ---------------------------------------------------------------------------
# Task 2 (Plan 03): profile_prepare_run per-phase hotspots (ALGO-03)
# ---------------------------------------------------------------------------

_PROFILE_PREPARE_RUN_WALL_LIMIT_S = 30.0


def test_profile_hotspots() -> None:
    from linum_basic.benchmark.audit import FIX_CLASSES, profile_prepare_run

    stack, _, _ = make_handrolled_stack(n=12, ws=64, estimate_darkfield=False)

    t0 = time.perf_counter()
    profile = profile_prepare_run(stack, working_size=64, estimate_darkfield=False)
    elapsed = time.perf_counter() - t0
    assert elapsed < _PROFILE_PREPARE_RUN_WALL_LIMIT_S

    for phase in ("prepare", "run", "normalize"):
        assert phase in profile
        phase_data = profile[phase]
        assert "wall_ms" in phase_data
        assert "peak_bytes" in phase_data
        assert "hotspots" in phase_data
        assert phase_data["wall_ms"] >= 0.0
        assert np.isfinite(phase_data["wall_ms"])
        assert phase_data["peak_bytes"] > 0
        assert isinstance(phase_data["hotspots"], list)

    run_hotspots = profile["run"]["hotspots"]
    assert len(run_hotspots) > 0
    run_functions = " ".join(h["function"] for h in run_hotspots).lower()
    assert "dct" in run_functions or "_alm_core_step" in run_functions

    for phase in ("prepare", "run", "normalize"):
        for hotspot in profile[phase]["hotspots"]:
            assert hotspot["fix_class"] in FIX_CLASSES


# ---------------------------------------------------------------------------
# Task 1 (Plan 04): build_audit_report schema (ALGO-04)
# ---------------------------------------------------------------------------

AUDIT_REPORT_TOP_KEYS = {
    "schema_version",
    "git_commit",
    "traceability",
    "micro_benchmarks",
    "profile_hotspots",
    "ranked_findings",
    "phase9_handoff",
}

FINDING_FIELD_KEYS = {
    "finding_id",
    "rank",
    "description",
    "fix_class",
    "measured_cost_ms",
    "measured_cost_pct",
    "paper_ref",
    "code_refs",
    "phase9_change_class",
    "phase11_priority",
    "suggested_action",
}

PHASE9_HANDOFF_KEYS = {"forensics_hypotheses", "fore02_change_classes", "deferred_to_phase9"}

MARKDOWN_SECTION_MARKERS = [
    "## Executive summary",
    "## Traceability matrix",
    "## Micro-benchmark results",
    "## Profiling hot paths",
    "## Ranked findings",
    "## Phase 9 handoff",
    "## Intentional invariants appendix",
]


def _sample_audit_inputs() -> tuple[list, list, dict]:
    traceability = build_traceability_matrix()
    stack, _, _ = make_handrolled_stack(n=12, ws=64, estimate_darkfield=False)
    micro = [run_micro_benchmark(stack, working_size=64, estimate_darkfield=False)]
    from linum_basic.benchmark.audit import profile_prepare_run

    profile = profile_prepare_run(stack, working_size=64, estimate_darkfield=False)
    return traceability, micro, profile


def test_audit_report_schema() -> None:
    from linum_basic.benchmark.audit import build_audit_report
    from linum_basic.benchmark.metadata import collect_git_commit

    traceability, micro, profile = _sample_audit_inputs()
    report = build_audit_report(traceability, micro, profile)

    assert set(report.keys()) == AUDIT_REPORT_TOP_KEYS
    assert report["schema_version"] == "8.1.0"
    assert report["git_commit"] == collect_git_commit()
    assert report["git_commit"] != ""
    assert report["git_commit"] != "unknown" or collect_git_commit() == "unknown"

    assert isinstance(report["traceability"], list)
    assert isinstance(report["micro_benchmarks"], list)
    assert isinstance(report["profile_hotspots"], list)
    for row in report["traceability"]:
        assert isinstance(row, dict)
        assert "paper_step" in row
        assert "status" in row

    findings = report["ranked_findings"]
    assert isinstance(findings, list)
    assert len(findings) >= 2
    ranks = [f["rank"] for f in findings]
    assert ranks == sorted(ranks)
    for finding in findings:
        assert set(finding.keys()) == FINDING_FIELD_KEYS
        assert finding["fix_class"] in {"algorithm", "data_layout", "backend", "allocation"}

    eq8 = [f for f in findings if f["finding_id"] == "missing-eq8-two-step-b"]
    assert len(eq8) == 1
    assert eq8[0]["fix_class"] == "algorithm"
    assert eq8[0]["phase9_change_class"] == "algorithm"
    assert eq8[0]["phase11_priority"] == "high"

    reweight = [f for f in findings if f["finding_id"] == "reweighting-iter-production-500"]
    assert len(reweight) == 1
    assert reweight[0]["phase9_change_class"] == "harness_defaults"

    for finding in findings:
        desc = finding["description"].lower()
        fid = finding["finding_id"].lower()
        assert "b1" not in fid
        assert "dual shrink" not in desc
        assert "opencv" not in desc or "transpose" not in desc
        assert "power-iteration" not in desc and "svd" not in fid

    handoff = report["phase9_handoff"]
    assert set(handoff.keys()) == PHASE9_HANDOFF_KEYS
    assert isinstance(handoff["forensics_hypotheses"], list)
    assert len(handoff["forensics_hypotheses"]) > 0
    assert "algorithm" in handoff["fore02_change_classes"]
    assert any("basicpy" in item.lower() for item in handoff["deferred_to_phase9"])

    traceability[0] = dataclasses.replace(traceability[0], notes="mutated")
    micro[0] = dataclasses.replace(micro[0], name="mutated")
    profile["prepare"]["wall_ms"] = -1.0
    assert report["traceability"][0]["notes"] != "mutated"
    assert report["micro_benchmarks"][0]["name"] != "mutated"
    assert report["profile_hotspots"]  # unchanged snapshot


# ---------------------------------------------------------------------------
# Task 2 (Plan 04): write_audit_artifacts + full pipeline (ALGO-04)
# ---------------------------------------------------------------------------


def test_audit_report_artifacts(tmp_path: Path) -> None:
    from linum_basic.benchmark.audit import build_audit_report, write_audit_artifacts

    traceability, micro, profile = _sample_audit_inputs()
    report = build_audit_report(traceability, micro, profile)
    allowed = tmp_path.resolve()
    out_dir = allowed / "audit-out"
    write_audit_artifacts(out_dir, report, allowed_base=allowed)

    json_path = out_dir / "audit-report.json"
    md_path = out_dir / "audit-report.md"
    assert json_path.is_file()
    assert md_path.is_file()

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded == report
    assert json_path.read_text(encoding="utf-8").endswith("\n")

    md_text = md_path.read_text(encoding="utf-8")
    for marker in MARKDOWN_SECTION_MARKERS:
        assert marker in md_text

    escape_dir = allowed / "nested" / ".." / ".." / "escape-audit"
    with pytest.raises(ValueError, match=r"escape|outside|confined|base"):
        write_audit_artifacts(escape_dir, report, allowed_base=allowed)


def test_full_audit_pipeline(tmp_path: Path) -> None:
    from linum_basic.benchmark import (
        build_audit_report,
        build_traceability_matrix,
        make_handrolled_stack,
        profile_prepare_run,
        run_micro_benchmark,
        write_audit_artifacts,
    )

    traceability = build_traceability_matrix()
    stack, _, _ = make_handrolled_stack(n=12, ws=64, estimate_darkfield=False)
    micro = run_micro_benchmark(stack, working_size=64, estimate_darkfield=False)
    profile = profile_prepare_run(stack, working_size=64, estimate_darkfield=False)
    report = build_audit_report(traceability, [micro], profile)

    assert report["schema_version"] == "8.1.0"
    assert len(report["ranked_findings"]) >= 2

    allowed = tmp_path.resolve()
    write_audit_artifacts(allowed / "reports", report, allowed_base=allowed)
    assert (allowed / "reports" / "audit-report.json").exists()
    assert (allowed / "reports" / "audit-report.md").exists()
