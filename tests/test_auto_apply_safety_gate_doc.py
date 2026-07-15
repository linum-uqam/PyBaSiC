"""Structural regression test: lock the required sections of the auto-apply
safety gate design doc (M008/S01).

The design doc (``docs/auto_apply_safety_gate.md``) is the slice's sole
deliverable. Without a mechanical guard, a future edit could silently strip a
required section (e.g. the fallback-to-defaults contract or the D015
advisory-vs-release-gate distinction) and no CI signal would catch it,
defeating the purpose of the slice's success criteria.

This mirrors :mod:`tests.test_adaptive_working_size_doc.py`: a plain pytest
module asserting required headings exist in the git-tracked markdown doc, plus
drift-checks against the real code it cites
(``FIRST_CLASS_METRICS`` / ``CALIBRATION_POLICY`` / the quality primitives in
:mod:`linum_basic.benchmark.quality`, and ``recommend_bounds`` /
``BoundsRecommendation`` in :mod:`linum_basic.tuning`). The drift-checks catch
scope-narrowing or a renamed primitive, not just a missing heading.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOC = Path("docs/auto_apply_safety_gate.md")

# ``##``-level headings the design must contain. Matched as substrings of the
# full heading line, so the "(D023)" / "two full-volume fits" suffixes do not
# break the match.
REQUIRED_SECTIONS = [
    "Background and scope",
    "Gate mechanism",  # "Gate mechanism: fixed non-regression margin (D023)"
    "Cost bound",  # "Cost bound: two full-volume fits, no N-repeat calibration"
    "Load profile",
    "Failure modes and fallback-to-defaults contract",
    "Distinction from the advisory and release gates",
    "S02 integration contract",
    "Observability contract",
    "Negative tests",
    "Validation plan",
    "Non-goals",
]

# Quality primitives the gate mechanism must cite as REUSED (the K01-aligned
# metric definitions the gate composes, not reinvents).
REQUIRED_QUALITY_PRIMITIVES = [
    "compute_quality_report",
    "compute_deltas",
    "MetricDelta",
    "QualityReport",
    "FIRST_CLASS_METRICS",
]

# Release-gate machinery the distinction section must cite as the DISTINCT
# precedent (D015) the auto-apply gate deliberately does NOT use as its policy.
REQUIRED_RELEASE_GATE_SYMBOLS = [
    "CALIBRATION_POLICY",
    "calibrate_tolerances",
    "evaluate_quality_gate",
]

# Tuning surfaces the integration contract consumes as-is.
REQUIRED_TUNING_SYMBOLS = [
    "recommend_bounds",
    "BoundsRecommendation",
    "tune",
    "TuneResult",
]

# S02 integration-contract symbols the doc must specify.
REQUIRED_S02_CONTRACT = [
    "auto_tune",
    "AutoTuneResult",
    "--auto-apply",
    "fit_mosaic",
]

# Observability sub-dict keys the contract must record, mirroring the existing
# ``_strategy`` / ``_working_size_selector`` explainability pattern.
REQUIRED_OBSERVABILITY_KEYS = [
    "gate_verdict",
    "fallback_reason",
    "failing_metrics",
    "deltas",
]

# Fallback reasons enumerated in the failure-modes table -- the contract that
# lets auto-apply never ship a worse correction than the default-bounds fit.
REQUIRED_FALLBACK_REASONS = [
    "regression-detected",
    "degenerate-trial-history",
]

# Architectural decisions / repo rules the distinction and non-goals anchor to.
REQUIRED_DECISION_ANCHORS = ["D015", "D023", "K01", "K02"]


@pytest.fixture(scope="module")
def doc_text() -> str:
    """Full text of the design doc, loaded once for the module."""
    return DOC.read_text(encoding="utf-8")


def test_doc_exists_and_is_substantial() -> None:
    """The design artifact is present and non-trivially populated."""
    assert DOC.exists(), f"{DOC} is missing"
    text = DOC.read_text(encoding="utf-8")
    assert len(text) > 2000, "design doc is suspiciously short"
    assert text.strip(), "design doc is empty"


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_required_section_headings_present(doc_text: str, section: str) -> None:
    """Each must-have section exists as a markdown heading."""
    pattern = re.compile(rf"(?m)^#{{1,6}}\s+.*{re.escape(section)}.*$")
    assert pattern.search(doc_text), f"design doc is missing required section heading: {section!r}"


@pytest.mark.parametrize("symbol", REQUIRED_QUALITY_PRIMITIVES)
def test_doc_cites_quality_primitive(doc_text: str, symbol: str) -> None:
    """The gate mechanism must cite each reused quality primitive by name."""
    assert symbol in doc_text, f"doc must cite reused quality primitive {symbol!r}"


@pytest.mark.parametrize("symbol", REQUIRED_RELEASE_GATE_SYMBOLS)
def test_doc_distinguishes_release_gate_symbol(doc_text: str, symbol: str) -> None:
    """The distinction section must name the release-gate precedent it does NOT use."""
    assert symbol in doc_text, f"doc must distinguish release-gate symbol {symbol!r}"


@pytest.mark.parametrize("symbol", REQUIRED_TUNING_SYMBOLS)
def test_doc_cites_tuning_surface(doc_text: str, symbol: str) -> None:
    """The integration contract must name each tuning surface it consumes."""
    assert symbol in doc_text, f"doc must cite tuning surface {symbol!r}"


@pytest.mark.parametrize("symbol", REQUIRED_S02_CONTRACT)
def test_doc_specifies_s02_contract_symbol(doc_text: str, symbol: str) -> None:
    """The S02 integration contract names its new symbols and the reused fitter."""
    assert symbol in doc_text, f"doc must specify S02 contract symbol {symbol!r}"


@pytest.mark.parametrize("key", REQUIRED_OBSERVABILITY_KEYS)
def test_doc_specifies_observability_key(doc_text: str, key: str) -> None:
    """The observability sub-dict must define each explainability key."""
    assert key in doc_text, f"doc must define observability key {key!r}"


@pytest.mark.parametrize("reason", REQUIRED_FALLBACK_REASONS)
def test_doc_enumerates_fallback_reason(doc_text: str, reason: str) -> None:
    """Each fallback reason in the failure-modes table is enumerated."""
    assert reason in doc_text, f"doc must enumerate fallback reason {reason!r}"


@pytest.mark.parametrize("anchor", REQUIRED_DECISION_ANCHORS)
def test_doc_anchors_to_decision(doc_text: str, anchor: str) -> None:
    """The distinction and non-goals anchor to the locked decisions / repo rules."""
    assert anchor in doc_text, f"doc must anchor to decision/rule {anchor!r}"


# --- drift checks against the real cited code --------------------------------


def test_quality_primitives_exist_in_code() -> None:
    """The doc reuses linum_basic.benchmark.quality primitives -- verify they
    actually exist. If one is renamed or removed, the design is invalid and S02
    cannot reuse it; fail here rather than at implementation time."""
    from linum_basic.benchmark import quality

    for name in REQUIRED_QUALITY_PRIMITIVES:
        assert hasattr(quality, name), f"quality.{name} missing (design cites it)"
    for name in REQUIRED_RELEASE_GATE_SYMBOLS:
        assert hasattr(quality, name), f"quality.{name} missing (design cites it)"


def test_tuning_surfaces_exist_in_code() -> None:
    """recommend_bounds / BoundsRecommendation / tune / TuneResult exist."""
    from linum_basic import tuning

    for name in REQUIRED_TUNING_SYMBOLS:
        assert hasattr(tuning, name), f"tuning.{name} missing (design cites it)"


def test_doc_first_class_metrics_matches_code(doc_text: str) -> None:
    """The doc's stated FIRST_CLASS_METRICS tuple must equal the code tuple.

    Drift check: if the code tuple changes, the design must change with it, or
    the gate would test metrics that no longer exist in the code."""
    from linum_basic.benchmark.quality import FIRST_CLASS_METRICS

    code_tuple = tuple(FIRST_CLASS_METRICS)
    doc_match = re.search(r"FIRST_CLASS_METRICS\s*==\s*\(([^)]*)\)", doc_text)
    assert doc_match is not None, "doc must state FIRST_CLASS_METRICS == (...)"
    doc_tuple = tuple(m.strip().strip('"').strip("'") for m in doc_match.group(1).split(",") if m.strip())
    assert doc_tuple == code_tuple, f"doc FIRST_CLASS_METRICS {doc_tuple} != code {code_tuple}; drifted apart"


def test_doc_calibration_policy_matches_code(doc_text: str) -> None:
    """The doc must cite the exact CALIBRATION_POLICY value from the code.

    Drift check: if the release-gate policy string changes in the code, the
    design's distinction must reflect it."""
    from linum_basic.benchmark.quality import CALIBRATION_POLICY

    assert CALIBRATION_POLICY in doc_text, f"doc must cite the code CALIBRATION_POLICY value {CALIBRATION_POLICY!r}"


# --- contract-shape checks ---------------------------------------------------


def test_doc_states_two_full_volume_fit_cost_bound(doc_text: str) -> None:
    """The cost bound is exactly two full-volume fits, never an N-repeat
    mean+3std calibration -- the milestone's literal cost contract."""
    lower = doc_text.lower()
    assert "two full-volume fits" in lower, "doc must state the two-full-volume-fits cost bound"
    assert "mean+3std" in lower, "doc must name the release-gate policy it explicitly does NOT use"


def test_doc_states_lower_is_better_regression_rule(doc_text: str) -> None:
    """The non-regression rule is anchored to 'lower is better' on both metrics."""
    assert "lower is better" in doc_text.lower()
    assert "seam_l1" in doc_text
    assert "seam_curvature" in doc_text


def test_doc_specifies_observability_explainability_precedent(doc_text: str) -> None:
    """The gate sub-dict mirrors the existing _strategy / _working_size_selector
    explainability pattern (D-19 / M006)."""
    assert "_working_size_selector" in doc_text
    assert "_strategy" in doc_text


def test_doc_requires_no_solver_invariant_changes(doc_text: str) -> None:
    """Auto-apply composes existing surfaces and touches no ALM/backend BaSiC
    invariant (K02); the non-goals must state this explicitly."""
    assert "_alm" in doc_text or "backend.py" in doc_text
    lower = doc_text.lower()
    assert "solver" in lower or "invariant" in lower
