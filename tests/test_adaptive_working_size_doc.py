"""Structural regression test: lock the required sections of the adaptive
``working_size`` design doc (M006/S01).

The design doc is the slice's deliverable. Without a mechanical guard, a future
edit could silently strip a required section (e.g. the cost-bound or
quality-safety clause) and no CI signal would catch it, defeating the purpose
of the slice's success criteria.

Unlike :mod:`tests.test_ws_sweep_runbook` (which guards a gitignored
``.planning/`` artifact and therefore skips on clean checkouts),
``docs/adaptive_working_size.md`` is git-tracked, so this guard runs
unconditionally in CI.

Beyond asserting that required headings exist, several checks cross-validate
the design against the real code it claims to reuse (the Optuna candidate grid
in :mod:`linum_basic.tuning` and the ``dct_energy`` primitive in
:mod:`linum_basic.core`). Those cross-checks catch *drift* between the design
and the implementation, not just a missing heading.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOC = Path("docs/adaptive_working_size.md")

# ``##``-level headings the design must contain. ``Cost bound`` is matched as a
# substring of "## Cost bound: no probe fits", hence the regex below.
REQUIRED_SECTIONS = [
    "Signal inputs",  # the cheap signals the decision consumes
    "Selection rule",  # maps signals -> a value drawn from the grid
    "Cost bound",  # strict no-probe-fit bound
    "Quality-safety clause",  # ws=128-anchored fail-safe + K01 gate
    "S02 integration contract",  # how fit_mosaic / tune / CLI wire it
    "Observability contract",  # record resolution in MosaicFit.params
]

# Cheap signal inputs the must-haves require the doc to name explicitly.
REQUIRED_SIGNALS = [
    "n_z",
    "n_tiles",
    "tile_shape",
    "field_mode",
    "memory_budget_bytes",
    "torch.cuda.mem_get_info",
    "dct_energy",
]

# Solver primitives the cost-bound clause must enumerate as *forbidden*.
FORBIDDEN_SOLVER_PRIMITIVES = [
    "BaSiC.prepare",
    "BaSiC.run",
    "inexact_alm_l1",
]

# The ``rule_path`` explainability enum the observability contract must list.
RULE_PATHS = [
    "baseline-default",
    "memory-ceiling-shrink",
    "quality-floor-raise",
    "fallback-safe-default",
]

# Two formulations the doc may use to state the candidate grid; both cross-
# checked against linum_basic.tuning._DEFAULT_SEARCH_SPACE["working_size"].
_DOC_GRID_EQUALS_RE = re.compile(r'_DEFAULT_SEARCH_SPACE\["working_size"\]\s*==\s*\[([^\]]+)\]')
_DOC_GRID_ASSIGN_RE = re.compile(r"GRID\s*=\s*\(([^)]+)\)")


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


@pytest.mark.parametrize("signal", REQUIRED_SIGNALS)
def test_doc_names_required_signal_inputs(doc_text: str, signal: str) -> None:
    """Every must-have cheap signal input is named in the design doc."""
    assert signal in doc_text, f"design doc does not name required signal input {signal!r}"


def test_doc_preview_quality_reuses_core_dct_energy(doc_text: str) -> None:
    """The preview-quality signal must reuse linum_basic.core.dct_energy."""
    assert "linum_basic.core.dct_energy" in doc_text
    assert "preview_quality" in doc_text


def test_dct_energy_primitive_exists_in_core() -> None:
    """The doc reuses linum_basic.core.dct_energy -- verify it actually exists.

    If the primitive is renamed or removed, the design is invalid and S02
    cannot reuse it; fail here rather than at implementation time.
    """
    from linum_basic.core import dct_energy

    assert callable(dct_energy)


def test_doc_candidate_grid_matches_code_search_space(doc_text: str) -> None:
    """The doc's candidate grid must equal the validated Optuna grid in code.

    This is a drift check: if ``_DEFAULT_SEARCH_SPACE["working_size"]`` changes
    in :mod:`linum_basic.tuning`, the design's stated grid must change with it
    (or the selector would invent an unvalidated size).
    """
    from linum_basic.tuning import _DEFAULT_SEARCH_SPACE

    code_grid = [int(v) for v in _DEFAULT_SEARCH_SPACE["working_size"]]

    equals = _DOC_GRID_EQUALS_RE.search(doc_text)
    assign = _DOC_GRID_ASSIGN_RE.search(doc_text)
    assert equals is not None or assign is not None, (
        "doc must state the candidate grid as _DEFAULT_SEARCH_SPACE['working_size'] == [...] or GRID = (...)"
    )
    src = (equals or assign).group(1)
    doc_grid = [int(x.strip()) for x in src.split(",")]
    assert doc_grid == code_grid, (
        f"doc grid {doc_grid} != code _DEFAULT_SEARCH_SPACE['working_size'] "
        f"{code_grid}; the design and the validated grid have drifted apart"
    )


def test_doc_states_no_probe_fit_cost_bound(doc_text: str) -> None:
    """The cost bound forbids running any BaSiC/ALM solve at a candidate size."""
    lower = doc_text.lower()
    assert "probe fit" in lower, "doc must state the strict no-probe-fit cost bound"
    for primitive in FORBIDDEN_SOLVER_PRIMITIVES:
        assert primitive in doc_text, f"cost bound must enumerate forbidden solver primitive {primitive!r}"


@pytest.mark.parametrize("primitive", FORBIDDEN_SOLVER_PRIMITIVES)
def test_cost_bound_enumerates_forbidden_primitive(doc_text: str, primitive: str) -> None:
    """Each forbidden solver primitive is explicitly named (belt-and-suspenders
    view of :func:`test_doc_states_no_probe_fit_cost_bound`)."""
    assert primitive in doc_text


def test_doc_anchors_quality_safety_to_ws128_baseline(doc_text: str) -> None:
    """128 is the safe default, anchored to the D002/MEM002 evidence."""
    assert "128" in doc_text
    assert "MEM002" in doc_text or "D002" in doc_text


def test_doc_requires_k01_real_subject_seam_gate(doc_text: str) -> None:
    """Promotion of a non-128 size must pass the K01 real-subject seam gate."""
    assert "K01" in doc_text
    assert "seam_l1" in doc_text
    assert "seam_curvature" in doc_text


def test_doc_requires_opt_in_and_fail_safe_to_128(doc_text: str) -> None:
    """``auto`` must be opt-in and fail-safe to 128 on ambiguous signals."""
    assert '"auto"' in doc_text
    lower = doc_text.lower()
    assert "opt-in" in lower or "opt in" in lower
    assert "fallback" in lower or "fail-safe" in lower


def test_doc_requires_auto_resolved_before_basic_init(doc_text: str) -> None:
    """The ``auto`` sentinel must be resolved before reaching BaSiC.__init__."""
    assert '"auto"' in doc_text
    assert "BaSiC.__init__" in doc_text


def test_doc_specifies_fit_mosaic_and_tune_wiring(doc_text: str) -> None:
    """The S02 integration contract covers fit_mosaic and tune wiring."""
    assert "fit_mosaic" in doc_text
    assert "tune" in doc_text


def test_doc_cites_strategy_resolver_precedent(doc_text: str) -> None:
    """The resolver mirrors the WorkloadContext / resolve_auto_strategy precedent."""
    assert "resolve_auto_strategy" in doc_text
    assert "WorkloadContext" in doc_text


def test_doc_specifies_observability_metadata(doc_text: str) -> None:
    """Resolution is recorded under _working_size_selector on MosaicFit.params."""
    assert "_working_size_selector" in doc_text
    assert "MosaicFit" in doc_text


@pytest.mark.parametrize("path", RULE_PATHS)
def test_doc_enumerates_rule_path(doc_text: str, path: str) -> None:
    """Each rule_path explainability value is enumerated."""
    assert path in doc_text, f"doc must enumerate rule_path {path!r}"
