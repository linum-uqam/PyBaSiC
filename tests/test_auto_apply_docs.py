"""Structural regression test: lock the auto-apply *operator* docs (M008/S03).

T01/T02 added the fully-automated auto-apply workflow to the two operator-facing
docs:

- ``docs/tuning.md`` — the ``## Auto-apply safety gate`` narrative section
  (library usage, gate mechanism, ``.gate`` observability sub-dict,
  fallback-to-defaults contract, three-mechanism distinction, Validation Log).
- ``docs/runbooks.md`` — the ``## Auto-apply (fully automated)`` runbook
  (when / why / reproducible ``basic tune --auto-apply`` command /
  what-to-look-for table / gotchas).

Without a mechanical guard, a future edit could silently strip any of these —
dropping the ``--auto-apply`` flag, the fallback-reason enumeration, or the
observability keys — and no CI signal would catch it, leaving the R057
workflow undocumented for operators.

This mirrors :mod:`tests.test_auto_apply_safety_gate_doc.py` (the S01 guard on
the *design* doc ``docs/auto_apply_safety_gate.md``): a plain pytest module
asserting required headings/symbols exist in git-tracked markdown, plus
drift-checks against the real code the docs cite
(``linum_basic.tuning.auto_tune`` / ``AutoTuneResult`` / ``AutoApplyError`` /
``NO_REGRESSION_MARGIN``, ``FIRST_CLASS_METRICS``, and the ``basic tune`` CLI
flags). The S01 guard and this guard are deliberately distinct: S01 locks the
design contract, this guard locks the operator-facing surface that S03 added.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

TUNING_DOC = Path("docs/tuning.md")
RUNBOOKS_DOC = Path("docs/runbooks.md")

# --- docs/tuning.md: required "Auto-apply safety gate" subsection headings ---
# Matched as substrings of the heading line (so suffixes like the em dash in
# "Three mechanisms — do not conflate them" do not break the match).
TUNING_REQUIRED_SECTIONS = [
    "Auto-apply safety gate",  # the H2 section itself
    "Why the gate exists",
    "The workflow",
    "The safety gate",
    "The fallback-to-defaults contract",
    "observability sub-dict",  # "The `.gate` observability sub-dict"
    "Three mechanisms",  # "Three mechanisms — do not conflate them"
    "Validation Log",
]

# docs/tuning.md: code symbols the operator narrative must cite by name.
TUNING_REQUIRED_SYMBOLS = [
    "auto_tune",
    "AutoTuneResult",
    "AutoApplyError",
    "NO_REGRESSION_MARGIN",
    "recommend_bounds",
    "fit_mosaic",
    "apply_fit",
    "compute_quality_report",
    "compute_deltas",
]

# docs/tuning.md: the two first-class metrics the gate composes (K01).
TUNING_REQUIRED_METRICS = ["seam_l1", "seam_curvature"]

# docs/tuning.md: the ``.gate`` observability sub-dict keys (the
# explainability contract, mirroring the existing _strategy pattern).
TUNING_REQUIRED_OBSERVABILITY_KEYS = [
    "gate_verdict",
    "applied",
    "no_regression_margin",
    "failing_metrics",
    "fallback_reason",
    "deltas",
    "baseline",
    "candidate",
    "recommendation",
]

# docs/tuning.md: fallback reasons enumerated in the failure-modes table -- the
# contract that lets auto-apply never ship a worse correction than default.
FALLBACK_REASONS = [
    "regression-detected",
    "tune-failed",
    "degenerate-trial-history",
    "invalid-margin",
    "candidate-fit-failed",
    "quality-report-failed",
]

# docs/tuning.md: decision / cross-reference anchors.
TUNING_REQUIRED_ANCHORS = ["auto_apply_safety_gate", "D023", "mean+3std"]

# --- docs/runbooks.md: required "Auto-apply (fully automated)" structure -----
# These H3 subsections must live *inside* the auto-apply runbook, mirroring the
# gpu_smoke / streaming runbook structure exactly.
RUNBOOK_REQUIRED_SUBSECTIONS = [
    "When",
    "Why it exists",
    "Reproducible command",
    "What to look for",
    "Gotchas",
]

# docs/runbooks.md: the reproducible command must cite these real CLI flags.
RUNBOOK_REQUIRED_CLI_FLAGS = [
    "--auto-apply",
    "--apply",
    "--out-json",
    "--bounds-json",
    "--n-trials",
    "--verbose",
]

# docs/runbooks.md: verdict / applied values in the what-to-look-for table.
RUNBOOK_REQUIRED_VERDICTS = ["pass", "fail", "fallback", "baseline-default", "candidate"]

# docs/runbooks.md: the AutoApplyError-vs-fallback distinction is conveyed to
# operators via terminal signals (exit code + error message), not the Python
# class name. These are the operator-facing contract the gotcha must keep.
RUNBOOK_ERROR_SIGNALS = ["exit 1", "auto-apply failed"]


# --- helpers -----------------------------------------------------------------


def _h2_section(text: str, title_substr: str) -> str:
    """Return the body of the ``## `` section whose heading contains
    ``title_substr``, from its heading line through the next ``## `` heading
    (or end of document). Raises an AssertionError via the caller if absent.
    """
    start = re.search(rf"(?m)^##\s+.*{re.escape(title_substr)}.*$", text)
    assert start is not None, f"H2 section matching {title_substr!r} not found"
    after = start.end()
    nxt = re.search(r"(?m)^##\s+", text[after:])
    end = after + nxt.start() if nxt else len(text)
    return text[start.start() : end]


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(scope="module")
def tuning_text() -> str:
    """Full text of docs/tuning.md, loaded once for the module."""
    return TUNING_DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tuning_section(tuning_text: str) -> str:
    """The isolated ``## Auto-apply safety gate`` section of docs/tuning.md."""
    return _h2_section(tuning_text, "Auto-apply safety gate")


@pytest.fixture(scope="module")
def runbooks_text() -> str:
    """Full text of docs/runbooks.md, loaded once for the module."""
    return RUNBOOKS_DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def runbook_section(runbooks_text: str) -> str:
    """The isolated ``## Auto-apply (fully automated)`` section of docs/runbooks.md."""
    return _h2_section(runbooks_text, "Auto-apply (fully automated)")


# --- docs exist and are substantial -----------------------------------------


def test_tuning_doc_exists_and_is_substantial() -> None:
    """docs/tuning.md is present and the auto-apply section is non-trivial."""
    assert TUNING_DOC.exists(), f"{TUNING_DOC} is missing"
    text = TUNING_DOC.read_text(encoding="utf-8")
    assert text.strip(), "docs/tuning.md is empty"
    assert len(_h2_section(text, "Auto-apply safety gate")) > 1500, "Auto-apply safety gate section is suspiciously short"


def test_runbooks_doc_exists_and_is_substantial() -> None:
    """docs/runbooks.md is present and the auto-apply runbook is non-trivial."""
    assert RUNBOOKS_DOC.exists(), f"{RUNBOOKS_DOC} is missing"
    text = RUNBOOKS_DOC.read_text(encoding="utf-8")
    assert text.strip(), "docs/runbooks.md is empty"
    assert len(_h2_section(text, "Auto-apply (fully automated)")) > 1200, (
        "Auto-apply (fully automated) runbook is suspiciously short"
    )


# --- docs/tuning.md: structure ----------------------------------------------


@pytest.mark.parametrize("section", TUNING_REQUIRED_SECTIONS)
def test_tuning_section_heading_present(tuning_text: str, section: str) -> None:
    """Each must-have subsection exists as a markdown heading under the section."""
    pattern = re.compile(rf"(?m)^#{{1,6}}\s+.*{re.escape(section)}.*$")
    assert pattern.search(tuning_text), f"docs/tuning.md is missing required heading: {section!r}"


@pytest.mark.parametrize("symbol", TUNING_REQUIRED_SYMBOLS)
def test_tuning_doc_cites_symbol(tuning_text: str, symbol: str) -> None:
    """The narrative must cite each auto-apply surface by name."""
    assert symbol in tuning_text, f"docs/tuning.md must cite {symbol!r}"


@pytest.mark.parametrize("metric", TUNING_REQUIRED_METRICS)
def test_tuning_doc_cites_first_class_metric(tuning_text: str, metric: str) -> None:
    """The gate composes both first-class metrics (K01)."""
    assert metric in tuning_text, f"docs/tuning.md must cite metric {metric!r}"


@pytest.mark.parametrize("key", TUNING_REQUIRED_OBSERVABILITY_KEYS)
def test_tuning_doc_defines_observability_key(tuning_section: str, key: str) -> None:
    """The ``.gate`` sub-dict must define each explainability key inside the
    auto-apply section (not just elsewhere in the doc)."""
    assert key in tuning_section, f"docs/tuning.md auto-apply section must define gate key {key!r}"


@pytest.mark.parametrize("reason", FALLBACK_REASONS)
def test_tuning_doc_enumerates_fallback_reason(tuning_section: str, reason: str) -> None:
    """Each fallback reason in the failure-modes table is enumerated in-section."""
    assert reason in tuning_section, f"docs/tuning.md auto-apply section must enumerate fallback reason {reason!r}"


@pytest.mark.parametrize("anchor", TUNING_REQUIRED_ANCHORS)
def test_tuning_doc_anchors_to_reference(tuning_text: str, anchor: str) -> None:
    """The section anchors to its design contract, decision, and the policy it
    deliberately does NOT use."""
    assert anchor in tuning_text, f"docs/tuning.md must anchor to {anchor!r}"


# --- docs/runbooks.md: structure --------------------------------------------


@pytest.mark.parametrize("subsection", RUNBOOK_REQUIRED_SUBSECTIONS)
def test_runbook_has_subsection(runbook_section: str, subsection: str) -> None:
    """The auto-apply runbook mirrors the gpu_smoke/streaming subsection
    structure: each H3 must live *inside* the auto-apply section, not merely
    somewhere else in docs/runbooks.md (where 'When'/'Gotchas' recur)."""
    pattern = re.compile(rf"(?m)^###\s+{re.escape(subsection)}\b")
    assert pattern.search(runbook_section), f"Auto-apply runbook is missing subsection heading: {subsection!r}"


@pytest.mark.parametrize("flag", RUNBOOK_REQUIRED_CLI_FLAGS)
def test_runbook_cites_cli_flag(runbook_section: str, flag: str) -> None:
    """The reproducible command must cite each real ``basic tune`` flag."""
    assert flag in runbook_section, f"Auto-apply runbook must cite CLI flag {flag!r}"


@pytest.mark.parametrize("reason", FALLBACK_REASONS)
def test_runbook_enumerates_fallback_reason(runbook_section: str, reason: str) -> None:
    """The runbook's fallback_reason enumeration must stay in lockstep with the
    design doc / tuning.md enumeration."""
    assert reason in runbook_section, f"Auto-apply runbook must enumerate fallback reason {reason!r}"


@pytest.mark.parametrize("verdict", RUNBOOK_REQUIRED_VERDICTS)
def test_runbook_documents_verdict(runbook_section: str, verdict: str) -> None:
    """The what-to-look-for table must document each verdict/applied value."""
    assert verdict in runbook_section, f"Auto-apply runbook must document verdict/applied value {verdict!r}"


@pytest.mark.parametrize("signal", RUNBOOK_ERROR_SIGNALS)
def test_runbook_documents_baseline_error_path(runbook_section: str, signal: str) -> None:
    """The gotcha must keep the AutoApplyError (baseline-fit failure) path
    distinct from a fallback, via the operator-facing terminal signals."""
    assert signal in runbook_section, f"Auto-apply runbook must document the baseline-error signal {signal!r}"


def test_runbook_cross_references_design_contract(runbook_section: str) -> None:
    """The runbook must link back to the full design contract."""
    assert "auto_apply_safety_gate" in runbook_section, "Auto-apply runbook must cross-reference docs/auto_apply_safety_gate"


def test_runbook_distinguishes_from_release_gate(runbook_section: str) -> None:
    """The gotcha must keep the auto-apply gate distinct from the mean+3std
    release gate, mirroring the three-mechanism distinction in tuning.md."""
    assert "mean+3std" in runbook_section


# --- drift checks against the real cited code --------------------------------


def test_auto_apply_symbols_exist_in_code() -> None:
    """The docs cite linum_basic.tuning auto-apply surfaces -- verify they
    actually exist. If one is renamed/removed, the operator docs are invalid;
    fail here rather than at operator run time."""
    from linum_basic import tuning

    for name in ["auto_tune", "AutoTuneResult", "AutoApplyError", "NO_REGRESSION_MARGIN"]:
        assert hasattr(tuning, name), f"linum_basic.tuning.{name} missing (docs cite it)"


def test_tuning_doc_no_regression_margin_matches_code(tuning_text: str) -> None:
    """The doc's stated NO_REGRESSION_MARGIN default must equal the code value.

    Drift check: if the constant's default changes in the code, the operator
    docs must change with it, or the gate semantics the docs describe would be
    wrong."""
    from linum_basic.tuning import NO_REGRESSION_MARGIN

    code_value = repr(NO_REGRESSION_MARGIN)
    # The doc shows the default inline ("default `0.0`") and in code snippets.
    assert str(NO_REGRESSION_MARGIN) in tuning_text or code_value in tuning_text, (
        f"docs/tuning.md must state the code NO_REGRESSION_MARGIN value {code_value}"
    )


def test_tuning_doc_first_class_metrics_match_code(tuning_section: str) -> None:
    """The gate's two-metric rule must cite exactly the code's FIRST_CLASS_METRICS.

    Drift check: if FIRST_CLASS_METRICS changes in the code, the operator docs'
    gate description must follow, or the gate would test metrics that no longer
    exist."""
    from linum_basic.benchmark.quality import FIRST_CLASS_METRICS

    for metric in FIRST_CLASS_METRICS:
        assert metric in tuning_section, f"auto-apply section must cite first-class metric {metric!r} from FIRST_CLASS_METRICS"


def test_runbook_cli_flags_exist_in_parser() -> None:
    """The runbook's reproducible command cites ``basic tune`` flags -- verify
    they are actually registered on the tune subcommand. If a flag is renamed or
    removed, the runbook command would fail at operator run time."""
    parser = argparse.ArgumentParser(prog="basic")
    subs = parser.add_subparsers(dest="subcommand")
    from linum_basic import cli

    cli._add_tune_subcommand(subs)  # type: ignore[attr-defined]

    flags: set[str] = set()
    for action in parser._actions:  # type: ignore[attr-defined]
        flags.update(action.option_strings)
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                for sub_action in sub._actions:  # type: ignore[attr-defined]
                    flags.update(sub_action.option_strings)

    for flag in RUNBOOK_REQUIRED_CLI_FLAGS:
        assert flag in flags, f"basic tune CLI flag {flag!r} missing from parser (runbook cites it)"


# --- contract-shape checks ---------------------------------------------------


def test_tuning_doc_states_two_full_volume_fit_cost_bound(tuning_section: str) -> None:
    """The cost bound is exactly two full-volume fits, never an N-repeat
    calibration -- the milestone's literal cost contract."""
    assert "two full-volume fits" in tuning_section, "auto-apply section must state the two-full-volume-fits cost bound"


def test_tuning_doc_states_lower_is_better_regression_rule(tuning_section: str) -> None:
    """The non-regression rule is anchored to 'lower is better' on both metrics."""
    assert "lower is better" in tuning_section.lower()


def test_runbook_states_fallback_is_success(runbook_section: str) -> None:
    """The headline gotcha: a fallback returns a valid default-bounds volume
    (exit 0), distinct from the baseline-fit error (exit 1)."""
    lower = runbook_section.lower()
    assert "fallback is a success" in lower
    assert "exit 0" in runbook_section or "exit 1" in runbook_section
