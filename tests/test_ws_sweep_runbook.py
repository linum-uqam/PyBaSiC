"""CI guard: WS_SWEEP_RUNBOOK.md keeps production-faithful sweep flags."""

from __future__ import annotations

import re
from pathlib import Path

RUNBOOK = Path(".planning/phases/02-working-size-sweep/WS_SWEEP_RUNBOOK.md")

_BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def _runbook_bash_blocks() -> str:
    """Return concatenated bash fenced blocks (copy-pasteable commands only)."""
    text = RUNBOOK.read_text(encoding="utf-8")
    blocks = _BASH_BLOCK.findall(text)
    return "\n".join(blocks)


def test_runbook_production_faithful_flags() -> None:
    content = _runbook_bash_blocks()
    assert "working-size 64" in content
    assert "working-size 96" in content
    assert "--max-reweighting-iterations 500" in content
    assert " sweep " in content or " sweep \\" in content
    assert "sub-22" in content
    assert "ws64-candidate" in content
    assert "ws96-candidate" in content


def test_runbook_omits_z_selection_overrides() -> None:
    content = _runbook_bash_blocks()
    z_opt = "--" + "z-"
    assert f"{z_opt}indices" not in content
    assert f"{z_opt}sample" not in content
