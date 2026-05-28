#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the vignette validation tests and save their diagnostic figures.

This script does not duplicate the test pipeline.  It simply invokes pytest
on ``tests/test_vignette_validation.py`` with ``LINUM_BASIC_VIGNETTE_ARTIFACT_DIR``
set, which causes each test to write a 6-panel PNG into the output directory.

The exit code matches pytest's: 0 on success, non-zero on test failure (so the
figures reflect whatever the algorithm is actually doing today, including
failures).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_TEST_FILE = _REPO_ROOT / "tests" / "test_vignette_validation.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vignette_results"),
        metavar="DIR",
        help="Directory for output PNGs. (default: %(default)s)",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LINUM_BASIC_VIGNETTE_ARTIFACT_DIR"] = str(args.output.resolve())

    cmd = [sys.executable, "-m", "pytest", str(_TEST_FILE), "-v", "--tb=short", "-s"]
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, env=env, cwd=_REPO_ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
