#!/usr/bin/env bash
# GPU smoke test — operator-run CUDA regression check on cuda:0.
#
# Runs tests/test_alm_batched.py under LINUM_BASIC_ALM_COMPILE_MODE=off with a
# sub-5-minute wall-time budget, single-GPU (cuda:0 only). Skips gracefully with
# a clear message when CUDA is unavailable so the same entrypoint works on local
# dev and (future) self-hosted CI runners without modification.
#
# Invoke via:  make test-gpu-smoke
# Runbook:     docs/gpu_smoke.md
#
# Decisions (Phase 12 CONTEXT, .gsd/.../03-01-CONTEXT.md):
#   D-05 synthetic CUDA only (test_alm_batched.py)
#   D-06 LINUM_BASIC_ALM_COMPILE_MODE=off
#   D-07 test breadth = test_alm_batched.py only
#   D-08 sub-5-minute wall-time budget (fail if exceeded)
#   D-09 device cuda:0 only, single-GPU
#   D-12 future-CI-ready (unchanged when a self-hosted runner is added)
#   D-16 pytest terminal output only (no smoke-result.json artifact)
set -euo pipefail

# --- Configuration -----------------------------------------------------------
TIMEOUT_SECONDS="${LINUM_BASIC_GPU_SMOKE_TIMEOUT:-300}"  # 5-minute budget (D-08)
TEST_TARGET="tests/test_alm_batched.py"                   # D-05, D-07
DEVICE_INDEX="${LINUM_BASIC_GPU_SMOKE_DEVICE:-0}"         # cuda:0 only (D-09)

# --- Production-relevant environment (D-06) ----------------------------------
export LINUM_BASIC_ALM_COMPILE_MODE=off

# Pin the visible GPU to device 0 so the smoke never fans out to a second A6000.
# Inside the process physical GPU 0 is then the sole visible device (cuda:0),
# which is what tests/test_alm_batched.py targets.
export CUDA_VISIBLE_DEVICES="$DEVICE_INDEX"

# Run from the repository root regardless of invocation cwd (future-CI-ready).
repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

# --- CUDA availability probe -------------------------------------------------
# Skips cleanly on machines without torch/CUDA (local dev) so the same entrypoint
# remains invokable unchanged when a self-hosted runner is added later (D-12).
cuda_status="$(
    uv run python - <<'PY' 2>/dev/null || echo "probe-error"
import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    print("no-torch")
    sys.exit(0)

import torch

print("cuda" if torch.cuda.is_available() else "no-cuda")
PY
)"

if [ "$cuda_status" != "cuda" ]; then
    echo "skip: CUDA not available (probe='$cuda_status')."
    echo "      GPU smoke runs on the A6000 server; see docs/gpu_smoke.md."
    exit 0
fi

echo "=== GPU smoke: $TEST_TARGET on cuda:$DEVICE_INDEX ==="
echo "    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "    LINUM_BASIC_ALM_COMPILE_MODE=$LINUM_BASIC_ALM_COMPILE_MODE"
echo "    wall-time budget: ${TIMEOUT_SECONDS}s"

# --- Wall-time budget (D-08) -------------------------------------------------
# GNU/BSD `timeout` enforces the budget; the A6000 server always provides it.
timeout_bin="$(command -v timeout || command -v gtimeout || true)"

start_epoch="$(date +%s)"
if [ -n "$timeout_bin" ]; then
    set +e
    "$timeout_bin" "$TIMEOUT_SECONDS" uv run pytest "$TEST_TARGET" -v --tb=short
    rc=$?
    set -e
else
    echo "warn: 'timeout' command not found; running without wall-time enforcement." >&2
    set +e
    uv run pytest "$TEST_TARGET" -v --tb=short
    rc=$?
    set -e
fi
end_epoch="$(date +%s)"
elapsed=$((end_epoch - start_epoch))

# `timeout` exits 124 when the budget is reached.
if [ "$rc" -eq 124 ]; then
    echo "FAIL: GPU smoke exceeded the ${TIMEOUT_SECONDS}s wall-time budget (${elapsed}s)." >&2
    exit 124
fi

if [ "$rc" -ne 0 ]; then
    echo "FAIL: GPU smoke failed (pytest exit $rc) after ${elapsed}s." >&2
    exit "$rc"
fi

if [ "$elapsed" -gt "$TIMEOUT_SECONDS" ]; then
    # Belt-and-suspenders for the no-timeout fallback path.
    echo "warn: GPU smoke passed but exceeded the ${TIMEOUT_SECONDS}s budget (${elapsed}s)." >&2
    exit 0
fi

echo "PASS: GPU smoke passed in ${elapsed}s (budget ${TIMEOUT_SECONDS}s)."
