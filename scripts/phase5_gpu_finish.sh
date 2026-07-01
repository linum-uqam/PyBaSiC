#!/usr/bin/env bash
# Phase 5 GPU finish — levers 2-3 + optimize after dct-kernel-tuning complete.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /home/frans/code/linum-basic

SLICE_ID="${SLICE_ID:-27}"
SUB22_ZARR="/scratch/workspace/sub-22/output/${SLICE_ID}/resample_mosaic_grid/mosaic_grid_z27_resampled.ome.zarr"
OPT_OUT="/scratch/workspace/sub-22/runs/ws128-opt/${SLICE_ID}"
LOG="$OPT_OUT/phase5-gpu-finish.log"
BASELINE_ID="${BASELINE_ID:-baseline-20260701T020128-be1e880-sub-22}"
GIT_COMMIT=$(git rev-parse HEAD)
export TORCHINDUCTOR_CACHE_DIR="$OPT_OUT/inductor-cache"
mkdir -p "$OPT_OUT" "$TORCHINDUCTOR_CACHE_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "=== Phase 5 GPU finish $(date -Is) baseline=$BASELINE_ID ==="

run_pytest_gate() {
  uv run pytest tests/test_alm_parity.py tests/test_backend_parity.py \
    tests/test_darkfield.py tests/test_vignette_validation.py \
    tests/test_benchmark_quality.py -x -q
}

run_compare() {
  local cand_id="$1"
  uv run python scripts/benchmark_speedup.py compare \
    --baseline "$OPT_OUT/$BASELINE_ID/baseline-bundle.json" \
    --candidate "$OPT_OUT/$cand_id/candidate-artifact.json" \
    --output-dir "$OPT_OUT" || true
}

run_code_path_candidate() {
  local label="$1"
  shift
  local logfile="$OPT_OUT/${label}.log"
  env "$@" uv run python scripts/benchmark_speedup.py candidate \
    --input "$SUB22_ZARR" \
    --subject-id sub-22 \
    --baseline-id "$BASELINE_ID" \
    --output-dir "$OPT_OUT" \
    --strategy baseline \
    --working-size 128 \
    --max-reweighting-iterations 500 \
    --estimate-darkfield \
    --repeats 3 \
    --warmup 1 \
    --allow-large-run \
    --config "$OPT_OUT/lever-stack.json" \
    --run-label "$label" 2>&1 | tee "$logfile" >/dev/null
  grep -oE 'candidate-[0-9T]+-[a-f0-9]+-sub-22' "$logfile" | tail -1
}

echo '{}' > "$OPT_OUT/lever-stack.json"
declare -a CANDIDATE_PATHS=(
  "$OPT_OUT/candidate-20260701T023858-498b2f7-sub-22/candidate-artifact.json"
)

echo "--- Lever 1 dct-kernel-tuning: already complete (promote) ---"
run_pytest_gate

echo "--- Lever 2: inductor-cache-warm-policy ---"
unset LINUM_BASIC_DCT_KERNEL LINUM_BASIC_ALM_COMPILE_MODE
export LINUM_BASIC_INDUCTOR_WARM_PASSES=2
CAND_ID=$(run_code_path_candidate lever-inductor-cache-warm-policy)
if [ -n "$CAND_ID" ]; then
  run_compare "$CAND_ID"
  run_pytest_gate || echo "pytest gate failed for inductor-cache-warm-policy"
  CANDIDATE_PATHS+=("$OPT_OUT/$CAND_ID/candidate-artifact.json")
fi

echo "--- Lever 3: compile-shape-stability ---"
unset LINUM_BASIC_DCT_KERNEL LINUM_BASIC_INDUCTOR_WARM_PASSES
export LINUM_BASIC_ALM_COMPILE_MODE=reduce-overhead
CAND_ID=$(run_code_path_candidate lever-compile-shape-stability)
if [ -n "$CAND_ID" ]; then
  run_compare "$CAND_ID"
  run_pytest_gate || echo "pytest gate failed for compile-shape-stability"
  CANDIDATE_PATHS+=("$OPT_OUT/$CAND_ID/candidate-artifact.json")
fi

echo "--- Batched ws=128 diagnostic ---"
unset LINUM_BASIC_DCT_KERNEL LINUM_BASIC_ALM_COMPILE_MODE LINUM_BASIC_INDUCTOR_WARM_PASSES
uv run python scripts/benchmark_speedup.py profile \
  --mode batched-diagnostic \
  --input "$SUB22_ZARR" \
  --subject-id sub-22 \
  --baseline-id "$BASELINE_ID" \
  --output-dir "$OPT_OUT" \
  --strategy baseline \
  --working-size 128 \
  --max-reweighting-iterations 500 \
  --estimate-darkfield \
  --repeats 3 \
  --warmup 1 \
  --allow-large-run

echo "--- Optimize aggregation ---"
OPT_ARGS=(optimize --output-dir "$OPT_OUT" --baseline-id "$BASELINE_ID")
for p in "${CANDIDATE_PATHS[@]}"; do
  OPT_ARGS+=(--candidate "$p")
done
uv run python scripts/benchmark_speedup.py "${OPT_ARGS[@]}"

run_pytest_gate

python3 - <<PY
import json
from pathlib import Path
out = Path("$OPT_OUT")
summary = {"baseline_id": "$BASELINE_ID", "git_commit": "$GIT_COMMIT"}
for name in ("lever-attempt-table.json", "phase5-fast-path.json", "phase5-backlog.json", "bottleneck-report.json", "compare-summary.json", "batched-handoff.json"):
    p = out / name
    if p.is_file():
        summary[name.replace(".json", "")] = json.loads(p.read_text())
(out / "phase5-uat-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("Wrote phase5-uat-summary.json")
PY

echo "=== Phase 5 GPU finish complete $(date -Is) ==="
