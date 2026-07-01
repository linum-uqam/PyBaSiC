#!/usr/bin/env bash
# Phase 5 A6000 GPU optimization program — 500-iter re-baseline + code-path levers.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /home/frans/code/linum-basic

SLICE_ID="${SLICE_ID:-27}"
SUB22_ZARR="/scratch/workspace/sub-22/output/${SLICE_ID}/resample_mosaic_grid/mosaic_grid_z27_resampled.ome.zarr"
OPT_OUT="/scratch/workspace/sub-22/runs/ws128-opt/${SLICE_ID}"
LOG="$OPT_OUT/phase5-gpu.log"
GIT_COMMIT=$(git rev-parse HEAD)
Z_INDICES="0,13,27,40,54"
export TORCHINDUCTOR_CACHE_DIR="$OPT_OUT/inductor-cache"
mkdir -p "$OPT_OUT" "$TORCHINDUCTOR_CACHE_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "=== Phase 5 GPU workflow started $(date -Is) commit=$GIT_COMMIT ==="

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
  # shellcheck disable=SC2068
  env "$@" uv run python scripts/benchmark_speedup.py candidate \
    --input "$SUB22_ZARR" \
    --subject-id sub-22 \
    --z-indices "$Z_INDICES" \
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

echo "--- Wave 0: 500-iter re-baseline (D-09) ---"
uv run python scripts/benchmark_speedup.py baseline \
  --input "$SUB22_ZARR" \
  --subject-id sub-22 \
  --z-indices "$Z_INDICES" \
  --output-dir "$OPT_OUT" \
  --strategy baseline \
  --working-size 128 \
  --max-reweighting-iterations 500 \
  --estimate-darkfield \
  --repeats 3 \
  --warmup 1 \
  --allow-large-run 2>&1 | tee "$OPT_OUT/phase5-rebaseline.log"

BASELINE_ID=$(grep -oE 'baseline-[0-9T]+-[a-f0-9]+-sub-22' "$OPT_OUT/phase5-rebaseline.log" | tail -1)
if [ -z "$BASELINE_ID" ]; then
  echo "FATAL: could not parse BASELINE_ID from re-baseline log" >&2
  exit 1
fi
echo "BASELINE_ID=$BASELINE_ID"
export BASELINE_ID

if [ -f "$OPT_OUT/phase5-backlog.json" ]; then
  cp "$OPT_OUT/phase5-backlog.json" "$OPT_OUT/phase5-backlog-prior.json"
fi
echo '{}' > "$OPT_OUT/lever-stack.json"
: > "$OPT_OUT/candidate-paths.txt"

echo "--- Profile refresh against Phase 5 baseline ---"
uv run python scripts/benchmark_speedup.py profile \
  --mode sequential \
  --input "$SUB22_ZARR" \
  --subject-id sub-22 \
  --z-indices "$Z_INDICES" \
  --baseline-id "$BASELINE_ID" \
  --output-dir "$OPT_OUT" \
  --strategy baseline \
  --working-size 128 \
  --max-reweighting-iterations 500 \
  --estimate-darkfield \
  --repeats 3 \
  --warmup 1 \
  --allow-large-run

python3 - <<PY
import json
from pathlib import Path
r = json.loads(Path("$OPT_OUT/bottleneck-report.json").read_text())
print("primary_limit:", r.get("primary_limit"))
print("ranked_levers:", [x["lever_id"] for x in r.get("ranked_levers", [])])
PY

echo "--- Tier 1: pytest gate stack (synthetic candidate skipped — requires input+baseline) ---"
run_pytest_gate

declare -a CANDIDATE_PATHS=()

echo "--- Lever 1: dct-kernel-tuning ---"
unset LINUM_BASIC_ALM_COMPILE_MODE LINUM_BASIC_INDUCTOR_WARM_PASSES
export LINUM_BASIC_DCT_KERNEL=tuned
CAND_ID=$(run_code_path_candidate lever-dct-kernel-tuning)
if [ -n "$CAND_ID" ]; then
  run_compare "$CAND_ID"
  run_pytest_gate || echo "pytest gate failed for dct-kernel-tuning"
  CANDIDATE_PATHS+=("$OPT_OUT/$CAND_ID/candidate-artifact.json")
  echo "$OPT_OUT/$CAND_ID/candidate-artifact.json" >> "$OPT_OUT/candidate-paths.txt"
fi

echo "--- Lever 2: inductor-cache-warm-policy ---"
unset LINUM_BASIC_DCT_KERNEL LINUM_BASIC_ALM_COMPILE_MODE
export LINUM_BASIC_INDUCTOR_WARM_PASSES=2
CAND_ID=$(run_code_path_candidate lever-inductor-cache-warm-policy)
if [ -n "$CAND_ID" ]; then
  run_compare "$CAND_ID"
  run_pytest_gate || echo "pytest gate failed for inductor-cache-warm-policy"
  CANDIDATE_PATHS+=("$OPT_OUT/$CAND_ID/candidate-artifact.json")
  echo "$OPT_OUT/$CAND_ID/candidate-artifact.json" >> "$OPT_OUT/candidate-paths.txt"
fi

echo "--- Lever 3: compile-shape-stability ---"
unset LINUM_BASIC_DCT_KERNEL LINUM_BASIC_INDUCTOR_WARM_PASSES
export LINUM_BASIC_ALM_COMPILE_MODE=reduce-overhead
CAND_ID=$(run_code_path_candidate lever-compile-shape-stability)
if [ -n "$CAND_ID" ]; then
  run_compare "$CAND_ID"
  run_pytest_gate || echo "pytest gate failed for compile-shape-stability"
  CANDIDATE_PATHS+=("$OPT_OUT/$CAND_ID/candidate-artifact.json")
  echo "$OPT_OUT/$CAND_ID/candidate-artifact.json" >> "$OPT_OUT/candidate-paths.txt"
fi

echo "--- Batched ws=128 diagnostic (Phase 6 handoff) ---"
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

echo "--- Final pytest gate stack ---"
run_pytest_gate

python3 - <<PY
import json
from pathlib import Path
out = Path("$OPT_OUT")
summary = {
    "baseline_id": "$BASELINE_ID",
    "git_commit": "$GIT_COMMIT",
    "artifacts": {
        "lever_attempt_table": str(out / "lever-attempt-table.json"),
        "phase5_backlog": str(out / "phase5-backlog.json"),
        "phase5_fast_path": str(out / "phase5-fast-path.json"),
        "bottleneck_report": str(out / "bottleneck-report.json"),
    },
}
for name in ("lever-attempt-table.json", "phase5-fast-path.json", "phase5-backlog.json"):
    p = out / name
    if p.is_file():
        summary[name.replace(".json", "")] = json.loads(p.read_text())
print(json.dumps(summary, indent=2))
summary_path = out / "phase5-uat-summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print("Wrote", summary_path)
PY

echo "=== Phase 5 GPU workflow finished $(date -Is) ==="
