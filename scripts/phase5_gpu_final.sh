#!/usr/bin/env bash
# Phase 5 GPU final steps — lever 3 + optimize after levers 1-2 complete.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /home/frans/code/linum-basic

OPT_OUT="/scratch/workspace/sub-22/runs/ws128-opt/27"
SUB22_ZARR="/scratch/workspace/sub-22/output/27/resample_mosaic_grid/mosaic_grid_z27_resampled.ome.zarr"
BASELINE_ID="baseline-20260701T020128-be1e880-sub-22"
GIT_COMMIT=$(git rev-parse HEAD)
LOG="$OPT_OUT/phase5-gpu-final.log"
export TORCHINDUCTOR_CACHE_DIR="$OPT_OUT/inductor-cache"

exec > >(tee -a "$LOG") 2>&1
echo "=== Phase 5 final $(date -Is) ==="

run_pytest_gate() {
  uv run pytest tests/test_alm_parity.py tests/test_backend_parity.py \
    tests/test_darkfield.py tests/test_vignette_validation.py \
    tests/test_benchmark_quality.py -x -q
}

run_compare() {
  uv run python scripts/benchmark_speedup.py compare \
    --baseline "$OPT_OUT/$BASELINE_ID/baseline-bundle.json" \
    --candidate "$OPT_OUT/$1/candidate-artifact.json" \
    --output-dir "$OPT_OUT" || true
}

run_code_path_candidate() {
  local label="$1"
  shift
  local logfile="$OPT_OUT/${label}.log"
  env "$@" uv run python scripts/benchmark_speedup.py candidate \
    --input "$SUB22_ZARR" --subject-id sub-22 --baseline-id "$BASELINE_ID" \
    --output-dir "$OPT_OUT" --strategy baseline --working-size 128 \
    --max-reweighting-iterations 500 --estimate-darkfield --repeats 3 --warmup 1 \
    --allow-large-run --config "$OPT_OUT/lever-stack.json" --run-label "$label" \
    2>&1 | tee "$logfile" >/dev/null
  grep -oE 'candidate-[0-9T]+-[a-f0-9]+-sub-22' "$logfile" | tail -1
}

echo '{}' > "$OPT_OUT/lever-stack.json"
CANDIDATES=(
  "candidate-20260701T023858-498b2f7-sub-22"
  "candidate-20260701T032306-c32c753-sub-22"
)

echo "--- Lever 3: compile-shape-stability ---"
unset LINUM_BASIC_DCT_KERNEL LINUM_BASIC_INDUCTOR_WARM_PASSES
export LINUM_BASIC_ALM_COMPILE_MODE=reduce-overhead
CAND_ID=$(run_code_path_candidate lever-compile-shape-stability)
if [ -n "$CAND_ID" ]; then
  run_compare "$CAND_ID"
  run_pytest_gate || true
  CANDIDATES+=("$CAND_ID")
fi

echo "--- Batched diagnostic ---"
unset LINUM_BASIC_DCT_KERNEL LINUM_BASIC_ALM_COMPILE_MODE LINUM_BASIC_INDUCTOR_WARM_PASSES
uv run python scripts/benchmark_speedup.py profile --mode batched-diagnostic \
  --input "$SUB22_ZARR" --subject-id sub-22 --baseline-id "$BASELINE_ID" \
  --output-dir "$OPT_OUT" --strategy baseline --working-size 128 \
  --max-reweighting-iterations 500 --estimate-darkfield --repeats 3 --warmup 1 --allow-large-run

echo "--- Optimize ---"
OPT_ARGS=(optimize --output-dir "$OPT_OUT" --baseline-id "$BASELINE_ID")
for c in "${CANDIDATES[@]}"; do
  OPT_ARGS+=(--candidate "$OPT_OUT/$c/candidate-artifact.json")
done
uv run python scripts/benchmark_speedup.py "${OPT_ARGS[@]}"

run_pytest_gate

python3 - <<PY
import json
from pathlib import Path
out = Path("$OPT_OUT")
summary = {"baseline_id": "$BASELINE_ID", "git_commit": "$GIT_COMMIT"}
for name in ("lever-attempt-table.json", "phase5-fast-path.json", "phase5-backlog.json",
             "bottleneck-report.json", "compare-summary.json", "batched-handoff.json"):
    p = out / name
    if p.is_file():
        summary[name.replace(".json", "")] = json.loads(p.read_text())
(out / "phase5-uat-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("COMPLETE")
PY
