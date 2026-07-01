#!/usr/bin/env bash
# Phase 7 A6000 integration check — harness integration-check vs frozen Phase 5 baseline.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /home/frans/code/linum-basic

SLICE_ID="${SLICE_ID:-27}"
SUB22_ZARR="/scratch/workspace/sub-22/output/${SLICE_ID}/resample_mosaic_grid/mosaic_grid_z27_resampled.ome.zarr"
OPT_OUT="/scratch/workspace/sub-22/runs/ws128-opt/${SLICE_ID}"
BASELINE_ID="baseline-20260701T020128-be1e880-sub-22"
FAST_PATH_REF="$OPT_OUT/phase5-fast-path.json"
LOG="$OPT_OUT/phase7-integration.log"
GIT_COMMIT=$(git rev-parse HEAD)
export TORCHINDUCTOR_CACHE_DIR="$OPT_OUT/inductor-cache"
mkdir -p "$OPT_OUT" "$TORCHINDUCTOR_CACHE_DIR"
chmod 700 "$TORCHINDUCTOR_CACHE_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "=== Phase 7 integration check started $(date -Is) commit=$GIT_COMMIT ==="

if [ ! -f "$OPT_OUT/$BASELINE_ID/baseline-bundle.json" ]; then
  echo "FATAL: frozen baseline bundle missing: $OPT_OUT/$BASELINE_ID/baseline-bundle.json" >&2
  exit 1
fi
if [ ! -f "$FAST_PATH_REF" ]; then
  echo "FATAL: phase5-fast-path.json missing: $FAST_PATH_REF" >&2
  exit 1
fi

export BASELINE_ID
export LINUM_BASIC_DCT_KERNEL=tuned
unset LINUM_BASIC_INDUCTOR_WARM_PASSES LINUM_BASIC_ALM_COMPILE_MODE

echo "--- GPU / env inventory ---"
nvidia-smi -L
N_GPUS=$(uv run python -c "import torch; print(torch.cuda.device_count())")
echo "torch.cuda.device_count()=$N_GPUS"
echo "LINUM_BASIC_DCT_KERNEL=${LINUM_BASIC_DCT_KERNEL:-unset}"
echo "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

run_integration_candidate() {
  local logfile="$OPT_OUT/integration-candidate.log"
  uv run python scripts/benchmark_speedup.py candidate \
    --input "$SUB22_ZARR" \
    --subject-id sub-22 \
    --baseline-id "$BASELINE_ID" \
    --output-dir "$OPT_OUT" \
    --strategy sequential \
    --working-size 128 \
    --max-reweighting-iterations 500 \
    --estimate-darkfield \
    --repeats 1 \
    --warmup 0 \
    --allow-large-run \
    --run-label integration-candidate 2>&1 | tee "$logfile" >/dev/null
  grep -oE 'candidate-[0-9T]+-[a-f0-9]+-sub-22' "$logfile" | tail -1
}

echo "--- Integration candidate (strategy=auto, no z-selection) ---"
CAND_ID=$(run_integration_candidate)
if [ -z "$CAND_ID" ]; then
  echo "FATAL: integration candidate run did not emit candidate id" >&2
  exit 1
fi
echo "Integration candidate: $CAND_ID"

CAND_ARTIFACT="$OPT_OUT/$CAND_ID/candidate-artifact.json"
WALLCLOCK_BASELINE_MS=660000

echo "--- Harness integration-check (NFLO-05 layer 1) ---"
uv run python scripts/benchmark_speedup.py integration-check \
  --baseline "$OPT_OUT/$BASELINE_ID/baseline-bundle.json" \
  --candidate "$CAND_ARTIFACT" \
  --output-dir "$OPT_OUT" \
  --fast-path-ref "$FAST_PATH_REF" \
  --wallclock-baseline-ms "$WALLCLOCK_BASELINE_MS" || true

INTEGRATION_SUMMARY="$OPT_OUT/phase7-integration-summary.json"
if [ ! -f "$INTEGRATION_SUMMARY" ]; then
  echo "FATAL: phase7-integration-summary.json not written: $INTEGRATION_SUMMARY" >&2
  exit 1
fi

echo "--- Env snapshot summary ---"
uv run python - <<PY
import json
from pathlib import Path

out = Path("$OPT_OUT")
integration_path = out / "phase7-integration-summary.json"
integration = json.loads(integration_path.read_text(encoding="utf-8"))
summary = {
    "baseline_id": "$BASELINE_ID",
    "git_commit": "$GIT_COMMIT",
    "candidate_id": "$CAND_ID",
    "shared_inductor_cache": "$TORCHINDUCTOR_CACHE_DIR",
    "artifacts": {
        "phase7_integration_summary": str(integration_path),
        "phase5_fast_path": "$FAST_PATH_REF",
        "candidate_artifact": "$CAND_ARTIFACT",
        "phase7_integration_log": "$LOG",
    },
    "integration_summary": integration,
}
summary_path = out / "phase7-uat-summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
print("Wrote", summary_path)
PY

echo "=== Phase 7 integration check finished $(date -Is) ==="
