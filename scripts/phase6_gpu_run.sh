#!/usr/bin/env bash
# Phase 6 A6000 GPU concurrency A/B — multi vs batched against frozen Phase 5 baseline.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /home/frans/code/linum-basic

SLICE_ID="${SLICE_ID:-27}"
SUB22_ZARR="/scratch/workspace/sub-22/output/${SLICE_ID}/resample_mosaic_grid/mosaic_grid_z27_resampled.ome.zarr"
OPT_OUT="/scratch/workspace/sub-22/runs/ws128-opt/${SLICE_ID}"
BASELINE_ID="baseline-20260701T020128-be1e880-sub-22"
FAST_PATH_REF="$OPT_OUT/phase5-fast-path.json"
LOG="$OPT_OUT/phase6-gpu.log"
GIT_COMMIT=$(git rev-parse HEAD)
export TORCHINDUCTOR_CACHE_DIR="$OPT_OUT/inductor-cache"
mkdir -p "$OPT_OUT" "$TORCHINDUCTOR_CACHE_DIR"
chmod 700 "$TORCHINDUCTOR_CACHE_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "=== Phase 6 GPU concurrency workflow started $(date -Is) commit=$GIT_COMMIT ==="

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

echo "--- GPU inventory ---"
nvidia-smi -L
N_GPUS=$(uv run python -c "import torch; print(torch.cuda.device_count())")
echo "torch.cuda.device_count()=$N_GPUS"
echo "Shared inductor cache: $TORCHINDUCTOR_CACHE_DIR"

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

run_strategy_candidate() {
  local strategy="$1"
  local label="$2"
  local logfile="$OPT_OUT/${label}.log"
  uv run python scripts/benchmark_speedup.py candidate \
    --input "$SUB22_ZARR" \
    --subject-id sub-22 \
    --baseline-id "$BASELINE_ID" \
    --output-dir "$OPT_OUT" \
    --strategy "$strategy" \
    --working-size 128 \
    --max-reweighting-iterations 500 \
    --estimate-darkfield \
    --repeats 3 \
    --warmup 1 \
    --allow-large-run \
    --run-label "$label" 2>&1 | tee "$logfile" >/dev/null
  grep -oE 'candidate-[0-9T]+-[a-f0-9]+-sub-22' "$logfile" | tail -1
}

MULTI_CAND_ID=""
BATCHED_CAND_ID=""
MULTI_SKIPPED="false"
MULTI_SKIP_RATIONALE=""

if [ "$N_GPUS" -ge 2 ]; then
  echo "--- Multi arm (--strategy multi) ---"
  MULTI_CAND_ID=$(run_strategy_candidate multi concurrency-multi)
  if [ -n "$MULTI_CAND_ID" ]; then
    run_compare "$MULTI_CAND_ID"
    echo "Multi candidate: $MULTI_CAND_ID"
    echo "Checking shared inductor cache references in multi log..."
    grep -E "inductor|TORCHINDUCTOR_CACHE" "$OPT_OUT/concurrency-multi.log" | head -20 || true
  else
    MULTI_SKIPPED="true"
    MULTI_SKIP_RATIONALE="multi candidate run did not emit candidate id"
    echo "WARN: $MULTI_SKIP_RATIONALE"
  fi
else
  MULTI_SKIPPED="true"
  MULTI_SKIP_RATIONALE="fewer than 2 GPUs visible (Assumption A1 batched-only evidence)"
  echo "SKIP multi arm: $MULTI_SKIP_RATIONALE"
fi

echo "--- Batched arm (--strategy batched) ---"
BATCHED_CAND_ID=$(run_strategy_candidate batched concurrency-batched)
if [ -z "$BATCHED_CAND_ID" ]; then
  echo "FATAL: batched candidate run did not emit candidate id" >&2
  exit 1
fi
run_compare "$BATCHED_CAND_ID"
echo "Batched candidate: $BATCHED_CAND_ID"

echo "--- Pytest gate stack ---"
run_pytest_gate

echo "--- Concurrency verdict aggregation ---"
if [ "$MULTI_SKIPPED" = "true" ] || [ -z "$MULTI_CAND_ID" ]; then
  uv run python - <<PY
import json
from pathlib import Path

from linum_basic.benchmark.artifacts import CandidateArtifact, read_artifact
from linum_basic.benchmark.profile import build_phase6_concurrency_verdict

opt_out = Path("$OPT_OUT")
batched = read_artifact(
    opt_out / "$BATCHED_CAND_ID" / "candidate-artifact.json",
    CandidateArtifact,
)
meta = batched.metadata or {}
concurrency = meta.get("concurrency") or {}
operator = meta.get("operator_timing") or {}
telemetry = meta.get("telemetry") or {}
quality = meta.get("quality_verdict") or {}

mode = {
    "strategy": concurrency.get("strategy", "batched"),
    "fork_model": concurrency.get("fork_model"),
    "end_to_end_ms": float(operator.get("end_to_end_ms") or 0),
    "per_z_ms": float(operator.get("per_z_ms") or 0),
    "steady_state_ms": float(telemetry.get("steady_state_ms") or 0),
    "quality_verdict": quality,
    "artifact_id": batched.candidate_id,
    "peak_vram_bytes": int(concurrency.get("peak_vram_bytes") or 0),
    "gpu_map": concurrency.get("gpu_map"),
}

verdict = build_phase6_concurrency_verdict(
    [mode],
    baseline_id="$BASELINE_ID",
    phase5_fast_path_ref="$FAST_PATH_REF",
    evidence_artifact_ids=["$BASELINE_ID", batched.candidate_id],
    git_commit="$GIT_COMMIT",
)
verdict["selection_rationale"] = "$MULTI_SKIP_RATIONALE"
verdict_path = opt_out / "phase6-concurrency-verdict.json"
verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Wrote batched-only verdict to {verdict_path}")
PY
else
  uv run python scripts/benchmark_speedup.py concurrency \
    --output-dir "$OPT_OUT" \
    --baseline-id "$BASELINE_ID" \
    --candidate "$OPT_OUT/$MULTI_CAND_ID/candidate-artifact.json" \
    --candidate "$OPT_OUT/$BATCHED_CAND_ID/candidate-artifact.json" \
    --fast-path-ref "$FAST_PATH_REF"
fi

uv run python - <<PY
import json
from pathlib import Path

out = Path("$OPT_OUT")
verdict_path = out / "phase6-concurrency-verdict.json"
verdict = json.loads(verdict_path.read_text()) if verdict_path.is_file() else {}
summary = {
    "baseline_id": "$BASELINE_ID",
    "git_commit": "$GIT_COMMIT",
    "multi_skipped": "$MULTI_SKIPPED" == "true",
    "multi_skip_rationale": "$MULTI_SKIP_RATIONALE",
    "multi_candidate_id": "$MULTI_CAND_ID" or None,
    "batched_candidate_id": "$BATCHED_CAND_ID",
    "shared_inductor_cache": "$TORCHINDUCTOR_CACHE_DIR",
    "artifacts": {
        "phase6_concurrency_verdict": str(verdict_path),
        "phase5_fast_path": "$FAST_PATH_REF",
        "phase6_gpu_log": "$LOG",
    },
    "verdict": verdict,
}
summary_path = out / "phase6-uat-summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
print("Wrote", summary_path)
PY

echo "=== Phase 6 GPU concurrency workflow finished $(date -Is) ==="
