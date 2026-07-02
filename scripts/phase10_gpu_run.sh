#!/usr/bin/env bash
# Phase 10 A6000 GPU workflow — NFLO-03 inductor cache audit + NFLO-04 wall-clock reconciliation.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /home/frans/code/linum-basic

SLICE_ID="${SLICE_ID:-27}"
OPT_OUT="/scratch/workspace/sub-22/runs/ws128-opt/${SLICE_ID}"
SUB22_ZARR="/scratch/workspace/sub-22/output/${SLICE_ID}/resample_mosaic_grid/mosaic_grid_z27_resampled.ome.zarr"
BASELINE_ID="baseline-20260701T020128-be1e880-sub-22"
FAST_PATH_REF="$OPT_OUT/phase5-fast-path.json"
LOG="$OPT_OUT/phase10-gpu.log"
NFLO03_PATH="$OPT_OUT/nflo03-inductor-cache-audit.json"
NFLO04_PATH="$OPT_OUT/nflo04-wallclock-reconciliation.json"
RUN_LABEL="nflo03-multi-shared-cache"
GIT_COMMIT=$(git rev-parse HEAD)

export TORCHINDUCTOR_CACHE_DIR="$OPT_OUT/inductor-cache"
mkdir -p "$OPT_OUT" "$TORCHINDUCTOR_CACHE_DIR"
chmod 700 "$TORCHINDUCTOR_CACHE_DIR"

export LINUM_BASIC_DCT_KERNEL=tuned
unset LINUM_BASIC_INDUCTOR_WARM_PASSES LINUM_BASIC_ALM_COMPILE_MODE

# Optional supplementary Nextflow operator wall-clock (ms); harness-primary when unset (D-13).
NEXTFLOW_WALLCLOCK_MS="${NEXTFLOW_WALLCLOCK_MS:-}"

exec > >(tee -a "$LOG") 2>&1
echo "=== Phase 10 GPU workflow started $(date -Is) commit=$GIT_COMMIT ==="
echo "Shared inductor cache: $TORCHINDUCTOR_CACHE_DIR (mode 700)"

preflight_frozen_baseline() {
  local bundle="$OPT_OUT/$BASELINE_ID/baseline-bundle.json"
  local sidecar="$OPT_OUT/$BASELINE_ID/tolerance-sidecar.json"
  for path in "$bundle" "$sidecar" "$FAST_PATH_REF"; do
    if [ ! -f "$path" ]; then
      echo "FATAL: frozen baseline pre-flight failed — missing $path" >&2
      exit 1
    fi
  done
  echo "Frozen baseline pre-flight OK: $BASELINE_ID"
}

cache_inventory() {
  uv run python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

cache_dir = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
if not cache_dir.is_dir():
    print(json.dumps({"file_count": 0, "file_list_hash": None}))
    raise SystemExit(0)
files = sorted(p.relative_to(cache_dir).as_posix() for p in cache_dir.rglob("*") if p.is_file())
digest = hashlib.sha256("\n".join(files).encode()).hexdigest() if files else None
print(json.dumps({"file_count": len(files), "file_list_hash": digest}))
PY
}

run_multi_candidate() {
  local logfile="$OPT_OUT/${RUN_LABEL}.log"
  uv run python scripts/benchmark_speedup.py candidate \
    --input "$SUB22_ZARR" \
    --subject-id sub-22 \
    --baseline-id "$BASELINE_ID" \
    --output-dir "$OPT_OUT" \
    --strategy multi \
    --working-size 128 \
    --max-reweighting-iterations 500 \
    --estimate-darkfield \
    --repeats 3 \
    --warmup 1 \
    --allow-large-run \
    --run-label "$RUN_LABEL" 2>&1 | tee "$logfile" >/dev/null
  grep -oE 'candidate-[0-9T]+-[a-f0-9]+-sub-22' "$logfile" | tail -1
}

preflight_frozen_baseline

if [ ! -e "$SUB22_ZARR" ]; then
  echo "FATAL: input zarr missing: $SUB22_ZARR" >&2
  exit 1
fi

echo "--- GPU inventory ---"
nvidia-smi -L
N_GPUS=$(uv run python -c "import torch; print(torch.cuda.device_count())")
echo "torch.cuda.device_count()=$N_GPUS"
if [ "$N_GPUS" -lt 2 ]; then
  echo "FATAL: NFLO-03 multi-worker audit requires at least 2 GPUs (got $N_GPUS)" >&2
  exit 1
fi

echo "--- NFLO-03 pre-run inductor cache inventory ---"
export PRE_CACHE_JSON
PRE_CACHE_JSON=$(cache_inventory)
echo "pre_cache=$PRE_CACHE_JSON"

echo "--- NFLO-03 concurrent multi-GPU candidate (no z-selection override) ---"
if [ -n "${CANDIDATE_ID:-}" ]; then
  echo "Reusing CANDIDATE_ID=$CANDIDATE_ID (skip candidate run)"
else
  CANDIDATE_ID=$(run_multi_candidate)
fi
if [ -z "$CANDIDATE_ID" ]; then
  echo "FATAL: multi candidate run did not emit candidate id" >&2
  exit 1
fi
echo "CANDIDATE_ID=$CANDIDATE_ID"

echo "--- NFLO-03 post-run inductor cache inventory ---"
export POST_CACHE_JSON
POST_CACHE_JSON=$(cache_inventory)
echo "post_cache=$POST_CACHE_JSON"

echo "--- NFLO-03/04 artifact aggregation ---"
uv run python - <<PY
import json
import os
import re
from pathlib import Path

from linum_basic.benchmark.artifacts import CandidateArtifact, read_artifact
from linum_basic.benchmark.profile import build_nflo04_reconciliation

opt_out = Path("$OPT_OUT")
candidate_id = "$CANDIDATE_ID"
log_path = opt_out / "${RUN_LABEL}.log"
shared_cache = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).resolve()
pre_cache = json.loads(os.environ["PRE_CACHE_JSON"])
post_cache = json.loads(os.environ["POST_CACHE_JSON"])

candidate = read_artifact(
    opt_out / candidate_id / "candidate-artifact.json",
    CandidateArtifact,
)
meta = candidate.metadata or {}
concurrency = meta.get("concurrency") or {}
operator = meta.get("operator_timing") or {}
resolved_cache = concurrency.get("inductor_cache_path") or meta.get("inductor_cache_path")
gpu_map = concurrency.get("gpu_map") or {}
n_gpus = int(concurrency.get("n_gpus") or len(gpu_map) or 0)

worker_pids: list[int] = []
if log_path.is_file():
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    worker_pids = sorted({int(match) for match in re.findall(r"PID[:=\s]+(\d+)", log_text)})
    if not worker_pids:
        worker_pids = sorted({int(match) for match in re.findall(r"loky[_-]?(\d+)", log_text, flags=re.I)})

per_worker_cache_paths = {str(device): str(shared_cache) for device in gpu_map}
if not per_worker_cache_paths and n_gpus > 0:
    per_worker_cache_paths = {f"worker-{index}": str(shared_cache) for index in range(n_gpus)}

all_paths_match = all(Path(path).resolve() == shared_cache for path in per_worker_cache_paths.values())
cache_not_truncated = post_cache["file_count"] >= pre_cache["file_count"]
stable_shared_cache = (
    all_paths_match
    and resolved_cache is not None
    and Path(str(resolved_cache)).resolve() == shared_cache
    and cache_not_truncated
)

nflo03 = {
    "schema_version": "1",
    "fixture": False,
    "git_commit": "$GIT_COMMIT",
    "baseline_id": "$BASELINE_ID",
    "candidate_id": candidate_id,
    "shared_inductor_cache_dir": str(shared_cache),
    "pre_run_cache": pre_cache,
    "post_run_cache": post_cache,
    "worker_pids": worker_pids,
    "per_worker_cache_paths": per_worker_cache_paths,
    "resolved_inductor_cache_path": resolved_cache,
    "stable_shared_cache": stable_shared_cache,
    "audit_rationale": (
        "All workers inherit one TORCHINDUCTOR_CACHE_DIR; post-run file count did not shrink."
        if stable_shared_cache
        else "Shared-cache assertion failed — inspect per_worker_cache_paths and cache inventories."
    ),
    "log_path": str(log_path),
}
nflo03_path = Path("$NFLO03_PATH")
nflo03_path.write_text(json.dumps(nflo03, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Wrote {nflo03_path} stable_shared_cache={stable_shared_cache}")

nextflow_ms = os.environ.get("NEXTFLOW_WALLCLOCK_MS", "").strip()
nextflow_wallclock_ms = float(nextflow_ms) if nextflow_ms else None
harness_end_to_end_ms = float(operator.get("end_to_end_ms") or 0)
per_z_ms = operator.get("per_z_ms")
n_z = operator.get("n_z")

reconciliation = build_nflo04_reconciliation(
    harness_end_to_end_ms=harness_end_to_end_ms,
    nextflow_wallclock_ms=nextflow_wallclock_ms,
    per_z_ms=float(per_z_ms) if per_z_ms is not None else None,
    n_z=int(n_z) if n_z is not None else None,
)
nflo04 = {
    "schema_version": "1",
    "fixture": False,
    "git_commit": "$GIT_COMMIT",
    "baseline_id": "$BASELINE_ID",
    "candidate_id": candidate_id,
    "operator_timing_source": "candidate.metadata.operator_timing",
    **reconciliation,
}
nflo04_path = Path("$NFLO04_PATH")
nflo04_path.write_text(json.dumps(nflo04, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Wrote {nflo04_path} within_tolerance={reconciliation['within_tolerance']}")
PY

echo "=== Phase 10 GPU workflow finished $(date -Is) ==="
echo "Artifacts:"
echo "  $NFLO03_PATH"
echo "  $NFLO04_PATH"
