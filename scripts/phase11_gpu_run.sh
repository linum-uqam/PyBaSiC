#!/usr/bin/env bash
# Phase 11 A6000 GPU harness — worker-compile-off at ws=128, 500 reweight iterations (D-01, D-18).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /home/frans/code/linum-basic

WS=128
MAX_REWEIGHT=500
Z_INDICES="0,13,27,40,54"
REPEATS=3
WARMUP=1
RUN_LABEL="lever-worker-compile-off"
BOTTLENECK_FIXTURE="tests/fixtures/bottleneck-report-worker-compile-off.json"
ITERATION_AB_FIXTURE="tests/fixtures/iteration-ab-z27.json"
GIT_COMMIT=$(git rev-parse HEAD)

# Phase 11 env levers (DCT tuned from v1.0 promotion + compile-off candidate).
export LINUM_BASIC_DCT_KERNEL=tuned
export LINUM_BASIC_ALM_COMPILE_MODE=off
unset LINUM_BASIC_INDUCTOR_WARM_PASSES

# sub-22: reuse v1.0 frozen baseline bundle (D-05, D-06).
SUB22_BASELINE_ID="baseline-20260701T020128-be1e880-sub-22"
SUB22_OPT_OUT="/scratch/workspace/sub-22/runs/ws128-opt/27"
SUB22_ZARR="/scratch/workspace/sub-22/output/27/resample_mosaic_grid/mosaic_grid_z27_resampled.ome.zarr"
SUB22_COMPARE_DIR="$SUB22_OPT_OUT/phase11-compare"
SUB22_FROZEN_BASELINE="$SUB22_OPT_OUT/$SUB22_BASELINE_ID/baseline-bundle.json"

# sub-21: fresh 500-iter baseline on canonical slice 48.
SUB21_OPT_OUT="/scratch_nvme/workspace/sub-21/runs/phase11/48"
SUB21_ZARR="/scratch_nvme/workspace/sub-21/output/48/resample_mosaic_grid/mosaic_grid_z48_resampled.ome.zarr"
SUB21_COMPARE_DIR="$SUB21_OPT_OUT/phase11-compare"

# sub-18: fresh 500-iter baseline on mosaic-grids slice 27.
SUB18_OPT_OUT="/scratch/workspace/sub-18/runs/phase11/27"
SUB18_ZARR="/scratch/workspace/sub-18/mosaic-grids/mosaic_grid_3d_z27.ome.zarr"
SUB18_COMPARE_DIR="$SUB18_OPT_OUT/phase11-compare"

FORENSICS_DIR="/scratch/workspace/sub-22/runs/forensics/27"
PHASE11_LOG="/scratch/workspace/sub-22/runs/ws128-opt/27/phase11-gpu.log"
PHASE11_SUMMARY="$SUB22_OPT_OUT/phase11-multisubject-summary.json"

mkdir -p "$SUB22_OPT_OUT" "$SUB21_OPT_OUT" "$SUB18_OPT_OUT" "$FORENSICS_DIR"
export TORCHINDUCTOR_CACHE_DIR="$SUB22_OPT_OUT/inductor-cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

exec > >(tee -a "$PHASE11_LOG") 2>&1
echo "=== Phase 11 GPU workflow started $(date -Is) commit=$GIT_COMMIT ==="
echo "LINUM_BASIC_DCT_KERNEL=${LINUM_BASIC_DCT_KERNEL:-unset}"
echo "LINUM_BASIC_ALM_COMPILE_MODE=${LINUM_BASIC_ALM_COMPILE_MODE:-unset}"

echo "--- Wave 0: pytest worker compile-off guard (511c88c) ---"
uv run pytest tests/test_parallel.py -k cuda_joblib_worker_init -x -q

copy_bottleneck_fixture() {
  local opt_out="$1"
  if [ ! -f "$BOTTLENECK_FIXTURE" ]; then
    echo "FATAL: bottleneck fixture missing: $BOTTLENECK_FIXTURE" >&2
    exit 1
  fi
  cp "$BOTTLENECK_FIXTURE" "$opt_out/bottleneck-report.json"
}

preflight_frozen_baseline() {
  local bundle="$SUB22_OPT_OUT/$SUB22_BASELINE_ID/baseline-bundle.json"
  local sidecar="$SUB22_OPT_OUT/$SUB22_BASELINE_ID/tolerance-sidecar.json"
  local fast_path="$SUB22_OPT_OUT/phase5-fast-path.json"
  for path in "$bundle" "$sidecar" "$fast_path"; do
    if [ ! -f "$path" ]; then
      echo "FATAL: frozen baseline pre-flight failed — missing $path" >&2
      exit 1
    fi
  done
  echo "Frozen baseline pre-flight OK: $SUB22_BASELINE_ID"
}

run_fresh_baseline() {
  local zarr="$1"
  local subject_id="$2"
  local opt_out="$3"
  local logfile="$opt_out/phase11-baseline.log"
  copy_bottleneck_fixture "$opt_out"
  uv run python scripts/benchmark_speedup.py baseline \
    --input "$zarr" \
    --subject-id "$subject_id" \
    --z-indices "$Z_INDICES" \
    --output-dir "$opt_out" \
    --strategy sequential \
    --working-size "$WS" \
    --max-reweighting-iterations "$MAX_REWEIGHT" \
    --estimate-darkfield \
    --repeats "$REPEATS" \
    --warmup "$WARMUP" \
    --allow-large-run 2>&1 | tee "$logfile" >/dev/null
  local baseline_id
  baseline_id=$(grep -oE "baseline-[0-9T]+-[a-f0-9]+-${subject_id}" "$logfile" | tail -1)
  if [ -z "$baseline_id" ]; then
    echo "FATAL: could not parse baseline id for $subject_id from $logfile" >&2
    exit 1
  fi
  echo "$baseline_id"
}

run_candidate() {
  local zarr="$1"
  local subject_id="$2"
  local opt_out="$3"
  local baseline_id="$4"
  local logfile="$opt_out/phase11-candidate-${subject_id}.log"
  local z_arg=()
  if [ "$subject_id" = "sub-22" ]; then
    z_arg=(--z-indices "$Z_INDICES")
  fi
  uv run python scripts/benchmark_speedup.py candidate \
    --input "$zarr" \
    --subject-id "$subject_id" \
    --baseline-id "$baseline_id" \
    --output-dir "$opt_out" \
    --strategy sequential \
    --working-size "$WS" \
    --max-reweighting-iterations "$MAX_REWEIGHT" \
    --estimate-darkfield \
    --repeats "$REPEATS" \
    --warmup "$WARMUP" \
    --allow-large-run \
    --run-label "$RUN_LABEL" \
    "${z_arg[@]}" 2>&1 | tee "$logfile" >/dev/null
  grep -oE "candidate-[0-9T]+-[a-f0-9]+-${subject_id}" "$logfile" | tail -1
}

run_compare() {
  local opt_out="$1"
  local baseline_id="$2"
  local cand_id="$3"
  local compare_dir="$4"
  mkdir -p "$compare_dir"
  uv run python scripts/benchmark_speedup.py compare \
    --baseline "$opt_out/$baseline_id/baseline-bundle.json" \
    --candidate "$opt_out/$cand_id/candidate-artifact.json" \
    --output-dir "$compare_dir" || true
}

print_promotion_verdict() {
  local subject_id="$1"
  local compare_dir="$2"
  local summary="$compare_dir/compare-summary.json"
  if [ ! -f "$summary" ]; then
    echo "FATAL: missing compare summary for $subject_id: $summary" >&2
    exit 1
  fi
  uv run python - <<PY
import json
from pathlib import Path
summary = json.loads(Path("$summary").read_text(encoding="utf-8"))
pv = summary.get("promotion_verdict", {})
print(f"--- $subject_id promotion_verdict ---")
print(json.dumps(pv, indent=2, sort_keys=True))
PY
}

declare -a SUBJECT_ORDER=(sub-22 sub-21 sub-18)
declare -A SUBJECT_ZARR=(
  [sub-22]="$SUB22_ZARR"
  [sub-21]="$SUB21_ZARR"
  [sub-18]="$SUB18_ZARR"
)
declare -A SUBJECT_OPT_OUT=(
  [sub-22]="$SUB22_OPT_OUT"
  [sub-21]="$SUB21_OPT_OUT"
  [sub-18]="$SUB18_OPT_OUT"
)
declare -A SUBJECT_COMPARE_DIR=(
  [sub-22]="$SUB22_COMPARE_DIR"
  [sub-21]="$SUB21_COMPARE_DIR"
  [sub-18]="$SUB18_COMPARE_DIR"
)
declare -A SUBJECT_BASELINE_ID=()
declare -A SUBJECT_CANDIDATE_ID=()

for subject_id in "${SUBJECT_ORDER[@]}"; do
  zarr="${SUBJECT_ZARR[$subject_id]}"
  opt_out="${SUBJECT_OPT_OUT[$subject_id]}"
  compare_dir="${SUBJECT_COMPARE_DIR[$subject_id]}"
  if [ ! -e "$zarr" ]; then
    echo "FATAL: input zarr missing for $subject_id: $zarr" >&2
    exit 1
  fi
  echo "=== Subject $subject_id ==="
  echo "input=$zarr"
  echo "output=$opt_out"

  if [ "$subject_id" = "sub-22" ]; then
    preflight_frozen_baseline
    baseline_id="$SUB22_BASELINE_ID"
    copy_bottleneck_fixture "$opt_out"
  else
    echo "--- Fresh 500-iter baseline for $subject_id ---"
    baseline_id=$(run_fresh_baseline "$zarr" "$subject_id" "$opt_out")
    echo "BASELINE_ID=$baseline_id"
  fi
  SUBJECT_BASELINE_ID[$subject_id]="$baseline_id"

  echo "--- Candidate ($RUN_LABEL) for $subject_id ---"
  cand_id=$(run_candidate "$zarr" "$subject_id" "$opt_out" "$baseline_id")
  if [ -z "$cand_id" ]; then
    echo "FATAL: candidate run did not emit id for $subject_id" >&2
    exit 1
  fi
  SUBJECT_CANDIDATE_ID[$subject_id]="$cand_id"
  echo "CANDIDATE_ID=$cand_id"

  echo "--- Compare for $subject_id ---"
  run_compare "$opt_out" "$baseline_id" "$cand_id" "$compare_dir"
  print_promotion_verdict "$subject_id" "$compare_dir"
done

echo "--- D-23 forensics ingestion + cumulative speed ratio (PERF-03) ---"
uv run python - <<'PY'
import json
import subprocess
from pathlib import Path

from linum_basic.benchmark.profile import (
    build_forensics_report,
    compute_stack_speed_ratio,
    load_historical_baselines_from_harness_candidate,
    load_historical_baselines_from_iteration_ab,
    write_forensics_report_bundle,
)

git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
iteration_ab = Path("tests/fixtures/iteration-ab-z27.json")
current_slow_ms = 229_000.0

subjects = {
    "sub-22": {
        "compare_dir": Path("/scratch/workspace/sub-22/runs/ws128-opt/27/phase11-compare"),
        "opt_out": Path("/scratch/workspace/sub-22/runs/ws128-opt/27"),
        "baseline_id": "baseline-20260701T020128-be1e880-sub-22",
        "slice_id": 27,
    },
    "sub-21": {
        "compare_dir": Path("/scratch_nvme/workspace/sub-21/runs/phase11/48/phase11-compare"),
        "opt_out": Path("/scratch_nvme/workspace/sub-21/runs/phase11/48"),
        "slice_id": 48,
    },
    "sub-18": {
        "compare_dir": Path("/scratch/workspace/sub-18/runs/phase11/27/phase11-compare"),
        "opt_out": Path("/scratch/workspace/sub-18/runs/phase11/27"),
        "slice_id": 27,
    },
}

# Resolve dynamic baseline ids for sub-21/sub-18 from latest baseline-* dirs.
for sid in ("sub-21", "sub-18"):
    opt = subjects[sid]["opt_out"]
    baseline_dirs = sorted(opt.glob("baseline-*"))
    if not baseline_dirs:
        raise SystemExit(f"FATAL: no baseline bundle under {opt} for {sid}")
    subjects[sid]["baseline_id"] = baseline_dirs[-1].name

historical_rows = list(
    load_historical_baselines_from_iteration_ab(iteration_ab, current_steady_state_ms=current_slow_ms)
)

subject_summaries: dict[str, dict] = {}
evidence_ids: list[str] = []

for sid, cfg in subjects.items():
    compare_summary_path = cfg["compare_dir"] / "compare-summary.json"
    compare_summary = json.loads(compare_summary_path.read_text(encoding="utf-8"))
    cand_id = compare_summary["candidate_id"]
    cand_artifact = cfg["opt_out"] / cand_id / "candidate-artifact.json"
    rows = load_historical_baselines_from_harness_candidate(
        cand_artifact,
        compare_summary_path,
        current_steady_state_ms=current_slow_ms,
    )
    historical_rows.extend(rows)
    evidence_ids.append(cand_id)
    promotion = compare_summary.get("promotion_verdict", {})
    timing = compare_summary.get("timing_report", {})
    subject_summaries[sid] = {
        "subject_id": sid,
        "slice_id": cfg["slice_id"],
        "baseline_id": cfg["baseline_id"],
        "candidate_id": cand_id,
        "promotion_verdict": promotion,
        "timing_report": timing,
    }

forensics_dir = Path("/scratch/workspace/sub-22/runs/forensics/27")
forensics_json = forensics_dir / "forensics-report.json"
if forensics_json.is_file():
    existing = json.loads(forensics_json.read_text(encoding="utf-8"))
    seen = {row.get("artifact_id") for row in existing.get("historical_baselines", [])}
    merged = list(existing.get("historical_baselines", []))
    for row in historical_rows:
        artifact_id = row.artifact_id
        if artifact_id in seen:
            continue
        seen.add(artifact_id)
        merged.append(
            {
                "git_commit": row.git_commit,
                "steady_state_ms": row.steady_state_ms,
                "end_to_end_ms": row.end_to_end_ms,
                "artifact_id": row.artifact_id,
                "change_class": row.change_class,
                "is_fast_era": row.is_fast_era,
            }
        )
    # Rebuild from merged HistoricalBaseline objects for triage consistency.
    from linum_basic.benchmark.profile import HistoricalBaseline

    historical_for_report = tuple(
        HistoricalBaseline(
            git_commit=str(item["git_commit"]),
            steady_state_ms=float(item["steady_state_ms"]),
            end_to_end_ms=item.get("end_to_end_ms"),
            artifact_id=str(item["artifact_id"]),
            change_class=item.get("change_class"),
            is_fast_era=bool(item.get("is_fast_era", False)),
        )
        for item in merged
    )
else:
    historical_for_report = tuple(historical_rows)

all_promotion_eligible = all(
    subject_summaries[sid]["promotion_verdict"].get("promotion_eligible") for sid in subjects
)
harness_overall = "promote" if all_promotion_eligible else "reject"

report = build_forensics_report(
    slice_id=27,
    historical_baselines=historical_for_report,
    current_steady_state_ms=current_slow_ms,
    current_git_commit=git_commit,
    harness_compare_overall=harness_overall,
    evidence_artifact_ids=evidence_ids,
)
write_forensics_report_bundle(forensics_dir, report)

frozen_baseline_path = Path(
    "/scratch/workspace/sub-22/runs/ws128-opt/27/baseline-20260701T020128-be1e880-sub-22/baseline-bundle.json"
)
frozen_payload = json.loads(frozen_baseline_path.read_text(encoding="utf-8"))
frozen_steady_ms = float(frozen_payload["metadata"]["telemetry"]["steady_state_ms"])

sub22_cand_path = subjects["sub-22"]["opt_out"] / subject_summaries["sub-22"]["candidate_id"] / "candidate-artifact.json"
sub22_cand_payload = json.loads(sub22_cand_path.read_text(encoding="utf-8"))
candidate_steady_ms = float(sub22_cand_payload["metadata"]["telemetry"]["steady_state_ms"])
candidate_e2e_ms = sub22_cand_payload["metadata"].get("operator_timing", {}).get("end_to_end_ms")
frozen_e2e_ms = frozen_payload["metadata"].get("operator_timing", {}).get("end_to_end_ms")

stack_ratio = compute_stack_speed_ratio(baseline_ms=frozen_steady_ms, candidate_ms=candidate_steady_ms)

summary = {
    "schema_version": "1",
    "git_commit": git_commit,
    "lever": "worker-compile-off",
    "working_size": 128,
    "max_reweighting_iterations": 500,
    "all_subjects_promotion_eligible": all_promotion_eligible,
    "subjects": subject_summaries,
    "cumulative_stack_speed_ratio_vs_v1_frozen_baseline": stack_ratio,
    "timing": {
        "primary_metric": "steady_state_ms",
        "frozen_baseline_steady_state_ms": frozen_steady_ms,
        "sub22_candidate_steady_state_ms": candidate_steady_ms,
        "frozen_baseline_end_to_end_ms": frozen_e2e_ms,
        "sub22_candidate_end_to_end_ms": candidate_e2e_ms,
    },
    "forensics_report": str(forensics_dir / "forensics-report.json"),
}

summary_path = Path("/scratch/workspace/sub-22/runs/ws128-opt/27/phase11-multisubject-summary.json")
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
print(f"Wrote {summary_path}")
print(f"Cumulative stack speed ratio vs v1.0 frozen baseline: {stack_ratio:.2f}x (steady_state_ms)")
PY

echo "=== Phase 11 GPU workflow finished $(date -Is) ==="
