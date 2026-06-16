#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RESULTS="${RESULTS:-$REPO_ROOT/outputs/review-response-multigpu-sharding}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-multigpu-sharding}"
GPU_COUNTS="${GPU_COUNTS:-1 2 4 8}"
LF="${LF:-0.75}"
DIM="${DIM:-32}"
CAPACITY="${CAPACITY:-16777216}"
BATCH_SIZE="${BATCH_SIZE:-1048576}"
WARMUP="${WARMUP:-2}"
RUNS="${RUNS:-5}"
HBM_GB="${HBM_GB:-8}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
API_LOCK="${API_LOCK:-1}"
SM="${SM:-90}"
MODE="${MODE:-value_returning}"
QUERY="${QUERY:-random}"

mkdir -p "$RESULTS/raw" "$RESULTS/logs"

ngpu_available="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$ngpu_available" -lt 1 ]]; then
  echo "No GPUs visible." >&2
  exit 1
fi

echo "Multi-GPU sharding audit"
echo "Visible GPUs: $ngpu_available"
echo "GPU counts requested: $GPU_COUNTS"
echo "Config: LF=$LF DIM=$DIM CAPACITY=$CAPACITY BATCH_SIZE=$BATCH_SIZE WARMUP=$WARMUP RUNS=$RUNS HBM_GB=$HBM_GB API_LOCK=$API_LOCK MODE=$MODE QUERY=$QUERY"

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
  -Dsm="$SM" \
  -DCMAKE_BUILD_TYPE=Release \
  -DHKV_LOOKUP_DIM="$DIM" \
  -DHKV_LOOKUP_CAPACITY="$CAPACITY" \
  -DHKV_LOOKUP_BATCH_SIZE="$BATCH_SIZE" \
  -DHKV_LOOKUP_WARMUP="$WARMUP" \
  -DHKV_LOOKUP_RUNS="$RUNS" \
  -DHKV_LOOKUP_HBM_GB="$HBM_GB" \
  -DHKV_LOOKUP_BLOCK_SIZE="$BLOCK_SIZE" \
  -DHKV_LOOKUP_API_LOCK="$API_LOCK" \
  > "$RESULTS/logs/cmake.log" 2>&1
cmake --build "$BUILD_DIR" --target hkv_lookup_modes_benchmark -j"$(nproc)" \
  > "$RESULTS/logs/build.log" 2>&1

binary="$BUILD_DIR/hkv_lookup_modes_benchmark"
if [[ ! -x "$binary" ]]; then
  echo "Missing benchmark binary: $binary" >&2
  exit 1
fi

for count in $GPU_COUNTS; do
  if [[ "$count" -gt "$ngpu_available" ]]; then
    echo "Skipping ${count} GPUs; only ${ngpu_available} visible." | tee "$RESULTS/logs/run-${count}gpu.skip"
    continue
  fi

  echo "Running ${count}-GPU sharded audit..."
  run_dir="$RESULTS/raw/${count}gpu"
  mkdir -p "$run_dir"
  pids=()
  start_ns="$(date +%s%N)"
  for ((gpu = 0; gpu < count; gpu++)); do
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$binary" --mode "$MODE" --query "$QUERY" "$LF" \
        > "$run_dir/gpu${gpu}.csv" \
        2> "$RESULTS/logs/run-${count}gpu-gpu${gpu}.log"
    ) &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  end_ns="$(date +%s%N)"
  wall_s="$(python3 - <<PY
start_ns = int("$start_ns")
end_ns = int("$end_ns")
print(f"{(end_ns - start_ns) / 1e9:.6f}")
PY
)"
  if [[ "$failed" -ne 0 ]]; then
    echo "${count}-GPU run failed" >&2
    exit 1
  fi
  echo "$wall_s" > "$RESULTS/raw/${count}gpu/wall_seconds.txt"
done

python3 - "$RESULTS" <<'PY'
import csv
import glob
import os
import statistics
import sys

root = sys.argv[1]
rows = []
one_gpu_mean = None
for run_dir in sorted(glob.glob(os.path.join(root, "raw", "*gpu")),
                      key=lambda p: int(os.path.basename(p).replace("gpu", ""))):
    count = int(os.path.basename(run_dir).replace("gpu", ""))
    per_gpu = []
    for path in sorted(glob.glob(os.path.join(run_dir, "gpu*.csv"))):
        vals = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["mode"] == "value_returning" and row["operation"] == "find":
                    vals.append(float(row["throughput_bkvs"]))
        if not vals:
            raise SystemExit(f"No value_returning/find rows in {path}")
        per_gpu.append(statistics.mean(vals))
    if len(per_gpu) != count:
        raise SystemExit(f"Expected {count} GPU files in {run_dir}, found {len(per_gpu)}")
    aggregate = sum(per_gpu)
    mean_per_gpu = statistics.mean(per_gpu)
    stdev_per_gpu = statistics.pstdev(per_gpu) if len(per_gpu) > 1 else 0.0
    if count == 1:
        one_gpu_mean = aggregate
    efficiency = aggregate / (count * one_gpu_mean) if one_gpu_mean else 1.0
    wall_path = os.path.join(run_dir, "wall_seconds.txt")
    wall_s = float(open(wall_path).read().strip()) if os.path.exists(wall_path) else 0.0
    rows.append({
        "gpus": count,
        "aggregate_bops_s": aggregate,
        "mean_per_gpu_bops_s": mean_per_gpu,
        "stdev_per_gpu_bops_s": stdev_per_gpu,
        "scale_vs_1gpu": aggregate / one_gpu_mean if one_gpu_mean else 1.0,
        "efficiency": efficiency,
        "wall_seconds": wall_s,
    })

summary_path = os.path.join(root, "multigpu_sharding_summary.csv")
with open(summary_path, "w", newline="") as f:
    fieldnames = [
        "gpus",
        "aggregate_bops_s",
        "mean_per_gpu_bops_s",
        "stdev_per_gpu_bops_s",
        "scale_vs_1gpu",
        "efficiency",
        "wall_seconds",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})

print("summary:", summary_path)
for row in rows:
    print(
        f"{row['gpus']} GPUs: aggregate={row['aggregate_bops_s']:.3f} Bops/s, "
        f"per_gpu={row['mean_per_gpu_bops_s']:.3f}+-{row['stdev_per_gpu_bops_s']:.3f}, "
        f"scale={row['scale_vs_1gpu']:.2f}x, efficiency={row['efficiency']:.2%}, "
        f"wall={row['wall_seconds']:.1f}s"
    )
PY

