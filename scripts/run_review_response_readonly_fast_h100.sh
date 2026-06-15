#!/usr/bin/env bash
# H100 read-only fast-path audit for R5 WarpCore/WarpSpeed fairness.
#
# This diagnostic experiment keeps HKV's table layout and single-bucket search,
# but uses a read-only all-hit path that skips cache-side score updates, miss
# bookkeeping, pointer materialization, and eviction/admission side effects.
#
# Default setup:
#   H100, LF=0.25, capacity=100M, 25M successful all-hit queries,
#   value dimensions 1/2/4/8/16/32 fp32.

set -euo pipefail

HKV_ROOT="${HKV_ROOT:-$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}"
SM="${SM:-90}"
RESULTS="${RESULTS:-$HKV_ROOT/results/review-response-readonly-fast-h100}"
LOOKUP_CAPACITY="${LOOKUP_CAPACITY:-100000000}"
LOOKUP_BATCH_SIZE="${LOOKUP_BATCH_SIZE:-25000000}"
LOOKUP_WARMUP="${LOOKUP_WARMUP:-1}"
LOOKUP_RUNS="${LOOKUP_RUNS:-5}"
LOOKUP_HBM_GB="${LOOKUP_HBM_GB:-16}"
LOOKUP_API_LOCK="${LOOKUP_API_LOCK:-0}"
VALUE_DIMS="${VALUE_DIMS:-1 2 4 8 16 32}"
LF_ARGS=("$@")

if [ "${#LF_ARGS[@]}" -eq 0 ]; then
  LF_ARGS=(0.25)
fi

if [ -d /usr/local/cuda-12.9/bin ]; then
  export PATH=/usr/local/cuda-12.9/bin:$PATH
fi

mkdir -p "$RESULTS"/{logs,raw}

{
  echo "HKV read-only fast-path H100 audit"
  echo "  HKV_ROOT          = $HKV_ROOT"
  echo "  git_commit        = $(git -C "$HKV_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "  SM                = $SM"
  echo "  RESULTS           = $RESULTS"
  echo "  LOOKUP_CAPACITY   = $LOOKUP_CAPACITY"
  echo "  LOOKUP_BATCH_SIZE = $LOOKUP_BATCH_SIZE"
  echo "  LOOKUP_WARMUP     = $LOOKUP_WARMUP"
  echo "  LOOKUP_RUNS       = $LOOKUP_RUNS"
  echo "  LOOKUP_HBM_GB     = $LOOKUP_HBM_GB"
  echo "  LOOKUP_API_LOCK   = $LOOKUP_API_LOCK"
  echo "  VALUE_DIMS        = $VALUE_DIMS"
  echo "  LF_ARGS           = ${LF_ARGS[*]}"
  echo "  Started           = $(date)"
} | tee "$RESULTS/run-info.txt"

cd "$HKV_ROOT"
if [ ! -f tests/googletest/CMakeLists.txt ]; then
  rm -rf tests/googletest
  git clone --depth 1 https://github.com/google/googletest.git tests/googletest
fi

printf 'library,mode,operation,value_dim,value_bytes,load_factor,run,throughput_bops\n' \
  > "$RESULTS/readonly_fast_h100_summary.csv"

for dim in $VALUE_DIMS; do
  value_bytes=$((dim * 4))
  echo "=== value_dim=$dim value_bytes=$value_bytes ===" | tee -a "$RESULTS/run-info.txt"

  hkv_build="$HKV_ROOT/build-readonly-fast-h100-dim${dim}"
  rm -rf "$hkv_build"
  cmake -S "$HKV_ROOT" -B "$hkv_build" \
    -Dsm="$SM" \
    -DCMAKE_BUILD_TYPE=Release \
    -DHKV_LOOKUP_DIM="$dim" \
    -DHKV_LOOKUP_CAPACITY="$LOOKUP_CAPACITY" \
    -DHKV_LOOKUP_BATCH_SIZE="$LOOKUP_BATCH_SIZE" \
    -DHKV_LOOKUP_WARMUP="$LOOKUP_WARMUP" \
    -DHKV_LOOKUP_RUNS="$LOOKUP_RUNS" \
    -DHKV_LOOKUP_HBM_GB="$LOOKUP_HBM_GB" \
    -DHKV_LOOKUP_API_LOCK="$LOOKUP_API_LOCK" \
    > "$RESULTS/logs/cmake-hkv-dim${dim}.log" 2>&1
  cmake --build "$hkv_build" --target hkv_lookup_modes_benchmark -j"$(nproc)" \
    > "$RESULTS/logs/build-hkv-dim${dim}.log" 2>&1

  "$hkv_build/hkv_lookup_modes_benchmark" \
    --mode value_returning --query sequential "${LF_ARGS[@]}" \
    > "$RESULTS/raw/hkv_value_returning_dim${dim}.csv" \
    2> "$RESULTS/logs/hkv_value_returning_dim${dim}.log"
  "$hkv_build/hkv_lookup_modes_benchmark" \
    --mode readonly_fast_sweep --query sequential "${LF_ARGS[@]}" \
    > "$RESULTS/raw/hkv_readonly_fast_dim${dim}.csv" \
    2> "$RESULTS/logs/hkv_readonly_fast_dim${dim}.log"

  wc_build="$HKV_ROOT/baselines/build-readonly-fast-h100-dim${dim}"
  rm -rf "$wc_build"
  cmake -S "$HKV_ROOT/baselines" -B "$wc_build" \
    -DGPU_ARCH="$SM" \
    -DENABLE_CUCO_BENCH=OFF \
    -DBASELINE_DIM="$dim" \
    -DBASELINE_CAPACITY="$LOOKUP_CAPACITY" \
    -DBASELINE_BATCH_SIZE="$LOOKUP_BATCH_SIZE" \
    -DBASELINE_WARMUP="$LOOKUP_WARMUP" \
    -DBASELINE_RUNS="$LOOKUP_RUNS" \
    > "$RESULTS/logs/cmake-warpcore-dim${dim}.log" 2>&1
  cmake --build "$wc_build" --target warpcore_warpspeed_value_size_bench -j"$(nproc)" \
    > "$RESULTS/logs/build-warpcore-dim${dim}.log" 2>&1
  "$wc_build/warpcore_warpspeed_value_size_bench" \
    --mode both "${LF_ARGS[@]}" \
    > "$RESULTS/raw/warpcore_retrieve_dim${dim}.csv" \
    2> "$RESULTS/logs/warpcore_retrieve_dim${dim}.log"

  awk -F, -v dim="$dim" -v bytes="$value_bytes" \
    'NR>1 {printf "%s,%s,%s,%s,%s,%s,%s,%.6f\n", $1, $2, $3, dim, bytes, $4, $5, $6 * 1.073741824}' \
    "$RESULTS/raw/hkv_value_returning_dim${dim}.csv" >> "$RESULTS/readonly_fast_h100_summary.csv"
  awk -F, -v dim="$dim" -v bytes="$value_bytes" \
    'NR>1 {printf "%s,%s,%s,%s,%s,%s,%s,%.6f\n", $1, $2, $3, dim, bytes, $4, $5, $6 * 1.073741824}' \
    "$RESULTS/raw/hkv_readonly_fast_dim${dim}.csv" >> "$RESULTS/readonly_fast_h100_summary.csv"
  awk -F, -v dim="$dim" -v bytes="$value_bytes" \
    'NR>1 {print $1 "," $2 "," $3 "," dim "," bytes "," $4 "," $5 "," $6}' \
    "$RESULTS/raw/warpcore_retrieve_dim${dim}.csv" >> "$RESULTS/readonly_fast_h100_summary.csv"
done

{
  echo "Finished: $(date)"
  echo "Summary: $RESULTS/readonly_fast_h100_summary.csv"
} | tee -a "$RESULTS/run-info.txt"
