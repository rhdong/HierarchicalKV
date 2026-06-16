#!/usr/bin/env bash
# A40 value-size audit for R5 WarpCore/WarpSpeed fairness.
#
# Runs WarpCore SingleValueHashTable retrieve and HKV find with values of
# 1/2/4/16 fp32 under a WarpSpeed-style A40 profile:
#   capacity=100M, LF=0.25, 25M successful positive queries.

set -euo pipefail

HKV_ROOT="${HKV_ROOT:-$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}"
SM="${SM:-86}"
RESULTS="${RESULTS:-$HKV_ROOT/results/review-response-value-size-a40}"
LOOKUP_CAPACITY="${LOOKUP_CAPACITY:-100000000}"
LOOKUP_BATCH_SIZE="${LOOKUP_BATCH_SIZE:-25000000}"
LOOKUP_WARMUP="${LOOKUP_WARMUP:-1}"
LOOKUP_RUNS="${LOOKUP_RUNS:-5}"
LOOKUP_HBM_GB="${LOOKUP_HBM_GB:-16}"
VALUE_DIMS="${VALUE_DIMS:-1 2 4 16}"
LF_ARGS=("$@")

if [ "${#LF_ARGS[@]}" -eq 0 ]; then
  LF_ARGS=(0.25)
fi

if [ -d /usr/local/cuda-12.9/bin ]; then
  export PATH=/usr/local/cuda-12.9/bin:$PATH
fi

mkdir -p "$RESULTS"/{logs,raw}

{
  echo "HKV/WarpCore A40 value-size audit"
  echo "  HKV_ROOT          = $HKV_ROOT"
  echo "  git_commit        = $(git -C "$HKV_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "  SM                = $SM"
  echo "  RESULTS           = $RESULTS"
  echo "  LOOKUP_CAPACITY   = $LOOKUP_CAPACITY"
  echo "  LOOKUP_BATCH_SIZE = $LOOKUP_BATCH_SIZE"
  echo "  LOOKUP_WARMUP     = $LOOKUP_WARMUP"
  echo "  LOOKUP_RUNS       = $LOOKUP_RUNS"
  echo "  LOOKUP_HBM_GB     = $LOOKUP_HBM_GB"
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
  > "$RESULTS/value_size_summary.csv"

for dim in $VALUE_DIMS; do
  value_bytes=$((dim * 4))
  echo "=== value_dim=$dim value_bytes=$value_bytes ===" | tee -a "$RESULTS/run-info.txt"

  hkv_build="$HKV_ROOT/build-value-size-a40-dim${dim}"
  cmake -S "$HKV_ROOT" -B "$hkv_build" \
    -Dsm="$SM" \
    -DCMAKE_BUILD_TYPE=Release \
    -DHKV_LOOKUP_DIM="$dim" \
    -DHKV_LOOKUP_CAPACITY="$LOOKUP_CAPACITY" \
    -DHKV_LOOKUP_BATCH_SIZE="$LOOKUP_BATCH_SIZE" \
    -DHKV_LOOKUP_WARMUP="$LOOKUP_WARMUP" \
    -DHKV_LOOKUP_RUNS="$LOOKUP_RUNS" \
    -DHKV_LOOKUP_HBM_GB="$LOOKUP_HBM_GB" \
    > "$RESULTS/logs/cmake-hkv-dim${dim}.log" 2>&1
  cmake --build "$hkv_build" --target hkv_lookup_modes_benchmark -j"$(nproc)" \
    > "$RESULTS/logs/build-hkv-dim${dim}.log" 2>&1
  "$hkv_build/hkv_lookup_modes_benchmark" --mode value_returning "${LF_ARGS[@]}" \
    > "$RESULTS/raw/hkv_find_dim${dim}.csv" \
    2> "$RESULTS/logs/hkv_find_dim${dim}.log"

  wc_build="$HKV_ROOT/baselines/build-value-size-a40-dim${dim}"
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
  "$wc_build/warpcore_warpspeed_value_size_bench" --mode both "${LF_ARGS[@]}" \
    > "$RESULTS/raw/warpcore_retrieve_dim${dim}.csv" \
    2> "$RESULTS/logs/warpcore_retrieve_dim${dim}.log"

  awk -F, -v dim="$dim" -v bytes="$value_bytes" \
    'NR>1 {printf "%s,%s,%s,%s,%s,%s,%s,%.6f\n", $1, $2, $3, dim, bytes, $4, $5, $6 * 1.073741824}' \
    "$RESULTS/raw/hkv_find_dim${dim}.csv" >> "$RESULTS/value_size_summary.csv"
  awk -F, -v dim="$dim" -v bytes="$value_bytes" \
    'NR>1 {print $1 "," $2 "," $3 "," dim "," bytes "," $4 "," $5 "," $6}' \
    "$RESULTS/raw/warpcore_retrieve_dim${dim}.csv" >> "$RESULTS/value_size_summary.csv"
done

{
  echo "Finished: $(date)"
  echo "Summary: $RESULTS/value_size_summary.csv"
} | tee -a "$RESULTS/run-info.txt"
