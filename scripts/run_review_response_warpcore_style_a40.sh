#!/usr/bin/env bash
# A40 WarpCore-style key-only sanity anchor for R5.
#
# WarpSpeed's public scripts run lf_test with -c 100000000.  At LF=0.25, that
# means 25M inserted keys and 25M successful queries.  This script uses that
# profile for native WarpCore HashSet key-only lookup and HKV contains.

set -euo pipefail

HKV_ROOT="${HKV_ROOT:-$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}"
SM="${SM:-86}"
RESULTS="${RESULTS:-$HKV_ROOT/results/review-response-warpcore-style-a40}"
LOOKUP_DIM="${LOOKUP_DIM:-32}"
LOOKUP_CAPACITY="${LOOKUP_CAPACITY:-100000000}"
LOOKUP_BATCH_SIZE="${LOOKUP_BATCH_SIZE:-25000000}"
LOOKUP_WARMUP="${LOOKUP_WARMUP:-0}"
LOOKUP_RUNS="${LOOKUP_RUNS:-3}"
LOOKUP_HBM_GB="${LOOKUP_HBM_GB:-16}"
LF_ARGS=("$@")

if [ "${#LF_ARGS[@]}" -eq 0 ]; then
  LF_ARGS=(0.25)
fi

if [ -d /usr/local/cuda-12.9/bin ]; then
  export PATH=/usr/local/cuda-12.9/bin:$PATH
fi

mkdir -p "$RESULTS"/{logs,raw}

{
  echo "HKV/WarpCore WarpSpeed-style A40 key-only audit"
  echo "  HKV_ROOT          = $HKV_ROOT"
  echo "  git_commit        = $(git -C "$HKV_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "  SM                = $SM"
  echo "  RESULTS           = $RESULTS"
  echo "  LOOKUP_DIM        = $LOOKUP_DIM"
  echo "  LOOKUP_CAPACITY   = $LOOKUP_CAPACITY"
  echo "  LOOKUP_BATCH_SIZE = $LOOKUP_BATCH_SIZE"
  echo "  LOOKUP_WARMUP     = $LOOKUP_WARMUP"
  echo "  LOOKUP_RUNS       = $LOOKUP_RUNS"
  echo "  LOOKUP_HBM_GB     = $LOOKUP_HBM_GB"
  echo "  LF_ARGS           = ${LF_ARGS[*]}"
  echo "  Started           = $(date)"
} | tee "$RESULTS/run-info.txt"

cd "$HKV_ROOT"
if [ ! -f tests/googletest/CMakeLists.txt ]; then
  rm -rf tests/googletest
  git clone --depth 1 https://github.com/google/googletest.git tests/googletest
fi

cmake -S "$HKV_ROOT" -B "$HKV_ROOT/build-warpcore-style-a40" \
  -Dsm="$SM" \
  -DCMAKE_BUILD_TYPE=Release \
  -DHKV_LOOKUP_DIM="$LOOKUP_DIM" \
  -DHKV_LOOKUP_CAPACITY="$LOOKUP_CAPACITY" \
  -DHKV_LOOKUP_BATCH_SIZE="$LOOKUP_BATCH_SIZE" \
  -DHKV_LOOKUP_WARMUP="$LOOKUP_WARMUP" \
  -DHKV_LOOKUP_RUNS="$LOOKUP_RUNS" \
  -DHKV_LOOKUP_HBM_GB="$LOOKUP_HBM_GB" \
  > "$RESULTS/logs/cmake-hkv.log" 2>&1
cmake --build "$HKV_ROOT/build-warpcore-style-a40" \
  --target hkv_lookup_modes_benchmark -j"$(nproc)" \
  > "$RESULTS/logs/build-hkv.log" 2>&1

"$HKV_ROOT/build-warpcore-style-a40/hkv_lookup_modes_benchmark" \
  --mode key_only "${LF_ARGS[@]}" \
  > "$RESULTS/raw/hkv_key_only.csv" \
  2> "$RESULTS/logs/hkv_key_only.log"

cmake -S "$HKV_ROOT/baselines" -B "$HKV_ROOT/baselines/build-warpcore-style-a40" \
  -DGPU_ARCH="$SM" \
  -DENABLE_CUCO_BENCH=OFF \
  -DBASELINE_DIM="$LOOKUP_DIM" \
  -DBASELINE_CAPACITY="$LOOKUP_CAPACITY" \
  -DBASELINE_BATCH_SIZE="$LOOKUP_BATCH_SIZE" \
  -DBASELINE_WARMUP="$LOOKUP_WARMUP" \
  -DBASELINE_RUNS="$LOOKUP_RUNS" \
  > "$RESULTS/logs/cmake-warpcore.log" 2>&1
cmake --build "$HKV_ROOT/baselines/build-warpcore-style-a40" \
  --target warpcore_lookup_modes_bench -j"$(nproc)" \
  > "$RESULTS/logs/build-warpcore.log" 2>&1

"$HKV_ROOT/baselines/build-warpcore-style-a40/warpcore_lookup_modes_bench" \
  --mode key_only "${LF_ARGS[@]}" \
  > "$RESULTS/raw/warpcore_key_only.csv" \
  2> "$RESULTS/logs/warpcore_key_only.log"

{
  echo "Finished: $(date)"
  echo "Artifacts:"
  echo "  $RESULTS/raw/hkv_key_only.csv"
  echo "  $RESULTS/raw/warpcore_key_only.csv"
} | tee -a "$RESULTS/run-info.txt"
