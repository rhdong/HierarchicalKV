#!/usr/bin/env bash
# Review-response P0 experiments for paper feedback follow-up.
#
# Produces lookup-mode fairness audit artifacts:
#   - HKV contains vs find
#   - WarpCore HashSet key-only vs SingleValueHashTable 128B value-returning
#
# Usage:
#   RESULTS="$PWD/results/review-response-p0" SM=90 \
#     bash scripts/run_review_response_p0.sh

set -euo pipefail

HKV_ROOT="${HKV_ROOT:-$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}"
SM="${SM:-90}"
RESULTS="${RESULTS:-$HKV_ROOT/results/review-response-p0}"
LF_ARGS=("$@")

if [ "${#LF_ARGS[@]}" -eq 0 ]; then
  LF_ARGS=(0.25 0.50 0.75 1.00)
fi


mkdir -p "$RESULTS"/{logs,raw}

{
  echo "HKV review-response P0"
  echo "  HKV_ROOT = $HKV_ROOT"
  echo "  SM       = $SM"
  echo "  RESULTS  = $RESULTS"
  echo "  LF_ARGS  = ${LF_ARGS[*]}"
  echo "  Started  = $(date)"
} | tee "$RESULTS/run-info.txt"

cd "$HKV_ROOT"
if [ ! -f tests/googletest/CMakeLists.txt ]; then
  git submodule update --init --recursive
fi

mkdir -p build
cd build
cmake -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release -DHKV_BUILD_REPRO_BENCHMARKS=ON .. \
  > "$RESULTS/logs/cmake-hkv.log" 2>&1
make hkv_lookup_modes_benchmark e8_hkv_baseline -j"$(nproc)" \
  > "$RESULTS/logs/build-hkv.log" 2>&1

./hkv_lookup_modes_benchmark "${LF_ARGS[@]}" \
  > "$RESULTS/raw/hkv_lookup_modes.csv" \
  2> "$RESULTS/logs/hkv_lookup_modes.log"

cd "$HKV_ROOT/baselines"
mkdir -p build
cd build
cmake -DGPU_ARCH="$SM" -DENABLE_CUCO_BENCH=OFF .. \
  > "$RESULTS/logs/cmake-baselines.log" 2>&1
make warpcore_lookup_modes_bench warpcore_bench -j"$(nproc)" \
  > "$RESULTS/logs/build-warpcore.log" 2>&1

timeout 1800 ./warpcore_lookup_modes_bench "${LF_ARGS[@]}" \
  > "$RESULTS/raw/warpcore_lookup_modes.csv" \
  2> "$RESULTS/logs/warpcore_lookup_modes.log" \
  || echo "warpcore_lookup_modes_bench failed rc=$?" \
    >> "$RESULTS/logs/warpcore_lookup_modes.log"

{
  echo "Finished: $(date)"
  echo "Artifacts:"
  echo "  $RESULTS/raw/hkv_lookup_modes.csv"
  echo "  $RESULTS/raw/warpcore_lookup_modes.csv"
} | tee -a "$RESULTS/run-info.txt"
