#!/usr/bin/env bash
# Review-response adversarial performance curves for R5.Q3.
#
# Builds and runs HKV's value-returning cache workload under uniform,
# bucket-skewed, single-bucket, Zipf-hotset, and legal R/U/I-overlap stress.

set -euo pipefail

HKV_ROOT="${HKV_ROOT:-$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}"
SM="${SM:-90}"
RESULTS="${RESULTS:-$HKV_ROOT/results/review-response-adversarial-performance}"
RUN_MODE="${RUN_MODE:-quick}"
RUN_ARGS=("$@")

if [ "${#RUN_ARGS[@]}" -eq 0 ]; then
  RUN_ARGS=(--mode="$RUN_MODE")
fi

if [ -d /usr/local/cuda-12.9/bin ]; then
  export PATH=/usr/local/cuda-12.9/bin:$PATH
fi

mkdir -p "$RESULTS"/{logs,raw}

{
  echo "HKV adversarial performance curves"
  echo "  HKV_ROOT   = $HKV_ROOT"
  echo "  git_commit = $(git -C "$HKV_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "  SM         = $SM"
  echo "  RESULTS    = $RESULTS"
  echo "  RUN_ARGS   = ${RUN_ARGS[*]}"
  echo "  Started    = $(date)"
} | tee "$RESULTS/run-info.txt"

cd "$HKV_ROOT"
if [ ! -f tests/googletest/CMakeLists.txt ]; then
  rm -rf tests/googletest
  git clone --depth 1 https://github.com/google/googletest.git tests/googletest
fi

BUILD_DIR="$HKV_ROOT/build-adversarial-performance"
cmake -S "$HKV_ROOT" -B "$BUILD_DIR" \
  -Dsm="$SM" \
  -DCMAKE_BUILD_TYPE=Release \
  > "$RESULTS/logs/cmake.log" 2>&1
cmake --build "$BUILD_DIR" --target hkv_adversarial_performance_benchmark \
  -j"$(nproc)" \
  > "$RESULTS/logs/build.log" 2>&1

"$BUILD_DIR/hkv_adversarial_performance_benchmark" "${RUN_ARGS[@]}" \
  > "$RESULTS/raw/hkv_adversarial_performance.csv" \
  2> "$RESULTS/logs/hkv_adversarial_performance.log"

{
  echo "Finished: $(date)"
  echo "Artifacts:"
  echo "  $RESULTS/raw/hkv_adversarial_performance.csv"
} | tee -a "$RESULTS/run-info.txt"
