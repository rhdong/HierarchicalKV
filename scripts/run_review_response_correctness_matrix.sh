#!/usr/bin/env bash
# Review-response correctness/adversarial CSV matrix.
#
# Default quick run:
#   RESULTS=/workspace/results/review-response-correctness-matrix SM=90 \
#     bash scripts/run_review_response_correctness_matrix.sh
#
# Full scalable run:
#   MODE=full TABLE_MODE=both RESULTS=/workspace/results/full-correctness-matrix \
#     bash scripts/run_review_response_correctness_matrix.sh --streams=8

set -euo pipefail

HKV_ROOT="${HKV_ROOT:-$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel 2>/dev/null || cd "$(dirname "$0")/.." && pwd)}"
SM="${SM:-90}"
MODE="${MODE:-quick}"
TABLE_MODE="${TABLE_MODE:-both}"
SCENARIO="${SCENARIO:-all}"
BUILD_DIR="${BUILD_DIR:-$HKV_ROOT/build-review-correctness-matrix}"
RESULTS="${RESULTS:-$HKV_ROOT/results/review-response-correctness-matrix}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"
BENCH_ARGS=("--mode=$MODE" "--table_mode=$TABLE_MODE" "--scenario=$SCENARIO")

if [ "$#" -gt 0 ]; then
  BENCH_ARGS+=("$@")
fi

if [ -d /usr/local/cuda-12.9/bin ]; then
  export PATH=/usr/local/cuda-12.9/bin:$PATH
fi

mkdir -p "$RESULTS"/{logs,raw}

{
  echo "HKV review-response correctness/adversarial matrix"
  echo "  HKV_ROOT   = $HKV_ROOT"
  echo "  SM         = $SM"
  echo "  MODE       = $MODE"
  echo "  TABLE_MODE = $TABLE_MODE"
  echo "  SCENARIO   = $SCENARIO"
  echo "  BUILD_DIR  = $BUILD_DIR"
  echo "  RESULTS    = $RESULTS"
  echo "  BENCH_ARGS = ${BENCH_ARGS[*]}"
  echo "  Started    = $(date)"
  git -C "$HKV_ROOT" rev-parse HEAD 2>/dev/null | sed 's/^/  Commit     = /' || true
  nvidia-smi -L 2>/dev/null | sed 's/^/  GPU        = /' || true
} | tee "$RESULTS/run-info.txt"

cd "$HKV_ROOT"
if [ ! -f tests/googletest/CMakeLists.txt ]; then
  rm -rf tests/googletest
  git clone --depth 1 https://github.com/google/googletest.git tests/googletest
fi

cmake -S "$HKV_ROOT" -B "$BUILD_DIR" -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release \
  > "$RESULTS/logs/cmake.log" 2>&1
cmake --build "$BUILD_DIR" \
  --target correctness_stress_test hkv_correctness_stress_benchmark \
  -j"$JOBS" > "$RESULTS/logs/build.log" 2>&1

"$BUILD_DIR/correctness_stress_test" \
  --gtest_filter='CorrectnessStress.*' \
  > "$RESULTS/logs/correctness_stress_test.log" 2>&1

"$BUILD_DIR/hkv_correctness_stress_benchmark" "${BENCH_ARGS[@]}" \
  > "$RESULTS/raw/hkv_correctness_matrix.csv" \
  2> "$RESULTS/logs/hkv_correctness_matrix.log"

{
  echo "Finished: $(date)"
  echo "Artifacts:"
  echo "  $RESULTS/logs/correctness_stress_test.log"
  echo "  $RESULTS/raw/hkv_correctness_matrix.csv"
  echo "  $RESULTS/logs/hkv_correctness_matrix.log"
} | tee -a "$RESULTS/run-info.txt"
