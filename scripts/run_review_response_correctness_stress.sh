#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-review-correctness}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results/review-response-correctness-stress}"
SM="${SM:-90}"

mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/raw"

{
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "root=$ROOT_DIR"
  echo "build_dir=$BUILD_DIR"
  echo "sm=$SM"
  git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null | sed 's/^/commit=/' || true
  nvidia-smi -L 2>/dev/null | sed 's/^/gpu=/'
} > "$RESULTS_DIR/run-info.txt"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release \
  > "$RESULTS_DIR/logs/cmake.log" 2>&1
cmake --build "$BUILD_DIR" --target correctness_stress_test -j"$(nproc)" \
  > "$RESULTS_DIR/logs/build.log" 2>&1

"$BUILD_DIR/correctness_stress_test" \
  --gtest_filter='CorrectnessStress.*' \
  > "$RESULTS_DIR/logs/correctness_stress_test.log" 2>&1

cat > "$RESULTS_DIR/raw/correctness_stress_summary.csv" <<'CSV'
suite,status,log
CorrectnessStress,pass,logs/correctness_stress_test.log
CSV

echo "Wrote $RESULTS_DIR"
