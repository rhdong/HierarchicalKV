#!/usr/bin/env bash
# HKV-level alternate-hash review-response experiment.
#
# Rebuilds the real HKV kernels with several standard 64-bit hash finalizers
# and runs insert/find/admission/eviction checks through HKV itself.

set -euo pipefail

HKV_ROOT="${HKV_ROOT:-$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}"
SM="${SM:-90}"
RESULTS="${RESULTS:-$HKV_ROOT/results/review-response-hkv-hash-sensitivity}"
HASH_MODE="${HASH_MODE:-quick}"
HASH_ARGS=("$@")

if [ "${#HASH_ARGS[@]}" -eq 0 ]; then
  HASH_ARGS=(--mode "$HASH_MODE")
fi


mkdir -p "$RESULTS"/{logs,raw}

{
  echo "HKV-level hash sensitivity"
  echo "  HKV_ROOT   = $HKV_ROOT"
  echo "  git_commit = $(git -C "$HKV_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "  SM         = $SM"
  echo "  RESULTS    = $RESULTS"
  echo "  HASH_ARGS  = ${HASH_ARGS[*]}"
  echo "  Started    = $(date)"
} | tee "$RESULTS/run-info.txt"

cd "$HKV_ROOT"
if [ ! -f tests/googletest/CMakeLists.txt ]; then
  git submodule update --init --recursive
fi

variants=(
  "0:murmur3"
  "1:splitmix64"
  "2:xxhash_avalanche"
  "3:wyhash_final"
)

if [ "${INCLUDE_IDENTITY:-0}" = "1" ]; then
  variants+=("4:identity")
fi

for entry in "${variants[@]}"; do
  variant_id="${entry%%:*}"
  variant_name="${entry#*:}"
  build_dir="$HKV_ROOT/build-hash-${variant_name}"

  echo "Building hash variant ${variant_name} (${variant_id})" | tee -a "$RESULTS/run-info.txt"
  cmake -S "$HKV_ROOT" -B "$build_dir" \
    -Dsm="$SM" \
    -DCMAKE_BUILD_TYPE=Release -DHKV_BUILD_REPRO_BENCHMARKS=ON \
    -DHKV_REPRO_HASH_VARIANT="$variant_id" \
    > "$RESULTS/logs/cmake-${variant_name}.log" 2>&1
  cmake --build "$build_dir" --target hkv_hash_sensitivity_benchmark \
    -j"$(nproc)" \
    > "$RESULTS/logs/build-${variant_name}.log" 2>&1

  echo "Running hash variant ${variant_name}" | tee -a "$RESULTS/run-info.txt"
  "$build_dir/hkv_hash_sensitivity_benchmark" "${HASH_ARGS[@]}" \
    > "$RESULTS/raw/hkv_hash_sensitivity_${variant_name}.csv" \
    2> "$RESULTS/logs/hkv_hash_sensitivity_${variant_name}.log"
done

awk 'FNR == 1 && NR != 1 { next } { print }' \
  "$RESULTS"/raw/hkv_hash_sensitivity_*.csv \
  > "$RESULTS/raw/hkv_hash_sensitivity_combined.csv"

{
  echo "Finished: $(date)"
  echo "Artifacts:"
  echo "  $RESULTS/raw/hkv_hash_sensitivity_combined.csv"
} | tee -a "$RESULTS/run-info.txt"
