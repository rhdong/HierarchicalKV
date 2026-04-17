#!/bin/bash
# =============================================================================
# HierarchicalKV — SIGMOD reviewer reproduction runner
# -----------------------------------------------------------------------------
# Orchestrates every experiment referenced by the HKV SIGMOD paper.
# Produces CSV/TSV artifacts under ${RESULTS} matching the layout consumed
# by the figure-regeneration scripts and the paper's data tables.
#
# Usage:
#   bash scripts/run_all_benchmarks.sh                  # full suite (~6-8 h)
#
# Environment overrides:
#   HKV_ROOT     Repository root. Default: current git top-level.
#   SM           CUDA SM arch (90=H100, 86=A6000, 80=A100). Default: 90.
#   RESULTS      Output directory. Default: $HKV_ROOT/results.
#   SKIP_BUILD   If set non-empty, skip the clean rebuild step.
#
# Paper ↔ experiment mapping (see benchmark/README.md for details):
#   Exp #1  ← E8 (HKV) + E8b/E18 (WarpCore, BGHT, cuCollections, P2BHT)
#   Exp #2  ← E12/E13/E14 (merlin_hashtable_benchmark) + E4 (latency)
#   Exp #3a ← E6  (digest_ablation_benchmark, dual compile)
#   Exp #3c ← E5  (score_strategy_benchmark) + e19_cache_quality_sweep
#   Exp #3d ← P25 (p25_admission_control)
#   Exp #3e ← E7  (concurrency_benchmark, dual compile)
#   Exp #4  ← E10 + E17 (e10_dual_bucket_analysis, e17_single_vs_dual)
#   Supporting: E2 (batch size), E3 (error bars), E16 (LF-vs-baseline sweep)
# =============================================================================

set +e   # Baseline hang/crash at high LF is expected; keep going.

HKV_ROOT="${HKV_ROOT:-$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}"
SM="${SM:-90}"
RESULTS="${RESULTS:-$HKV_ROOT/results}"

# Auto-pick CUDA toolkit on Ubuntu if present, otherwise rely on PATH.
if [ -d /usr/local/cuda-12.9/bin ]; then
  export PATH=/usr/local/cuda-12.9/bin:$PATH
fi

echo "=========================================="
echo "HKV benchmark suite"
echo "  HKV_ROOT = $HKV_ROOT"
echo "  SM       = $SM"
echo "  RESULTS  = $RESULTS"
echo "  Started  = $(date)"
echo "=========================================="

mkdir -p "$RESULTS"/{E2,E3,E4,E5,E6,E7,E8,E9,E10,E11,E12-E14,E16,E17,E18,E19,P25}

# ============================================================
# Step 0: Build HKV + googletest submodule
# ============================================================
if [ -z "${SKIP_BUILD:-}" ]; then
  echo "[$(date)] === Step 0: Building HKV ==="
  cd "$HKV_ROOT"
  if [ ! -f tests/googletest/CMakeLists.txt ]; then
    echo "Cloning googletest ..."
    rm -rf tests/googletest
    git clone --depth 1 https://github.com/google/googletest.git tests/googletest
  fi
  rm -rf build && mkdir -p build && cd build
  cmake -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release .. 2>&1 | tail -10
  make -j"$(nproc)" 2>&1 | tail -10
  echo "[$(date)] Build DONE"

  ./merlin_hashtable_test --gtest_filter="*test_basic" 2>&1 | tail -3 || {
    echo "FATAL: sanity test failed"; exit 1;
  }
  echo "[$(date)] Sanity check PASSED"
else
  cd "$HKV_ROOT/build"
fi

BUILD_DIR="$HKV_ROOT/build"

# ============================================================
# E11: Hardware Info
# ============================================================
echo "[$(date)] === E11: Hardware Info ==="
lscpu > "$RESULTS/E11/cpu.txt"
free -h > "$RESULTS/E11/mem.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
  --format=csv > "$RESULTS/E11/gpu.txt"
nvidia-smi > "$RESULTS/E11/nvidia-smi.txt"
echo "[$(date)] E11 DONE"

# ============================================================
# E2: Batch Size Sensitivity (methodology: justifies 1M default)
# ============================================================
echo "[$(date)] === E2: Batch Size Sensitivity ==="
cd "$BUILD_DIR"
./batch_size_benchmark > "$RESULTS/E2/e2_results.csv" 2> "$RESULTS/E2/e2_progress.log"
echo "[$(date)] E2 DONE"

# ============================================================
# E3: Statistical Rigor — Error Bars (§5.1)
# ============================================================
echo "[$(date)] === E3: Error Bars ==="
./error_bars_benchmark > "$RESULTS/E3/e3_results.csv" 2> "$RESULTS/E3/e3_progress.log"
echo "[$(date)] E3 DONE"

# ============================================================
# E4: Latency Distribution (Exp #2 latency paragraph)
# ============================================================
echo "[$(date)] === E4: Latency Distribution ==="
./latency_benchmark > "$RESULTS/E4/e4_results.csv" 2> "$RESULTS/E4/e4_progress.log"
echo "[$(date)] E4 DONE"

# ============================================================
# E5: Score Strategy (precursor to Exp #3c)
# ============================================================
echo "[$(date)] === E5: Score Strategy ==="
echo "strategy,insert_throughput_bkvs,find_throughput_bkvs,hit_rate" \
  > "$RESULTS/E5/e5_results.csv"
for s in LRU LFU EpochLRU EpochLFU Custom; do
  ./score_strategy_benchmark "$s" >> "$RESULTS/E5/e5_results.csv" \
    2>> "$RESULTS/E5/e5_progress.log"
done
echo "[$(date)] E5 DONE"

# ============================================================
# E6: Exp #3a — Digest Ablation (dual compile)
# ============================================================
echo "[$(date)] === E6: Digest Ablation ==="
cmake -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release -DDISABLE_DIGEST=OFF .. 2>&1 | tail -3
make digest_ablation_benchmark -j"$(nproc)" 2>&1 | tail -3
cp digest_ablation_benchmark digest_ablation_with_digest

cmake -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release -DDISABLE_DIGEST=ON .. 2>&1 | tail -3
make digest_ablation_benchmark -j"$(nproc)" 2>&1 | tail -3
cp digest_ablation_benchmark digest_ablation_no_digest

cmake -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release -DDISABLE_DIGEST=OFF .. 2>&1 | tail -3

./digest_ablation_with_digest > "$RESULTS/E6/e6_with_digest.csv" \
  2> "$RESULTS/E6/e6_with_progress.log"
./digest_ablation_no_digest   > "$RESULTS/E6/e6_no_digest.csv" \
  2> "$RESULTS/E6/e6_no_progress.log"

cat "$RESULTS/E6/e6_with_digest.csv" > "$RESULTS/E6/e6_results.csv"
tail -n +2 "$RESULTS/E6/e6_no_digest.csv" >> "$RESULTS/E6/e6_results.csv"
echo "[$(date)] E6 DONE"

# ============================================================
# E7: Exp #3e — Concurrency Modes (dual compile)
# ============================================================
echo "[$(date)] === E7: Concurrency Modes ==="
cmake -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release -DUSE_RW_LOCK=OFF .. 2>&1 | tail -3
make concurrency_benchmark -j"$(nproc)" 2>&1 | tail -3
cp concurrency_benchmark concurrency_triple_group

cmake -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release -DUSE_RW_LOCK=ON .. 2>&1 | tail -3
make concurrency_benchmark -j"$(nproc)" 2>&1 | tail -3
cp concurrency_benchmark concurrency_rw_lock

cmake -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release -DUSE_RW_LOCK=OFF .. 2>&1 | tail -3

./concurrency_triple_group triple_group > "$RESULTS/E7/e7_triple_group.csv" \
  2> "$RESULTS/E7/e7_tg_progress.log"
./concurrency_rw_lock      rw_lock      > "$RESULTS/E7/e7_rw_lock.csv" \
  2> "$RESULTS/E7/e7_rw_progress.log"
echo "[$(date)] E7 DONE"

# ============================================================
# E8: HKV baseline throughput (feeds Exp #1 HKV line)
# ============================================================
echo "[$(date)] === E8: HKV Baseline ==="
cmake -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release .. 2>&1 | tail -3
make e8_hkv_baseline -j"$(nproc)" 2>&1 | tail -3
./e8_hkv_baseline > "$RESULTS/E8/e8_hkv_results.csv" 2> "$RESULTS/E8/e8_hkv_progress.log"
echo "[$(date)] E8 HKV DONE"

# ============================================================
# E9: Eviction comparison (co-design ablation)
# ============================================================
echo "[$(date)] === E9: Eviction Comparison ==="
make e9_eviction_comparison -j"$(nproc)" 2>&1 | tail -3
./e9_eviction_comparison > "$RESULTS/E9/e9_results.csv" 2> "$RESULTS/E9/e9_progress.log"
echo "[$(date)] E9 DONE"

# ============================================================
# E10: Dual-bucket analysis (precursor to Exp #4)
# ============================================================
echo "[$(date)] === E10: Dual-Bucket Analysis ==="
make e10_dual_bucket_analysis -j"$(nproc)" 2>&1 | tail -3
./e10_dual_bucket_analysis 128 > "$RESULTS/E10/e10_results.csv" \
  2> "$RESULTS/E10/e10_progress.log"
echo "[$(date)] E10 DONE"

# ============================================================
# E12/E13/E14: Exp #2 — End-to-end throughput (pure HBM + hybrid)
# ============================================================
echo "[$(date)] === E12/E13/E14: Standard Benchmark ==="
make merlin_hashtable_benchmark -j"$(nproc)" 2>&1 | tail -3
./merlin_hashtable_benchmark > "$RESULTS/E12-E14/full_results.txt" 2>&1
echo "[$(date)] E12/E13/E14 DONE"

# ============================================================
# E17: Exp #4 — Single vs Dual Bucket
# ============================================================
echo "[$(date)] === E17: Single vs Dual Bucket ==="
make e17_single_vs_dual -j"$(nproc)" 2>&1 | tail -3
./e17_single_vs_dual > "$RESULTS/E17/e17_results.csv" 2> "$RESULTS/E17/e17_progress.log"
echo "[$(date)] E17 DONE"

# ============================================================
# E19: Exp #3c — Cache quality sweep
# ============================================================
echo "[$(date)] === E19: Cache Quality Sweep ==="
make e19_cache_quality_sweep -j"$(nproc)" 2>&1 | tail -3
./e19_cache_quality_sweep > "$RESULTS/E19/e19_results.csv" 2> "$RESULTS/E19/e19_progress.log"
echo "[$(date)] E19 DONE"

# ============================================================
# P25: Exp #3d — Admission control
# ============================================================
echo "[$(date)] === P25: Admission Control ==="
make p25_admission_control -j"$(nproc)" 2>&1 | tail -3
./p25_admission_control > "$RESULTS/P25/p25_results.csv" 2> "$RESULTS/P25/p25_progress.log"
echo "[$(date)] P25 DONE"

# ============================================================
# External Baselines (Exp #1): WarpCore / BGHT / cuCollections / P2BHT
# ============================================================
echo "[$(date)] === Building Baselines ==="
cd "$HKV_ROOT/baselines"
rm -rf build && mkdir -p build && cd build
cmake -DGPU_ARCH="$SM" .. 2>&1 | tail -10
make -j"$(nproc)" 2>&1 | tail -10
BASELINES_DIR="$HKV_ROOT/baselines/build"
echo "[$(date)] Baselines Build DONE"

# E8b: full default LF sweep per baseline (captures Exp #1 HKV-vs-baseline table)
echo "[$(date)] === E8b: WarpCore ==="
"$BASELINES_DIR/warpcore_bench" > "$RESULTS/E8/e8_warpcore_results.csv" \
  2> "$RESULTS/E8/e8_warpcore_progress.log" \
  || echo "WarpCore FAILED ($?)" >> "$RESULTS/E8/e8_warpcore_progress.log"

echo "[$(date)] === E8b: BGHT ==="
"$BASELINES_DIR/bght_bench" > "$RESULTS/E8/e8_bght_results.csv" \
  2> "$RESULTS/E8/e8_bght_progress.log" \
  || echo "BGHT FAILED ($?)" >> "$RESULTS/E8/e8_bght_progress.log"

echo "[$(date)] === E8b: cuCollections ==="
"$BASELINES_DIR/cuco_bench" > "$RESULTS/E8/e8_cuco_results.csv" \
  2> "$RESULTS/E8/e8_cuco_progress.log" \
  || echo "cuCollections FAILED ($?)" >> "$RESULTS/E8/e8_cuco_progress.log"

echo "[$(date)] === E18: P2BHT ==="
"$BASELINES_DIR/p2bht_bench" > "$RESULTS/E18/e18_p2bht_results.csv" \
  2> "$RESULTS/E18/e18_p2bht_progress.log" \
  || echo "P2BHT FAILED ($?)" >> "$RESULTS/E18/e18_p2bht_progress.log"

# ============================================================
# E16: Exp #1 curve — per-LF timeout-protected sweep for each baseline
# ============================================================
echo "[$(date)] === E16: LF Degradation ==="
cd "$BUILD_DIR"
./e8_hkv_baseline > "$RESULTS/E16/e16_hkv_results.csv" \
  2> "$RESULTS/E16/e16_hkv_progress.log"

sweep_baseline() {
  local name="$1" bin="$2" out="$3" log="$4"
  echo "library,operation,load_factor,run,throughput_bkvs" > "$out"
  for lf in 0.10 0.25 0.50 0.75 0.80 0.90 0.95 1.00; do
    echo "=== $name LF=$lf ===" >> "$log"
    timeout 120 "$bin" "$lf" >> "$out" 2>> "$log"
    rc=$?
    if [ "$rc" -eq 124 ]; then
      echo "$name,insert,$lf,0,TIMEOUT" >> "$out"
      echo "$name,find,$lf,0,TIMEOUT"   >> "$out"
    elif [ "$rc" -ne 0 ]; then
      echo "$name,insert,$lf,0,CRASH_$rc" >> "$out"
      echo "$name,find,$lf,0,CRASH_$rc"   >> "$out"
    fi
  done
}

sweep_baseline "WarpCore"      "$BASELINES_DIR/warpcore_bench" \
               "$RESULTS/E16/e16_warpcore_results.csv" \
               "$RESULTS/E16/e16_warpcore_progress.log"
sweep_baseline "BGHT"          "$BASELINES_DIR/bght_bench" \
               "$RESULTS/E16/e16_bght_results.csv" \
               "$RESULTS/E16/e16_bght_progress.log"
sweep_baseline "cuCollections" "$BASELINES_DIR/cuco_bench" \
               "$RESULTS/E16/e16_cuco_results.csv" \
               "$RESULTS/E16/e16_cuco_progress.log"
sweep_baseline "P2BHT"         "$BASELINES_DIR/p2bht_bench" \
               "$RESULTS/E16/e16_p2bht_results.csv" \
               "$RESULTS/E16/e16_p2bht_progress.log"

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "Finished: $(date)"
echo "Results: $RESULTS"
echo "=========================================="
