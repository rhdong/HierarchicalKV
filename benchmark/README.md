# HKV Benchmark Reproducibility Guide (SIGMOD)

This directory contains every benchmark used to produce the figures and
tables in the HierarchicalKV (HKV) SIGMOD paper. The guide below tells a
reviewer, for each paper claim, **which binary to build, which command
to run, and what output to expect**. All measurements reported in the
paper were collected on a single NVIDIA H100 NVL (94 GB HBM3) with
`nvcc -O3 -arch=sm_90` under CUDA 12.9.

> **One-shot runner.** If you just want everything in one go, see
> [`../scripts/run_all_benchmarks.sh`](../scripts/run_all_benchmarks.sh)
> (~6–8 h total). Individual experiments below are faster and can be
> run in isolation.

---

## Quick reference: paper claim ↔ script

| Paper location | Claim / artifact | Binary | Status |
|:--|:--|:--|:--|
| **Exp #1** fig `lf-degradation`, tab `lf-comparison` | LF sensitivity vs. 4 baselines (§5.2) | `e8_hkv_baseline` + `baselines/{warpcore,bght,p2bht,cuco}_bench` | reproducible |
| **Exp #2** fig `throughput-bar` + "Latency and scaling" + "Hybrid storage impact" (§5.3) | End-to-end throughput, Configs A–C (HBM) + D (HBM+HMEM) | `merlin_hashtable_benchmark` (+ `latency_benchmark` for P50) | reproducible |
| **Exp #3a** tab `digest-ablation` (§5.4) | Digest pre-filter speedup (with/without) | `digest_ablation_benchmark` (two compiles) | reproducible |
| **Exp #3b** "Eviction overhead" paragraph (§5.4) | `insert_or_assign` λ=0.5 vs λ=1.0 delta | derived from `merlin_hashtable_benchmark` | reproducible |
| **Exp #3c** tab `cache-quality` (§5.4) | Hit rate × Zipfian α for 5 eviction policies | `e19_cache_quality_sweep` | reproducible |
| **Exp #3d** tab `admission` (§5.4) | Score-based admission control ablation | `p25_admission_control` | reproducible |
| **Exp #3e** concurrency paragraph (§5.4) | Triple-group vs R/W lock | `concurrency_benchmark` (two compiles) | reproducible |
| **Exp #4** tab `single-vs-dual` (§5.5) | Single-bucket vs dual-bucket (LF, top-N, HR) | `e17_single_vs_dual` | reproducible |
| **Exp #4 footnote** L2-residency crossover | Dual ≥ single at cap≲1 M, single > dual at cap ≥ 16 M | `dual_bucket_benchmark` (see `REPRODUCE_e17v3_dual_bucket.md`) | reproducible |
| **Exp #5** fig `company-a-deployment` (§5.6) | Industrial deployment (Company A/B) | *No code*: live production traces from partners | not reviewer-reproducible |

Supporting binaries (not tied to a single paper claim but useful when
auditing the evaluation methodology):

| Binary | Purpose | Paper reference |
|:--|:--|:--|
| `batch_size_benchmark` | Batch size sweep {1 K…10 M} at λ=1.0 | Methodology: justifies 1 M default batch |
| `error_bars_benchmark` | 5-run medians for all APIs / Configs at λ=0.5 | §5.1 "statistical methodology" |
| `latency_benchmark` | Per-batch P50/P95/P99 at λ∈{0.5, 1.0} | §5.3 "Latency and scaling" |
| `bucket_size_benchmark` | Bucket size ∈ {32, 64, 128, 256} | Motivates bucket=128 default |
| `score_strategy_benchmark` | Per-policy throughput + hit rate under Zipfian α=0.99 | Precursor / cross-check for Exp #3c |
| `e9_eviction_comparison` | Built-in `insert_and_evict` vs external sort-and-drop | Design discussion / co-design ablation |
| `e10_dual_bucket_analysis` | Standalone dual-bucket (bucket sweep 64 vs 128) | Precursor to Exp #4 |
| `find_with_missed_keys_benchmark` | `find` behavior with controlled miss rate | Feature demo |

---

## Prerequisites

### Hardware
- NVIDIA GPU with **compute capability ≥ 8.0** (Ampere or newer).
  Paper numbers are on H100 NVL (sm_90). Code also works on
  A100 (sm_80), A6000 (sm_86), H100 SXM5 (sm_90, NVSwitch required).
- ≥ 40 GB free HBM for Configs B/C (dim=32, cap=128 M).
- ≥ 96 GB host DRAM for hybrid Config D (dim=64, cap=128 M).

### Software
- CUDA Toolkit 12.6+ (paper uses 12.9).
- CMake ≥ 3.18, GCC 11+, `make`.
- `nvidia-container-toolkit` + Docker recommended.
  Reference image: `rapidsai/devcontainers:25.10-cpp-cuda12.9-ubuntu24.04`.

### Docker quick start (matches paper environment)
```bash
docker run -d --name hkv-bench --gpus all --shm-size=16g \
  -v "$PWD":/workspace -w /workspace \
  rapidsai/devcontainers:25.10-cpp-cuda12.9-ubuntu24.04 sleep infinity

docker exec -it hkv-bench bash
export PATH=/usr/local/cuda-12.9/bin:$PATH
```

---

## Build

All benchmarks are built from the repository root CMakeLists.txt.
Two CMake options toggle the ablation compiles for Exp #3a and #3e:
`-DDISABLE_DIGEST=ON` and `-DUSE_RW_LOCK=ON`. Change
`-Dsm=<cc>` to match your GPU (H100=90, A100=80, A6000=86).

```bash
# 0. googletest submodule (required by tests; some benchmark targets
#    depend on it through the shared include path).
git submodule update --init --recursive 2>/dev/null \
  || git clone --depth 1 https://github.com/google/googletest.git tests/googletest

# 1. HKV benchmarks (default compile, digest ON, triple-group ON)
mkdir -p build && cd build
cmake -Dsm=90 -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)    # ≈ 8 min on a 32-core host

# 2. External baselines (cloned via CMake FetchContent)
cd ../baselines && mkdir -p build && cd build
cmake -DGPU_ARCH=90 ..
make -j$(nproc)    # pulls WarpCore, BGHT, cuCollections; ≈ 5 min
```

All binaries below are produced in `build/` (HKV) or
`baselines/build/` (baselines).

---

## Reproducing each experiment

### Exp #1 — Load-factor analysis (§5.2, fig `lf-degradation`, tab `lf-comparison`)

**What the paper claims.** HKV's `find` throughput varies < 1 % across
λ ∈ [0.25, 1.00] (3.37–3.40 B-KV/s); WarpCore / BGHT / cuCollections
collapse beyond λ ≈ 0.95; BP2HT silently drops insertions (48 %
success at λ=1.0).

**Binaries.**
- HKV line: `build/e8_hkv_baseline`
- Baselines: `baselines/build/warpcore_bench`, `bght_bench`,
  `p2bht_bench`, `cuco_bench`

**Shared CLI.** All five accept an optional single `<load_factor>`
argument. With no argument they sweep
`{0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00}` (8 points,
5 runs/point).

**Config.** dim=32, capacity=128 M, batch=1 M, EvictStrategy=kLru
for HKV. All baselines use 128-byte values (float32×32) end-to-end
(key + scatter/gather for indirection-based libraries).

**Reproduce (HKV + four baselines, full LF sweep).**
```bash
cd build
./e8_hkv_baseline > e1_hkv.csv

cd ../../baselines/build
for lf in 0.25 0.50 0.75 0.80 0.85 0.90 0.95 1.00; do
  timeout 120 ./warpcore_bench $lf >> e1_warpcore.csv
  timeout 120 ./bght_bench     $lf >> e1_bght.csv
  timeout 120 ./p2bht_bench    $lf >> e1_p2bht.csv
  timeout 120 ./cuco_bench     $lf >> e1_cuco.csv
done
```
`timeout 120` protects against the known hang-at-capacity behavior of
dictionary-semantic baselines — a timeout IS the data point for
λ ≥ 0.95 baseline failures.

**Output format (same CSV across all five).**
```
library,operation,load_factor,run,throughput_bkvs
HKV,find,0.50,1,3.372
...
WarpCore,insert,1.00,0,TIMEOUT   ← paper's "crash zone"
```

**Expected numbers (H100 NVL, median of 5 runs).**

| System | λ=0.50 find | λ=1.00 find |
|:--|--:|--:|
| HKV | 3.37 | 3.40 |
| WarpCore | 2.45 | 0.25 |
| BGHT | 1.31 | 0.92 |
| cuCollections | 0.36 | ~0 (stall/timeout) |
| BP2HT | 1.09 | 1.07 |

Numbers within ± 3 % across 5 runs are consistent with the paper
(§5.1: "CV < 3 % for all data points").

---

### Exp #2 — End-to-end throughput (§5.3, fig `throughput-bar`, latency + hybrid paragraphs)

**What the paper claims.**
- Pure HBM: `find` 3.61–3.89 B-KV/s across Configs A/B/C;
  `find*` ~7.05 B-KV/s (dim-independent).
- Latency: P50 find ≈ 0.30 ms at both λ=0.5 and λ=1.0 (Config B).
- Hybrid (Config D): `find*` retains 96 % of pure-HBM throughput;
  `find` drops to 0.172 B-KV/s (PCIe-bound).

**Binary.** `build/merlin_hashtable_benchmark` — *one* binary that
covers Configs A, B, C (pure HBM) AND Configs D + (dim=64,
cap=512 M, HBM=32 GB) (HBM+HMEM) for all 10 public APIs at
λ ∈ {0.50, 0.75, 1.00}. It hard-codes the paper's configuration
table (see `main()` in `merlin_hashtable_benchmark.cc.cu`).

**Reproduce.**
```bash
cd build
./merlin_hashtable_benchmark > e2_throughput_full.txt 2>&1     # ~30 min

# Latency paragraph (§5.3)
./latency_benchmark > e2_latency.csv 2> e2_latency.log          # ~3 min
```

**Output.** `merlin_hashtable_benchmark` prints Markdown tables with
λ as the first column and one column per API; see
`results-h100-nvl/E12-E14/e12_e13_e14_full_results.txt` for a
reference transcript. `latency_benchmark` emits per-batch latency in
ms as CSV (1 000 batches × {find, insert_or_assign, assign} × {λ=0.5,
1.0}).

**Reviewer check — Config B, λ=0.50 row (excerpt):**

| λ | insert_or_assign | find | find_or_insert | assign | find* | find_or_insert* | insert_and_evict |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 0.50 | 1.73 | 3.72 | 2.37 | 2.90 | 7.05 | 4.85 | 1.52 |

The paper's 7.2 % reduction for `find` at dim=64 vs dim=8 (§5.3)
appears as the Config A→C gradient (3.89 → 3.61 B-KV/s).

---

### Exp #3a — Digest contribution (§5.4, tab `digest-ablation`)

**What the paper claims.** Digest pre-filter contributes 1.65×
(λ=0.5) – 2.61× (λ=1.0) speedup, growing because without digest a
miss scans all 128 bucket slots. Converges to ~1.83 B-KV/s.

**Benchmark uses the miss path.** `digest_ablation_benchmark.cc.cu`
generates lookup keys starting at `4 × init_capacity + 1` so every
query is guaranteed to miss. This is the path the paper measured
(s5 line 215: "without digest every miss compares all 128 keys").
Under a 100%-hit workload the lookup kernel short-circuits on the
first successful key compare and masks the digest contribution,
collapsing the speedup to ~1.0×.

**Binary.** `build/digest_ablation_benchmark`, built twice.

**Reproduce (dual compile + merge).**
```bash
cd build
# 1) With digest (default): skip rebuild if already compiled
./digest_ablation_benchmark > e6_with.csv

# 2) Without digest
cmake -Dsm=90 -DCMAKE_BUILD_TYPE=Release -DDISABLE_DIGEST=ON ..
make digest_ablation_benchmark -j$(nproc)
./digest_ablation_benchmark > e6_no.csv

# Restore default build so subsequent experiments are unaffected
cmake -Dsm=90 -DCMAKE_BUILD_TYPE=Release -DDISABLE_DIGEST=OFF ..
make -j$(nproc)

# Merge (skip no-digest header row)
cat e6_with.csv >  e6_results.csv
tail -n +2 e6_no.csv >> e6_results.csv
```

**Output format.**
```
mode,config,dim,load_factor,run,throughput_bkvs
with_digest,A,8,0.50,1,3.380
no_digest,A,8,0.50,1,2.051
```

**Expected.** At Config B, λ=1.00: with_digest ≈ 4.77, no_digest ≈
1.83 → 2.61× (matches paper tab:digest-ablation).

---

### Exp #3b — Eviction overhead (§5.4, no dedicated binary)

Derived from `merlin_hashtable_benchmark` output: compare the
`insert_or_assign` column at λ=0.50 vs λ=1.00 within each Config.
Paper reports 32–41 % slowdown.

No extra reproduction step — the number is already in Exp #2 output.

---

### Exp #3c — Cache quality under Zipfian skew (§5.4, tab `cache-quality`)

**What the paper claims.** Hit rate sweep over α ∈ {0.50, 0.75, 0.99,
1.25} × {kLru, kLfu, kEpochLru, kEpochLfu, kCustomized}. At
production-typical α=0.99: LFU 88.3 %, LRU-family 83.9 %. All
policies converge above α=1.25.

**Binary.** `build/e19_cache_quality_sweep`.

**CLI.** `./e19_cache_quality_sweep [policy] [alpha]`
(`policy` ∈ `{all, kLru, kLfu, kEpochLru, kEpochLfu, kCustomized}`;
`alpha` is a specific value or `all`). With no args, runs all 5
policies × 4 α = 20 combinations (≈ 90 min on H100 NVL).

**Protocol (per combination).**
1. Fresh table, pre-populate to capacity with sequential keys.
2. 5 × capacity steady-state Zipfian inserts over a 10 × capacity key
   range (seed 42).
3. 5 rounds × 1 M Zipfian finds (seed 12345).
4. Report hit ratio + find throughput.

**Reproduce.**
```bash
cd build
./e19_cache_quality_sweep > e3c_cache_quality.csv 2> e3c.log
```

**Output.**
```
policy,alpha,hit_ratio,total_found,total_queries,find_throughput_bkvs
kLru,0.50,0.186,932341,5000000,2.974
kLru,0.75,0.430,...
```

**Expected.** The LFU row at α=0.99 should be ≈ 0.883 (i.e. 88.3 %),
matching paper tab:cache-quality within ± 0.5 pp run-to-run.

---

### Exp #3d — Admission control (§5.4, tab `admission`)

**What the paper claims.** At a 16 M-slot table filled to λ=0.96,
injecting 4 M new keys with **low** scores changes hit rate by
+0.00 pp (admission rejects them); injecting with **high** scores
degrades hit rate by −21.48 pp (originals displaced).

**Binary.** `build/p25_admission_control`.

**Reproduce.**
```bash
cd build
./p25_admission_control > e3d_admission.csv 2> e3d.log
```

The binary internally runs both conditions (low-score burst,
high-score burst) and prints a single CSV row per condition plus the
baseline hit rate before the burst. `EvictStrategy::kCustomized` is
required to supply explicit scores for the burst.

---

### Exp #3e — Concurrency ablation (§5.4, codesign row "Triple-group")

**What the paper claims.** Triple-group R/U/I locking reaches
2.569 B-KV/s vs 0.535 B-KV/s for an R/W lock under 10-updater
workload (4.8×); gap varies with mix (update-heavy 3.21×,
insert-heavy 1.20×, read-heavy 1.03×).

**Binary.** `build/concurrency_benchmark` (CLI: `triple_group` or
`rw_lock`), built twice.

**Reproduce.**
```bash
cd build
# Triple-group (default)
./concurrency_benchmark triple_group > e3e_triple.csv 2> e3e_triple.log

# R/W lock (ablation compile)
cmake -Dsm=90 -DCMAKE_BUILD_TYPE=Release -DUSE_RW_LOCK=ON ..
make concurrency_benchmark -j$(nproc)
./concurrency_benchmark rw_lock > e3e_rw.csv 2> e3e_rw.log

# Restore default
cmake -Dsm=90 -DCMAKE_BUILD_TYPE=Release -DUSE_RW_LOCK=OFF ..
make -j$(nproc)
```

**Output format.**
```
mode,workload,threads,total_ops,wall_time_s,throughput_bkvs,find_ops,find_bkvs,assign_ops,assign_bkvs,insert_ops,insert_bkvs
triple_group,read_heavy,10,...
```

**Workloads swept.** `read_heavy` (8F/1U/1I), `update_heavy`
(4F/5U/1I), `insert_heavy` (4F/2U/4I), `assign_only` × {1,2,5,10} t,
`assign5_insert5`.

**Paper-spec config.** Constants in `concurrency_benchmark.cc.cu` match
paper Exp #3e (§5.4 line 282): `DIM=16`, `BATCH_SIZE=64*1024`,
`BATCHES_PER_THREAD=200`, `LOAD_FACTOR=0.75`. The paper's 4.8×
triple-group vs R/W-lock ratio at 10 updaters requires the small
dim / small batch combination — a 1 M batch at dim=32 amortizes lock
acquisition across a heavier kernel and masks the contention
behavior that the 4.8× claim depends on. Results under both settings
are archived in `results-h100-nvl/E7{,v2}/` for reference.

---

### Exp #4 — Single- vs dual-bucket (§5.5, tab `single-vs-dual`)

**What the paper claims.**
- First-eviction λ: single 0.633 → dual 0.977 (+54.3 %).
- Top-N score retention: 95.4 % → 99.4 % (+4.05 pp).
- Cache hit ratio: 83.88 % → 84.02 %.
- Throughput trade-off: single ≈ 2.3× faster per op at DRAM scale.

**Binary.** `build/e17_single_vs_dual` (Parts 1–4 correspond to the
four rows of tab:single-vs-dual + Throughput sweep).

**CLI.** `./e17_single_vs_dual [part]` — `part` ∈ {0 = all,
1 = first-eviction LF, 2 = hit ratio, 3 = throughput sweep,
4 = top-N retention}. Full run ≈ 70 min on H100 NVL.

**Config.** Fixed: dim=32, capacity=128 M, bucket_size=128, Pure
HBM, EvictStrategy=kLru (paper Config B).

**Reproduce.**
```bash
cd build
./e17_single_vs_dual > e4_single_vs_dual.csv 2> e4.log
```

**Output.**
```
mode,metric,bucket_size,load_factor,run,value
single,first_eviction_lf,128,,1,0.6334
dual,first_eviction_lf,128,,1,0.9774
single,hit_ratio,128,1.00,1,0.8388
dual,hit_ratio,128,1.00,1,0.8402
single,topN_retention,128,1.00,1,0.9539
dual,topN_retention,128,1.00,1,0.9944
single,insert_or_assign_bkvs,128,0.50,1,0.061
dual,insert_or_assign_bkvs,128,0.50,1,0.027
...
```

### Exp #4 footnote — L2-residency crossover

The paper's footnote ("At sub-L2-resident table sizes…dual-bucket
matches or exceeds single-bucket throughput") is reproduced with
`dual_bucket_benchmark`, which accepts `<capacity> <dim>` and runs
the full LF sweep in both modes. See the dedicated document
[`REPRODUCE_e17v3_dual_bucket.md`](./REPRODUCE_e17v3_dual_bucket.md)
for exact commands and expected tables at cap = 1 Mi, 1 Mi (dim=32),
and 128 Mi.

---

### Exp #5 — Industrial deployment (§5.6)

Not reviewer-reproducible: the Company A / Company B curves are live
production traces from partner deployments. The paper figures were
generated by `hkv-paper-sigmod/figures/fig-company-a-deployment.py`
and `fig-company-a-hit-rate.py` consuming anonymized CSVs that are
not part of this repository.

---

## Full suite in one command

The paper's end-to-end reproduction script is checked in at
[`../scripts/run_all_benchmarks.sh`](../scripts/run_all_benchmarks.sh). It:

1. Builds HKV (`-Dsm=$SM -DCMAKE_BUILD_TYPE=Release`; default `SM=90`).
2. Runs sanity test (`merlin_hashtable_test --gtest_filter="*test_basic"`).
3. Executes every per-experiment binary plus the two ablation recompiles
   (E6 digest, E7 concurrency) in sequence.
4. Builds external baselines and runs Exp #1 LF sweeps.
5. Writes every CSV + progress log under `$RESULTS` (default
   `$HKV_ROOT/results/`).

Environment knobs: `HKV_ROOT`, `SM`, `RESULTS`, `SKIP_BUILD`. Example:

```bash
HKV_ROOT=/workspace SM=90 RESULTS=/workspace/results-h100 \
  bash scripts/run_all_benchmarks.sh
```

Total wall-clock ≈ 6–8 h on H100 NVL. The script tolerates expected
failures (e.g. dictionary-baseline hangs at λ ≥ 0.95 trigger the
120 s timeout by design) and writes `TIMEOUT` / `CRASH_<rc>` rows
so downstream analysis can distinguish "no data" from "data".

Reference raw outputs from the authors' H100 NVL run are in
`results-h100-nvl/` of the paper supplementary archive.

---

## Interpreting throughput numbers

All HKV binaries report `throughput_bkvs` = Billion-KV pairs per
second = `batch_size / wall_time_s × 1e-9`. Baselines emit the same
unit. One "operation" is one key look-up or one (key, value) upsert
including scatter/gather for indirection-based tables.

- Cold-start: every experiment creates a fresh `HashTable` per run
  to avoid cross-run warm-up bleed.
- Warmup: 3–10 warm iterations (per binary) are discarded before
  timing (see `NUM_WARMUP` at the top of each source file).
- Timing primitive: CUDA events on the table's own stream; all
  `cudaStreamSynchronize` calls are outside the timed region.
- Aggregation: paper reports median of 5 runs. Error bars were
  checked against `error_bars_benchmark.cc.cu` (5 runs × 3 Configs
  × 10 APIs at λ=0.5) and found < 3 % CV, hence omitted from plots.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|:--|:--|:--|
| `free HBM is not enough, ignore current benchmark!` | GPU has < 16 GB free when Config B/C/D tries to allocate | Run with `CUDA_VISIBLE_DEVICES` pinned to a clean GPU; close other processes |
| Baseline hangs at high λ | Expected at λ ≥ 0.95 for WarpCore/BGHT/cuCollections | Wrap invocation with `timeout 120` as shown in Exp #1 |
| `dual_bucket_test` fails to compile (`vector<bool>::data()`) | Known bug in test harness, not the benchmark | Skip: `make $TARGET` instead of `make` for benchmarks only |
| Rebuild after ablation toggle shows stale behavior | Ablation flags add `-DDISABLE_DIGEST` / `-DUSE_RW_LOCK` at the compile-unit level; incremental CMake may miss them | Always re-invoke `cmake -D…` before `make` on the toggled target, and pass the exact target (`make digest_ablation_benchmark`) rather than bare `make` |
| `InvalidArgument: dim*sizeof(V) > 896` (dual-bucket only) | Dual-bucket mode imposes a 128 × 8 B = 1024 B vector line budget; 896 B leaves room for key + digest + score on-chip | Reduce `dim` for dual-bucket experiments (Exp #4 uses dim=32 throughout) |

---

## Files in this directory

| File | Role |
|:--|:--|
| `merlin_hashtable_benchmark.cc.cu` | Exp #2 (canonical end-to-end throughput) |
| `latency_benchmark.cc.cu` | Exp #2 latency paragraph |
| `digest_ablation_benchmark.cc.cu` | Exp #3a |
| `e19_cache_quality_sweep.cc.cu` | Exp #3c |
| `p25_admission_control.cc.cu` | Exp #3d |
| `concurrency_benchmark.cc.cu` | Exp #3e |
| `e17_single_vs_dual.cc.cu` | Exp #4 |
| `dual_bucket_benchmark.cc.cu` | Exp #4 footnote (L2 crossover) |
| `e8_hkv_baseline.cc.cu` | Exp #1 (HKV line + LF sweep) |
| `batch_size_benchmark.cc.cu` | Methodology: batch size sweep |
| `error_bars_benchmark.cc.cu` | Methodology: 5-run medians / CV |
| `bucket_size_benchmark.cc.cu` | Methodology: bucket size ∈ {32, 64, 128, 256} |
| `score_strategy_benchmark.cc.cu` | Cross-check for Exp #3c (per-policy throughput) |
| `e9_eviction_comparison.cc.cu` | Design discussion: built-in vs external eviction |
| `e10_dual_bucket_analysis.cc.cu` | Precursor to Exp #4 (bucket sweep 64 vs 128) |
| `find_with_missed_keys_benchmark.cc.cu` | Feature demo (miss-aware find) |
| `benchmark_util.cuh` | Shared CUDA timing / key-generation helpers |
| `REPRODUCE_e17v3_dual_bucket.md` | Detailed L2-crossover reproduction guide |
| `BUILD` | Bazel rules (unused by the paper; CMake is authoritative) |

For cross-binary utilities see `../baselines/` (WarpCore/BGHT/
P2BHT/cuCollections wrappers) and `../tests/` (correctness suite).
