# HKV Paper Benchmarks (VLDB 2027)

This branch accompanies the paper *HKV: A GPU Hash Table with Cache Semantics
for Continuous Online Embedding Storage* (PVLDB Vol. 20). It is based on
upstream HierarchicalKV
[`c7917583`](https://github.com/NVIDIA-Merlin/HierarchicalKV/tree/c79175830d5110b0d76868dc465ee0415ba080e4)
and adds the benchmark sources, external baselines, and scripts used for the
paper's experiments. The core library (`include/`) and tests are identical to
upstream.

| Path | Contents |
|---|---|
| `benchmark/` | HKV benchmark sources for Exp #1–#5 (this guide) |
| `benchmark/native_tbe/` | Native FBGEMM TBE comparison recipes and recorded results |
| `baselines/` | External GPU hash table baselines (WarpCore, BGHT, BP2HT, cuCollections) |
| `scripts/` | Suite runners and review-response audits |

## Requirements

- NVIDIA GPU. The paper uses an H100 NVL (SM 90).
- CUDA toolkit, a supported host C++ compiler, CMake, and Git.
- CMake 4.0 or newer for the external baselines only (required by the pinned
  cuCollections dependency); network access for CMake FetchContent.

All benchmark targets on this branch compile from a clean checkout with
CUDA 13.0.88, GCC 12.3, CMake 3.31.8 (HKV targets), and CMake 4.4.4 (baselines).

## Build

```sh
git clone https://github.com/rhdong/HierarchicalKV.git
cd HierarchicalKV
git checkout paper-benchmark        # or a specific commit for a fixed revision
git submodule update --init --recursive
cmake -S . -B build-paper -Dsm=90 -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_BUILD_TYPE=Release -DHKV_BUILD_REPRO_BENCHMARKS=ON
cmake --build build-paper --target paper_benchmarks -j4
```

Set the SM architecture for your GPU. `HKV_BUILD_REPRO_BENCHMARKS=ON` enables
the optional paper targets. The ablation binaries `digest_ablation_no_digest`
and `concurrency_rw_lock` carry their own target-local definitions
(`DISABLE_DIGEST`, `USE_RW_LOCK`); do not set these globally.

External baselines:

```sh
cmake -S baselines -B baselines/build-paper -DGPU_ARCH=90
cmake --build baselines/build-paper --target warpcore_bench bght_bench \
  p2bht_bench cuco_bench -j4
```

See the [baseline build notes](../baselines/README.md) and
[dependency pins](../baselines/CMakeLists.txt). Each dependency keeps its own
license.

## Experiment index

Run commands from the repository root after building. All are GPU workloads;
paper-scale defaults need substantial GPU and host memory. Redirect stdout and
stderr to a results directory of your choice.

| Experiment | Paper | Source | Command |
|---|---|---|---|
| Exp #1: Load factor | §5.2, Fig. 6, Table 5 | [e8_hkv_baseline](e8_hkv_baseline.cc.cu), [baselines](../baselines/) | [Load-factor sweep](#exp-1-load-factor-sweep) below |
| Exp #2: API throughput, hybrid storage | §5.3, Fig. 7 | [merlin_hashtable_benchmark](merlin_hashtable_benchmark.cc.cu) | `build-paper/merlin_hashtable_benchmark` |
| Exp #2: Latency | §5.3 | [latency_benchmark](latency_benchmark.cc.cu) | `build-paper/latency_benchmark` |
| Exp #2: Embedding lookup vs. native TBE | §5.3, Table 6 | [native_tbe](native_tbe/README.md) | [Native TBE comparison](#exp-2-native-tbe-comparison) below |
| Exp #3a: Digest pre-filter | §5.4, Table 7 | [digest_ablation_benchmark](digest_ablation_benchmark.cc.cu) | `build-paper/digest_ablation_benchmark`<br>`build-paper/digest_ablation_no_digest` |
| Exp #3b: Eviction overhead | §5.4 | [merlin_hashtable_benchmark](merlin_hashtable_benchmark.cc.cu) | Compare `insert_or_assign` at λ=0.5 vs. λ=1.0 (Exp #2 output) |
| Exp #3c: Cache quality | §5.4, Table 8 | [e19_cache_quality_sweep](e19_cache_quality_sweep.cc.cu) | `build-paper/e19_cache_quality_sweep all all` |
| Exp #3d: Admission control | §5.4, Table 9 | [p25_admission_control](p25_admission_control.cc.cu) | `build-paper/p25_admission_control` |
| Exp #3e: Concurrency | §5.4, Table 11 | [concurrency_benchmark](concurrency_benchmark.cc.cu) | `build-paper/concurrency_benchmark triple_group`<br>`build-paper/concurrency_rw_lock rw_lock` |
| Exp #4: Single vs. dual bucket | §5.5, Table 10 | [e17_single_vs_dual](e17_single_vs_dual.cc.cu) | `build-paper/e17_single_vs_dual 0` |
| Co-design summary | Table 11 | Derived from Exp #3–#4 | No separate benchmark |
| Exp #5: Industrial deployment | §5.6, Fig. 8 | [company_a_hkv_realtime](company_a_hkv_realtime/README.md) | Production data not public ([see notes](#notes-on-results)) |

Arguments: `e19_cache_quality_sweep` also accepts `[policy] [alpha]`;
`e17_single_vs_dual` parts `1`–`4` select first eviction, hit rate,
throughput, and retention. The Exp #4 throughput reported in the paper used a
customized-score configuration; the default command runs the source's current
policy. `latency_benchmark` measures a fixed-capacity latency distribution and
does not run a capacity sweep.

### Exp #1: Load-factor sweep

```sh
mkdir -p results-paper/exp1
for lf in .25 .50 .75 .80 .85 .90 .95 1.00; do
  build-paper/e8_hkv_baseline "$lf" > "results-paper/exp1/hkv-$lf.csv" 2> "results-paper/exp1/hkv-$lf.log"
  for backend in warpcore bght p2bht cuco; do
    rc=0
    timeout 120 "baselines/build-paper/${backend}_bench" "$lf" > "results-paper/exp1/$backend-$lf.csv" 2> "results-paper/exp1/$backend-$lf.log" || rc=$?
    if [ "$rc" -ne 0 ]; then printf '%s\n' "$rc" > "results-paper/exp1/$backend-$lf.failed"; fi
  done
done
```

A baseline that fails or times out is missing data, not zero throughput; keep
its logs and `.failed` status.

### Exp #2: Native TBE comparison

Follow [native_tbe/README.md](native_tbe/README.md) for the FBGEMM TBE
comparison and the pinned DynamicEmb overlay. Two timing scopes are recorded
separately and must not be merged:

| Scope | Recipe and recorded results | Samples |
|---|---|---|
| Full lookup service (misses, backing store, transfers, final flush) | [shared-cache-five-pair.json](native_tbe/shared-cache-five-pair.json), [timings-new-sdk-five-pair.csv](native_tbe/timings-new-sdk-five-pair.csv) | 5 independent pairs (10 fresh processes) |
| Resident single-batch forward | [cache-forward-host-five-pair.json](native_tbe/cache-forward-host-five-pair.json), [timings-cache-forward-host.csv](native_tbe/timings-cache-forward-host.csv) | 5 independent pairs (10 fresh processes) |

The full-service scope uses the historical overlay at its
[fixed public commit](https://github.com/rhdong/HierarchicalKV/blob/cb04241a2977aa0bd353aa5e600ef62a0605310d/third_party/recsys-examples/legacy-250aaa.patch);
verify the recipe hash before applying it. Inputs are synthetic, generated by
the overlay. The synthetic [tbe_like_cache_baseline](tbe_like_cache_baseline.cc.cu)
is a separate diagnostic, not native TBE.

## Notes on results

- **Paper numbers.** The paper's results were measured on an H100 NVL with the
  historical builds. Fresh builds from this branch use the current toolchain and
  produce new measurements; absolute values depend on GPU, driver, and CUDA
  version. Apart from the native TBE summaries in `native_tbe/`, the raw paper
  result archives are not included in this repository.
- **Throughput units.** [e8_hkv_baseline](e8_hkv_baseline.cc.cu) and
  [merlin_hashtable_benchmark](merlin_hashtable_benchmark.cc.cu) divide by
  2^30 despite the `throughput_bkvs` field name; other benchmarks use 10^9.
- **Repetitions.** In-process loops (batches, query rounds, warm-ups, repeated
  forwards) are repetitions within one run, not independent runs.
- **Production data.** Company A and Company B production inputs are not
  public. [company_a_hkv_realtime](company_a_hkv_realtime/README.md) holds the
  previously published anonymized aggregate statistics; its `sanitize.sh` only
  verifies that snapshot and is not a general log sanitizer. The snapshot alone
  cannot regenerate the industrial figure.
- **Native TBE validation.** The resident-forward checker
  [validate_cache_forward_result.py](native_tbe/validate_cache_forward_result.py)
  pins the historical extension hash. Validating a fresh build requires binding
  the new build's identity; editing the JSON pins alone is not sufficient.

## Additional scripts

- [scripts/run_all_benchmarks.sh](../scripts/run_all_benchmarks.sh) runs the
  microbenchmarks and dictionary baselines (not native TBE or production replays).
- `scripts/run_review_response_*.sh` run the review-response audits.
- The upstream `scripts/run_repro_*.sh` were written for newer benchmark CLIs;
  their compatibility with these sources has not been verified. Use the
  commands in the experiment index for paper workloads.
- [REPRODUCE_e17v3_dual_bucket.md](REPRODUCE_e17v3_dual_bucket.md) contains
  optional capacity-crossover commands beyond the paper's experiments.
