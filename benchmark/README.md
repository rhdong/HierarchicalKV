# HKV paper benchmarks (VLDB 2027)

This branch combines upstream HKV `c79175830d5110b0d76868dc465ee0415ba080e4`
with the historical paper benchmark sources and review audit scripts from
`7b661bacca8088f9e71c2689ef36c0d47f1285a7`. Core library headers and tests remain
upstream. Benchmark sources retain the historical workloads; scripts use the
upstream opt-in build targets, pinned dependencies and target-local ablations.
New builds have new identities and are not the binaries that produced the paper.

**Availability:** `available` means the linked source or record is included;
`unverified` means its exact relationship to a paper number or fresh runtime
has not been established; `restricted` means original production inputs are
not distributed. Source availability is not a claim that every figure reproduces.

## Build from a clean checkout

Use the full commit from the supplemental-material URL as `ARTIFACT_COMMIT`:

```sh
git clone https://github.com/rhdong/HierarchicalKV.git
cd HierarchicalKV
git checkout --detach "$ARTIFACT_COMMIT"
git submodule update --init --recursive
cmake -S . -B build-paper -Dsm=90 -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_BUILD_TYPE=Release -DHKV_BUILD_REPRO_BENCHMARKS=ON
cmake --build build-paper --target paper_benchmarks -j4
```

The public URL will resolve only after this branch is published. Set
`ARTIFACT_COMMIT` before running the checkout command. CUDA, a supported host
C++ compiler, CMake and Git are required even when only compiling without a GPU.
SM 90 above is an explicit H100 build choice; select the architecture for your
GPU. The GoogleTest submodule must be initialized at its checked-in revision.
No CUDA program needs to run to compile `paper_benchmarks`.

`HKV_BUILD_REPRO_BENCHMARKS=ON` is required for optional targets.
`digest_ablation_no_digest` and `concurrency_rw_lock` have target-local
`DISABLE_DIGEST` and `USE_RW_LOCK` definitions; do not use global CMake toggles
or overwrite one binary with another. The root build declares Debug test
targets; optional benchmarks explicitly retain host `-O3`. Record actual
compiler flags rather than inferring every target's flags from the build type.

Prepare the separately built external dictionary baselines:

```sh
cmake -S baselines -B baselines/build-paper -DGPU_ARCH=90
cmake --build baselines/build-paper --target warpcore_bench bght_bench \
  p2bht_bench cuco_bench -j4
```

[Baseline build notes](../baselines/README.md) and
[dependency pins](../baselines/CMakeLists.txt) define fresh builds, not historical
dependency identities. FetchContent requires network access and the pinned cuCollections dependency
requires **CMake 4.0 or newer** (the HKV-only build does not require that upgrade). Dependencies retain
their own licenses. A failed or timed-out baseline is missing data, not zero
throughput and not a successful reproduction.

## Paper experiment index

Commands below run from the repository root after the build above. Redirect
stdout and stderr to a new result directory when collecting evidence. They are
GPU commands; reserve appropriate GPU resources before executing them. Default
paper-scale workloads can require substantial GPU and host memory. The `build-paper/`
paths are generated CMake targets, not binaries distributed in Git.

| Paper experiment | Binary/script | Command | Paper figure/table | Expected output / recorded provenance and status |
|---|---|---|---|---|
| Exp #1: load factor | [e8_hkv_baseline](e8_hkv_baseline.cc.cu), [baseline wrappers](../baselines/) | `"build-paper/e8_hkv_baseline"`; run the explicit LF loop below for each baseline | `fig:lf-degradation`, `tab:lf-comparison` | CSV by API/backend/LF. Source **available**; `E16-lf-degradation/e16_all_combined.csv` (author archive) and exact measured binary identities **unverified/not bundled**; independent-process N **unverified**. Retain timeout/failure rows. |
| Exp #2: API and hybrid storage | [merlin_hashtable_benchmark](merlin_hashtable_benchmark.cc.cu) | `"build-paper/merlin_hashtable_benchmark"` | `fig:throughput-bar`; hybrid paragraph | API/config/LF stdout. Source **available**; the paper uses author archive `E12-E14/e12_e13_e14_optimized_find_star.txt`. Exact historical binary identity and independent-process N **unverified/not bundled**. |
| Exp #2: latency and scaling | [latency_benchmark](latency_benchmark.cc.cu) | `"build-paper/latency_benchmark"` | Latency/scaling paragraph | `E4/e4_results.csv` (author archive); fixed-capacity latency distribution; source **available**. 1,000 batches are within-process. Separate capacity sweep and historical independent N **unverified**; this command does not run a capacity sweep. |
| Exp #2: native full service | [native TBE guide](native_tbe/README.md#separate-recorded-scopes); pinned RecSys overlay | `python -m benchmark.criteo_experiment.native_tbe_comparison` with full prepare/run/summarize argv, env and order in [shared-cache-five-pair.json](native_tbe/shared-cache-five-pair.json) | `tab:native-tbe-comparison`, lookup service row | [timings-new-sdk-five-pair.csv](native_tbe/timings-new-sdk-five-pair.csv) **available**. **N=5 independent pairs, 10 fresh processes**. Includes transfers/miss/backing/final flush. Historical service uses its older overlay; preserve configured-native diagnostic qualification. Fresh-build checker requires evidenced identity rebinding; new-build reproduction **unverified**. |
| Exp #2: native resident forward | [native TBE guide](native_tbe/README.md#separate-recorded-scopes); current pinned overlay | Same module with full argv/env/order in [cache-forward-host-five-pair.json](native_tbe/cache-forward-host-five-pair.json) | Same table, resident forward row | [timings-cache-forward-host.csv](native_tbe/timings-cache-forward-host.csv) **available**. **N=5 independent pairs, 10 fresh processes**. GPU-resident inputs; 20 warmups/200 forwards per process are internal repetitions. Native forward + completion synchronization timer. Fresh-build checker identity remains **unverified**. |
| Exp #3a: digest | [digest_ablation_benchmark](digest_ablation_benchmark.cc.cu) | `"build-paper/digest_ablation_benchmark"`; `"build-paper/digest_ablation_no_digest"` | `tab:digest-ablation` | CSV by config/LF/run. Source **available**, no CLI arguments, pure-miss queries. Current-table historical row lineage **unverified**. Five in-process repetitions; independent-process N **unverified**. |
| Exp #3b: eviction overhead | Reuse API benchmark | Compare `insert_or_assign` rows at LF .5 and 1.0 within the same config | Eviction-overhead paragraph | Source **available**; historical identity follows Exp #2. Derived comparison adds no independent samples. |
| Exp #3c: score policy | [e19_cache_quality_sweep](e19_cache_quality_sweep.cc.cu) | `"build-paper/e19_cache_quality_sweep" all all`; optional `[policy] [alpha]` | `tab:cache-quality` | CSV across five policies/Zipfian skew settings; source **available**. `E19/e19_results_with_throughput.csv` (author archive) / policy identity **unverified/not bundled**. Five query rounds do not establish independent-process N. |
| Exp #3d: admission | [p25_admission_control](p25_admission_control.cc.cu) | `"build-paper/p25_admission_control"` | `tab:admission` | Low/high-score burst hit-rate output. Source **available**; author-held `e3d_admission.csv` binding and independent N **unverified/not bundled**. Hit preservation alone does not prove a zero-admission counter. |
| Exp #3e: concurrency | [concurrency_benchmark](concurrency_benchmark.cc.cu); independent RW-lock target | `"build-paper/concurrency_benchmark" triple_group`; `"build-paper/concurrency_rw_lock" rw_lock` | Concurrency paragraph; `tab:codesign-ablation` | Eight workload rows. Compile definition selects synchronization; argv labels run. Source **available**; candidate `E7v2-dim16-cached/e7v2_summary.csv` does not bind current prose; exact cohort mapping and independent N **unverified**. Batches/thread are internal repetitions. |
| Exp #4: single/dual bucket | [e17_single_vs_dual](e17_single_vs_dual.cc.cu) | `"build-paper/e17_single_vs_dual" 0`; parts `1,2,3,4` select first eviction, hit rate, throughput, retention | `tab:single-vs-dual` | Phase/metric CSV. Source **available**; historical binary identity **unverified/not bundled** (`E17v2/e17_part3_rerun.csv`, author archive). Paper throughput used a customized-score historical cohort; do not relabel it as the current source LRU run. Five throughput repeats are within-process. |
| Exp #3–4: co-design synthesis | Cross-reference ablation/bucket targets | Reuse preceding outputs; no separate benchmark | `tab:codesign-ablation` | Numeric rows inherit source experiment status; structural/qualitative rows are explanatory. No additional dataset or independent N. |
| Exp #5: deployment | [Previously public anonymous example](company_a_hkv_realtime/README.md) | No public production-system reproduction command | `fig:company-a-deployment`; Company B prose | Production datasets/pipelines **restricted**. Anonymous example does not establish full production-figure or Company B reproduction. Production independent-run count not public. |

Exp #1's explicit LF grid (each stdout file contains that invocation's samples):

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

Retain failed CSVs/logs and `.failed` statuses; never replace
failed arms with synthetic samples. No historical paper-scale command was run
on a GPU during migration.


## Native TBE in Exp #2

Follow [native_tbe/README.md](native_tbe/README.md) for the actual FBGEMM/TBE
comparison and the pinned DynamicEmb overlay. The synthetic
[`tbe_like_cache_baseline.cc.cu`](tbe_like_cache_baseline.cc.cu) is a separate
cache-pipeline diagnostic and cannot substitute for native TBE.

| Recorded scope | Public command recipe and record | Evidence boundary |
| --- | --- | --- |
| Full service, including misses/backing/transfer/final flush | [shared-cache-five-pair.json](native_tbe/shared-cache-five-pair.json), [timings-new-sdk-five-pair.csv](native_tbe/timings-new-sdk-five-pair.csv) | Five independent paired seeds/processes; use the historical overlay pinned in the recipe. |
| Resident single-batch cache forward | [cache-forward-host-five-pair.json](native_tbe/cache-forward-host-five-pair.json), [timings-cache-forward-host.csv](native_tbe/timings-cache-forward-host.csv) | Five independent paired seeds/processes; 20 warm-ups and 200 forwards per process are not additional independent samples. |

The historical full-service overlay is available at its [fixed public commit](https://github.com/rhdong/HierarchicalKV/blob/cb04241a2977aa0bd353aa5e600ef62a0605310d/third_party/recsys-examples/legacy-250aaa.patch);
verify the recipe hash before applying it to its separate RecSys checkout.
Both recipes contain prepare/run/summarize argv templates, alternating backend
order and input contracts. The included CSVs are historical summaries; original
input blobs, native binaries and per-arm raw logs are not bundled. Inputs are
synthetically generated by the overlay, not production Criteo data. Do not merge
the two timing scopes, claim equivalent total memory/policies, or turn repeated
forwards into a larger independent N.

Fresh native execution remains **unverified**: the current resident checker
[`validate_cache_forward_result.py`](native_tbe/validate_cache_forward_result.py)
also hardcodes the historical extension SHA, so editing its JSON pins alone
does not bind a new build. Full-output, input, pair, library, dirty/backing and
flush checks must remain intact; no validation gate was weakened here. Keep
historical validation and any future freshly bound results separate.

## Historical evidence and limitations

The original benchmark branch contains code and this anonymous statistics
snapshot, not a complete public raw-result archive. The historical records
listed in the index are archive-relative names for provenance, not local
paths to files shipped in this repository. They must not be treated as successful
checks of this branch. Missing exact source/binary/dependency identity, per-run
records or paper-table correspondence stays `unverified`.

The [Company A snapshot](company_a_hkv_realtime/README.md) is unchanged from
the previously public branch and contains anonymous aggregate statistics.
Original enterprise inputs, partner configurations and Company B traces remain
`restricted`; the snapshot alone cannot regenerate both industrial figures.
Its `sanitize.sh` verifies/copies only the reviewed public bytes and does not
sanitize new enterprise logs.

Throughput units must be read from each source: for example,
[`e8_hkv_baseline.cc.cu`](e8_hkv_baseline.cc.cu) and
[`merlin_hashtable_benchmark.cc.cu`](merlin_hashtable_benchmark.cc.cu) divide
by **2^30**, despite the historical `throughput_bkvs` name; other cohorts use
10^9. Keep denominators, successful/attempted operation counts, hit/miss mix,
timing scope, warm-ups, seeds, cold starts and independent processes explicit.
Within-process measurement loops do not establish an independent-run count.

## Additional historical diagnostics

The [historical suite runner](../scripts/run_all_benchmarks.sh) runs
microbenchmarks and dictionary baselines only, not native TBE or production
replays. It initializes the pinned submodule and uses separate ablation targets
without deleting build/dependency directories; failures still require inspection.
Its measured outputs are new results with the current core, not guaranteed paper
values. This migration did not run the suite on a GPU.

`scripts/run_review_response_*.sh` preserve the historical review audits, with
opt-in CMake and `HKV_REPRO_*` option mappings. The inherited upstream
`scripts/run_repro_*.sh` were written for newer benchmark CLIs/output schemas;
their runtime compatibility with the historical benchmark sources is
**unverified**. Use the explicit indexed binary commands for paper workloads.
Hash-variant and read-only audits also need their output/oracle and core-path
support checked before making a performance claim.

[`REPRODUCE_e17v3_dual_bucket.md`](REPRODUCE_e17v3_dual_bucket.md) retains
optional capacity-crossover commands; it is not an additional current-paper
experiment or verified historical result table.
