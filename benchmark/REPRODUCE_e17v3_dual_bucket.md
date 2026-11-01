# E17v3: Reproducing Single vs Dual-Bucket Throughput Comparison

This is the historical E17v3 experiment recipe. The commands below exercise
the upstream `dual_bucket_benchmark` at the listed capacities. Historical H100
tables and their original per-run provenance are not bundled in this branch;
their paper-number correspondence is **unverified** for this migrated source.
No GPU run was performed during this migration. Select an SM and CUDA toolkit
appropriate to your GPU, and record source, toolchain and independent-run count.

## Build

```bash
mkdir -p build && cd build
cmake -Dsm=90 -DCMAKE_BUILD_TYPE=Release ..
make dual_bucket_benchmark -j$(nproc)
```

## Run

The binary takes two arguments: `<capacity> <dim>`. It runs the full load-factor
sweep `[0.25, 0.50, 0.75, 0.90, 0.95, 1.00]` in `kThroughput` (single-bucket) mode,
then repeats in `kMemory` (dual-bucket) mode, reusing the same table across load
factors. Metric: Mops/s (million operations per second) for `insert_or_assign` and
`find`.

```bash
# Config 1: PR #246 default (L2-resident, small table)
./dual_bucket_benchmark 1048576 64

# Config 2: L2-resident, dim=32 (isolate dim effect)
./dual_bucket_benchmark 1048576 32

# Config 3: paper scale (DRAM-bound, 128Mi cap, dim=32)
./dual_bucket_benchmark 134217728 32
```

The source prints single/dual-bucket load-factor sweeps. Capture its actual
output and units; do not relabel output from this build as a historical sample.
See [the experiment index](README.md) for evidence availability and limitations.
