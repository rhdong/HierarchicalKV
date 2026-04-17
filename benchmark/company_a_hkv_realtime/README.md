# Company A — HKV Realtime Stat Log (Exp #5 raw data)

This directory ships the anonymized realtime statistics log that backs
the **Company A** (social-network) production deployment reported in
§5.6 of the HKV SIGMOD paper (Exp #5, figure `company-a-deployment`).
The log was captured by the HKV runtime on an 8-GPU training host
during a live continuous-training run; per-batch upsert / find /
eviction counts and wall-time costs are preserved verbatim so any
reviewer can re-derive the paper's hit-rate and latency claims.

## Files

| File | Purpose |
|:--|:--|
| `hkv_realtime.log` | Sanitized raw log (9 840 lines, 8 GPUs, 14 min window) |
| `sanitize.sh` | Sanitization pipeline used to produce `hkv_realtime.log` |

## Provenance and anonymization

- Captured on an 8-GPU training server at partner Company A, 2026-04-15.
- Embedding dim = 64 (`embedding/company_a_dim64`, unified embedding
  table).
- HBM budget 27 % of full vocabulary (matches paper §5.6).
- `sanitize.sh` removes the following categories before release:
    - Internal source-file paths and line numbers → `<internal>`.
    - Internal class / method names that expose the partner's storage
      architecture (`AsyncMultiLevelTable`, `UpsertL2Buffer`,
      `DumpL2Buffer`, `DoDeltaUpdate`, `FindKeysFromWeps`, etc.) →
      generic equivalents (`HybridTable`, `UpsertStage2`,
      `DumpStage2`, `DeltaUpdate`, `FindKeysFromStore`).
    - Partner-proprietary configuration path (`/home/.../<internal
      service>.cache`), internal service codenames, table IDs, and
      location tokens → `<redacted>`.
    - Internal config prefix `amlt_weps_` → `config_`.
    - Standalone acronym `weps` / `Weps` → `store` / `Store`.
    - Real table name `unified_embedding_dim64_gpu` → neutral
      `company_a_dim64`.
- Retained: timestamps, GPU indices (0–7), PIDs/TIDs, device pointers
  (ephemeral VM addresses), table sizes, upsert / evict / find counts,
  per-stage wall-time in microseconds.

`sanitize.sh` asserts a zero-leak fail-closed check before writing the
output, so re-running it against the same input is deterministic.

## How the paper claims are derived from this log

The paper reports the following for Company A at steady state (§5.6):

| Metric | Value | Log source |
|:--|:--|:--|
| GPU cache hit rate | 83 % | `[find_impl] find from device … found X, left Y`; hit rate = found / (found + left missing after both device+host) |
| Per-batch find latency warm-up | 392 ms → 95 ms in 6 min | `[find] [step N] find … total Nus, len M` |
| Continuous eviction (every batch) | non-zero `left` on every upsert | `[upsert] upsert N keys to device(...) left M cost Tus` |

### Quick grep recipes

```bash
# Per-batch upsert + per-batch eviction count (cache churn)
grep 'keys to device' hkv_realtime.log | head
# ...yields: "upsert N keys to device(0x...) left M cost Tus"
# where M is the number evicted/spilled to host.

# Per-batch find miss and latency
grep 'find from device' hkv_realtime.log | head
# ...yields: "find from device cost Tus, found F, left L, device table
# size: S, dim: 64, step: N".  Hit rate (device) = F / (F + L).

# End-to-end per-batch find wall-time (used for fig 4.1b)
grep '\[find\] \[step' hkv_realtime.log | head
# ...yields: "[step N] find wait Wus, impl Ius, total Tus, len M"
```

### Post-processing tip

The log interleaves 8 GPUs. Filter by `GPU ID:6` (or any fixed GPU)
for a single-stream view, as the paper does. Steps are numbered
monotonically per GPU via the `step N` suffix on `[find]` and the
`info.step` field on `[Prepare]`; pair them by step to align the
find-time with its upsert backdrop.

## License and reuse

These measurements are produced by the Company A deployment and are
released under the same Apache-2.0 license as the rest of the HKV
repository solely to support SIGMOD reviewer reproduction. Do not
attempt to de-anonymize the partner or recover the redacted internal
identifiers.
