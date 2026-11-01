# Company A: previously public anonymous statistics (Exp #5)

`hkv_realtime.log` is the unchanged 9,840-line anonymous statistics snapshot
previously published in the historical paper branch (`7b661bacca8088f9e71c2689ef36c0d47f1285a7`).
Its SHA256 is `379557f217500026b41c9d2a41e3c76ad984eb101ac47d4770e8d298ba881646`.
This branch adds no original enterprise data.

The snapshot contains aggregate operation counts/timings, timestamps, GPU indices,
process/thread IDs, transient device addresses and anonymous table labels. A full
text/field audit found no network addresses, personal paths, credentials, original
partner identifiers, or user/feature ID payloads. This is a bounded disclosure
review, not permission to publish any additional production input.

Full production traces, training data, configurations and Company B data remain
**restricted**. The snapshot alone does not reproduce both companies' paper
curves; the exact plot-to-log and independent-run binding is **unverified**.
Do not attempt to identify the partner from the remaining operational counters.

`sanitize.sh` now only verifies and copies this exact already-public snapshot;
it rejects different input bytes and existing destinations. It cannot sanitize
new production logs. The original partner-specific cleaning rules are omitted
because the rules themselves exposed identifying names and paths.

From this directory, inspect aggregate counters without reconstructing any
private inputs:

```sh
rg 'keys to device' hkv_realtime.log -m 3
rg 'find from device' hkv_realtime.log -m 3
rg -F '[find] [step' hkv_realtime.log -m 3
```

For device-cache statistics, `found / (found + left)` describes only that
logged lookup stage. Do not silently equate it with end-to-end cache hit rate
or infer paper latency percentiles without the original aggregation protocol.
The repository license applies as in the previously published branch.
