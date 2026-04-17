#!/bin/bash
# =============================================================================
# Sanitize Company-A HKV realtime stat log for double-blind release
# -----------------------------------------------------------------------------
# Removes:
#   * internal source-file paths and line numbers
#   * internal class/method names revealing architecture
#   * partner-internal config paths, service names, table IDs
#   * identifying acronyms ("weps", "amlt", "mmfinder*", "qspace", "shenzhen")
# Preserves:
#   * timestamps
#   * GPU IDs, PIDs, device pointers, byte/time counters, table sizes
#   * high-level operation verbs (upsert / find / evict)
#
# Usage: bash sanitize.sh <input.txt> <output.txt>
#
# Compatible with GNU sed and BSD/macOS sed (uses perl for word-boundary pass).
# =============================================================================

set -euo pipefail
IN="${1:?input log}"
OUT="${2:?output file}"

# Pass 1: file path / tag rewrites (order matters — long patterns first)
sed -E \
  -e 's#(\./)?(rflow|tensorflow)_recommenders_addons/[A-Za-z0-9_./]+:[0-9]+#<internal>#g' \
  -e 's#templ/customized_table/[A-Za-z0-9_./]+:[0-9]+#<internal>#g' \
  -e 's#(async_multilevel_table|context)\.(hh|cc|h|cpp):[0-9]+#<internal>#g' \
  -e 's#\[AsyncMultiLevelTable\]#[HybridTable]#g' \
  -e 's#AsyncMultiLevelTable#HybridTable#g' \
  -e 's#\[UpsertL2Buffer\]#[UpsertStage2]#g' \
  -e 's#\[UpsertL1Buffer\]#[UpsertStage1]#g' \
  -e 's#\[DumpL2Buffer\]#[DumpStage2]#g' \
  -e 's#\[DoDeltaUpdate\]#[DeltaUpdate]#g' \
  -e 's#\[DoPrepare\]#[Prepare]#g' \
  -e 's#\[FindKeysFromWeps\]#[FindKeysFromStore]#g' \
  -e 's#FindKeysFromWeps#FindKeysFromStore#g' \
  -e 's#amlt_weps_#config_#g' \
  -e 's#weps_conf:[[:space:]]*[^ ,]+#store_conf: <redacted>#g' \
  -e 's#weps_table_id:[[:space:]]*[0-9]+#store_table_id: <redacted>#g' \
  -e 's#weps_input_name:[[:space:]]*[A-Za-z0-9_/]+#store_input_name: <redacted>#g' \
  -e 's#weps_thread_num#store_thread_num#g' \
  -e 's#embedding/unified_embedding_dim64_gpu#embedding/company_a_dim64#g' \
  -e 's#mmfindermetisdroutepsstor_[A-Za-z0-9_.]+#<redacted>#g' \
  -e 's#mmkvcfgsvr_[A-Za-z0-9_.]+#<redacted>#g' \
  -e 's#/home/qspace/etc/client/shenzhen/[A-Za-z0-9_.\-]+#<redacted>#g' \
  -e 's#mmfindermetisdroutepsstor#<redacted>#g' \
  -e 's#mmkvcfgsvr#<redacted>#g' \
  -e 's#qspace#<redacted>#g' \
  -e 's#shenzhen#<redacted>#g' \
  "$IN" > "$OUT.tmp1"

# Pass 2: word-boundary replacements for bare "weps" / "Weps" / residual TFRA
# prefixes that were fragmented by concurrent log writes. Perl handles \b
# identically across platforms.
perl -pe '
  s/\btensorflow_recommenders_addons\S*/<internal>/g;
  s/\brflow_recommenders_addons\S*/<internal>/g;
  s/\bweps\b/store/g;
  s/\bWeps\b/Store/g;
' "$OUT.tmp1" > "$OUT"
rm -f "$OUT.tmp1"

echo "Wrote: $OUT ($(wc -l < "$OUT") lines)"

# Verification pass: fail loudly if any known-sensitive token leaks through
LEAKS=$(grep -cE \
  'tensorflow_recommenders_addons|rflow_recommenders_addons|hybrid_lookup_impl|customized_table/async_multilevel_table|\.(cc|hh):[0-9]+|AsyncMultiLevelTable|FindKeysFromWeps|amlt_weps_|[^a-z]weps[^a-z_]|[^a-z]Weps[^a-z_]|mmfinder|mmkvcfgsvr|qspace|shenzhen|unified_embedding_dim64_gpu' \
  "$OUT" || true)
if [ "$LEAKS" -gt 0 ]; then
  echo "FAIL: sanitization missed $LEAKS occurrences. Review patterns." >&2
  exit 1
fi
echo "OK: no sensitive tokens remaining."
