#!/usr/bin/env bash
# Review-response TBE-like baseline runner.
#
# Produces CSV artifacts for the faithful TBE-like cache baseline:
#   - 32-way set-associative cache by default
#   - 128B value rows
#   - staged lookup/miss/victim/replacement/value-returning timings
#   - LF sweep: 0.25, 0.50, 0.75, 1.00
#
# Usage:
#   RESULTS=/workspace/results/review-response-tbe-like SM=90 \
#     bash scripts/run_review_response_tbe_like.sh --mode quick --policy lru
#
#   RESULTS=/workspace/results/review-response-tbe-like SM=90 \
#     bash scripts/run_review_response_tbe_like.sh --mode full --policy all

set -euo pipefail

HKV_ROOT="${HKV_ROOT:-$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}"
SM="${SM:-90}"
RESULTS="${RESULTS:-$HKV_ROOT/results/review-response-tbe-like}"
MODE="quick"
POLICY="lru"
LFS="0.25,0.50,0.75,1.00"
MISS_RATE="0.25"
ASSOC="32"
VALUE_BYTES="128"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: run_review_response_tbe_like.sh [options]

Options:
  --mode quick|full       quick uses small local sanity settings; full uses
                          Config-B-sized capacity/batch for GPU runs.
  --quick                 Alias for --mode quick.
  --full                  Alias for --mode full.
  --policy lru|lfu|all    Default: lru.
  --lf LIST               Comma-separated LF sweep. Default: 0.25,0.50,0.75,1.00.
  --miss-rate P           Default: 0.25.
  --assoc N               Default: 32.
  --value-bytes N         Default: 128.
  --dry-run               Print planned build/run commands without executing.
  --                      Pass remaining args to tbe_like_cache_baseline.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#--mode=}"
      shift
      ;;
    --quick)
      MODE="quick"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --policy)
      POLICY="$2"
      shift 2
      ;;
    --policy=*)
      POLICY="${1#--policy=}"
      shift
      ;;
    --lf)
      LFS="$2"
      shift 2
      ;;
    --lf=*)
      LFS="${1#--lf=}"
      shift
      ;;
    --miss-rate)
      MISS_RATE="$2"
      shift 2
      ;;
    --miss-rate=*)
      MISS_RATE="${1#--miss-rate=}"
      shift
      ;;
    --assoc)
      ASSOC="$2"
      shift 2
      ;;
    --assoc=*)
      ASSOC="${1#--assoc=}"
      shift
      ;;
    --value-bytes)
      VALUE_BYTES="$2"
      shift 2
      ;;
    --value-bytes=*)
      VALUE_BYTES="${1#--value-bytes=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

case "$MODE" in
  quick|full) ;;
  *)
    echo "Invalid --mode: $MODE" >&2
    exit 1
    ;;
esac

case "$POLICY" in
  lru|lfu|all) ;;
  *)
    echo "Invalid --policy: $POLICY" >&2
    exit 1
    ;;
esac

if [ -d /usr/local/cuda-12.9/bin ]; then
  export PATH=/usr/local/cuda-12.9/bin:$PATH
fi

if command -v nproc >/dev/null 2>&1; then
  JOBS="${JOBS:-$(nproc)}"
elif command -v sysctl >/dev/null 2>&1; then
  JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"
else
  JOBS="${JOBS:-8}"
fi

BUILD_DIR="$HKV_ROOT/build"
RUN_CMD=(
  "$BUILD_DIR/tbe_like_cache_baseline"
  --mode "$MODE"
  --policy "$POLICY"
  --lf "$LFS"
  --miss-rate "$MISS_RATE"
  --assoc "$ASSOC"
  --value-bytes "$VALUE_BYTES"
  "${EXTRA_ARGS[@]}"
)

mkdir -p "$RESULTS"/{logs,raw}

{
  echo "HKV review-response TBE-like baseline"
  echo "  HKV_ROOT    = $HKV_ROOT"
  echo "  SM          = $SM"
  echo "  RESULTS     = $RESULTS"
  echo "  MODE        = $MODE"
  echo "  POLICY      = $POLICY"
  echo "  LFS         = $LFS"
  echo "  MISS_RATE   = $MISS_RATE"
  echo "  ASSOC       = $ASSOC"
  echo "  VALUE_BYTES = $VALUE_BYTES"
  echo "  EXTRA_ARGS  = ${EXTRA_ARGS[*]:-}"
  echo "  Started     = $(date)"
} | tee "$RESULTS/run-info.txt"

if [ "$DRY_RUN" -eq 1 ]; then
  {
    echo "Dry run: commands not executed."
    echo "cd $HKV_ROOT"
    echo "cmake -Dsm=\"$SM\" -DCMAKE_BUILD_TYPE=Release -S . -B \"$BUILD_DIR\""
    echo "cmake --build \"$BUILD_DIR\" --target tbe_like_cache_baseline -j\"$JOBS\""
    printf '%q ' "${RUN_CMD[@]}"
    echo "> \"$RESULTS/raw/tbe_like_cache_baseline.csv\" 2> \"$RESULTS/logs/tbe_like_cache_baseline.log\""
  } | tee -a "$RESULTS/run-info.txt"
  exit 0
fi

cd "$HKV_ROOT"
if [ ! -f tests/googletest/CMakeLists.txt ]; then
  rm -rf tests/googletest
  git clone --depth 1 https://github.com/google/googletest.git tests/googletest
fi

cmake -Dsm="$SM" -DCMAKE_BUILD_TYPE=Release -S . -B "$BUILD_DIR" \
  > "$RESULTS/logs/cmake-tbe-like.log" 2>&1
cmake --build "$BUILD_DIR" --target tbe_like_cache_baseline -j"$JOBS" \
  > "$RESULTS/logs/build-tbe-like.log" 2>&1

"${RUN_CMD[@]}" \
  > "$RESULTS/raw/tbe_like_cache_baseline.csv" \
  2> "$RESULTS/logs/tbe_like_cache_baseline.log"

{
  echo "Finished: $(date)"
  echo "Artifacts:"
  echo "  $RESULTS/raw/tbe_like_cache_baseline.csv"
  echo "  $RESULTS/logs/tbe_like_cache_baseline.log"
} | tee -a "$RESULTS/run-info.txt"
