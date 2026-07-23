#!/usr/bin/env bash
# TimeSformer forgery window-MIL benchmark — single dataset shortcut.
#
# Usage:
#   bash forgery/scripts/infer/run_timesformer_forgery_benchmark.sh temporal200
#   bash forgery/scripts/infer/run_timesformer_forgery_benchmark.sh mvtb200-temporal
#   bash forgery/scripts/infer/run_timesformer_forgery_benchmark.sh csvted200-temporal
#   bash forgery/scripts/infer/run_timesformer_forgery_benchmark.sh mvtb200      # mixed (spatial included)
#   bash forgery/scripts/infer/run_timesformer_forgery_benchmark.sh csvted200    # mixed (spatial included)
#
# Env:
#   CKPT=.../forgery_head.pt
#   AGGREGATE=max|topk   (prod csvted KPI: max)
#   THRESHOLD=0.12       (prod Youden; AUC는 threshold 무관)
set -eu

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  echo "usage: $0 {csvted200-temporal|mvtb200-temporal|mvtb200|csvted200|temporal200}" >&2
  echo "  prod KPI: csvted200-temporal (csvted-200-temporal-only, aggregate=max)" >&2
  exit 1
fi

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai/forgery}"
REPO="${FORENSHIELD_REPO:-$HOME/forenShield-ai}"
TS="$(date -u +%Y%m%d-%H%M)"
PRETRAINED="${PRETRAINED:-facebook/timesformer-base-finetuned-k400}"
GPU="${GPU:-0}"
DECODE_CACHE="${DECODE_CACHE:-$ROOT/cache/decode-mp4-temporal}"
AGGREGATE="${AGGREGATE:-max}"
THRESHOLD="${THRESHOLD:-0.12}"

CKPT="${CKPT:-$ROOT/models/train/temporal/timesformer-forgery/timesformer-forgery-v1.8-csvted-v4-20260708-0022/forgery_head.pt}"
BENCH_PY="$ROOT/scripts/infer/timesformer_forgery_benchmark.py"

case "$TARGET" in
  temporal200)
    # legacy alias → prod CSVTED temporal KPI (GMFlow lane removed)
    echo "NOTE: temporal200 → csvted-200-temporal-only (prod v1.8-v4 lane)" >&2
    DATA="$ROOT/data/pull/evidence/csvted-200-temporal-only"
    RUN_ID="timesformer-forgery-temporal200-${TS}"
    ;;
  mvtb200-temporal)
    DATA="$ROOT/data/pull/evidence/mvtamperbench-200-temporal-only"
    RUN_ID="timesformer-forgery-mvtb200-temporal-${TS}"
    ;;
  csvted200-temporal)
    DATA="$ROOT/data/pull/evidence/csvted-200-temporal-only"
    RUN_ID="timesformer-forgery-csvted200-temporal-${TS}"
    ;;
  mvtb200)
    DATA="$ROOT/data/pull/evidence/mvtamperbench-200-s3"
    RUN_ID="timesformer-forgery-mvtb200-mixed-${TS}"
    ;;
  csvted200)
    DATA="$ROOT/data/pull/evidence/csvted-200-balanced"
    RUN_ID="timesformer-forgery-csvted200-mixed-${TS}"
    ;;
  *)
    echo "unknown target: $TARGET" >&2
    exit 1
    ;;
esac

cd "$REPO"
source "$REPO/.venv/bin/activate"
export FORENSHIELD_AI_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  echo "Set CKPT=... or train first with run_timesformer_forgery_v1_temporal.sh" >&2
  exit 1
fi

python3 "$BENCH_PY" \
  --root "$REPO" \
  --data-root "$DATA" \
  --checkpoint "$CKPT" \
  --pretrained-id "$PRETRAINED" \
  --run-id "$RUN_ID" \
  --aggregate "$AGGREGATE" \
  --threshold "$THRESHOLD" \
  --decode-cache "$DECODE_CACHE" \
  --gpu "$GPU"

cat "$ROOT/results/eval/${RUN_ID}/metrics.json"
