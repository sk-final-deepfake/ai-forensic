#!/usr/bin/env bash
# mvtb 500 hold-out eval — NO retrain, NO re-tune on 500 (locked thr from mvtb200).
#
# Prereq: build hold-out folder first (prepare_mvtb_holdout_benchmark.py).
#
# Usage:
#   sed -i 's/\r$//' scripts/train/run_mvtb500_holdout_eval.sh
#   bash scripts/train/run_mvtb500_holdout_eval.sh
#
# Env overrides:
#   CKPT=...  DATA_ROOT=...  LOCKED_THR=0.185  RUN_ID=...

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source ../.venv/bin/activate 2>/dev/null || true

CKPT="${CKPT:-models/dev/spatial/trufor/v1.0.0/forgery-r5-20260702-0722/trufor.pth.tar}"
DATA_ROOT="${DATA_ROOT:-data/pull/evidence/mvtamperbench-500-holdout}"
LOCKED_THR="${LOCKED_THR:-0.185}"
RUN_DATE="$(date +%Y%m%d-%H%M)"
RUN_ID="${RUN_ID:-trufor-mvtb500-holdout-r5-${RUN_DATE}}"
MANIFEST="${DATA_ROOT}/manifest.json"
CALIB_PRED="${CALIB_PRED:-results/infer/trufor-mvtb200-forgery-r5-20260702-0722-20260702-0722/predictions.json}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: missing ckpt: $CKPT"
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: missing hold-out data: $DATA_ROOT"
  echo "  Run prepare_mvtb_holdout_benchmark.py first."
  exit 1
fi

echo "[1/3] infer @0.5 (scores only; thr applied in eval)"
python3 scripts/infer/spatial_mvtamperbench_benchmark.py \
  --root "$ROOT" \
  --data-root "$DATA_ROOT" \
  --model trufor --num-frames 8 --threshold 0.5 \
  --trufor-weights "$CKPT" \
  --run-id "$RUN_ID"

PRED="results/infer/${RUN_ID}/predictions.json"

echo "[2/3] locked thr metrics @${LOCKED_THR} (do NOT gate-sweep on 500 for adoption)"
python3 scripts/train/spatial_benchmark_calibrate_from_predictions.py \
  --predictions "$PRED" \
  --threshold "$LOCKED_THR" \
  --weights "$CKPT" \
  --note "mvtb500 hold-out @ locked thr ${LOCKED_THR} (no re-tune)"

echo "[3/3] subset report (calibration_200 vs ood_new)"
python3 scripts/train/evaluate_mvtb_holdout_predictions.py \
  --predictions "$PRED" \
  --manifest "$MANIFEST" \
  --locked-thr "$LOCKED_THR" \
  --out-json "results/infer/${RUN_ID}/holdout_eval.json"

echo ""
echo "=== done ==="
echo "  run_id:     $RUN_ID"
echo "  metrics:    results/infer/${RUN_ID}/metrics.json"
echo "  holdout:    results/infer/${RUN_ID}/holdout_eval.json"
echo "  PRIMARY:    ood_new @ thr=${LOCKED_THR} (must generalize; no re-calibration on 500)"
