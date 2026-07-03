#!/usr/bin/env bash
# Record R5 mvtb dev adoption (R5+A hybrid + Phase 0 calibration) on GPU.
#
# Usage (from ~/forenShield-ai/forgery):
#   sed -i 's/\r$//' scripts/train/record_r5_mvtb_dev_adoption.sh
#   bash scripts/train/record_r5_mvtb_dev_adoption.sh
#
# Restores mvtb metrics @0.185 from predictions, copies calibration manifest.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RUN_NAME="forgery-r5-20260702-0722"
CKPT_DIR="models/dev/spatial/trufor/v1.0.0/${RUN_NAME}"
CKPT="${CKPT_DIR}/trufor.pth.tar"
CALIB_SRC="config/forgery/trufor_r5_mvtb_dev_calibration.json"
CALIB_DST="${CKPT_DIR}/calibration.json"
MVTB_PRED="results/infer/trufor-mvtb200-${RUN_NAME}-20260702-0722/predictions.json"
MVTB_METRICS="results/infer/trufor-mvtb200-${RUN_NAME}-20260702-0722/metrics.json"
MVTB_THR="0.185"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: missing checkpoint: $CKPT"
  exit 1
fi

if [[ ! -f "$CALIB_SRC" ]]; then
  echo "ERROR: missing $CALIB_SRC (deploy config from repo first)"
  exit 1
fi

if [[ -f "$MVTB_PRED" ]]; then
  python3 scripts/train/spatial_benchmark_calibrate_from_predictions.py \
    --predictions "$MVTB_PRED" \
    --threshold "$MVTB_THR" \
    --weights "$CKPT" \
    --note "R5 mvtb dev adopted"
else
  echo "WARN: $MVTB_PRED not found — skip metrics restore"
fi

mkdir -p "$CKPT_DIR"
cp "$CALIB_SRC" "$CALIB_DST"
echo "wrote $CALIB_DST"

if [[ -f "$MVTB_METRICS" ]]; then
  cp "$MVTB_METRICS" "${CKPT_DIR}/metrics_mvtb_calibrated.json"
  echo "copied $MVTB_METRICS -> ${CKPT_DIR}/metrics_mvtb_calibrated.json"
fi

echo ""
echo "=== R5 mvtb dev adopted (supersedes R3) ==="
echo "  ckpt:        $CKPT"
echo "  calibration: $CALIB_DST"
echo "  mvtb thr:    $MVTB_THR"
echo "  csvted:      not adopted (fusion/temporal separate)"
echo "  models/test · deploy: not promoted"
ls -la "$CKPT_DIR"
