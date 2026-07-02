#!/usr/bin/env bash
# Record R3 mvtb dev adoption (Option A: Phase 0 calibration) on GPU.
#
# Usage (from ~/forenShield-ai/forgery):
#   sed -i 's/\r$//' scripts/train/record_r3_mvtb_dev_adoption.sh
#   bash scripts/train/record_r3_mvtb_dev_adoption.sh
#
# Copies calibration manifest next to dev checkpoint and verifies paths.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RUN_NAME="forgery-r3-20260702-0338"
CKPT_DIR="models/dev/spatial/trufor/v1.0.0/${RUN_NAME}"
CKPT="${CKPT_DIR}/trufor.pth.tar"
CALIB_SRC="config/forgery/trufor_r3_mvtb_dev_calibration.json"
CALIB_DST="${CKPT_DIR}/calibration.json"
MVTB_METRICS="results/infer/trufor-mvtb200-${RUN_NAME}-20260702-0353/metrics.json"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: missing checkpoint: $CKPT"
  exit 1
fi

if [[ ! -f "$CALIB_SRC" ]]; then
  echo "ERROR: missing $CALIB_SRC (deploy config from repo first)"
  exit 1
fi

mkdir -p "$CKPT_DIR"
cp "$CALIB_SRC" "$CALIB_DST"
echo "wrote $CALIB_DST"

if [[ -f "$MVTB_METRICS" ]]; then
  cp "$MVTB_METRICS" "${CKPT_DIR}/metrics_mvtb_calibrated.json"
  echo "copied $MVTB_METRICS -> ${CKPT_DIR}/metrics_mvtb_calibrated.json"
else
  echo "WARN: $MVTB_METRICS not found — generate from predictions.json first"
fi

echo ""
echo "=== R3 mvtb dev adopted ==="
echo "  ckpt:       $CKPT"
echo "  calibration: $CALIB_DST"
echo "  mvtb thr:   0.158"
echo "  models/test · deploy: not promoted"
ls -la "$CKPT_DIR"
