#!/usr/bin/env bash
# Xception staged fine-tune: local RAW (golden excluded) -> 1차 -> 2차.
#
# Usage (GPU):
#   cd ~/forenShield-ai && source .venv/bin/activate && unset AWS_PROFILE
#   bash scripts/infer/run_xception_finetune_staged.sh
#
# Optional env:
#   SKIP_PREPARE=1          # use existing data/pull/train/video/xception/*
#   UPLOAD_S3=1             # upload prepare output to S3 (default 0)
#   SKIP_S3_PULL=1          # default 1; set 0 to pull from S3 instead of local prepare
#   SKIP_STAGE1=1 | SKIP_STAGE2=1
#   FINETUNE_FRESH=1
#   TRAIN_FAKE_POOL=...   # stage1 FF++ fake pool override
#   TRAIN_REAL_DIR=...    # stage1 Vox real override
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "${ROOT}"

SKIP_PREPARE="${SKIP_PREPARE:-0}"
UPLOAD_S3="${UPLOAD_S3:-0}"
SKIP_S3_PULL="${SKIP_S3_PULL:-1}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"
INIT_WEIGHTS="${INIT_WEIGHTS:-models/test/video/xception/v1.0.0/xception_best.pth}"
STAGE1_OUT="${STAGE1_OUT:-models/test/video/xception/v1.0.0/xception_finetuned_stage1.pth}"
STAGE2_OUT="${STAGE2_OUT:-models/test/video/xception/v1.0.0/xception_finetuned_stage2.pth}"

PREPARE_EXTRA=()
if [[ "${UPLOAD_S3}" == "1" ]]; then
  PREPARE_EXTRA+=(--upload-s3)
fi
if [[ -n "${TRAIN_FAKE_POOL:-}" ]]; then
  PREPARE_EXTRA+=(--fake-dir "${TRAIN_FAKE_POOL}")
fi
if [[ -n "${TRAIN_REAL_DIR:-}" ]]; then
  PREPARE_EXTRA+=(--real-dir "${TRAIN_REAL_DIR}")
fi

FT_EXTRA=()
if [[ "${FINETUNE_FRESH:-0}" == "1" ]]; then
  FT_EXTRA+=(--fresh)
fi

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  echo "=== Prepare train clips from local RAW (exclude golden 200) ==="
  python3 scripts/download/data/prepare_xception_finetune_train.py \
    --root "${ROOT}" --stage stage1 --write-docs "${PREPARE_EXTRA[@]}"
  python3 scripts/download/data/prepare_xception_finetune_train.py \
    --root "${ROOT}" --stage stage2 --write-docs "${PREPARE_EXTRA[@]}"
fi

if [[ "${SKIP_S3_PULL}" != "1" ]]; then
  echo "=== S3 pull train clips ==="
  python3 scripts/download/data/s3_pull_xception_finetune_train.py --root "${ROOT}" --stage all
fi

if [[ "${SKIP_STAGE1}" != "1" ]]; then
  echo ""
  echo "=== 1차 fine-tune (Core FF++ + Vox, backbone freeze) ==="
  python3 scripts/infer/video_xception_finetune.py \
    --root "${ROOT}" \
    --stage stage1 \
    --weights "${INIT_WEIGHTS}" \
    --output "${STAGE1_OUT}" \
    --train-pull-dir data/pull/train/video/xception/stage1 \
    --write-train-manifest-docs \
    "${FT_EXTRA[@]}"
fi

if [[ "${SKIP_STAGE2}" != "1" ]]; then
  echo ""
  echo "=== 2차 fine-tune (Celeb proxy, partial unfreeze) ==="
  python3 scripts/infer/video_xception_finetune.py \
    --root "${ROOT}" \
    --stage stage2 \
    --weights "${STAGE1_OUT}" \
    --output "${STAGE2_OUT}" \
    --train-pull-dir data/pull/train/video/xception/stage2 \
    --unfreeze-backbone \
    --write-train-manifest-docs \
    --fresh
fi

echo ""
echo "Done."
echo "  1차 weights: ${STAGE1_OUT}"
echo "  2차 weights: ${STAGE2_OUT}"
echo "  train manifests: docs/xception_finetune_train_stage1.json"
echo "                   docs/xception_finetune_train_stage2.json"
