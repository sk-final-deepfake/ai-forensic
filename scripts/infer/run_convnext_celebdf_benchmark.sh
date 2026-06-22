#!/usr/bin/env bash
# Celeb-DF v2 50+50: ConvNeXt fine-tune + infer + bundle + S3.
#
# Note: DeepfakeBench v1.0.1 has no convnext_best.pth — we fine-tune ImageNet ConvNeXt
# on FF++ fake + Vox/FF++ real (same train pool as VideoMAE/TimeSformer).
#
# Usage:
#   cd ~/forenShield-ai && source .venv/bin/activate && unset AWS_PROFILE
#   sed -i 's/\r$//' scripts/infer/run_convnext_celebdf_benchmark.sh
#   bash scripts/infer/run_convnext_celebdf_benchmark.sh
#
# Optional env:
#   VARIANT=small|base
#   RUN_ID=convnext-celebdf-benchmark-20260619-1200
#   SKIP_FINETUNE=1
#   SKIP_INFER=1
#   THRESHOLD=0.5
#   UPLOAD_VIDEOS=1
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
unset AWS_PROFILE

VARIANT="${VARIANT:-small}"
DATASET_DIR="${DATASET_DIR:-data/test/video/celeb-df-v2}"
THRESHOLD="${THRESHOLD:-0.5}"
UPLOAD_VIDEOS="${UPLOAD_VIDEOS:-1}"
SKIP_FINETUNE="${SKIP_FINETUNE:-0}"
SKIP_INFER="${SKIP_INFER:-0}"
WEIGHTS="${WEIGHTS:-models/test/video/convnext/v1.0.0/convnext_finetuned.pth}"
S3_REPORT_PREFIX="${S3_REPORT_PREFIX:-cases/test/video-convnext-celebdf-benchmark/reports}"
RUN_ID="${RUN_ID:-convnext-celebdf-benchmark-$(date -u +%Y%m%d-%H%M)}"

FAKE_DIR="$ROOT/$DATASET_DIR/fake"
REAL_DIR="$ROOT/$DATASET_DIR/real"

for dir in "$FAKE_DIR" "$REAL_DIR"; do
  count="$(find "$dir" -maxdepth 1 -name '*.mp4' | wc -l)"
  if [[ "$count" -lt 1 ]]; then
    echo "ERROR: no mp4 under $dir (run download_celebdf_v2.sh first)"
    exit 1
  fi
  echo "found $count videos in $dir"
done

if [[ "$SKIP_FINETUNE" != "1" ]]; then
  echo "==> fine-tune ConvNeXt variant=$VARIANT"
  python3 "$ROOT/scripts/infer/video_convnext_finetune.py" \
    --root "$ROOT" \
    --variant "$VARIANT" \
    --output "$WEIGHTS" \
    --exclude-dirs "$DATASET_DIR/fake" "$DATASET_DIR/real" \
    --max-per-class "${MAX_PER_CLASS:-100}" \
    --epochs "${FINETUNE_EPOCHS:-5}" \
    --batch-size "${FINETUNE_BATCH_SIZE:-16}" \
    ${UNFREEZE_BACKBONE:+--unfreeze-backbone}
fi

if [[ "$SKIP_INFER" != "1" ]]; then
  echo "==> infer variant=$VARIANT run_id=$RUN_ID threshold=$THRESHOLD"
  python3 "$ROOT/scripts/infer/video_convnext_benchmark_infer.py" \
    --root "$ROOT" \
    --variant "$VARIANT" \
    --run-id "$RUN_ID" \
    --weights "$WEIGHTS" \
    --fake-dir "$FAKE_DIR" \
    --real-dir "$REAL_DIR" \
    --threshold "$THRESHOLD"
fi

echo "==> bundle report (Celeb-DF profile)"
python3 "$ROOT/scripts/infer/bundle_xception_benchmark_report.py" \
  "$RUN_ID" \
  --root "$ROOT" \
  --profile celebdf

echo "==> upload to S3"
S3_REPORT_PREFIX="$S3_REPORT_PREFIX" \
UPLOAD_VIDEOS="$UPLOAD_VIDEOS" \
bash "$ROOT/scripts/upload/s3_upload_video_infer_results.sh" "$RUN_ID"

echo ""
echo "DONE RUN_ID=$RUN_ID"
echo "S3: s3://forenshield-evidence-877044078824/${S3_REPORT_PREFIX}/${RUN_ID}/"
echo "metrics: cat $ROOT/results/eval/$RUN_ID/metrics.json"
