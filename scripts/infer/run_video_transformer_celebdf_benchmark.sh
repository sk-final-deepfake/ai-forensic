#!/usr/bin/env bash
# Celeb-DF v2 50+50: fine-tune + infer + bundle + S3 for TimeSformer or Video Swin.
#
# Usage:
#   cd ~/forenShield-ai && source .venv/bin/activate && unset AWS_PROFILE
#   sed -i 's/\r$//' scripts/infer/run_video_transformer_celebdf_benchmark.sh
#   MODEL=timesformer bash scripts/infer/run_video_transformer_celebdf_benchmark.sh
#   MODEL=video-swin bash scripts/infer/run_video_transformer_celebdf_benchmark.sh
#
# Optional env:
#   MODEL=timesformer|video-swin   (required)
#   RUN_ID=timesformer-celebdf-benchmark-20260619-1200
#   SKIP_FINETUNE=1
#   SKIP_INFER=1
#   THRESHOLD=0.5
#   UPLOAD_VIDEOS=1
set -euo pipefail

MODEL="${MODEL:?set MODEL=timesformer or MODEL=video-swin}"
ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
unset AWS_PROFILE

DATASET_DIR="${DATASET_DIR:-data/test/video/celeb-df-v2}"
THRESHOLD="${THRESHOLD:-0.5}"
MAX_CLIPS="${MAX_CLIPS:-4}"
UPLOAD_VIDEOS="${UPLOAD_VIDEOS:-1}"
SKIP_FINETUNE="${SKIP_FINETUNE:-0}"
SKIP_INFER="${SKIP_INFER:-0}"

case "$MODEL" in
  timesformer)
    WEIGHTS="${WEIGHTS:-models/test/video/timesformer/v1.0.0/timesformer_finetuned.pth}"
    S3_REPORT_PREFIX="${S3_REPORT_PREFIX:-cases/test/video-timesformer-celebdf-benchmark/reports}"
    RUN_ID="${RUN_ID:-timesformer-celebdf-benchmark-$(date -u +%Y%m%d-%H%M)}"
    ;;
  video-swin)
    WEIGHTS="${WEIGHTS:-models/test/video/video-swin/v1.0.0/video_swin_finetuned.pth}"
    S3_REPORT_PREFIX="${S3_REPORT_PREFIX:-cases/test/video-swin-celebdf-benchmark/reports}"
    RUN_ID="${RUN_ID:-video-swin-celebdf-benchmark-$(date -u +%Y%m%d-%H%M)}"
    ;;
  *)
    echo "ERROR: MODEL must be timesformer or video-swin"
    exit 1
    ;;
esac

FAKE_DIR="$ROOT/$DATASET_DIR/fake"
REAL_DIR="$ROOT/$DATASET_DIR/real"

for dir in "$FAKE_DIR" "$REAL_DIR"; do
  count="$(find "$dir" -maxdepth 1 -name '*.mp4' | wc -l)"
  if [[ "$count" -lt 1 ]]; then
    echo "ERROR: no mp4 under $dir"
    exit 1
  fi
  echo "found $count videos in $dir"
done

if [[ "$SKIP_FINETUNE" != "1" ]]; then
  echo "==> fine-tune model=$MODEL"
  python3 "$ROOT/scripts/infer/video_transformer_finetune.py" \
    --root "$ROOT" \
    --model "$MODEL" \
    --output "$WEIGHTS" \
    --exclude-dirs "$DATASET_DIR/fake" "$DATASET_DIR/real" \
    --max-per-class "${MAX_PER_CLASS:-100}" \
    --epochs "${FINETUNE_EPOCHS:-3}" \
    --batch-size "${FINETUNE_BATCH_SIZE:-1}"
fi

if [[ "$SKIP_INFER" != "1" ]]; then
  echo "==> infer model=$MODEL run_id=$RUN_ID threshold=$THRESHOLD"
  python3 "$ROOT/scripts/infer/video_transformer_benchmark_infer.py" \
    --root "$ROOT" \
    --model "$MODEL" \
    --run-id "$RUN_ID" \
    --weights "$WEIGHTS" \
    --fake-dir "$FAKE_DIR" \
    --real-dir "$REAL_DIR" \
    --threshold "$THRESHOLD" \
    --max-clips "$MAX_CLIPS"
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
echo "DONE MODEL=$MODEL RUN_ID=$RUN_ID"
echo "S3: s3://forenshield-evidence-877044078824/${S3_REPORT_PREFIX}/${RUN_ID}/"
echo "metrics: cat $ROOT/results/eval/$RUN_ID/metrics.json"
