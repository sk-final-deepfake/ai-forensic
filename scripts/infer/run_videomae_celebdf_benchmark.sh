#!/usr/bin/env bash
# Celeb-DF v2 100-video benchmark: VideoMAE infer + bundle + S3 (videos + reports).
#
# Prereqs on GPU:
#   data/test/video/celeb-df-v2/real/*.mp4 (50)
#   data/test/video/celeb-df-v2/fake/*.mp4 (50)
#   models/test/video/videomae/v1.0.0/videomae_finetuned_unfrozen.pth
#
# Usage:
#   cd ~/forenShield-ai && source .venv/bin/activate && unset AWS_PROFILE
#   sed -i 's/\r$//' scripts/infer/run_videomae_celebdf_benchmark.sh
#   bash scripts/infer/run_videomae_celebdf_benchmark.sh
#
# Optional env:
#   RUN_ID=videomae-celebdf-benchmark-20260619-1200
#   THRESHOLD=0.5
#   WEIGHTS=models/test/video/videomae/v1.0.0/videomae_finetuned_unfrozen.pth
#   UPLOAD_VIDEOS=1
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
unset AWS_PROFILE

DATASET_DIR="${DATASET_DIR:-data/test/video/celeb-df-v2}"
THRESHOLD="${THRESHOLD:-0.5}"
MAX_CLIPS="${MAX_CLIPS:-4}"
WEIGHTS="${WEIGHTS:-models/test/video/videomae/v1.0.0/videomae_finetuned_unfrozen.pth}"
UPLOAD_VIDEOS="${UPLOAD_VIDEOS:-1}"
S3_REPORT_PREFIX="${S3_REPORT_PREFIX:-cases/test/video-videomae-celebdf-benchmark/reports}"
RUN_ID="${RUN_ID:-videomae-celebdf-benchmark-$(date -u +%Y%m%d-%H%M)}"

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

echo "==> VideoMAE infer run_id=$RUN_ID threshold=$THRESHOLD"
python3 "$ROOT/scripts/infer/video_videomae_benchmark_infer.py" \
  --root "$ROOT" \
  --run-id "$RUN_ID" \
  --weights "$WEIGHTS" \
  --fake-dir "$FAKE_DIR" \
  --real-dir "$REAL_DIR" \
  --threshold "$THRESHOLD" \
  --max-clips "$MAX_CLIPS"

echo "==> bundle report (Celeb-DF profile)"
python3 "$ROOT/scripts/infer/bundle_xception_benchmark_report.py" \
  "$RUN_ID" \
  --root "$ROOT" \
  --profile celebdf

echo "==> upload to S3 (reports + videos)"
S3_REPORT_PREFIX="$S3_REPORT_PREFIX" \
UPLOAD_VIDEOS="$UPLOAD_VIDEOS" \
bash "$ROOT/scripts/upload/s3_upload_video_infer_results.sh" "$RUN_ID"

echo ""
echo "DONE RUN_ID=$RUN_ID"
echo "S3: s3://forenshield-evidence-877044078824/${S3_REPORT_PREFIX}/${RUN_ID}/"
echo "metrics: cat $ROOT/results/eval/$RUN_ID/metrics.json"
