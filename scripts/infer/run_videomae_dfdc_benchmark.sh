#!/usr/bin/env bash
# DFDC 50-video benchmark: VideoMAE infer + bundle + S3 (videos + reports).
#
# Prereqs on GPU:
#   data/test/video/dfdc/manifest.json + *.mp4
#   models/test/video/videomae/v1.0.0/videomae_finetuned_unfrozen.pth (or finetuned.pth)
#
# Usage:
#   cd ~/forenShield-ai && source .venv/bin/activate && unset AWS_PROFILE
#   bash scripts/infer/run_videomae_dfdc_benchmark.sh
#
# Optional env:
#   RUN_ID=videomae-dfdc-benchmark-20260619-1200
#   THRESHOLD=0.39
#   WEIGHTS=models/test/video/videomae/v1.0.0/videomae_finetuned_unfrozen.pth
#   UPLOAD_VIDEOS=1
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
unset AWS_PROFILE

DATASET_DIR="${DATASET_DIR:-data/test/video/dfdc}"
THRESHOLD="${THRESHOLD:-0.39}"
MAX_CLIPS="${MAX_CLIPS:-4}"
WEIGHTS="${WEIGHTS:-models/test/video/videomae/v1.0.0/videomae_finetuned_unfrozen.pth}"
UPLOAD_VIDEOS="${UPLOAD_VIDEOS:-1}"
S3_REPORT_PREFIX="${S3_REPORT_PREFIX:-cases/test/video-videomae-dfdc-benchmark/reports}"
RUN_ID="${RUN_ID:-videomae-dfdc-benchmark-$(date -u +%Y%m%d-%H%M)}"

echo "==> prepare fake/real dirs from manifest"
python3 "$ROOT/scripts/download/data/prepare_dfdc_infer_dirs.py" \
  --root "$ROOT" \
  --dataset-dir "$DATASET_DIR"

FAKE_DIR="$ROOT/$DATASET_DIR/fake"
REAL_DIR="$ROOT/$DATASET_DIR/real"

echo "==> VideoMAE infer run_id=$RUN_ID threshold=$THRESHOLD"
python3 "$ROOT/scripts/infer/video_videomae_benchmark_infer.py" \
  --root "$ROOT" \
  --run-id "$RUN_ID" \
  --weights "$WEIGHTS" \
  --fake-dir "$FAKE_DIR" \
  --real-dir "$REAL_DIR" \
  --threshold "$THRESHOLD" \
  --max-clips "$MAX_CLIPS"

echo "==> bundle report (DFDC profile)"
python3 "$ROOT/scripts/infer/bundle_xception_benchmark_report.py" \
  "$RUN_ID" \
  --root "$ROOT" \
  --profile dfdc

echo "==> upload to S3 (reports + videos)"
S3_REPORT_PREFIX="$S3_REPORT_PREFIX" \
UPLOAD_VIDEOS="$UPLOAD_VIDEOS" \
bash "$ROOT/scripts/upload/s3_upload_video_infer_results.sh" "$RUN_ID"

echo ""
echo "DONE RUN_ID=$RUN_ID"
echo "S3: s3://forenshield-evidence-877044078824/${S3_REPORT_PREFIX}/${RUN_ID}/"
echo "metrics: cat $ROOT/results/eval/$RUN_ID/metrics.json"
