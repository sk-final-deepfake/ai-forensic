#!/usr/bin/env bash
# Short VideoMAE fine-tune -> 100-video benchmark -> bundle -> S3 upload.
#
# Usage (on GPU host, forenShield-ai root):
#   unset AWS_PROFILE
#   source .venv/bin/activate   # needs: torch, cv2, transformers, numpy
#   sed -i 's/\r$//' scripts/infer/run_videomae_benchmark_pipeline.sh
#   bash scripts/infer/run_videomae_benchmark_pipeline.sh#
# Optional env:
#   SKIP_FINETUNE=1
#   SKIP_INFER=1
#   SKIP_UPLOAD=1
#   UPLOAD_VIDEOS=0
#   RUN_ID=videomae-benchmark-YYYYMMDD-HHMM
set -eu

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "${ROOT}"

WEIGHTS="${WEIGHTS:-models/test/video/videomae/v1.0.0/videomae_finetuned.pth}"
RUN_ID="${RUN_ID:-videomae-benchmark-$(date -u +%Y%m%d-%H%M)}"
S3_PREFIX="${S3_REPORT_PREFIX:-cases/test/video-videomae-benchmark/reports}"
UPLOAD_VIDEOS="${UPLOAD_VIDEOS:-0}"
TRAIN_REAL="${TRAIN_REAL:-data/train/video/voxceleb/real}"
TRAIN_FAKE_POOL="${TRAIN_FAKE_POOL:-data/raw/faceforensics/manipulated_sequences/DeepFakeDetection/c40/videos}"
TRAIN_COUNT="${TRAIN_COUNT:-100}"
BENCHMARK_REAL="${BENCHMARK_REAL:-data/test/video/voxceleb/real}"
BENCHMARK_FAKE="${BENCHMARK_FAKE:-data/test/video/ffpp/fake_over60s}"

if [[ "${SKIP_TRAIN_DATA:-0}" != "1" && "${SKIP_FINETUNE:-0}" != "1" ]]; then
  if [[ ! -d "${ROOT}/${TRAIN_REAL}" ]] || [[ "$(find "${ROOT}/${TRAIN_REAL}" -maxdepth 1 -name '*.mp4' | wc -l)" -lt "${TRAIN_COUNT}" ]]; then
    echo "=== Prepare train real (${TRAIN_COUNT} VoxCeleb, exclude benchmark) ==="
    sed -i 's/\r$//' scripts/download/data/prepare_videomae_train_data.sh 2>/dev/null || true
    SKIP_FINETUNE=1 bash scripts/download/data/prepare_videomae_train_data.sh
  fi
fi

if [[ "${SKIP_FINETUNE:-0}" != "1" ]]; then
  echo "=== VideoMAE short fine-tune (${TRAIN_COUNT}+${TRAIN_COUNT}) ==="
  FINETUNE_ARGS=(
    --root "${ROOT}"
    --epochs "${FINETUNE_EPOCHS:-3}"
    --max-per-class "${TRAIN_COUNT}"
    --batch-size "${FINETUNE_BATCH_SIZE:-2}"
    --output "${WEIGHTS}"
    --train-real-dir "${TRAIN_REAL}"
    --train-fake-dir "${TRAIN_FAKE_POOL}"
    --exclude-dirs "${BENCHMARK_FAKE}" "${BENCHMARK_REAL}"
  )
  python3 scripts/infer/video_videomae_finetune.py "${FINETUNE_ARGS[@]}"
fi

if [[ "${SKIP_INFER:-0}" != "1" ]]; then
  echo "=== VideoMAE 100-video benchmark infer ==="
  python3 scripts/infer/video_videomae_benchmark_infer.py \
    --root "${ROOT}" \
    --weights "${WEIGHTS}" \
    --run-id "${RUN_ID}"
fi

echo "=== Bundle report ==="
python3 scripts/infer/bundle_xception_benchmark_report.py "${RUN_ID}" --root "${ROOT}"

if [[ "${SKIP_UPLOAD:-0}" != "1" ]]; then
  echo "=== S3 upload ==="
  S3_REPORT_PREFIX="${S3_PREFIX}" UPLOAD_VIDEOS="${UPLOAD_VIDEOS}" \
    bash scripts/upload/s3_upload_video_infer_results.sh "${RUN_ID}"
fi

echo "done run_id=${RUN_ID}"
