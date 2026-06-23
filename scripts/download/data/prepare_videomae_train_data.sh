#!/usr/bin/env bash
# Prepare VideoMAE train set:
#   - 100 VoxCeleb real clips (excluding benchmark 50 video_ids)
#   - finetune uses 100 FF++ fake from raw pool (excluding benchmark 50 sources)
#
# Usage (GPU, forenShield-ai root):
#   bash scripts/download/data/prepare_videomae_train_data.sh
#
# Optional env:
#   SKIP_VOX_DOWNLOAD=1
#   SKIP_FINETUNE=1
set -eu

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "${ROOT}"

BENCHMARK_REAL="${BENCHMARK_REAL:-data/test/video/voxceleb/real}"
BENCHMARK_FAKE="${BENCHMARK_FAKE:-data/test/video/ffpp/fake_over60s}"
TRAIN_REAL="${TRAIN_REAL:-data/train/video/voxceleb/real}"
TRAIN_FAKE_POOL="${TRAIN_FAKE_POOL:-data/raw/faceforensics/manipulated_sequences/DeepFakeDetection/c40/videos}"
TRAIN_COUNT="${TRAIN_COUNT:-100}"
VOX_SEED="${VOX_SEED:-42}"

if [[ "${SKIP_VOX_DOWNLOAD:-0}" != "1" ]]; then
  echo "=== Download ${TRAIN_COUNT} VoxCeleb real (exclude benchmark) ==="
  python3 scripts/download/data/download_voxceleb_long.py \
    --out-dir "${TRAIN_REAL}" \
    --exclude-dir "${BENCHMARK_REAL}" \
    --target "${TRAIN_COUNT}" \
    --seed "${VOX_SEED}"
fi

if [[ "${SKIP_FINETUNE:-0}" != "1" ]]; then
  echo "=== VideoMAE fine-tune (${TRAIN_COUNT}+${TRAIN_COUNT}) ==="
  python3 scripts/infer/video_videomae_finetune.py \
    --root "${ROOT}" \
    --train-real-dir "${TRAIN_REAL}" \
    --train-fake-dir "${TRAIN_FAKE_POOL}" \
    --exclude-dirs "${BENCHMARK_FAKE}" "${BENCHMARK_REAL}" \
    --max-per-class "${TRAIN_COUNT}" \
    --epochs "${FINETUNE_EPOCHS:-3}" \
    --batch-size "${FINETUNE_BATCH_SIZE:-2}" \
    --output "${WEIGHTS:-models/test/video/videomae/v1.0.0/videomae_finetuned.pth}"
fi

echo "train real: ${TRAIN_REAL}"
echo "train fake pool: ${TRAIN_FAKE_POOL} (sample ${TRAIN_COUNT}, exclude benchmark)"
echo "done."
