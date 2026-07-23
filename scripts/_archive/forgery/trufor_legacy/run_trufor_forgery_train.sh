#!/usr/bin/env bash
# ForenShield TruFor video forgery smoke train
# Usage (from ~/forenShield-ai/forgery):
#   bash scripts/train/run_trufor_forgery_train.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

source "${HOME}/forenShield-ai/.venv/bin/activate"

DATA_ROOT="${ROOT}/data/train/video/forgery-gmflow-train-400"
CACHE_ROOT="${ROOT}/data/processed/trufor-gmflow-train-400"
EXP_NAME="${EXP_NAME:-forgery-$(date +%Y%m%d-%H%M)}"
GPU="${GPU:-0}"
CONFIG_EXP="${CONFIG_EXP:-trufor_forgery_video}"

# Optional: baseline/deploy TruFor weights
PRETRAINED="${PRETRAINED:-${ROOT}/models/test/spatial/trufor/v1.0.0/trufor.pth.tar}"

echo "[1/3] prepare frames"
python3 scripts/train/prepare_trufor_video_frames.py \
  --data-root "$DATA_ROOT" \
  --out-dir "$CACHE_ROOT" \
  --frames-per-video 8 \
  --valid-ratio 0.1 \
  --seed 42

echo "[2/3] ensure vendor dataset file"
PATCH_SRC="scripts/train/vendor_patches/dataset_ForenShieldVideo.py"
PATCH_DST="vendor/TruFor/TruFor_train_test/dataset/dataset_ForenShieldVideo.py"
if [[ ! -f "$PATCH_DST" ]]; then
  cp "$PATCH_SRC" "$PATCH_DST"
  echo "copied $PATCH_DST"
fi

CFG_SRC="scripts/train/vendor_patches/trufor_forgery_video.yaml"
CFG_DST="vendor/TruFor/TruFor_train_test/lib/config/trufor_forgery_video.yaml"
if [[ ! -f "$CFG_DST" ]]; then
  cp "$CFG_SRC" "$CFG_DST"
  echo "copied $CFG_DST"
fi

echo "[3/3] train"
PRETRAIN_ARG=()
if [[ -f "$PRETRAINED" ]]; then
  PRETRAIN_ARG=(--pretrained-checkpoint "$PRETRAINED")
fi

python3 scripts/train/train_trufor_video_forgery.py \
  -exp "$CONFIG_EXP" \
  --run-name "$EXP_NAME" \
  -g "$GPU" \
  --cache-root "$CACHE_ROOT" \
  "${PRETRAIN_ARG[@]}" \
  TRAIN.END_EPOCH 2 \
  TRAIN.BATCH_SIZE_PER_GPU 4

echo "done: $EXP_NAME"
echo "checkpoint dir: vendor/TruFor/TruFor_train_test/log/train/$EXP_NAME"
