#!/usr/bin/env bash
# ForenShield TruFor recipe v2 — Phase 1 smoke (prepare v2 + 3 epoch)
# Usage (from ~/forenShield-ai/forgery):
#   sed -i 's/\r$//' scripts/train/run_trufor_forgery_train_v2.sh
#   bash scripts/train/run_trufor_forgery_train_v2.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

source "${HOME}/forenShield-ai/.venv/bin/activate"

DATA_ROOT="${ROOT}/data/train/video/forgery-gmflow-train-400"
CACHE_ROOT="${ROOT}/data/processed/trufor-gmflow-train-400-v2"
EXP_NAME="${EXP_NAME:-forgery-v2-smoke-$(date +%Y%m%d-%H%M)}"
GPU="${GPU:-0}"
CONFIG_EXP="${CONFIG_EXP:-trufor_forgery_video_v2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
WORKERS="${WORKERS:-2}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

PRETRAINED="${PRETRAINED:-${ROOT}/models/test/spatial/trufor/v1.0.0/trufor.pth.tar}"

if [[ "$SKIP_PREPARE" != "1" ]]; then
  echo "[1/4] prepare frames (recipe v2)"
  python3 scripts/train/prepare_trufor_video_frames.py \
    --data-root "$DATA_ROOT" \
    --out-dir "$CACHE_ROOT" \
    --frames-per-video 8 \
    --valid-ratio 0.1 \
    --seed 42 \
    --require-middle-window
else
  echo "[1/4] prepare skipped (SKIP_PREPARE=1, cache: $CACHE_ROOT)"
fi

echo "[2/4] ensure vendor dataset + config"
PATCH_SRC="scripts/train/vendor_patches/dataset_ForenShieldVideo.py"
PATCH_DST="vendor/TruFor/TruFor_train_test/dataset/dataset_ForenShieldVideo.py"
if [[ ! -f "$PATCH_DST" ]]; then
  cp "$PATCH_SRC" "$PATCH_DST"
  echo "copied $PATCH_DST"
fi

CFG_SRC="scripts/train/vendor_patches/trufor_forgery_video_v2.yaml"
CFG_DST="vendor/TruFor/TruFor_train_test/lib/config/trufor_forgery_video_v2.yaml"
cp "$CFG_SRC" "$CFG_DST"
echo "copied $CFG_DST"

echo "[3/4] train (smoke 3 epoch, batch=$BATCH_SIZE, workers=$WORKERS, gpu=$GPU)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | sed "s/^/  GPU /" || true
fi
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
  TRAIN.END_EPOCH 3 \
  TRAIN.BATCH_SIZE_PER_GPU "$BATCH_SIZE" \
  WORKERS "$WORKERS"

echo "[4/4] merge hint"
echo "  python3 scripts/train/merge_trufor_infer_checkpoint.py \\"
echo "    --base $PRETRAINED \\"
echo "    --tuned vendor/TruFor/TruFor_train_test/weights/$EXP_NAME/best.pth.tar \\"
echo "    --out models/dev/spatial/trufor/v1.0.0/$EXP_NAME/trufor.pth.tar"
echo "done: $EXP_NAME"
echo "weights: vendor/TruFor/TruFor_train_test/weights/$EXP_NAME"
