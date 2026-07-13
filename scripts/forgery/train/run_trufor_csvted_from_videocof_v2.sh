#!/usr/bin/env bash
# Branch C: continue-FT from VideoCoF-v2 → csvted temporal
#
# Init:  videocof-v2 best (adopted)
# Train: forgery-csvted-train-temporal
# Eval:  csvted-200-balanced + separate thr sweep (no mvtb/VideoCoF thr share)
#
# Caveat: TruFor is spatial — C is an upper-bound experiment; fusion/GMFlow may still win.
#
# GPU:
#   sed -i 's/\r$//' scripts/train/run_trufor_csvted_from_videocof_v2.sh
#   bash scripts/train/run_trufor_csvted_from_videocof_v2.sh
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai/forgery}"
cd "$ROOT"
source "${HOME}/forenShield-ai/.venv/bin/activate"
export FORENSHIELD_AI_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="${DATA_ROOT:-data/train/video/forgery-csvted-train-temporal}"
CACHE_ROOT="${CACHE_ROOT:-data/processed/trufor-csvted-from-videocof-v2}"
EXP_NAME="${EXP_NAME:-csvted-from-videocof-v2-$(date +%Y%m%d-%H%M)}"
GPU="${GPU:-0}"
CONFIG_EXP="${CONFIG_EXP:-trufor_forgery_video_r5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
WORKERS="${WORKERS:-2}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
RESUME="${RESUME:-0}"
INFER_THRESHOLD="${INFER_THRESHOLD:-0.5}"
END_EPOCH="${END_EPOCH:-8}"
FRAMES="${FRAMES:-16}"
AGGREGATE="${AGGREGATE:-top3_mean}"

PRETRAINED="${PRETRAINED:-models/dev/spatial/trufor/v1.0.0/videocof-v2-20260710-0800/trufor.pth.tar}"
if [[ ! -f "$PRETRAINED" ]]; then
  PRETRAINED="models/train/spatial/trufor/videocof-v2/trufor-videocof-v2-20260710-0800/trufor_videocof_v2_ft/best.pth.tar"
fi
CKPT_DEV="models/dev/spatial/trufor/v1.0.0/${EXP_NAME}/trufor.pth.tar"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d-%H%M)}"
CSVTED_RUN_ID="trufor-csvted-${EXP_NAME}-${RUN_DATE}"
CSVTED_PRED="results/infer/${CSVTED_RUN_ID}/predictions.json"

echo "=== Branch C: csvted from VideoCoF-v2 ==="
echo "DATA_ROOT=$DATA_ROOT"
echo "PRETRAINED=$PRETRAINED"
echo "EXP_NAME=$EXP_NAME"

[[ -f "$PRETRAINED" ]] || { echo "ERROR: missing VideoCoF init: $PRETRAINED" >&2; exit 1; }
[[ -d "$DATA_ROOT" ]] || { echo "ERROR: missing $DATA_ROOT" >&2; exit 1; }

if [[ "$SKIP_TRAIN" != "1" ]]; then
  if [[ "$SKIP_PREPARE" != "1" ]]; then
    echo "[1/6] prepare (include temporal fakes; weak full-frame masks)"
    python3 scripts/train/prepare_trufor_video_frames.py \
      --data-root "$DATA_ROOT" \
      --out-dir "$CACHE_ROOT" \
      --frames-per-video "$FRAMES" \
      --valid-ratio 0.1 \
      --seed 42 \
      --include-temporal-fakes \
      --no-require-middle-window \
      --spatial-types masking substitution rotate \
      --recipe-tag csvted-from-vc2
  else
    echo "[1/6] prepare skipped ($CACHE_ROOT)"
  fi

  echo "[2/6] vendor patches"
  PATCH_DST="vendor/TruFor/TruFor_train_test/dataset/dataset_ForenShieldVideo.py"
  [[ -f "$PATCH_DST" ]] || cp scripts/train/vendor_patches/dataset_ForenShieldVideo.py "$PATCH_DST"
  cp scripts/train/vendor_patches/trufor_forgery_video_r5.yaml \
    "vendor/TruFor/TruFor_train_test/lib/config/${CONFIG_EXP}.yaml"

  echo "[3/6] train (init=VideoCoF-v2, epoch=${END_EPOCH}, resume=${RESUME})"
  TRAIN_ARGS=(
    -exp "$CONFIG_EXP"
    --run-name "$EXP_NAME"
    -g "$GPU"
    --cache-root "$CACHE_ROOT"
    --pretrained-checkpoint "$PRETRAINED"
  )
  if [[ "$RESUME" == "1" ]]; then
    TRAIN_ARGS+=(--resume)
  fi
  python3 scripts/train/train_trufor_video_forgery.py \
    "${TRAIN_ARGS[@]}" \
    TRAIN.BATCH_SIZE_PER_GPU "$BATCH_SIZE" \
    TRAIN.END_EPOCH "$END_EPOCH" \
    WORKERS "$WORKERS"

  echo "[4/6] merge → $CKPT_DEV"
  mkdir -p "$(dirname "$CKPT_DEV")"
  python3 scripts/train/merge_trufor_infer_checkpoint.py \
    --base "$PRETRAINED" \
    --tuned "vendor/TruFor/TruFor_train_test/weights/${EXP_NAME}/best.pth.tar" \
    --out "$CKPT_DEV"
else
  echo "[1-4/6] train skipped"
  [[ -f "$CKPT_DEV" ]] || { echo "ERROR: missing $CKPT_DEV"; exit 1; }
fi

echo "[5/6] infer csvted @${INFER_THRESHOLD}"
python3 scripts/infer/spatial_mvtamperbench_benchmark.py \
  --root "$ROOT" \
  --data-root data/pull/evidence/csvted-200-balanced \
  --model trufor \
  --num-frames "$FRAMES" \
  --aggregate "$AGGREGATE" \
  --threshold "$INFER_THRESHOLD" \
  --trufor-weights "$CKPT_DEV" \
  --run-id "$CSVTED_RUN_ID"

echo "[6/6] thr sweep (csvted-only; do not share with mvtb/VideoCoF)"
python3 scripts/infer/sweep_spatial_benchmark_threshold.py \
  --predictions "$CSVTED_PRED" \
  --step 0.01 || true

echo "DONE Branch C"
echo "  ckpt: $CKPT_DEV"
echo "  pred: $CSVTED_PRED"
echo "  Pick thr from sweep; VideoCoF 0.5 / mvtb thr 공유 금지."
