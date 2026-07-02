#!/usr/bin/env bash
# Baseline-line R2 — cause #3 only (skip hard-negative out-of-window fake frames)
# Independent from Legacy v2~v5 and from R1 (separate cache / yaml / run_name).
#
# Usage (from ~/forenShield-ai/forgery):
#   sed -i 's/\r$//' scripts/train/run_trufor_forgery_train_r2.sh
#   bash scripts/train/run_trufor_forgery_train_r2.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

source "${HOME}/forenShield-ai/.venv/bin/activate"

DATA_ROOT="${ROOT}/data/train/video/forgery-gmflow-train-400"
CACHE_ROOT="${ROOT}/data/processed/trufor-gmflow-train-400-r2"
EXP_NAME="${EXP_NAME:-forgery-r2-$(date +%Y%m%d-%H%M)}"
GPU="${GPU:-0}"
CONFIG_EXP="${CONFIG_EXP:-trufor_forgery_video_r2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
WORKERS="${WORKERS:-2}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
INFER_THRESHOLD="${INFER_THRESHOLD:-0.5}"

PRETRAINED="${PRETRAINED:-${ROOT}/models/test/spatial/trufor/v1.0.0/trufor.pth.tar}"

if [[ "$SKIP_PREPARE" != "1" ]]; then
  echo "[1/5] prepare frames (R2: middle window, no hard-negative OOW frames)"
  python3 scripts/train/prepare_trufor_video_frames.py \
    --data-root "$DATA_ROOT" \
    --out-dir "$CACHE_ROOT" \
    --frames-per-video 8 \
    --valid-ratio 0.1 \
    --seed 42 \
    --require-middle-window \
    --skip-out-of-window-fake \
    --recipe-tag r2
else
  echo "[1/5] prepare skipped (SKIP_PREPARE=1, cache: $CACHE_ROOT)"
fi

echo "[2/5] ensure vendor dataset + config"
PATCH_SRC="scripts/train/vendor_patches/dataset_ForenShieldVideo.py"
PATCH_DST="vendor/TruFor/TruFor_train_test/dataset/dataset_ForenShieldVideo.py"
if [[ ! -f "$PATCH_DST" ]]; then
  cp "$PATCH_SRC" "$PATCH_DST"
  echo "copied $PATCH_DST"
fi

CFG_SRC="scripts/train/vendor_patches/trufor_forgery_video_r2.yaml"
CFG_DST="vendor/TruFor/TruFor_train_test/lib/config/trufor_forgery_video_r2.yaml"
cp "$CFG_SRC" "$CFG_DST"
echo "copied $CFG_DST"

echo "[3/5] train R2 (cause #3 data only; BEST_KEY=avg_p-F1, batch=$BATCH_SIZE, gpu=$GPU)"
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
  TRAIN.BATCH_SIZE_PER_GPU "$BATCH_SIZE" \
  WORKERS "$WORKERS"

CKPT_DEV="models/dev/spatial/trufor/v1.0.0/${EXP_NAME}/trufor.pth.tar"

echo "[4/5] merge"
python3 scripts/train/merge_trufor_infer_checkpoint.py \
  --base "$PRETRAINED" \
  --tuned "vendor/TruFor/TruFor_train_test/weights/${EXP_NAME}/best.pth.tar" \
  --out "$CKPT_DEV"

RUN_DATE="$(date +%Y%m%d-%H%M)"

echo "[5/5] infer @ threshold=${INFER_THRESHOLD} (mvtb first, then csvted)"
python3 scripts/infer/spatial_mvtamperbench_benchmark.py \
  --root "$ROOT" \
  --data-root data/pull/evidence/mvtamperbench-200-s3 \
  --model trufor --num-frames 8 --threshold "$INFER_THRESHOLD" \
  --trufor-weights "$CKPT_DEV" \
  --run-id "trufor-mvtb200-${EXP_NAME}-${RUN_DATE}"

python3 scripts/infer/spatial_mvtamperbench_benchmark.py \
  --root "$ROOT" \
  --data-root data/pull/evidence/csvted-200-balanced \
  --model trufor --num-frames 8 --threshold "$INFER_THRESHOLD" \
  --trufor-weights "$CKPT_DEV" \
  --run-id "trufor-csvted200-${EXP_NAME}-${RUN_DATE}"

MVTB_PRED="results/infer/trufor-mvtb200-${EXP_NAME}-${RUN_DATE}/predictions.json"
echo ""
echo "=== R2 gate @${INFER_THRESHOLD} (compare baseline only) ==="
echo "  baseline @0.5: mvtb TP63 FP51 | csvted TP24 FP11"
echo "  R1 (separate): forgery-r1-20260702-0101 — do not mix runs"
echo "  cat results/infer/trufor-mvtb200-${EXP_NAME}-${RUN_DATE}/metrics.json"
echo ""
echo "=== threshold sweep ==="
echo "  python3 scripts/infer/sweep_spatial_benchmark_threshold.py \\"
echo "    --predictions ${MVTB_PRED} --step 0.01"
echo ""
echo "done: $EXP_NAME"
echo "cache: $CACHE_ROOT"
echo "weights: vendor/TruFor/TruFor_train_test/weights/$EXP_NAME"
echo "dev ckpt: $CKPT_DEV"
