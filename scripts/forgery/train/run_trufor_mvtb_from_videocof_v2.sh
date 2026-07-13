#!/usr/bin/env bash
# Branch B: continue-FT from VideoCoF-v2 → mvtb spatial
#
# Init:  videocof-v2 best (adopted)
# Train: forgery-gmflow-train-400 (or MVTB_1K if present)
# Eval:  mvtamperbench-200-s3 + Phase0 thr sweep (gate TP>=63 FP<=51)
#
# GPU:
#   cd ~/forenShield-ai/forgery
#   source ~/forenShield-ai/.venv/bin/activate
#   export FORENSHIELD_AI_ROOT=$HOME/forenShield-ai/forgery
#   sed -i 's/\r$//' scripts/train/run_trufor_mvtb_from_videocof_v2.sh
#   bash scripts/train/run_trufor_mvtb_from_videocof_v2.sh
#
# Optional:
#   SKIP_TRAIN=1 EXP_NAME=...  # infer+calibrate only
#   DATA_ROOT=data/train/video/forgery-gmflow-train-mvtb-1k
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai/forgery}"
cd "$ROOT"
source "${HOME}/forenShield-ai/.venv/bin/activate"
export FORENSHIELD_AI_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Prefer restored 400; allow override to 1k when available
if [[ -z "${DATA_ROOT:-}" ]]; then
  if [[ -d data/train/video/forgery-gmflow-train-mvtb-1k ]]; then
    DATA_ROOT="data/train/video/forgery-gmflow-train-mvtb-1k"
  else
    DATA_ROOT="data/train/video/forgery-gmflow-train-400"
  fi
fi

CACHE_ROOT="${CACHE_ROOT:-data/processed/trufor-mvtb-from-videocof-v2}"
EXP_NAME="${EXP_NAME:-mvtb-from-videocof-v2-$(date +%Y%m%d-%H%M)}"
GPU="${GPU:-0}"
CONFIG_EXP="${CONFIG_EXP:-trufor_forgery_video_r5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
WORKERS="${WORKERS:-2}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
INFER_THRESHOLD="${INFER_THRESHOLD:-0.5}"
END_EPOCH="${END_EPOCH:-8}"
OVERSAMPLE_POSITIVE="${OVERSAMPLE_POSITIVE:-3}"
FRAMES="${FRAMES:-16}"
AGGREGATE="${AGGREGATE:-top3_mean}"
MVTB_GATE_MIN_TP="${MVTB_GATE_MIN_TP:-63}"
MVTB_GATE_MAX_FP="${MVTB_GATE_MAX_FP:-51}"

# VideoCoF-v2 adopted init (prefer models/dev copy)
PRETRAINED="${PRETRAINED:-models/dev/spatial/trufor/v1.0.0/videocof-v2-20260710-0800/trufor.pth.tar}"
if [[ ! -f "$PRETRAINED" ]]; then
  PRETRAINED="models/train/spatial/trufor/videocof-v2/trufor-videocof-v2-20260710-0800/trufor_videocof_v2_ft/best.pth.tar"
fi
BASELINE="${BASELINE:-models/test/spatial/trufor/v1.0.0/trufor.pth.tar}"
CKPT_DEV="models/dev/spatial/trufor/v1.0.0/${EXP_NAME}/trufor.pth.tar"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d-%H%M)}"
MVTB_RUN_ID="trufor-mvtb200-${EXP_NAME}-${RUN_DATE}"
MVTB_PRED="results/infer/${MVTB_RUN_ID}/predictions.json"

echo "=== Branch B: mvtb from VideoCoF-v2 ==="
echo "DATA_ROOT=$DATA_ROOT"
echo "PRETRAINED=$PRETRAINED"
echo "EXP_NAME=$EXP_NAME FRAMES=$FRAMES AGGREGATE=$AGGREGATE"

[[ -f "$PRETRAINED" ]] || { echo "ERROR: missing VideoCoF init: $PRETRAINED" >&2; exit 1; }
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: mvtb train pool missing: $DATA_ROOT" >&2
  echo "Restore forgery-gmflow-train-400 or *-mvtb-1k under data/train/video/ first." >&2
  exit 1
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  if [[ "$SKIP_PREPARE" != "1" ]]; then
    echo "[1/6] prepare (spatial middle + skip-OOW + oversample x${OVERSAMPLE_POSITIVE})"
    python3 scripts/train/prepare_trufor_video_frames.py \
      --data-root "$DATA_ROOT" \
      --out-dir "$CACHE_ROOT" \
      --frames-per-video "$FRAMES" \
      --valid-ratio 0.1 \
      --seed 42 \
      --require-middle-window \
      --skip-out-of-window-fake \
      --oversample-positive "$OVERSAMPLE_POSITIVE" \
      --spatial-types masking substitution rotate \
      --recipe-tag mvtb-from-vc2
  else
    echo "[1/6] prepare skipped ($CACHE_ROOT)"
  fi

  echo "[2/6] vendor patches"
  PATCH_DST="vendor/TruFor/TruFor_train_test/dataset/dataset_ForenShieldVideo.py"
  [[ -f "$PATCH_DST" ]] || cp scripts/train/vendor_patches/dataset_ForenShieldVideo.py "$PATCH_DST"
  cp scripts/train/vendor_patches/trufor_forgery_video_r5.yaml \
    "vendor/TruFor/TruFor_train_test/lib/config/${CONFIG_EXP}.yaml"

  echo "[3/6] train (init=VideoCoF-v2, epoch=${END_EPOCH})"
  python3 scripts/train/train_trufor_video_forgery.py \
    -exp "$CONFIG_EXP" \
    --run-name "$EXP_NAME" \
    -g "$GPU" \
    --cache-root "$CACHE_ROOT" \
    --pretrained-checkpoint "$PRETRAINED" \
    TRAIN.BATCH_SIZE_PER_GPU "$BATCH_SIZE" \
    TRAIN.END_EPOCH "$END_EPOCH" \
    WORKERS "$WORKERS"

  echo "[4/6] merge → $CKPT_DEV"
  mkdir -p "$(dirname "$CKPT_DEV")"
  # Prefer merging onto VideoCoF init (domain prior); fallback baseline
  MERGE_BASE="$PRETRAINED"
  [[ -f "$BASELINE" ]] || MERGE_BASE="$PRETRAINED"
  python3 scripts/train/merge_trufor_infer_checkpoint.py \
    --base "$MERGE_BASE" \
    --tuned "vendor/TruFor/TruFor_train_test/weights/${EXP_NAME}/best.pth.tar" \
    --out "$CKPT_DEV"
else
  echo "[1-4/6] train skipped"
  [[ -f "$CKPT_DEV" ]] || { echo "ERROR: missing $CKPT_DEV"; exit 1; }
fi

echo "[5/6] infer mvtb200 @${INFER_THRESHOLD} frames=${FRAMES} aggregate=${AGGREGATE}"
python3 scripts/infer/spatial_mvtamperbench_benchmark.py \
  --root "$ROOT" \
  --data-root data/pull/evidence/mvtamperbench-200-s3 \
  --model trufor \
  --num-frames "$FRAMES" \
  --aggregate "$AGGREGATE" \
  --threshold "$INFER_THRESHOLD" \
  --trufor-weights "$CKPT_DEV" \
  --run-id "$MVTB_RUN_ID"

echo "[6/6] Phase 0 thr sweep (gate TP>=${MVTB_GATE_MIN_TP} FP<=${MVTB_GATE_MAX_FP})"
python3 scripts/infer/sweep_spatial_benchmark_threshold.py \
  --predictions "$MVTB_PRED" \
  --step 0.01 || true

if [[ -f scripts/train/spatial_benchmark_calibrate_from_predictions.py ]]; then
  python3 scripts/train/spatial_benchmark_calibrate_from_predictions.py \
    --predictions "$MVTB_PRED" \
    --min-tp "$MVTB_GATE_MIN_TP" \
    --max-fp "$MVTB_GATE_MAX_FP" \
    --out-dir "models/dev/spatial/trufor/v1.0.0/${EXP_NAME}" \
    --note "Branch B mvtb-from-videocof-v2" || true
fi

echo "DONE Branch B"
echo "  ckpt: $CKPT_DEV"
echo "  pred: $MVTB_PRED"
echo "  Do NOT reuse VideoCoF thr=0.5 or R5 thr=0.185 without checking sweep."
