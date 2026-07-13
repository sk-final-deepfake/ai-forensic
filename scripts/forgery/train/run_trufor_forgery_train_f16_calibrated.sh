#!/usr/bin/env bash
# F16 line — R5 recipe + 16 frames/video (prepare + infer aligned)
#
# Same training recipe as R5 (recall BEST_KEY, skip-OOW, oversample x3, epoch 8).
# Only change vs R5: --frames-per-video 16 and infer --num-frames 16.
#
# Usage (from ~/forenShield-ai/forgery):
#   sed -i 's/\r$//' scripts/train/run_trufor_forgery_train_f16_calibrated.sh
#   nohup bash scripts/train/run_trufor_forgery_train_f16_calibrated.sh \
#     > logs/f16-exp/trufor-f16-$(date +%Y%m%d-%H%M).log 2>&1 &
#
# Optional:
#   SKIP_TRAIN=1 EXP_NAME=forgery-f16-... bash ...
#   FRAMES=16 (default)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

source "${HOME}/forenShield-ai/.venv/bin/activate"

FRAMES="${FRAMES:-16}"
DATA_ROOT="${ROOT}/data/train/video/forgery-gmflow-train-400"
CACHE_ROOT="${ROOT}/data/processed/trufor-gmflow-train-400-f16"
EXP_NAME="${EXP_NAME:-forgery-f16-$(date +%Y%m%d-%H%M)}"
GPU="${GPU:-0}"
CONFIG_EXP="${CONFIG_EXP:-trufor_forgery_video_r5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
WORKERS="${WORKERS:-2}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
INFER_THRESHOLD="${INFER_THRESHOLD:-0.5}"
END_EPOCH="${END_EPOCH:-8}"
OVERSAMPLE_POSITIVE="${OVERSAMPLE_POSITIVE:-3}"
MVTB_GATE_MIN_TP="${MVTB_GATE_MIN_TP:-63}"
MVTB_GATE_MAX_FP="${MVTB_GATE_MAX_FP:-51}"
CALIB_STEP="${CALIB_STEP:-0.005}"

PRETRAINED="${PRETRAINED:-${ROOT}/models/test/spatial/trufor/v1.0.0/trufor.pth.tar}"
CKPT_DEV="models/dev/spatial/trufor/v1.0.0/${EXP_NAME}/trufor.pth.tar"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d-%H%M)}"
MVTB_RUN_ID="trufor-mvtb200-${EXP_NAME}-${RUN_DATE}"
CSVTED_RUN_ID="trufor-csvted200-${EXP_NAME}-${RUN_DATE}"
MVTB_PRED="results/infer/${MVTB_RUN_ID}/predictions.json"
CSVTED_PRED="results/infer/${CSVTED_RUN_ID}/predictions.json"

mkdir -p logs/f16-exp

if [[ "$SKIP_TRAIN" != "1" ]]; then
  if [[ "$SKIP_PREPARE" != "1" ]]; then
    echo "[1/7] prepare frames (f16: ${FRAMES} frames/video, skip-OOW + oversample x${OVERSAMPLE_POSITIVE})"
    python3 scripts/train/prepare_trufor_video_frames.py \
      --data-root "$DATA_ROOT" \
      --out-dir "$CACHE_ROOT" \
      --frames-per-video "$FRAMES" \
      --valid-ratio 0.1 \
      --seed 42 \
      --require-middle-window \
      --skip-out-of-window-fake \
      --oversample-positive "$OVERSAMPLE_POSITIVE" \
      --recipe-tag "f16-${FRAMES}"
  else
    echo "[1/7] prepare skipped (cache: $CACHE_ROOT)"
  fi

  echo "[2/7] vendor dataset + config (R5 yaml, f16 cache)"
  PATCH_DST="vendor/TruFor/TruFor_train_test/dataset/dataset_ForenShieldVideo.py"
  if [[ ! -f "$PATCH_DST" ]]; then
    cp scripts/train/vendor_patches/dataset_ForenShieldVideo.py "$PATCH_DST"
  fi
  cp scripts/train/vendor_patches/trufor_forgery_video_r5.yaml \
    vendor/TruFor/TruFor_train_test/lib/config/trufor_forgery_video_r5.yaml

  echo "[3/7] train f16/R5-recipe (epoch=${END_EPOCH}, gpu=$GPU, frames=${FRAMES})"
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
    TRAIN.END_EPOCH "$END_EPOCH" \
    WORKERS "$WORKERS"

  echo "[4/7] merge → dev ckpt"
  python3 scripts/train/merge_trufor_infer_checkpoint.py \
    --base "$PRETRAINED" \
    --tuned "vendor/TruFor/TruFor_train_test/weights/${EXP_NAME}/best.pth.tar" \
    --out "$CKPT_DEV"
else
  echo "[1-4/7] train skipped (SKIP_TRAIN=1, EXP_NAME=$EXP_NAME)"
  if [[ ! -f "$CKPT_DEV" ]]; then
    echo "ERROR: missing $CKPT_DEV"
    exit 1
  fi
fi

echo "[5/7] infer @${INFER_THRESHOLD}, num-frames=${FRAMES} (mvtb + csvted)"
python3 scripts/infer/spatial_mvtamperbench_benchmark.py \
  --root "$ROOT" \
  --data-root data/pull/evidence/mvtamperbench-200-s3 \
  --model trufor --num-frames "$FRAMES" --threshold "$INFER_THRESHOLD" \
  --trufor-weights "$CKPT_DEV" \
  --run-id "$MVTB_RUN_ID"

python3 scripts/infer/spatial_mvtamperbench_benchmark.py \
  --root "$ROOT" \
  --data-root data/pull/evidence/csvted-200-balanced \
  --model trufor --num-frames "$FRAMES" --threshold "$INFER_THRESHOLD" \
  --trufor-weights "$CKPT_DEV" \
  --run-id "$CSVTED_RUN_ID"

echo "[6/7] Phase 0 calibration — mvtb gate TP>=${MVTB_GATE_MIN_TP} FP<=${MVTB_GATE_MAX_FP}"
python3 scripts/infer/sweep_spatial_benchmark_threshold.py \
  --predictions "$MVTB_PRED" \
  --step 0.01 || true

MVTB_CALIB_OUT="$(python3 scripts/train/spatial_benchmark_calibrate_from_predictions.py \
  --predictions "$MVTB_PRED" \
  --weights "$CKPT_DEV" \
  --gate \
  --min-tp "$MVTB_GATE_MIN_TP" \
  --max-fp "$MVTB_GATE_MAX_FP" \
  --step "$CALIB_STEP" \
  --note "F16: gate-calibrated mvtb threshold (${FRAMES} frames)")"

echo "$MVTB_CALIB_OUT"
MVTB_CAL_THR="$(echo "$MVTB_CALIB_OUT" | sed -n 's/^gate thr=\([0-9.]*\).*/\1/p' | head -1)"
if [[ -z "$MVTB_CAL_THR" ]]; then
  echo "WARN: mvtb gate not satisfied — keeping @${INFER_THRESHOLD} metrics only"
  MVTB_CAL_THR="$INFER_THRESHOLD"
  python3 scripts/train/spatial_benchmark_calibrate_from_predictions.py \
    --predictions "$MVTB_PRED" \
    --threshold "$MVTB_CAL_THR" \
    --weights "$CKPT_DEV" \
    --note "F16: gate failed, fallback infer threshold"
fi

echo "[7/7] write calibration.json"
CALIB_DIR="$(dirname "$CKPT_DEV")"
mkdir -p "$CALIB_DIR"
cp "results/infer/${MVTB_RUN_ID}/metrics.json" "${CALIB_DIR}/metrics_mvtb_calibrated.json"

python3 - <<PY
import json
from pathlib import Path

exp = "${EXP_NAME}"
mvtb_thr = float("${MVTB_CAL_THR}")
ckpt = "${CKPT_DEV}"
frames = int("${FRAMES}")
mvtb_metrics = json.loads(Path("results/infer/${MVTB_RUN_ID}/metrics.json").read_text())

doc = {
    "line": "F16",
    "model": "trufor",
    "modality": "spatial",
    "version": "v1.0.0",
    "run_name": exp,
    "status": "f16_candidate",
    "strategy": f"R5_recipe + {frames}_frames_per_video",
    "frames_per_video": frames,
    "checkpoint": ckpt,
    "pretrain": "${PRETRAINED}",
    "cache": "${CACHE_ROOT}",
    "config_exp": "${CONFIG_EXP}",
    "benchmarks": {
        "mvtb": {
            "threshold": mvtb_thr,
            "num_frames": frames,
            "run_id": "${MVTB_RUN_ID}",
            "confusion": mvtb_metrics["confusion"],
            "accuracy": mvtb_metrics["accuracy"],
            "roc_auc": mvtb_metrics.get("roc_auc"),
            "gate": "TP>=${MVTB_GATE_MIN_TP} & FP<=${MVTB_GATE_MAX_FP}",
        },
        "csvted": {
            "adopted": False,
            "note": "separate threshold; do not share mvtb thr",
            "num_frames": frames,
            "run_id": "${CSVTED_RUN_ID}",
        },
    },
}
out = Path("${CALIB_DIR}/calibration.json")
out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
PY

echo ""
echo "=== F16 done ==="
echo "  EXP_NAME:     $EXP_NAME"
echo "  FRAMES:       $FRAMES"
echo "  dev ckpt:     $CKPT_DEV"
echo "  mvtb thr:     $MVTB_CAL_THR"
echo "  calibration:  ${CALIB_DIR}/calibration.json"
echo ""
echo "  Next (hold-out, same frame count):"
echo "    CKPT=$CKPT_DEV LOCKED_THR=$MVTB_CAL_THR NUM_FRAMES=$FRAMES \\"
echo "    RUN_ID=trufor-mvtb500-holdout-f16-\$(date +%Y%m%d-%H%M) \\"
echo "    bash scripts/train/run_mvtb500_holdout_eval.sh"
