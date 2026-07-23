#!/usr/bin/env bash
# R5 + Phase 0 calibration (Option A+B hybrid)
#
# B: R5 combined training (R1 BEST_KEY + R2 skip-OOW + R3 oversample + R4 epoch)
# A: post-infer threshold sweep → mvtb gate (TP>=63 FP<=51) → metrics.json + calibration.json
#
# Usage (from ~/forenShield-ai/forgery):
#   sed -i 's/\r$//' scripts/train/run_trufor_forgery_train_r5_calibrated.sh
#   bash scripts/train/run_trufor_forgery_train_r5_calibrated.sh
#
# Optional:
#   SKIP_TRAIN=1 EXP_NAME=forgery-r5-... bash ...   # infer+calibrate only
#   MVTB_GATE_MIN_TP=63 MVTB_GATE_MAX_FP=51

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

source "${HOME}/forenShield-ai/.venv/bin/activate"

DATA_ROOT="${ROOT}/data/train/video/forgery-gmflow-train-400"
CACHE_ROOT="${ROOT}/data/processed/trufor-gmflow-train-400-r5"
EXP_NAME="${EXP_NAME:-forgery-r5-$(date +%Y%m%d-%H%M)}"
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

if [[ "$SKIP_TRAIN" != "1" ]]; then
  if [[ "$SKIP_PREPARE" != "1" ]]; then
    echo "[1/7] prepare frames (R5: skip-OOW + oversample x${OVERSAMPLE_POSITIVE})"
    python3 scripts/train/prepare_trufor_video_frames.py \
      --data-root "$DATA_ROOT" \
      --out-dir "$CACHE_ROOT" \
      --frames-per-video 8 \
      --valid-ratio 0.1 \
      --seed 42 \
      --require-middle-window \
      --skip-out-of-window-fake \
      --oversample-positive "$OVERSAMPLE_POSITIVE" \
      --recipe-tag r5
  else
    echo "[1/7] prepare skipped (cache: $CACHE_ROOT)"
  fi

  echo "[2/7] vendor dataset + config (R5)"
  PATCH_DST="vendor/TruFor/TruFor_train_test/dataset/dataset_ForenShieldVideo.py"
  if [[ ! -f "$PATCH_DST" ]]; then
    cp scripts/train/vendor_patches/dataset_ForenShieldVideo.py "$PATCH_DST"
  fi
  cp scripts/train/vendor_patches/trufor_forgery_video_r5.yaml \
    vendor/TruFor/TruFor_train_test/lib/config/trufor_forgery_video_r5.yaml

  echo "[3/7] train R5 (epoch=${END_EPOCH}, gpu=$GPU)"
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

echo "[5/7] infer @${INFER_THRESHOLD} (mvtb + csvted)"
python3 scripts/infer/spatial_mvtamperbench_benchmark.py \
  --root "$ROOT" \
  --data-root data/pull/evidence/mvtamperbench-200-s3 \
  --model trufor --num-frames 8 --threshold "$INFER_THRESHOLD" \
  --trufor-weights "$CKPT_DEV" \
  --run-id "$MVTB_RUN_ID"

python3 scripts/infer/spatial_mvtamperbench_benchmark.py \
  --root "$ROOT" \
  --data-root data/pull/evidence/csvted-200-balanced \
  --model trufor --num-frames 8 --threshold "$INFER_THRESHOLD" \
  --trufor-weights "$CKPT_DEV" \
  --run-id "$CSVTED_RUN_ID"

echo "[6/7] Phase 0 calibration — mvtb gate TP>=${MVTB_GATE_MIN_TP} FP<=${MVTB_GATE_MAX_FP}"
python3 scripts/infer/sweep_spatial_benchmark_threshold.py \
  --predictions "$MVTB_PRED" \
  --step 0.01 || true

MVTB_CALIB_OUT="$(python3 scripts/infer/spatial_benchmark_calibrate_from_predictions.py \
  --predictions "$MVTB_PRED" \
  --weights "$CKPT_DEV" \
  --gate \
  --min-tp "$MVTB_GATE_MIN_TP" \
  --max-fp "$MVTB_GATE_MAX_FP" \
  --step "$CALIB_STEP" \
  --note "R5+A hybrid: gate-calibrated mvtb threshold")"

echo "$MVTB_CALIB_OUT"
MVTB_CAL_THR="$(echo "$MVTB_CALIB_OUT" | sed -n 's/^gate thr=\([0-9.]*\).*/\1/p' | head -1)"
if [[ -z "$MVTB_CAL_THR" ]]; then
  echo "WARN: mvtb gate not satisfied — keeping @${INFER_THRESHOLD} metrics only"
  MVTB_CAL_THR="$INFER_THRESHOLD"
  python3 scripts/infer/spatial_benchmark_calibrate_from_predictions.py \
    --predictions "$MVTB_PRED" \
    --threshold "$MVTB_CAL_THR" \
    --weights "$CKPT_DEV" \
    --note "R5+A hybrid: gate failed, fallback infer threshold"
fi

echo "[7/7] write calibration.json (+ csvted FP<=11 reference)"
CALIB_DIR="$(dirname "$CKPT_DEV")"
mkdir -p "$CALIB_DIR"
cp "results/infer/${MVTB_RUN_ID}/metrics.json" "${CALIB_DIR}/metrics_mvtb_calibrated.json"

python3 - <<PY
import json
from pathlib import Path

exp = "${EXP_NAME}"
mvtb_thr = float("${MVTB_CAL_THR}")
ckpt = "${CKPT_DEV}"
mvtb_metrics = json.loads(Path("results/infer/${MVTB_RUN_ID}/metrics.json").read_text())
csvted_pred = Path("${CSVTED_PRED}")

# csvted: best TP with FP<=11 (baseline FP anchor)
csvted_thr = None
csvted_conf = None
if csvted_pred.exists():
    items = json.loads(csvted_pred.read_text())["items"]
    scores = [(float(x["tamper_score"]), 1 if x["ground_truth_label"] == "fake" else 0) for x in items]
    best = None
    for i in range(10, 60):
        thr = round(i * 0.01, 2)
        tp=fp=fn=tn=0
        for s, y in scores:
            pred = 1 if s >= thr else 0
            if y == 1 and pred == 1: tp += 1
            elif y == 0 and pred == 1: fp += 1
            elif y == 1 and pred == 0: fn += 1
            else: tn += 1
        if fp <= 11:
            rank = (tp, -fp)
            if best is None or rank > best[0]:
                best = (rank, thr, {"tp": tp, "tn": tn, "fp": fp, "fn": fn})
    if best:
        csvted_thr = best[1]
        csvted_conf = best[2]
        note = "R5+A hybrid: csvted FP<=11 reference (not adopted for mvtb)"
        from pathlib import Path as P
        import subprocess, sys
        subprocess.run([
            sys.executable, "scripts/infer/spatial_benchmark_calibrate_from_predictions.py",
            "--predictions", str(csvted_pred),
            "--threshold", str(csvted_thr),
            "--weights", ckpt,
            "--note", note,
        ], check=True)

doc = {
    "model": "trufor",
    "modality": "spatial",
    "version": "v1.0.0",
    "run_name": exp,
    "status": "dev_candidate_ab_hybrid",
    "strategy": "R5_combined_training + Phase0_mvtb_gate_calibration",
    "checkpoint": ckpt,
    "pretrain": "${PRETRAINED}",
    "cache": "${CACHE_ROOT}",
    "config_exp": "${CONFIG_EXP}",
    "benchmarks": {
        "mvtb": {
            "adopted": mvtb_metrics["confusion"]["tp"] >= ${MVTB_GATE_MIN_TP} and mvtb_metrics["confusion"]["fp"] <= ${MVTB_GATE_MAX_FP},
            "threshold": mvtb_thr,
            "run_id": "${MVTB_RUN_ID}",
            "metrics_path": f"results/infer/${MVTB_RUN_ID}/metrics.json",
            "confusion": mvtb_metrics["confusion"],
            "accuracy": mvtb_metrics["accuracy"],
            "roc_auc": mvtb_metrics.get("roc_auc"),
            "gate": "TP>=${MVTB_GATE_MIN_TP} & FP<=${MVTB_GATE_MAX_FP}",
        },
        "csvted": {
            "adopted": False,
            "note": "reference only; temporal/fusion separate",
            "optional_threshold_fp_le_11": csvted_thr,
            "optional_confusion": csvted_conf,
            "run_id": "${CSVTED_RUN_ID}",
        },
    },
    "doc": "docs/ai/17-TruFor-v5-원인1-BEST_KEY-개선-실험.md §9.19",
}
out = Path("${CALIB_DIR}/calibration.json")
out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
PY

echo ""
echo "=== R5+A hybrid done ==="
echo "  EXP_NAME:     $EXP_NAME"
echo "  dev ckpt:     $CKPT_DEV"
echo "  mvtb RUN_ID:  $MVTB_RUN_ID"
echo "  mvtb thr:     $MVTB_CAL_THR (gate TP>=${MVTB_GATE_MIN_TP} FP<=${MVTB_GATE_MAX_FP})"
echo "  calibration:  ${CALIB_DIR}/calibration.json"
echo "  metrics:      cat results/infer/${MVTB_RUN_ID}/metrics.json"
echo ""
echo "  Compare vs R3 dev (thr=0.158): forgery-r3-20260702-0338"
echo "  models/test/deploy: not promoted until team review"
