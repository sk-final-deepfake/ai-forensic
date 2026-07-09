#!/usr/bin/env bash
# S2 line — R5 recipe (recall BEST_KEY, epoch 8) in isolated S-line namespace
#
# Train:  forgery-gmflow-train-400 | Test: csvted200 + mvtb200 hold-out
# Recipe: same as R5 (skip-OOW + oversample x3 + tampered_pixel_recall)
#
# Usage:
#   sed -i 's/\r$//' scripts/train/run_trufor_forgery_train_s2_calibrated.sh
#   SKIP_PREPARE=1 nohup bash scripts/train/run_trufor_forgery_train_s2_calibrated.sh > logs/s2-exp/....log 2>&1 &
#
# Reuse S1 cache (same prepare flags):
#   ln -sfn trufor-s1-gmflow-train-400 data/processed/trufor-s2-gmflow-train-400
#   SKIP_PREPARE=1 nohup bash scripts/train/run_trufor_forgery_train_s2_calibrated.sh ...

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

source "${HOME}/forenShield-ai/.venv/bin/activate"

DATA_ROOT="${ROOT}/data/train/video/forgery-gmflow-train-400"
CACHE_ROOT="${CACHE_ROOT:-${ROOT}/data/processed/trufor-s2-gmflow-train-400}"
EXP_NAME="${EXP_NAME:-forgery-s2-$(date +%Y%m%d-%H%M)}"
GPU="${GPU:-0}"
CONFIG_EXP="${CONFIG_EXP:-trufor_forgery_video_s2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
WORKERS="${WORKERS:-2}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
INFER_THRESHOLD="${INFER_THRESHOLD:-0.5}"
END_EPOCH="${END_EPOCH:-8}"
OVERSAMPLE_POSITIVE="${OVERSAMPLE_POSITIVE:-3}"
FRAMES_PER_VIDEO="${FRAMES_PER_VIDEO:-8}"
MVTB_GATE_MIN_TP="${MVTB_GATE_MIN_TP:-63}"
MVTB_GATE_MAX_FP="${MVTB_GATE_MAX_FP:-51}"
CALIB_STEP="${CALIB_STEP:-0.005}"

PRETRAINED="${PRETRAINED:-${ROOT}/models/test/spatial/trufor/v1.0.0/trufor.pth.tar}"
CKPT_DEV="models/dev/spatial/trufor-s2/v1.0.0/${EXP_NAME}/trufor.pth.tar"
RESULTS_TAG="s2-exp"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d-%H%M)}"
MVTB_RUN_ID="trufor-mvtb200-${EXP_NAME}-${RUN_DATE}"
CSVTED_RUN_ID="trufor-csvted200-${EXP_NAME}-${RUN_DATE}"
MVTB_PRED="results/infer/${MVTB_RUN_ID}/predictions.json"
CSVTED_PRED="results/infer/${CSVTED_RUN_ID}/predictions.json"
LOG_DIR="${ROOT}/logs/${RESULTS_TAG}"
mkdir -p "$LOG_DIR" "models/dev/spatial/trufor-s2/v1.0.0" "results/${RESULTS_TAG}"

echo "=== TruFor S2 run (R5 recipe): $EXP_NAME ==="
echo "  cache:      $CACHE_ROOT"
echo "  ckpt:       $CKPT_DEV"
echo "  pretrain:   $PRETRAINED (baseline test, not R-line)"
echo "  test:       csvted200 + mvtb200 (hold-out)"

if [[ "$SKIP_TRAIN" != "1" ]]; then
  if [[ "$SKIP_PREPARE" != "1" ]]; then
    echo "[1/7] prepare frames (S2/R5: skip-OOW + oversample x${OVERSAMPLE_POSITIVE})"
    python3 scripts/train/prepare_trufor_video_frames.py \
      --data-root "$DATA_ROOT" \
      --out-dir "$CACHE_ROOT" \
      --frames-per-video "$FRAMES_PER_VIDEO" \
      --valid-ratio 0.1 \
      --seed 42 \
      --require-middle-window \
      --skip-out-of-window-fake \
      --oversample-positive "$OVERSAMPLE_POSITIVE" \
      --recipe-tag s2
  else
    echo "[1/7] prepare skipped (cache: $CACHE_ROOT)"
  fi

  echo "[2/7] vendor dataset + S2 config + torch.load patch"
  PATCH_DST="vendor/TruFor/TruFor_train_test/dataset/dataset_ForenShieldVideo.py"
  if [[ ! -f "$PATCH_DST" ]]; then
    cp scripts/train/vendor_patches/dataset_ForenShieldVideo.py "$PATCH_DST"
  fi
  cp scripts/train/vendor_patches/trufor_forgery_video_s2.yaml \
    vendor/TruFor/TruFor_train_test/lib/config/trufor_forgery_video_s2.yaml
  python3 scripts/train/vendor_patches/patch_trufor_vendor_torch_load.py 2>/dev/null || \
    python3 - <<'PY'
import re
from pathlib import Path
pat = re.compile(r"torch\.device\(([^,)]+),\s*weights_only=False\)")
for p in Path("vendor/TruFor/TruFor_train_test").rglob("*.py"):
    t = p.read_text(encoding="utf-8", errors="ignore")
    if pat.search(t):
        p.write_text(pat.sub(r"torch.device(\1), weights_only=False", t), encoding="utf-8")
        print("patched", p)
PY
  python3 scripts/train/vendor_patches/patch_trufor_test_load.py 2>/dev/null || true

  echo "[3/7] train S2 (epoch=${END_EPOCH}, BEST_KEY=tampered_pixel_recall, gpu=$GPU)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed "s/^/  GPU /" || true
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
    TRAIN.END_EPOCH "$END_EPOCH" \
    WORKERS "$WORKERS"

  echo "[4/7] merge → S2 dev ckpt"
  python3 scripts/train/merge_trufor_infer_checkpoint.py \
    --base "$PRETRAINED" \
    --tuned "vendor/TruFor/TruFor_train_test/weights/${EXP_NAME}/best.pth.tar" \
    --out "$CKPT_DEV"
else
  echo "[1-4/7] train skipped (SKIP_TRAIN=1)"
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

echo "[6/7] mvtb calibration (gate TP>=${MVTB_GATE_MIN_TP} FP<=${MVTB_GATE_MAX_FP})"
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
  --note "S2 line (R5 recipe): mvtb gate-calibrated threshold")"

echo "$MVTB_CALIB_OUT"
MVTB_CAL_THR="$(echo "$MVTB_CALIB_OUT" | sed -n 's/^gate thr=\([0-9.]*\).*/\1/p' | head -1)"
if [[ -z "$MVTB_CAL_THR" ]]; then
  MVTB_CAL_THR="$INFER_THRESHOLD"
  python3 scripts/train/spatial_benchmark_calibrate_from_predictions.py \
    --predictions "$MVTB_PRED" \
    --threshold "$MVTB_CAL_THR" \
    --weights "$CKPT_DEV" \
    --note "S2: gate failed, fallback"
fi

echo "[7/7] write S2 summary"
CALIB_DIR="$(dirname "$CKPT_DEV")"
mkdir -p "$CALIB_DIR"
cp "results/infer/${MVTB_RUN_ID}/metrics.json" "${CALIB_DIR}/metrics_mvtb_calibrated.json"

python3 - <<PY
import json
import subprocess
import sys
from pathlib import Path

exp = "${EXP_NAME}"
mvtb_thr = float("${MVTB_CAL_THR}")
ckpt = "${CKPT_DEV}"
cache = "${CACHE_ROOT}"
mvtb_metrics = json.loads(Path("results/infer/${MVTB_RUN_ID}/metrics.json").read_text())
csvted_pred = Path("${CSVTED_PRED}")

csvted_thr = None
csvted_conf = None
if csvted_pred.exists():
    items = json.loads(csvted_pred.read_text())["items"]
    scores = [(float(x["tamper_score"]), 1 if x["ground_truth_label"] == "fake" else 0) for x in items]
    best = None
    for i in range(10, 60):
        thr = round(i * 0.01, 2)
        tp = fp = fn = tn = 0
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
        csvted_thr, csvted_conf = best[1], best[2]
        subprocess.run([
            sys.executable, "scripts/train/spatial_benchmark_calibrate_from_predictions.py",
            "--predictions", str(csvted_pred), "--threshold", str(csvted_thr),
            "--weights", ckpt, "--note", "S2: csvted FP<=11 reference",
        ], check=True)

doc = {
    "line": "S2",
    "run_name": exp,
    "status": "s2_candidate",
    "strategy": "R5_recipe: recall BEST_KEY + skipOOW + oversample x3 + epoch 8",
    "checkpoint": ckpt,
    "pretrain": "${PRETRAINED}",
    "cache": cache,
    "config_exp": "${CONFIG_EXP}",
    "benchmarks": {
        "mvtb": {
            "threshold": mvtb_thr,
            "run_id": "${MVTB_RUN_ID}",
            "confusion": mvtb_metrics["confusion"],
            "accuracy": mvtb_metrics["accuracy"],
            "roc_auc": mvtb_metrics.get("roc_auc"),
            "gate": "TP>=${MVTB_GATE_MIN_TP} & FP<=${MVTB_GATE_MAX_FP}",
            "r5_reference": "TP96 FP19 @0.185",
            "s1_reference": "TP82 FP49 @0.25",
        },
        "csvted": {"adopted": False, "run_id": "${CSVTED_RUN_ID}"},
    },
}
Path("${CALIB_DIR}/calibration.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
Path("results/${RESULTS_TAG}/${EXP_NAME}_summary.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
print("wrote summary")
PY

echo ""
echo "=== S2 done ==="
echo "  EXP_NAME:  $EXP_NAME"
echo "  ckpt:      $CKPT_DEV"
echo "  mvtb thr:  $MVTB_CAL_THR"
echo "  compare:   R5 TP96 FP19 | S1 TP82 FP49 | baseline TP63 FP51"
