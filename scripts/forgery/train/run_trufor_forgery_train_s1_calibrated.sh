#!/usr/bin/env bash
# S1 line ??TruFor spatial fine-tune + fixed 400 eval + mvtb calibration
# Isolated from baseline models/test and R-line (R1~R5) paths.
#
# Train:  forgery-gmflow-train-400 (400 video, test 400 excluded by manifest)
# Test:   csvted-200-balanced + mvtamperbench-200-s3
#
# Usage (from ~/forenShield-ai/forgery):
#   sed -i 's/\r$//' scripts/train/run_trufor_forgery_train_s1_calibrated.sh
#   nohup bash scripts/train/run_trufor_forgery_train_s1_calibrated.sh \
#     > logs/trufor-s1-$(date +%Y%m%d-%H%M).log 2>&1 &
#
# Optional:
#   SKIP_TRAIN=1 EXP_NAME=forgery-s1-... bash ...
#   END_EPOCH=12 OVERSAMPLE_POSITIVE=3 bash ...

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

source "${HOME}/forenShield-ai/.venv/bin/activate"

DATA_ROOT="${ROOT}/data/train/video/forgery-gmflow-train-400"
CACHE_ROOT="${ROOT}/data/processed/trufor-s1-gmflow-train-400"
EXP_NAME="${EXP_NAME:-forgery-s1-$(date +%Y%m%d-%H%M)}"
GPU="${GPU:-0}"
CONFIG_EXP="${CONFIG_EXP:-trufor_forgery_video_s1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
WORKERS="${WORKERS:-2}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
INFER_THRESHOLD="${INFER_THRESHOLD:-0.5}"
END_EPOCH="${END_EPOCH:-10}"
OVERSAMPLE_POSITIVE="${OVERSAMPLE_POSITIVE:-3}"
FRAMES_PER_VIDEO="${FRAMES_PER_VIDEO:-8}"
MVTB_GATE_MIN_TP="${MVTB_GATE_MIN_TP:-63}"
MVTB_GATE_MAX_FP="${MVTB_GATE_MAX_FP:-51}"
CALIB_STEP="${CALIB_STEP:-0.005}"

# S1 namespace ??do not write into R-line or models/test
PRETRAINED="${PRETRAINED:-${ROOT}/models/test/spatial/trufor/v1.0.0/trufor.pth.tar}"
CKPT_DEV="models/dev/spatial/trufor-s1/v1.0.0/${EXP_NAME}/trufor.pth.tar"
RESULTS_TAG="s1-exp"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d-%H%M)}"
MVTB_RUN_ID="trufor-mvtb200-${EXP_NAME}-${RUN_DATE}"
CSVTED_RUN_ID="trufor-csvted200-${EXP_NAME}-${RUN_DATE}"
MVTB_PRED="results/infer/${MVTB_RUN_ID}/predictions.json"
CSVTED_PRED="results/infer/${CSVTED_RUN_ID}/predictions.json"
LOG_DIR="${ROOT}/logs/${RESULTS_TAG}"
mkdir -p "$LOG_DIR" "models/dev/spatial/trufor-s1/v1.0.0" "results/${RESULTS_TAG}"

echo "=== TruFor S1 run: $EXP_NAME ==="
echo "  cache:      $CACHE_ROOT"
echo "  ckpt:       $CKPT_DEV"
echo "  pretrain:   $PRETRAINED (baseline test, not R-line)"
echo "  test:       csvted200 + mvtb200 (hold-out)"

if [[ "$SKIP_TRAIN" != "1" ]]; then
  if [[ "$SKIP_PREPARE" != "1" ]]; then
    echo "[1/7] prepare frames (S1: skip-OOW + oversample x${OVERSAMPLE_POSITIVE})"
    python3 scripts/train/prepare_trufor_video_frames.py \
      --data-root "$DATA_ROOT" \
      --out-dir "$CACHE_ROOT" \
      --frames-per-video "$FRAMES_PER_VIDEO" \
      --valid-ratio 0.1 \
      --seed 42 \
      --require-middle-window \
      --skip-out-of-window-fake \
      --oversample-positive "$OVERSAMPLE_POSITIVE" \
      --recipe-tag s1
  else
    echo "[1/7] prepare skipped (cache: $CACHE_ROOT)"
  fi

  echo "[2/7] vendor dataset + S1 config + torch.load patch"
  PATCH_DST="vendor/TruFor/TruFor_train_test/dataset/dataset_ForenShieldVideo.py"
  if [[ ! -f "$PATCH_DST" ]]; then
    cp scripts/train/vendor_patches/dataset_ForenShieldVideo.py "$PATCH_DST"
  fi
  cp scripts/train/vendor_patches/trufor_forgery_video_s1.yaml \
    vendor/TruFor/TruFor_train_test/lib/config/trufor_forgery_video_s1.yaml
  python3 scripts/train/vendor_patches/patch_trufor_vendor_torch_load.py || true
  python3 scripts/train/vendor_patches/patch_trufor_test_load.py 2>/dev/null || true

  echo "[3/7] train S1 (epoch=${END_EPOCH}, BEST_KEY=tampered_pixel_f1, gpu=$GPU)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader | sed "s/^/  GPU /" || true
  fi
  PRETRAIN_ARG=()
  if [[ -f "$PRETRAINED" ]]; then
    PRETRAIN_ARG=(--pretrained-checkpoint "$PRETRAINED")
  else
    echo "WARN: baseline pretrained missing at $PRETRAINED"
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

  echo "[4/7] merge ??S1 dev ckpt"
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

echo "[5/7] infer @${INFER_THRESHOLD} (mvtb + csvted hold-out 400)"
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

echo "[6/7] mvtb threshold calibration (gate TP>=${MVTB_GATE_MIN_TP} FP<=${MVTB_GATE_MAX_FP})"
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
  --note "S1 line: mvtb gate-calibrated threshold")"

echo "$MVTB_CALIB_OUT"
MVTB_CAL_THR="$(echo "$MVTB_CALIB_OUT" | sed -n 's/^gate thr=\([0-9.]*\).*/\1/p' | head -1)"
if [[ -z "$MVTB_CAL_THR" ]]; then
  echo "WARN: mvtb gate not satisfied ??fallback @${INFER_THRESHOLD}"
  MVTB_CAL_THR="$INFER_THRESHOLD"
  python3 scripts/infer/spatial_benchmark_calibrate_from_predictions.py \
    --predictions "$MVTB_PRED" \
    --threshold "$MVTB_CAL_THR" \
    --weights "$CKPT_DEV" \
    --note "S1: gate failed, fallback infer threshold"
fi

echo "[7/7] write S1 summary"
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
baseline_mvtb = {"tp": 63, "fp": 51, "acc": 0.56, "auc": 0.5709}
baseline_csvted = {"tp": 24, "fp": 11, "acc": 0.565, "auc": 0.6285}

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
            if y == 1 and pred == 1:
                tp += 1
            elif y == 0 and pred == 1:
                fp += 1
            elif y == 1 and pred == 0:
                fn += 1
            else:
                tn += 1
        if fp <= 11:
            rank = (tp, -fp)
            if best is None or rank > best[0]:
                best = (rank, thr, {"tp": tp, "tn": tn, "fp": fp, "fn": fn})
    if best:
        csvted_thr = best[1]
        csvted_conf = best[2]
        subprocess.run([
            sys.executable,
            "scripts/infer/spatial_benchmark_calibrate_from_predictions.py",
            "--predictions", str(csvted_pred),
            "--threshold", str(csvted_thr),
            "--weights", ckpt,
            "--note", "S1: csvted FP<=11 reference (not mvtb-shared)",
        ], check=True)

doc = {
    "line": "S1",
    "model": "trufor",
    "modality": "spatial",
    "version": "v1.0.0",
    "run_name": exp,
    "status": "s1_candidate",
    "strategy": "S1_f1_bestkey + skipOOW + oversample + mvtb_gate_calibration",
    "checkpoint": ckpt,
    "pretrain": "${PRETRAINED}",
    "cache": cache,
    "config_exp": "${CONFIG_EXP}",
    "isolated_from": ["models/test baseline infer", "R1-R5 dev checkpoints"],
    "benchmarks": {
        "mvtb": {
            "threshold": mvtb_thr,
            "run_id": "${MVTB_RUN_ID}",
            "metrics_path": f"results/infer/${MVTB_RUN_ID}/metrics.json",
            "confusion": mvtb_metrics["confusion"],
            "accuracy": mvtb_metrics["accuracy"],
            "roc_auc": mvtb_metrics.get("roc_auc"),
            "gate": "TP>=${MVTB_GATE_MIN_TP} & FP<=${MVTB_GATE_MAX_FP}",
            "baseline_anchor": baseline_mvtb,
        },
        "csvted": {
            "adopted": False,
            "note": "temporal-heavy; spatial TruFor limited",
            "optional_threshold_fp_le_11": csvted_thr,
            "optional_confusion": csvted_conf,
            "run_id": "${CSVTED_RUN_ID}",
            "baseline_anchor": baseline_csvted,
        },
    },
}
out = Path("${CALIB_DIR}/calibration.json")
out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
summary = Path("results/${RESULTS_TAG}/${EXP_NAME}_summary.json")
summary.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
print("wrote", summary)
PY

echo ""
echo "=== S1 done ==="
echo "  EXP_NAME:     $EXP_NAME"
echo "  dev ckpt:     $CKPT_DEV"
echo "  mvtb RUN_ID:  $MVTB_RUN_ID"
echo "  mvtb thr:     $MVTB_CAL_THR"
echo "  summary:      results/${RESULTS_TAG}/${EXP_NAME}_summary.json"
echo ""
echo "  baseline @0.5: mvtb TP63 FP51 | csvted TP24 FP11"
echo "  R5 dev ref:    mvtb TP96 FP19 @thr0.185 (for comparison only)"
echo "  models/test & R-line paths untouched"
