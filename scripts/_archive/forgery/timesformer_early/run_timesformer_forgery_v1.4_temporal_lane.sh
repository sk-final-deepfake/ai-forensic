#!/usr/bin/env bash
# TimeSformer forgery v1.4 — temporal-only lane (spatial/masking excluded).
#
# Lane design:
#   Train : forgery-gmflow-train-temporal (~633, dropping/repetition/substitution + CSVTED temporal)
#   Gate  : gmflow-test-temporal-200 ONLY (AUC > 0.55)
#   Fusion: MVTB200 / CSVTED200 optional (RUN_FUSION_EVAL=1) — system sign-off, NOT lane KPI
#
# Excludes from train: MVTB masking, CSVTED spatial-tampering (see prepare_gmflow_temporal_dataset.py)
#
# Usage (GPU):
#   cd ~/forenShield-ai && source .venv/bin/activate
#   export FORENSHIELD_AI_ROOT=~/forenShield-ai/forgery
#   bash forgery/scripts/train/run_timesformer_forgery_v1.4_temporal_lane.sh
#
# Optional FT (temporal train + v2 init):
#   MODE=ft INIT_HEAD=$FORENSHIELD_AI_ROOT/models/train/temporal/timesformer-forgery/\
timesformer-forgery-v2-ft-last1-20260704-0725/forgery_head.pt \
#   bash forgery/scripts/train/run_timesformer_forgery_v1.4_temporal_lane.sh
#
# Rebuild expanded temporal train pool (if 633 too small):
#   python3 forgery/scripts/data/prepare_gmflow_temporal_dataset.py train --fresh ...
set -eu

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai/forgery}"
REPO="${FORENSHIELD_REPO:-$HOME/forenShield-ai}"
TS="$(date -u +%Y%m%d-%H%M)"
MODE="${MODE:-clip}"
RUN_ID="${RUN_ID:-timesformer-forgery-v1.4-temporal-${MODE}-${TS}}"

TRAIN_DATA="${TRAIN_DATA:-$ROOT/data/train/video/forgery-gmflow-train-temporal}"
TEST_LANE="${TEST_LANE:-$ROOT/data/pull/evidence/gmflow-test-temporal-200}"
TEST_MIXED_MVTB="${TEST_MIXED_MVTB:-$ROOT/data/pull/evidence/mvtamperbench-200-s3}"
TEST_MIXED_CSVTED="${TEST_MIXED_CSVTED:-$ROOT/data/pull/evidence/csvted-200-balanced}"

OUT_DIR="${OUT_DIR:-$ROOT/models/train/temporal/timesformer-forgery/${RUN_ID}}"
DECODE_CACHE="${DECODE_CACHE:-$ROOT/cache/decode-mp4-temporal}"

# clip (frozen K400 head)
EPOCHS="${EPOCHS:-40}"
CLIP_FRAMES="${CLIP_FRAMES:-8}"
POS_WEIGHT="${POS_WEIGHT:-1.0}"
LR="${LR:-1e-3}"

# ft (optional MODE=ft)
INIT_HEAD="${INIT_HEAD:-$ROOT/models/train/temporal/timesformer-forgery/timesformer-forgery-v2-ft-last1-20260704-0725/forgery_head.pt}"
UNFREEZE_LAYERS="${UNFREEZE_LAYERS:-1}"
FT_EPOCHS="${FT_EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-2}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
HEAD_LR="${HEAD_LR:-1e-4}"

PRETRAINED="${PRETRAINED:-facebook/timesformer-base-finetuned-k400}"
GPU="${GPU:-0}"
RUN_FUSION_EVAL="${RUN_FUSION_EVAL:-0}"
GATE_AUC_MIN="${GATE_AUC_MIN:-0.55}"

cd "$REPO"
source "$REPO/.venv/bin/activate"
export FORENSHIELD_AI_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

TRAIN_CLIP_PY="$ROOT/scripts/train/train_timesformer_forgery_clip_mil.py"
TRAIN_FT_PY="$ROOT/scripts/train/train_timesformer_forgery_clip_ft.py"
BENCH_PY="$ROOT/scripts/infer/timesformer_forgery_clip_benchmark.py"
SWEEP_PY="$ROOT/scripts/infer/sweep_timesformer_forgery_threshold.py"
PREP_PY="$ROOT/scripts/data/prepare_gmflow_temporal_dataset.py"

echo "==> v1.4 temporal lane  MODE=$MODE  RUN_ID=$RUN_ID"
echo "    train: $TRAIN_DATA"
echo "    lane gate eval: $TEST_LANE (AUC >= $GATE_AUC_MIN)"
echo "    fusion eval: RUN_FUSION_EVAL=$RUN_FUSION_EVAL"

if [[ ! -d "$TRAIN_DATA/original" ]] || [[ ! -d "$TRAIN_DATA/tampered" ]]; then
  echo "ERROR: temporal train not found: $TRAIN_DATA" >&2
  echo "  build with: python3 $PREP_PY train --fresh ..." >&2
  exit 1
fi

if find "$TRAIN_DATA/tampered" -mindepth 1 -maxdepth 1 -type d -iname 'masking' 2>/dev/null | grep -q .; then
  echo "WARN: tampered/masking found under temporal train — lane purity violated" >&2
fi

N_TRAIN=$(find "$TRAIN_DATA" -type f \( -name '*.mp4' -o -name '*.webm' \) | wc -l)
echo "    train clips: $N_TRAIN"

if [[ "$MODE" == "clip" ]]; then
  echo ""
  echo "==> [1] Train clip head (frozen K400, temporal-only)"
  python3 "$TRAIN_CLIP_PY" \
    --root "$REPO" \
    --data-root "$TRAIN_DATA" \
    --run-id "$RUN_ID" \
    --pretrained-id "$PRETRAINED" \
    --clip-frames "$CLIP_FRAMES" \
    --pos-weight "$POS_WEIGHT" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --out-dir "$OUT_DIR" \
    --decode-cache "$DECODE_CACHE" \
    --gpu "$GPU"
elif [[ "$MODE" == "ft" ]]; then
  if [[ ! -f "$INIT_HEAD" ]]; then
    echo "ERROR: INIT_HEAD not found: $INIT_HEAD" >&2
    exit 1
  fi
  echo ""
  echo "==> [1] FT last-${UNFREEZE_LAYERS} layer(s) on temporal-only (init: $INIT_HEAD)"
  python3 "$TRAIN_FT_PY" \
    --root "$REPO" \
    --data-root "$TRAIN_DATA" \
    --init-head "$INIT_HEAD" \
    --run-id "$RUN_ID" \
    --pretrained-id "$PRETRAINED" \
    --unfreeze-layers "$UNFREEZE_LAYERS" \
    --clip-frames "$CLIP_FRAMES" \
    --epochs "$FT_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --backbone-lr "$BACKBONE_LR" \
    --head-lr "$HEAD_LR" \
    --out-dir "$OUT_DIR" \
    --decode-cache "$DECODE_CACHE" \
    --gpu "$GPU"
else
  echo "ERROR: MODE must be clip or ft (got: $MODE)" >&2
  exit 1
fi

CKPT="$OUT_DIR/forgery_head.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

cat "$OUT_DIR/train_summary.json"

COMMON_BENCH=(
  --root "$REPO"
  --checkpoint "$CKPT"
  --pretrained-id "$PRETRAINED"
  --clip-frames "$CLIP_FRAMES"
  --decode-cache "$DECODE_CACHE"
  --gpu "$GPU"
)

run_eval_sweep() {
  tag="$1"
  data_root="$2"
  label="$3"
  eval_run_id="${RUN_ID}-${tag}"
  echo ""
  echo "==> Eval ${tag} (${label})"
  python3 "$BENCH_PY" \
    "${COMMON_BENCH[@]}" \
    --data-root "$data_root" \
    --run-id "$eval_run_id"
  python3 "$SWEEP_PY" \
    --run-id "$eval_run_id" \
    --forgery-root "$ROOT" \
    --write-metrics
  jq '{roc_auc, accuracy: ."metrics_at_0.5".accuracy, youden: .best_youden_j | {threshold, accuracy, recall}}' \
    "$ROOT/results/eval/${eval_run_id}/metrics_threshold_sweep.json"
}

run_eval_sweep "temporal200" "$TEST_LANE" "LANE GATE"

LANE_AUC=$(jq -r '.roc_auc // empty' "$ROOT/results/eval/${RUN_ID}-temporal200/metrics.json")
if [[ -n "$LANE_AUC" ]]; then
  echo ""
  echo "==> Lane gate: temporal200 AUC=$LANE_AUC (min $GATE_AUC_MIN)"
  python3 - "$LANE_AUC" "$GATE_AUC_MIN" <<'PY'
import sys
auc, gate = float(sys.argv[1]), float(sys.argv[2])
print("PASS" if auc >= gate else "FAIL")
PY
fi

if [[ "$RUN_FUSION_EVAL" == "1" ]]; then
  run_eval_sweep "mvtb200" "$TEST_MIXED_MVTB" "fusion sign-off (spatial mixed)"
  run_eval_sweep "csvted200" "$TEST_MIXED_CSVTED" "fusion sign-off (OOD mixed)"
else
  echo ""
  echo "==> Skipping MVTB/CSVTED (set RUN_FUSION_EVAL=1 for system sign-off)"
fi

echo ""
echo "DONE"
echo "  checkpoint: $CKPT"
echo "  compare lane baselines:"
echo "    v2 FT temporal200: timesformer-forgery-v2-ft-last1-20260704-0725-temporal200"
echo "    v1.3 temporal200:  timesformer-forgery-v1.3-mixed1k-clip-20260704-0710-temporal200"
