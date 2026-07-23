#!/usr/bin/env bash
# TimeSformer forgery v1.6 — ranking-focused window-MIL (train = infer).
#
# Goals:
#   - Init MIL head + FT backbone from v1.4 temporal FT (not scratch)
#   - real_windows label mode: suppress shake/noise on real clips at window level
#   - Fake clips: clip bag BCE only (avoid weak segment priors)
#   - BCEWithLogitsLoss + top-k bag aggregation
#
# Usage (GPU):
#   cd ~/forenShield-ai && source .venv/bin/activate
#   export FORENSHIELD_AI_ROOT=~/forenShield-ai/forgery
#   unset RUN_ID
#   bash forgery/scripts/train/run_timesformer_forgery_v1.6_rank.sh
#
# tmux (recommended):
#   tmux new -s ts-v16 -d "cd ~/forenShield-ai && source .venv/bin/activate && \
#     export FORENSHIELD_AI_ROOT=~/forenShield-ai/forgery && unset RUN_ID && \
#     bash forgery/scripts/train/run_timesformer_forgery_v1.6_rank.sh 2>&1 | tee /tmp/ts-v16-rank.log"
set -eu

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai/forgery}"
REPO="${FORENSHIELD_REPO:-$HOME/forenShield-ai}"
TS="$(date -u +%Y%m%d-%H%M)"
RUN_ID="${RUN_ID:-timesformer-forgery-v1.6-rank-${TS}}"

TRAIN_DATA="${TRAIN_DATA:-$ROOT/data/train/video/forgery-gmflow-train-temporal}"
TEST_LANE="${TEST_LANE:-$ROOT/data/pull/evidence/gmflow-test-temporal-200}"
TEST_MVTB_TEMP="${TEST_MVTB_TEMP:-$ROOT/data/pull/evidence/mvtamperbench-200-temporal-only}"
TEST_CSVTED_TEMP="${TEST_CSVTED_TEMP:-$ROOT/data/pull/evidence/csvted-200-temporal-only}"

INIT_HEAD="${INIT_HEAD:-$ROOT/models/train/temporal/timesformer-forgery/timesformer-forgery-v1.4-temporal-ft-20260704-0837/forgery_head.pt}"
OUT_DIR="${OUT_DIR:-$ROOT/models/train/temporal/timesformer-forgery/${RUN_ID}}"

EPOCHS="${EPOCHS:-30}"
CLIP_FRAMES="${CLIP_FRAMES:-8}"
MAX_SIDE="${MAX_SIDE:-512}"
TOP_K="${TOP_K:-3}"
WINDOW_SEC="${WINDOW_SEC:-1.0}"
STRIDE_SEC="${STRIDE_SEC:-0.5}"
WINDOW_LABEL_MODE="${WINDOW_LABEL_MODE:-real_windows}"
WINDOW_LOSS_WEIGHT="${WINDOW_LOSS_WEIGHT:-0.35}"
WINDOW_POS_WEIGHT="${WINDOW_POS_WEIGHT:-4.0}"
LR="${LR:-5e-4}"
PRETRAINED="${PRETRAINED:-facebook/timesformer-base-finetuned-k400}"
GPU="${GPU:-0}"
DECODE_CACHE="${DECODE_CACHE:-$ROOT/cache/decode-mp4-temporal}"
GATE_AUC_MIN="${GATE_AUC_MIN:-0.60}"

cd "$REPO"
source "$REPO/.venv/bin/activate"
export FORENSHIELD_AI_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

TRAIN_PY="$ROOT/scripts/train/train_timesformer_forgery_window_mil.py"
BENCH_PY="$ROOT/scripts/infer/timesformer_forgery_benchmark.py"
SWEEP_PY="$ROOT/scripts/infer/sweep_timesformer_forgery_threshold.py"

if [[ ! -f "$INIT_HEAD" ]]; then
  echo "ERROR: INIT_HEAD not found: $INIT_HEAD" >&2
  exit 1
fi

if [[ ! -d "$TRAIN_DATA/original" ]]; then
  echo "ERROR: train data missing: $TRAIN_DATA" >&2
  exit 1
fi

echo "==> v1.6 rank window-MIL"
echo "    RUN_ID=$RUN_ID"
echo "    init=$INIT_HEAD"
echo "    label_mode=$WINDOW_LABEL_MODE window_loss=$WINDOW_LOSS_WEIGHT"
echo "    train=$TRAIN_DATA ($(find "$TRAIN_DATA" -type f \( -name '*.mp4' -o -name '*.webm' -o -name '*.avi' -o -name '*.mov' \) | wc -l) clips)"
echo "    gate: temporal200 AUC >= $GATE_AUC_MIN"

python3 "$TRAIN_PY" \
  --root "$REPO" \
  --data-root "$TRAIN_DATA" \
  --run-id "$RUN_ID" \
  --init-head "$INIT_HEAD" \
  --pretrained-id "$PRETRAINED" \
  --scan-mode window_1s \
  --window-label-mode "$WINDOW_LABEL_MODE" \
  --window-loss-weight "$WINDOW_LOSS_WEIGHT" \
  --window-pos-weight "$WINDOW_POS_WEIGHT" \
  --window-sec "$WINDOW_SEC" \
  --stride-sec "$STRIDE_SEC" \
  --clip-frames "$CLIP_FRAMES" \
  --max-side "$MAX_SIDE" \
  --aggregate topk \
  --top-k "$TOP_K" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --val-ratio 0.15 \
  --out-dir "$OUT_DIR" \
  --decode-cache "$DECODE_CACHE" \
  --gpu "$GPU"

CKPT="$OUT_DIR/forgery_head.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

COMMON_BENCH=(
  --root "$REPO"
  --checkpoint "$CKPT"
  --aggregate topk
  --top-k "$TOP_K"
  --window-sec "$WINDOW_SEC"
  --stride-sec "$STRIDE_SEC"
  --clip-frames "$CLIP_FRAMES"
  --max-side "$MAX_SIDE"
  --decode-cache "$DECODE_CACHE"
  --gpu "$GPU"
)

run_eval() {
  tag="$1"
  data_root="$2"
  if [[ ! -d "$data_root/original" ]]; then
    echo "SKIP: $data_root"
    return 0
  fi
  eval_id="${RUN_ID}-${tag}"
  echo ""
  echo "==> Eval $tag ($data_root)"
  python3 "$BENCH_PY" \
    "${COMMON_BENCH[@]}" \
    --data-root "$data_root" \
    --run-id "$eval_id"
  python3 "$SWEEP_PY" \
    --run-id "$eval_id" \
    --forgery-root "$ROOT" \
    --write-metrics
  jq '{roc_auc, real_avg: .real.avg_tamper_score, fake_avg: .fake.avg_tamper_score, youden: .best_youden_j}' \
    "$ROOT/results/eval/${eval_id}/metrics_threshold_sweep.json" 2>/dev/null \
    || jq '{roc_auc, real: .real.avg_tamper_score, fake: .fake.avg_tamper_score}' \
    "$ROOT/results/eval/${eval_id}/metrics.json"
}

run_eval "temporal200" "$TEST_LANE"
run_eval "mvtb200-temporal" "$TEST_MVTB_TEMP"
run_eval "csvted200-temporal" "$TEST_CSVTED_TEMP"

LANE_AUC=$(jq -r '.roc_auc // empty' "$ROOT/results/eval/${RUN_ID}-temporal200/metrics.json" 2>/dev/null || true)
echo ""
echo "==> Lane gate: temporal200 AUC=${LANE_AUC:-?} (min $GATE_AUC_MIN, v1.4 baseline 0.601)"
echo "DONE checkpoint: $CKPT"
