#!/usr/bin/env bash
# TimeSformer forgery v1.7 — real-FP vs fake-TP contrastive window-MIL.
#
# Design:
#   1) Reuse v1.6 window embedding cache (no 633-clip re-extract)
#   2) Mine real_fp (high-score real windows) vs fake_tp (tamper-prior / high-score fake windows)
#   3) Train: bag_BCE + real_window_BCE(0) + margin(logit_fake_tp - logit_real_fp)
#
# Usage (GPU):
#   cd ~/forenShield-ai && source .venv/bin/activate
#   export FORENSHIELD_AI_ROOT=~/forenShield-ai/forgery
#   unset RUN_ID
#   bash forgery/scripts/train/run_timesformer_forgery_v1.7_contrastive.sh
set -eu

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai/forgery}"
REPO="${FORENSHIELD_REPO:-$HOME/forenShield-ai}"
TS="$(date -u +%Y%m%d-%H%M)"
RUN_ID="${RUN_ID:-timesformer-forgery-v1.7-contrastive-${TS}}"

INIT_HEAD="${INIT_HEAD:-$ROOT/models/train/temporal/timesformer-forgery/timesformer-forgery-v1.4-temporal-ft-20260704-0837/forgery_head.pt}"
V16_RUN="${V16_RUN:-timesformer-forgery-v1.6-rank-20260707-0437}"
FEATURE_CACHE="${FEATURE_CACHE:-$ROOT/results/train/${V16_RUN}/timesformer_window_bags_real_windows_initforgery_head_w1.0.npz}"

TEST_LANE="${TEST_LANE:-$ROOT/data/pull/evidence/gmflow-test-temporal-200}"
OUT_DIR="${OUT_DIR:-$ROOT/models/train/temporal/timesformer-forgery/${RUN_ID}}"

EPOCHS="${EPOCHS:-25}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.5}"
CONTRASTIVE_MARGIN="${CONTRASTIVE_MARGIN:-0.35}"
REAL_WINDOW_WEIGHT="${REAL_WINDOW_WEIGHT:-0.2}"
LR="${LR:-3e-4}"
GPU="${GPU:-0}"
GATE_AUC_MIN="${GATE_AUC_MIN:-0.60}"

cd "$REPO"
source "$REPO/.venv/bin/activate"
export FORENSHIELD_AI_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

MINE_PY="$ROOT/scripts/train/mine_timesformer_forgery_contrastive_pairs.py"
TRAIN_PY="$ROOT/scripts/train/train_timesformer_forgery_contrastive_mil.py"
BENCH_PY="$ROOT/scripts/infer/timesformer_forgery_benchmark.py"
SWEEP_PY="$ROOT/scripts/infer/sweep_timesformer_forgery_threshold.py"

if [[ ! -f "$FEATURE_CACHE" ]]; then
  echo "ERROR: feature cache not found: $FEATURE_CACHE" >&2
  echo "  set FEATURE_CACHE or V16_RUN to v1.6 run with window npz" >&2
  exit 1
fi

if [[ ! -f "$INIT_HEAD" ]]; then
  echo "ERROR: INIT_HEAD not found: $INIT_HEAD" >&2
  exit 1
fi

echo "==> v1.7 contrastive (real-FP vs fake-TP)"
echo "    RUN_ID=$RUN_ID"
echo "    cache=$FEATURE_CACHE"
echo "    init=$INIT_HEAD"
echo "    margin=$CONTRASTIVE_MARGIN weight=$CONTRASTIVE_WEIGHT"

echo ""
echo "==> [1] Mine contrastive pairs (analysis)"
MINE_RUN="timesformer-contrastive-mine-${TS}"
python3 "$MINE_PY" \
  --root "$REPO" \
  --feature-cache "$FEATURE_CACHE" \
  --init-head "$INIT_HEAD" \
  --run-id "$MINE_RUN" \
  --gpu "$GPU"

PAIR_CACHE="$ROOT/results/train/${MINE_RUN}/contrastive_pairs.npz"
jq '{n_pairs, score_gap_mean, pct_pairs_fake_higher}' "$ROOT/results/train/${MINE_RUN}/mine_summary.json"

echo ""
echo "==> [2] Contrastive MIL train"
python3 "$TRAIN_PY" \
  --root "$REPO" \
  --feature-cache "$FEATURE_CACHE" \
  --init-head "$INIT_HEAD" \
  --pair-cache "$PAIR_CACHE" \
  --run-id "$RUN_ID" \
  --out-dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --contrastive-weight "$CONTRASTIVE_WEIGHT" \
  --contrastive-margin "$CONTRASTIVE_MARGIN" \
  --real-window-weight "$REAL_WINDOW_WEIGHT" \
  --gpu "$GPU"

CKPT="$OUT_DIR/forgery_head.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint missing: $CKPT" >&2
  exit 1
fi

echo ""
echo "==> [3] Eval temporal200"
EVAL_ID="${RUN_ID}-temporal200"
python3 "$BENCH_PY" \
  --root "$REPO" \
  --checkpoint "$CKPT" \
  --data-root "$TEST_LANE" \
  --run-id "$EVAL_ID" \
  --gpu "$GPU"

python3 "$SWEEP_PY" \
  --run-id "$EVAL_ID" \
  --forgery-root "$ROOT" \
  --write-metrics

LANE_AUC=$(jq -r '.roc_auc // empty' "$ROOT/results/eval/${EVAL_ID}/metrics.json")
echo ""
echo "==> Lane gate: temporal200 AUC=${LANE_AUC:-?} (min $GATE_AUC_MIN, v1.4=0.601, v1.6=0.594)"
echo "DONE checkpoint: $CKPT"
