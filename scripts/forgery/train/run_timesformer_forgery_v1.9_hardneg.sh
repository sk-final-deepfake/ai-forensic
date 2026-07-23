#!/usr/bin/env bash
# TimeSformer forgery v1.9 — hard-negative retune on top of v1.8 CSVTED ckpt.
#
# Label policy:
#   fake = temporal edits only
#   real = authentic + deepfake + spatial/local (hard negatives under original/)
#
# Pipeline:
#   [0] (optional) prepare train tree
#   [1] window-MIL from INIT_HEAD (fresh embeddings on hardneg set)
#   [2] contrastive boost (real-FP vs fake-TP) on new feature cache
#   [3] eval on csvted-200-temporal-only + threshold sweep
#
# Usage (GPU / welabs):
#   cd ~/forenShield-ai && source .venv/bin/activate
#   export FORENSHIELD_AI_ROOT=~/forenShield-ai/forgery
#   unset RUN_ID
#
#   # 1) build dataset (edit HARDNEG paths to your deepfake/spatial pools)
#   bash forgery/scripts/train/run_timesformer_forgery_v1.9_hardneg.sh prepare
#
#   # 2) train + eval
#   bash forgery/scripts/train/run_timesformer_forgery_v1.9_hardneg.sh train
set -eu

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai/forgery}"
REPO="${FORENSHIELD_REPO:-$HOME/forenShield-ai}"
TS="$(date -u +%Y%m%d-%H%M)"
CMD="${1:-train}"

TRAIN_DATA="${TRAIN_DATA:-$ROOT/data/train/video/forgery-ts-hardneg-train}"
TEST_CSVTED_TEMP="${TEST_CSVTED_TEMP:-$ROOT/data/pull/evidence/csvted-200-temporal-only}"

INIT_HEAD="${INIT_HEAD:-$ROOT/models/train/temporal/timesformer-forgery/timesformer-forgery-v1.8-csvted-v4-20260708-0022/forgery_head.pt}"

# --- prepare defaults (override with env) ---
TEMPORAL_ROOT="${TEMPORAL_ROOT:-$ROOT/data/train/video/forgery-csvted-train-temporal}"
AUTH_DIR="${AUTH_DIR:-$ROOT/data/pull/evidence/csvted-200-temporal-only/original}"
# Deepfake / spatial hardneg roots — comma-separated paths
HARDNEG_DIRS="${HARDNEG_DIRS:-}"

MAX_TEMPORAL_FAKE="${MAX_TEMPORAL_FAKE:-200}"
MAX_AUTH="${MAX_AUTH:-80}"
MAX_HARDNEG="${MAX_HARDNEG:-120}"

RUN_ID="${RUN_ID:-timesformer-forgery-v1.9-hardneg-${TS}}"
OUT_DIR="${OUT_DIR:-$ROOT/models/train/temporal/timesformer-forgery/${RUN_ID}}"

EPOCHS_MIL="${EPOCHS_MIL:-20}"
EPOCHS_CTR="${EPOCHS_CTR:-20}"
WINDOW_LABEL_MODE="${WINDOW_LABEL_MODE:-real_windows}"
WINDOW_LOSS_WEIGHT="${WINDOW_LOSS_WEIGHT:-0.55}"
REAL_WINDOW_WEIGHT="${REAL_WINDOW_WEIGHT:-0.55}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.75}"
CONTRASTIVE_MARGIN="${CONTRASTIVE_MARGIN:-0.42}"
LR_MIL="${LR_MIL:-1e-4}"
LR_CTR="${LR_CTR:-5e-5}"
PRETRAINED="${PRETRAINED:-facebook/timesformer-base-finetuned-k400}"
GPU="${GPU:-0}"
DECODE_CACHE="${DECODE_CACHE:-$ROOT/cache/decode-mp4-temporal}"
GATE_AUC_MIN="${GATE_AUC_MIN:-0.68}"

cd "$REPO"
source "$REPO/.venv/bin/activate"
export FORENSHIELD_AI_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

PREPARE_PY="$ROOT/scripts/data/prepare_timesformer_hardneg_trainset.py"
TRAIN_MIL_PY="$ROOT/scripts/train/train_timesformer_forgery_window_mil.py"
MINE_PY="$ROOT/scripts/train/mine_timesformer_forgery_contrastive_pairs.py"
TRAIN_CTR_PY="$ROOT/scripts/train/train_timesformer_forgery_contrastive_mil.py"
BENCH_PY="$ROOT/scripts/infer/timesformer_forgery_benchmark.py"
SWEEP_PY="$ROOT/scripts/infer/sweep_timesformer_forgery_threshold.py"

do_prepare() {
  if [[ ! -f "$PREPARE_PY" ]]; then
    echo "ERROR: missing $PREPARE_PY (sync scripts to GPU forgery/ tree)" >&2
    exit 1
  fi
  if [[ -z "$HARDNEG_DIRS" ]]; then
    echo "ERROR: set HARDNEG_DIRS to deepfake/spatial pools, e.g.:" >&2
    echo "  export HARDNEG_DIRS=\"\$ROOT/data/test/video/spatial-videocof-benchmark/tampered,/path/to/deepfake\"" >&2
    exit 1
  fi

  # shellcheck disable=SC2206
  HN_ARR=(${HARDNEG_DIRS//,/ })
  echo "==> prepare hardneg train set -> $TRAIN_DATA"
  python3 "$PREPARE_PY" \
    --out-root "$TRAIN_DATA" \
    --temporal-root "$TEMPORAL_ROOT" \
    --auth-dirs "$AUTH_DIR" \
    --hardneg-dirs "${HN_ARR[@]}" \
    --max-temporal-fake "$MAX_TEMPORAL_FAKE" \
    --max-auth "$MAX_AUTH" \
    --max-hardneg "$MAX_HARDNEG" \
    --mode symlink \
    --fresh \
    --seed 43
}

do_train() {
  if [[ ! -f "$INIT_HEAD" ]]; then
    echo "ERROR: INIT_HEAD not found: $INIT_HEAD" >&2
    exit 1
  fi
  if [[ ! -d "$TRAIN_DATA/original" || ! -d "$TRAIN_DATA/tampered" ]]; then
    echo "ERROR: train data missing under $TRAIN_DATA — run: $0 prepare" >&2
    exit 1
  fi

  echo "==> v1.9 hardneg window-MIL"
  echo "    RUN_ID=$RUN_ID"
  echo "    init=$INIT_HEAD"
  echo "    train=$TRAIN_DATA"
  echo "    policy: original/=real(auth+hardneg) tampered/=temporal-fake"

  python3 "$TRAIN_MIL_PY" \
    --root "$REPO" \
    --data-root "$TRAIN_DATA" \
    --run-id "$RUN_ID-mil" \
    --init-head "$INIT_HEAD" \
    --pretrained-id "$PRETRAINED" \
    --scan-mode window_1s \
    --window-label-mode "$WINDOW_LABEL_MODE" \
    --window-loss-weight "$WINDOW_LOSS_WEIGHT" \
    --window-sec 1.0 \
    --stride-sec 0.5 \
    --clip-frames 8 \
    --max-side 512 \
    --aggregate topk \
    --top-k 3 \
    --epochs "$EPOCHS_MIL" \
    --lr "$LR_MIL" \
    --val-ratio 0.15 \
    --out-dir "${OUT_DIR}-mil" \
    --decode-cache "$DECODE_CACHE" \
    --gpu "$GPU"

  MIL_CKPT="${OUT_DIR}-mil/forgery_head.pt"
  FEATURE_CACHE="$(ls -1t "$ROOT/results/train/${RUN_ID}-mil"/timesformer_window_bags_*.npz 2>/dev/null | head -1 || true)"
  if [[ ! -f "$MIL_CKPT" ]]; then
    echo "ERROR: MIL ckpt missing: $MIL_CKPT" >&2
    exit 1
  fi
  if [[ -z "$FEATURE_CACHE" || ! -f "$FEATURE_CACHE" ]]; then
    echo "ERROR: feature cache not found under results/train/${RUN_ID}-mil/" >&2
    exit 1
  fi

  echo "==> v1.9 contrastive boost"
  MINE_RUN="${RUN_ID}-mine"
  python3 "$MINE_PY" \
    --root "$REPO" \
    --feature-cache "$FEATURE_CACHE" \
    --init-head "$MIL_CKPT" \
    --run-id "$MINE_RUN" \
    --gpu "$GPU"

  PAIR_CACHE="$ROOT/results/train/${MINE_RUN}/contrastive_pairs.npz"
  python3 "$TRAIN_CTR_PY" \
    --root "$REPO" \
    --feature-cache "$FEATURE_CACHE" \
    --init-head "$MIL_CKPT" \
    --pair-cache "$PAIR_CACHE" \
    --run-id "$RUN_ID" \
    --out-dir "$OUT_DIR" \
    --epochs "$EPOCHS_CTR" \
    --lr "$LR_CTR" \
    --contrastive-weight "$CONTRASTIVE_WEIGHT" \
    --contrastive-margin "$CONTRASTIVE_MARGIN" \
    --real-window-weight "$REAL_WINDOW_WEIGHT" \
    --gpu "$GPU"

  CKPT="$OUT_DIR/forgery_head.pt"
  if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: final ckpt missing: $CKPT" >&2
    exit 1
  fi

  echo "==> eval csvted temporal-only"
  EVAL_ID="${RUN_ID}-csvted-max"
  python3 "$BENCH_PY" \
    --root "$REPO" \
    --checkpoint "$CKPT" \
    --data-root "$TEST_CSVTED_TEMP" \
    --run-id "$EVAL_ID" \
    --aggregate max \
    --threshold 0.12 \
    --decode-cache "$DECODE_CACHE" \
    --gpu "$GPU"

  python3 "$SWEEP_PY" \
    --run-id "$EVAL_ID" \
    --forgery-root "$ROOT" \
    --write-metrics

  echo ""
  echo "DONE checkpoint: $CKPT"
  echo "metrics: $ROOT/results/eval/${EVAL_ID}/metrics.json"
  echo "sweep:   $ROOT/results/eval/${EVAL_ID}/metrics_threshold_sweep.json"
  echo "gate: CSVTED temporal AUC >= $GATE_AUC_MIN (v1.8 was ~0.70)"
  jq '{roc_auc, accuracy, confusion, real, fake}' "$ROOT/results/eval/${EVAL_ID}/metrics.json" || true
}

case "$CMD" in
  prepare) do_prepare ;;
  train) do_train ;;
  all)
    do_prepare
    do_train
    ;;
  *)
    echo "usage: $0 {prepare|train|all}" >&2
    exit 1
    ;;
esac
