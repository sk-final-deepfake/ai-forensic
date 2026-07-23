#!/usr/bin/env bash
# TimeSformer CSVTED v1.8 — boost contrastive from v1.7-1347 (target AUC 0.70+).
#
# Prerequisite:
#   - forgery-csvted-train-temporal (group split)
#   - v1.6-rank CSVTED window NPZ cache
#   - v1.7-contrastive-1347 ckpt (or v1.6-1159 fallback)
#
# Usage (welabs):
#   cd ~/forenShield-ai && source .venv/bin/activate
#   export FORENSHIELD_AI_ROOT=~/forenShield-ai/forgery
#   bash forgery/scripts/train/run_timesformer_forgery_v1.8_csvted_boost.sh
set -eu

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai/forgery}"
REPO="${FORENSHIELD_REPO:-$HOME/forenShield-ai}"

V16_RUN="${V16_RUN:-timesformer-forgery-v1.6-rank-20260707-1159}"
INIT_HEAD="${INIT_HEAD:-$ROOT/models/train/temporal/timesformer-forgery/timesformer-forgery-v1.7-contrastive-20260707-1347/forgery_head.pt}"
FEATURE_CACHE="${FEATURE_CACHE:-$ROOT/results/train/${V16_RUN}/timesformer_window_bags_real_windows_initforgery_head_w1.0.npz}"
TEST_LANE="${TEST_LANE:-$ROOT/data/pull/evidence/csvted-200-temporal-only}"

# Stronger contrastive + real-FP suppression (tune on val inside trainer)
EPOCHS="${EPOCHS:-40}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.85}"
CONTRASTIVE_MARGIN="${CONTRASTIVE_MARGIN:-0.45}"
REAL_WINDOW_WEIGHT="${REAL_WINDOW_WEIGHT:-0.45}"
LR="${LR:-2e-4}"
GATE_AUC_MIN="${GATE_AUC_MIN:-0.70}"

cd "$REPO"
source "$REPO/.venv/bin/activate"
export FORENSHIELD_AI_ROOT="$ROOT"

if [[ ! -f "$FEATURE_CACHE" ]]; then
  echo "ERROR: missing v1.6 NPZ: $FEATURE_CACHE" >&2
  exit 1
fi
if [[ ! -f "$INIT_HEAD" ]]; then
  echo "ERROR: missing init head: $INIT_HEAD" >&2
  exit 1
fi

echo "==> v1.8 CSVTED boost (contrastive from v1.7)"
echo "    init=$INIT_HEAD"
echo "    cache=$FEATURE_CACHE"
echo "    test=$TEST_LANE"
echo "    ctr_w=$CONTRASTIVE_WEIGHT margin=$CONTRASTIVE_MARGIN real_w=$REAL_WINDOW_WEIGHT epochs=$EPOCHS"

INIT_HEAD="$INIT_HEAD" \
V16_RUN="$V16_RUN" \
FEATURE_CACHE="$FEATURE_CACHE" \
TEST_LANE="$TEST_LANE" \
EPOCHS="$EPOCHS" \
CONTRASTIVE_WEIGHT="$CONTRASTIVE_WEIGHT" \
CONTRASTIVE_MARGIN="$CONTRASTIVE_MARGIN" \
REAL_WINDOW_WEIGHT="$REAL_WINDOW_WEIGHT" \
LR="$LR" \
GATE_AUC_MIN="$GATE_AUC_MIN" \
RUN_ID="timesformer-forgery-v1.8-csvted-boost-$(date -u +%Y%m%d-%H%M)" \
bash "$ROOT/scripts/train/run_timesformer_forgery_v1.7_contrastive.sh"
