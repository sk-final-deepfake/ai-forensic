#!/usr/bin/env bash
# Rebuild / verify temporal-only train pool for TimeSformer v1.4 (no masking / spatial).
#
# Uses prepare_gmflow_temporal_dataset.py — same whitelist as GMFlow temporal lane:
#   MVTB fake: dropping, repetition, substitution (NOT masking)
#   CSVTED: frame-deletion/duplication/insertion, eop-* (NOT spatial-tampering)
#   Hold-out: MVTB200 + CSVTED200 manifests excluded from train
#
# Usage (GPU):
#   cd ~/forenShield-ai && source .venv/bin/activate
#   export FORENSHIELD_AI_ROOT=~/forenShield-ai/forgery
#   bash forgery/scripts/train/run_timesformer_forgery_v1.4_prepare_temporal_train.sh
set -eu

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai/forgery}"
REPO="${FORENSHIELD_REPO:-$HOME/forenShield-ai}"

TRAIN_STAGE="${TRAIN_STAGE:-$ROOT/data/train/video/forgery-gmflow-train-temporal}"
VAL_STAGE="${VAL_STAGE:-$ROOT/data/train/video/forgery-gmflow-val-temporal}"
MVTB_TRAIN="${MVTB_TRAIN:-$ROOT/data/train/video/forgery-gmflow-train-mvtb-1k}"
CSVTED_ROOT="${CSVTED_ROOT:-$ROOT/data/test/video/csvted}"
MVTB_TEST_MANIFEST="${MVTB_TEST_MANIFEST:-$ROOT/data/pull/evidence/mvtamperbench-200-s3/manifest.json}"
CSVTED_TEST_MANIFEST="${CSVTED_TEST_MANIFEST:-$ROOT/data/pull/evidence/csvted-200-balanced/manifest.json}"
VAL_RATIO="${VAL_RATIO:-0.15}"
SEED="${SEED:-123}"
FRESH="${FRESH:-}"

cd "$REPO"
source "$REPO/.venv/bin/activate"

PREP="$ROOT/scripts/data/prepare_gmflow_temporal_dataset.py"

FRESH_ARG=()
if [[ "$FRESH" == "1" ]]; then
  FRESH_ARG=(--fresh)
fi

echo "==> Build temporal-only train + val"
python3 "$PREP" train \
  --train-stage "$TRAIN_STAGE" \
  --val-stage "$VAL_STAGE" \
  --mvtb-train-dir "$MVTB_TRAIN" \
  --csvted-root "$CSVTED_ROOT" \
  --mvtb-test-manifest "$MVTB_TEST_MANIFEST" \
  --csvted-test-manifest "$CSVTED_TEST_MANIFEST" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED" \
  "${FRESH_ARG[@]}"

echo ""
echo "==> Stats"
python3 "$PREP" stats "$TRAIN_STAGE" "$VAL_STAGE"

echo ""
echo "==> Tamper types under train (expect NO masking, NO spatial-tampering)"
find "$TRAIN_STAGE/tampered" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort || true

echo ""
echo "Next:"
echo "  bash forgery/scripts/train/run_timesformer_forgery_v1.4_temporal_lane.sh"
