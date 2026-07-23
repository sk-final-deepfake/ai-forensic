#!/usr/bin/env bash
# TimeSformer 1k fine-tune: ff1k (FF++ fake + Vox real) → celeb1k (Celeb-DF).
#
# Reuses Xception 1k train manifests/clips:
#   data/pull/train/video/xception/{ff1k,celeb1k}/manifest.json
#
# Usage (GPU) — tomorrow one-liner:
#   cd ~/forenShield-ai && source .venv/bin/activate && unset AWS_PROFILE
#   sed -i 's/\r$//' scripts/infer/run_timesformer_finetune_1k.sh
#   bash scripts/infer/run_timesformer_finetune_1k.sh
#
# Optional env:
#   SKIP_FF1K=1 | SKIP_CELEB1K=1
#   FINETUNE_FRESH=1        # --fresh for ff1k
#   CELEB_FRESH=1           # --fresh for celeb1k (default 1)
#   VAL_HOLDOUT=120         # 7:3 split when re-preparing Xception data
#   INIT_WEIGHTS=...        # ff1k init (default: timesformer_finetuned.pth if exists)
#   MAX_PER_CLASS=200
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "${ROOT}"

SKIP_FF1K="${SKIP_FF1K:-0}"
SKIP_CELEB1K="${SKIP_CELEB1K:-0}"
CELEB_FRESH="${CELEB_FRESH:-1}"
VAL_HOLDOUT="${VAL_HOLDOUT:-120}"

WEIGHTS_ROOT="${TIMESFORMER_WEIGHTS_DIR:-models/test/video/timesformer/v1.0.0}"
BASELINE="${INIT_WEIGHTS:-${WEIGHTS_ROOT}/timesformer_finetuned.pth}"
FF1K_OUT="${FF1K_OUT:-${WEIGHTS_ROOT}/timesformer_finetuned_ff1k.pth}"
CELEB1K_OUT="${CELEB1K_OUT:-${WEIGHTS_ROOT}/timesformer_finetuned_celeb1k.pth}"

FT_COMMON=(--model timesformer --root "${ROOT}" --val-holdout "${VAL_HOLDOUT}")
if [[ -n "${MAX_PER_CLASS:-}" ]]; then
  FT_COMMON+=(--max-per-class "${MAX_PER_CLASS}")
fi

FF_INIT=()
if [[ -f "${ROOT}/${BASELINE}" || -f "${ROOT}/deepfake/${BASELINE}" ]]; then
  FF_INIT=(--init-weights "${BASELINE}")
else
  echo "WARN: baseline weights not found (${BASELINE}); ff1k starts from K400 pretrained head only"
fi

FF_EXTRA=()
CELEB_EXTRA=()
if [[ "${FINETUNE_FRESH:-0}" == "1" ]]; then
  FF_EXTRA+=(--fresh)
fi
if [[ "${CELEB_FRESH}" == "1" ]]; then
  CELEB_EXTRA+=(--fresh)
fi

echo "=== TimeSformer 1k fine-tune ==="
echo "  train manifests: data/pull/train/video/xception/{ff1k,celeb1k}"
echo "  val_holdout:     ${VAL_HOLDOUT}"
echo "  ff1k out:        ${FF1K_OUT}"
echo "  celeb1k out:     ${CELEB1K_OUT}"
echo ""

if [[ "${SKIP_FF1K}" != "1" ]]; then
  echo "=== ff1k (backbone frozen, manifest train/val) ==="
  python3 scripts/infer/video_transformer_finetune.py \
    "${FT_COMMON[@]}" \
    --stage ff1k \
    --output "${FF1K_OUT}" \
    --train-pull-dir data/pull/train/video/xception/ff1k \
    "${FF_INIT[@]}" \
    "${FF_EXTRA[@]}"
fi

if [[ "${SKIP_CELEB1K}" != "1" ]]; then
  echo ""
  echo "=== celeb1k (partial backbone unfreeze, init from ff1k) ==="
  python3 scripts/infer/video_transformer_finetune.py \
    "${FT_COMMON[@]}" \
    --stage celeb1k \
    --output "${CELEB1K_OUT}" \
    --train-pull-dir data/pull/train/video/xception/celeb1k \
    --init-weights "${FF1K_OUT}" \
    --unfreeze-backbone \
    "${CELEB_EXTRA[@]}"
fi

echo ""
echo "Done."
echo "  ff1k:    ${FF1K_OUT}"
echo "  celeb1k: ${CELEB1K_OUT}"
echo "  checkpoints: checkpoints/video/timesformer/timesformer_finetuned_ff1k/"
echo "               checkpoints/video/timesformer/timesformer_finetuned_celeb1k/"
echo "  epochs cap: ff1k≤30 celeb1k≤30 (early-stop on val AUC)"
echo ""
echo "Golden-200 eval:"
echo "  python3 scripts/eval/run_timesformer_1k_finetune_rebenchmark.py"
