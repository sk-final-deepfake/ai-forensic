#!/usr/bin/env bash
# Xception 1k fine-tune: FF++ fake + Vox real (prior 100-clip train excluded).
#
# ff1k real pools (merged, deduped):
#   data/raw/voxceleb/tmp_full, data/raw/voxceleb, data/train/video/voxceleb/real
# GPU often has ~166-266 usable real clips (not 1000). Script warns and uses max available.
# Optional: MAX_PER_CLASS=200 to balance fake/real counts.
#
# Usage (GPU):
#   cd ~/forenShield-ai && source .venv/bin/activate && unset AWS_PROFILE
#   sed -i 's/\r$//' scripts/infer/run_xception_finetune_1k.sh
#   bash scripts/infer/run_xception_finetune_1k.sh
#
# Optional env:
#   SKIP_PREPARE=1          # use existing data/pull/train/video/xception/{ff1k,celeb1k}
#   UPLOAD_S3=1             # upload prepare output to S3
#   SKIP_S3_PULL=1          # default 1; set 0 to pull from S3 instead of local prepare
#   SKIP_FF1K=1 | SKIP_CELEB1K=1
#   FINETUNE_FRESH=1        # ignore savepoint for ff1k
#   CELEB_FRESH=1           # --fresh for celeb1k stage (default 1)
#   MAX_PER_CLASS=200       # cap fake/real per class (recommended on GPU)
#   TRAIN_FAKE_POOL=...     # ff1k FF++ fake override
#   TRAIN_REAL_DIR=...      # ff1k Vox real override
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "${ROOT}"

SKIP_PREPARE="${SKIP_PREPARE:-0}"
UPLOAD_S3="${UPLOAD_S3:-0}"
SKIP_S3_PULL="${SKIP_S3_PULL:-1}"
SKIP_FF1K="${SKIP_FF1K:-0}"
SKIP_CELEB1K="${SKIP_CELEB1K:-0}"
INIT_WEIGHTS="${INIT_WEIGHTS:-models/test/video/xception/v1.0.0/xception_best.pth}"
FF1K_OUT="${FF1K_OUT:-models/test/video/xception/v1.0.0/xception_finetuned_ff1k.pth}"
CELEB1K_OUT="${CELEB1K_OUT:-models/test/video/xception/v1.0.0/xception_finetuned_celeb1k.pth}"
CELEB_FRESH="${CELEB_FRESH:-1}"
MAX_PER_CLASS="${MAX_PER_CLASS:-}"

PREPARE_EXTRA=(--exclude-prior-train)
FT_MAX=()
if [[ -n "${MAX_PER_CLASS}" ]]; then
  PREPARE_EXTRA+=(--max-per-class "${MAX_PER_CLASS}")
  FT_MAX=(--max-per-class "${MAX_PER_CLASS}")
fi
if [[ "${UPLOAD_S3}" == "1" ]]; then
  PREPARE_EXTRA+=(--upload-s3)
fi
if [[ -n "${TRAIN_FAKE_POOL:-}" ]]; then
  PREPARE_EXTRA+=(--fake-dir "${TRAIN_FAKE_POOL}")
fi
if [[ -n "${TRAIN_REAL_DIR:-}" ]]; then
  PREPARE_EXTRA+=(--real-dir "${TRAIN_REAL_DIR}")
fi

FT_FF_EXTRA=(--exclude-prior-train)
FT_CELEB_EXTRA=(--exclude-prior-train)
if [[ "${FINETUNE_FRESH:-0}" == "1" ]]; then
  FT_FF_EXTRA+=(--fresh)
fi
if [[ "${CELEB_FRESH}" == "1" ]]; then
  FT_CELEB_EXTRA+=(--fresh)
fi

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  echo "=== Prepare ff1k (FF++ fake + Vox real, exclude golden + prior train) ==="
  python3 scripts/download/data/prepare_xception_finetune_train.py \
    --root "${ROOT}" --stage ff1k --write-docs "${PREPARE_EXTRA[@]}"

  echo ""
  echo "=== Prepare celeb1k (Celeb-DF 1000 each, exclude golden + prior + ff1k) ==="
  python3 scripts/download/data/prepare_xception_finetune_train.py \
    --root "${ROOT}" --stage celeb1k --write-docs "${PREPARE_EXTRA[@]}"
fi

if [[ "${SKIP_S3_PULL}" != "1" ]]; then
  echo "=== S3 pull 1k train clips ==="
  python3 scripts/download/data/s3_pull_xception_finetune_train.py --root "${ROOT}" --stage all-1k
fi

if [[ "${SKIP_FF1K}" != "1" ]]; then
  echo ""
  echo "=== FF 1k fine-tune (backbone freeze, early-stop + epoch checkpoints) ==="
  python3 scripts/infer/video_xception_finetune.py \
    --root "${ROOT}" \
    --stage ff1k \
    --weights "${INIT_WEIGHTS}" \
    --output "${FF1K_OUT}" \
    --train-pull-dir data/pull/train/video/xception/ff1k \
    --write-train-manifest-docs \
    "${FT_FF_EXTRA[@]}" \
    "${FT_MAX[@]}"
fi

if [[ "${SKIP_CELEB1K}" != "1" ]]; then
  echo ""
  echo "=== Celeb 1k fine-tune (partial unfreeze, init from ff1k) ==="
  python3 scripts/infer/video_xception_finetune.py \
    --root "${ROOT}" \
    --stage celeb1k \
    --weights "${FF1K_OUT}" \
    --output "${CELEB1K_OUT}" \
    --train-pull-dir data/pull/train/video/xception/celeb1k \
    --unfreeze-backbone \
    --write-train-manifest-docs \
    "${FT_CELEB_EXTRA[@]}" \
    "${FT_MAX[@]}"
fi

echo ""
echo "Done."
echo "  ff1k weights:    ${FF1K_OUT}"
echo "  celeb1k weights: ${CELEB1K_OUT}"
echo "  checkpoints:     checkpoints/video/xception/xception_finetuned_ff1k/"
echo "                   checkpoints/video/xception/xception_finetuned_celeb1k/"
echo "  train manifests: docs/xception_finetune_train_ff1k.json"
echo "                   docs/xception_finetune_train_celeb1k.json"
echo ""
echo "Golden-200 eval:"
echo "  python3 scripts/eval/run_xception_1k_finetune_rebenchmark.py"
