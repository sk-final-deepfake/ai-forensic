#!/usr/bin/env bash
# Merge csvted-from-videocof FT best → infer csvted + mvtb + thr sweep
# GPU:
#   cd ~/forenShield-ai/forgery
#   sed -i 's/\r$//' scripts/train/run_csvted_eval_tmux.sh
#   bash scripts/train/run_csvted_eval_tmux.sh
set -euo pipefail

cd "${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai/forgery}"
source "${HOME}/forenShield-ai/.venv/bin/activate"
export FORENSHIELD_AI_ROOT="${FORENSHIELD_AI_ROOT:-$PWD}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXP="${EXP:-csvted-from-videocof-v2-20260713-0412}"
PRETRAIN="${PRETRAIN:-models/dev/spatial/trufor/v1.0.0/videocof-v2-20260710-0800/trufor.pth.tar}"
CKPT="${CKPT:-models/dev/spatial/trufor/v1.0.0/${EXP}/trufor.pth.tar}"
DATE="${DATE:-$(date +%Y%m%d-%H%M)}"
FRAMES="${FRAMES:-16}"
AGGREGATE="${AGGREGATE:-top3_mean}"
INFER_THRESHOLD="${INFER_THRESHOLD:-0.5}"

if [[ -f "weights/${EXP}/best.pth.tar" ]]; then
  TUNED="weights/${EXP}/best.pth.tar"
elif [[ -f "vendor/TruFor/TruFor_train_test/weights/${EXP}/best.pth.tar" ]]; then
  TUNED="vendor/TruFor/TruFor_train_test/weights/${EXP}/best.pth.tar"
else
  echo "ERROR: best.pth.tar not found for EXP=$EXP" >&2
  find weights vendor -name 'best.pth.tar' -path "*${EXP}*" 2>/dev/null || true
  exit 1
fi

[[ -f "$PRETRAIN" ]] || { echo "ERROR: missing base $PRETRAIN" >&2; exit 1; }

echo "=== merge ==="
echo "TUNED=$TUNED"
echo "OUT=$CKPT"
mkdir -p "$(dirname "$CKPT")"
python3 scripts/train/merge_trufor_infer_checkpoint.py \
  --base "$PRETRAIN" \
  --tuned "$TUNED" \
  --out "$CKPT"
ls -lah "$CKPT"

CSVTED_RUN="trufor-csvted-${EXP}-${DATE}"
MVTB_RUN="trufor-mvtb200-${EXP}-xeval-${DATE}"

echo "=== infer csvted → $CSVTED_RUN ==="
python3 scripts/infer/spatial_mvtamperbench_benchmark.py \
  --root "$FORENSHIELD_AI_ROOT" \
  --data-root data/pull/evidence/csvted-200-balanced \
  --model trufor \
  --num-frames "$FRAMES" \
  --aggregate "$AGGREGATE" \
  --threshold "$INFER_THRESHOLD" \
  --trufor-weights "$CKPT" \
  --run-id "$CSVTED_RUN"

[[ -f "results/infer/${CSVTED_RUN}/predictions.json" ]] || {
  echo "ERROR: csvted predictions missing" >&2
  ls -la "results/infer/${CSVTED_RUN}/" >&2 || true
  exit 1
}

python3 scripts/infer/sweep_spatial_benchmark_threshold.py \
  --predictions "results/infer/${CSVTED_RUN}/predictions.json" \
  --step 0.01

echo "=== infer mvtb → $MVTB_RUN ==="
python3 scripts/infer/spatial_mvtamperbench_benchmark.py \
  --root "$FORENSHIELD_AI_ROOT" \
  --data-root data/pull/evidence/mvtamperbench-200-s3 \
  --model trufor \
  --num-frames "$FRAMES" \
  --aggregate "$AGGREGATE" \
  --threshold "$INFER_THRESHOLD" \
  --trufor-weights "$CKPT" \
  --run-id "$MVTB_RUN"

[[ -f "results/infer/${MVTB_RUN}/predictions.json" ]] || {
  echo "ERROR: mvtb predictions missing" >&2
  ls -la "results/infer/${MVTB_RUN}/" >&2 || true
  exit 1
}

python3 scripts/infer/sweep_spatial_benchmark_threshold.py \
  --predictions "results/infer/${MVTB_RUN}/predictions.json" \
  --step 0.01

echo "DONE"
echo "  ckpt=$CKPT"
echo "  csvted=$CSVTED_RUN"
echo "  mvtb=$MVTB_RUN"
