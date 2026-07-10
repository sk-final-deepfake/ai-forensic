#!/usr/bin/env bash
# Video Swin on S3 bench datasets (celebdf + ffpp_vox, 50+50 each = 200)
# -> bundle under deepfake/results/infer/video-swin/{profile}/
#
# S3 input:
#   s3://.../deepfake/datasets/bench/celebdf/{fake,real}/
#   s3://.../deepfake/datasets/bench/ffpp_vox/{fake,real}/
#
# S3 output:
#   .../deepfake/results/infer/video-swin/celebdf/{infer_summary.json,metrics.json,fake/,real/}
#   .../deepfake/results/infer/video-swin/ffpp_vox/...
#
# Usage (GPU):
#   cd ~/forenShield-ai && source .venv/bin/activate && unset AWS_PROFILE
#   sed -i 's/\r$//' scripts/infer/run_video_swin_video_benchmark_datasets.sh
#   bash scripts/infer/run_video_swin_video_benchmark_datasets.sh
#
# Optional env:
#   SKIP_S3_PULL=1
#   SKIP_S3_UPLOAD=1
#   SKIP_FINETUNE=1          # use existing video_swin_finetuned.pth
#   THRESHOLD=0.5  MAX_CLIPS=4
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
unset AWS_PROFILE

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/s3_deepfake_paths.sh
source "${_SCRIPT_DIR}/../common/s3_deepfake_paths.sh"

BUCKET="${S3_EVIDENCE_BUCKET:-forenshield-evidence-877044078824}"
MODEL_SLUG="video-swin"
S3_DATA_BASE="s3://${BUCKET}/${S3_DEEPFAKE_DATASETS_BENCH}"
S3_MODEL_BASE="s3://${BUCKET}/${S3_DEEPFAKE_RESULTS_INFER}/${MODEL_SLUG}"
DATA_ROOT="${DATA_ROOT:-data/benchmark/video-benchmark-datasets}"
OUT_DIR="${OUT_DIR:-results/video-swin-benchmark-bundle}"
WEIGHTS="${WEIGHTS:-models/test/video/video-swin/v1.0.0/video_swin_finetuned.pth}"
THRESHOLD="${THRESHOLD:-0.5}"
MAX_CLIPS="${MAX_CLIPS:-4}"
SKIP_S3_PULL="${SKIP_S3_PULL:-0}"
SKIP_S3_UPLOAD="${SKIP_S3_UPLOAD:-0}"
SKIP_FINETUNE="${SKIP_FINETUNE:-1}"
RUN_TS="$(date -u +%Y%m%d-%H%M)"

pull_profile() {
  local profile="$1"
  echo "==> S3 pull ${profile}"
  aws s3 sync "${S3_DATA_BASE}/${profile}/fake/" "${ROOT}/${DATA_ROOT}/${profile}/fake/" \
    --exclude "*" --include "*.mp4"
  aws s3 sync "${S3_DATA_BASE}/${profile}/real/" "${ROOT}/${DATA_ROOT}/${profile}/real/" \
    --exclude "*" --include "*.mp4"
  echo "  fake: $(find "${ROOT}/${DATA_ROOT}/${profile}/fake" -maxdepth 1 -name '*.mp4' | wc -l)"
  echo "  real: $(find "${ROOT}/${DATA_ROOT}/${profile}/real" -maxdepth 1 -name '*.mp4' | wc -l)"
}

run_profile() {
  local profile="$1"
  local run_id="video-swin-benchmark-${profile}-${RUN_TS}"
  echo ""
  echo "==> Video Swin infer profile=${profile} run_id=${run_id}"

  python3 "$ROOT/scripts/infer/video_transformer_benchmark_infer.py" \
    --root "$ROOT" \
    --model video-swin \
    --run-id "$run_id" \
    --weights "$WEIGHTS" \
    --fake-dir "${DATA_ROOT}/${profile}/fake" \
    --real-dir "${DATA_ROOT}/${profile}/real" \
    --threshold "$THRESHOLD" \
    --max-clips "$MAX_CLIPS"

  python3 "$ROOT/scripts/infer/classifier_benchmark_profile_bundle.py" \
    --root "$ROOT" \
    --run-id "$run_id" \
    --profile "$profile" \
    --model-slug "$MODEL_SLUG" \
    --out-dir "$OUT_DIR" \
    --s3-dataset-prefix "$(s3_bench_profile "${profile}")"
}

upload_profile() {
  local profile="$1"
  local local_dir="${ROOT}/${OUT_DIR}/${profile}"
  if [[ ! -d "$local_dir" ]]; then
    echo "ERROR: missing ${local_dir}"
    exit 1
  fi
  echo "==> S3 upload ${profile} -> ${S3_MODEL_BASE}/${profile}/"
  aws s3 sync "${local_dir}/fake/" "${S3_MODEL_BASE}/${profile}/fake/" \
    --exclude "*" --include "*.json" --content-type "application/json"
  aws s3 sync "${local_dir}/real/" "${S3_MODEL_BASE}/${profile}/real/" \
    --exclude "*" --include "*.json" --content-type "application/json"
  aws s3 cp "${local_dir}/infer_summary.json" "${S3_MODEL_BASE}/${profile}/infer_summary.json" \
    --content-type "application/json"
  aws s3 cp "${local_dir}/metrics.json" "${S3_MODEL_BASE}/${profile}/metrics.json" \
    --content-type "application/json"
}

if [[ "$SKIP_FINETUNE" != "1" ]]; then
  echo "==> fine-tune Video Swin (FF++100 + Vox100, exclude celebdf test)"
  python3 "$ROOT/scripts/infer/video_transformer_finetune.py" \
    --root "$ROOT" \
    --model video-swin \
    --output "$WEIGHTS" \
    --exclude-dirs data/test/video/celeb-df-v2/fake data/test/video/celeb-df-v2/real \
    --max-per-class "${MAX_PER_CLASS:-100}" \
    --epochs "${FINETUNE_EPOCHS:-3}" \
    --batch-size "${FINETUNE_BATCH_SIZE:-1}"
fi

if [[ "$SKIP_S3_PULL" != "1" ]]; then
  pull_profile celebdf
  pull_profile ffpp_vox
else
  echo "SKIP_S3_PULL=1 — using ${ROOT}/${DATA_ROOT}"
  for profile in celebdf ffpp_vox; do
    echo "  ${profile} fake: $(find "${ROOT}/${DATA_ROOT}/${profile}/fake" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)"
    echo "  ${profile} real: $(find "${ROOT}/${DATA_ROOT}/${profile}/real" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)"
  done
fi

run_profile celebdf
run_profile ffpp_vox

if [[ "$SKIP_S3_UPLOAD" != "1" ]]; then
  upload_profile celebdf
  upload_profile ffpp_vox
  echo ""
  echo "S3 output:"
  echo "  ${S3_MODEL_BASE}/celebdf/"
  echo "  ${S3_MODEL_BASE}/ffpp_vox/"
else
  echo "SKIP_S3_UPLOAD=1 — local bundle: ${ROOT}/${OUT_DIR}/"
fi

echo ""
echo "DONE RUN_TS=${RUN_TS}"
echo "local: ${ROOT}/${OUT_DIR}/celebdf/ ${ROOT}/${OUT_DIR}/ffpp_vox/"
