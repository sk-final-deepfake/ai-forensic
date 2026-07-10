#!/usr/bin/env bash
# RAFT on S3 bench datasets (celebdf + ffpp_vox, 50+50 each)
# -> bundle under deepfake/results/infer/raft/{profile}/
#
# S3 input:
#   s3://.../deepfake/datasets/bench/celebdf/{fake,real}/
#   s3://.../deepfake/datasets/bench/ffpp_vox/{fake,real}/
#
# S3 output:
#   .../deepfake/results/infer/raft/celebdf/{infer_summary.json,metrics.json,fake/,real/}
#   .../deepfake/results/infer/raft/ffpp_vox/...
#
# Usage (GPU):
#   cd ~/forenShield-ai && source .venv/bin/activate && unset AWS_PROFILE
#   sed -i 's/\r$//' scripts/infer/run_raft_video_benchmark_datasets.sh
#   bash scripts/infer/run_raft_video_benchmark_datasets.sh
#
# Optional env:
#   SKIP_S3_PULL=1          # use local DATA_ROOT only
#   SKIP_S3_UPLOAD=1
#   MAX_PAIRS=8  MAX_SIDE=384   # lower MAX_SIDE if VRAM tight
#   RUN_ID=raft-benchmark-20260622-1200
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
S3_DATA_BASE="s3://${BUCKET}/${S3_DEEPFAKE_DATASETS_BENCH}"
S3_RAFT_BASE="s3://${BUCKET}/${S3_DEEPFAKE_RESULTS_INFER}/raft"
DATA_ROOT="${DATA_ROOT:-data/benchmark/video-benchmark-datasets}"
OUT_DIR="${OUT_DIR:-results/raft-benchmark-bundle}"
MAX_PAIRS="${MAX_PAIRS:-8}"
MAX_SIDE="${MAX_SIDE:-512}"
SKIP_S3_PULL="${SKIP_S3_PULL:-0}"
SKIP_S3_UPLOAD="${SKIP_S3_UPLOAD:-0}"
RUN_ID="${RUN_ID:-raft-benchmark-$(date -u +%Y%m%d-%H%M)}"

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
  echo ""
  echo "==> RAFT infer profile=${profile} run_id=${RUN_ID}"
  python3 "$ROOT/scripts/infer/raft_benchmark_profile_bundle.py" \
    --root "$ROOT" \
    --profile "$profile" \
    --fake-dir "${DATA_ROOT}/${profile}/fake" \
    --real-dir "${DATA_ROOT}/${profile}/real" \
    --out-dir "$OUT_DIR" \
    --run-id "${RUN_ID}" \
    --max-pairs "$MAX_PAIRS" \
    --max-side "$MAX_SIDE" \
    --s3-dataset-prefix "$(s3_bench_profile "${profile}")"
}

upload_profile() {
  local profile="$1"
  local local_dir="${ROOT}/${OUT_DIR}/${profile}"
  if [[ ! -d "$local_dir" ]]; then
    echo "ERROR: missing ${local_dir}"
    exit 1
  fi
  echo "==> S3 upload ${profile} -> ${S3_RAFT_BASE}/${profile}/"
  aws s3 sync "${local_dir}/fake/" "${S3_RAFT_BASE}/${profile}/fake/" \
    --exclude "*" --include "*.json" --content-type "application/json"
  aws s3 sync "${local_dir}/real/" "${S3_RAFT_BASE}/${profile}/real/" \
    --exclude "*" --include "*.json" --content-type "application/json"
  aws s3 cp "${local_dir}/infer_summary.json" "${S3_RAFT_BASE}/${profile}/infer_summary.json" \
    --content-type "application/json"
  aws s3 cp "${local_dir}/metrics.json" "${S3_RAFT_BASE}/${profile}/metrics.json" \
    --content-type "application/json"
}

if [[ "$SKIP_S3_PULL" != "1" ]]; then
  pull_profile celebdf
  pull_profile ffpp_vox
else
  echo "SKIP_S3_PULL=1 — using ${ROOT}/${DATA_ROOT}"
fi

run_profile celebdf
run_profile ffpp_vox

if [[ "$SKIP_S3_UPLOAD" != "1" ]]; then
  upload_profile celebdf
  upload_profile ffpp_vox
  echo ""
  echo "S3 output:"
  echo "  ${S3_RAFT_BASE}/celebdf/"
  echo "  ${S3_RAFT_BASE}/ffpp_vox/"
else
  echo "SKIP_S3_UPLOAD=1 — local bundle: ${ROOT}/${OUT_DIR}/"
fi

echo ""
echo "DONE RUN_ID=${RUN_ID}"
echo "local: ${ROOT}/${OUT_DIR}/celebdf/ ${ROOT}/${OUT_DIR}/ffpp_vox/"
