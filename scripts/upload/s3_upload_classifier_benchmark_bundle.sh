#!/usr/bin/env bash
# Upload classifier benchmark bundle to S3 (e.g. video-swin/celebdf, video-swin/ffpp_vox).
#
# Usage:
#   unset AWS_PROFILE
#   MODEL_SLUG=video-swin bash scripts/upload/s3_upload_classifier_benchmark_bundle.sh
#   OUT_DIR=results/video-swin-benchmark-bundle MODEL_SLUG=video-swin bash scripts/upload/s3_upload_classifier_benchmark_bundle.sh
set -eu

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/s3_deepfake_paths.sh
source "${_SCRIPT_DIR}/../common/s3_deepfake_paths.sh"

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
MODEL_SLUG="${MODEL_SLUG:?set MODEL_SLUG=video-swin (or convnext, etc.)}"
OUT_DIR="${OUT_DIR:-results/${MODEL_SLUG}-benchmark-bundle}"
BUCKET="${S3_EVIDENCE_BUCKET:-forenshield-evidence-877044078824}"
S3_BASE="s3://${BUCKET}/${S3_DEEPFAKE_RESULTS_INFER}/${MODEL_SLUG}"

upload_profile() {
  local profile="$1"
  local local_dir="${ROOT}/${OUT_DIR}/${profile}"
  [[ -d "$local_dir" ]] || { echo "missing $local_dir"; return 1; }
  echo "upload ${MODEL_SLUG}/${profile}..."
  aws s3 sync "${local_dir}/fake/" "${S3_BASE}/${profile}/fake/" \
    --exclude "*" --include "*.json" --content-type "application/json"
  aws s3 sync "${local_dir}/real/" "${S3_BASE}/${profile}/real/" \
    --exclude "*" --include "*.json" --content-type "application/json"
  aws s3 cp "${local_dir}/infer_summary.json" "${S3_BASE}/${profile}/infer_summary.json" \
    --content-type "application/json"
  aws s3 cp "${local_dir}/metrics.json" "${S3_BASE}/${profile}/metrics.json" \
    --content-type "application/json"
}

upload_profile celebdf
upload_profile ffpp_vox
echo "done: ${S3_BASE}/"
