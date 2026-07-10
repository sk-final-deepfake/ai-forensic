#!/usr/bin/env bash
# Upload local RAFT benchmark bundle to S3 (raft/celebdf, raft/ffpp_vox).
#
# Usage:
#   unset AWS_PROFILE
#   bash scripts/upload/s3_upload_raft_benchmark_bundle.sh
#   OUT_DIR=results/raft-benchmark-bundle bash scripts/upload/s3_upload_raft_benchmark_bundle.sh
set -eu

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/s3_deepfake_paths.sh
source "${_SCRIPT_DIR}/../common/s3_deepfake_paths.sh"

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
OUT_DIR="${OUT_DIR:-results/raft-benchmark-bundle}"
BUCKET="${S3_EVIDENCE_BUCKET:-forenshield-evidence-877044078824}"
S3_RAFT_BASE="s3://${BUCKET}/${S3_DEEPFAKE_RESULTS_INFER}/raft"

upload_profile() {
  local profile="$1"
  local local_dir="${ROOT}/${OUT_DIR}/${profile}"
  [[ -d "$local_dir" ]] || { echo "missing $local_dir"; return 1; }
  echo "upload ${profile}..."
  aws s3 sync "${local_dir}/fake/" "${S3_RAFT_BASE}/${profile}/fake/" \
    --exclude "*" --include "*.json" --content-type "application/json"
  aws s3 sync "${local_dir}/real/" "${S3_RAFT_BASE}/${profile}/real/" \
    --exclude "*" --include "*.json" --content-type "application/json"
  aws s3 cp "${local_dir}/infer_summary.json" "${S3_RAFT_BASE}/${profile}/infer_summary.json" \
    --content-type "application/json"
  aws s3 cp "${local_dir}/metrics.json" "${S3_RAFT_BASE}/${profile}/metrics.json" \
    --content-type "application/json"
}

upload_profile celebdf
upload_profile ffpp_vox
echo "done: ${S3_RAFT_BASE}/"
