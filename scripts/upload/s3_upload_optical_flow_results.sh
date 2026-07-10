#!/usr/bin/env bash
# Upload per-model optical-flow results (raft + gmflow) and infer_summary_*.json to S3.
#
# Layout (local):
#   results/infer/<RUN_ID>/raft/json/*.json
#   results/infer/<RUN_ID>/gmflow/json/*.json
#   results/infer/<RUN_ID>/datasets/infer_summary_raft.json
#   results/infer/<RUN_ID>/datasets/infer_summary_gmflow.json
#   results/eval/<RUN_ID>/metrics_raft.json
#   results/eval/<RUN_ID>/metrics_gmflow.json
#
# Usage:
#   unset AWS_PROFILE
#   bash scripts/upload/s3_upload_optical_flow_results.sh optical-flow-celebdf-20260619-1200
set -eu

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/s3_deepfake_paths.sh
source "${_SCRIPT_DIR}/../common/s3_deepfake_paths.sh"

RUN_ID="${1:?usage: $0 <run_id>}"
ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
BUCKET="${S3_EVIDENCE_BUCKET:-forenshield-evidence-877044078824}"
PREFIX="${S3_REPORT_PREFIX:-$(s3_legacy_reports video-optical-flow-benchmark)}"
UPLOAD_VIDEOS="${UPLOAD_VIDEOS:-1}"
S3_DEST="s3://${BUCKET}/${PREFIX}/${RUN_ID}"

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws CLI not found"
  exit 1
fi

upload_model() {
  local model="$1"
  local local_json="${ROOT}/results/infer/${RUN_ID}/${model}/json"
  local local_pred="${ROOT}/results/infer/${RUN_ID}/${model}/predictions.json"
  local local_metrics="${ROOT}/results/eval/${RUN_ID}/metrics_${model}.json"

  if [[ ! -d "${local_json}" ]]; then
    echo "WARN: skip ${model} (missing ${local_json})"
    return 0
  fi

  local count
  count="$(find "${local_json}" -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')"
  echo "upload ${model}: ${count} json -> ${S3_DEST}/${model}/json/"
  aws s3 sync "${local_json}/" "${S3_DEST}/${model}/json/"

  if [[ -f "${local_pred}" ]]; then
    aws s3 cp "${local_pred}" "${S3_DEST}/${model}/predictions.json"
  fi
  if [[ -f "${local_metrics}" ]]; then
    aws s3 cp "${local_metrics}" "${S3_DEST}/eval/metrics_${model}.json"
  fi
}

upload_model raft
upload_model gmflow

DATASETS="${ROOT}/results/infer/${RUN_ID}/datasets"
for summary in infer_summary_raft.json infer_summary_gmflow.json; do
  if [[ -f "${DATASETS}/${summary}" ]]; then
    echo "upload summary: ${DATASETS}/${summary}"
    aws s3 cp "${DATASETS}/${summary}" "${S3_DEST}/datasets/${summary}"
  fi
done

if [[ "${UPLOAD_VIDEOS}" == "1" ]] && [[ -f "${ROOT}/results/infer/${RUN_ID}/raft/predictions.json" ]]; then
  FAKE_DIR="$(python3 -c "import json; print(json.load(open('${ROOT}/results/infer/${RUN_ID}/raft/predictions.json'))['fake_dir'])")"
  REAL_DIR="$(python3 -c "import json; print(json.load(open('${ROOT}/results/infer/${RUN_ID}/raft/predictions.json'))['real_dir'])")"

  if [[ -d "${FAKE_DIR}" ]]; then
    echo "upload dataset fake mp4 (once)"
    aws s3 sync "${FAKE_DIR}/" "${S3_DEST}/datasets/fake/" --exclude "*" --include "*.mp4"
  fi
  if [[ -d "${REAL_DIR}" ]]; then
    echo "upload dataset real mp4 (once)"
    aws s3 sync "${REAL_DIR}/" "${S3_DEST}/datasets/real/" --exclude "*" --include "*.mp4"
  fi
fi

echo "done."
echo "raft summary:  ${S3_DEST}/datasets/infer_summary_raft.json"
echo "gmflow summary: ${S3_DEST}/datasets/infer_summary_gmflow.json"
echo "raft json:     ${S3_DEST}/raft/json/"
echo "gmflow json:   ${S3_DEST}/gmflow/json/"
