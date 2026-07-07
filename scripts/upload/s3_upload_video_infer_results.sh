#!/usr/bin/env bash
# Upload infer JSON results + bundled report + benchmark dataset videos (mp4) to S3.
#
# Usage:
#   unset AWS_PROFILE
#   bash scripts/upload/s3_upload_video_infer_results.sh xception-benchmark-20260618-0411
#
# Optional env:
#   FORENSHIELD_AI_ROOT=~/forenShield-ai
#   S3_EVIDENCE_BUCKET=forenshield-evidence-877044078824
#   S3_REPORT_PREFIX=cases/test/video-xception-benchmark/reports
#   UPLOAD_VIDEOS=1          # default 1; set 0 to skip mp4 upload
set -eu

RUN_ID="${1:?usage: $0 <run_id>}"
ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
BUCKET="${S3_EVIDENCE_BUCKET:-forenshield-evidence-877044078824}"
PREFIX="${S3_REPORT_PREFIX:-cases/test/video-xception-benchmark/reports}"
UPLOAD_VIDEOS="${UPLOAD_VIDEOS:-1}"
LOCAL_JSON="${ROOT}/results/infer/${RUN_ID}/json"
LOCAL_PRED="${ROOT}/results/infer/${RUN_ID}/predictions.json"
LOCAL_METRICS="${ROOT}/results/eval/${RUN_ID}/metrics.json"
LOCAL_BUNDLE="${ROOT}/results/infer/${RUN_ID}/benchmark_report.json"
LOCAL_FAKE_MANIFEST="${ROOT}/results/infer/${RUN_ID}/datasets/fake_manifest.json"
LOCAL_REAL_MANIFEST="${ROOT}/results/infer/${RUN_ID}/datasets/real_manifest.json"
LOCAL_INFER_SUMMARY="${ROOT}/results/infer/${RUN_ID}/datasets/infer_summary.json"
S3_DEST="s3://${BUCKET}/${PREFIX}/${RUN_ID}"
BUNDLE_SCRIPT="${ROOT}/scripts/infer/bundle_xception_benchmark_report.py"

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws CLI not found"
  exit 1
fi

if [[ ! -d "${LOCAL_JSON}" ]]; then
  echo "ERROR: missing json dir: ${LOCAL_JSON}"
  exit 1
fi

if [[ -f "${BUNDLE_SCRIPT}" ]]; then
  echo "building bundled report..."
  python3 "${BUNDLE_SCRIPT}" "${RUN_ID}" --root "${ROOT}"
else
  echo "WARN: bundle script missing: ${BUNDLE_SCRIPT}"
fi

COUNT="$(find "${LOCAL_JSON}" -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')"
echo "uploading ${COUNT} per-video json files from ${LOCAL_JSON}"
echo "destination: ${S3_DEST}/json/"

aws s3 sync "${LOCAL_JSON}/" "${S3_DEST}/json/"

if [[ -f "${LOCAL_PRED}" ]]; then
  aws s3 cp "${LOCAL_PRED}" "${S3_DEST}/predictions.json"
fi
if [[ -f "${LOCAL_METRICS}" ]]; then
  aws s3 cp "${LOCAL_METRICS}" "${S3_DEST}/metrics.json"
fi
if [[ -f "${LOCAL_BUNDLE}" ]]; then
  echo "uploading bundled report: ${LOCAL_BUNDLE}"
  aws s3 cp "${LOCAL_BUNDLE}" "${S3_DEST}/benchmark_report.json"
fi
if [[ -f "${LOCAL_INFER_SUMMARY}" ]]; then
  echo "uploading infer summary: ${LOCAL_INFER_SUMMARY}"
  aws s3 cp "${LOCAL_INFER_SUMMARY}" "${S3_DEST}/datasets/infer_summary.json"
fi

if [[ -f "${LOCAL_PRED}" ]] && command -v python3 >/dev/null 2>&1; then
  read -r FAKE_DIR REAL_DIR <<PY
$(python3 - <<EOF
import json
from pathlib import Path
data = json.loads(Path("${LOCAL_PRED}").read_text())
print(data["fake_dir"])
print(data["real_dir"])
EOF
)
PY

  FAKE_DIR="${FAKE_DIR:-}"
  REAL_DIR="${REAL_DIR:-}"

  if [[ -d "${FAKE_DIR}" ]]; then
    if [[ -f "${LOCAL_FAKE_MANIFEST}" ]]; then
      aws s3 cp "${LOCAL_FAKE_MANIFEST}" "${S3_DEST}/datasets/fake/manifest.json"
      echo "uploaded normalized fake manifest"
    fi
    if [[ "${UPLOAD_VIDEOS}" == "1" ]]; then
      FAKE_COUNT="$(find "${FAKE_DIR}" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')"
      echo "uploading ${FAKE_COUNT} fake mp4 from ${FAKE_DIR}"
      aws s3 sync "${FAKE_DIR}/" "${S3_DEST}/datasets/fake/" --exclude "*" --include "*.mp4"
    fi
  fi

  if [[ -d "${REAL_DIR}" ]]; then
    if [[ -f "${LOCAL_REAL_MANIFEST}" ]]; then
      aws s3 cp "${LOCAL_REAL_MANIFEST}" "${S3_DEST}/datasets/real/manifest.json"
      echo "uploaded normalized real manifest"
    fi
    if [[ "${UPLOAD_VIDEOS}" == "1" ]]; then
      REAL_COUNT="$(find "${REAL_DIR}" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')"
      echo "uploading ${REAL_COUNT} real mp4 from ${REAL_DIR}"
      aws s3 sync "${REAL_DIR}/" "${S3_DEST}/datasets/real/" --exclude "*" --include "*.mp4"
    fi
  fi
fi

echo "done."
echo "bundled report: ${S3_DEST}/benchmark_report.json"
echo "infer summary:  ${S3_DEST}/datasets/infer_summary.json"
echo "dataset videos: ${S3_DEST}/datasets/fake/  ${S3_DEST}/datasets/real/"
echo "list json:      aws s3 ls ${S3_DEST}/json/ --human-readable"
