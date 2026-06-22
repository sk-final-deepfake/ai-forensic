#!/usr/bin/env bash
# Celeb-DF v2 50+50: RAFT and GMFlow run separately -> infer_summary_raft.json + infer_summary_gmflow.json -> S3
#
# Usage:
#   cd ~/forenShield-ai && source .venv/bin/activate && unset AWS_PROFILE
#   sed -i 's/\r$//' scripts/infer/run_optical_flow_celebdf_benchmark.sh
#   bash scripts/infer/run_optical_flow_celebdf_benchmark.sh
#
# Optional env:
#   RUN_ID=optical-flow-celebdf-20260619-1200
#   MAX_PAIRS=8
#   MAX_SIDE=512
#   UPLOAD_VIDEOS=1
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
unset AWS_PROFILE

DATASET_DIR="${DATASET_DIR:-data/test/video/celeb-df-v2}"
MAX_PAIRS="${MAX_PAIRS:-8}"
MAX_SIDE="${MAX_SIDE:-512}"
UPLOAD_VIDEOS="${UPLOAD_VIDEOS:-1}"
S3_REPORT_PREFIX="${S3_REPORT_PREFIX:-cases/test/video-optical-flow-benchmark/reports}"
RUN_ID="${RUN_ID:-optical-flow-celebdf-$(date -u +%Y%m%d-%H%M)}"

FAKE_DIR="$ROOT/$DATASET_DIR/fake"
REAL_DIR="$ROOT/$DATASET_DIR/real"

for dir in "$FAKE_DIR" "$REAL_DIR"; do
  count="$(find "$dir" -maxdepth 1 -name '*.mp4' | wc -l)"
  if [[ "$count" -lt 1 ]]; then
    echo "ERROR: no mp4 under $dir"
    exit 1
  fi
  echo "found $count videos in $dir"
done

run_model() {
  local model="$1"
  echo ""
  echo "==> infer model=$model run_id=$RUN_ID"
  python3 "$ROOT/scripts/infer/optical_flow_infer_model.py" \
    --root "$ROOT" \
    --run-id "$RUN_ID" \
    --model "$model" \
    --fake-dir "$FAKE_DIR" \
    --real-dir "$REAL_DIR" \
    --max-pairs "$MAX_PAIRS" \
    --max-side "$MAX_SIDE"
}

run_model raft
run_model gmflow

echo ""
echo "==> upload to S3"
S3_REPORT_PREFIX="$S3_REPORT_PREFIX" \
UPLOAD_VIDEOS="$UPLOAD_VIDEOS" \
bash "$ROOT/scripts/upload/s3_upload_optical_flow_results.sh" "$RUN_ID"

echo ""
echo "DONE RUN_ID=$RUN_ID"
echo "S3: s3://forenshield-evidence-877044078824/${S3_REPORT_PREFIX}/${RUN_ID}/"
echo "  datasets/infer_summary_raft.json"
echo "  datasets/infer_summary_gmflow.json"
echo "  raft/json/  gmflow/json/"
