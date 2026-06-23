#!/usr/bin/env bash
# Ad-hoc optical-flow test (YouTube / custom mp4 folders).
#
# Requires fake/ and real/ mp4 folders in the SAME run so motion_anomaly
# baseline (real cohort z-score) is meaningful.
#
# Usage (GPU):
#   cd ~/forenShield-ai && source .venv/bin/activate
#   sed -i 's/\r$//' scripts/infer/run_custom_optical_flow_test.sh
#   bash scripts/infer/run_custom_optical_flow_test.sh youtube-ai-test-20260622 \
#     data/test/video/youtube-pilot/fake data/test/video/youtube-pilot/real
#
# Optional env:
#   MAX_PAIRS=8  MAX_SIDE=512  MODELS=raft,gmflow
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
RUN_ID="${1:?usage: $0 <run_id> <fake_dir> <real_dir>}"
FAKE_DIR="${2:?missing fake_dir}"
REAL_DIR="${3:?missing real_dir}"
MAX_PAIRS="${MAX_PAIRS:-8}"
MAX_SIDE="${MAX_SIDE:-512}"
MODELS="${MODELS:-raft,gmflow}"

cd "$ROOT"

if [[ ! -d "$FAKE_DIR" ]] || [[ -z "$(find "$FAKE_DIR" -maxdepth 1 -name '*.mp4' | head -1)" ]]; then
  echo "ERROR: no mp4 in fake_dir: $FAKE_DIR" >&2
  exit 1
fi
if [[ ! -d "$REAL_DIR" ]] || [[ -z "$(find "$REAL_DIR" -maxdepth 1 -name '*.mp4' | head -1)" ]]; then
  echo "ERROR: no mp4 in real_dir: $REAL_DIR (needed for motion baseline)" >&2
  exit 1
fi

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"

for model in "${MODEL_LIST[@]}"; do
  model="$(echo "$model" | tr -d ' ')"
  [[ -n "$model" ]] || continue
  echo ""
  echo "========== infer: $model =========="
  python3 "$ROOT/scripts/infer/optical_flow_infer_model.py" \
    --root "$ROOT" \
    --model "$model" \
    --run-id "$RUN_ID" \
    --fake-dir "$FAKE_DIR" \
    --real-dir "$REAL_DIR" \
    --max-pairs "$MAX_PAIRS" \
    --max-side "$MAX_SIDE"

  echo ""
  echo "========== report: $model =========="
  python3 "$ROOT/scripts/infer/regenerate_optical_flow_reports.py" "$RUN_ID" \
    --root "$ROOT" \
    --run-dir "$ROOT/results/infer/$RUN_ID/$model" \
    --model "$model" \
    --no-sync-s3
done

echo ""
echo "DONE run_id=$RUN_ID"
for model in "${MODEL_LIST[@]}"; do
  model="$(echo "$model" | tr -d ' ')"
  html="$ROOT/results/infer/$RUN_ID/$model/benchmark_report.html"
  json="$ROOT/results/infer/$RUN_ID/$model/benchmark_report.json"
  echo "  [$model] HTML  $html"
  echo "  [$model] JSON  $json"
done
echo ""
echo "Tip: scp benchmark_report.html to PC and open in a browser."
