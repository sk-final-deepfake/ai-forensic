#!/usr/bin/env bash
# Record VideoCoF-v2 as models/dev adoption (ckpt + calibration JSON).
#
# GPU:
#   cd ~/forenShield-ai/forgery
#   sed -i 's/\r$//' scripts/train/record_videocof_v2_dev_adoption.sh
#   bash scripts/train/record_videocof_v2_dev_adoption.sh
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai/forgery}"
cd "$ROOT"

RUN_NAME="videocof-v2-20260710-0800"
SRC_BEST="models/train/spatial/trufor/videocof-v2/trufor-videocof-v2-20260710-0800/trufor_videocof_v2_ft/best.pth.tar"
DEV_DIR="models/dev/spatial/trufor/v1.0.0/${RUN_NAME}"
REPO_JSON_CANDIDATES=(
  "${HOME}/FINAL/ai-forensic/config/forgery/trufor_videocof_v2_dev_adoption.json"
  "${HOME}/forenShield-ai/ai-forensic/config/forgery/trufor_videocof_v2_dev_adoption.json"
  "config/forgery/trufor_videocof_v2_dev_adoption.json"
  "/tmp/trufor_videocof_v2_dev_adoption.json"
)

[[ -f "$SRC_BEST" ]] || { echo "ERROR: missing $SRC_BEST" >&2; exit 1; }

mkdir -p "$DEV_DIR"
cp -f "$SRC_BEST" "$DEV_DIR/trufor.pth.tar"
cp -f "models/train/spatial/trufor/videocof-v2/trufor-videocof-v2-20260710-0800/resolved_best.txt" \
  "$DEV_DIR/resolved_best.txt" 2>/dev/null || true

JSON_SRC=""
for c in "${REPO_JSON_CANDIDATES[@]}"; do
  if [[ -f "$c" ]]; then
    JSON_SRC="$c"
    break
  fi
done
if [[ -n "$JSON_SRC" ]]; then
  cp -f "$JSON_SRC" "$DEV_DIR/calibration.json"
  mkdir -p config/forgery
  DEST_CFG="config/forgery/trufor_videocof_v2_dev_adoption.json"
  if [[ "$(readlink -f "$JSON_SRC")" != "$(readlink -f "$DEST_CFG" 2>/dev/null || echo "")" ]]; then
    cp -f "$JSON_SRC" "$DEST_CFG"
  fi
  echo "copied adoption json from $JSON_SRC"
else
  echo "WARN: adoption json not found locally — write calibration stub"
  cat > "$DEV_DIR/calibration.json" <<'EOF'
{
  "model": "trufor",
  "line": "videocof-v2",
  "run_name": "videocof-v2-20260710-0800",
  "status": "dev_adopted",
  "scope": "videocof_only",
  "checkpoint": "models/dev/spatial/trufor/v1.0.0/videocof-v2-20260710-0800/trufor.pth.tar",
  "threshold": 0.5,
  "infer_recipe": { "num_frames": 16, "aggregate": "top3_mean", "align_pairs": true },
  "doc": "docs/ai/21-TruFor-VideoCoF-채택-및-분기-튜닝.md"
}
EOF
fi

# Lightweight pointer metrics for official test400
cat > "$DEV_DIR/metrics_videocof_official_test400.json" <<'EOF'
{
  "run_id": "trufor-videocof-v2-official-test400-f16-align-top3",
  "threshold": 0.5,
  "accuracy": 0.755,
  "confusion": { "tp": 143, "tn": 159, "fp": 41, "fn": 57 },
  "num_frames": 16,
  "aggregate": "top3_mean",
  "align_pairs": true,
  "data_root": "data/test/video/spatial-videocof-benchmark"
}
EOF

echo "=== VideoCoF-v2 dev adoption OK ==="
ls -la "$DEV_DIR"
echo "ckpt: $DEV_DIR/trufor.pth.tar"
echo "json: $DEV_DIR/calibration.json"
