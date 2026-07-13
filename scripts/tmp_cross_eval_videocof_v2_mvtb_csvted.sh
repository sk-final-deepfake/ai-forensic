#!/usr/bin/env bash
# Cross-eval: VideoCoF-v2 adopted ckpt -> mvtb200 + csvted (same infer recipe as official test400)
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai/forgery}"
REPO="${FORENSHIELD_REPO:-$HOME/forenShield-ai}"
cd "$REPO"
source "$REPO/.venv/bin/activate"
export FORENSHIELD_AI_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BENCH_PY="$ROOT/scripts/infer/spatial_mvtamperbench_benchmark.py"
WEIGHTS="$ROOT/models/train/spatial/trufor/videocof-v2/trufor-videocof-v2-20260710-0800/trufor_videocof_v2_ft/best.pth.tar"
STAMP="$(date -u +%Y%m%d-%H%M)"
LOG_DIR="$ROOT/results/infer/_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/cross-eval-videocof-v2-to-mvtb-csvted-${STAMP}.log"

MVTB_ROOT="$ROOT/data/pull/evidence/mvtamperbench-200-s3"
CSVTED_ROOT="$ROOT/data/pull/evidence/csvted-200-balanced"
MVTB_RUN="trufor-mvtb200-videocof-v2-xeval-f16-top3-${STAMP}"
CSVTED_RUN="trufor-csvted-videocof-v2-xeval-f16-top3-${STAMP}"

require() { [[ -e "$1" ]] || { echo "MISSING: $1" >&2; exit 1; }; }
require "$BENCH_PY"
require "$WEIGHTS"
require "$MVTB_ROOT"
require "$CSVTED_ROOT"

{
  echo "==> cross-eval start $STAMP"
  echo "weights: $WEIGHTS"
  echo "recipe: num-frames=16 aggregate=top3_mean threshold=0.5 (no align-pairs; mvtb/csvted unpaired)"
  echo "mvtb_run: $MVTB_RUN"
  echo "csvted_run: $CSVTED_RUN"

  echo ""
  echo "======== MVTB 200 ========"
  python3 "$BENCH_PY" \
    --root "$ROOT" \
    --data-root "$MVTB_ROOT" \
    --model trufor \
    --run-id "$MVTB_RUN" \
    --trufor-weights "$WEIGHTS" \
    --num-frames 16 \
    --aggregate top3_mean \
    --threshold 0.5 \
    --gpu 0

  echo ""
  echo "======== CSVTED ========"
  python3 "$BENCH_PY" \
    --root "$ROOT" \
    --data-root "$CSVTED_ROOT" \
    --model trufor \
    --run-id "$CSVTED_RUN" \
    --trufor-weights "$WEIGHTS" \
    --num-frames 16 \
    --aggregate top3_mean \
    --threshold 0.5 \
    --gpu 0

  echo ""
  echo "======== SUMMARY ========"
  python3 - <<PY
import json
from pathlib import Path
root = Path("$ROOT") / "results" / "infer"
for rid in ["$MVTB_RUN", "$CSVTED_RUN"]:
    p = root / rid / "predictions.json"
    m = root / rid / "metrics.json"
    print("===", rid, "===")
    if m.exists():
        d = json.loads(m.read_text())
        c = d.get("confusion", {})
        print("metrics:", {k: d.get(k) for k in ["threshold","accuracy","roc_auc"]})
        print("confusion:", c)
    elif p.exists():
        items = [x for x in json.loads(p.read_text())["items"] if x.get("status")=="ok"]
        tp=sum(1 for x in items if x.get("ground_truth_label")=="fake" and x.get("pred_label")=="fake")
        tn=sum(1 for x in items if x.get("ground_truth_label")=="real" and x.get("pred_label")=="real")
        fp=sum(1 for x in items if x.get("ground_truth_label")=="real" and x.get("pred_label")=="fake")
        fn=sum(1 for x in items if x.get("ground_truth_label")=="fake" and x.get("pred_label")=="real")
        n=len(items)
        print(f"from predictions: n={n} TP={tp} TN={tn} FP={fp} FN={fn} acc={(tp+tn)/n if n else 0:.4f}")
    else:
        print("MISSING results")
PY
  echo "DONE"
} 2>&1 | tee "$LOG"

echo "LOG=$LOG"
