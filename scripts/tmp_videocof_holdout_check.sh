#!/bin/bash
set -euo pipefail
cd ~/forenShield-ai/forgery

echo "=== counts ==="
for d in \
  data/test/video/spatial-videocof-benchmark \
  data/test/video/spatial-videocof \
  data/train/video/spatial-videocof \
  data/train/video/spatial-videocof-benchmark
do
  n=$(find "$d" -type f \( -name '*.mp4' -o -name '*.webm' \) 2>/dev/null | wc -l)
  echo "$n  $d"
done

echo "=== overlap train vs official test400 ==="
find data/train/video/spatial-videocof -type f -name '*.mp4' -printf '%f\n' | sort -u > /tmp/vc_train_names.txt
python3 - <<'PY'
import json
from pathlib import Path
pred = json.loads(Path("results/infer/trufor-videocof-v2-official-test400-f16-align-top3/predictions.json").read_text())
files = sorted({x["file"] for x in pred["items"]})
Path("/tmp/vc_test_names.txt").write_text("\n".join(files) + "\n")
print("test400 unique files:", len(files))
PY
wc -l /tmp/vc_train_names.txt /tmp/vc_test_names.txt
echo -n "overlap: "
comm -12 /tmp/vc_train_names.txt /tmp/vc_test_names.txt | wc -l

echo "=== split-half OOD check on existing predictions (thr locked 0.5) ==="
python3 - <<'PY'
import json
from pathlib import Path

pred = json.loads(Path("results/infer/trufor-videocof-v2-official-test400-f16-align-top3/predictions.json").read_text())
items = [x for x in pred["items"] if x.get("status") == "ok"]
thr = float(pred.get("threshold", 0.5))

def metrics(rows):
    tp = sum(1 for x in rows if x["ground_truth_label"] == "fake" and x["pred_label"] == "fake")
    tn = sum(1 for x in rows if x["ground_truth_label"] == "real" and x["pred_label"] == "real")
    fp = sum(1 for x in rows if x["ground_truth_label"] == "real" and x["pred_label"] == "fake")
    fn = sum(1 for x in rows if x["ground_truth_label"] == "fake" and x["pred_label"] == "real")
    n = len(rows)
    acc = (tp + tn) / n if n else 0.0
    return n, tp, tn, fp, fn, acc

# deterministic half split by filename
items_sorted = sorted(items, key=lambda x: x["file"])
half = len(items_sorted) // 2
a, b = items_sorted[:half], items_sorted[half:]
for name, rows in [("all", items_sorted), ("half_A", a), ("half_B", b)]:
    n, tp, tn, fp, fn, acc = metrics(rows)
    print(f"{name}: n={n} TP={tp} TN={tn} FP={fp} FN={fn} acc={acc:.4f} thr={thr}")

print("note: half_A/half_B use SAME locked thr=0.5 (no retune) — if one half collapses, fragile/overfit to subset")
PY

echo "=== infer script hints ==="
grep -n "aggregate\|align_pairs\|add_argument\|top3" scripts/infer/spatial_mvtamperbench_benchmark.py | head -40 || true
echo "--- ft script head ---"
head -60 scripts/train/run_trufor_videocof_v2.2_ft.sh || true
