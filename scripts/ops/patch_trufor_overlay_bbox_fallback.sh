#!/usr/bin/env bash
# Patch GPU TruFor overlay bake: when BE omits bboxes, synthesize center boxes from scores.
# Target: /home/sk4team/ai-forensic/app/services/module_overlays.py
#
# Usage (on GPU):
#   bash scripts/ops/patch_trufor_overlay_bbox_fallback.sh
#   bash /tmp/start_overlay_worker_method_b.sh
set -euo pipefail

TARGET="${1:-/home/sk4team/ai-forensic/app/services/module_overlays.py}"
python3 - "$TARGET" <<'PY'
from pathlib import Path
import ast
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
bak = path.with_suffix(path.suffix + ".bak-trufor-bbox-fallback")
if not bak.exists():
    bak.write_text(text, encoding="utf-8")

if "using score-centered fallback boxes" in text and "def _synthesize_center_bboxes" in text:
    print(f"already patched: {path}")
    raise SystemExit(0)

old = '''    bboxes_by_frame = _frame_risks_to_bboxes(frame_risks)
    frame_scores = _frame_risks_to_frame_scores(frame_risks)
    if not bboxes_by_frame and not frame_scores:
        return None
    if not bboxes_by_frame:
        logger.warning(
            "TruFor overlay has scores but no bboxes (evidenceId=%s analysisRequestId=%s); "
            "refusing full-frame border fallback",
            evidence_id,
            analysis_request_id,
        )
        return None
    if os.getenv("AI_VISUALIZATION_OVERLAY", "1").lower() in {"0", "false", "no"}:
        return None

    work_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return None
'''

new = '''    bboxes_by_frame = _frame_risks_to_bboxes(frame_risks)
    frame_scores = _frame_risks_to_frame_scores(frame_risks)
    if not bboxes_by_frame and not frame_scores:
        return None
    if os.getenv("AI_VISUALIZATION_OVERLAY", "1").lower() in {"0", "false", "no"}:
        return None

    work_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return None

    # Legacy BE payloads may only have scores (bboxes stripped). Keep overlay usable.
    if not bboxes_by_frame and frame_scores:
        logger.warning(
            "TruFor overlay has scores but no bboxes (evidenceId=%s analysisRequestId=%s); "
            "using score-centered fallback boxes",
            evidence_id,
            analysis_request_id,
        )
        bboxes_by_frame = _synthesize_center_bboxes(frame_scores, width, height)
'''

helper = '''
def _synthesize_center_bboxes(
    frame_scores: dict[int, float],
    frame_w: int,
    frame_h: int,
) -> dict[int, list[dict[str, Any]]]:
    """Approximate localization when BE omitted TruFor bboxes."""
    side = max(32, int(min(frame_w, frame_h) * 0.28))
    cx, cy = frame_w // 2, frame_h // 2
    x = max(0, cx - side // 2)
    y = max(0, cy - side // 2)
    w = min(side, frame_w - x)
    h = min(side, frame_h - y)
    out: dict[int, list[dict[str, Any]]] = {}
    for frame_index, score in frame_scores.items():
        out[int(frame_index)] = [
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "score": float(score),
            }
        ]
    return out

'''

if "using score-centered fallback boxes" not in text:
    if old not in text:
        raise SystemExit(f"OLD BLOCK NOT FOUND in {path}")
    text = text.replace(old, new, 1)

if "def _synthesize_center_bboxes" not in text:
    anchor = "def _frame_risks_to_bboxes("
    if anchor not in text:
        raise SystemExit("helper anchor missing")
    text = text.replace(anchor, helper + anchor, 1)

ast.parse(text)
path.write_text(text, encoding="utf-8")
print(f"patched: {path}")
print(f"backup:  {bak}")
PY
