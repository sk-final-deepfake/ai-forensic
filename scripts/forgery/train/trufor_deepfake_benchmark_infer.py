#!/usr/bin/env python3
"""
TruFor baseline deepfake benchmark (fake vs real folders).

Input data format (absolute or relative paths):
  --fake-dir: directory containing fake videos (recursively scanned)
  --real-dir: directory containing real videos (recursively scanned)

Video sampling:
  - uniform sampling of N frames (default 8)
  - frames are extracted with OpenCV and written to work/ for TruFor inference

TruFor inference:
  - calls vendor/TruFor/TruFor_train_test/test.py on extracted frames
  - uses npz['score'] (0..1) to compute video-level score by mean aggregation

Outputs:
  - results/infer/<run_id>/predictions.json
  - results/infer/<run_id>/metrics.json
  - results/infer/<run_id>/json/ per-video json files

Notes:
  - This is for *testing* baseline; it does not require masks/GT.
  - Video-level score is aggregated from sampled frame scores.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}


def iter_videos(root: Path) -> list[Path]:
    if not root.exists():
        return []
    videos: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
    videos.sort()
    return videos


def uniform_indices(total: int, num_samples: int) -> list[int]:
    if total <= 0:
        return []
    if total <= num_samples:
        return list(range(total))
    return [int(i * (total - 1) / (num_samples - 1)) for i in range(num_samples)]


def extract_frames_cv2(video_path: Path, out_dir: Path, frames_per_video: int) -> list[Path]:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = uniform_indices(total_frames, frames_per_video)
    if not indices:
        cap.release()
        return []

    extracted: list[Path] = []
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_path = out_dir / f"frame_{i:03d}.jpg"
        # OpenCV expects BGR for imwrite
        frame_bgr_out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok_write = cv2.imwrite(str(frame_path), frame_bgr_out)
        if ok_write:
            extracted.append(frame_path)

    cap.release()
    return extracted


@dataclass(frozen=True)
class Item:
    video_rel: str
    video_path: str
    gt: int  # 0=real, 1=fake
    score: float

    @property
    def pred(self) -> int:
        return 1 if self.score >= 0.5 else 0


def run_trufor_on_frames(
    *,
    gpu: int,
    frames_root: Path,
    npz_out_dir: Path,
    trufor_test_py: Path,
    experiment: str,
    model_file: Path,
) -> None:
    npz_out_dir.mkdir(parents=True, exist_ok=True)
    trufor_root = trufor_test_py.parent
    config_yaml = trufor_root / "lib" / "config" / f"{experiment}.yaml"
    if not config_yaml.exists():
        raise FileNotFoundError(
            f"Missing TruFor config: {config_yaml} (run test.py from TruFor_train_test cwd)"
        )

    cmd = [
        sys.executable,
        str(trufor_test_py),
        "-g",
        str(gpu),
        "-in",
        str(frames_root.resolve()),
        "-out",
        str(npz_out_dir.resolve()),
        "-exp",
        experiment,
        "TEST.MODEL_FILE",
        str(model_file.resolve()),
    ]
    # test.py loads lib/config/<exp>.yaml relative to TruFor_train_test cwd.
    subprocess.run(cmd, check=True, cwd=trufor_root)


def video_score_from_npz(video_npz_dir: Path) -> float:
    scores: list[float] = []
    for npz_path in video_npz_dir.glob("*.npz"):
        data = np.load(npz_path)
        if "score" in data:
            s = float(data["score"])
            if math.isfinite(s):
                scores.append(s)
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="TruFor deepfake benchmark infer (baseline only)")
    parser.add_argument("--root", default=".", help="forgery repo root (default '.')")
    parser.add_argument("--fake-dir", required=True, help="fake videos directory")
    parser.add_argument("--real-dir", required=True, help="real videos directory")
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--experiment", default="trufor_ph3", help="TruFor TEST exp (default: trufor_ph3)")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to TruFor checkpoint (.pth.tar). If not set, uses models/test/spatial/trufor/v1.0.0/trufor.pth.tar",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--clean-work", action="store_true", help="delete previous work dir before extracting")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    fake_dir = Path(args.fake_dir).expanduser()
    real_dir = Path(args.real_dir).expanduser()
    if not fake_dir.is_absolute():
        fake_dir = (root / fake_dir).resolve()
    if not real_dir.is_absolute():
        real_dir = (root / real_dir).resolve()

    if args.weights is None:
        model_file = root / "models" / "test" / "spatial" / "trufor" / "v1.0.0" / "trufor.pth.tar"
    else:
        model_file = Path(args.weights)
        if not model_file.is_absolute():
            model_file = (root / model_file).resolve()

    run_id = args.run_id
    if run_id is None:
        run_id = f"trufor-deepfake-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"

    infer_dir = root / "results" / "infer" / run_id
    json_dir = infer_dir / "json"
    work_dir = infer_dir / "work"
    frames_root = work_dir / "frames"
    npz_out_dir = work_dir / "npz_frames"
    eval_dir = infer_dir  # keep same level

    json_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)

    trufor_test_py = root / "vendor" / "TruFor" / "TruFor_train_test" / "test.py"
    if not trufor_test_py.exists():
        raise FileNotFoundError(f"Missing TruFor test.py: {trufor_test_py}")

    if args.clean_work and work_dir.exists():
        shutil.rmtree(work_dir)

    # 1) Extract frames
    splits = [("real", real_dir, 0), ("fake", fake_dir, 1)]
    video_items: list[Item] = []

    for split_name, split_root, gt in splits:
        vids = iter_videos(split_root)
        for vp in vids:
            # Keep relative identity for filenames/logs
            rel = str(vp.relative_to(split_root))
            # Create stable folder names (avoid path separators in npz parsing)
            video_stem = vp.stem
            safe_rel = rel.replace("\\", "_").replace("/", "_")
            frame_dir = frames_root / split_name / safe_rel
            extracted = extract_frames_cv2(vp, frame_dir, args.frames_per_video)
            if not extracted:
                # Still create folder to keep parsing simple; it will score as NaN
                frame_dir.mkdir(parents=True, exist_ok=True)

    # 2) Run TruFor inference on all extracted frames
    run_trufor_on_frames(
        gpu=args.gpu,
        frames_root=frames_root,
        npz_out_dir=npz_out_dir,
        trufor_test_py=trufor_test_py,
        experiment=args.experiment,
        model_file=model_file,
    )

    # 3) Aggregate video scores from npz
    items: list[dict] = []
    total_fake = 0
    total_real = 0

    for split_name, split_root, gt in splits:
        vids = iter_videos(split_root)
        for vp in vids:
            rel = str(vp.relative_to(split_root))
            safe_rel = rel.replace("\\", "_").replace("/", "_")
            video_npz_dir = npz_out_dir / split_name / safe_rel

            score = video_score_from_npz(video_npz_dir)
            if gt == 1:
                total_fake += 1
            else:
                total_real += 1

            # pred at provided threshold
            pred = 1 if (math.isfinite(score) and score >= args.threshold) else 0
            correct = 1 if pred == gt else 0

            item = {
                "video_rel": rel,
                "video_path": str(vp),
                "gt": gt,
                "score": score,
                "pred": pred,
                "correct": correct,
            }
            items.append(item)

            # save per-video json (for UI inspection parity)
            (json_dir / f"{split_name}__{safe_rel}.json").write_text(
                json.dumps(item, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # 4) Metrics
    labels = [x["gt"] for x in items]
    scores = [x["score"] for x in items]
    # Remove NaN scores
    labels_np = np.array(labels, dtype=np.int64)
    scores_np = np.array(scores, dtype=np.float32)
    mask = np.isfinite(scores_np)
    labels_np = labels_np[mask]
    scores_np = scores_np[mask]

    from sklearn.metrics import confusion_matrix, roc_auc_score

    pred_np = (scores_np >= args.threshold).astype(np.int64)
    # Confusion matrix expects labels order [0,1]
    tn, fp, fn, tp = confusion_matrix(labels_np, pred_np, labels=[0, 1]).ravel()
    total = len(items)
    ok = int(sum(x["correct"] for x in items))

    auc = roc_auc_score(labels_np, scores_np) if len(np.unique(labels_np)) > 1 else float("nan")
    accuracy = ok / max(1, len(items))

    metrics = {
        "run_id": run_id,
        "model": "trufor",
        "threshold": args.threshold,
        "total": total,
        "ok": ok,
        "accuracy": accuracy,
        "confusion": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
        "real": {
            "count": int(total_real),
            "avg_tamper_score": float(np.mean(scores_np[labels_np == 0])) if np.any(labels_np == 0) else float("nan"),
        },
        "fake": {
            "count": int(total_fake),
            "avg_tamper_score": float(np.mean(scores_np[labels_np == 1])) if np.any(labels_np == 1) else float("nan"),
        },
        "roc_auc": float(auc),
        "frames_per_video": args.frames_per_video,
        "fake_dir": str(fake_dir),
        "real_dir": str(real_dir),
        "weights": str(model_file),
    }

    (eval_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    predictions = {
        "run_id": run_id,
        "model": "trufor",
        "threshold": args.threshold,
        "weights": str(model_file),
        "fake_dir": str(fake_dir),
        "real_dir": str(real_dir),
        "items": items,
    }
    (infer_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("run_id:", run_id)
    print("metrics:", eval_dir / "metrics.json")
    print("predictions:", infer_dir / "predictions.json")


if __name__ == "__main__":
    main()

