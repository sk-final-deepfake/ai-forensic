#!/usr/bin/env python3
"""Run Xception infer on real + fake video benchmark folders.

Writes one JSON per video (100 total when each folder has 50 mp4 files).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_xception_infer import (
    compute_metrics,
    load_model,
    run_directory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Xception benchmark infer (real + fake)")
    parser.add_argument("--weights", default="models/test/video/xception/v1.0.0/xception_best.pth")
    parser.add_argument("--fake-dir", default="data/test/video/ffpp/fake_over60s")
    parser.add_argument("--real-dir", default="data/test/video/voxceleb/real")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--root", default=".")
    parser.add_argument("--threshold", type=float, default=0.5, help="fake_score >= threshold => fake")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    fake_dir = Path(args.fake_dir)
    real_dir = Path(args.real_dir)
    if not fake_dir.is_absolute():
        fake_dir = (root / fake_dir).resolve()
    if not real_dir.is_absolute():
        real_dir = (root / real_dir).resolve()
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = (root / weights).resolve()

    run_id = args.run_id or f"xception-benchmark-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    infer_dir = root / "results/infer" / run_id
    eval_dir = root / "results/eval" / run_id
    json_dir = infer_dir / "json"
    infer_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print("run_id:", run_id)
    print("device:", device)
    print("threshold:", args.threshold)
    print("fake:", fake_dir)
    print("real:", real_dir)
    print()

    fake_items = run_directory(
        model, face_cascade, device, fake_dir, "fake", run_id, weights, json_dir,
        threshold=args.threshold,
    )
    print()
    real_items = run_directory(
        model, face_cascade, device, real_dir, "real", run_id, weights, json_dir,
        threshold=args.threshold,
    )

    all_items = fake_items + real_items
    metrics = {
        "run_id": run_id,
        "threshold": args.threshold,
        "total": len(all_items),
        "fake": compute_metrics(fake_items, "fake"),
        "real": compute_metrics(real_items, "real"),
        "overall_accuracy": round(
            sum(1 for x in all_items if x.get("correct")) / max(1, sum(1 for x in all_items if x["status"] == "ok")),
            4,
        ),
    }

    payload = {
        "run_id": run_id,
        "model": "xception/v1.0.0",
        "threshold": args.threshold,
        "weights": str(weights),
        "fake_dir": str(fake_dir),
        "real_dir": str(real_dir),
        "device": str(device),
        "items": all_items,
    }
    (infer_dir / "predictions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    json_count = len(list(json_dir.glob("*.json")))
    print()
    print("done:", json_count, "json files in", json_dir)
    print("metrics:", eval_dir / "metrics.json")
    print()
    print("S3 upload:")
    print(
        "  bash scripts/upload/s3_upload_video_infer_results.sh "
        f"{run_id}"
    )


if __name__ == "__main__":
    main()
