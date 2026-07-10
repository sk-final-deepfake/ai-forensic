#!/usr/bin/env python3
"""Run VideoMAE infer on real + fake video benchmark folders (100 videos)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_videomae_infer import load_model, run_directory
from video_xception_infer import compute_metrics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import s3_deepfake_paths as s3p


def main() -> None:
    parser = argparse.ArgumentParser(description="VideoMAE benchmark infer (real + fake)")
    parser.add_argument("--weights", default="models/test/video/videomae/v1.0.0/videomae_finetuned.pth")
    parser.add_argument("--pretrained-id", default="MCG-NJU/videomae-base")
    parser.add_argument("--fake-dir", default="data/test/video/ffpp/fake_over60s")
    parser.add_argument("--real-dir", default="data/test/video/voxceleb/real")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--root", default=".")
    parser.add_argument("--threshold", type=float, default=0.5, help="prob_fake >= threshold => fake")
    parser.add_argument("--export-embedding", action="store_true")
    parser.add_argument("--max-clips", type=int, default=4)
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

    run_id = args.run_id or f"videomae-benchmark-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    infer_dir = root / "results/infer" / run_id
    eval_dir = root / "results/eval" / run_id
    json_dir = infer_dir / "json"
    infer_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device, pretrained_id=args.pretrained_id)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    model_id = "videomae/v1.0.0"

    print("run_id:", run_id)
    print("device:", device)
    print("threshold:", args.threshold)
    print("weights:", weights)
    print("fake:", fake_dir)
    print("real:", real_dir)
    print()

    fake_items = run_directory(
        model, face_cascade, device, fake_dir, "fake", run_id, weights, json_dir, model_id,
        threshold=args.threshold,
        export_embedding=args.export_embedding,
        max_clips=args.max_clips,
    )
    print()
    real_items = run_directory(
        model, face_cascade, device, real_dir, "real", run_id, weights, json_dir, model_id,
        threshold=args.threshold,
        export_embedding=args.export_embedding,
        max_clips=args.max_clips,
    )

    all_items = fake_items + real_items
    metrics = {
        "run_id": run_id,
        "threshold": args.threshold,
        "max_clips": args.max_clips,
        "export_embedding": args.export_embedding,
        "total": len(all_items),
        "fake": compute_metrics(fake_items, "fake"),
        "real": compute_metrics(real_items, "real"),
        "overall_accuracy": round(
            sum(1 for x in all_items if x.get("correct"))
            / max(1, sum(1 for x in all_items if x["status"] == "ok")),
            4,
        ),
    }

    payload = {
        "run_id": run_id,
        "model": model_id,
        "threshold": args.threshold,
        "max_clips": args.max_clips,
        "export_embedding": args.export_embedding,
        "weights": str(weights),
        "fake_dir": str(fake_dir),
        "real_dir": str(real_dir),
        "device": str(device),
        "items": all_items,
    }
    (infer_dir / "predictions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    print("done:", len(list(json_dir.glob("*.json"))), "json files in", json_dir)
    print("metrics:", eval_dir / "metrics.json")
    print("overall_accuracy:", metrics["overall_accuracy"])
    print()
    print("bundle + S3:")
    print(f"  python3 scripts/infer/bundle_xception_benchmark_report.py {run_id} --root .")
    print(f"  S3_REPORT_PREFIX={s3p.LEGACY_VIDEOMAE} \\")
    print(f"    UPLOAD_VIDEOS=0 bash scripts/upload/s3_upload_video_infer_results.sh {run_id}")


if __name__ == "__main__":
    main()
