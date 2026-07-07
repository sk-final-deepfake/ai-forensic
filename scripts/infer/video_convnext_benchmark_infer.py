#!/usr/bin/env python3
"""Run ConvNeXt infer on real + fake video benchmark folders."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_convnext_infer import DEFAULT_VARIANT, MODEL_ID, load_model
from video_xception_infer import compute_metrics, run_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="ConvNeXt benchmark infer (real + fake)")
    parser.add_argument("--weights", default="models/test/video/convnext/v1.0.0/convnext_finetuned.pth")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=["small", "base"])
    parser.add_argument("--fake-dir", default="data/test/video/celeb-df-v2/fake")
    parser.add_argument("--real-dir", default="data/test/video/celeb-df-v2/real")
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

    run_id = args.run_id or f"convnext-celebdf-benchmark-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    infer_dir = root / "results/infer" / run_id
    eval_dir = root / "results/eval" / run_id
    json_dir = infer_dir / "json"
    infer_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device, variant=args.variant)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print("run_id:", run_id)
    print("device:", device)
    print("variant:", args.variant)
    print("threshold:", args.threshold)
    print("fake:", fake_dir)
    print("real:", real_dir)
    print()

    fake_items = run_directory(
        model,
        face_cascade,
        device,
        fake_dir,
        "fake",
        run_id,
        weights,
        json_dir,
        MODEL_ID,
        threshold=args.threshold,
    )
    print()
    real_items = run_directory(
        model,
        face_cascade,
        device,
        real_dir,
        "real",
        run_id,
        weights,
        json_dir,
        MODEL_ID,
        threshold=args.threshold,
    )

    all_items = fake_items + real_items
    ok_count = sum(1 for x in all_items if x["status"] == "ok")
    metrics = {
        "run_id": run_id,
        "threshold": args.threshold,
        "total": len(all_items),
        "fake": compute_metrics(fake_items, "fake"),
        "real": compute_metrics(real_items, "real"),
        "overall_accuracy": round(
            sum(1 for x in all_items if x.get("correct")) / max(1, ok_count),
            4,
        ),
    }

    payload = {
        "run_id": run_id,
        "model": MODEL_ID,
        "variant": args.variant,
        "threshold": args.threshold,
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
    print()
    print("bundle + S3:")
    print(f"  python3 scripts/infer/bundle_xception_benchmark_report.py {run_id} --root . --profile celebdf")
    print(f"  S3_REPORT_PREFIX=cases/test/video-convnext-celebdf-benchmark/reports \\")
    print(f"    bash scripts/upload/s3_upload_video_infer_results.sh {run_id}")


if __name__ == "__main__":
    main()
