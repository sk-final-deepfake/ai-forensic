#!/usr/bin/env python3
"""Run TimeSformer or Video Swin infer on real + fake benchmark folders (100 videos)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_clip_transformer_common import run_clip_infer_directory
from video_swin3d_infer import (
    CLIP_FRAMES as SWIN_CLIP_FRAMES,
    MODEL_ID as SWIN_MODEL_ID,
    load_model as load_swin_model,
    clip_to_tensor as swin_clip_to_tensor,
)
from video_timesformer_infer import (
    CLIP_FRAMES as TS_CLIP_FRAMES,
    DEFAULT_PRETRAINED,
    MODEL_ID as TS_MODEL_ID,
    load_model as load_ts_model,
    clip_to_tensor as ts_clip_to_tensor,
)
from video_xception_infer import compute_metrics

SPECS = {
    "timesformer": {
        "model_id": TS_MODEL_ID,
        "clip_frames": TS_CLIP_FRAMES,
        "method": "timesformer_clip_classification_outputs",
        "default_weights": "models/test/video/timesformer/v1.0.0/timesformer_finetuned.pth",
        "run_prefix": "timesformer-celebdf-benchmark",
        "s3_prefix": "cases/test/video-timesformer-celebdf-benchmark/reports",
    },
    "video-swin": {
        "model_id": SWIN_MODEL_ID,
        "clip_frames": SWIN_CLIP_FRAMES,
        "method": "video_swin_clip_classification_outputs",
        "default_weights": "models/test/video/video-swin/v1.0.0/video_swin_finetuned.pth",
        "run_prefix": "video-swin-celebdf-benchmark",
        "s3_prefix": "cases/test/video-swin-celebdf-benchmark/reports",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Video transformer benchmark infer (TimeSformer / Video Swin)")
    parser.add_argument("--model", required=True, choices=sorted(SPECS.keys()))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--pretrained-id", default=DEFAULT_PRETRAINED)
    parser.add_argument("--fake-dir", default="data/test/video/celeb-df-v2/fake")
    parser.add_argument("--real-dir", default="data/test/video/celeb-df-v2/real")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--root", default=".")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-clips", type=int, default=4)
    parser.add_argument("--export-embedding", action="store_true")
    args = parser.parse_args()

    spec = SPECS[args.model]
    root = Path(args.root).resolve()
    fake_dir = Path(args.fake_dir)
    real_dir = Path(args.real_dir)
    if not fake_dir.is_absolute():
        fake_dir = (root / fake_dir).resolve()
    if not real_dir.is_absolute():
        real_dir = (root / real_dir).resolve()

    weights = Path(args.weights or spec["default_weights"])
    if not weights.is_absolute():
        weights = (root / weights).resolve()

    run_id = args.run_id or f"{spec['run_prefix']}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    infer_dir = root / "results/infer" / run_id
    eval_dir = root / "results/eval" / run_id
    json_dir = infer_dir / "json"
    infer_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "timesformer":
        model = load_ts_model(weights, device, pretrained_id=args.pretrained_id)
        clip_to_tensor = ts_clip_to_tensor
    else:
        model = load_swin_model(weights, device)
        clip_to_tensor = swin_clip_to_tensor

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print("run_id:", run_id)
    print("model:", args.model)
    print("device:", device)
    print("threshold:", args.threshold)
    print("weights:", weights)
    print("fake:", fake_dir)
    print("real:", real_dir)
    print()

    fake_items = run_clip_infer_directory(
        model,
        face_cascade,
        device,
        fake_dir,
        "fake",
        run_id,
        weights,
        json_dir,
        spec["model_id"],
        clip_to_tensor=clip_to_tensor,
        method=spec["method"],
        threshold=args.threshold,
        export_embedding=args.export_embedding,
        clip_frames=spec["clip_frames"],
        clip_size=224,
        max_clips=args.max_clips,
    )
    print()
    real_items = run_clip_infer_directory(
        model,
        face_cascade,
        device,
        real_dir,
        "real",
        run_id,
        weights,
        json_dir,
        spec["model_id"],
        clip_to_tensor=clip_to_tensor,
        method=spec["method"],
        threshold=args.threshold,
        export_embedding=args.export_embedding,
        clip_frames=spec["clip_frames"],
        clip_size=224,
        max_clips=args.max_clips,
    )

    all_items = fake_items + real_items
    metrics = {
        "run_id": run_id,
        "model": spec["model_id"],
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
        "model": spec["model_id"],
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
    print(f"  python3 scripts/infer/bundle_xception_benchmark_report.py {run_id} --root . --profile celebdf")
    print(f"  S3_REPORT_PREFIX={spec['s3_prefix']} UPLOAD_VIDEOS=1 \\")
    print(f"    bash scripts/upload/s3_upload_video_infer_results.sh {run_id}")


if __name__ == "__main__":
    main()
