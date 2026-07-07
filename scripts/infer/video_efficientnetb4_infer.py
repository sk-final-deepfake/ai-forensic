#!/usr/bin/env python3
"""Smoke infer for DeepfakeBench effnb4_best.pth on local mp4 folders."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from video_xception_infer import (
    compute_metrics,
    infer_video,
    run_directory,
    write_per_file_json,
)


class EfficientNetB4(nn.Module):
    """DeepfakeBench efficientnetb4 backbone (mode=Original from efficientnetb4.yaml)."""

    def __init__(self, num_classes=2, inc=3, dropout=False, mode="Original"):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.mode = mode
        self.efficientnet = EfficientNet.from_name("efficientnet-b4")
        self.efficientnet._conv_stem = nn.Conv2d(inc, 48, kernel_size=3, stride=2, bias=False)
        self.efficientnet._fc = nn.Identity()
        self.last_layer = nn.Linear(1792, num_classes)
        if dropout:
            self.dropout_layer = nn.Dropout(p=dropout)
        if mode == "adjust_channel":
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(1792, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

    def features(self, x):
        x = self.efficientnet.extract_features(x)
        if self.mode == "adjust_channel":
            x = self.adjust_channel(x)
        return x

    def classifier(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = self.dropout_layer(x)
        return self.last_layer(x)


class EfficientNetB4DetectorLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNetB4()

    def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.features(images)
        return self.backbone.classifier(feat)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward_logits(images), dim=1)[:, 1]


def load_model(weights_path: Path, device: torch.device) -> EfficientNetB4DetectorLite:
    model = EfficientNetB4DetectorLite().to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="EfficientNet-B4 video infer (DeepfakeBench weights)")
    parser.add_argument("--weights", default="models/test/video/efficientnetb4/v1.0.0/effnb4_best.pth")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--label", default=None, choices=["real", "fake"])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--root", default=".")
    parser.add_argument("--per-file-json", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5, help="fake_score >= threshold => fake")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = (root / input_dir).resolve()
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = (root / weights).resolve()

    run_id = args.run_id or f"efficientnetb4-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    infer_dir = root / "results/infer" / run_id
    eval_dir = root / "results/eval" / run_id
    json_dir = infer_dir / "json" if args.per_file_json else None
    infer_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    items = run_directory(
        model,
        face_cascade,
        device,
        input_dir,
        args.label,
        run_id,
        weights,
        json_dir,
        model_id="efficientnetb4/v1.0.0",
        threshold=args.threshold,
    )
    metrics = compute_metrics(items, args.label)

    payload = {
        "run_id": run_id,
        "model": "efficientnetb4/v1.0.0",
        "threshold": args.threshold,
        "weights": str(weights),
        "input_dir": str(input_dir),
        "device": str(device),
        "items": items,
    }
    (infer_dir / "predictions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
