#!/usr/bin/env python3
"""Video infer for ConvNeXt deepfake classifier (ImageNet backbone + binary head)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Small_Weights,
    convnext_base,
    convnext_small,
)

from video_xception_infer import (
    compute_metrics,
    run_directory,
)

MODEL_ID = "convnext/v1.0.0"
DEFAULT_VARIANT = "small"

VARIANTS = {
    "small": (convnext_small, ConvNeXt_Small_Weights.IMAGENET1K_V1),
    "base": (convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1),
}


class ConvNeXtBackbone(nn.Module):
    """ConvNeXt backbone with DeepfakeBench-style features/classifier split."""

    def __init__(self, num_classes: int = 2, variant: str = DEFAULT_VARIANT, pretrained: bool = False):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        builder, weights_enum = VARIANTS[variant]
        weights = weights_enum if pretrained else None
        net = builder(weights=weights)
        self.stem_and_stages = net.features
        self.avgpool = net.avgpool
        self.norm = net.classifier[0]
        in_features = net.classifier[2].in_features
        self.head = nn.Linear(in_features, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem_and_stages(x)

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(features)
        x = self.norm(x)
        x = torch.flatten(x, 1)
        return self.head(x)


class ConvNeXtDetectorLite(nn.Module):
    def __init__(self, variant: str = DEFAULT_VARIANT, pretrained: bool = False):
        super().__init__()
        self.variant = variant
        self.backbone = ConvNeXtBackbone(num_classes=2, variant=variant, pretrained=pretrained)

    def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.features(images)
        return self.backbone.classifier(feat)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward_logits(images), dim=1)[:, 1]


def load_model(
    weights_path: Path,
    device: torch.device,
    *,
    variant: str = DEFAULT_VARIANT,
) -> ConvNeXtDetectorLite:
    model = ConvNeXtDetectorLite(variant=variant, pretrained=False).to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="ConvNeXt video infer")
    parser.add_argument("--weights", default="models/test/video/convnext/v1.0.0/convnext_finetuned.pth")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=sorted(VARIANTS))
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

    run_id = args.run_id or f"convnext-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    infer_dir = root / "results/infer" / run_id
    eval_dir = root / "results/eval" / run_id
    json_dir = infer_dir / "json" if args.per_file_json else None
    infer_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device, variant=args.variant)
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
        model_id=MODEL_ID,
        threshold=args.threshold,
    )
    metrics = compute_metrics(items, args.label)

    payload = {
        "run_id": run_id,
        "model": MODEL_ID,
        "variant": args.variant,
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
