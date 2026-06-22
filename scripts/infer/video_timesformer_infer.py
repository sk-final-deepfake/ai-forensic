#!/usr/bin/env python3
"""TimeSformer clip infer (Kinetics-pretrained backbone + binary head)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import TimesformerModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_clip_transformer_common import (
    MAX_CLIPS,
    SAMPLE_FRAMES,
    normalize_face_crops,
    run_clip_infer_directory,
)
from video_xception_infer import compute_metrics

CLIP_FRAMES = 8
CLIP_SIZE = 224
MODEL_ID = "timesformer/v1.0.0"
DEFAULT_PRETRAINED = "facebook/timesformer-base-finetuned-k400"


class TimeSformerDetectorLite(nn.Module):
    def __init__(self, pretrained_id: str = DEFAULT_PRETRAINED):
        super().__init__()
        self.backbone = TimesformerModel.from_pretrained(pretrained_id)
        hidden = self.backbone.config.hidden_size
        self.fc_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 2)
        self.embedding_dim = hidden

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        return self.fc_norm(outputs.last_hidden_state.mean(dim=1))

    def forward_logits(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(pixel_values))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward_logits(pixel_values), dim=1)[:, 1]


def clip_to_tensor(crops: list[np.ndarray], device: torch.device) -> torch.Tensor:
    arr = normalize_face_crops(crops)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    return tensor


def load_model(weights_path: Path, device: torch.device, pretrained_id: str = DEFAULT_PRETRAINED) -> TimeSformerDetectorLite:
    model = TimeSformerDetectorLite(pretrained_id=pretrained_id).to(device)
    if weights_path.is_file():
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="TimeSformer video infer")
    parser.add_argument("--weights", default="models/test/video/timesformer/v1.0.0/timesformer_finetuned.pth")
    parser.add_argument("--pretrained-id", default=DEFAULT_PRETRAINED)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--label", default=None, choices=["real", "fake"])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--root", default=".")
    parser.add_argument("--per-file-json", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--clip-frames", type=int, default=CLIP_FRAMES)
    parser.add_argument("--max-clips", type=int, default=MAX_CLIPS)
    parser.add_argument("--export-embedding", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = (root / input_dir).resolve()
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = (root / weights).resolve()

    run_id = args.run_id or f"timesformer-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    infer_dir = root / "results/infer" / run_id
    eval_dir = root / "results/eval" / run_id
    json_dir = infer_dir / "json" if args.per_file_json else None
    infer_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device, pretrained_id=args.pretrained_id)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    items = run_clip_infer_directory(
        model,
        face_cascade,
        device,
        input_dir,
        args.label,
        run_id,
        weights,
        json_dir,
        MODEL_ID,
        clip_to_tensor=clip_to_tensor,
        method="timesformer_clip_classification_outputs",
        threshold=args.threshold,
        export_embedding=args.export_embedding,
        clip_frames=args.clip_frames,
        clip_size=CLIP_SIZE,
        max_clips=args.max_clips,
    )
    metrics = compute_metrics(items, args.label)

    payload = {
        "run_id": run_id,
        "model": MODEL_ID,
        "threshold": args.threshold,
        "clip_frames": args.clip_frames,
        "max_clips": args.max_clips,
        "export_embedding": args.export_embedding,
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
