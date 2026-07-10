#!/usr/bin/env python3
"""Smoke infer for DeepfakeBench xception_best.pth on local mp4 folders.

No DeepfakeBench / dlib required. Face crop via Haar or MediaPipe + frame sampling.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from face_crop import FaceCropper, create_face_cropper, crop_face as _legacy_crop_face
from xception_operating_defaults import (
    DEFAULT_AGGREGATE,
    DEFAULT_CROP_METHOD,
    DEFAULT_CROP_PADDING,
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_NUM_FRAMES,
    DEFAULT_TOP_K,
)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.conv1(x))


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super().__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        rep = []
        filters = in_filters
        if grow_first:
            rep.extend([nn.ReLU(inplace=True), SeparableConv2d(in_filters, out_filters, 3, 1, 1, bias=False), nn.BatchNorm2d(out_filters)])
            filters = out_filters
        for _ in range(reps - 1):
            rep.extend([nn.ReLU(inplace=True), SeparableConv2d(filters, filters, 3, 1, 1, bias=False), nn.BatchNorm2d(filters)])
        if not grow_first:
            rep.extend([nn.ReLU(inplace=True), SeparableConv2d(in_filters, out_filters, 3, 1, 1, bias=False), nn.BatchNorm2d(out_filters)])
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skipbn(self.skip(inp))
        else:
            skip = inp
        return x + skip


class Xception(nn.Module):
    """DeepfakeBench xception backbone (mode=original from xception.yaml)."""

    def __init__(self, num_classes=2, inc=3, dropout=False, mode="original"):
        super().__init__()
        self.num_classes = num_classes
        self.mode = mode

        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        final_channel = 2048
        if mode == "adjust_channel_iid":
            final_channel = 512
            mode = "adjust_channel"
            self.mode = mode
        self.last_linear = nn.Linear(final_channel, num_classes)
        if dropout:
            self.last_linear = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(final_channel, num_classes))

        # Present in checkpoint but unused when mode != "adjust_channel".
        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )

    def features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block3(self.block2(self.block1(x)))
        x = self.block7(self.block6(self.block5(self.block4(x))))
        x = self.block12(self.block11(self.block10(self.block9(self.block8(x)))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        if self.mode == "adjust_channel":
            x = self.adjust_channel(x)
        return x

    def classifier(self, features):
        if self.mode == "adjust_channel":
            x = features
        else:
            x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        return self.last_linear(x)


class XceptionDetectorLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Xception()

    def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.features(images)
        return self.backbone.classifier(feat)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward_logits(images), dim=1)[:, 1]


SCORE_BREAKDOWN_SCHEMA_VERSION = "1.1"


def load_model(weights_path: Path, device: torch.device) -> XceptionDetectorLite:
    model = XceptionDetectorLite().to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model


def read_frame_samples(video_path: Path, num_frames: int = 32) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []

    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = [int(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]

    samples: list[dict] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            samples.append({"frame_index": idx, "frame": frame})
    cap.release()
    return samples


def read_frames(video_path: Path, num_frames: int = 32) -> list[np.ndarray]:
    return [s["frame"] for s in read_frame_samples(video_path, num_frames)]


def crop_face(frame: np.ndarray, face_cascade: cv2.CascadeClassifier, size: int = 256) -> np.ndarray | None:
    """Backward-compatible Haar crop for other infer scripts."""
    return _legacy_crop_face(frame, face_cascade, size=size, padding=0.2, square=False)


def frames_to_tensor(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)
    return tensor


def _round4(value: float) -> float:
    return round(float(value), 4)


def _binary_entropy(prob_real: float, prob_fake: float) -> float:
    total = 0.0
    for prob in (prob_real, prob_fake):
        if prob > 0.0:
            total -= prob * math.log(prob)
    return total


def classification_row_from_logits(logit_real: float, logit_fake: float, *, threshold: float) -> dict:
    logits = np.array([logit_real, logit_fake], dtype=np.float64)
    probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
    prob_real = float(probs[0])
    prob_fake = float(probs[1])
    margin = float(logit_fake - logit_real)
    entropy = _binary_entropy(prob_real, prob_fake)
    confidence = max(prob_real, prob_fake)
    pred_label = "fake" if prob_fake >= threshold else "real"
    return {
        "logit_real": _round4(logit_real),
        "logit_fake": _round4(logit_fake),
        "prob_real": _round4(prob_real),
        "prob_fake": _round4(prob_fake),
        "margin": _round4(margin),
        "entropy": _round4(entropy),
        "confidence": _round4(confidence),
        "pred_label": pred_label,
    }


def logits_to_frame_rows(
    logits: np.ndarray,
    face_samples: list[dict],
    *,
    threshold: float,
) -> list[dict]:
    rows: list[dict] = []
    for meta, (logit_real, logit_fake) in zip(face_samples, logits):
        row = classification_row_from_logits(float(logit_real), float(logit_fake), threshold=threshold)
        row["frame_index"] = int(meta["frame_index"])
        row["face_index"] = int(meta.get("face_index", 0))
        row["face_detected"] = True
        bbox = meta.get("bbox")
        if bbox is not None:
            x, y, w, h = bbox
            row["bbox"] = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        rows.append(row)
    return rows


def _distribution_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "min": None,
            "max": None,
            "median": None,
            "std": None,
            "p25": None,
            "p75": None,
        }
    return {
        "min": _round4(float(np.min(values))),
        "max": _round4(float(np.max(values))),
        "median": _round4(float(np.median(values))),
        "std": _round4(float(np.std(values))),
        "p25": _round4(float(np.percentile(values, 25))),
        "p75": _round4(float(np.percentile(values, 75))),
    }


def aggregate_prob_fake(
    prob_fake: np.ndarray,
    method: str,
    *,
    top_k: int = 5,
    frame_threshold: float = 0.5,
) -> float:
    if prob_fake.size == 0:
        return 0.0
    if method == "mean":
        return float(np.mean(prob_fake))
    if method == "median":
        return float(np.median(prob_fake))
    if method == "max":
        return float(np.max(prob_fake))
    if method == "topk":
        k = min(top_k, int(prob_fake.size))
        return float(np.mean(np.sort(prob_fake)[-k:]))
    if method == "vote":
        return float(np.mean(prob_fake >= frame_threshold))
    raise ValueError(f"unknown aggregate method: {method}")


def _mean_classification_row(per_frame: list[dict], *, threshold: float) -> dict:
    keys = ("logit_real", "logit_fake", "prob_real", "prob_fake", "margin", "entropy", "confidence")
    aggregate = {key: _round4(float(np.mean([row[key] for row in per_frame]))) for key in keys}
    aggregate["pred_label"] = "fake" if aggregate["prob_fake"] >= threshold else "real"
    return aggregate


def empty_score_breakdown(*, threshold: float, frames_sampled: int, frames_without_face: int) -> dict:
    return {
        "schema_version": SCORE_BREAKDOWN_SCHEMA_VERSION,
        "method": "mean_classification_outputs_over_face_frames",
        "threshold": threshold,
        "margin_definition": "logit_fake_minus_logit_real",
        "entropy_log_base": "natural",
        "frames_sampled": frames_sampled,
        "frames_with_face": 0,
        "frames_without_face": frames_without_face,
        "aggregate": None,
        "aggregate_fake_score": None,
        "score_stats": None,
        "frame_votes": None,
        "per_frame": [],
        "per_frame_scores": [],
    }


def build_score_breakdown(
    face_samples: list[dict],
    logits: np.ndarray,
    *,
    threshold: float = 0.5,
    frames_sampled: int,
    frames_without_face: int,
    aggregate: str = "mean",
    top_k: int = 5,
) -> dict:
    per_frame = logits_to_frame_rows(logits, face_samples, threshold=threshold)
    prob_fake_values = np.array([row["prob_fake"] for row in per_frame], dtype=np.float64)
    margin_values = np.array([row["margin"] for row in per_frame], dtype=np.float64)
    entropy_values = np.array([row["entropy"] for row in per_frame], dtype=np.float64)
    logit_real_values = np.array([row["logit_real"] for row in per_frame], dtype=np.float64)
    logit_fake_values = np.array([row["logit_fake"] for row in per_frame], dtype=np.float64)
    prob_real_values = np.array([row["prob_real"] for row in per_frame], dtype=np.float64)

    fake_frame_count = int(np.sum(prob_fake_values >= threshold))
    real_frame_count = int(len(per_frame) - fake_frame_count)
    aggregate_row = _mean_classification_row(per_frame, threshold=threshold)
    fake_score = aggregate_prob_fake(
        prob_fake_values,
        aggregate,
        top_k=top_k,
        frame_threshold=threshold,
    )
    aggregate_row["prob_fake"] = _round4(fake_score)
    aggregate_row["pred_label"] = "fake" if fake_score >= threshold else "real"

    per_frame_scores = [
        {
            "frame_index": row["frame_index"],
            "face_index": row.get("face_index", 0),
            "fake_score": row["prob_fake"],
            **({"bbox": row["bbox"]} if row.get("bbox") else {}),
        }
        for row in per_frame
    ]

    unique_frames_with_face = len({row["frame_index"] for row in per_frame})
    method_label = f"{aggregate}_prob_fake_over_face_crops"
    if aggregate == "topk":
        method_label = f"top{top_k}_mean_prob_fake_over_face_frames"

    return {
        "schema_version": SCORE_BREAKDOWN_SCHEMA_VERSION,
        "method": method_label,
        "aggregate_method": aggregate,
        "top_k": top_k if aggregate == "topk" else None,
        "threshold": threshold,
        "margin_definition": "logit_fake_minus_logit_real",
        "entropy_log_base": "natural",
        "frames_sampled": frames_sampled,
        "frames_with_face": len(per_frame),
        "unique_frames_with_face": unique_frames_with_face,
        "frames_without_face": frames_without_face,
        "multi_face": True,
        "aggregate": aggregate_row,
        "aggregate_fake_score": fake_score,
        "score_stats": {
            "prob_fake": _distribution_stats(prob_fake_values),
            "prob_real": _distribution_stats(prob_real_values),
            "margin": _distribution_stats(margin_values),
            "entropy": _distribution_stats(entropy_values),
            "logit_real": _distribution_stats(logit_real_values),
            "logit_fake": _distribution_stats(logit_fake_values),
        },
        "frame_votes": {
            "fake": fake_frame_count,
            "real": real_frame_count,
        },
        "per_frame": per_frame,
        "per_frame_scores": per_frame_scores,
    }


@torch.no_grad()
def infer_video(
    model: XceptionDetectorLite,
    video_path: Path,
    face_cropper: FaceCropper,
    device: torch.device,
    *,
    threshold: float = 0.5,
    num_frames: int = 32,
    aggregate: str = "mean",
    top_k: int = 5,
) -> dict:
    samples = read_frame_samples(video_path, num_frames=num_frames)
    face_samples: list[dict] = []
    for sample in samples:
        if hasattr(face_cropper, "crop_all"):
            face_entries = face_cropper.crop_all(sample["frame"])
        else:
            crop = face_cropper.crop(sample["frame"])
            face_entries = (
                [{"face_index": 0, "bbox": None, "crop": crop}]
                if crop is not None
                else []
            )
        for entry in face_entries:
            face_samples.append(
                {
                    "frame_index": sample["frame_index"],
                    "face_index": int(entry.get("face_index", 0)),
                    "bbox": entry.get("bbox"),
                    "crop": entry["crop"],
                }
            )

    if not face_samples:
        breakdown = empty_score_breakdown(
            threshold=threshold,
            frames_sampled=len(samples),
            frames_without_face=len(samples),
        )
        breakdown.update(face_cropper.to_metadata())
        return {
            "file": video_path.name,
            "status": face_cropper.no_face_status(),
            "fake_score": None,
            "pred_label": None,
            "frames_used": 0,
            "score_breakdown": breakdown,
        }

    min_faces = max(1, int(getattr(face_cropper.config, "min_sample_faces", 1)))
    unique_frames_with_face = len({sample["frame_index"] for sample in face_samples})
    if unique_frames_with_face < min_faces:
        breakdown = empty_score_breakdown(
            threshold=threshold,
            frames_sampled=len(samples),
            frames_without_face=len(samples) - unique_frames_with_face,
        )
        breakdown.update(face_cropper.to_metadata())
        breakdown["frames_with_face"] = len(face_samples)
        breakdown["unique_frames_with_face"] = unique_frames_with_face
        return {
            "file": video_path.name,
            "status": face_cropper.no_face_status(),
            "fake_score": None,
            "pred_label": None,
            "frames_used": len(face_samples),
            "score_breakdown": breakdown,
        }

    crops = [s["crop"] for s in face_samples]
    batch = frames_to_tensor(crops, device)
    logits = model.forward_logits(batch).detach().cpu().numpy()
    breakdown = build_score_breakdown(
        face_samples,
        logits,
        threshold=threshold,
        frames_sampled=len(samples),
        frames_without_face=len(samples) - unique_frames_with_face,
        aggregate=aggregate,
        top_k=top_k,
    )
    breakdown.update(face_cropper.to_metadata())
    fake_score = breakdown["aggregate_fake_score"]
    pred_label = breakdown["aggregate"]["pred_label"]
    return {
        "file": video_path.name,
        "status": "ok",
        "fake_score": fake_score,
        "pred_label": pred_label,
        "frames_used": len(crops),
        "score_breakdown": breakdown,
    }


def build_item(
    video_path: Path,
    result: dict,
    *,
    run_id: str,
    weights: Path,
    device: torch.device,
    ground_truth_label: str | None,
    model_id: str = "xception/v1.0.0",
) -> dict:
    analyzed_at = datetime.now(timezone.utc).isoformat()
    item = {
        "run_id": run_id,
        "file": video_path.name,
        "source_path": str(video_path.resolve()),
        "ground_truth_label": ground_truth_label,
        "status": result["status"],
        "fake_score": result["fake_score"],
        "pred_label": result["pred_label"],
        "frames_used": result["frames_used"],
        "score_breakdown": result.get("score_breakdown"),
        "model": model_id,
        "weights": str(weights),
        "device": str(device),
        "analyzed_at": analyzed_at,
    }
    if ground_truth_label in {"real", "fake"} and result["status"] == "ok":
        item["correct"] = result["pred_label"] == ground_truth_label
    return item


def write_per_file_json(json_dir: Path, item: dict) -> Path:
    json_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(item["file"]).stem
    out = json_dir / f"{stem}.json"
    out.write_text(json.dumps(item, indent=2), encoding="utf-8")
    return out


def run_directory(
    model: XceptionDetectorLite,
    face_cropper: FaceCropper,
    device: torch.device,
    input_dir: Path,
    ground_truth_label: str | None,
    run_id: str,
    weights: Path,
    per_file_json_dir: Path | None,
    model_id: str = "xception/v1.0.0",
    threshold: float = 0.5,
    num_frames: int = 32,
    aggregate: str = "mean",
    top_k: int = 5,
) -> list[dict]:
    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No mp4 files in {input_dir}")

    items: list[dict] = []
    for video_path in videos:
        result = infer_video(
            model,
            video_path,
            face_cropper,
            device,
            threshold=threshold,
            num_frames=num_frames,
            aggregate=aggregate,
            top_k=top_k,
        )
        item = build_item(
            video_path,
            result,
            run_id=run_id,
            weights=weights,
            device=device,
            ground_truth_label=ground_truth_label,
            model_id=model_id,
        )
        items.append(item)
        if per_file_json_dir is not None:
            write_per_file_json(per_file_json_dir, item)
        print(
            f"{video_path.name}: {item['status']} "
            f"pred={item['pred_label']} fake_score={item['fake_score']}",
            flush=True,
        )
    return items


def compute_metrics(items: list[dict], expected_label: str | None) -> dict:
    ok_items = [x for x in items if x["status"] == "ok"]
    metrics = {
        "total": len(items),
        "ok": len(ok_items),
        "no_face": sum(1 for x in items if x["status"] == "no_face"),
        "expected_label": expected_label,
    }
    if expected_label in {"real", "fake"} and ok_items:
        correct = sum(1 for x in ok_items if x.get("pred_label") == expected_label)
        metrics["accuracy"] = round(correct / len(ok_items), 4)
        metrics["correct"] = correct
        fake_scores = [x["fake_score"] for x in ok_items if x["fake_score"] is not None]
        metrics["avg_fake_score"] = round(float(np.mean(fake_scores)), 4)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Xception video infer (DeepfakeBench weights)")
    parser.add_argument("--weights", default="models/test/video/xception/v1.0.0/xception_best.pth")
    parser.add_argument("--input-dir", required=True, help="Directory with mp4 files")
    parser.add_argument("--label", default=None, choices=["real", "fake"], help="Ground-truth label for eval")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--root", default=".", help="forenShield-ai root")
    parser.add_argument(
        "--per-file-json",
        action="store_true",
        help="write one JSON per video under results/infer/<run_id>/json/",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="fake_score >= threshold => fake")
    parser.add_argument(
        "--aggregate",
        default="mean",
        choices=("mean", "median", "max", "topk", "vote"),
        help="how to aggregate per-frame prob_fake into video fake_score",
    )
    parser.add_argument("--top-k", type=int, default=5, help="k for --aggregate topk")
    parser.add_argument("--num-frames", type=int, default=32, help="frames sampled per video")
    parser.add_argument(
        "--crop-method",
        default="haar",
        choices=("haar", "mediapipe"),
        help="face detector for crop (step-2: mediapipe)",
    )
    parser.add_argument("--crop-padding", type=float, default=0.2, help="bbox padding ratio (step-2: 0.3)")
    parser.add_argument(
        "--crop-square",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="force square face bbox before resize",
    )
    parser.add_argument(
        "--operating",
        action="store_true",
        help="Step 2/3 ops: MediaPipe p0.3 + topk k=5 + threshold 0.78",
    )
    args = parser.parse_args()
    if args.operating:
        args.threshold = DEFAULT_FAKE_THRESHOLD
        args.aggregate = DEFAULT_AGGREGATE
        args.top_k = DEFAULT_TOP_K
        args.num_frames = DEFAULT_NUM_FRAMES
        args.crop_method = DEFAULT_CROP_METHOD
        args.crop_padding = DEFAULT_CROP_PADDING
        args.crop_square = True

    root = Path(args.root).resolve()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = (root / input_dir).resolve()
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = (root / weights).resolve()

    run_id = args.run_id or f"xception-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    infer_dir = root / "results/infer" / run_id
    eval_dir = root / "results/eval" / run_id
    report_dir = root / "results/reports"
    json_dir = infer_dir / "json" if args.per_file_json else None
    infer_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)
    face_cropper = create_face_cropper(
        method=args.crop_method,
        padding=args.crop_padding,
        square=args.crop_square,
    )

    try:
        items = run_directory(
            model,
            face_cropper,
            device,
            input_dir,
            args.label,
            run_id,
            weights,
            json_dir,
            threshold=args.threshold,
            num_frames=args.num_frames,
            aggregate=args.aggregate,
            top_k=args.top_k,
        )
    finally:
        face_cropper.close()
    metrics = compute_metrics(items, args.label)

    pred_path = infer_dir / "predictions.json"
    metrics_path = eval_dir / "metrics.json"
    report_path = report_dir / f"{run_id}_summary.md"

    payload = {
        "run_id": run_id,
        "model": "xception/v1.0.0",
        "threshold": args.threshold,
        "aggregate": args.aggregate,
        "top_k": args.top_k,
        "num_frames": args.num_frames,
        "crop_method": args.crop_method,
        "crop_padding": args.crop_padding,
        "crop_square": args.crop_square,
        "weights": str(weights),
        "input_dir": str(input_dir),
        "device": str(device),
        "items": items,
    }
    pred_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        f"# Xception infer summary — {run_id}",
        "",
        f"- input: `{input_dir}`",
        f"- weights: `{weights}`",
        f"- device: `{device}`",
        f"- total videos: {metrics['total']}",
        f"- ok: {metrics['ok']}",
        f"- no_face: {metrics['no_face']}",
    ]
    if "accuracy" in metrics:
        lines.append(f"- accuracy (expected `{args.label}`): {metrics['accuracy']}")
        lines.append(f"- avg_fake_score: {metrics['avg_fake_score']}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"predictions: {pred_path}")
    if json_dir is not None:
        print(f"per-file json: {json_dir} ({len(list(json_dir.glob('*.json')))} files)")
    print(f"metrics:     {metrics_path}")
    print(f"report:      {report_path}")


if __name__ == "__main__":
    main()
