#!/usr/bin/env python3
"""VideoMAE clip infer (DeepfakeBench-style head on MCG-NJU/videomae-base)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import VideoMAEModel

from video_xception_infer import (
    SCORE_BREAKDOWN_SCHEMA_VERSION,
    _distribution_stats,
    _mean_classification_row,
    _round4,
    build_item,
    classification_row_from_logits,
    compute_metrics,
    crop_face,
    empty_score_breakdown,
    read_frame_samples,
    write_per_file_json,
)

CLIP_FRAMES = 16
CLIP_SIZE = 224
SAMPLE_FRAMES = 32
MAX_CLIPS = 4


class VideoMAEDetectorLite(nn.Module):
    """Matches DeepfakeBench videomae keys: backbone.*, fc_norm.*, head.*."""

    def __init__(self, pretrained_id: str = "MCG-NJU/videomae-base"):
        super().__init__()
        self.backbone = VideoMAEModel.from_pretrained(pretrained_id)
        hidden = self.backbone.config.hidden_size
        self.fc_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 2)
        self.embedding_dim = hidden

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values)
        seq = outputs.last_hidden_state
        return self.fc_norm(seq.mean(dim=1))

    def forward_logits(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(pixel_values))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward_logits(pixel_values), dim=1)[:, 1]


def load_model(weights_path: Path, device: torch.device, pretrained_id: str = "MCG-NJU/videomae-base") -> VideoMAEDetectorLite:
    model = VideoMAEDetectorLite(pretrained_id=pretrained_id).to(device)
    if weights_path.is_file():
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model


def clip_to_tensor(crops: list[np.ndarray], device: torch.device) -> torch.Tensor:
    arr = np.stack(crops, axis=0).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    return tensor


def representation_summary(embedding: np.ndarray, *, export_vector: bool = False) -> dict:
    vec = embedding.astype(np.float64)
    summary = {
        "dim": int(vec.shape[0]),
        "l2_norm": _round4(float(np.linalg.norm(vec))),
        "mean": _round4(float(np.mean(vec))),
        "std": _round4(float(np.std(vec))),
        "min": _round4(float(np.min(vec))),
        "max": _round4(float(np.max(vec))),
    }
    if export_vector:
        summary["vector"] = [_round4(float(x)) for x in vec]
    return summary


def pick_clip_windows(face_samples: list[dict], *, clip_frames: int, max_clips: int) -> list[list[dict]]:
    if len(face_samples) < clip_frames:
        return []
    if len(face_samples) == clip_frames:
        return [face_samples]

    windows: list[list[dict]] = []
    if len(face_samples) <= clip_frames + 4:
        step = max(1, len(face_samples) - clip_frames + 1)
        for start in range(0, len(face_samples) - clip_frames + 1, step):
            windows.append(face_samples[start : start + clip_frames])
            if len(windows) >= max_clips:
                break
        return windows[:max_clips]

    stride = max(1, (len(face_samples) - clip_frames) // max(1, max_clips - 1))
    for start in range(0, len(face_samples) - clip_frames + 1, stride):
        windows.append(face_samples[start : start + clip_frames])
        if len(windows) >= max_clips:
            break
    return windows[:max_clips]


def build_videomae_score_breakdown(
    per_clip: list[dict],
    *,
    threshold: float,
    frames_sampled: int,
    frames_without_face: int,
    clip_frames: int,
    clip_size: int,
    max_clips: int,
) -> dict:
    classification_rows = [
        {key: row[key] for key in ("logit_real", "logit_fake", "prob_real", "prob_fake", "margin", "entropy", "confidence", "pred_label")}
        for row in per_clip
    ]
    aggregate = _mean_classification_row(classification_rows, threshold=threshold)

    prob_fake_values = np.array([row["prob_fake"] for row in per_clip], dtype=np.float64)
    margin_values = np.array([row["margin"] for row in per_clip], dtype=np.float64)
    entropy_values = np.array([row["entropy"] for row in per_clip], dtype=np.float64)
    logit_real_values = np.array([row["logit_real"] for row in per_clip], dtype=np.float64)
    logit_fake_values = np.array([row["logit_fake"] for row in per_clip], dtype=np.float64)
    prob_real_values = np.array([row["prob_real"] for row in per_clip], dtype=np.float64)

    fake_clip_count = int(np.sum(prob_fake_values >= threshold))
    real_clip_count = int(len(per_clip) - fake_clip_count)

    per_clip_scores = [
        {
            "clip_index": row["clip_index"],
            "fake_score": row["prob_fake"],
            "frame_indices": row["frame_indices"],
        }
        for row in per_clip
    ]

    return {
        "schema_version": SCORE_BREAKDOWN_SCHEMA_VERSION,
        "method": "videomae_clip_classification_outputs",
        "threshold": threshold,
        "margin_definition": "logit_fake_minus_logit_real",
        "entropy_log_base": "natural",
        "clip_frames": clip_frames,
        "clip_size": clip_size,
        "max_clips": max_clips,
        "clips_used": len(per_clip),
        "frames_sampled": frames_sampled,
        "frames_with_face": len({idx for row in per_clip for idx in row["frame_indices"]}),
        "frames_without_face": frames_without_face,
        "aggregate": aggregate,
        "aggregate_fake_score": aggregate["prob_fake"],
        "score_stats": {
            "prob_fake": _distribution_stats(prob_fake_values),
            "prob_real": _distribution_stats(prob_real_values),
            "margin": _distribution_stats(margin_values),
            "entropy": _distribution_stats(entropy_values),
            "logit_real": _distribution_stats(logit_real_values),
            "logit_fake": _distribution_stats(logit_fake_values),
        },
        "clip_votes": {
            "fake": fake_clip_count,
            "real": real_clip_count,
        },
        "frame_votes": {
            "fake": fake_clip_count,
            "real": real_clip_count,
        },
        "per_clip": per_clip,
        "per_clip_scores": per_clip_scores,
        "per_frame_scores": per_clip_scores,
    }


def empty_videomae_score_breakdown(
    *,
    threshold: float,
    frames_sampled: int,
    frames_without_face: int,
    clip_frames: int,
    clip_size: int,
    max_clips: int,
) -> dict:
    breakdown = empty_score_breakdown(
        threshold=threshold,
        frames_sampled=frames_sampled,
        frames_without_face=frames_without_face,
    )
    breakdown["method"] = "videomae_clip_classification_outputs"
    breakdown["clip_frames"] = clip_frames
    breakdown["clip_size"] = clip_size
    breakdown["max_clips"] = max_clips
    breakdown["clips_used"] = 0
    breakdown["clip_votes"] = None
    breakdown["per_clip"] = []
    breakdown["per_clip_scores"] = []
    return breakdown


@torch.no_grad()
def infer_video(
    model: VideoMAEDetectorLite,
    video_path: Path,
    face_cascade: cv2.CascadeClassifier,
    device: torch.device,
    *,
    num_frames: int = SAMPLE_FRAMES,
    clip_frames: int = CLIP_FRAMES,
    clip_size: int = CLIP_SIZE,
    max_clips: int = MAX_CLIPS,
    threshold: float = 0.5,
    export_embedding: bool = False,
) -> dict:
    samples = read_frame_samples(video_path, num_frames=num_frames)
    face_samples: list[dict] = []
    for sample in samples:
        crop = crop_face(sample["frame"], face_cascade, size=clip_size)
        if crop is not None:
            face_samples.append({"frame_index": sample["frame_index"], "crop": crop})

    if len(face_samples) < max(4, clip_frames // 2):
        return {
            "file": video_path.name,
            "status": "no_face",
            "fake_score": None,
            "pred_label": None,
            "frames_used": len(face_samples),
            "score_breakdown": empty_videomae_score_breakdown(
                threshold=threshold,
                frames_sampled=len(samples),
                frames_without_face=len(samples) - len(face_samples),
                clip_frames=clip_frames,
                clip_size=clip_size,
                max_clips=max_clips,
            ),
        }

    windows = pick_clip_windows(face_samples, clip_frames=clip_frames, max_clips=max_clips)
    per_clip: list[dict] = []
    for clip_index, window in enumerate(windows):
        crops = [entry["crop"] for entry in window]
        frame_indices = [entry["frame_index"] for entry in window]
        clip = clip_to_tensor(crops, device)
        logits = model.forward_logits(clip).detach().cpu().numpy().reshape(-1)
        features = model.forward_features(clip).detach().cpu().numpy().reshape(-1)
        row = classification_row_from_logits(float(logits[0]), float(logits[1]), threshold=threshold)
        row.update(
            {
                "clip_index": clip_index,
                "frame_indices": frame_indices,
                "clip_start_frame": frame_indices[0],
                "clip_end_frame": frame_indices[-1],
                "representation": representation_summary(features, export_vector=export_embedding),
            }
        )
        per_clip.append(row)

    breakdown = build_videomae_score_breakdown(
        per_clip,
        threshold=threshold,
        frames_sampled=len(samples),
        frames_without_face=len(samples) - len(face_samples),
        clip_frames=clip_frames,
        clip_size=clip_size,
        max_clips=max_clips,
    )
    fake_score = breakdown["aggregate_fake_score"]
    pred_label = breakdown["aggregate"]["pred_label"]
    return {
        "file": video_path.name,
        "status": "ok",
        "fake_score": fake_score,
        "pred_label": pred_label,
        "frames_used": breakdown["frames_with_face"],
        "score_breakdown": breakdown,
    }


def run_directory(
    model: VideoMAEDetectorLite,
    face_cascade: cv2.CascadeClassifier,
    device: torch.device,
    input_dir: Path,
    ground_truth_label: str | None,
    run_id: str,
    weights: Path,
    per_file_json_dir: Path | None,
    model_id: str = "videomae/v1.0.0",
    *,
    threshold: float = 0.5,
    export_embedding: bool = False,
    clip_frames: int = CLIP_FRAMES,
    max_clips: int = MAX_CLIPS,
) -> list[dict]:
    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No mp4 files in {input_dir}")

    items: list[dict] = []
    for video_path in videos:
        result = infer_video(
            model,
            video_path,
            face_cascade,
            device,
            clip_frames=clip_frames,
            max_clips=max_clips,
            threshold=threshold,
            export_embedding=export_embedding,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="VideoMAE video infer")
    parser.add_argument("--weights", default="models/test/video/videomae/v1.0.0/videomae_finetuned.pth")
    parser.add_argument("--pretrained-id", default="MCG-NJU/videomae-base")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--label", default=None, choices=["real", "fake"])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--root", default=".")
    parser.add_argument("--per-file-json", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5, help="prob_fake >= threshold => fake")
    parser.add_argument("--clip-frames", type=int, default=CLIP_FRAMES)
    parser.add_argument("--max-clips", type=int, default=MAX_CLIPS)
    parser.add_argument(
        "--export-embedding",
        action="store_true",
        help="include full 768-d embedding vector in per_clip.representation.vector",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = (root / input_dir).resolve()
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = (root / weights).resolve()

    run_id = args.run_id or f"videomae-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    infer_dir = root / "results/infer" / run_id
    eval_dir = root / "results/eval" / run_id
    json_dir = infer_dir / "json" if args.per_file_json else None
    infer_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device, pretrained_id=args.pretrained_id)
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
        model_id="videomae/v1.0.0",
        threshold=args.threshold,
        export_embedding=args.export_embedding,
        clip_frames=args.clip_frames,
        max_clips=args.max_clips,
    )
    metrics = compute_metrics(items, args.label)

    payload = {
        "run_id": run_id,
        "model": "videomae/v1.0.0",
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
