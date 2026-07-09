"""Shared clip sampling / score breakdown for video transformer detectors."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from video_xception_infer import (
    SCORE_BREAKDOWN_SCHEMA_VERSION,
    _distribution_stats,
    _mean_classification_row,
    _round4,
    build_item,
    classification_row_from_logits,
    crop_face,
    empty_score_breakdown,
    read_frame_samples,
    write_per_file_json,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SAMPLE_FRAMES = 32
MAX_CLIPS = 4


def normalize_face_crops(crops: list[np.ndarray]) -> np.ndarray:
    arr = np.stack(crops, axis=0).astype(np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


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


def build_clip_score_breakdown(
    per_clip: list[dict],
    *,
    method: str,
    threshold: float,
    frames_sampled: int,
    frames_without_face: int,
    clip_frames: int,
    clip_size: int,
    max_clips: int,
) -> dict:
    classification_rows = [
        {
            key: row[key]
            for key in (
                "logit_real",
                "logit_fake",
                "prob_real",
                "prob_fake",
                "margin",
                "entropy",
                "confidence",
                "pred_label",
            )
        }
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
        "method": method,
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
        "clip_votes": {"fake": fake_clip_count, "real": real_clip_count},
        "frame_votes": {"fake": fake_clip_count, "real": real_clip_count},
        "per_clip": per_clip,
        "per_clip_scores": per_clip_scores,
        "per_frame_scores": per_clip_scores,
    }


def empty_clip_score_breakdown(
    *,
    method: str,
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
    breakdown["method"] = method
    breakdown["clip_frames"] = clip_frames
    breakdown["clip_size"] = clip_size
    breakdown["max_clips"] = max_clips
    breakdown["clips_used"] = 0
    breakdown["clip_votes"] = None
    breakdown["per_clip"] = []
    breakdown["per_clip_scores"] = []
    return breakdown


class ClipDetectorProtocol:
    embedding_dim: int

    def forward_logits(self, clip: torch.Tensor) -> torch.Tensor: ...

    def forward_features(self, clip: torch.Tensor) -> torch.Tensor: ...


@torch.no_grad()
def infer_video_clip_model(
    model: ClipDetectorProtocol,
    video_path: Path,
    face_cascade: cv2.CascadeClassifier | None,
    device: torch.device,
    *,
    clip_to_tensor,
    method: str,
    num_frames: int = SAMPLE_FRAMES,
    clip_frames: int,
    clip_size: int,
    max_clips: int = MAX_CLIPS,
    threshold: float = 0.5,
    export_embedding: bool = False,
    face_cropper: object | None = None,
) -> dict:
    samples = read_frame_samples(video_path, num_frames=num_frames)
    face_samples: list[dict] = []
    for sample in samples:
        if face_cropper is not None:
            crop = face_cropper.crop(sample["frame"])
        else:
            crop = crop_face(sample["frame"], face_cascade, size=clip_size)
        if crop is not None:
            face_samples.append({"frame_index": sample["frame_index"], "crop": crop})

    no_face_status = (
        face_cropper.no_face_status()
        if face_cropper is not None and hasattr(face_cropper, "no_face_status")
        else "no_face"
    )
    min_faces = max(4, clip_frames // 2)
    if face_cropper is not None and hasattr(face_cropper, "config"):
        min_faces = max(min_faces, int(getattr(face_cropper.config, "min_sample_faces", 4)))
    if len(face_samples) < min_faces:
        breakdown = empty_clip_score_breakdown(
            method=method,
            threshold=threshold,
            frames_sampled=len(samples),
            frames_without_face=len(samples) - len(face_samples),
            clip_frames=clip_frames,
            clip_size=clip_size,
            max_clips=max_clips,
        )
        if face_cropper is not None and hasattr(face_cropper, "to_metadata"):
            breakdown.update(face_cropper.to_metadata())
        return {
            "file": video_path.name,
            "status": no_face_status,
            "fake_score": None,
            "pred_label": None,
            "frames_used": len(face_samples),
            "score_breakdown": breakdown,
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

    breakdown = build_clip_score_breakdown(
        per_clip,
        method=method,
        threshold=threshold,
        frames_sampled=len(samples),
        frames_without_face=len(samples) - len(face_samples),
        clip_frames=clip_frames,
        clip_size=clip_size,
        max_clips=max_clips,
    )
    return {
        "file": video_path.name,
        "status": "ok",
        "fake_score": breakdown["aggregate_fake_score"],
        "pred_label": breakdown["aggregate"]["pred_label"],
        "frames_used": breakdown["frames_with_face"],
        "score_breakdown": breakdown,
    }


def run_clip_infer_directory(
    model: ClipDetectorProtocol,
    face_cascade: cv2.CascadeClassifier,
    device: torch.device,
    input_dir: Path,
    ground_truth_label: str | None,
    run_id: str,
    weights: Path,
    per_file_json_dir: Path | None,
    model_id: str,
    *,
    clip_to_tensor,
    method: str,
    threshold: float = 0.5,
    export_embedding: bool = False,
    clip_frames: int,
    clip_size: int,
    max_clips: int = MAX_CLIPS,
) -> list[dict]:
    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No mp4 files in {input_dir}")

    items: list[dict] = []
    for video_path in videos:
        result = infer_video_clip_model(
            model,
            video_path,
            face_cascade,
            device,
            clip_to_tensor=clip_to_tensor,
            method=method,
            clip_frames=clip_frames,
            clip_size=clip_size,
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
