"""Xception video deepfake inference — models/test/video/xception weights."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("gpu_worker.xception")

INPUT_SIZE = 299
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_MODEL_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class XceptionVideoResult:
    model_id: str
    model_version: str
    checkpoint: str
    device: str
    frames_sampled: int
    deepfake_score: float
    confidence_score: float
    frame_scores: list[float]
    frame_indices: list[int]
    video_fps: float
    elapsed_sec: float

    @property
    def frame_timestamps_sec(self) -> list[float]:
        fps = self.video_fps if self.video_fps > 0 else 25.0
        return [idx / fps for idx in self.frame_indices]

    def to_dict(self) -> dict[str, Any]:
        return {
            "modelId": self.model_id,
            "modelVersion": self.model_version,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "framesSampled": self.frames_sampled,
            "deepfakeScore": round(self.deepfake_score, 4),
            "confidenceScore": round(self.confidence_score, 4),
            "frameScores": [round(s, 4) for s in self.frame_scores],
            "elapsedSec": round(self.elapsed_sec, 3),
        }


def resolve_checkpoint(models_test_dir: Path, explicit: str = "") -> Path:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        raise FileNotFoundError(f"MODEL_CHECKPOINT_PATH not found: {path}")

    base = models_test_dir / "video" / "xception"
    patterns = [
        "v1.0.0/xception_best.pth",
        "v1/xception_best.pth",
        "xception_best.pth",
        "**/xception_best.pth",
        "**/*.pth",
    ]
    for pattern in patterns:
        hits = sorted(base.glob(pattern)) if "**" in pattern else sorted(base.glob(pattern))
        if hits:
            logger.info("Using checkpoint %s", hits[0])
            return hits[0]

    raise FileNotFoundError(
        f"No .pth checkpoint under {base}. "
        "Place weights at models/test/video/xception/xception_best.pth"
    )


def _load_torch_model(checkpoint: Path, device: str):
    import torch

    from gpu_worker.models.xception_ffpp_net import build_xception_model

    cache_key = f"{checkpoint.resolve()}::v5::{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint}")

    model = build_xception_model(state)
    model.eval()
    model.to(device)

    _MODEL_CACHE[cache_key] = model
    return model


def _sample_frame_indices(frame_count: int, fps: float, video_fps: float, max_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    if video_fps <= 0:
        video_fps = 25.0
    step = max(1, int(round(video_fps / max(fps, 0.1))))
    indices = list(range(0, frame_count, step))
    if len(indices) > max_frames:
        stride = max(1, len(indices) // max_frames)
        indices = indices[::stride][:max_frames]
    return indices


def _preprocess_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))


def _fake_probability(logits) -> float:
    import torch

    if logits.ndim == 1 and logits.shape[0] == 1:
        return float(torch.sigmoid(logits[0]).item())
    probs = torch.softmax(logits, dim=-1)
    if probs.shape[-1] == 1:
        return float(probs[0].item())
    return float(probs[-1].item())


def run_xception_video(
    video_path: Path,
    *,
    checkpoint: Path,
    device: str = "cuda",
    sample_fps: float = 1.0,
    max_frames: int = 32,
    model_id: str = "xception",
    model_version: str = "test",
) -> XceptionVideoResult:
    import time
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — falling back to CPU")
        device = "cpu"

    t0 = time.perf_counter()
    model = _load_torch_model(checkpoint, device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    indices = _sample_frame_indices(frame_count, sample_fps, video_fps, max_frames)

    frame_scores: list[float] = []
    frame_indices: list[int] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        tensor = torch.from_numpy(_preprocess_bgr(frame)).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            logits = logits.squeeze()
            frame_scores.append(_fake_probability(logits))
            frame_indices.append(idx)

    cap.release()

    if not frame_scores:
        raise RuntimeError(
            f"No frames sampled from {video_path}. "
            "Video may be AV1/HEVC — upload H.264 mp4 or install ffmpeg on GPU for auto-transcode."
        )

    deepfake_score = float(sum(frame_scores) / len(frame_scores))
    confidence = float(min(0.99, 0.55 + abs(deepfake_score - 0.5) * 0.8))
    elapsed = time.perf_counter() - t0
    logger.info(
        "Xception inference done video=%s frames=%d deepfake=%.4f device=%s elapsed=%.2fs",
        video_path.name,
        len(frame_scores),
        deepfake_score,
        device,
        elapsed,
    )

    return XceptionVideoResult(
        model_id=model_id,
        model_version=model_version,
        checkpoint=str(checkpoint),
        device=device,
        frames_sampled=len(frame_scores),
        deepfake_score=deepfake_score,
        confidence_score=confidence,
        frame_scores=frame_scores,
        frame_indices=frame_indices,
        video_fps=video_fps,
        elapsed_sec=elapsed,
    )


def save_infer_json(result: XceptionVideoResult, out_path: Path, extra: dict[str, Any] | None = None) -> None:
    payload = result.to_dict()
    if extra:
        payload.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
