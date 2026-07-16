"""Shared helpers for forgery visualization overlays."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def format_timestamp(time_sec: float) -> str:
    total = max(0, int(time_sec))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def colormap_from_scalar_map(map_2d: np.ndarray, *, cmap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Normalize 2D float map [0,1] to BGR heatmap."""
    arr = np.asarray(map_2d, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected HxW map, got shape={arr.shape}")
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi > lo:
        norm = (arr - lo) / (hi - lo)
    else:
        norm = np.zeros_like(arr)
    gray = (norm * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cmap)


def blend_heatmap(frame_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if heatmap_bgr.shape[:2] != (h, w):
        heatmap_bgr = cv2.resize(heatmap_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)


def risk_to_bgr(score: float, *, low: tuple[int, int, int] = (40, 180, 40), high: tuple[int, int, int] = (40, 40, 220)) -> tuple[int, int, int]:
    """Map risk 0..1 to green->red BGR."""
    t = float(np.clip(score, 0.0, 1.0))
    b = int(low[0] * (1 - t) + high[0] * t)
    g = int(low[1] * (1 - t) + high[1] * t)
    r = int(low[2] * (1 - t) + high[2] * t)
    return b, g, r


def draw_temporal_band(frame_bgr: np.ndarray, score: float, *, band_ratio: float = 0.12, alpha: float = 0.55) -> np.ndarray:
    """Bottom band tint for temporal risk."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    band_h = max(8, int(h * band_ratio))
    color = np.array(risk_to_bgr(score), dtype=np.uint8)
    overlay = out.copy()
    overlay[h - band_h : h, :] = color
    mask = np.zeros((h, w), dtype=np.float32)
    mask[h - band_h : h, :] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3, sigmaY=3)
    mask = mask[..., None]
    blended = (out.astype(np.float32) * (1.0 - mask * alpha) + overlay.astype(np.float32) * (mask * alpha)).astype(np.uint8)
    cv2.putText(
        blended,
        f"temporal {score:.2f}",
        (12, h - max(10, band_h // 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return blended


def write_mp4(frames_bgr: list[np.ndarray], out_path: Path, *, fps: float) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames_bgr:
        raise ValueError("no frames to encode")
    h, w = frames_bgr[0].shape[:2]
    tmp = out_path.with_suffix(".tmp.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed: {tmp}")
    try:
        for frame in frames_bgr:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
    finally:
        writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(tmp),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.unlink(missing_ok=True)
    else:
        tmp.rename(out_path)
    return out_path


def save_jpeg(frame_bgr: np.ndarray, path: Path, *, quality: int = 92) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return path
