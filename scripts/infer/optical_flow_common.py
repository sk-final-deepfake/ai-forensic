"""Shared helpers for optical-flow benchmark inference."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def resize_for_flow(image_rgb: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image_rgb
    scale = max_side / float(longest)
    nh = max(8, int(round(h * scale)))
    nw = max(8, int(round(w * scale)))
    return cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)


def read_video_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def sample_frame_pairs(
    video_path: Path,
    max_pairs: int = 8,
    *,
    max_side: int = 512,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < 2:
        cap.release()
        return []

    if total <= max_pairs + 1:
        indices = list(range(total - 1))
    else:
        step = max(1, (total - 2) // max_pairs)
        indices = [min(i * step, total - 2) for i in range(max_pairs)]

    pairs: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    for idx in indices:
        f1 = read_video_frame(cap, idx)
        f2 = read_video_frame(cap, idx + 1)
        if f1 is not None and f2 is not None:
            pairs.append(
                (
                    resize_for_flow(f1, max_side=max_side),
                    resize_for_flow(f2, max_side=max_side),
                    idx,
                    idx + 1,
                )
            )
    cap.release()
    return pairs


def flow_to_numpy(flow) -> np.ndarray:
    if hasattr(flow, "detach"):
        arr = flow.detach().float().cpu().numpy()
    else:
        arr = np.asarray(flow, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 2:
        arr = np.transpose(arr, (1, 2, 0))
    return arr.astype(np.float32)


def summarize_flow(flow: np.ndarray) -> dict:
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    angle = np.arctan2(fy, fx)
    return {
        "magnitude_mean": round(float(mag.mean()), 6),
        "magnitude_std": round(float(mag.std()), 6),
        "magnitude_max": round(float(mag.max()), 6),
        "magnitude_p95": round(float(np.percentile(mag, 95)), 6),
        "magnitude_median": round(float(np.median(mag)), 6),
        "angle_std": round(float(angle.std()), 6),
        "flow_x_mean": round(float(fx.mean()), 6),
        "flow_y_mean": round(float(fy.mean()), 6),
    }


def aggregate_pair_stats(pair_stats: list[dict]) -> dict:
    if not pair_stats:
        return {}
    keys = [k for k in pair_stats[0].keys() if k not in {"frame_index_a", "frame_index_b"}]
    out: dict = {"pair_count": len(pair_stats)}
    for key in keys:
        values = [row[key] for row in pair_stats]
        out[f"{key}_mean"] = round(float(np.mean(values)), 6)
        out[f"{key}_std"] = round(float(np.std(values)), 6)
    return out
