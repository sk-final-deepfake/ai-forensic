"""TimeSformer temporal overlay: per-window risk tint on video frames."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .overlay_common import draw_temporal_band, format_timestamp, save_jpeg


@dataclass
class TemporalWindowRisk:
    start_frame: int
    end_frame: int
    score: float

    @property
    def mid_frame(self) -> int:
        return int((self.start_frame + self.end_frame) / 2)


def parse_timesformer_windows(mil_detail: dict[str, Any]) -> list[TemporalWindowRisk]:
    rows: list[TemporalWindowRisk] = []
    for w in mil_detail.get("top_windows", []):
        rows.append(
            TemporalWindowRisk(
                start_frame=int(w.get("start", 0)),
                end_frame=int(w.get("end", 0)),
                score=float(w.get("score", 0.0)),
            )
        )
    return rows


def risk_for_frame(frame_idx: int, windows: list[TemporalWindowRisk]) -> float:
    best = 0.0
    for w in windows:
        if w.start_frame <= frame_idx < w.end_frame:
            best = max(best, w.score)
    return best


def build_temporal_risk_timeline(
    total_frames: int,
    per_window_probs: list[float],
    per_window_meta: list[dict[str, Any]],
) -> list[TemporalWindowRisk]:
    windows: list[TemporalWindowRisk] = []
    for score, meta in zip(per_window_probs, per_window_meta):
        windows.append(
            TemporalWindowRisk(
                start_frame=int(meta["frame_index_start"]),
                end_frame=int(meta["frame_index_end"]),
                score=float(score),
            )
        )
    return windows


def render_temporal_overlay_video(
    video_path: Path,
    windows: list[TemporalWindowRisk],
    out_path: Path,
    *,
    max_sec: float | None = 60.0,
    sample_every: int = 1,
) -> tuple[Path, list[dict]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = total
    if max_sec is not None and fps > 0:
        max_frames = min(total, int(max_sec * fps))

    frames_out: list[np.ndarray] = []
    timeline: list[dict] = []
    idx = 0
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % sample_every == 0:
            risk = risk_for_frame(idx, windows)
            painted = draw_temporal_band(frame, risk)
            frames_out.append(painted)
            timeline.append({"frameNumber": idx, "timeSec": round(idx / fps, 3), "temporalRisk": round(risk, 4)})
        idx += 1
    cap.release()

    from .overlay_common import write_mp4

    write_mp4(frames_out, out_path, fps=fps / sample_every)
    return out_path, timeline


def export_temporal_representatives(
    video_path: Path,
    windows: list[TemporalWindowRisk],
    out_dir: Path,
    *,
    fps: float,
    max_frames: int = 3,
) -> list[dict]:
    if not windows:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    picked = sorted(windows, key=lambda w: w.score, reverse=True)[:max_frames]
    payload: list[dict] = []
    try:
        for i, win in enumerate(picked):
            cap.set(cv2.CAP_PROP_POS_FRAMES, win.mid_frame)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            painted = draw_temporal_band(frame, win.score)
            path = save_jpeg(painted, out_dir / f"timesformer_rep_{i:02d}.jpg")
            t = win.mid_frame / fps if fps > 0 else 0.0
            payload.append(
                {
                    "timeSec": round(t, 3),
                    "timestamp": format_timestamp(t),
                    "frameNumber": win.mid_frame,
                    "score": round(win.score, 4),
                    "imagePath": str(path),
                    "heatmapImagePath": None,
                    "module": "temporal",
                    "modelName": "TimeSformer",
                }
            )
    finally:
        cap.release()
    return payload
