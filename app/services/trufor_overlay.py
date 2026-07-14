"""TruFor spatial overlay: localization map -> heatmap on frame."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .overlay_common import blend_heatmap, colormap_from_scalar_map, format_timestamp, save_jpeg


@dataclass
class TruForFrameArtifact:
    frame_number: int
    time_sec: float
    score: float
    frame_path: Path | None
    npz_path: Path
    overlay_bgr: np.ndarray
    heatmap_bgr: np.ndarray


def load_trufor_map(npz_path: Path) -> tuple[np.ndarray, float]:
    data = np.load(npz_path)
    if "map" in data:
        m = np.asarray(data["map"], dtype=np.float32)
        score = float(np.max(m))
        return m, score
    if "score" in data:
        score = float(np.asarray(data["score"]).reshape(-1)[0])
        return np.full((64, 64), score, dtype=np.float32), score
    raise KeyError(f"npz missing map/score: {npz_path}")


def overlay_trufor_on_frame(frame_bgr: np.ndarray, npz_path: Path, *, alpha: float = 0.45) -> tuple[np.ndarray, np.ndarray, float]:
    tamper_map, score = load_trufor_map(npz_path)
    heatmap = colormap_from_scalar_map(tamper_map)
    blended = blend_heatmap(frame_bgr, heatmap, alpha=alpha)
    return blended, heatmap, score


def collect_trufor_frame_pairs(frames_dir: Path, npz_dir: Path) -> list[tuple[Path, Path, str]]:
    """Match extracted JPEG frames with TruFor test.py NPZ outputs by stem."""
    pairs: list[tuple[Path, Path, str]] = []
    for npz in sorted(npz_dir.rglob("*.npz")):
        stem = npz.stem
        if "_f" not in stem:
            continue
        jpg = frames_dir / f"{stem}.jpg"
        if not jpg.is_file():
            alt = next(frames_dir.glob(f"{stem}.*"), None)
            if alt is None:
                continue
            jpg = alt
        pairs.append((jpg, npz, stem))
    return pairs


def build_trufor_frame_artifacts(
    frame_npz_pairs: list[tuple[Path, Path]],
    *,
    fps: float,
    alpha: float = 0.45,
) -> list[TruForFrameArtifact]:
    rows: list[TruForFrameArtifact] = []
    for frame_path, npz_path in frame_npz_pairs:
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            continue
        overlay, heatmap, score = overlay_trufor_on_frame(bgr, npz_path, alpha=alpha)
        stem = frame_path.stem
        try:
            frame_idx = int(stem.rsplit("_f", 1)[1])
        except (IndexError, ValueError):
            frame_idx = 0
        time_sec = frame_idx / fps if fps > 0 else 0.0
        rows.append(
            TruForFrameArtifact(
                frame_number=frame_idx,
                time_sec=time_sec,
                score=score,
                frame_path=frame_path,
                npz_path=npz_path,
                overlay_bgr=overlay,
                heatmap_bgr=heatmap,
            )
        )
    return rows


def pick_representative_trufor_frames(artifacts: list[TruForFrameArtifact], *, max_frames: int = 3) -> list[TruForFrameArtifact]:
    return sorted(artifacts, key=lambda x: x.score, reverse=True)[: max(1, max_frames)]


def export_trufor_representatives(
    artifacts: list[TruForFrameArtifact],
    out_dir: Path,
    *,
    max_frames: int = 3,
    include_heatmap: bool = True,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    picked = pick_representative_trufor_frames(artifacts, max_frames=max_frames)
    payload: list[dict] = []
    for i, art in enumerate(picked):
        frame_uri = save_jpeg(art.overlay_bgr, out_dir / f"trufor_rep_{i:02d}_overlay.jpg")
        heat_uri = None
        if include_heatmap:
            heat_uri = save_jpeg(art.heatmap_bgr, out_dir / f"trufor_rep_{i:02d}_heatmap.jpg")
        payload.append(
            {
                "timeSec": round(art.time_sec, 3),
                "timestamp": format_timestamp(art.time_sec),
                "frameNumber": art.frame_number,
                "score": round(art.score, 4),
                "imagePath": str(frame_uri),
                "heatmapImagePath": str(heat_uri) if heat_uri else None,
                "module": "spatial",
                "modelName": "TruFor",
            }
        )
    return payload
