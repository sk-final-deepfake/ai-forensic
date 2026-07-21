"""TruFor spatial overlay: localization map -> heatmap / tamper bboxes on frame."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .overlay_common import blend_heatmap, colormap_from_scalar_map, format_timestamp, risk_to_bgr, save_jpeg


@dataclass
class TruForFrameArtifact:
    frame_number: int
    time_sec: float
    score: float
    frame_path: Path | None
    npz_path: Path
    overlay_bgr: np.ndarray
    heatmap_bgr: np.ndarray


@dataclass(frozen=True)
class TamperBBox:
    x: int
    y: int
    w: int
    h: int
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "w": int(self.w),
            "h": int(self.h),
            "score": round(float(self.score), 4),
        }


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


def tamper_map_to_bboxes(
    tamper_map: np.ndarray,
    frame_w: int,
    frame_h: int,
    *,
    threshold: float | None = None,
    min_area_ratio: float = 0.0008,
    max_boxes: int = 6,
    pad_ratio: float = 0.04,
) -> list[TamperBBox]:
    """Connected-component bboxes from TruFor localization map (pixel-level → rectangles)."""
    arr = np.asarray(tamper_map, dtype=np.float32)
    if arr.ndim != 2 or frame_w <= 0 or frame_h <= 0:
        return []

    peak = float(np.max(arr)) if arr.size else 0.0
    if peak <= 1e-6:
        return []

    # Adaptive floor: keep high-confidence regions even when global score is moderate.
    thr = float(threshold) if threshold is not None else max(0.25, 0.45 * peak)
    thr = min(thr, max(0.15, peak * 0.9))

    resized = cv2.resize(arr, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    mask = (resized >= thr).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(16, int(frame_w * frame_h * min_area_ratio))
    pad_x = max(2, int(frame_w * pad_ratio))
    pad_y = max(2, int(frame_h * pad_ratio))

    boxes: list[TamperBBox] = []
    for label in range(1, num):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < min_area or w < 4 or h < 4:
            continue
        region = resized[y : y + h, x : x + w]
        score = float(np.max(region)) if region.size else peak
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(frame_w, x + w + pad_x)
        y1 = min(frame_h, y + h + pad_y)
        boxes.append(TamperBBox(x=x0, y=y0, w=max(1, x1 - x0), h=max(1, y1 - y0), score=score))

    boxes.sort(key=lambda b: b.score, reverse=True)
    return boxes[: max(1, max_boxes)]


def _bbox_area(box: TamperBBox) -> int:
    return max(0, int(box.w)) * max(0, int(box.h))


def _bbox_intersection(a: TamperBBox, b: TamperBBox) -> int:
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x + a.w, b.x + b.w)
    y1 = min(a.y + a.h, b.y + b.h)
    return max(0, x1 - x0) * max(0, y1 - y0)


def pick_localized_bboxes(
    boxes: list[TamperBBox],
    frame_w: int,
    frame_h: int,
    *,
    max_boxes: int = 1,
    max_area_ratio: float = 0.35,
    contain_overlap: float = 0.55,
) -> list[TamperBBox]:
    """Prefer compact localization peaks; drop broad parent blobs (display/API only).

    Among compact boxes (area <= max_area_ratio) the highest local map score wins,
    so a tiny low-score noise blob does not beat the real chest/face peak.
    """
    if not boxes or frame_w <= 0 or frame_h <= 0:
        return []

    frame_area = float(frame_w * frame_h)
    sorted_boxes = sorted(boxes, key=lambda b: (-b.score, _bbox_area(b)))

    picked: list[TamperBBox] = []
    for box in sorted_boxes:
        area = _bbox_area(box)
        if area <= 0:
            continue
        if area / frame_area > max_area_ratio:
            continue

        mostly_inside = any(
            _bbox_intersection(keep, box) / max(float(area), 1e-9) >= contain_overlap for keep in picked
        )
        if mostly_inside:
            continue

        contains_picked = any(
            _bbox_intersection(box, keep) / max(float(_bbox_area(keep)), 1e-9) >= contain_overlap
            for keep in picked
        )
        if contains_picked:
            continue

        picked.append(box)
        if len(picked) >= max(1, max_boxes):
            break
    return picked


def draw_trufor_bboxes(
    frame_bgr: np.ndarray,
    bboxes: list[TamperBBox] | list[dict[str, Any]],
    *,
    label: str = "TruFor",
) -> np.ndarray:
    """Xception-style risk-colored rectangles over suspected tamper regions."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    for raw in bboxes:
        if isinstance(raw, TamperBBox):
            x, y, bw, bh, score = raw.x, raw.y, raw.w, raw.h, raw.score
        else:
            x = int(raw.get("x", 0))
            y = int(raw.get("y", 0))
            bw = int(raw.get("w", 0))
            bh = int(raw.get("h", 0))
            score = float(raw.get("score", 0.0))
        if bw <= 0 or bh <= 0:
            continue
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        color = risk_to_bgr(score)
        thickness = max(2, min(w, h) // 180)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, thickness)
        # Soft fill like Xception heatmap hint (light)
        overlay = out.copy()
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, thickness=-1)
        out = cv2.addWeighted(out, 0.82, overlay, 0.18, 0)
        tag = f"{label} {score:.2f}"
        text_y = max(16, y - 6)
        cv2.putText(
            out,
            tag,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def bboxes_from_npz(
    npz_path: Path,
    frame_w: int,
    frame_h: int,
    *,
    threshold: float | None = None,
) -> tuple[list[TamperBBox], float]:
    """Load TruFor NPZ → real localization bboxes only.

    Flat / score-only maps return empty boxes (no invented center-frame box).
    DET score is still returned when present so timeline risk can stay.
    """
    tamper_map, score = load_trufor_map(npz_path)
    boxes = tamper_map_to_bboxes(tamper_map, frame_w, frame_h, threshold=threshold)
    # Intentionally no center / full-frame fallback when localization is missing.
    return boxes, score


def overlay_trufor_on_frame(
    frame_bgr: np.ndarray,
    npz_path: Path,
    *,
    alpha: float = 0.45,
    style: str = "bbox",
) -> tuple[np.ndarray, np.ndarray, float]:
    tamper_map, score = load_trufor_map(npz_path)
    heatmap = colormap_from_scalar_map(tamper_map)
    if style == "heatmap":
        blended = blend_heatmap(frame_bgr, heatmap, alpha=alpha)
        return blended, heatmap, score

    h, w = frame_bgr.shape[:2]
    boxes = pick_localized_bboxes(tamper_map_to_bboxes(tamper_map, w, h), w, h, max_boxes=5)
    blended = draw_trufor_bboxes(frame_bgr, boxes)
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
