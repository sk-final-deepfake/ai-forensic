from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.paths import ensure_infer_scripts_on_path
from app.services.s3_artifact_upload import (
    artifact_bucket,
    artifact_prefix,
    s3_upload_enabled,
    upload_file,
)


@dataclass(frozen=True)
class VisualizationArtifacts:
    representative_frames: list[dict[str, Any]]
    heatmap_image_url: str | None
    overlay_video_url: str | None


def _enabled() -> bool:
    return os.getenv("AI_VISUALIZATION_ENABLED", "1").lower() not in {"0", "false", "no"}


def _max_representative_frames() -> int:
    return max(1, int(os.getenv("AI_VISUALIZATION_MAX_FRAMES", "3")))


def _overlay_max_seconds() -> float:
    return max(1.0, float(os.getenv("AI_VISUALIZATION_OVERLAY_MAX_SEC", "60")))


def _timestamp_label(seconds: float) -> str:
    total = max(0, int(seconds))
    mm, ss = divmod(total, 60)
    return f"{mm:02d}:{ss:02d}"


def _risk_colormap_bgr(risk: float) -> tuple[int, int, int]:
    value = int(np.clip(risk, 0.0, 1.0) * 255)
    color = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
    return int(color[0]), int(color[1]), int(color[2])


def _face_bbox_on_frame(cropper: Any, frame: np.ndarray) -> tuple[int, int, int, int] | None:
    if hasattr(cropper, "detect_human_face_bbox"):
        return cropper.detect_human_face_bbox(frame)
    return None


def _render_heatmap_layer(
    frame_shape: tuple[int, ...],
    bbox: tuple[int, int, int, int],
    risk_score: float,
) -> np.ndarray:
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    x, y, w, h = bbox
    cx = x + w // 2
    cy = y + h // 2
    radius = max(8, int(max(w, h) * 0.55))
    cv2.circle(mask, (cx, cy), radius, float(np.clip(risk_score, 0.0, 1.0)), thickness=-1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(radius * 0.35, 1.0))
    heat = cv2.applyColorMap(np.clip(mask * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heat


def _draw_face_overlay(frame: np.ndarray, bbox: tuple[int, int, int, int], risk_score: float) -> np.ndarray:
    output = frame.copy()
    x, y, w, h = bbox
    color = _risk_colormap_bgr(risk_score)
    cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    heat = _render_heatmap_layer(output.shape, bbox, risk_score)
    return cv2.addWeighted(output, 0.65, heat, 0.35, 0)


def _read_frame_at_index(video_path: Path, frame_index: int) -> tuple[np.ndarray | None, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None, 0.0
    return frame, frame_index / fps if fps > 0 else 0.0


def _score_map(per_frame_scores: list[dict[str, Any]]) -> dict[int, float]:
    mapping: dict[int, float] = {}
    for row in per_frame_scores:
        frame_index = row.get("frame_index")
        score = row.get("fake_score", row.get("prob_fake"))
        if frame_index is None or score is None:
            continue
        mapping[int(frame_index)] = float(score)
    return mapping


def _pick_representative_rows(per_frame_scores: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows = [
        row
        for row in per_frame_scores
        if row.get("frame_index") is not None and row.get("fake_score", row.get("prob_fake")) is not None
    ]
    rows.sort(key=lambda row: float(row.get("fake_score", row.get("prob_fake"))), reverse=True)
    return rows[:limit]


def _maybe_upload(local_path: Path, *, evidence_id: int, analysis_request_id: int, name: str) -> str | None:
    if not s3_upload_enabled():
        return None
    bucket = artifact_bucket()
    if not bucket:
        return None
    key = f"{artifact_prefix(evidence_id, analysis_request_id)}/{name}"
    return upload_file(local_path, bucket=bucket, key=key)


def build_visualization_artifacts(
    *,
    video_path: Path,
    per_frame_scores: list[dict[str, Any]],
    evidence_id: int,
    analysis_request_id: int,
    work_dir: Path,
) -> VisualizationArtifacts | None:
    if not _enabled() or not per_frame_scores:
        return None

    ensure_infer_scripts_on_path()
    from face_crop import create_face_cropper

    work_dir.mkdir(parents=True, exist_ok=True)
    cropper = create_face_cropper(method="yunet", padding=0.3, square=True, human_only=True)

    representative_rows = _pick_representative_rows(per_frame_scores, _max_representative_frames())
    representative_frames: list[dict[str, Any]] = []

    try:
        for idx, row in enumerate(representative_rows):
            frame_index = int(row["frame_index"])
            risk_score = float(row.get("fake_score", row.get("prob_fake")))
            frame, time_sec = _read_frame_at_index(video_path, frame_index)
            if frame is None:
                continue
            bbox = _face_bbox_on_frame(cropper, frame)
            if bbox is None:
                continue

            frame_path = work_dir / f"frame_{idx:02d}.jpg"
            heatmap_path = work_dir / f"frame_{idx:02d}_heatmap.jpg"
            cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            heat_layer = _render_heatmap_layer(frame.shape, bbox, risk_score)
            cv2.imwrite(str(heatmap_path), heat_layer, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            image_url = _maybe_upload(
                frame_path,
                evidence_id=evidence_id,
                analysis_request_id=analysis_request_id,
                name=frame_path.name,
            )
            heatmap_url = _maybe_upload(
                heatmap_path,
                evidence_id=evidence_id,
                analysis_request_id=analysis_request_id,
                name=heatmap_path.name,
            )

            representative_frames.append(
                {
                    "timeSec": round(time_sec, 3),
                    "timestamp": _timestamp_label(time_sec),
                    "frameNumber": frame_index,
                    "score": round(risk_score, 6),
                    "imageUrl": image_url,
                    "heatmapUrl": heatmap_url,
                }
            )

        heatmap_image_url = representative_frames[0]["heatmapUrl"] if representative_frames else None
        overlay_video_url = _build_overlay_video(
            video_path=video_path,
            score_by_frame=_score_map(per_frame_scores),
            cropper=cropper,
            work_dir=work_dir,
            evidence_id=evidence_id,
            analysis_request_id=analysis_request_id,
        )

        if not representative_frames and overlay_video_url is None:
            return None

        return VisualizationArtifacts(
            representative_frames=representative_frames,
            heatmap_image_url=heatmap_image_url,
            overlay_video_url=overlay_video_url,
        )
    finally:
        cropper.close()


def _build_overlay_video(
    *,
    video_path: Path,
    score_by_frame: dict[int, float],
    cropper: Any,
    work_dir: Path,
    evidence_id: int,
    analysis_request_id: int,
) -> str | None:
    if os.getenv("AI_VISUALIZATION_OVERLAY", "1").lower() in {"0", "false", "no"}:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return None

    max_frames = int(_overlay_max_seconds() * fps)
    overlay_path = work_dir / "overlay.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        return None

    frame_index = 0
    try:
        while frame_index < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            risk = score_by_frame.get(frame_index)
            if risk is not None:
                bbox = _face_bbox_on_frame(cropper, frame)
                if bbox is not None:
                    frame = _draw_face_overlay(frame, bbox, risk)
            writer.write(frame)
            frame_index += 1
    finally:
        writer.release()
        cap.release()

    if frame_index == 0 or not overlay_path.is_file():
        return None

    return _maybe_upload(
        overlay_path,
        evidence_id=evidence_id,
        analysis_request_id=analysis_request_id,
        name=overlay_path.name,
    )
