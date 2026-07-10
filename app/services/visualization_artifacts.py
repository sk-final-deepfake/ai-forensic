from __future__ import annotations

import logging
import os
import shutil
import subprocess
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

logger = logging.getLogger("ai_fastapi.visualization_artifacts")


@dataclass(frozen=True)
class VisualizationArtifacts:
    representative_frames: list[dict[str, Any]]
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


def _face_bboxes_on_frame(cropper: Any, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    if hasattr(cropper, "detect_all_human_face_bboxes"):
        return list(cropper.detect_all_human_face_bboxes(frame))
    if hasattr(cropper, "detect_human_face_bbox"):
        bbox = cropper.detect_human_face_bbox(frame)
        return [bbox] if bbox is not None else []
    return []


def _bbox_from_row(row: dict[str, Any]) -> tuple[int, int, int, int] | None:
    bbox = row.get("bbox")
    if isinstance(bbox, dict):
        keys = ("x", "y", "w", "h")
        if all(key in bbox for key in keys):
            return int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    return None


def _score_map_by_frame(per_frame_scores: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    mapping: dict[int, list[dict[str, Any]]] = {}
    for row in per_frame_scores:
        frame_index = row.get("frame_index")
        score = row.get("fake_score", row.get("prob_fake"))
        if frame_index is None or score is None:
            continue
        mapping.setdefault(int(frame_index), []).append(
            {
                "score": float(score),
                "bbox": _bbox_from_row(row),
                "face_index": int(row.get("face_index", 0)),
            }
        )
    for faces in mapping.values():
        faces.sort(key=lambda item: item["face_index"])
    return mapping


def _score_map(per_frame_scores: list[dict[str, Any]]) -> dict[int, float]:
    mapping: dict[int, float] = {}
    for frame_index, faces in _score_map_by_frame(per_frame_scores).items():
        mapping[frame_index] = max(face["score"] for face in faces)
    return mapping


def _draw_faces_overlay(frame: np.ndarray, faces: list[dict[str, Any]], cropper: Any) -> np.ndarray:
    output = frame
    for face in faces:
        bbox = face.get("bbox")
        if bbox is None:
            detected = _face_bboxes_on_frame(cropper, frame)
            face_index = int(face.get("face_index", 0))
            if face_index < len(detected):
                bbox = detected[face_index]
        if bbox is None:
            continue
        output = _draw_face_overlay(output, bbox, float(face["score"]))
    return output

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


def _pick_representative_rows(per_frame_scores: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows = [
        row
        for row in per_frame_scores
        if row.get("frame_index") is not None and row.get("fake_score", row.get("prob_fake")) is not None
    ]
    rows.sort(key=lambda row: float(row.get("fake_score", row.get("prob_fake"))), reverse=True)
    return rows[:limit]


def _ffmpeg_path() -> str | None:
    return os.getenv("FFMPEG_PATH") or shutil.which("ffmpeg")


def _transcode_overlay_to_h264(source: Path, dest: Path) -> bool:
    """Re-encode OpenCV mp4v output to browser-playable H.264 (yuv420p)."""
    ffmpeg = _ffmpeg_path()
    if not ffmpeg or not source.is_file():
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        os.getenv("AI_VISUALIZATION_OVERLAY_PRESET", "veryfast"),
        "-crf",
        os.getenv("AI_VISUALIZATION_OVERLAY_CRF", "23"),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("Overlay H.264 transcode failed: %s", exc)
        return False

    return dest.is_file() and dest.stat().st_size > 0


def _finalize_overlay_video(raw_path: Path, output_path: Path) -> Path | None:
    if not raw_path.is_file() or raw_path.stat().st_size == 0:
        return None

    if _transcode_overlay_to_h264(raw_path, output_path):
        try:
            raw_path.unlink(missing_ok=True)
        except OSError:
            pass
        return output_path

    logger.warning("ffmpeg unavailable or transcode failed; overlay may not play in browsers: %s", raw_path)
    if output_path != raw_path:
        try:
            shutil.move(str(raw_path), str(output_path))
        except OSError:
            return raw_path
    return output_path


def _maybe_upload(local_path: Path, *, evidence_id: int, analysis_request_id: int, name: str) -> str | None:
    if not s3_upload_enabled():
        logger.warning(
            "S3 visualization upload is disabled: evidenceId=%s analysisRequestId=%s file=%s",
            evidence_id,
            analysis_request_id,
            name,
        )
        return None
    bucket = artifact_bucket()
    if not bucket:
        logger.warning(
            "S3 visualization bucket is not configured: evidenceId=%s analysisRequestId=%s file=%s",
            evidence_id,
            analysis_request_id,
            name,
        )
        return None
    key = f"{artifact_prefix(evidence_id, analysis_request_id)}/{name}"
    url = upload_file(local_path, bucket=bucket, key=key)
    logger.info(
        "Visualization artifact upload result: evidenceId=%s analysisRequestId=%s key=%s uploaded=%s",
        evidence_id,
        analysis_request_id,
        key,
        bool(url),
    )
    return url


def build_visualization_artifacts(
    *,
    video_path: Path,
    per_frame_scores: list[dict[str, Any]],
    evidence_id: int,
    analysis_request_id: int,
    work_dir: Path,
) -> VisualizationArtifacts | None:
    if not _enabled():
        logger.warning(
            "Visualization artifacts are disabled: evidenceId=%s analysisRequestId=%s",
            evidence_id,
            analysis_request_id,
        )
        return None
    if not per_frame_scores:
        logger.warning(
            "Visualization artifacts skipped because frame scores are empty: evidenceId=%s analysisRequestId=%s",
            evidence_id,
            analysis_request_id,
        )
        return None

    logger.info(
        "Building visualization artifacts: evidenceId=%s analysisRequestId=%s scores=%s video=%s",
        evidence_id,
        analysis_request_id,
        len(per_frame_scores),
        video_path,
    )

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
                logger.warning(
                    "Representative frame could not be read: evidenceId=%s analysisRequestId=%s frameIndex=%s",
                    evidence_id,
                    analysis_request_id,
                    frame_index,
                )
                continue
            bbox = _bbox_from_row(row)
            if bbox is None:
                detected = _face_bboxes_on_frame(cropper, frame)
                face_index = int(row.get("face_index", 0))
                if face_index < len(detected):
                    bbox = detected[face_index]
            if bbox is None:
                logger.warning(
                    "Representative frame has no detected face: evidenceId=%s analysisRequestId=%s frameIndex=%s",
                    evidence_id,
                    analysis_request_id,
                    frame_index,
                )
                continue

            frame_path = work_dir / f"frame_{idx:02d}.jpg"
            cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            image_url = _maybe_upload(
                frame_path,
                evidence_id=evidence_id,
                analysis_request_id=analysis_request_id,
                name=frame_path.name,
            )

            representative_frames.append(
                {
                    "timeSec": round(time_sec, 3),
                    "timestamp": _timestamp_label(time_sec),
                    "frameNumber": frame_index,
                    "score": round(risk_score, 6),
                    "imageUrl": image_url,
                }
            )

        overlay_video_url = _build_overlay_video(
            video_path=video_path,
            faces_by_frame=_score_map_by_frame(per_frame_scores),
            cropper=cropper,
            work_dir=work_dir,
            evidence_id=evidence_id,
            analysis_request_id=analysis_request_id,
        )

        if not representative_frames and overlay_video_url is None:
            logger.warning(
                "Visualization artifacts produced no output: evidenceId=%s analysisRequestId=%s",
                evidence_id,
                analysis_request_id,
            )
            return None

        logger.info(
            "Visualization artifacts built: evidenceId=%s analysisRequestId=%s frames=%s overlay=%s",
            evidence_id,
            analysis_request_id,
            len(representative_frames),
            bool(overlay_video_url),
        )
        return VisualizationArtifacts(
            representative_frames=representative_frames,
            overlay_video_url=overlay_video_url,
        )
    finally:
        cropper.close()


def _build_overlay_video(
    *,
    video_path: Path,
    faces_by_frame: dict[int, list[dict[str, Any]]],
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
    raw_overlay_path = work_dir / "overlay_raw.mp4"
    overlay_path = work_dir / "overlay.mp4"
    scored_frame_indices = sorted(faces_by_frame)   # ← faces_by_frame 으로 변경
    nearest_window = max(1, int(fps))
    writer = cv2.VideoWriter(
        str(raw_overlay_path),
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

            faces = faces_by_frame.get(frame_index)
            if faces is None and scored_frame_indices:
                nearest_index = min(scored_frame_indices, key=lambda idx: abs(idx - frame_index))
                if abs(nearest_index - frame_index) <= nearest_window:
                    faces = [
                        {
                            "score": face["score"],
                            "face_index": face["face_index"],
                            "bbox": None,
                        }
                        for face in faces_by_frame.get(nearest_index, [])
                    ]

            if faces:
                frame = _draw_faces_overlay(frame, faces, cropper)

            writer.write(frame)
            frame_index += 1
    finally:
        writer.release()
        cap.release()

    if frame_index == 0 or not raw_overlay_path.is_file():
        logger.warning(
            "Overlay video was not created: evidenceId=%s analysisRequestId=%s frames=%s path=%s",
            evidence_id,
            analysis_request_id,
            frame_index,
            raw_overlay_path,
        )
        return None

    playable_overlay = _finalize_overlay_video(raw_overlay_path, overlay_path)
    if playable_overlay is None:
        return None

    return _maybe_upload(
        playable_overlay,
        evidence_id=evidence_id,
        analysis_request_id=analysis_request_id,
        name=overlay_path.name,
    )
