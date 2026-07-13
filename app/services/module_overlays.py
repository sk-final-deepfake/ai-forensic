from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services.visualization_artifacts import (
    _build_overlay_video,
    _enabled,
    _finalize_overlay_video,
    _maybe_upload,
    _overlay_max_seconds,
    _overlay_yunet_threshold,
    _risk_colormap_bgr,
    _score_map_by_frame,
)
from app.core.paths import ensure_infer_scripts_on_path

logger = logging.getLogger("ai_fastapi.module_overlays")

MODULE_META = {
    "cnn": {
        "key": "deepfake:cnn",
        "label": "Xception",
        "description": "프레임별 얼굴 경계와 위험 점수를 영상 위에 표시합니다.",
    },
    "temporal": {
        "key": "deepfake:temporal",
        "label": "TimeSformer",
        "description": "시계열 이상이 감지된 클립 구간을 영상 위에 표시합니다.",
    },
    "optical": {
        "key": "deepfake:optical",
        "label": "GMFlow",
        "description": "연속 프레임쌍의 optical flow 이상을 영상 위에 표시합니다.",
    },
}


@dataclass(frozen=True)
class ModuleOverlaySet:
    """Per-module overlay URLs + flat FE artifact list. Legacy CNN also maps to overlay_video_url."""

    overlay_by_module: dict[str, str | None]
    model_overlay_artifacts: list[dict[str, Any]]
    legacy_cnn_overlay_url: str | None


def build_module_overlay_set(
    *,
    video_path: Path,
    evidence_id: int,
    analysis_request_id: int,
    work_dir: Path,
    cnn_per_frame_scores: list[dict[str, Any]] | None = None,
    clip_risks: list[dict[str, Any]] | None = None,
    pair_risks: list[dict[str, Any]] | None = None,
) -> ModuleOverlaySet:
    if not _enabled():
        return _empty_set()

    work_dir.mkdir(parents=True, exist_ok=True)
    ensure_infer_scripts_on_path()
    from face_crop import create_face_cropper

    cropper = create_face_cropper(
        method="yunet",
        padding=0.3,
        square=True,
        human_only=True,
        yunet_score_threshold=_overlay_yunet_threshold(),
    )

    urls: dict[str, str | None] = {"cnn": None, "temporal": None, "optical": None}
    try:
        if cnn_per_frame_scores:
            urls["cnn"] = _build_overlay_video(
                video_path=video_path,
                faces_by_frame=_score_map_by_frame(cnn_per_frame_scores),
                cropper=cropper,
                work_dir=work_dir / "cnn",
                evidence_id=evidence_id,
                analysis_request_id=analysis_request_id,
                upload_name="overlay_cnn.mp4",
            )

        urls["temporal"] = _build_segment_overlay_video(
            video_path=video_path,
            frame_scores=_clip_risks_to_frame_scores(clip_risks or []),
            work_dir=work_dir / "temporal",
            evidence_id=evidence_id,
            analysis_request_id=analysis_request_id,
            filename="overlay_temporal.mp4",
            banner_label="TimeSformer",
        )
        urls["optical"] = _build_segment_overlay_video(
            video_path=video_path,
            frame_scores=_pair_risks_to_frame_scores(pair_risks or []),
            work_dir=work_dir / "optical",
            evidence_id=evidence_id,
            analysis_request_id=analysis_request_id,
            filename="overlay_optical.mp4",
            banner_label="GMFlow",
        )
    finally:
        cropper.close()

    artifacts = [
        {
            "key": MODULE_META[module]["key"],
            "category": "deepfake",
            "label": MODULE_META[module]["label"],
            "overlayVideoUrl": urls.get(module),
            "status": "ready" if urls.get(module) else "pending",
            "description": MODULE_META[module]["description"],
        }
        for module in ("cnn", "temporal", "optical")
    ]
    return ModuleOverlaySet(
        overlay_by_module=urls,
        model_overlay_artifacts=artifacts,
        legacy_cnn_overlay_url=urls.get("cnn"),
    )


def _empty_set() -> ModuleOverlaySet:
    artifacts = [
        {
            "key": MODULE_META[module]["key"],
            "category": "deepfake",
            "label": MODULE_META[module]["label"],
            "overlayVideoUrl": None,
            "status": "pending",
            "description": MODULE_META[module]["description"],
        }
        for module in ("cnn", "temporal", "optical")
    ]
    return ModuleOverlaySet({}, artifacts, None)


def _clip_risks_to_frame_scores(clip_risks: list[dict[str, Any]]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for row in clip_risks:
        start = row.get("startFrameIndex", row.get("clip_start_frame", row.get("start_frame")))
        end = row.get("endFrameIndex", row.get("clip_end_frame", row.get("end_frame")))
        score = row.get("riskScore", row.get("fake_score", row.get("prob_fake")))
        if start is None or end is None or score is None:
            continue
        for frame_index in range(int(start), int(end) + 1):
            scores[frame_index] = max(scores.get(frame_index, 0.0), float(score))
    return scores


def _pair_risks_to_frame_scores(pair_risks: list[dict[str, Any]]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for row in pair_risks:
        score = row.get("riskScore", row.get("fake_score"))
        if score is None:
            continue
        for key in ("frameIndexA", "frameIndexB", "frame_index_a", "frame_index_b"):
            idx = row.get(key)
            if idx is None:
                continue
            frame_index = int(idx)
            scores[frame_index] = max(scores.get(frame_index, 0.0), float(score))
    return scores


def _build_segment_overlay_video(
    *,
    video_path: Path,
    frame_scores: dict[int, float],
    work_dir: Path,
    evidence_id: int,
    analysis_request_id: int,
    filename: str,
    banner_label: str,
) -> str | None:
    if not frame_scores:
        return None
    if os.getenv("AI_VISUALIZATION_OVERLAY", "1").lower() in {"0", "false", "no"}:
        return None

    work_dir.mkdir(parents=True, exist_ok=True)
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
    raw_path = work_dir / f"raw_{filename}"
    out_path = work_dir / filename
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return None

    frame_index = 0
    try:
        while frame_index < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            score = frame_scores.get(frame_index)
            if score is not None and score > 0:
                frame = _draw_score_banner(frame, score=float(score), label=banner_label)
            writer.write(frame)
            frame_index += 1
    finally:
        writer.release()
        cap.release()

    if frame_index == 0 or not raw_path.is_file():
        return None

    playable = _finalize_overlay_video(raw_path, out_path)
    if playable is None:
        return None
    return _maybe_upload(
        playable,
        evidence_id=evidence_id,
        analysis_request_id=analysis_request_id,
        name=filename,
    )


def _draw_score_banner(frame: np.ndarray, *, score: float, label: str) -> np.ndarray:
    output = frame.copy()
    color = _risk_colormap_bgr(score)
    overlay = output.copy()
    h, w = output.shape[:2]
    band = max(28, h // 14)
    cv2.rectangle(overlay, (0, 0), (w, band), color, thickness=-1)
    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), color, thickness=max(2, w // 180))
    mixed = cv2.addWeighted(output, 0.72, overlay, 0.28, 0)
    text = f"{label}  risk={score:.2f}"
    cv2.putText(
        mixed,
        text,
        (12, max(20, band - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return mixed
