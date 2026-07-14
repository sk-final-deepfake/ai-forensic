from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.schemas.messaging import (
    AnalysisJobMessage,
    AnalysisResponseMessage,
    AnalysisVideoResultItem,
    PerFrameFaceScoreItem,
    RepresentativeFrameItem,
)
from app.services.visualization_artifacts import build_visualization_artifacts

logger = logging.getLogger(__name__)

_S3_URI = re.compile(r"^s3://([^/]+)/(.+)$")


def _per_frame_scores_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for row in rows:
        frame_index = row.get("frame_index", row.get("frameIndex"))
        score = row.get("fake_score", row.get("prob_fake", row.get("riskScore")))
        if frame_index is None or score is None:
            continue
        scores.append({"frame_index": int(frame_index), "fake_score": float(score)})
    return scores


def per_frame_scores_from_cnn_raw(raw: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    breakdown = raw.get("score_breakdown") or {}
    rows = breakdown.get("per_frame_scores") or breakdown.get("per_frame") or []
    return _per_frame_scores_from_rows(rows)


def per_frame_scores_from_video_item(video: AnalysisVideoResultItem) -> list[dict[str, Any]]:
    face_rows = per_frame_face_scores_from_video_item(video)
    if face_rows:
        return face_rows
    if video.frameRisks:
        return _per_frame_scores_from_rows(
            [
                {
                    "frameIndex": item.frameIndex,
                    "riskScore": item.riskScore,
                }
                for item in video.frameRisks
            ]
        )
    return []


def per_frame_face_scores_from_video_item(video: AnalysisVideoResultItem) -> list[dict[str, Any]]:
    rows = video.perFrameFaceScores or []
    if not rows:
        return []
    scores: list[dict[str, Any]] = []
    for row in rows:
        entry: dict[str, Any] = {
            "frame_index": int(row.frameIndex),
            "face_index": int(row.faceIndex),
            "fake_score": float(row.riskScore),
        }
        if row.bbox is not None:
            entry["bbox"] = {
                "x": int(row.bbox.x),
                "y": int(row.bbox.y),
                "w": int(row.bbox.w),
                "h": int(row.bbox.h),
            }
        scores.append(entry)
    return scores


def build_visualization_payload(
    *,
    video_path: Path,
    per_frame_scores: list[dict[str, Any]],
    evidence_id: int,
    analysis_request_id: int,
    work_dir: Path,
) -> dict[str, Any] | None:
    if not per_frame_scores or not video_path.is_file():
        return None

    viz = build_visualization_artifacts(
        video_path=video_path,
        per_frame_scores=per_frame_scores,
        evidence_id=evidence_id,
        analysis_request_id=analysis_request_id,
        work_dir=work_dir,
    )
    if viz is None:
        return None

    return {
        "representativeFrames": viz.representative_frames,
        "overlayVideoUrl": viz.overlay_video_url,
    }


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    match = _S3_URI.match(uri.strip())
    if not match:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return match.group(1), match.group(2)


def download_messaging_job_video(job: AnalysisJobMessage, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(job.filePath or "evidence.mp4").suffix or ".mp4"
    local_path = dest_dir / f"evidence_{job.evidenceId}_{job.analysisRequestId}{suffix}"

    if job.presignedDownloadUrl:
        with httpx.Client(timeout=600.0) as client:
            response = client.get(job.presignedDownloadUrl)
            response.raise_for_status()
            local_path.write_bytes(response.content)
        return local_path

    bucket = (job.s3Bucket or os.getenv("S3_EVIDENCE_BUCKET") or os.getenv("S3_ARTIFACT_BUCKET") or "").strip()
    key = (job.s3ObjectKey or job.filePath or "").strip().lstrip("/")
    if not bucket and job.filePath.startswith("s3://"):
        bucket, key = _parse_s3_uri(job.filePath)
    if not bucket or not key:
        parsed = urlparse(job.filePath or "")
        if parsed.scheme in ("http", "https"):
            with httpx.Client(timeout=600.0) as client:
                response = client.get(job.filePath)
                response.raise_for_status()
                local_path.write_bytes(response.content)
            return local_path
        raise ValueError(
            f"Missing download location for evidenceId={job.evidenceId} bucket={bucket!r} key={key!r}"
        )

    import boto3

    region = job.s3Region or os.getenv("AWS_REGION", "ap-northeast-2")
    client = boto3.client("s3", region_name=region)
    client.download_file(bucket, key, str(local_path))
    return local_path


def _video_item_has_visualization(video: AnalysisVideoResultItem) -> bool:
    if video.overlayVideoUrl:
        return True
    artifacts = video.modelOverlayArtifacts or []
    if any(artifact.overlayVideoUrl for artifact in artifacts):
        return True
    return bool(video.representativeFrames)


def _gpu_already_provided_module_overlays(video: AnalysisVideoResultItem) -> bool:
    """GPU gateway may attach per-module MP4s; do not replace with legacy CNN-only overlay."""
    artifacts = video.modelOverlayArtifacts or []
    if any((artifact.overlayVideoUrl or "").strip() for artifact in artifacts):
        return True
    timelines = video.moduleTimelines or []
    return any(
        timeline.module in ("temporal", "optical") and (timeline.overlayVideoUrl or "").strip()
        for timeline in timelines
    )


def _gpu_ran_module_pipeline(video: AnalysisVideoResultItem) -> bool:
    """True when GPU Method-B already produced deepfake module timelines.

    Even if overlay encode OOM'd, falling back to EKS legacy overlay.mp4 takes
    many minutes and races the serial prefetch=1 queue — skip it.
    """
    modules = {
        str(timeline.module).strip().lower()
        for timeline in (video.moduleTimelines or [])
        if getattr(timeline, "module", None)
    }
    return "cnn" in modules and ("temporal" in modules or "optical" in modules)


def _apply_visualization_payload(
    video: AnalysisVideoResultItem,
    payload: dict[str, Any],
) -> AnalysisVideoResultItem:
    representative_frames = [
        RepresentativeFrameItem(**row) for row in payload.get("representativeFrames") or []
    ]
    return video.model_copy(
        update={
            "representativeFrames": representative_frames,
            "overlayVideoUrl": payload.get("overlayVideoUrl"),
        }
    )


def attach_visualization_artifacts(
    job: AnalysisJobMessage,
    response: AnalysisResponseMessage,
) -> AnalysisResponseMessage:
    if response.status != "COMPLETED" or not response.results:
        return response

    video = response.results[0]
    if _gpu_already_provided_module_overlays(video):
        logger.info(
            "Visualization skipped: GPU module overlays present analysisRequestId=%s evidenceId=%s",
            job.analysisRequestId,
            job.evidenceId,
        )
        return response

    if _gpu_ran_module_pipeline(video):
        logger.info(
            "Visualization skipped: GPU module pipeline present (no EKS legacy fallback) "
            "analysisRequestId=%s evidenceId=%s",
            job.analysisRequestId,
            job.evidenceId,
        )
        return response

    has_face_scores = bool(video.perFrameFaceScores)
    if _video_item_has_visualization(video) and not has_face_scores:
        return response

    per_frame_scores = per_frame_scores_from_video_item(video)
    if not per_frame_scores:
        logger.info(
            "Visualization skipped: no per-frame scores analysisRequestId=%s evidenceId=%s",
            job.analysisRequestId,
            job.evidenceId,
        )
        return response

    try:
        with tempfile.TemporaryDirectory(prefix="forenshield-viz-") as tmp:
            work_dir = Path(tmp)
            video_path = download_messaging_job_video(job, work_dir)
            payload = build_visualization_payload(
                video_path=video_path,
                per_frame_scores=per_frame_scores,
                evidence_id=job.evidenceId,
                analysis_request_id=job.analysisRequestId,
                work_dir=work_dir / "visualization",
            )
            if payload is None:
                logger.info(
                    "Visualization produced no artifacts analysisRequestId=%s evidenceId=%s",
                    job.analysisRequestId,
                    job.evidenceId,
                )
                return response

            updated_video = _apply_visualization_payload(video, payload)
            logger.info(
                "Visualization attached analysisRequestId=%s evidenceId=%s frames=%s overlay=%s",
                job.analysisRequestId,
                job.evidenceId,
                len(updated_video.representativeFrames or []),
                bool(updated_video.overlayVideoUrl),
            )
            return response.model_copy(update={"results": [updated_video, *response.results[1:]]})
    except Exception:
        logger.exception(
            "Visualization enrichment failed analysisRequestId=%s evidenceId=%s",
            job.analysisRequestId,
            job.evidenceId,
        )
        return response
