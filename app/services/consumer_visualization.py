from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.schemas.messaging import (
    AnalysisJobMessage,
    AnalysisResponseMessage,
    AnalysisVideoResultItem,
    FrameRiskItem,
    RepresentativeFrameItem,
)
from app.services.visualization_artifacts import build_visualization_artifacts

logger = logging.getLogger("ai_fastapi.consumer_visualization")


def attach_visualization_artifacts(
    message: AnalysisResponseMessage,
    job: AnalysisJobMessage,
) -> AnalysisResponseMessage:
    """Build overlay/heatmap artifacts in the EKS AI pod before publishing the result."""
    if message.status != "COMPLETED":
        return message

    video_result = _first_video_result(message)
    if video_result is None:
        logger.warning(
            "Visualization skipped because response has no video result: evidenceId=%s analysisRequestId=%s",
            job.evidenceId,
            job.analysisRequestId,
        )
        return message

    if video_result.overlayVideoUrl and video_result.heatmapImageUrl:
        logger.info(
            "Visualization already attached by gateway: evidenceId=%s analysisRequestId=%s",
            job.evidenceId,
            job.analysisRequestId,
        )
        return message

    try:
        with tempfile.TemporaryDirectory(prefix=f"forenshield-viz-{job.evidenceId}-{job.analysisRequestId}-") as tmp:
            tmp_dir = Path(tmp)
            video_path = _download_job_video(job, tmp_dir)
            frame_scores = _frame_scores(video_result)
            if not frame_scores:
                frame_scores = _fallback_frame_scores(
                    video_path,
                    video_result.deepfakeScore
                    if video_result.deepfakeScore is not None
                    else message.riskScore,
                )

            viz = build_visualization_artifacts(
                video_path=video_path,
                per_frame_scores=frame_scores,
                evidence_id=job.evidenceId,
                analysis_request_id=job.analysisRequestId,
                work_dir=tmp_dir / "visualization",
            )
            if viz is None:
                logger.warning(
                    "Visualization builder returned no artifacts before publish: evidenceId=%s analysisRequestId=%s",
                    job.evidenceId,
                    job.analysisRequestId,
                )
                return message

            if viz.representative_frames:
                video_result.representativeFrames = [
                    RepresentativeFrameItem(**row) for row in viz.representative_frames
                ]
            if viz.heatmap_image_url:
                video_result.heatmapImageUrl = viz.heatmap_image_url
            if viz.overlay_video_url:
                video_result.overlayVideoUrl = viz.overlay_video_url

            logger.info(
                "Visualization attached before publish: evidenceId=%s analysisRequestId=%s frames=%s heatmap=%s overlay=%s",
                job.evidenceId,
                job.analysisRequestId,
                len(video_result.representativeFrames or []),
                bool(video_result.heatmapImageUrl),
                bool(video_result.overlayVideoUrl),
            )
    except Exception:
        logger.exception(
            "Visualization attachment failed before publish: evidenceId=%s analysisRequestId=%s",
            job.evidenceId,
            job.analysisRequestId,
        )

    return message


def _first_video_result(message: AnalysisResponseMessage) -> AnalysisVideoResultItem | None:
    return next((item for item in message.results if item.type == "video"), None)


def _download_job_video(job: AnalysisJobMessage, tmp_dir: Path) -> Path:
    suffix = Path(job.filePath or job.s3ObjectKey or "evidence.mp4").suffix or ".mp4"
    local_path = tmp_dir / f"evidence_{job.evidenceId}_{job.analysisRequestId}{suffix}"

    if job.presignedDownloadUrl:
        _download_http(job.presignedDownloadUrl, local_path)
        return local_path

    if job.filePath and job.filePath.startswith(("http://", "https://")):
        _download_http(job.filePath, local_path)
        return local_path

    bucket, key = _s3_location(job)
    if bucket and key:
        _download_s3(bucket, key, local_path, job.s3Region)
        return local_path

    if job.filePath:
        local_candidate = Path(job.filePath)
        if local_candidate.is_file():
            return local_candidate

    raise ValueError(
        f"Missing downloadable evidence source for evidenceId={job.evidenceId} "
        f"analysisRequestId={job.analysisRequestId}"
    )


def _s3_location(job: AnalysisJobMessage) -> tuple[str | None, str | None]:
    if job.filePath and job.filePath.startswith("s3://"):
        parsed = urlparse(job.filePath)
        return parsed.netloc, parsed.path.lstrip("/")

    key = (job.filePath or job.s3ObjectKey or "").strip().lstrip("/")
    bucket = (job.s3Bucket or os.getenv("S3_EVIDENCE_BUCKET") or "").strip()
    if bucket and key:
        return bucket, key
    return None, None


def _download_http(url: str, local_path: Path) -> None:
    with httpx.Client(timeout=600.0) as client:
        response = client.get(url)
        response.raise_for_status()
    local_path.write_bytes(response.content)


def _download_s3(bucket: str, key: str, local_path: Path, region: str | None) -> None:
    import boto3

    client = boto3.client("s3", region_name=region or os.getenv("AWS_REGION", "ap-northeast-2"))
    client.download_file(bucket, key, str(local_path))


def _frame_scores(result: AnalysisVideoResultItem) -> list[dict[str, Any]]:
    candidates: list[FrameRiskItem] = []
    if result.frameRisks:
        candidates.extend(result.frameRisks)
    if result.moduleTimelines:
        for timeline in result.moduleTimelines:
            if timeline.frameRisks:
                candidates.extend(timeline.frameRisks)

    seen: set[int] = set()
    scores: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda row: row.riskScore, reverse=True):
        if item.frameIndex in seen:
            continue
        seen.add(item.frameIndex)
        scores.append(
            {
                "frame_index": item.frameIndex,
                "fake_score": _normalize_score(item.riskScore),
            }
        )
    return scores


def _fallback_frame_scores(video_path: Path, score: float | None) -> list[dict[str, Any]]:
    risk = _normalize_score(score if score is not None else 50.0)
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if frame_count > 0:
            indices = {
                max(0, min(frame_count - 1, int(frame_count * ratio)))
                for ratio in (0.25, 0.5, 0.75)
            }
            return [{"frame_index": idx, "fake_score": risk} for idx in sorted(indices)]
    except Exception:
        logger.exception("Failed to create fallback visualization frame scores from video=%s", video_path)
    return [{"frame_index": 0, "fake_score": risk}]


def _normalize_score(score: float) -> float:
    value = float(score)
    if value > 1.0:
        value /= 100.0
    return max(0.0, min(1.0, value))
