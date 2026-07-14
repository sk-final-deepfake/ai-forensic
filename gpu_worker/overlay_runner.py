"""On-demand module overlay generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from app.services.module_overlays import build_single_module_overlay
from gpu_worker.config import WorkerConfig
from gpu_worker.inference_runner import _utc_now
from gpu_worker.schemas import OverlayJobMessage, OverlayResultMessage

logger = logging.getLogger("gpu_worker.overlay")

ProgressCallback = Callable[[int, str | None], None]


def run_overlay_job(
    job: OverlayJobMessage,
    local_path: Path,
    cfg: WorkerConfig,
    on_progress: ProgressCallback | None = None,
) -> OverlayResultMessage:
    def report(percent: int, message: str | None = None) -> None:
        if on_progress is None:
            return
        try:
            on_progress(max(0, min(99, int(percent))), message)
        except Exception:
            logger.exception("Overlay progress callback failed")

    report(10, "오버레이 준비 중")
    work_dir = cfg.work_dir / "overlays" / f"{job.evidenceId}_{job.overlayJobId}_{job.module}"
    report(40, f"{job.module} 오버레이 렌더링 중")

    artifact = build_single_module_overlay(
        module=job.module,
        video_path=local_path,
        evidence_id=job.evidenceId,
        analysis_request_id=job.analysisRequestId,
        work_dir=work_dir,
        cnn_per_frame_scores=None,
        clip_risks=[row.model_dump(mode="json") for row in (job.clipRisks or [])],
        pair_risks=[row.model_dump(mode="json") for row in (job.pairRisks or [])],
        frame_risks=[row.model_dump(mode="json") for row in (job.frameRisks or [])],
    )
    report(90, "오버레이 업로드 정리 중")

    url = None if artifact is None else artifact.get("overlayVideoUrl")
    if not url:
        return OverlayResultMessage(
            overlayJobId=job.overlayJobId,
            analysisRequestId=job.analysisRequestId,
            evidenceId=job.evidenceId,
            module=job.module,
            status="FAILED",
            progressPercent=100,
            analyzedAt=_utc_now(),
            errorCode="OVERLAY_EMPTY",
            message="오버레이를 생성할 점수 데이터가 없거나 렌더링에 실패했습니다.",
        )

    return OverlayResultMessage(
        overlayJobId=job.overlayJobId,
        analysisRequestId=job.analysisRequestId,
        evidenceId=job.evidenceId,
        module=job.module,
        status="COMPLETED",
        progressPercent=100,
        overlayVideoUrl=str(url),
        analyzedAt=_utc_now(),
        message=f"{job.module} 오버레이 생성 완료",
    )
