from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.analysis import AnalysisRequest
from app.schemas.queue_job import AnalysisJobMessage


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def job_to_analysis_request(job: AnalysisJobMessage) -> AnalysisRequest:
    original = job.originalHash or job.originalSha256
    return AnalysisRequest(
        analysisRequestId=job.analysisRequestId,
        evidenceId=job.evidenceId,
        fileType="video",
        filePath=job.filePath or job.s3ObjectKey,
        s3ObjectKey=job.s3ObjectKey or job.filePath,
        presignedDownloadUrl=job.presignedDownloadUrl,
        localVideoPath=job.localVideoPath,
        originalHash=original,
        originalSha256=original,
        requestedAt=job.requestedAt or _utc_now_iso(),
    )
