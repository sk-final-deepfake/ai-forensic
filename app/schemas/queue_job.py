from typing import Any, Literal

from pydantic import BaseModel, Field


FileType = Literal["video"]


class FrameAnalysisSpec(BaseModel):
    extractionIntervalSec: float | None = None
    highRiskFrameScoreThreshold: float | None = None
    minSuspiciousSegmentSec: float | None = None
    pixelFormat: str | None = None
    imageEncoding: str | None = None
    sampleTimestampsSec: list[float] | None = None


class AnalysisJobMessage(BaseModel):
    """Backend → AI worker (RabbitMQ). Matches backend AnalysisJobMessage."""

    analysisRequestId: int
    evidenceId: int
    fileType: FileType = "video"
    filePath: str | None = None
    s3ObjectKey: str | None = None
    s3Bucket: str | None = None
    s3Region: str | None = None
    presignedDownloadUrl: str | None = None
    originalHash: str | None = None
    originalSha256: str | None = None
    caseName: str | None = None
    requestedAt: str | None = None
    frameAnalysis: FrameAnalysisSpec | None = None
    localVideoPath: str | None = Field(
        default=None,
        description="Dev-only override when publishing test jobs manually.",
    )

    model_config = {"extra": "ignore"}
