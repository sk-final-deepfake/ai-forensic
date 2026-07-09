from typing import Literal

from pydantic import BaseModel, Field


FileType = Literal["video", "audio", "image"]


class AnalysisRequest(BaseModel):
    analysisRequestId: int = Field(..., examples=[101])
    evidenceId: int | None = Field(default=None, examples=[15])
    fileId: int | None = Field(default=None, examples=[15])
    caseId: int | None = Field(default=None, examples=[3])
    fileType: FileType = Field(..., examples=["video"])
    filePath: str | None = Field(default=None, examples=["cases/3/15/copy/evidence.mp4"])
    s3ObjectKey: str | None = Field(default=None, examples=["original-files/3/15/original.mp4"])
    presignedDownloadUrl: str | None = Field(default=None, examples=["https://example.com/presigned-url"])
    localVideoPath: str | None = Field(
        default=None,
        description="Dev-only local mp4 path (skips download). Not used in production queue.",
    )
    originalSha256: str | None = Field(default=None, examples=["abc123..."])
    originalHash: str | None = Field(default=None, examples=["abc123..."])
    requestedAt: str = Field(..., examples=["2026-06-10T10:30:00"])


class AnalysisResponse(BaseModel):
    """Legacy mock response — kept for backward compatibility."""

    analysisRequestId: int
    fileId: int
    fileType: FileType
    status: str
    rawScore: float
    confidence: int
    evidence: list[str]
    modelName: str
    modelVersion: str
