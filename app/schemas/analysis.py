from typing import Literal

from pydantic import BaseModel, Field


FileType = Literal["video", "audio", "image"]


class AnalysisRequest(BaseModel):
    analysisRequestId: int = Field(..., examples=[101])
    fileId: int = Field(..., examples=[15])
    caseId: int = Field(..., examples=[3])
    fileType: FileType = Field(..., examples=["video"])
    s3ObjectKey: str = Field(..., examples=["original-files/3/15/original.mp4"])
    presignedDownloadUrl: str = Field(..., examples=["https://example.com/presigned-url"])
    originalSha256: str = Field(..., examples=["abc123..."])
    requestedAt: str = Field(..., examples=["2026-06-10T10:30:00"])


class AnalysisResponse(BaseModel):
    analysisRequestId: int
    fileId: int
    fileType: FileType
    status: str
    rawScore: float
    confidence: int
    evidence: list[str]
    modelName: str
    modelVersion: str
