"""Backend RabbitMQ job/result contract (ai-json.md)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalysisJobMessage(BaseModel):
    analysisRequestId: int
    evidenceId: int
    fileType: str = "video"
    filePath: str
    s3ObjectKey: str | None = None
    s3Bucket: str | None = None
    s3Region: str | None = None
    presignedDownloadUrl: str | None = None
    originalHash: str | None = None
    originalSha256: str | None = None
    caseName: str | None = None
    requestedAt: str | None = None


class FrameRiskItem(BaseModel):
    frameIndex: int
    timestampSec: float
    riskScore: float


class AnalysisVideoResultItem(BaseModel):
    type: Literal["video"] = "video"
    modelName: str | None = None
    modelVersion: str | None = None
    lipSyncDetected: bool | None = None
    lipSyncScore: float | None = None
    frameEditDetected: bool | None = None
    frameEditScore: float | None = None
    deepfakeDetected: bool | None = None
    deepfakeScore: float | None = None
    splicingDetected: bool | None = None
    splicingScore: float | None = None
    reEncodingDetected: bool | None = None
    reEncodingScore: float | None = None
    frameRisks: list[FrameRiskItem] | None = None


class AnalysisResponseMessage(BaseModel):
    analysisRequestId: int
    evidenceId: int
    status: Literal["COMPLETED", "FAILED"]
    riskScore: float | None = None
    confidenceScore: float | None = None
    riskLevel: Literal["LOW", "MEDIUM", "HIGH"] | None = None
    modelName: str | None = None
    modelVersion: str | None = None
    analysisReasons: list[str] = Field(default_factory=list)
    results: list[AnalysisVideoResultItem] = Field(default_factory=list)
    analyzedAt: str
    errorCode: str | None = None
    message: str | None = None
