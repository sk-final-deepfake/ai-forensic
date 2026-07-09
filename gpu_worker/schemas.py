"""Pydantic schemas matching backend ai-json contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalysisJobMessage(BaseModel):
    """backend AnalysisJobMessage + S3/GPU 확장 필드."""

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


class ClipRiskItem(BaseModel):
    clipIndex: int
    startFrameIndex: int
    endFrameIndex: int
    startTimeSec: float
    endTimeSec: float
    riskScore: float


class PairRiskItem(BaseModel):
    pairIndex: int
    frameIndexA: int
    frameIndexB: int
    timestampSec: float
    riskScore: float
    motionMagnitude: float | None = None


class SuspiciousSegmentItem(BaseModel):
    startTime: float
    endTime: float
    maxRiskScore: float
    reason: str | None = None


class ModuleTimelineItem(BaseModel):
    module: str
    modelName: str
    modelVersion: str
    videoScore: float
    threshold: float
    detected: bool
    frameRisks: list[FrameRiskItem] | None = None
    clipRisks: list[ClipRiskItem] | None = None
    pairRisks: list[PairRiskItem] | None = None
    suspiciousSegments: list[SuspiciousSegmentItem] | None = None


class ModelScoreItem(BaseModel):
    moduleName: str
    detected: bool
    score: float
    modelName: str
    modelVersion: str


class RepresentativeFrameItem(BaseModel):
    timeSec: float | None = None
    timestamp: str | None = None
    frameNumber: int | None = None
    score: float | None = None
    imageUrl: str | None = None
    heatmapUrl: str | None = None


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
    clipRisks: list[ClipRiskItem] | None = None
    pairRisks: list[PairRiskItem] | None = None
    suspiciousSegments: list[SuspiciousSegmentItem] | None = None
    temporalSuspiciousSegments: list[SuspiciousSegmentItem] | None = None
    opticalSuspiciousSegments: list[SuspiciousSegmentItem] | None = None
    moduleTimelines: list[ModuleTimelineItem] | None = None
    modelScores: list[ModelScoreItem] | None = None
    representativeFrames: list[RepresentativeFrameItem] | None = None
    heatmapImageUrl: str | None = None
    overlayVideoUrl: str | None = None


class AnalysisResponseMessage(BaseModel):
    """docs/integrations/ai-json.md — BE AnalysisResponseMessage."""

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
    modelScores: list[ModelScoreItem] | None = None
