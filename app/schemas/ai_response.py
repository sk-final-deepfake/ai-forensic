from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ModuleKind = Literal["cnn", "temporal", "optical"]


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
    reason: str


class ModuleTimelineItem(BaseModel):
    """Per-module timeline for detail-page charts (BE/FE contract extension)."""

    module: ModuleKind
    modelName: str
    modelVersion: str | None = None
    videoScore: float
    threshold: float
    detected: bool
    frameRisks: list[FrameRiskItem] = Field(default_factory=list)
    clipRisks: list[ClipRiskItem] = Field(default_factory=list)
    pairRisks: list[PairRiskItem] = Field(default_factory=list)
    suspiciousSegments: list[SuspiciousSegmentItem] = Field(default_factory=list)


class ModelScoreItem(BaseModel):
    moduleName: str
    detected: bool
    score: float
    modelName: str | None = None
    modelVersion: str | None = None


class RepresentativeFrameItem(BaseModel):
    """High-risk frame thumbnails for detail UI (FE contract)."""

    timeSec: float | None = None
    timestamp: str | None = None
    frameNumber: int | None = None
    score: float | None = None
    imageUrl: str | None = None
    heatmapUrl: str | None = None


class AnalysisVideoResultItem(BaseModel):
    type: Literal["video"] = "video"
    lipSyncDetected: bool = False
    lipSyncScore: float = 0.0
    frameEditDetected: bool = False
    frameEditScore: float = 0.0
    deepfakeDetected: bool
    deepfakeScore: float
    splicingDetected: bool = False
    splicingScore: float = 0.0
    reEncodingDetected: bool = False
    reEncodingScore: float = 0.0
    frameRisks: list[FrameRiskItem] = Field(default_factory=list)
    clipRisks: list[ClipRiskItem] = Field(default_factory=list)
    pairRisks: list[PairRiskItem] = Field(default_factory=list)
    suspiciousSegments: list[SuspiciousSegmentItem] = Field(default_factory=list)
    temporalSuspiciousSegments: list[SuspiciousSegmentItem] = Field(default_factory=list)
    opticalSuspiciousSegments: list[SuspiciousSegmentItem] = Field(default_factory=list)
    moduleTimelines: list[ModuleTimelineItem] = Field(default_factory=list)
    modelName: str | None = None
    modelVersion: str | None = None
    modelScores: list[ModelScoreItem] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    representativeFrames: list[RepresentativeFrameItem] = Field(default_factory=list)
    heatmapImageUrl: str | None = None
    overlayVideoUrl: str | None = None


class AnalysisResponseMessage(BaseModel):
    analysisRequestId: int
    evidenceId: int
    status: Literal["COMPLETED", "FAILED"]
    riskScore: float | None = None
    confidenceScore: float | None = None
    riskLevel: Literal["LOW", "MEDIUM", "HIGH"] | None = None
    analysisReasons: list[str] = Field(default_factory=list)
    results: list[AnalysisVideoResultItem] = Field(default_factory=list)
    analyzedAt: str
    errorCode: str | None = None
    message: str | None = None
    modelName: str | None = None
    modelVersion: str | None = None
    modelScores: list[ModelScoreItem] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
