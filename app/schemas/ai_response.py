from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FrameRiskItem(BaseModel):
    frameIndex: int
    timestampSec: float
    riskScore: float


class SuspiciousSegmentItem(BaseModel):
    startTime: float
    endTime: float
    maxRiskScore: float
    reason: str


class ModelScoreItem(BaseModel):
    moduleName: str
    detected: bool
    score: float
    modelName: str | None = None
    modelVersion: str | None = None


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
    suspiciousSegments: list[SuspiciousSegmentItem] = Field(default_factory=list)
    modelName: str | None = None
    modelVersion: str | None = None
    modelScores: list[ModelScoreItem] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


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
