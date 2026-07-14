"""Pydantic schemas for GPU worker visualization payload."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RepresentativeFrameViz(BaseModel):
    timeSec: float
    timestamp: str
    frameNumber: int
    score: float
    imageUrl: str | None = None
    heatmapImageUrl: str | None = None
    module: str | None = None
    modelName: str | None = None


class VisualizationArtifacts(BaseModel):
    representativeFrames: list[RepresentativeFrameViz] = Field(default_factory=list)
    overlayVideoUrl: str | None = None
    heatmapImageUrl: str | None = None
    spatialOverlayVideoUrl: str | None = None
    temporalOverlayVideoUrl: str | None = None


class VideoAnalysisVizResult(BaseModel):
    analysisRequestId: int
    evidenceId: int
    visualization: VisualizationArtifacts
