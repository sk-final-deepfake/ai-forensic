"""Merge forgery TruFor/TimeSformer outputs into GPU worker AnalysisResponseMessage."""
from __future__ import annotations

import logging
from typing import Any

from gpu_worker.pipeline.forgery_infer import ForgeryLaneResult

logger = logging.getLogger("gpu_worker.forgery")


def _model_score_item(
    *,
    module_name: str,
    score: float,
    detected: bool,
    model_name: str,
    model_version: str,
) -> Any:
    try:
        from gpu_worker.schemas import ModelScoreItem  # noqa: WPS433

        return ModelScoreItem(
            moduleName=module_name,
            score=round(float(score), 6),
            detected=bool(detected),
            modelName=model_name,
            modelVersion=model_version,
        )
    except Exception:
        return {
            "moduleName": module_name,
            "score": round(float(score), 6),
            "detected": bool(detected),
            "modelName": model_name,
            "modelVersion": model_version,
        }


def _module_timeline_item(
    *,
    module: str,
    model_name: str,
    model_version: str,
    video_score: float,
    threshold: float,
    detected: bool,
    frame_risks: list[dict] | None = None,
    clip_risks: list[dict] | None = None,
    suspicious_segments: list[dict] | None = None,
) -> Any:
    try:
        from gpu_worker.schemas import (  # noqa: WPS433
            ClipRiskItem,
            FrameRiskItem,
            ModuleTimelineItem,
            SuspiciousSegmentItem,
        )

        fr = [
            FrameRiskItem(
                frameIndex=int(r.get("frameIndex", i)),
                timestampSec=float(r.get("timestampSec", 0.0)),
                riskScore=float(r.get("riskScore", 0.0)),
            )
            for i, r in enumerate(frame_risks or [])
        ]
        cr = [
            ClipRiskItem(
                clipIndex=int(r.get("clipIndex", i)),
                startFrameIndex=int(r.get("startFrameIndex", 0)),
                endFrameIndex=int(r.get("endFrameIndex", 0)),
                startTimeSec=float(r.get("startTimeSec", 0.0)),
                endTimeSec=float(r.get("endTimeSec", 0.0)),
                riskScore=float(r.get("riskScore", 0.0)),
            )
            for i, r in enumerate(clip_risks or [])
        ]
        seg = [
            SuspiciousSegmentItem(
                startTime=float(s.get("startTime", 0.0)),
                endTime=float(s.get("endTime", 0.0)),
                maxRiskScore=float(s.get("maxRiskScore", 0.0)),
                reason=str(s.get("reason", "")),
            )
            for s in (suspicious_segments or [])
        ]
        return ModuleTimelineItem(
            module=module,
            modelName=model_name,
            modelVersion=model_version,
            videoScore=round(float(video_score), 6),
            threshold=float(threshold),
            detected=bool(detected),
            frameRisks=fr,
            clipRisks=cr,
            suspiciousSegments=seg,
        )
    except Exception:
        return {
            "module": module,
            "modelName": model_name,
            "modelVersion": model_version,
            "videoScore": round(float(video_score), 6),
            "threshold": float(threshold),
            "detected": bool(detected),
            "frameRisks": frame_risks or [],
            "clipRisks": clip_risks or [],
            "suspiciousSegments": suspicious_segments or [],
        }


def merge_forgery_into_response(response: Any, forgery: ForgeryLaneResult, *, worker_cfg: Any = None) -> Any:
    """Append forgery_spatial / forgery_temporal to modelScores and moduleTimelines."""
    trufor_threshold = float(getattr(worker_cfg, "trufor_threshold", 0.515) if worker_cfg else 0.515)
    ts_threshold = float(getattr(worker_cfg, "ts_threshold", 0.173386) if worker_cfg else 0.173386)
    try:
        from gpu_worker.pipeline.forgery_infer import ForgeryInferConfig

        fcfg = ForgeryInferConfig.from_worker_config(worker_cfg) if worker_cfg else None
        if fcfg:
            trufor_threshold = fcfg.trufor_threshold
            ts_threshold = fcfg.ts_threshold
    except Exception:
        pass

    spatial_score_item = _model_score_item(
        module_name="forgery_spatial",
        score=forgery.spatial_score,
        detected=forgery.spatial_detected,
        model_name="TruFor",
        model_version=forgery.model_spatial_version,
    )
    temporal_score_item = _model_score_item(
        module_name="forgery_temporal",
        score=forgery.temporal_score,
        detected=forgery.temporal_detected,
        model_name="TimeSformer",
        model_version=forgery.model_temporal_version,
    )

    spatial_timeline = _module_timeline_item(
        module="forgery_spatial",
        model_name="TruFor",
        model_version=forgery.model_spatial_version,
        video_score=forgery.spatial_score,
        threshold=trufor_threshold,
        detected=forgery.spatial_detected,
        frame_risks=forgery.frame_risks,
        suspicious_segments=forgery.spatial_segments,
    )
    temporal_timeline = _module_timeline_item(
        module="forgery_temporal",
        model_name="TimeSformer",
        model_version=forgery.model_temporal_version,
        video_score=forgery.temporal_score,
        threshold=ts_threshold,
        detected=forgery.temporal_detected,
        clip_risks=forgery.clip_risks,
        suspicious_segments=forgery.temporal_segments,
    )

    # Top-level modelScores
    top_scores = list(getattr(response, "modelScores", None) or [])
    top_scores = [
        s
        for s in top_scores
        if str(getattr(s, "moduleName", s.get("moduleName") if isinstance(s, dict) else "")).lower()
        not in ("forgery_spatial", "forgery_temporal")
    ]
    top_scores.extend([spatial_score_item, temporal_score_item])
    if hasattr(response, "modelScores"):
        response.modelScores = top_scores
    elif isinstance(response, dict):
        response["modelScores"] = top_scores

    if not getattr(response, "results", None):
        logger.warning("Response has no results[]; cannot attach forgery moduleTimelines")
        return response

    video = response.results[0]
    video_scores = list(getattr(video, "modelScores", None) or [])
    video_scores = [
        s
        for s in video_scores
        if str(getattr(s, "moduleName", s.get("moduleName") if isinstance(s, dict) else "")).lower()
        not in ("forgery_spatial", "forgery_temporal")
    ]
    video_scores.extend([spatial_score_item, temporal_score_item])
    if hasattr(video, "modelScores"):
        video.modelScores = video_scores

    timelines = list(getattr(video, "moduleTimelines", None) or [])
    # Drop stale forgery timelines (e.g. face-gate TruFor) before attaching lane output.
    timelines = [
        t
        for t in timelines
        if str(getattr(t, "module", t.get("module") if isinstance(t, dict) else "")).lower()
        not in ("forgery_spatial", "forgery_temporal")
    ]
    timelines.extend([spatial_timeline, temporal_timeline])
    if hasattr(video, "moduleTimelines"):
        video.moduleTimelines = timelines

    # Do not append forgery lines to analysisReasons / 종합 소견 —
    # keep the deepfake late-fusion narrative as before.

    logger.info(
        "Merged forgery lane spatial=%.4f temporal=%.4f frameRisks=%d clipRisks=%d",
        forgery.spatial_score,
        forgery.temporal_score,
        len(forgery.frame_risks),
        len(forgery.clip_risks),
    )
    return response
