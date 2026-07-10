from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.visualization_artifacts import build_visualization_artifacts
from gpu_worker.config import WorkerConfig
from gpu_worker.pipeline.fusion import FusionResult, apply_late_fusion, load_fusion_config
from gpu_worker.pipeline.module_infer import (
    ModuleRunResult,
    _video_fps,
    run_gmflow_module,
    run_timesformer_module,
    run_xception_module,
)
from gpu_worker.pipeline.paths import resolve_under_root
from gpu_worker.schemas import (
    AnalysisResponseMessage,
    AnalysisVideoResultItem,
    ClipRiskItem,
    FrameRiskItem,
    ModelScoreItem,
    ModuleTimelineItem,
    PairRiskItem,
    RepresentativeFrameItem,
    SuspiciousSegmentItem,
)

logger = logging.getLogger("gpu_worker.pipeline.response_builder")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _model_meta(config: dict[str, Any], key: str) -> dict[str, str]:
    models = config.get("models") or {}
    model_versions = config.get("model_versions") or {}
    meta = dict(models.get(key) or {})
    if "modelVersion" not in meta and key in model_versions:
        meta["modelVersion"] = str(model_versions[key])
    if "modelName" not in meta:
        default_names = {
            "fusion": "Late Fusion",
            "cnn": "Xception",
            "temporal": "TimeSformer",
            "optical": "GMFlow",
        }
        meta["modelName"] = default_names.get(key, key)
    return meta


def _module_thresholds(config: dict[str, Any]) -> dict[str, float]:
    defaults = {"cnn": 0.5, "temporal": 0.5, "optical": 0.405315}
    overrides = config.get("module_thresholds") or {}
    return {key: float(overrides.get(key, value)) for key, value in defaults.items()}


def _to_frame_risks(rows: list[dict[str, Any]]) -> list[FrameRiskItem]:
    return [FrameRiskItem(**row) for row in rows]


def _to_clip_risks(rows: list[dict[str, Any]]) -> list[ClipRiskItem]:
    return [ClipRiskItem(**row) for row in rows]


def _to_pair_risks(rows: list[dict[str, Any]]) -> list[PairRiskItem]:
    return [PairRiskItem(**row) for row in rows]


def _to_segments(rows: list[dict[str, Any]]) -> list[SuspiciousSegmentItem]:
    return [SuspiciousSegmentItem(**row) for row in rows]


def _to_visualization_scores(frame_risks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "frame_index": int(row["frameIndex"]),
            "fake_score": float(row["riskScore"]),
        }
        for row in frame_risks
        if row.get("frameIndex") is not None and row.get("riskScore") is not None
    ]


def _fallback_visualization_scores(video_path: Path, score: float, fps: float) -> list[dict[str, Any]]:
    """Create coarse visualization points when the model returns only a video-level score."""
    try:
        import cv2
    except ImportError:
        return [{"frame_index": 0, "fake_score": score}]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [{"frame_index": 0, "fake_score": score}]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    step = max(1, int(fps))
    max_frames = min(frame_count if frame_count > 0 else step, int(max(fps, fps * 60)))
    return [
        {"frame_index": frame_index, "fake_score": score}
        for frame_index in range(0, max_frames, step)
    ] or [{"frame_index": 0, "fake_score": score}]


def _build_model_scores(
    fusion: FusionResult,
    modules: dict[str, ModuleRunResult],
    config: dict[str, Any],
) -> list[ModelScoreItem]:
    fusion_meta = _model_meta(config, "fusion")
    scores = [
        ModelScoreItem(
            moduleName="deepfake",
            detected=fusion.detected,
            score=fusion.score,
            modelName=str(fusion_meta.get("modelName", "Late Fusion")),
            modelVersion=str(fusion_meta.get("modelVersion", config.get("fusion_version", "fusion-v4-ts-gated"))),
        )
    ]
    mapping = {
        "cnn": "deepfake_cnn",
        "temporal": "deepfake_temporal",
        "optical": "deepfake_optical",
    }
    for key, module_name in mapping.items():
        module = modules[key]
        meta = _model_meta(config, key)
        scores.append(
            ModelScoreItem(
                moduleName=module_name,
                detected=module.detected,
                score=module.video_score,
                modelName=str(meta.get("modelName", module.model_name)),
                modelVersion=str(meta.get("modelVersion", module.model_version)),
            )
        )
    return scores


def _build_module_timelines(modules: dict[str, ModuleRunResult], config: dict[str, Any]) -> list[ModuleTimelineItem]:
    timelines: list[ModuleTimelineItem] = []
    for key in ("cnn", "temporal", "optical"):
        module = modules[key]
        meta = _model_meta(config, key)
        timelines.append(
            ModuleTimelineItem(
                module=module.module,
                modelName=str(meta.get("modelName", module.model_name)),
                modelVersion=str(meta.get("modelVersion", module.model_version)),
                videoScore=module.video_score,
                threshold=module.threshold,
                detected=module.detected,
                frameRisks=_to_frame_risks(module.frame_risks) or None,
                clipRisks=_to_clip_risks(module.clip_risks) or None,
                pairRisks=_to_pair_risks(module.pair_risks) or None,
                suspiciousSegments=_to_segments(module.suspicious_segments or module.temporal_suspicious_segments or module.optical_suspicious_segments) or None,
            )
        )
    return timelines


def build_analysis_response(
    *,
    analysis_request_id: int,
    evidence_id: int,
    video_path: Path,
    cfg: WorkerConfig,
) -> AnalysisResponseMessage:
    fusion_path = resolve_under_root(cfg, cfg.fusion_config_path)
    if not fusion_path.is_file():
        raise FileNotFoundError(f"Fusion config not found: {fusion_path}")

    fusion_config = load_fusion_config(fusion_path)
    thresholds = _module_thresholds(fusion_config)
    fps = _video_fps(video_path)

    cnn = run_xception_module(video_path, cfg, threshold=thresholds["cnn"], fps=fps)
    temporal = run_timesformer_module(video_path, cfg, threshold=thresholds["temporal"], fps=fps)
    optical = run_gmflow_module(video_path, cfg, threshold=thresholds["optical"], fps=fps)
    modules = {"cnn": cnn, "temporal": temporal, "optical": optical}

    module_meta = {key: _model_meta(fusion_config, key) for key in modules}
    fusion = apply_late_fusion(
        cnn_score=cnn.video_score,
        temporal_score=temporal.video_score,
        optical_score=optical.video_score,
        config=fusion_config,
        module_meta=module_meta,
    )
    model_scores = _build_model_scores(fusion, modules, fusion_config)
    module_timelines = _build_module_timelines(modules, fusion_config)
    fusion_meta = _model_meta(fusion_config, "fusion")
    representative_frames = None
    heatmap_image_url = None
    overlay_video_url = None

    try:
        visualization_scores = _to_visualization_scores(cnn.frame_risks)
        if not visualization_scores:
            visualization_scores = _fallback_visualization_scores(video_path, cnn.video_score, fps)
            logger.info(
                "Using fallback visualization scores: evidenceId=%s analysisRequestId=%s score=%s points=%s",
                evidence_id,
                analysis_request_id,
                cnn.video_score,
                len(visualization_scores),
            )
        else:
            logger.info(
                "Using CNN frame risks for visualization: evidenceId=%s analysisRequestId=%s points=%s",
                evidence_id,
                analysis_request_id,
                len(visualization_scores),
            )

        viz = build_visualization_artifacts(
            video_path=video_path,
            per_frame_scores=visualization_scores,
            evidence_id=evidence_id,
            analysis_request_id=analysis_request_id,
            work_dir=cfg.work_dir / "visualization" / f"{evidence_id}_{analysis_request_id}",
        )
        if viz is not None:
            representative_frames = [
                RepresentativeFrameItem(**row) for row in viz.representative_frames
            ]
            heatmap_image_url = viz.heatmap_image_url
            overlay_video_url = viz.overlay_video_url
            logger.info(
                "Visualization artifacts attached: evidenceId=%s analysisRequestId=%s frames=%s heatmap=%s overlay=%s",
                evidence_id,
                analysis_request_id,
                len(representative_frames or []),
                bool(heatmap_image_url),
                bool(overlay_video_url),
            )
        else:
            logger.warning(
                "Visualization artifact builder returned no artifacts: evidenceId=%s analysisRequestId=%s points=%s",
                evidence_id,
                analysis_request_id,
                len(visualization_scores),
            )
    except Exception:
        logger.exception(
            "Failed to build visualization artifacts: evidenceId=%s analysisRequestId=%s",
            evidence_id,
            analysis_request_id,
        )

    video_item = AnalysisVideoResultItem(
        modelName=str(fusion_meta.get("modelName", "Late Fusion")),
        modelVersion=str(fusion_meta.get("modelVersion", fusion_config.get("fusion_version", "fusion-v4-ts-gated"))),
        deepfakeDetected=fusion.detected,
        deepfakeScore=fusion.score,
        frameRisks=_to_frame_risks(cnn.frame_risks) or None,
        clipRisks=_to_clip_risks(temporal.clip_risks) or None,
        pairRisks=_to_pair_risks(optical.pair_risks) or None,
        suspiciousSegments=_to_segments(cnn.suspicious_segments) or None,
        temporalSuspiciousSegments=_to_segments(temporal.temporal_suspicious_segments) or None,
        opticalSuspiciousSegments=_to_segments(optical.optical_suspicious_segments) or None,
        moduleTimelines=module_timelines,
        modelScores=model_scores,
        representativeFrames=representative_frames,
        heatmapImageUrl=heatmap_image_url,
        overlayVideoUrl=overlay_video_url,
    )

    return AnalysisResponseMessage(
        analysisRequestId=analysis_request_id,
        evidenceId=evidence_id,
        status="COMPLETED",
        riskScore=fusion.risk_score,
        confidenceScore=fusion.confidence,
        riskLevel=fusion.risk_level,  # type: ignore[arg-type]
        modelName=video_item.modelName,
        modelVersion=video_item.modelVersion,
        analysisReasons=fusion.reasons,
        results=[video_item],
        analyzedAt=_utc_now(),
        modelScores=model_scores,
    )
