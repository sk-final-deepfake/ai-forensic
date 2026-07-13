from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.late_fusion import build_fused_per_frame_scores, collapse_frame_risks_by_frame
from app.services.module_overlays import build_module_overlay_set
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
    FaceBBoxItem,
    FrameRiskItem,
    ModelOverlayArtifactItem,
    ModelScoreItem,
    ModuleTimelineItem,
    PairRiskItem,
    PerFrameFaceScoreItem,
    RepresentativeFrameItem,
    SuspiciousSegmentItem,
)

logger = logging.getLogger("gpu_worker.pipeline.response_builder")

NO_FACE_STATUSES = frozenset({"no_face", "no_human_face", "skipped_no_human_face"})


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
    scores: list[dict[str, Any]] = []
    for row in frame_risks:
        frame_index = row.get("frameIndex", row.get("frame_index"))
        score = row.get("riskScore", row.get("fake_score"))
        if frame_index is None or score is None:
            continue
        entry = {
            "frame_index": int(frame_index),
            "fake_score": float(score),
        }
        if row.get("face_index") is not None:
            entry["face_index"] = int(row["face_index"])
        if row.get("faceIndex") is not None:
            entry["face_index"] = int(row["faceIndex"])
        if row.get("bbox") is not None:
            entry["bbox"] = row["bbox"]
        scores.append(entry)
    return scores


def _module_per_frame_scores(module: ModuleRunResult) -> list[dict[str, Any]]:
    raw = module.raw or {}
    breakdown = raw.get("score_breakdown") or {}
    rows = breakdown.get("per_frame_scores") or breakdown.get("per_frame") or []
    if rows:
        normalized: list[dict[str, Any]] = []
        for row in rows:
            frame_index = row.get("frame_index", row.get("frameIndex"))
            score = row.get("fake_score", row.get("prob_fake", row.get("riskScore")))
            if frame_index is None or score is None:
                continue
            entry = {
                "frame_index": int(frame_index),
                "fake_score": float(score),
            }
            if row.get("face_index") is not None:
                entry["face_index"] = int(row["face_index"])
            if row.get("bbox") is not None:
                entry["bbox"] = row["bbox"]
            normalized.append(entry)
        return normalized
    if module.frame_risks:
        return _to_visualization_scores(module.frame_risks)
    return []


def _build_fused_visualization_scores(
    *,
    modules: dict[str, ModuleRunResult],
    fusion_config: dict[str, Any],
    module_meta: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    from app.services.late_fusion import build_fused_per_frame_scores

    cnn_scores = _module_per_frame_scores(modules["cnn"])
    temporal_scores = _module_per_frame_scores(modules["temporal"])
    if not cnn_scores:
        return []

    def fuse_fn(*, cnn_score: float, temporal_score: float, optical_score: float) -> float:
        return apply_late_fusion(
            cnn_score=cnn_score,
            temporal_score=temporal_score,
            optical_score=optical_score,
            config=fusion_config,
            module_meta=module_meta,
        ).score

    return build_fused_per_frame_scores(
        cnn_scores=cnn_scores,
        temporal_scores=temporal_scores,
        optical_score=float(modules["optical"].video_score),
        fuse_fn=fuse_fn,
        temporal_video_score=float(modules["temporal"].video_score),
    )


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


def _to_per_frame_face_score_items(scores: list[dict[str, Any]]) -> list[PerFrameFaceScoreItem]:
    items: list[PerFrameFaceScoreItem] = []
    for row in scores:
        frame_index = row.get("frame_index", row.get("frameIndex"))
        score = row.get("fake_score", row.get("prob_fake", row.get("riskScore")))
        if frame_index is None or score is None:
            continue
        bbox_item = None
        bbox = row.get("bbox")
        if isinstance(bbox, dict) and all(key in bbox for key in ("x", "y", "w", "h")):
            bbox_item = FaceBBoxItem(
                x=int(bbox["x"]),
                y=int(bbox["y"]),
                w=int(bbox["w"]),
                h=int(bbox["h"]),
            )
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            bbox_item = FaceBBoxItem(
                x=int(bbox[0]),
                y=int(bbox[1]),
                w=int(bbox[2]),
                h=int(bbox[3]),
            )
        items.append(
            PerFrameFaceScoreItem(
                frameIndex=int(frame_index),
                faceIndex=int(row.get("face_index", row.get("faceIndex", 0))),
                riskScore=round(float(score), 6),
                bbox=bbox_item,
            )
        )
    return items


def _no_human_face_response(
    *,
    analysis_request_id: int,
    evidence_id: int,
    modules: list[str],
    frames_sampled: int | None = None,
) -> AnalysisResponseMessage:
    detail = f" sampled_frames={frames_sampled}" if frames_sampled is not None else ""
    return AnalysisResponseMessage(
        analysisRequestId=analysis_request_id,
        evidenceId=evidence_id,
        status="FAILED",
        analyzedAt=_utc_now(),
        errorCode="NO_HUMAN_FACE",
        message=(
            "사람 얼굴이 검출되지 않아 딥페이크 판별을 수행할 수 없습니다."
            f" (modules={','.join(modules)}{detail})"
        ),
    )


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
    cnn_status = str((cnn.raw or {}).get("status", "ok"))
    if cnn_status in NO_FACE_STATUSES or (cnn.raw or {}).get("fake_score") is None:
        breakdown = (cnn.raw or {}).get("score_breakdown") or {}
        return _no_human_face_response(
            analysis_request_id=analysis_request_id,
            evidence_id=evidence_id,
            modules=["cnn"],
            frames_sampled=breakdown.get("frames_sampled"),
        )

    temporal = run_timesformer_module(video_path, cfg, threshold=thresholds["temporal"], fps=fps)
    temporal_status = str((temporal.raw or {}).get("status", "ok"))
    if temporal_status in NO_FACE_STATUSES or (temporal.raw or {}).get("fake_score") is None:
        breakdown = (temporal.raw or {}).get("score_breakdown") or {}
        return _no_human_face_response(
            analysis_request_id=analysis_request_id,
            evidence_id=evidence_id,
            modules=["cnn", "temporal"],
            frames_sampled=breakdown.get("frames_sampled"),
        )

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
    overlay_video_url = None
    model_overlay_artifacts: list[ModelOverlayArtifactItem] = []
    fused_visualization_scores: list[dict[str, Any]] = []

    try:
        fused_visualization_scores = _build_fused_visualization_scores(
            modules=modules,
            fusion_config=fusion_config,
            module_meta=module_meta,
        )
        visualization_scores = fused_visualization_scores
        if not visualization_scores:
            visualization_scores = _to_visualization_scores(cnn.frame_risks)
        if not visualization_scores:
            visualization_scores = _fallback_visualization_scores(video_path, fusion.score, fps)

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

        overlay_set = build_module_overlay_set(
            video_path=video_path,
            evidence_id=evidence_id,
            analysis_request_id=analysis_request_id,
            work_dir=cfg.work_dir / "visualization" / f"{evidence_id}_{analysis_request_id}" / "modules",
            cnn_per_frame_scores=_module_per_frame_scores(cnn),
            clip_risks=temporal.clip_risks,
            pair_risks=optical.pair_risks,
        )
        overlay_video_url = overlay_set.legacy_cnn_overlay_url
        model_overlay_artifacts = [
            ModelOverlayArtifactItem(**row) for row in overlay_set.model_overlay_artifacts
        ]
        for timeline in module_timelines:
            url = overlay_set.overlay_by_module.get(timeline.module)
            if url:
                timeline.overlayVideoUrl = url
        logger.info(
            "Module overlays attached: evidenceId=%s cnn=%s temporal=%s optical=%s",
            evidence_id,
            bool(overlay_set.overlay_by_module.get("cnn")),
            bool(overlay_set.overlay_by_module.get("temporal")),
            bool(overlay_set.overlay_by_module.get("optical")),
        )
    except Exception:
        logger.exception(
            "Failed to build visualization artifacts: evidenceId=%s analysisRequestId=%s",
            evidence_id,
            analysis_request_id,
        )

    fused_frame_risks = collapse_frame_risks_by_frame(
        fused_visualization_scores or _module_per_frame_scores(cnn),
        video_path,
    )

    video_item = AnalysisVideoResultItem(
        modelName=str(fusion_meta.get("modelName", "Late Fusion")),
        modelVersion=str(fusion_meta.get("modelVersion", fusion_config.get("fusion_version", "fusion-v4-ts-gated"))),
        deepfakeDetected=fusion.detected,
        deepfakeScore=fusion.score,
        frameRisks=_to_frame_risks(fused_frame_risks) or None,
        clipRisks=_to_clip_risks(temporal.clip_risks) or None,
        pairRisks=_to_pair_risks(optical.pair_risks) or None,
        suspiciousSegments=_to_segments(cnn.suspicious_segments) or None,
        temporalSuspiciousSegments=_to_segments(temporal.temporal_suspicious_segments) or None,
        opticalSuspiciousSegments=_to_segments(optical.optical_suspicious_segments) or None,
        moduleTimelines=module_timelines,
        modelScores=model_scores,
        representativeFrames=representative_frames,
        overlayVideoUrl=overlay_video_url,
        modelOverlayArtifacts=model_overlay_artifacts or None,
        perFrameFaceScores=_to_per_frame_face_score_items(fused_visualization_scores)
        if fused_visualization_scores
        else None,
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
