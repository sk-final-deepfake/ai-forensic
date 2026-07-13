from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.late_fusion import (
    build_fused_per_frame_scores,
    build_suspicious_segments,
    collapse_frame_risks_by_frame,
)
from app.services.module_overlays import build_module_overlay_set
from app.services.visualization_artifacts import build_visualization_artifacts
from gpu_worker.config import WorkerConfig
from gpu_worker.pipeline.fusion import FusionResult, apply_late_fusion, load_fusion_config
from gpu_worker.pipeline.module_infer import (
    ModuleRunResult,
    _video_fps,
    forgery_ran_successfully,
    run_gmflow_module,
    run_timesformer_module,
    run_trufor_module,
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


def _safe_score(value: float | int | None) -> float:
    """Never emit null/NaN module scores — EKS consumer Pydantic rejects them."""
    if value is None:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(parsed) or math.isinf(parsed):
        return 0.0
    return parsed


NO_FACE_STATUSES = frozenset({"no_face", "no_human_face", "skipped_no_human_face"})
FACE_QUALITY_STATUSES = frozenset({"face_too_small", "insufficient_face_samples"})
TEMPORAL_UNAVAILABLE_STATUSES = frozenset(
    {
        "insufficient_face_samples",
        "insufficient_temporal_clips",
        "face_too_small",
        "error",
        "skipped",
        *NO_FACE_STATUSES,
    }
)


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
                score=_safe_score(module.video_score),
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
                videoScore=_safe_score(module.video_score),
                threshold=_safe_score(module.threshold),
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


def _forgery_continuation_note(forgery: ModuleRunResult | None) -> str:
    if forgery is None:
        return "위변조(TruFor)는 가중치 또는 vendor가 없어 생략되었습니다."
    status = str((forgery.raw or {}).get("status", ""))
    if status == "ok":
        return "위변조(TruFor) 공간 분석을 이어서 수행했습니다."
    if status in {"skipped_unavailable", "skipped"}:
        detail = str((forgery.raw or {}).get("message") or "").strip()
        suffix = f" ({detail})" if detail else ""
        return f"위변조(TruFor)는 가중치 또는 vendor가 없어 생략되었습니다.{suffix}"
    detail = str((forgery.raw or {}).get("message") or "").strip()
    suffix = f" ({detail})" if detail else ""
    return f"위변조(TruFor) 실행 중 오류가 발생해 생략되었습니다.{suffix}"


def _soft_inconclusive_response(
    *,
    analysis_request_id: int,
    evidence_id: int,
    error_code: str,
    modules: list[str],
    frames_sampled: int | None = None,
    message: str | None = None,
    forgery: ModuleRunResult | None = None,
) -> AnalysisResponseMessage:
    detail = f" sampled_frames={frames_sampled}" if frames_sampled is not None else ""
    forgery_note = _forgery_continuation_note(forgery)
    defaults = {
        "NO_HUMAN_FACE": (
            "사람 얼굴이 검출되지 않아 딥페이크 판별을 수행할 수 없습니다. "
            f"{forgery_note}"
        ),
        "FACE_TOO_SMALL": (
            "검출된 얼굴이 너무 작아(전신·원거리 등) 신뢰 가능한 딥페이크 판별을 보류합니다. "
            f"{forgery_note}"
        ),
        "INSUFFICIENT_FACE_SAMPLES": (
            "분석에 쓸 수 있는 얼굴 프레임이 부족하여 딥페이크 판별을 보류합니다. "
            f"{forgery_note}"
        ),
    }
    base = message or defaults.get(
        error_code,
        f"딥페이크 판별을 수행할 수 없습니다. {forgery_note}",
    )
    full_message = f"{base} (modules={','.join(modules)}{detail})"
    reasons = [f"{error_code}: 딥페이크 모델 판단 보류", full_message]
    version = f"inconclusive-{error_code.lower().replace('_', '-')}"

    model_scores = [
        ModelScoreItem(
            moduleName="deepfake",
            detected=False,
            score=0.0,
            modelName="forenshield-late-fusion",
            modelVersion=version,
        )
    ]
    module_timelines: list[ModuleTimelineItem] | None = None
    frame_edit_detected = False
    frame_edit_score = 0.0

    if forgery_ran_successfully(forgery) and forgery is not None:
        model_scores.append(
            ModelScoreItem(
                moduleName=forgery.module,
                detected=forgery.detected,
                score=_safe_score(forgery.video_score),
                modelName=forgery.model_name,
                modelVersion=forgery.model_version,
            )
        )
        module_timelines = [
            ModuleTimelineItem(
                module=forgery.module,
                modelName=forgery.model_name,
                modelVersion=forgery.model_version,
                videoScore=_safe_score(forgery.video_score),
                threshold=_safe_score(forgery.threshold),
                detected=forgery.detected,
                frameRisks=_to_frame_risks(forgery.frame_risks) or None,
                clipRisks=None,
                pairRisks=None,
                suspiciousSegments=_to_segments(forgery.suspicious_segments) or None,
            )
        ]
        frame_edit_detected = forgery.detected
        frame_edit_score = _safe_score(forgery.video_score)
        reasons.append(
            f"forgery_spatial: TruFor score={_safe_score(forgery.video_score):.4f} "
            f"detected={forgery.detected}"
        )
    else:
        reasons.append(forgery_note)

    video_item = AnalysisVideoResultItem(
        modelName="forenshield-late-fusion",
        modelVersion=version,
        deepfakeDetected=False,
        deepfakeScore=0.0,
        frameEditDetected=frame_edit_detected,
        frameEditScore=frame_edit_score,
        frameRisks=None,
        clipRisks=None,
        pairRisks=None,
        suspiciousSegments=None,
        temporalSuspiciousSegments=None,
        opticalSuspiciousSegments=None,
        moduleTimelines=module_timelines,
        modelScores=model_scores,
        representativeFrames=None,
        overlayVideoUrl=None,
        perFrameFaceScores=None,
    )
    return AnalysisResponseMessage(
        analysisRequestId=analysis_request_id,
        evidenceId=evidence_id,
        status="COMPLETED",
        riskScore=0.0,
        confidenceScore=0.0,
        riskLevel="LOW",
        analyzedAt=_utc_now(),
        errorCode=error_code,
        message=full_message,
        analysisReasons=reasons,
        results=[video_item],
        modelScores=video_item.modelScores,
        modelName=video_item.modelName,
        modelVersion=video_item.modelVersion,
    )


def _no_human_face_response(
    *,
    analysis_request_id: int,
    evidence_id: int,
    modules: list[str],
    frames_sampled: int | None = None,
) -> AnalysisResponseMessage:
    return _soft_inconclusive_response(
        analysis_request_id=analysis_request_id,
        evidence_id=evidence_id,
        error_code="NO_HUMAN_FACE",
        modules=modules,
        frames_sampled=frames_sampled,
    )


def _cnn_status_to_error_code(status: str) -> str | None:
    if status in NO_FACE_STATUSES:
        return "NO_HUMAN_FACE"
    if status == "face_too_small":
        return "FACE_TOO_SMALL"
    if status == "insufficient_face_samples":
        return "INSUFFICIENT_FACE_SAMPLES"
    return None


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
    cnn_error = _cnn_status_to_error_code(cnn_status)
    if cnn_error is not None or (cnn.raw or {}).get("fake_score") is None:
        breakdown = (cnn.raw or {}).get("score_breakdown") or {}
        # Soft advisory only — still run TruFor spatial best-effort in the same response.
        forgery = run_trufor_module(
            video_path,
            cfg,
            fps=fps,
            work_dir=cfg.work_dir / "trufor" / f"{evidence_id}_{analysis_request_id}",
        )
        logger.info(
            "Soft face-gate → forgery continuation: evidenceId=%s errorCode=%s forgery_status=%s",
            evidence_id,
            cnn_error or "NO_HUMAN_FACE",
            (forgery.raw or {}).get("status"),
        )
        return _soft_inconclusive_response(
            analysis_request_id=analysis_request_id,
            evidence_id=evidence_id,
            error_code=cnn_error or "NO_HUMAN_FACE",
            modules=["cnn", "forgery_spatial"],
            frames_sampled=breakdown.get("frames_sampled"),
            forgery=forgery,
        )

    temporal = run_timesformer_module(video_path, cfg, threshold=thresholds["temporal"], fps=fps)
    temporal_status = str((temporal.raw or {}).get("status", "ok"))
    temporal_unavailable = (
        temporal_status in TEMPORAL_UNAVAILABLE_STATUSES
        or (temporal.raw or {}).get("fake_score") is None
    )
    soft_error_code: str | None = None
    soft_error_message: str | None = None
    if temporal_unavailable:
        soft_error_code = "TEMPORAL_MODULE_UNAVAILABLE"
        soft_error_message = (
            "TimeSformer(시계열) 모듈을 사용할 수 없어 CNN·광학 흐름 중심으로 판별했습니다. "
            f"(temporal_status={temporal_status})"
        )
        logger.warning(
            "Temporal module unavailable; continuing with CNN/optical: evidenceId=%s status=%s",
            evidence_id,
            temporal_status,
        )

    optical = run_gmflow_module(video_path, cfg, threshold=thresholds["optical"], fps=fps)
    modules = {"cnn": cnn, "temporal": temporal, "optical": optical}

    module_meta = {key: _model_meta(fusion_config, key) for key in modules}
    fusion = apply_late_fusion(
        cnn_score=cnn.video_score,
        temporal_score=0.0 if temporal_unavailable else temporal.video_score,
        optical_score=optical.video_score,
        config=fusion_config,
        module_meta=module_meta,
    )
    if soft_error_message:
        fusion = FusionResult(
            score=fusion.score,
            detected=fusion.detected,
            confidence=fusion.confidence,
            risk_score=fusion.risk_score,
            risk_level=fusion.risk_level,
            reasons=[soft_error_message, *fusion.reasons],
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
    suspicious_cfg = fusion_config.get("suspicious_segment") or {}
    fused_segments = build_suspicious_segments(
        fused_frame_risks,
        high_risk_threshold=float(suspicious_cfg.get("high_risk_frame_threshold", 0.65)),
        min_segment_sec=float(suspicious_cfg.get("min_segment_sec", 0.5)),
        reason="High fused frame-level fake probability cluster",
    )

    video_item = AnalysisVideoResultItem(
        modelName=str(fusion_meta.get("modelName", "Late Fusion")),
        modelVersion=str(fusion_meta.get("modelVersion", fusion_config.get("fusion_version", "fusion-v4-ts-gated"))),
        deepfakeDetected=fusion.detected,
        deepfakeScore=fusion.score,
        frameRisks=_to_frame_risks(fused_frame_risks) or None,
        clipRisks=_to_clip_risks(temporal.clip_risks) or None,
        pairRisks=_to_pair_risks(optical.pair_risks) or None,
        suspiciousSegments=_to_segments(fused_segments) or None,
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
        errorCode=soft_error_code,
        message=soft_error_message,
        modelScores=model_scores,
    )
