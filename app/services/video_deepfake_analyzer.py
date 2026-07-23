from __future__ import annotations

import hashlib
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from app.core.model_settings import ModelSettings, load_model_settings
from app.schemas.ai_response import (
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
from app.services.module_overlays import build_module_overlay_set
from app.services.visualization_artifacts import build_visualization_artifacts
from app.schemas.analysis import AnalysisRequest
from app.services.infer_bridge import InferRuntime, ModuleInferResult
from app.services.integrated_risk import (
    build_forgery_analysis_reasons,
    build_integrated_risk_reason,
    integrate_risk_score,
)
from app.services.late_fusion import (
    FusionConfig,
    build_analysis_reasons,
    build_clip_risks,
    build_fused_per_frame_scores,
    build_module_timelines,
    build_pair_risks,
    build_suspicious_segments,
    collapse_frame_risks_by_frame,
    confidence_from_module_scores,
    fuse_scores,
    load_fusion_config,
    score_detected,
)

logger = logging.getLogger("ai_fastapi.video_deepfake_analyzer")

FUSION_MODEL_NAME = "forenshield-late-fusion"
NO_HUMAN_FACE_STATUSES = frozenset({"no_face", "no_human_face", "skipped_no_human_face"})
FACE_QUALITY_STATUSES = frozenset({"face_too_small", "insufficient_face_samples"})
TEMPORAL_UNAVAILABLE_STATUSES = frozenset(
    {
        "insufficient_face_samples",
        "insufficient_temporal_clips",
        "face_too_small",
        "error",
        "skipped",
    }
)

SOFT_INCONCLUSIVE_MESSAGES = {
    "NO_HUMAN_FACE": "사람 얼굴이 검출되지 않아 딥페이크 판별을 수행할 수 없습니다.",
    "FACE_TOO_SMALL": "검출된 얼굴이 너무 작아(전신·원거리 등) 신뢰 가능한 딥페이크 판별을 보류합니다.",
    "INSUFFICIENT_FACE_SAMPLES": "분석에 쓸 수 있는 얼굴 프레임이 부족하여 딥페이크 판별을 보류합니다.",
}

FORGERY_RAN_NOTE = "위변조(TruFor) 공간 분석을 이어서 수행했습니다."
FORGERY_SKIPPED_NOTE = "위변조(TruFor)는 가중치 또는 vendor가 없어 생략되었습니다."
FORGERY_ERROR_NOTE = "위변조(TruFor) 실행 중 오류가 발생해 생략되었습니다."


def _forgery_continuation_note(forgery: ModuleInferResult | None) -> str:
    if forgery is None:
        return FORGERY_SKIPPED_NOTE
    if forgery.status == "ok" and forgery.fake_score is not None:
        return FORGERY_RAN_NOTE
    if forgery.status in {"skipped_unavailable", "skipped"}:
        return FORGERY_SKIPPED_NOTE
    return FORGERY_ERROR_NOTE


def _module_run_to_infer_result(module: Any) -> ModuleInferResult:
    """Adapt gpu_worker ModuleRunResult → ModuleInferResult for soft-gate forgery."""
    raw = getattr(module, "raw", None) or {}
    return ModuleInferResult(
        module=str(getattr(module, "module", "forgery_spatial")),
        model_name=str(getattr(module, "model_name", "TruFor")),
        model_version=str(getattr(module, "model_version", "v1.0.0")),
        status=str(raw.get("status", "ok")),
        fake_score=raw.get("fake_score", getattr(module, "video_score", None)),
        pred_label="fake" if getattr(module, "detected", False) else "real",
        details={
            "threshold": getattr(module, "threshold", 0.515),
            "frame_risks": getattr(module, "frame_risks", []) or [],
            "suspicious_segments": getattr(module, "suspicious_segments", []) or [],
            "raw": raw,
        },
    )


def _try_run_forgery_spatial(
    video_path: Path,
    *,
    work_dir: Path | None = None,
) -> ModuleInferResult | None:
    """Best-effort TruFor on soft face-gate. Never raises.

    Skipped when FORGERY_ENABLED (forgery lane owns TruFor — avoids dual run).
    """
    import os

    if os.getenv("FORGERY_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        logger.info("Soft TruFor skipped — FORGERY_ENABLED; defer to forgery lane")
        return None
    try:
        from gpu_worker.config import load_config
        from gpu_worker.pipeline.module_infer import run_trufor_module
    except Exception as exc:  # noqa: BLE001
        logger.warning("TruFor soft-continuation unavailable (import): %s", exc)
        return ModuleInferResult(
            module="forgery_spatial",
            model_name="TruFor",
            model_version="v1.0.0",
            status="skipped_unavailable",
            fake_score=None,
            pred_label=None,
            details={"message": str(exc)},
        )
    try:
        if not video_path.is_file():
            return ModuleInferResult(
                module="forgery_spatial",
                model_name="TruFor",
                model_version="v1.0.0",
                status="skipped",
                fake_score=None,
                pred_label=None,
                details={"message": f"video not found: {video_path}"},
            )
        cfg = load_config()
        forgery_work = None
        if work_dir is not None:
            forgery_work = work_dir / "trufor"
        result = run_trufor_module(video_path, cfg, work_dir=forgery_work)
        return _module_run_to_infer_result(result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("TruFor soft-continuation failed for %s", video_path)
        return ModuleInferResult(
            module="forgery_spatial",
            model_name="TruFor",
            model_version="v1.0.0",
            status="error",
            fake_score=None,
            pred_label=None,
            details={"message": str(exc)},
        )


def _to_per_frame_face_scores(rows: list[dict[str, Any]]) -> list[PerFrameFaceScoreItem]:
    items: list[PerFrameFaceScoreItem] = []
    for row in rows:
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_evidence_id(request: AnalysisRequest) -> int:
    if request.evidenceId is not None:
        return request.evidenceId
    if request.fileId is not None:
        return request.fileId
    raise ValueError("evidenceId or fileId is required")


def _resolve_original_hash(request: AnalysisRequest) -> str | None:
    return request.originalHash or request.originalSha256


def _download_video(request: AnalysisRequest, dest_dir: Path) -> Path:
    if request.localVideoPath:
        source = Path(request.localVideoPath)
        if not source.is_file():
            raise FileNotFoundError(f"localVideoPath not found: {source}")
        return source

    url = request.presignedDownloadUrl
    if not url:
        raise ValueError("presignedDownloadUrl or localVideoPath is required")

    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix or ".mp4"
    target = dest_dir / f"evidence_{request.analysisRequestId}{suffix}"
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def _verify_sha256(path: Path, expected: str | None) -> None:
    if not expected:
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest.lower() != expected.lower():
        raise ValueError(f"SHA-256 mismatch: expected={expected[:16]}..., actual={digest[:16]}...")


def _failed_response(
    request: AnalysisRequest,
    *,
    error_code: str,
    message: str,
) -> AnalysisResponseMessage:
    return AnalysisResponseMessage(
        analysisRequestId=request.analysisRequestId,
        evidenceId=_resolve_evidence_id(request),
        status="FAILED",
        analyzedAt=_utc_now_iso(),
        errorCode=error_code,
        message=message,
    )


def _inconclusive_soft_response(
    request: AnalysisRequest,
    *,
    error_code: str,
    blocked_modules: list[str],
    frames_sampled: int | None = None,
    forgery: ModuleInferResult | None = None,
) -> AnalysisResponseMessage:
    """Soft-complete and attach best-effort forgery_spatial (TruFor) when available."""
    detail = f" sampled_frames={frames_sampled}" if frames_sampled is not None else ""
    forgery_note = _forgery_continuation_note(forgery)
    base = SOFT_INCONCLUSIVE_MESSAGES.get(
        error_code,
        "딥페이크 판별을 수행할 수 없습니다.",
    )
    message = f"{base} {forgery_note} (modules={','.join(blocked_modules)}{detail})"
    reasons = [f"{error_code}: 딥페이크 모델 판단 보류", message]
    version = f"inconclusive-{error_code.lower().replace('_', '-')}"

    model_scores = [
        ModelScoreItem(
            moduleName="deepfake",
            detected=False,
            score=0.0,
            modelName=FUSION_MODEL_NAME,
            modelVersion=version,
        )
    ]
    module_timelines: list[ModuleTimelineItem] = []
    frame_edit_detected = False
    frame_edit_score = 0.0

    if forgery is not None and forgery.status == "ok" and forgery.fake_score is not None:
        threshold = 0.515
        if isinstance(forgery.details, dict) and forgery.details.get("threshold") is not None:
            threshold = float(forgery.details["threshold"])
        score = round(float(forgery.fake_score), 6)
        detected = score >= threshold
        model_scores.append(
            ModelScoreItem(
                moduleName="forgery_spatial",
                detected=detected,
                score=score,
                modelName=forgery.model_name,
                modelVersion=forgery.model_version,
            )
        )
        frame_risks_raw = []
        segments_raw = []
        if isinstance(forgery.details, dict):
            frame_risks_raw = forgery.details.get("frame_risks") or []
            segments_raw = forgery.details.get("suspicious_segments") or []
        module_timelines.append(
            ModuleTimelineItem(
                module="forgery_spatial",
                modelName=forgery.model_name,
                modelVersion=forgery.model_version,
                videoScore=score,
                threshold=threshold,
                detected=detected,
                frameRisks=[FrameRiskItem(**row) for row in frame_risks_raw],
                suspiciousSegments=[SuspiciousSegmentItem(**row) for row in segments_raw],
            )
        )
        frame_edit_detected = detected
        frame_edit_score = score
        forgery_lane_scores = [score]
        reasons.extend(
            build_forgery_analysis_reasons(
                spatial_score=score,
                spatial_detected=detected,
                spatial_threshold=threshold,
                include_temporal=False,
            )
        )
    else:
        reasons.append(forgery_note)
        forgery_lane_scores = []

    # Soft face-gate: deepfake unavailable → risk from forgery only (else 0 / LOW).
    integrated = integrate_risk_score(
        deepfake_score=None,
        deepfake_available=False,
        forgery_scores=forgery_lane_scores,
    )
    if forgery_lane_scores:
        reasons.append(build_integrated_risk_reason(integrated))

    video_result = AnalysisVideoResultItem(
        deepfakeDetected=False,
        deepfakeScore=0.0,
        frameEditDetected=frame_edit_detected,
        frameEditScore=frame_edit_score,
        frameRisks=[],
        clipRisks=[],
        pairRisks=[],
        suspiciousSegments=[],
        temporalSuspiciousSegments=[],
        opticalSuspiciousSegments=[],
        moduleTimelines=module_timelines,
        modelName=FUSION_MODEL_NAME,
        modelVersion=version,
        modelScores=model_scores,
        evidence=reasons,
        representativeFrames=[],
        overlayVideoUrl=None,
        perFrameFaceScores=[],
    )
    return AnalysisResponseMessage(
        analysisRequestId=request.analysisRequestId,
        evidenceId=_resolve_evidence_id(request),
        status="COMPLETED",
        riskScore=integrated.risk_score,
        confidenceScore=0.0,
        riskLevel=integrated.risk_level,  # type: ignore[arg-type]
        analysisReasons=reasons,
        results=[video_result],
        analyzedAt=_utc_now_iso(),
        errorCode=error_code,
        message=message,
        modelName=FUSION_MODEL_NAME,
        modelVersion=version,
        modelScores=video_result.modelScores,
        evidence=reasons,
    )


def _inconclusive_no_human_face_response(
    request: AnalysisRequest,
    *,
    blocked_modules: list[str],
    frames_sampled: int | None = None,
) -> AnalysisResponseMessage:
    return _inconclusive_soft_response(
        request,
        error_code="NO_HUMAN_FACE",
        blocked_modules=blocked_modules,
        frames_sampled=frames_sampled,
    )


def _cnn_gate_error_code(status: str | None) -> str | None:
    if status in NO_HUMAN_FACE_STATUSES:
        return "NO_HUMAN_FACE"
    if status == "face_too_small":
        return "FACE_TOO_SMALL"
    if status == "insufficient_face_samples":
        return "INSUFFICIENT_FACE_SAMPLES"
    return None


def _temporal_is_unavailable(temporal: ModuleInferResult | None) -> bool:
    if temporal is None:
        return True
    if temporal.fake_score is not None and temporal.status == "ok":
        return False
    if temporal.status in NO_HUMAN_FACE_STATUSES:
        # CNN already passed; treat temporal "no face" as module gap, not video-level no-human.
        return True
    if temporal.status in TEMPORAL_UNAVAILABLE_STATUSES or temporal.fake_score is None:
        return True
    return False


def _mock_module_results() -> list[ModuleInferResult]:
    return [
        ModuleInferResult(
            module="cnn",
            model_name="xception",
            model_version="xception/v1.0.0",
            status="ok",
            fake_score=0.72,
            pred_label="real",
            details={"per_frame_scores": [{"frame_index": 10, "fake_score": 0.81}]},
        ),
        ModuleInferResult(
            module="temporal",
            model_name="timesformer",
            model_version="timesformer/v1.0.0",
            status="ok",
            fake_score=0.68,
            pred_label="fake",
            details={
                "per_clip_scores": [
                    {
                        "clip_index": 0,
                        "fake_score": 0.81,
                        "clip_start_frame": 0,
                        "clip_end_frame": 80,
                    }
                ]
            },
        ),
        ModuleInferResult(
            module="optical",
            model_name="gmflow",
            model_version="gmflow/v1.0.0",
            status="ok",
            fake_score=0.31,
            pred_label="real",
            details={
                "pair_stats": [
                    {"frame_index_a": 0, "frame_index_b": 1, "magnitude_mean": 0.42},
                    {"frame_index_a": 35, "frame_index_b": 36, "magnitude_mean": 0.18},
                ]
            },
        ),
    ]


def _module_score_items(
    *,
    fusion_score: float,
    fusion_detected: bool,
    fusion_version: str,
    modules: list[ModuleInferResult],
    config: FusionConfig,
) -> list[ModelScoreItem]:
    items = [
        ModelScoreItem(
            moduleName="deepfake",
            detected=fusion_detected,
            score=round(fusion_score, 6),
            modelName=FUSION_MODEL_NAME,
            modelVersion=fusion_version,
        )
    ]
    module_map = {
        "cnn": "deepfake_cnn",
        "temporal": "deepfake_temporal",
        "optical": "deepfake_optical",
    }
    for module in modules:
        threshold = config.module_thresholds[module.module]
        score = module.fake_score if module.fake_score is not None else 0.0
        items.append(
            ModelScoreItem(
                moduleName=module_map.get(module.module, module.module),
                detected=score_detected(module.fake_score, threshold),
                score=round(score, 6),
                modelName=module.model_name,
                modelVersion=module.model_version,
            )
        )
    return items


def build_response_from_modules(
    request: AnalysisRequest,
    video_path: Path,
    modules: list[ModuleInferResult],
    *,
    config: FusionConfig,
    work_dir: Path | None = None,
) -> AnalysisResponseMessage:
    by_module = {item.module: item for item in modules}
    cnn = by_module.get("cnn")
    temporal = by_module.get("temporal")
    optical = by_module.get("optical")

    s_cnn = cnn.fake_score if cnn else None
    s_temporal = temporal.fake_score if temporal else None
    s_optical = optical.fake_score if optical and optical.fake_score is not None else 0.0

    frames_sampled = None
    if cnn and cnn.details:
        frames_sampled = (cnn.details.get("score_breakdown") or {}).get("frames_sampled")
        if not isinstance(frames_sampled, int):
            frames_sampled = None

    cnn_gate = _cnn_gate_error_code(cnn.status if cnn else None)
    if cnn is not None and cnn.status == "error" and s_cnn is None:
        return _failed_response(
            request,
            error_code="MODEL_INFERENCE_FAILED",
            message="CNN module inference failed.",
        )
    if cnn is None or s_cnn is None or cnn_gate is not None:
        error_code = cnn_gate or "NO_HUMAN_FACE"
        forgery = by_module.get("forgery_spatial")
        if forgery is None:
            forgery = _try_run_forgery_spatial(video_path, work_dir=work_dir)
        return _inconclusive_soft_response(
            request,
            error_code=error_code,
            blocked_modules=["cnn", "forgery_spatial"],
            frames_sampled=frames_sampled,
            forgery=forgery,
        )

    soft_error_code: str | None = None
    soft_error_message: str | None = None
    temporal_unavailable = _temporal_is_unavailable(temporal)
    if temporal_unavailable:
        s_temporal = 0.0
        soft_error_code = "TEMPORAL_MODULE_UNAVAILABLE"
        soft_error_message = (
            "TimeSformer(시계열) 모듈을 사용할 수 없어 CNN·광학 흐름 중심으로 판별했습니다. "
            f"(temporal_status={getattr(temporal, 'status', None)})"
        )

    fusion_score = fuse_scores(
        s_cnn=float(s_cnn),
        s_temporal=float(s_temporal),
        s_optical=float(s_optical),
        config=config,
    )
    fusion_detected = score_detected(fusion_score, config.threshold)
    confidence = confidence_from_module_scores([s_cnn, s_temporal, s_optical])
    reasons = build_analysis_reasons(
        s_cnn=float(s_cnn),
        s_temporal=float(s_temporal),
        s_optical=float(s_optical) if optical and optical.fake_score is not None else None,
        fusion_score=fusion_score,
        fusion_detected=fusion_detected,
        config=config,
    )
    if soft_error_message:
        reasons = [soft_error_message, *reasons]


    per_frame = []
    temporal_per_frame: list[dict[str, Any]] = []
    if cnn and cnn.details:
        per_frame = cnn.details.get("per_frame_scores") or []
    if temporal and temporal.details:
        breakdown = temporal.details.get("score_breakdown") or {}
        temporal_per_frame = breakdown.get("per_frame_scores") or temporal.details.get("per_frame_scores") or []

    fused_per_frame = build_fused_per_frame_scores(
        cnn_scores=per_frame,
        temporal_scores=temporal_per_frame,
        optical_score=float(s_optical),
        fuse_fn=lambda *, cnn_score, temporal_score, optical_score: fuse_scores(
            s_cnn=cnn_score,
            s_temporal=temporal_score,
            s_optical=optical_score,
            config=config,
        ),
        temporal_video_score=float(s_temporal) if s_temporal is not None else None,
    ) if per_frame else []

    frame_risks_raw = collapse_frame_risks_by_frame(fused_per_frame or per_frame, video_path)
    frame_risks = [FrameRiskItem(**row) for row in frame_risks_raw]
    suspicious_raw = build_suspicious_segments(
        frame_risks_raw,
        high_risk_threshold=config.suspicious_segment["high_risk_frame_threshold"],
        min_segment_sec=config.suspicious_segment["min_segment_sec"],
    )
    suspicious_segments = [SuspiciousSegmentItem(**row) for row in suspicious_raw]

    clip_risks_raw: list[dict] = []
    if temporal and temporal.details:
        breakdown = temporal.details.get("score_breakdown") or {}
        clip_risks_raw = build_clip_risks(
            video_path,
            per_clip_scores=temporal.details.get("per_clip_scores") or [],
            per_clip=breakdown.get("per_clip") or temporal.details.get("per_clip") or [],
        )
    clip_risks = [ClipRiskItem(**row) for row in clip_risks_raw]

    pair_risks_raw: list[dict] = []
    if optical and optical.details:
        pair_risks_raw = build_pair_risks(
            video_path,
            optical.details.get("pair_stats") or [],
            per_frame_pair=optical.details.get("per_frame_pair") or None,
        )
    pair_risks = [PairRiskItem(**row) for row in pair_risks_raw]

    module_timelines_raw = build_module_timelines(video_path, modules, config=config)
    module_timelines = [ModuleTimelineItem(**row) for row in module_timelines_raw]

    temporal_segments = [
        SuspiciousSegmentItem(**row)
        for row in next(
            (t["suspiciousSegments"] for t in module_timelines_raw if t["module"] == "temporal"),
            [],
        )
    ]
    optical_segments = [
        SuspiciousSegmentItem(**row)
        for row in next(
            (t["suspiciousSegments"] for t in module_timelines_raw if t["module"] == "optical"),
            [],
        )
    ]

    model_scores = _module_score_items(
        fusion_score=fusion_score,
        fusion_detected=fusion_detected,
        fusion_version=config.fusion_version,
        modules=modules,
        config=config,
    )

    forgery_spatial = by_module.get("forgery_spatial")
    forgery_temporal = by_module.get("forgery_temporal")
    forgery_lane_scores: list[float | None] = []
    frame_edit_detected = False
    frame_edit_score = 0.0
    if (
        forgery_spatial is not None
        and forgery_spatial.status == "ok"
        and forgery_spatial.fake_score is not None
    ):
        forgery_lane_scores.append(float(forgery_spatial.fake_score))
        frame_edit_score = float(forgery_spatial.fake_score)
        thr = 0.515
        if isinstance(forgery_spatial.details, dict) and forgery_spatial.details.get("threshold") is not None:
            thr = float(forgery_spatial.details["threshold"])
        frame_edit_detected = frame_edit_score >= thr
    if (
        forgery_temporal is not None
        and forgery_temporal.status == "ok"
        and forgery_temporal.fake_score is not None
    ):
        forgery_lane_scores.append(float(forgery_temporal.fake_score))

    integrated = integrate_risk_score(
        deepfake_score=fusion_score,
        deepfake_available=True,
        forgery_scores=forgery_lane_scores,
        medium_min=float(config.risk_levels["medium_min"]),
        high_min=float(config.risk_levels["high_min"]),
    )

    if forgery_lane_scores:
        spatial_thr = 0.515
        temporal_thr = 0.173386
        if forgery_spatial is not None and isinstance(forgery_spatial.details, dict):
            if forgery_spatial.details.get("threshold") is not None:
                spatial_thr = float(forgery_spatial.details["threshold"])
        if forgery_temporal is not None and isinstance(forgery_temporal.details, dict):
            if forgery_temporal.details.get("threshold") is not None:
                temporal_thr = float(forgery_temporal.details["threshold"])
        reasons.extend(
            build_forgery_analysis_reasons(
                spatial_score=float(forgery_spatial.fake_score)
                if forgery_spatial is not None and forgery_spatial.fake_score is not None
                else None,
                temporal_score=float(forgery_temporal.fake_score)
                if forgery_temporal is not None and forgery_temporal.fake_score is not None
                else None,
                spatial_detected=frame_edit_detected,
                temporal_detected=bool(
                    forgery_temporal is not None
                    and forgery_temporal.fake_score is not None
                    and float(forgery_temporal.fake_score) >= temporal_thr
                ),
                spatial_threshold=spatial_thr,
                temporal_threshold=temporal_thr,
                include_spatial=forgery_spatial is not None and forgery_spatial.fake_score is not None,
                include_temporal=forgery_temporal is not None and forgery_temporal.fake_score is not None,
            )
        )
        reasons.append(build_integrated_risk_reason(integrated))

    representative_frames: list[RepresentativeFrameItem] = []
    overlay_video_url: str | None = None
    model_overlay_artifacts: list[ModelOverlayArtifactItem] = []
    if work_dir is not None and video_path.is_file():
        try:
            # Representative frames still use fused scores when available.
            if fused_per_frame or per_frame:
                viz = build_visualization_artifacts(
                    video_path=video_path,
                    per_frame_scores=fused_per_frame or per_frame,
                    evidence_id=_resolve_evidence_id(request),
                    analysis_request_id=request.analysisRequestId,
                    work_dir=work_dir / "visualization",
                )
                if viz is not None:
                    representative_frames = [RepresentativeFrameItem(**row) for row in viz.representative_frames]

            overlay_set = build_module_overlay_set(
                video_path=video_path,
                evidence_id=_resolve_evidence_id(request),
                analysis_request_id=request.analysisRequestId,
                work_dir=work_dir / "visualization" / "modules",
                cnn_per_frame_scores=per_frame,
                clip_risks=clip_risks_raw,
                pair_risks=pair_risks_raw,
            )
            overlay_video_url = overlay_set.legacy_cnn_overlay_url
            model_overlay_artifacts = [
                ModelOverlayArtifactItem(**row) for row in overlay_set.model_overlay_artifacts
            ]
            for timeline in module_timelines:
                url = overlay_set.overlay_by_module.get(timeline.module)
                if url:
                    timeline.overlayVideoUrl = url
        except Exception:
            logger.exception(
                "Failed to build visualization artifacts: evidenceId=%s analysisRequestId=%s",
                request.evidenceId or request.fileId,
                request.analysisRequestId,
            )

    video_result = AnalysisVideoResultItem(
        deepfakeDetected=fusion_detected,
        deepfakeScore=round(fusion_score, 6),
        frameEditDetected=frame_edit_detected if forgery_lane_scores else None,
        frameEditScore=round(frame_edit_score, 6) if forgery_lane_scores else None,
        frameRisks=frame_risks,
        clipRisks=clip_risks,
        pairRisks=pair_risks,
        suspiciousSegments=suspicious_segments,
        temporalSuspiciousSegments=temporal_segments,
        opticalSuspiciousSegments=optical_segments,
        moduleTimelines=module_timelines,
        modelName=FUSION_MODEL_NAME,
        modelVersion=config.fusion_version,
        modelScores=model_scores,
        evidence=reasons,
        representativeFrames=representative_frames,
        overlayVideoUrl=overlay_video_url,
        modelOverlayArtifacts=model_overlay_artifacts,
        perFrameFaceScores=_to_per_frame_face_scores(fused_per_frame) if fused_per_frame else [],
    )

    return AnalysisResponseMessage(
        analysisRequestId=request.analysisRequestId,
        evidenceId=_resolve_evidence_id(request),
        status="COMPLETED",
        riskScore=integrated.risk_score,
        confidenceScore=confidence,
        riskLevel=integrated.risk_level,  # type: ignore[arg-type]
        analysisReasons=reasons,
        results=[video_result],
        analyzedAt=_utc_now_iso(),
        errorCode=soft_error_code,
        message=soft_error_message,
        modelName=FUSION_MODEL_NAME,
        modelVersion=config.fusion_version,
        modelScores=model_scores,
        evidence=reasons,
    )


def analyze_video_request(
    request: AnalysisRequest,
    *,
    settings: ModelSettings | None = None,
) -> AnalysisResponseMessage:
    settings = settings or load_model_settings()
    config = load_fusion_config(settings.fusion_config_path)

    try:
        _resolve_evidence_id(request)
    except ValueError as exc:
        return _failed_response(request, error_code="INVALID_REQUEST", message=str(exc))

    if request.fileType != "video":
        return _failed_response(
            request,
            error_code="UNSUPPORTED_FILE_TYPE",
            message=f"Only video is supported, got: {request.fileType}",
        )

    if settings.use_mock_infer:
        return build_response_from_modules(
            request,
            Path("mock.mp4"),
            _mock_module_results(),
            config=config,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="forenshield-analyze-") as tmp:
            video_path = _download_video(request, Path(tmp))
            _verify_sha256(video_path, _resolve_original_hash(request))
            runtime = InferRuntime(settings)
            modules = runtime.analyze_modules(video_path)
            return build_response_from_modules(
                request,
                video_path,
                modules,
                config=config,
                work_dir=Path(tmp),
            )
    except FileNotFoundError as exc:
        return _failed_response(request, error_code="MODEL_WEIGHTS_NOT_FOUND", message=str(exc))
    except requests.RequestException as exc:
        return _failed_response(request, error_code="VIDEO_DOWNLOAD_FAILED", message=str(exc))
    except ValueError as exc:
        return _failed_response(request, error_code="VALIDATION_FAILED", message=str(exc))
    except Exception as exc:  # noqa: BLE001
        return _failed_response(
            request,
            error_code="MODEL_INFERENCE_FAILED",
            message=str(exc),
        )
