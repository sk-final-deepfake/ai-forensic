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
    FrameRiskItem,
    ModelScoreItem,
    ModuleTimelineItem,
    PairRiskItem,
    RepresentativeFrameItem,
    SuspiciousSegmentItem,
)
from app.services.visualization_artifacts import build_visualization_artifacts
from app.schemas.analysis import AnalysisRequest
from app.services.infer_bridge import InferRuntime, ModuleInferResult
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
    risk_level_from_score,
    score_detected,
)

logger = logging.getLogger("ai_fastapi.video_deepfake_analyzer")

FUSION_MODEL_NAME = "forenshield-late-fusion"
NO_HUMAN_FACE_STATUSES = frozenset({"no_face", "no_human_face", "skipped_no_human_face"})


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

    blocked_modules = [
        m.module
        for m in (cnn, temporal)
        if m is not None and (m.status in NO_HUMAN_FACE_STATUSES or m.fake_score is None)
    ]
    if blocked_modules:
        frames_sampled = None
        if cnn and cnn.details:
            frames_sampled = (cnn.details.get("score_breakdown") or {}).get("frames_sampled")
        detail = (
            f" sampled_frames={frames_sampled}" if frames_sampled is not None else ""
        )
        return _failed_response(
            request,
            error_code="NO_HUMAN_FACE",
            message=(
                "사람 얼굴이 검출되지 않아 딥페이크 판별을 수행할 수 없습니다."
                f" (modules={','.join(blocked_modules)}{detail})"
            ),
        )

    missing = [
        name
        for name, score in (("cnn", s_cnn), ("temporal", s_temporal))
        if score is None
    ]
    if missing:
        return _failed_response(
            request,
            error_code="MODEL_INFERENCE_FAILED",
            message=f"Required module scores missing: {', '.join(missing)}",
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

    representative_frames: list[RepresentativeFrameItem] = []
    overlay_video_url: str | None = None
    if work_dir is not None and (fused_per_frame or per_frame) and video_path.is_file():
        try:
            viz = build_visualization_artifacts(
                video_path=video_path,
                per_frame_scores=fused_per_frame or per_frame,
                evidence_id=_resolve_evidence_id(request),
                analysis_request_id=request.analysisRequestId,
                work_dir=work_dir / "visualization",
            )
            if viz is not None:
                representative_frames = [RepresentativeFrameItem(**row) for row in viz.representative_frames]
                overlay_video_url = viz.overlay_video_url
        except Exception:
            logger.exception(
                "Failed to build visualization artifacts: evidenceId=%s analysisRequestId=%s",
                request.evidenceId or request.fileId,
                request.analysisRequestId,
            )

    video_result = AnalysisVideoResultItem(
        deepfakeDetected=fusion_detected,
        deepfakeScore=round(fusion_score, 6),
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
    )

    return AnalysisResponseMessage(
        analysisRequestId=request.analysisRequestId,
        evidenceId=_resolve_evidence_id(request),
        status="COMPLETED",
        riskScore=round(fusion_score * 100.0, 2),
        confidenceScore=confidence,
        riskLevel=risk_level_from_score(fusion_score, config),
        analysisReasons=reasons,
        results=[video_result],
        analyzedAt=_utc_now_iso(),
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
