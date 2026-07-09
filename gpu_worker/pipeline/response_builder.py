from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    SuspiciousSegmentItem,
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
