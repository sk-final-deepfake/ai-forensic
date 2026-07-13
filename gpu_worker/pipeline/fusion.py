from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.late_fusion import (
    FusionConfig,
    build_analysis_reasons,
    confidence_from_module_scores,
    fuse_scores,
    load_fusion_config as load_typed_fusion_config,
    risk_level_from_score,
    score_detected,
)


@dataclass(frozen=True)
class FusionResult:
    score: float
    detected: bool
    confidence: float
    risk_score: float
    risk_level: str
    reasons: list[str]


def load_fusion_config(path: Path) -> dict[str, Any]:
    """Load raw JSON for metadata helpers; scoring uses typed FusionConfig."""
    return json.loads(path.read_text(encoding="utf-8"))


def apply_late_fusion(
    *,
    cnn_score: float,
    temporal_score: float,
    optical_score: float,
    config: dict[str, Any],
    module_meta: dict[str, dict[str, str]],
) -> FusionResult:
    """Delegate gated/weighted fusion to the shared FastAPI late_fusion module."""
    del module_meta  # reasons are built from typed config + scores
    typed = FusionConfig.from_dict(config)
    fusion_score = fuse_scores(
        s_cnn=float(cnn_score),
        s_temporal=float(temporal_score),
        s_optical=float(optical_score),
        config=typed,
    )
    detected = score_detected(fusion_score, typed.threshold)
    confidence = confidence_from_module_scores([cnn_score, temporal_score, optical_score])
    risk_score = round(fusion_score * 100.0, 2)
    risk_level = risk_level_from_score(fusion_score, typed)
    reasons = build_analysis_reasons(
        s_cnn=float(cnn_score),
        s_temporal=float(temporal_score),
        s_optical=float(optical_score),
        fusion_score=fusion_score,
        fusion_detected=detected,
        config=typed,
    )
    return FusionResult(
        score=fusion_score,
        detected=detected,
        confidence=confidence,
        risk_score=risk_score,
        risk_level=risk_level,
        reasons=reasons,
    )


def load_typed_fusion_config_path(path: Path) -> FusionConfig:
    return load_typed_fusion_config(path)
