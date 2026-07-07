from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FusionResult:
    score: float
    detected: bool
    confidence: float
    risk_score: float
    risk_level: str
    reasons: list[str]


def load_fusion_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _risk_level(score_0_100: float, cfg: dict[str, Any]) -> str:
    levels = cfg.get("risk_levels") or {}
    high = float(levels.get("high", 70.0))
    medium = float(levels.get("medium", 40.0))
    if score_0_100 >= high:
        return "HIGH"
    if score_0_100 >= medium:
        return "MEDIUM"
    return "LOW"


def apply_late_fusion(
    *,
    cnn_score: float,
    temporal_score: float,
    optical_score: float,
    config: dict[str, Any],
    module_meta: dict[str, dict[str, str]],
) -> FusionResult:
    method = str(config.get("method", "logistic_gated"))
    threshold = float(config.get("threshold", 0.5))
    module_thresholds = config.get("module_thresholds") or {}
    gating = config.get("gating") or {}
    gated_enabled = bool(gating.get("enabled", True))

    inputs = {
        "cnn": float(cnn_score),
        "temporal": float(temporal_score),
        "optical": float(optical_score),
    }
    if gated_enabled:
        for key, value in list(inputs.items()):
            gate = float(module_thresholds.get(key, 0.0))
            if value < gate and gate > 0:
                inputs[key] = value

    if method in ("logistic", "logistic_gated"):
        coef = config.get("coefficients") or {}
        intercept = float(config.get("intercept", 0.0))
        logit = intercept
        for key, value in inputs.items():
            logit += float(coef.get(key, 0.0)) * value
        fusion_score = _sigmoid(logit)
    elif method == "weighted_avg":
        weights = config.get("weights") or {"cnn": 0.45, "temporal": 0.35, "optical": 0.2}
        total_w = sum(float(weights.get(k, 0.0)) for k in inputs)
        if total_w <= 0:
            fusion_score = sum(inputs.values()) / max(len(inputs), 1)
        else:
            fusion_score = sum(float(weights.get(k, 0.0)) * v for k, v in inputs.items()) / total_w
    else:
        fusion_score = sum(inputs.values()) / max(len(inputs), 1)

    fusion_score = round(min(1.0, max(0.0, fusion_score)), 4)
    detected = fusion_score >= threshold
    confidence = round(max(fusion_score, 1.0 - fusion_score), 4)
    risk_score = round(fusion_score * 100.0, 1)
    risk_level = _risk_level(risk_score, config)

    fusion_models = (config.get("models") or {}).get("fusion") or {}
    cnn_meta = module_meta.get("cnn") or (config.get("models") or {}).get("cnn") or {}
    temporal_meta = module_meta.get("temporal") or (config.get("models") or {}).get("temporal") or {}
    optical_meta = module_meta.get("optical") or (config.get("models") or {}).get("optical") or {}

    reasons = [
        (
            f"Late Fusion ({fusion_models.get('modelName', 'Late Fusion')}/"
            f"{fusion_models.get('modelVersion', config.get('fusion_version', 'v3'))}) "
            f"score {fusion_score:.2f} (threshold {threshold:.2f})"
        ),
        (
            f"Xception ({cnn_meta.get('modelName', 'Xception')}) "
            f"fake probability {inputs['cnn']:.2f}"
        ),
        (
            f"TimeSformer ({temporal_meta.get('modelName', 'TimeSformer')}) "
            f"fake probability {inputs['temporal']:.2f}"
        ),
        (
            f"GMFlow ({optical_meta.get('modelName', 'GMFlow')}) "
            f"fake probability {inputs['optical']:.2f}"
        ),
    ]
    if detected:
        reasons.append("Late Fusion threshold exceeded — deepfake detected")

    return FusionResult(
        score=fusion_score,
        detected=detected,
        confidence=confidence,
        risk_score=risk_score,
        risk_level=risk_level,
        reasons=reasons,
    )
