"""Integrated top-level riskScore from deepfake + forgery lanes.

Product rule (2026-07, dynamic weight):
  When both Late Fusion (F) and forgery-max (G) are available:
    riskScore = ((F^2 + G^2) / (F + G)) * 100
    i.e. weighted mean with weights proportional to each score.
  deepfakeScore stays fusion-only (0~1) for the deepfake tab.

Exceptions:
  - forgery missing/failed/skipped → deepfake only
  - deepfake soft-incomplete (face gate) → forgery only
  - both unavailable → 0.0 / LOW
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


DEFAULT_MEDIUM_MIN = 40.0
DEFAULT_HIGH_MIN = 70.0


@dataclass(frozen=True)
class IntegratedRiskResult:
    risk_score: float
    risk_level: str
    deepfake_score_01: float | None
    forgery_score_01: float | None
    method: str


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _risk_level_from_100(
    risk_score: float,
    *,
    medium_min: float = DEFAULT_MEDIUM_MIN,
    high_min: float = DEFAULT_HIGH_MIN,
) -> str:
    if risk_score >= high_min:
        return "HIGH"
    if risk_score >= medium_min:
        return "MEDIUM"
    return "LOW"


def _max_valid_scores(scores: Iterable[float | None]) -> float | None:
    valid = [_clamp01(s) for s in scores if s is not None]
    if not valid:
        return None
    return max(valid)


def dynamic_weighted_mean01(fusion: float, forgery: float) -> float:
    """Score-proportional weights: w_i = s_i / (F+G); result = w_F*F + w_G*G."""
    f = _clamp01(fusion)
    g = _clamp01(forgery)
    total = f + g
    if total <= 0.0:
        return 0.0
    return (f * f + g * g) / total


def integrate_risk_score(
    *,
    deepfake_score: float | None = None,
    forgery_scores: Sequence[float | None] = (),
    deepfake_available: bool = True,
    medium_min: float = DEFAULT_MEDIUM_MIN,
    high_min: float = DEFAULT_HIGH_MIN,
) -> IntegratedRiskResult:
    """Combine deepfake (0~1) and forgery lane scores (0~1 each) into riskScore 0~100.

    ``forgery_scores`` may include spatial and/or temporal; invalid/None entries are ignored
    and the forgery lane uses the max of remaining values.
    When both lanes exist, apply dynamic weighted mean (not plain max / not plain average).
    """
    df: float | None = None
    if deepfake_available and deepfake_score is not None:
        df = _clamp01(deepfake_score)

    forgery = _max_valid_scores(forgery_scores)

    if df is not None and forgery is not None:
        peak = dynamic_weighted_mean01(df, forgery)
        method = "dynamic_weighted_deepfake_forgery"
    elif df is not None:
        peak = df
        method = "deepfake_only"
    elif forgery is not None:
        peak = forgery
        method = "forgery_only"
    else:
        return IntegratedRiskResult(
            risk_score=0.0,
            risk_level="LOW",
            deepfake_score_01=None,
            forgery_score_01=None,
            method="none",
        )

    risk_score = round(peak * 100.0, 2)
    return IntegratedRiskResult(
        risk_score=risk_score,
        risk_level=_risk_level_from_100(risk_score, medium_min=medium_min, high_min=high_min),
        deepfake_score_01=df,
        forgery_score_01=forgery,
        method=method,
    )


def forgery_scores_from_success(
    *,
    spatial_score: float | None = None,
    temporal_score: float | None = None,
    spatial_ok: bool = False,
    temporal_ok: bool = False,
) -> list[float | None]:
    """Build forgery score list for integrate_risk_score from run success flags."""
    scores: list[float | None] = []
    if spatial_ok and spatial_score is not None:
        scores.append(spatial_score)
    if temporal_ok and temporal_score is not None:
        scores.append(temporal_score)
    return scores


def forgery_scores_from_lane_result(forgery: Any | None) -> list[float | None]:
    """Build forgery score list from GPU ForgeryLaneResult (skip when lane disabled)."""
    if forgery is None or not bool(getattr(forgery, "lane_ran", False)):
        return []
    return forgery_scores_from_success(
        spatial_score=float(getattr(forgery, "spatial_score", 0.0)),
        temporal_score=float(getattr(forgery, "temporal_score", 0.0)),
        spatial_ok=True,
        temporal_ok=True,
    )


def build_forgery_analysis_reasons(
    *,
    spatial_score: float | None = None,
    temporal_score: float | None = None,
    spatial_detected: bool = False,
    temporal_detected: bool = False,
    spatial_threshold: float = 0.515,
    temporal_threshold: float = 0.173386,
    include_spatial: bool = True,
    include_temporal: bool = True,
) -> list[str]:
    """Human-readable forgery lane lines for analysisReasons / 종합 소견."""
    lines: list[str] = []
    if include_spatial and spatial_score is not None:
        lines.append(
            f"Forgery spatial (TruFor) fake_score={_clamp01(spatial_score):.3f} "
            f"({'fake' if spatial_detected else 'real'}) @ T={spatial_threshold:.3f}"
        )
    if include_temporal and temporal_score is not None:
        lines.append(
            f"Forgery temporal (TimeSformer) fake_score={_clamp01(temporal_score):.3f} "
            f"({'fake' if temporal_detected else 'real'}) @ T={temporal_threshold:.3f}"
        )
    return lines


def build_integrated_risk_reason(result: IntegratedRiskResult) -> str:
    """Single-line integrated risk summary for analysisReasons / 종합 소견."""
    method_labels = {
        "dynamic_weighted_deepfake_forgery": "dynamic weighted deepfake+forgery",
        "deepfake_only": "deepfake only",
        "forgery_only": "forgery only",
        "none": "unavailable",
    }
    label = method_labels.get(result.method, result.method)
    return (
        f"Integrated risk ({label}) riskScore={result.risk_score:.2f} "
        f"→ {result.risk_level}"
    )
