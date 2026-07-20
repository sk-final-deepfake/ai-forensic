"""Integrated top-level riskScore from deepfake + forgery lanes.

Product rule (2026-07):
  riskScore (0~100) = max(available lanes) * 100
  deepfakeScore stays fusion-only (0~1) for the deepfake tab.

Exceptions:
  - forgery missing/failed/skipped → deepfake only
  - deepfake soft-incomplete (face gate) → forgery only
  - both unavailable → 0.0 / LOW
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


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
    """
    df: float | None = None
    if deepfake_available and deepfake_score is not None:
        df = _clamp01(deepfake_score)

    forgery = _max_valid_scores(forgery_scores)

    candidates: list[float] = []
    if df is not None:
        candidates.append(df)
    if forgery is not None:
        candidates.append(forgery)

    if not candidates:
        return IntegratedRiskResult(
            risk_score=0.0,
            risk_level="LOW",
            deepfake_score_01=None,
            forgery_score_01=None,
            method="none",
        )

    peak = max(candidates)
    risk_score = round(peak * 100.0, 2)
    if df is not None and forgery is not None:
        method = "max_deepfake_forgery"
    elif df is not None:
        method = "deepfake_only"
    else:
        method = "forgery_only"

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
