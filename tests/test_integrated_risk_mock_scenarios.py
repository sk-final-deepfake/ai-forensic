"""10 mock end-to-end scenarios for dynamic-weighted riskScore."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.services.integrated_risk import (
    dynamic_weighted_mean01,
    forgery_scores_from_lane_result,
    integrate_risk_score,
)


@dataclass
class _MockForgeryLane:
    lane_ran: bool
    spatial_score: float = 0.0
    temporal_score: float = 0.0


MOCK_SCENARIOS = [
    {
        "id": "01_normal_real_both_low",
        "kwargs": {"deepfake_score": 0.08, "forgery_scores": [0.05]},
        "expected_score": round(dynamic_weighted_mean01(0.08, 0.05) * 100, 2),
        "expected_method": "dynamic_weighted_deepfake_forgery",
    },
    {
        "id": "02_deepfake_miss_forgery_catches",
        "kwargs": {"deepfake_score": 0.12, "forgery_scores": [0.88]},
        "expected_score": round(dynamic_weighted_mean01(0.12, 0.88) * 100, 2),
        "expected_method": "dynamic_weighted_deepfake_forgery",
    },
    {
        "id": "03_both_high_dynamic_weight",
        "kwargs": {"deepfake_score": 0.75, "forgery_scores": [0.60, 0.82]},
        "expected_score": round(dynamic_weighted_mean01(0.75, 0.82) * 100, 2),
        "expected_method": "dynamic_weighted_deepfake_forgery",
    },
    {
        "id": "04_forgery_skip_deepfake_only",
        "kwargs": {"deepfake_score": 0.72, "forgery_scores": []},
        "expected_score": 72.0,
        "expected_method": "deepfake_only",
    },
    {
        "id": "05_soft_face_gate_forgery_only",
        "kwargs": {
            "deepfake_score": 0.99,
            "deepfake_available": False,
            "forgery_scores": [0.55],
        },
        "expected_score": 55.0,
        "expected_method": "forgery_only",
    },
    {
        "id": "06_lane_disabled_no_forgery_in_merge",
        "lane": _MockForgeryLane(lane_ran=False, spatial_score=0.0, temporal_score=0.0),
        "deepfake_score": 0.65,
        "expected_score": 65.0,
        "expected_method": "deepfake_only",
    },
    {
        "id": "07_lane_enabled_spatial_temporal_max",
        "lane": _MockForgeryLane(lane_ran=True, spatial_score=0.30, temporal_score=0.71),
        "deepfake_score": 0.22,
        "expected_score": round(dynamic_weighted_mean01(0.22, 0.71) * 100, 2),
        "expected_method": "dynamic_weighted_deepfake_forgery",
    },
    {
        "id": "08_medium_boundary",
        "kwargs": {"deepfake_score": 0.40, "forgery_scores": [0.40]},
        "expected_score": round(dynamic_weighted_mean01(0.40, 0.40) * 100, 2),
        "expected_method": "dynamic_weighted_deepfake_forgery",
    },
    {
        "id": "09_high_boundary",
        "kwargs": {"deepfake_score": 0.70, "forgery_scores": [0.70]},
        "expected_score": 70.0,
        "expected_method": "dynamic_weighted_deepfake_forgery",
    },
    {
        "id": "10_both_unavailable",
        "kwargs": {"deepfake_score": None, "forgery_scores": []},
        "expected_score": 0.0,
        "expected_method": "none",
    },
]


@pytest.mark.parametrize(
    "scenario",
    MOCK_SCENARIOS,
    ids=[s["id"] for s in MOCK_SCENARIOS],
)
def test_mock_scenario(scenario: dict) -> None:
    if "lane" in scenario:
        forgery_scores = forgery_scores_from_lane_result(scenario["lane"])
        result = integrate_risk_score(
            deepfake_score=scenario["deepfake_score"],
            forgery_scores=forgery_scores,
        )
    else:
        result = integrate_risk_score(**scenario["kwargs"])

    assert result.risk_score == scenario["expected_score"], (
        f"{scenario['id']}: score {result.risk_score} != {scenario['expected_score']}"
    )
    assert result.method == scenario["expected_method"], (
        f"{scenario['id']}: method {result.method} != {scenario['expected_method']}"
    )
