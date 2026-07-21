"""Unit tests for integrated riskScore = dynamic weighted (F,G) with fixed exceptions.

100 table-driven scenarios: normal, missed detection, soft gate, forgery skip, boundaries.
"""

from __future__ import annotations

import pytest

from app.services.integrated_risk import (
    dynamic_weighted_mean01,
    forgery_scores_from_lane_result,
    forgery_scores_from_success,
    integrate_risk_score,
)


def _level(score: float, *, medium: float = 40.0, high: float = 70.0) -> str:
    if score >= high:
        return "HIGH"
    if score >= medium:
        return "MEDIUM"
    return "LOW"


def _method(*, df: float | None, forgery: float | None) -> str:
    if df is not None and forgery is not None:
        return "dynamic_weighted_deepfake_forgery"
    if df is not None:
        return "deepfake_only"
    if forgery is not None:
        return "forgery_only"
    return "none"


def _peak100(df: float | None, forgery: float | None) -> float:
    if df is not None and forgery is not None:
        return round(dynamic_weighted_mean01(df, forgery) * 100.0, 2)
    if df is not None:
        return round(df * 100.0, 2)
    if forgery is not None:
        return round(forgery * 100.0, 2)
    return 0.0


def _case(
    case_id: str,
    *,
    deepfake_score: float | None = None,
    forgery_scores: list[float | None] | None = None,
    deepfake_available: bool = True,
    medium_min: float = 40.0,
    high_min: float = 70.0,
) -> tuple[str, dict, float, str, str]:
    """Build kwargs + expected tuple from product rule (independent oracle)."""
    forgery_scores = forgery_scores if forgery_scores is not None else []

    df: float | None = None
    if deepfake_available and deepfake_score is not None:
        df = max(0.0, min(1.0, float(deepfake_score)))

    forgery_vals = [max(0.0, min(1.0, float(s))) for s in forgery_scores if s is not None]
    forgery: float | None = max(forgery_vals) if forgery_vals else None

    risk = _peak100(df, forgery)
    kwargs: dict = {
        "deepfake_score": deepfake_score,
        "forgery_scores": forgery_scores,
        "deepfake_available": deepfake_available,
        "medium_min": medium_min,
        "high_min": high_min,
    }
    return (
        case_id,
        kwargs,
        risk,
        _level(risk, medium=medium_min, high=high_min),
        _method(df=df, forgery=forgery),
    )


def _build_100_cases() -> list[tuple[str, dict, float, str, str]]:
    cases: list[tuple[str, dict, float, str, str]] = []

    # --- A. 정상 / 클린 (both low) ---
    clean_pairs: list[tuple[float, list[float | None]]] = [
        (0.05, [0.03]),
        (0.08, [0.12]),
        (0.15, [0.10]),
        (0.02, [0.01]),
        (0.18, [0.22]),
        (0.25, [0.20]),
        (0.30, [0.28]),
        (0.12, []),
        (0.09, []),
        (0.11, [0.07, 0.04]),
    ]
    for i, (df, fg_list) in enumerate(clean_pairs, start=1):
        cases.append(_case(f"A{i:02d}_clean", deepfake_score=df, forgery_scores=fg_list))

    # --- B. 딥페이크만 높음 (위변조 낮음 / 미탐 아님 — DF가 주도) ---
    for i, (df, fg) in enumerate(
        [
            (0.92, 0.10),
            (0.85, 0.05),
            (0.78, 0.20),
            (0.71, 0.15),
            (0.65, 0.08),
            (0.58, 0.12),
            (0.95, 0.30),
            (0.88, 0.00),
            (0.82, 0.25),
            (0.76, 0.18),
        ],
        start=1,
    ):
        cases.append(_case(f"B{i:02d}_df_dominant", deepfake_score=df, forgery_scores=[fg]))

    # --- C. 위변조만 높음 (딥페이크 낮음 — DF 미탐, forgery가 구제) ---
    for i, (df, fg) in enumerate(
        [
            (0.15, 0.88),
            (0.22, 0.91),
            (0.10, 0.75),
            (0.08, 0.82),
            (0.30, 0.95),
            (0.12, 0.68),
            (0.05, 0.72),
            (0.18, 0.85),
            (0.25, 0.80),
            (0.20, 0.77),
        ],
        start=1,
    ):
        cases.append(_case(f"C{i:02d}_forgery_rescue", deepfake_score=df, forgery_scores=[fg]))

    # --- D. 둘 다 높음 (max는 더 높은 쪽) ---
    for i, (df, fg) in enumerate(
        [
            (0.90, 0.85),
            (0.85, 0.90),
            (0.95, 0.92),
            (0.70, 0.71),
            (0.71, 0.70),
            (0.88, 0.88),
            (0.75, 0.92),
            (0.92, 0.75),
            (0.99, 0.99),
            (0.60, 0.65),
        ],
        start=1,
    ):
        cases.append(_case(f"D{i:02d}_both_high", deepfake_score=df, forgery_scores=[fg]))

    # --- E. 위변조 스킵/실패 → deepfake only ---
    for i, df in enumerate(
        [0.0, 0.12, 0.45, 0.58, 0.66, 0.70, 0.81, 0.95, 0.578, 0.42],
        start=1,
    ):
        cases.append(_case(f"E{i:02d}_forgery_skip", deepfake_score=df, forgery_scores=[]))

    # --- F. Soft face-gate (딥페이크 보류) → forgery only ---
    for i, (df_leak, fg) in enumerate(
        [
            (0.99, 0.77),
            (0.95, 0.50),
            (0.88, 0.15),
            (0.0, 0.88),
            (None, 0.65),
            (0.75, 0.72),
            (0.50, 0.40),
            (0.80, 0.0),
            (0.91, 0.83),
            (None, 0.0),
        ],
        start=1,
    ):
        cases.append(
            _case(
                f"F{i:02d}_soft_gate",
                deepfake_score=df_leak,
                forgery_scores=[fg],
                deepfake_available=False,
            )
        )

    # --- G. Soft gate + forgery도 없음 ---
    for i, df_leak in enumerate([0.99, 0.50, 0.0, None], start=1):
        cases.append(
            _case(
                f"G{i:02d}_soft_no_forgery",
                deepfake_score=df_leak,
                forgery_scores=[None],
                deepfake_available=False,
            )
        )

    # --- H. 경계값 (40 / 70) ---
    boundaries = [
        (0.399, 0.10),
        (0.400, 0.10),
        (0.401, 0.10),
        (0.699, 0.10),
        (0.700, 0.10),
        (0.701, 0.10),
        (0.10, 0.399),
        (0.10, 0.400),
        (0.10, 0.700),
        (0.10, 0.701),
    ]
    for i, (df, fg) in enumerate(boundaries, start=1):
        cases.append(_case(f"H{i:02d}_boundary", deepfake_score=df, forgery_scores=[fg]))

    # --- I. 운영 유사 점수 (fusion T≈0.578, TruFor T≈0.515, temporal≈0.173) ---
    ops = [
        (0.578, 0.20, "fusion_at_threshold_forgery_low"),
        (0.577, 0.60, "fusion_just_below_forgery_wins"),
        (0.20, 0.515, "trufor_at_threshold"),
        (0.20, 0.516, "trufor_just_above"),
        (0.578, 0.515, "both_at_threshold_max_578"),
        (0.40, 0.173, "temporal_low_spatial_mid"),
        (0.90, 0.173386, "df_high_temporal_typical"),
        (0.15, 0.85, "df_miss_forgery_high"),
        (0.85, 0.15, "df_high_forgery_clean"),
        (0.578, None, "fusion_only_no_forgery"),
    ]
    for i, (df, fg, tag) in enumerate(ops, start=1):
        fg_list = [] if fg is None else [fg]
        cases.append(_case(f"I{i:02d}_{tag}", deepfake_score=df, forgery_scores=fg_list))

    # --- J. spatial + temporal forgery max ---
    dual_forgery = [
        (0.30, [0.40, 0.92]),
        (0.30, [0.92, 0.40]),
        (0.30, [None, 0.80]),
        (0.30, [0.55, 0.55]),
        (0.90, [0.40, 0.60]),
        (0.40, [0.60, 0.90]),
    ]
    for i, (df, fgs) in enumerate(dual_forgery, start=1):
        cases.append(_case(f"J{i:02d}_dual_forgery", deepfake_score=df, forgery_scores=fgs))

    # --- K. clamp / rounding ---
    clamp_cases = [
        (1.2, [0.1]),
        (0.1, [1.5]),
        (-0.3, [-0.1]),
        (0.12345, [0.12340]),
        (0.0, [0.0]),
        (0.001, [0.002]),
    ]
    for i, (df, fgs) in enumerate(clamp_cases, start=1):
        cases.append(_case(f"K{i:02d}_clamp", deepfake_score=df, forgery_scores=fgs))

    # --- L. custom risk bands ---
    custom = [
        (0.60, [0.10], 30.0, 60.0),
        (0.35, [0.10], 30.0, 60.0),
        (0.29, [0.10], 30.0, 60.0),
        (0.60, [0.65], 30.0, 60.0),
    ]
    for i, (df, fgs, med, hi) in enumerate(custom, start=1):
        cases.append(
            _case(
                f"L{i:02d}_custom_band",
                deepfake_score=df,
                forgery_scores=fgs,
                medium_min=med,
                high_min=hi,
            )
        )

    return cases


INTEGRATED_RISK_CASES = _build_100_cases()
assert len(INTEGRATED_RISK_CASES) == 100, f"expected 100 cases, got {len(INTEGRATED_RISK_CASES)}"


@pytest.mark.parametrize(
    "case_id,kwargs,expected_score,expected_level,expected_method",
    INTEGRATED_RISK_CASES,
    ids=[c[0] for c in INTEGRATED_RISK_CASES],
)
def test_integrate_risk_score_100_cases(
    case_id: str,
    kwargs: dict,
    expected_score: float,
    expected_level: str,
    expected_method: str,
) -> None:
    result = integrate_risk_score(**kwargs)
    assert result.risk_score == expected_score, f"{case_id}: score"
    assert result.risk_level == expected_level, f"{case_id}: level"
    assert result.method == expected_method, f"{case_id}: method"


# --- Scenario narratives (product semantics) ---


def test_scenario_normal_real_both_low() -> None:
    """정상 real: fusion·TruFor 모두 낮음 → 종합 LOW."""
    r = integrate_risk_score(deepfake_score=0.08, forgery_scores=[0.05])
    assert r.risk_score == round(dynamic_weighted_mean01(0.08, 0.05) * 100.0, 2)
    assert r.risk_level == "LOW"


def test_scenario_deepfake_miss_forgery_catches() -> None:
    """딥페이크 미탐(낮은 fusion) + 편집 흔적(높은 TruFor) → 동적 가중으로 반영."""
    r = integrate_risk_score(deepfake_score=0.12, forgery_scores=[0.88])
    expected = round(dynamic_weighted_mean01(0.12, 0.88) * 100.0, 2)
    assert r.risk_score == expected
    assert r.risk_level == _level(expected)
    assert r.method == "dynamic_weighted_deepfake_forgery"


def test_scenario_forgery_skip_deepfake_only() -> None:
    """TruFor 스킵 → fusion만 종합."""
    r = integrate_risk_score(deepfake_score=0.72, forgery_scores=[])
    assert r.risk_score == 72.0
    assert r.method == "deepfake_only"


def test_scenario_face_gate_forgery_only() -> None:
    """얼굴 없음 soft complete: fusion 0.99여도 무시, TruFor만."""
    r = integrate_risk_score(
        deepfake_score=0.99,
        deepfake_available=False,
        forgery_scores=[0.55],
    )
    assert r.risk_score == 55.0
    assert r.deepfake_score_01 is None


def test_scenario_face_gate_no_forgery_zero() -> None:
    """얼굴 없음 + TruFor도 없음 → 0."""
    r = integrate_risk_score(
        deepfake_score=0.99,
        deepfake_available=False,
        forgery_scores=[],
    )
    assert r.risk_score == 0.0
    assert r.method == "none"


def test_forgery_scores_from_success_both_modules() -> None:
    scores = forgery_scores_from_success(
        spatial_score=0.55,
        temporal_score=0.82,
        spatial_ok=True,
        temporal_ok=True,
    )
    assert scores == [0.55, 0.82]
    r = integrate_risk_score(deepfake_score=0.20, forgery_scores=scores)
    assert r.risk_score == round(dynamic_weighted_mean01(0.20, 0.82) * 100.0, 2)


def test_forgery_scores_from_success_spatial_only() -> None:
    scores = forgery_scores_from_success(
        spatial_score=0.70,
        temporal_score=0.90,
        spatial_ok=True,
        temporal_ok=False,
    )
    r = integrate_risk_score(deepfake_score=0.20, forgery_scores=scores)
    assert r.risk_score == round(dynamic_weighted_mean01(0.20, 0.70) * 100.0, 2)


def test_dynamic_weight_between_mean_and_max() -> None:
    """F=0.2, G=0.9 → 동적 가중은 단순평균(0.55)과 max(0.9) 사이."""
    peak = dynamic_weighted_mean01(0.2, 0.9)
    assert 0.55 < peak < 0.9
    assert abs(peak - (0.04 + 0.81) / 1.1) < 1e-9


def test_forgery_scores_from_lane_result_disabled_lane() -> None:
    class _Lane:
        lane_ran = False
        spatial_score = 0.0
        temporal_score = 0.0

    assert forgery_scores_from_lane_result(_Lane()) == []


def test_forgery_scores_from_lane_result_enabled_lane() -> None:
    class _Lane:
        lane_ran = True
        spatial_score = 0.12
        temporal_score = 0.88

    scores = forgery_scores_from_lane_result(_Lane())
    r = integrate_risk_score(deepfake_score=0.20, forgery_scores=scores)
    assert r.risk_score == round(dynamic_weighted_mean01(0.20, 0.88) * 100.0, 2)


def test_build_forgery_analysis_reasons_includes_spatial_and_temporal() -> None:
    from app.services.integrated_risk import (
        build_forgery_analysis_reasons,
        build_integrated_risk_reason,
    )

    lines = build_forgery_analysis_reasons(
        spatial_score=0.12,
        temporal_score=0.88,
        spatial_detected=False,
        temporal_detected=True,
        spatial_threshold=0.515,
        temporal_threshold=0.173,
    )
    assert len(lines) == 2
    assert "Forgery spatial (TruFor)" in lines[0]
    assert "Forgery temporal (TimeSformer)" in lines[1]
    integrated = integrate_risk_score(deepfake_score=0.20, forgery_scores=[0.88])
    summary = build_integrated_risk_reason(integrated)
    assert "Integrated risk" in summary
    assert "riskScore=" in summary


def test_strip_soft_trufor_reasons_when_forgery_lane_succeeds() -> None:
    from gpu_worker.pipeline.forgery_merge import _strip_soft_trufor_analysis_reasons

    reasons = [
        "CNN (Xception) fake_score=0.806 (fake)",
        "Late fusion (fusion-v4c-field-tuned) score=0.537 → REAL @ T=0.58",
        "위변조(TruFor) 실행 중 오류가 발생해 생략되었습니다. (TruFor produced no finite frame scores)",
        "forgery_spatial: TruFor score=0.0000 detected=False",
        "Forgery spatial (TruFor) fake_score=0.645 (fake) @ T=0.515",
    ]
    kept = _strip_soft_trufor_analysis_reasons(reasons)
    assert kept == [
        "CNN (Xception) fake_score=0.806 (fake)",
        "Late fusion (fusion-v4c-field-tuned) score=0.537 → REAL @ T=0.58",
        "Forgery spatial (TruFor) fake_score=0.645 (fake) @ T=0.515",
    ]
