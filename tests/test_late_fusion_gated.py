from __future__ import annotations

import pytest

from app.services.late_fusion import FusionConfig, fuse_scores, fuse_scores_gated


def _gated_config(**gating_overrides) -> FusionConfig:
    payload = {
        "fusion_version": "fusion-v3-gated",
        "method": "gated",
        "weights": {"cnn": 0.9, "temporal": 0.0, "optical": 0.1},
        "threshold": 0.60,
        "module_thresholds": {"cnn": 0.78, "temporal": 0.5, "optical": 0.417},
        "risk_levels": {"medium_min": 40.0, "high_min": 70.0},
        "suspicious_segment": {"high_risk_frame_threshold": 0.65, "min_segment_sec": 0.5},
        "model_versions": {},
        "gating": {
            "cnn_ambiguous_lo": 0.40,
            "cnn_ambiguous_hi": 0.78,
            "ts_rescue_min": 0.60,
            "ts_rescue_cnn_weight": 0.35,
            "ts_rescue_temporal_weight": 0.65,
            "gmflow_veto_max": 0.35,
            "cnn_discount_when_gmf_low": 0.18,
            "ambiguous_cnn_floor": True,
            "ambiguous_gmf_max": 0.30,
            **gating_overrides,
        },
    }
    return FusionConfig.from_dict(payload)


def test_ts_rescue_boosts_ambiguous_fake() -> None:
    config = _gated_config(ts_rescue_margin=0.15)
    score, meta = fuse_scores_gated(
        s_cnn=0.53,
        s_temporal=0.91,
        s_optical=0.39,
        config=config,
    )
    assert meta["ts_rescue_active"] is True
    assert score >= 0.35 * 0.53 + 0.65 * 0.91 - 0.01


def test_ts_rescue_skips_when_temporal_only_slightly_above_cnn() -> None:
    config = _gated_config(ts_rescue_min=0.50, ts_rescue_margin=0.15)
    _, meta = fuse_scores_gated(
        s_cnn=0.665,
        s_temporal=0.532,
        s_optical=0.0,
        config=config,
    )
    assert meta["ts_rescue_active"] is False


def test_gmflow_soft_veto_for_borderline_real() -> None:
    config = _gated_config()
    score, meta = fuse_scores_gated(
        s_cnn=0.665,
        s_temporal=0.0,
        s_optical=0.05,
        config=config,
    )
    assert meta["gmflow_soft_veto_active"] is True
    assert score < 0.665


def test_ambiguous_boost_for_borderline_fake() -> None:
    config = _gated_config(ambiguous_cnn_floor=False, ambiguous_boost=0.04)
    score, meta = fuse_scores_gated(
        s_cnn=0.62,
        s_temporal=0.0,
        s_optical=0.20,
        config=config,
    )
    assert meta["ambiguous_boost_active"] is True
    assert score >= 0.615


def test_dual_module_rescue_when_cnn_misses() -> None:
    config = _gated_config(
        dual_module_rescue=True,
        dual_module_ts_min=0.60,
        dual_module_gmf_min=0.50,
        dual_module_ts_weight=0.70,
        dual_module_gmf_weight=0.30,
    )
    score, meta = fuse_scores_gated(
        s_cnn=0.20,
        s_temporal=0.85,
        s_optical=0.70,
        config=config,
    )
    assert meta["dual_module_rescue_active"] is True
    assert score >= 0.70 * 0.85 + 0.30 * 0.70 - 0.01


def test_dual_module_rescue_keeps_cnn_only_high() -> None:
    config = _gated_config(dual_module_rescue=True)
    score, meta = fuse_scores_gated(
        s_cnn=0.92,
        s_temporal=0.10,
        s_optical=0.55,
        config=config,
    )
    assert meta["dual_module_rescue_active"] is False
    assert score >= 0.80


def test_fuse_scores_dispatches_gated() -> None:
    config = _gated_config()
    assert fuse_scores(s_cnn=0.53, s_temporal=0.91, s_optical=0.39, config=config) > 0.5
