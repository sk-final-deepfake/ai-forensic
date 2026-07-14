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


def test_gmflow_veto_skips_when_ts_agrees() -> None:
    """CNN high + GMF low should not discount when TimeSformer is also high."""
    config = _gated_config(gmflow_veto_max=0.30, gmflow_veto_max_ts=0.50)
    score, meta = fuse_scores_gated(
        s_cnn=0.90,
        s_temporal=0.86,
        s_optical=0.10,
        config=config,
    )
    assert meta["gmflow_veto_active"] is False
    assert score >= 0.80


def test_gmflow_veto_still_applies_when_ts_weak() -> None:
    config = _gated_config(gmflow_veto_max=0.30, gmflow_veto_max_ts=0.50)
    score, meta = fuse_scores_gated(
        s_cnn=0.90,
        s_temporal=0.20,
        s_optical=0.10,
        config=config,
    )
    assert meta["gmflow_veto_active"] is True
    assert score < 0.90 * 0.9 + 0.05


def test_dual_high_cap_disabled_does_not_crush_agreement() -> None:
    config = _gated_config(
        dual_high_fusion_cap=0.0,
        dual_high_agree_boost=0.04,
        dual_high_agree_cnn_min=0.78,
        dual_high_agree_ts_min=0.60,
    )
    score, meta = fuse_scores_gated(
        s_cnn=0.90,
        s_temporal=0.88,
        s_optical=0.20,
        config=config,
    )
    assert meta["dual_high_cap_active"] is False
    assert meta["dual_high_agree_active"] is True
    assert score >= 0.80


def test_fuse_scores_dispatches_gated() -> None:
    config = _gated_config()
    assert fuse_scores(s_cnn=0.53, s_temporal=0.91, s_optical=0.39, config=config) > 0.5


def test_dual_high_mid_gmf_cap_spares_low_gmf_fake() -> None:
    """Mid-GMF dual-high cap catches BBC-like real FP; low-GMF dual-high fake stays high."""
    config = _gated_config(
        dual_high_fusion_cap=0.55,
        dual_high_cnn_min=0.95,
        dual_high_ts_min=0.95,
        dual_high_gmf_min=0.35,
        dual_high_gmf_max=0.50,
    )
    real_fp, meta_real = fuse_scores_gated(
        s_cnn=1.0,
        s_temporal=1.0,
        s_optical=0.45,
        config=config,
    )
    assert meta_real["dual_high_cap_active"] is True
    assert real_fp == pytest.approx(0.55)

    fake_tp, meta_fake = fuse_scores_gated(
        s_cnn=1.0,
        s_temporal=1.0,
        s_optical=0.14,
        config=config,
    )
    assert meta_fake["dual_high_cap_active"] is False
    assert fake_tp > 0.80


def test_soft_mid_ts_discount() -> None:
    config = _gated_config(
        ambiguous_cnn_floor=False,
        ambiguous_boost=0.0,
        gmflow_soft_veto_max_ts=0.25,
        gmflow_soft_veto_mid_ts_min=0.10,
        cnn_soft_discount=0.05,
        cnn_soft_mid_discount=0.20,
        gmflow_soft_veto_max=0.15,
    )
    # Mild TS → stronger soft discount
    score_mid, meta_mid = fuse_scores_gated(
        s_cnn=0.70,
        s_temporal=0.17,
        s_optical=0.0,
        config=config,
    )
    assert meta_mid["gmflow_soft_veto_active"] is True
    assert score_mid == pytest.approx(0.9 * 0.70 * 0.80, abs=1e-4)

    # Near-zero TS keeps mild soft discount
    score_low, meta_low = fuse_scores_gated(
        s_cnn=0.70,
        s_temporal=0.0,
        s_optical=0.0,
        config=config,
    )
    assert meta_low["gmflow_soft_veto_active"] is True
    assert score_low == pytest.approx(0.9 * 0.70 * 0.95, abs=1e-4)
    assert score_low > score_mid


def test_ops_v4c_field_tuned_config_loads() -> None:
    from pathlib import Path

    from app.services.late_fusion import load_fusion_config

    cfg = load_fusion_config(Path(__file__).resolve().parents[1] / "config" / "fusion_v4_ts_gated.json")
    assert cfg.fusion_version == "fusion-v4c-field-tuned"
    assert cfg.threshold == pytest.approx(0.578)
    assert cfg.gating is not None
    assert cfg.gating.cnn_discount_when_gmf_low == pytest.approx(0.30)
    assert cfg.gating.gmflow_veto_max == pytest.approx(0.38)
    assert cfg.gating.dual_high_agree_boost == pytest.approx(0.0)
    assert cfg.gating.dual_high_fusion_cap == pytest.approx(0.55)

    # Real-like CNN-high / TS-weak / mid-GMF should be pulled down by stronger veto.
    score, meta = fuse_scores_gated(
        s_cnn=0.87,
        s_temporal=0.001,
        s_optical=0.30,
        config=cfg,
    )
    assert meta["gmflow_veto_active"] is True
    assert score < 0.70
