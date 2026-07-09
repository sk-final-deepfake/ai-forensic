from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class GatingConfig:
    """Rule-based fusion gates (no retrain): when to trust / rescue each module."""

    cnn_ambiguous_lo: float = 0.40
    cnn_ambiguous_hi: float = 0.78
    ts_rescue_min: float = 0.60
    ts_rescue_cnn_weight: float = 0.35
    ts_rescue_temporal_weight: float = 0.65
    ts_rescue_strong_min: float = 0.75
    ts_rescue_strong_cnn_weight: float = 0.10
    ts_rescue_strong_temporal_weight: float = 0.90
    ts_base_weight: float = 0.0
    ts_base_min: float = 0.45
    ts_base_requires_ambiguous_cnn: bool = True
    gmflow_veto_max: float = 0.35
    cnn_discount_when_gmf_low: float = 0.18
    ambiguous_cnn_floor: bool = True
    ambiguous_gmf_max: float = 0.30
    ambiguous_cnn_floor_min: float = 0.58
    ts_rescue_margin: float = 0.15
    gmflow_soft_veto_cnn_min: float = 0.62
    gmflow_soft_veto_max: float = 0.15
    cnn_soft_discount: float = 0.10
    ambiguous_boost: float = 0.04
    ambiguous_boost_cnn_max: float = 0.68
    dual_high_cnn_min: float = 0.85
    dual_high_ts_min: float = 0.85
    dual_high_gmf_max: float = 0.0
    dual_high_fusion_cap: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> GatingConfig:
        if not payload:
            return cls()
        return cls(
            cnn_ambiguous_lo=float(payload.get("cnn_ambiguous_lo", 0.40)),
            cnn_ambiguous_hi=float(payload.get("cnn_ambiguous_hi", 0.78)),
            ts_rescue_min=float(payload.get("ts_rescue_min", 0.60)),
            ts_rescue_cnn_weight=float(payload.get("ts_rescue_cnn_weight", 0.35)),
            ts_rescue_temporal_weight=float(payload.get("ts_rescue_temporal_weight", 0.65)),
            ts_rescue_strong_min=float(payload.get("ts_rescue_strong_min", 0.75)),
            ts_rescue_strong_cnn_weight=float(payload.get("ts_rescue_strong_cnn_weight", 0.10)),
            ts_rescue_strong_temporal_weight=float(payload.get("ts_rescue_strong_temporal_weight", 0.90)),
            ts_base_weight=float(payload.get("ts_base_weight", 0.0)),
            ts_base_min=float(payload.get("ts_base_min", 0.45)),
            ts_base_requires_ambiguous_cnn=bool(payload.get("ts_base_requires_ambiguous_cnn", True)),
            gmflow_veto_max=float(payload.get("gmflow_veto_max", 0.35)),
            cnn_discount_when_gmf_low=float(payload.get("cnn_discount_when_gmf_low", 0.18)),
            ambiguous_cnn_floor=bool(payload.get("ambiguous_cnn_floor", False)),
            ambiguous_gmf_max=float(payload.get("ambiguous_gmf_max", 0.30)),
            ambiguous_cnn_floor_min=float(payload.get("ambiguous_cnn_floor_min", 0.58)),
            ts_rescue_margin=float(payload.get("ts_rescue_margin", 0.15)),
            gmflow_soft_veto_cnn_min=float(payload.get("gmflow_soft_veto_cnn_min", 0.62)),
            gmflow_soft_veto_max=float(payload.get("gmflow_soft_veto_max", 0.15)),
            cnn_soft_discount=float(payload.get("cnn_soft_discount", 0.10)),
            ambiguous_boost=float(payload.get("ambiguous_boost", 0.04)),
            ambiguous_boost_cnn_max=float(payload.get("ambiguous_boost_cnn_max", 0.68)),
            dual_high_cnn_min=float(payload.get("dual_high_cnn_min", 0.85)),
            dual_high_ts_min=float(payload.get("dual_high_ts_min", 0.85)),
            dual_high_gmf_max=float(payload.get("dual_high_gmf_max", 0.0)),
            dual_high_fusion_cap=float(payload.get("dual_high_fusion_cap", 0.0)),
        )


@dataclass(frozen=True)
class FusionConfig:
    fusion_version: str
    method: str
    weights: dict[str, float]
    threshold: float
    module_thresholds: dict[str, float]
    risk_levels: dict[str, float]
    suspicious_segment: dict[str, float]
    model_versions: dict[str, str]
    gating: GatingConfig | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FusionConfig:
        gating_raw = payload.get("gating")
        return cls(
            fusion_version=str(payload.get("fusion_version", "fusion-v0")),
            method=str(payload.get("method", "weighted")),
            weights={
                "cnn": float(payload["weights"]["cnn"]),
                "temporal": float(payload["weights"]["temporal"]),
                "optical": float(payload["weights"]["optical"]),
            },
            threshold=float(payload.get("threshold", 0.57)),
            module_thresholds={
                "cnn": float(payload.get("module_thresholds", {}).get("cnn", 0.78)),
                "temporal": float(payload.get("module_thresholds", {}).get("temporal", 0.5)),
                "optical": float(payload.get("module_thresholds", {}).get("optical", 0.5)),
            },
            risk_levels={
                "medium_min": float(payload.get("risk_levels", {}).get("medium_min", 40.0)),
                "high_min": float(payload.get("risk_levels", {}).get("high_min", 70.0)),
            },
            suspicious_segment={
                "high_risk_frame_threshold": float(
                    payload.get("suspicious_segment", {}).get("high_risk_frame_threshold", 0.65)
                ),
                "min_segment_sec": float(
                    payload.get("suspicious_segment", {}).get("min_segment_sec", 0.5)
                ),
            },
            model_versions=dict(payload.get("model_versions", {})),
            gating=GatingConfig.from_dict(gating_raw) if gating_raw else None,
        )


def load_fusion_config(path: Path) -> FusionConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FusionConfig.from_dict(payload)


def fuse_scores_weighted(
    *,
    s_cnn: float,
    s_temporal: float,
    s_optical: float,
    config: FusionConfig,
    s_cnn_override: float | None = None,
) -> float:
    weights = config.weights
    cnn_val = float(s_cnn_override if s_cnn_override is not None else s_cnn)
    return (
        weights["cnn"] * cnn_val
        + weights["temporal"] * s_temporal
        + weights["optical"] * s_optical
    )


def fuse_scores_gated(
    *,
    s_cnn: float,
    s_temporal: float,
    s_optical: float,
    config: FusionConfig,
) -> tuple[float, dict[str, Any]]:
    """Adaptive fusion: TimeSformer rescue + GMFlow veto on overconfident CNN."""
    g = config.gating or GatingConfig()
    meta: dict[str, Any] = {
        "ts_rescue_active": False,
        "ts_rescue_tier": None,
        "ts_base_blend_active": False,
        "dual_high_cap_active": False,
        "gmflow_veto_active": False,
        "gmflow_soft_veto_active": False,
        "ambiguous_boost_active": False,
    }

    s_cnn_eff = float(s_cnn)
    cnn_th = config.module_thresholds["cnn"]
    if s_cnn_eff >= cnn_th and s_optical < g.gmflow_veto_max:
        s_cnn_eff = s_cnn_eff * (1.0 - g.cnn_discount_when_gmf_low)
        meta["gmflow_veto_active"] = True
    elif (
        g.gmflow_soft_veto_cnn_min <= s_cnn_eff < cnn_th
        and s_optical < g.gmflow_soft_veto_max
        and s_temporal < 0.12
    ):
        s_cnn_eff = s_cnn_eff * (1.0 - g.cnn_soft_discount)
        meta["gmflow_soft_veto_active"] = True

    weights = dict(config.weights)
    ambiguous = g.cnn_ambiguous_lo <= s_cnn < g.cnn_ambiguous_hi
    if (
        g.ts_base_weight > 0.0
        and s_temporal >= g.ts_base_min
        and (ambiguous or not g.ts_base_requires_ambiguous_cnn)
    ):
        weights["cnn"] = max(0.0, weights["cnn"] - g.ts_base_weight)
        weights["temporal"] = weights.get("temporal", 0.0) + g.ts_base_weight
        meta["ts_base_blend_active"] = True

    base = (
        weights["cnn"] * s_cnn_eff
        + weights["temporal"] * s_temporal
        + weights["optical"] * s_optical
    )
    fusion = base

    ts_beats_cnn = s_temporal >= s_cnn + g.ts_rescue_margin
    if ambiguous and ts_beats_cnn:
        rescue_candidates: list[tuple[str, float, float, float]] = []
        if s_temporal >= g.ts_rescue_strong_min:
            rescue_candidates.append(
                (
                    "strong",
                    g.ts_rescue_strong_min,
                    g.ts_rescue_strong_cnn_weight,
                    g.ts_rescue_strong_temporal_weight,
                )
            )
        if s_temporal >= g.ts_rescue_min:
            rescue_candidates.append(
                (
                    "medium",
                    g.ts_rescue_min,
                    g.ts_rescue_cnn_weight,
                    g.ts_rescue_temporal_weight,
                )
            )
        if rescue_candidates:
            tier, _, cnn_w, ts_w = max(rescue_candidates, key=lambda row: row[1])
            rescue = cnn_w * s_cnn + ts_w * s_temporal
            if rescue > fusion:
                fusion = rescue
                meta["ts_rescue_active"] = True
                meta["ts_rescue_tier"] = tier
    elif (
        g.ambiguous_boost > 0
        and g.ambiguous_cnn_floor_min <= s_cnn < g.ambiguous_boost_cnn_max
        and s_optical < g.ambiguous_gmf_max
        and s_temporal < min(g.ts_rescue_min, 0.15)
    ):
        fusion = max(fusion, base + g.ambiguous_boost)
        meta["ambiguous_boost_active"] = True
    elif (
        g.ambiguous_cnn_floor
        and g.ambiguous_cnn_floor_min <= s_cnn < g.cnn_ambiguous_hi
        and s_optical < g.ambiguous_gmf_max
        and s_temporal < g.ts_rescue_min
    ):
        fusion = max(fusion, s_cnn)
        meta["ambiguous_boost_active"] = True

    if (
        g.dual_high_fusion_cap > 0.0
        and s_cnn >= g.dual_high_cnn_min
        and s_temporal >= g.dual_high_ts_min
        and s_optical < g.dual_high_gmf_max
        and fusion > g.dual_high_fusion_cap
    ):
        fusion = g.dual_high_fusion_cap
        meta["dual_high_cap_active"] = True

    meta["fusion_base"] = round(base, 6)
    return round(fusion, 6), meta


def fuse_scores(
    *,
    s_cnn: float,
    s_temporal: float,
    s_optical: float,
    config: FusionConfig,
) -> float:
    if config.method == "gated":
        score, _ = fuse_scores_gated(
            s_cnn=s_cnn,
            s_temporal=s_temporal,
            s_optical=s_optical,
            config=config,
        )
        return score
    return round(
        fuse_scores_weighted(
            s_cnn=s_cnn,
            s_temporal=s_temporal,
            s_optical=s_optical,
            config=config,
        ),
        6,
    )


def score_detected(score: float | None, threshold: float) -> bool:
    if score is None:
        return False
    return float(score) >= threshold


def risk_level_from_score(fusion_score: float, config: FusionConfig) -> str:
    risk_score = fusion_score * 100.0
    if risk_score >= config.risk_levels["high_min"]:
        return "HIGH"
    if risk_score >= config.risk_levels["medium_min"]:
        return "MEDIUM"
    return "LOW"


def confidence_from_module_scores(scores: list[float | None]) -> float:
    valid = [float(s) for s in scores if s is not None]
    if not valid:
        return 0.0
    if len(valid) == 1:
        return round(abs(valid[0] - 0.5) * 2.0, 4)
    spread = max(valid) - min(valid)
    agreement = max(0.0, 1.0 - spread)
    strength = sum(abs(v - 0.5) for v in valid) / len(valid) / 0.5
    return round(min(1.0, agreement * 0.6 + strength * 0.4), 4)


def build_analysis_reasons(
    *,
    s_cnn: float | None,
    s_temporal: float | None,
    s_optical: float | None,
    fusion_score: float,
    fusion_detected: bool,
    config: FusionConfig,
) -> list[str]:
    reasons: list[str] = []
    if s_cnn is not None:
        reasons.append(
            f"CNN (Xception) fake_score={s_cnn:.3f} "
            f"({'fake' if score_detected(s_cnn, config.module_thresholds['cnn']) else 'real'})"
        )
    if s_temporal is not None:
        reasons.append(
            f"Temporal (TimeSformer) fake_score={s_temporal:.3f} "
            f"({'fake' if score_detected(s_temporal, config.module_thresholds['temporal']) else 'real'})"
        )
    if s_optical is not None:
        reasons.append(
            f"Optical (GMFlow) motion_score={s_optical:.3f} "
            f"({'anomaly' if score_detected(s_optical, config.module_thresholds['optical']) else 'normal'})"
        )
    gate_note = ""
    if config.method == "gated" and config.gating is not None:
        _, gate_meta = fuse_scores_gated(
            s_cnn=float(s_cnn or 0.0),
            s_temporal=float(s_temporal or 0.0),
            s_optical=float(s_optical or 0.0),
            config=config,
        )
        flags = []
        if gate_meta.get("ts_rescue_active"):
            tier = gate_meta.get("ts_rescue_tier")
            flags.append(f"TimeSformer rescue ({tier})" if tier else "TimeSformer rescue")
        if gate_meta.get("dual_high_cap_active"):
            flags.append("dual-high CNN+TS cap")
        if gate_meta.get("gmflow_veto_active"):
            flags.append("GMFlow veto on CNN")
        if gate_meta.get("ambiguous_boost_active"):
            flags.append("ambiguous CNN boost")
        if flags:
            gate_note = f" [{' + '.join(flags)}]"
    reasons.append(
        f"Late fusion ({config.fusion_version}) score={fusion_score:.3f} "
        f"→ {'FAKE' if fusion_detected else 'REAL'} @ T={config.threshold:.2f}{gate_note}"
    )
    return reasons


def frame_index_to_timestamp(video_path: Path, frame_index: int) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return float(frame_index)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    if fps <= 0:
        fps = 25.0
    return round(frame_index / fps, 3)


def build_frame_risks(
    video_path: Path,
    per_frame_scores: list[dict[str, Any]],
) -> list[dict[str, float | int]]:
    risks: list[dict[str, float | int]] = []
    for row in per_frame_scores:
        frame_index = row.get("frame_index")
        score = row.get("fake_score", row.get("prob_fake"))
        if frame_index is None or score is None:
            continue
        risks.append(
            {
                "frameIndex": int(frame_index),
                "timestampSec": frame_index_to_timestamp(video_path, int(frame_index)),
                "riskScore": round(float(score), 6),
            }
        )
    return risks


def build_suspicious_segments(
    frame_risks: list[dict[str, float | int]],
    *,
    high_risk_threshold: float,
    min_segment_sec: float,
    reason: str = "High CNN frame-level fake probability cluster",
) -> list[dict[str, float | str]]:
    if not frame_risks:
        return []

    ordered = sorted(frame_risks, key=lambda row: float(row["timestampSec"]))
    segments: list[dict[str, float | str]] = []
    current: list[dict[str, float | int]] = []

    def flush() -> None:
        if not current:
            return
        start = float(current[0]["timestampSec"])
        end = float(current[-1]["timestampSec"])
        if end - start < min_segment_sec and len(current) == 1:
            end = start + min_segment_sec
        max_score = max(float(row["riskScore"]) for row in current)
        segments.append(
            {
                "startTime": round(start, 3),
                "endTime": round(end, 3),
                "maxRiskScore": round(max_score, 6),
                "reason": reason,
            }
        )

    for row in ordered:
        if float(row["riskScore"]) >= high_risk_threshold:
            current.append(row)
        else:
            flush()
            current = []
    flush()
    return segments


def _clip_rows(per_clip_scores: list[dict[str, Any]], per_clip: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if per_clip:
        return per_clip
    rows: list[dict[str, Any]] = []
    for row in per_clip_scores:
        indices = row.get("frame_indices") or []
        start = row.get("clip_start_frame")
        end = row.get("clip_end_frame")
        if start is None and indices:
            start = indices[0]
        if end is None and indices:
            end = indices[-1]
        score = row.get("fake_score", row.get("prob_fake"))
        if start is None or end is None or score is None:
            continue
        rows.append(
            {
                "clip_index": row.get("clip_index", len(rows)),
                "clip_start_frame": int(start),
                "clip_end_frame": int(end),
                "prob_fake": float(score),
            }
        )
    return rows


def build_clip_risks(
    video_path: Path,
    *,
    per_clip_scores: list[dict[str, Any]] | None = None,
    per_clip: list[dict[str, Any]] | None = None,
) -> list[dict[str, float | int]]:
    risks: list[dict[str, float | int]] = []
    for row in _clip_rows(per_clip_scores or [], per_clip or []):
        start = int(row["clip_start_frame"])
        end = int(row["clip_end_frame"])
        score = row.get("prob_fake", row.get("fake_score"))
        if score is None:
            continue
        risks.append(
            {
                "clipIndex": int(row.get("clip_index", len(risks))),
                "startFrameIndex": start,
                "endFrameIndex": end,
                "startTimeSec": frame_index_to_timestamp(video_path, start),
                "endTimeSec": frame_index_to_timestamp(video_path, end),
                "riskScore": round(float(score), 6),
            }
        )
    return risks


def build_clip_segment_risks(clip_risks: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    """Convert clip windows to point samples for suspicious-segment clustering."""
    points: list[dict[str, float | int]] = []
    for row in clip_risks:
        midpoint = (float(row["startTimeSec"]) + float(row["endTimeSec"])) / 2.0
        points.append(
            {
                "timestampSec": round(midpoint, 3),
                "riskScore": row["riskScore"],
            }
        )
    return points


def _pair_motion_score(flow_mag_mean: float, *, median: float, span: float) -> float:
    if span <= 0:
        return 0.0
    relative = (flow_mag_mean - median) / span
    return round(min(1.0, max(0.0, 0.5 + relative / 2.0)), 6)


def build_pair_risks(
    video_path: Path,
    pair_stats: list[dict[str, Any]],
    *,
    per_frame_pair: list[dict[str, Any]] | None = None,
) -> list[dict[str, float | int | None]]:
    if per_frame_pair:
        mags = [float(row.get("flow_mag_mean", 0.0)) for row in per_frame_pair]
    else:
        mags = [float(row.get("magnitude_mean", 0.0)) for row in pair_stats]
    if not pair_stats and not per_frame_pair:
        return []

    median = float(np.median(mags)) if mags else 0.0
    span = float(max(mags) - min(mags)) if mags else 0.0

    risks: list[dict[str, float | int | None]] = []
    if per_frame_pair and pair_stats:
        iterable = zip(per_frame_pair, pair_stats)
    elif per_frame_pair:
        iterable = ((row, {}) for row in per_frame_pair)
    else:
        iterable = (({}, row) for row in pair_stats)

    for idx, (pair_row, stat_row) in enumerate(iterable):
        frame_a = stat_row.get("frame_index_a", pair_row.get("frame_index_a"))
        frame_b = stat_row.get("frame_index_b", pair_row.get("frame_index_b"))
        mag = pair_row.get("flow_mag_mean", stat_row.get("magnitude_mean"))
        if frame_a is None or frame_b is None or mag is None:
            continue
        frame_a = int(frame_a)
        frame_b = int(frame_b)
        mag = float(mag)
        midpoint = (frame_a + frame_b) / 2.0
        risks.append(
            {
                "pairIndex": idx,
                "frameIndexA": frame_a,
                "frameIndexB": frame_b,
                "timestampSec": frame_index_to_timestamp(video_path, int(midpoint)),
                "riskScore": _pair_motion_score(mag, median=median, span=span),
                "motionMagnitude": round(mag, 6),
            }
        )
    return risks


def build_module_timelines(
    video_path: Path,
    modules: list[Any],
    *,
    config: FusionConfig,
) -> list[dict[str, Any]]:
    """Build unified module timeline payloads for API response."""
    by_module = {item.module: item for item in modules}
    timelines: list[dict[str, Any]] = []

    cnn = by_module.get("cnn")
    if cnn:
        per_frame = (cnn.details or {}).get("per_frame_scores") or []
        frame_risks = build_frame_risks(video_path, per_frame)
        threshold = config.module_thresholds["cnn"]
        score = float(cnn.fake_score or 0.0)
        timelines.append(
            {
                "module": "cnn",
                "modelName": cnn.model_name,
                "modelVersion": cnn.model_version,
                "videoScore": round(score, 6),
                "threshold": threshold,
                "detected": score_detected(cnn.fake_score, threshold),
                "frameRisks": frame_risks,
                "clipRisks": [],
                "pairRisks": [],
                "suspiciousSegments": build_suspicious_segments(
                    frame_risks,
                    high_risk_threshold=config.suspicious_segment["high_risk_frame_threshold"],
                    min_segment_sec=config.suspicious_segment["min_segment_sec"],
                    reason="High CNN frame-level fake probability cluster",
                ),
            }
        )

    temporal = by_module.get("temporal")
    if temporal:
        details = temporal.details or {}
        breakdown = details.get("score_breakdown") or {}
        clip_risks = build_clip_risks(
            video_path,
            per_clip_scores=details.get("per_clip_scores") or breakdown.get("per_clip_scores") or [],
            per_clip=breakdown.get("per_clip") or [],
        )
        threshold = config.module_thresholds["temporal"]
        score = float(temporal.fake_score or 0.0)
        clip_points = build_clip_segment_risks(clip_risks)
        timelines.append(
            {
                "module": "temporal",
                "modelName": temporal.model_name,
                "modelVersion": temporal.model_version,
                "videoScore": round(score, 6),
                "threshold": threshold,
                "detected": score_detected(temporal.fake_score, threshold),
                "frameRisks": [],
                "clipRisks": clip_risks,
                "pairRisks": [],
                "suspiciousSegments": build_suspicious_segments(
                    clip_points,
                    high_risk_threshold=threshold,
                    min_segment_sec=config.suspicious_segment["min_segment_sec"],
                    reason="High TimeSformer clip-level fake probability cluster",
                ),
            }
        )

    optical = by_module.get("optical")
    if optical:
        details = optical.details or {}
        pair_stats = details.get("pair_stats") or []
        per_frame_pair = details.get("per_frame_pair") or []
        pair_risks = build_pair_risks(video_path, pair_stats, per_frame_pair=per_frame_pair or None)
        threshold = config.module_thresholds["optical"]
        score = float(optical.fake_score or 0.0)
        pair_points = [
            {"timestampSec": row["timestampSec"], "riskScore": row["riskScore"]}
            for row in pair_risks
        ]
        timelines.append(
            {
                "module": "optical",
                "modelName": optical.model_name,
                "modelVersion": optical.model_version,
                "videoScore": round(score, 6),
                "threshold": threshold,
                "detected": score_detected(optical.fake_score, threshold),
                "frameRisks": [],
                "clipRisks": [],
                "pairRisks": pair_risks,
                "suspiciousSegments": build_suspicious_segments(
                    pair_points,
                    high_risk_threshold=threshold,
                    min_segment_sec=config.suspicious_segment["min_segment_sec"],
                    reason="High GMFlow optical-flow motion anomaly cluster",
                ),
            }
        )

    return timelines


def optical_score_from_aggregate(
    aggregate: dict[str, Any],
    cohort: dict[str, float],
) -> float | None:
    flow_mean = aggregate.get("magnitude_mean_mean")
    if flow_mean is None:
        return None
    median = cohort.get("flow_mag_mean_median")
    std = cohort.get("flow_mag_mean_std")
    if median is None or std is None or std <= 0:
        return None
    z = max(0.0, (float(flow_mean) - float(median)) / float(std))
    return round(min(1.0, z / 3.0), 6)
