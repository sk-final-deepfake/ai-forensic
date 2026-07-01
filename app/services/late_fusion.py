from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FusionConfig:
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
        )


def load_fusion_config(path: Path) -> FusionConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FusionConfig.from_dict(payload)


def fuse_scores(
    *,
    s_cnn: float,
    s_temporal: float,
    s_optical: float,
    config: FusionConfig,
) -> float:
    weights = config.weights
    return round(
        weights["cnn"] * s_cnn
        + weights["temporal"] * s_temporal
        + weights["optical"] * s_optical,
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
    reasons.append(
        f"Late fusion ({config.fusion_version}) score={fusion_score:.3f} "
        f"→ {'FAKE' if fusion_detected else 'REAL'} @ T={config.threshold:.2f}"
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
                "reason": "High CNN frame-level fake probability cluster",
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
