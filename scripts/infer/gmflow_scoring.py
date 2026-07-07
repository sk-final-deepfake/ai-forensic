"""GMFlow motion heuristic scoring (no torch/backend deps)."""
from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_ANOMALY_THRESHOLD = 0.5
DEFAULT_SIGNAL_WEIGHTS: dict[str, float] = {
    "temporal_jitter": 0.30,
    "spatial_inconsistency_mean": 0.25,
    "angle_dispersion_mean": 0.25,
    "flow_mag_mean": 0.20,
}


def profile_from_filename(filename: str) -> str | None:
    if filename.startswith("fake_ffpp_") or filename.endswith("_long.mp4"):
        return "ffpp_vox"
    if filename.startswith("celebdf_"):
        return "celebdf"
    return None


def _report_matches_profile(report: dict[str, Any], profile: str) -> bool:
    tagged = profile_from_filename(str(report.get("file") or ""))
    return tagged == profile if tagged else False


def cohort_baselines_from_reports(
    reports: list[dict[str, Any]],
    *,
    profile: str | None = None,
) -> dict[str, float]:
    if profile:
        reports = [r for r in reports if _report_matches_profile(r, profile)]

    reals = [
        r["score_breakdown"]["aggregate"]
        for r in reports
        if r.get("status") == "ok"
        and r.get("ground_truth_label") == "real"
        and "score_breakdown" in r
    ]
    required = (
        "temporal_jitter",
        "spatial_inconsistency_mean",
        "angle_dispersion_mean",
        "flow_mag_mean",
    )
    complete = [row for row in reals if all(k in row and row[k] is not None for k in required)]
    if not complete:

        def med(key: str) -> float:
            return 0.0

        return {}

    def med(key: str) -> float:
        return float(np.median([row[key] for row in complete]))

    return {
        "temporal_jitter_median": med("temporal_jitter"),
        "spatial_inconsistency_median": med("spatial_inconsistency_mean"),
        "angle_dispersion_median": med("angle_dispersion_mean"),
        "flow_mag_mean_median": med("flow_mag_mean"),
        "temporal_jitter_std": float(np.std([row["temporal_jitter"] for row in complete])) or 1e-6,
        "spatial_inconsistency_std": float(np.std([row["spatial_inconsistency_mean"] for row in complete])) or 1e-6,
        "angle_dispersion_std": float(np.std([row["angle_dispersion_mean"] for row in complete])) or 1e-6,
        "flow_mag_mean_std": float(np.std([row["flow_mag_mean"] for row in complete])) or 1e-6,
    }


def motion_anomaly_score(
    aggregate: dict[str, float],
    cohort: dict[str, float],
    *,
    signal_weights: dict[str, float] | None = None,
) -> float:
    if not cohort:
        return 0.0
    weights = signal_weights or DEFAULT_SIGNAL_WEIGHTS
    mapping = [
        ("temporal_jitter", "temporal_jitter_median", "temporal_jitter_std"),
        ("spatial_inconsistency_mean", "spatial_inconsistency_median", "spatial_inconsistency_std"),
        ("angle_dispersion_mean", "angle_dispersion_median", "angle_dispersion_std"),
        ("flow_mag_mean", "flow_mag_mean_median", "flow_mag_mean_std"),
    ]
    weighted_sum = 0.0
    for value_key, med_key, std_key in mapping:
        w = float(weights.get(value_key, 0.0))
        if w <= 0.0:
            continue
        z = (aggregate[value_key] - cohort[med_key]) / cohort[std_key]
        weighted_sum += w * max(0.0, z)
    if weighted_sum <= 0.0:
        return 0.0
    return float(min(1.0, weighted_sum / 3.0))


def enrich_motion_scores(
    reports: list[dict[str, Any]],
    *,
    threshold: float = DEFAULT_ANOMALY_THRESHOLD,
    per_profile_cohort: bool = False,
    signal_weights: dict[str, float] | None = None,
    cohort_source: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    from gmflow_feature_extract import normalize_report

    source = cohort_source if cohort_source is not None else reports
    cohort_rows = [dict(r) for r in source]
    for row in cohort_rows:
        normalize_report(row)
    combined_cohort = cohort_baselines_from_reports(cohort_rows)
    profile_cohorts = {
        prof: cohort_baselines_from_reports(cohort_rows, profile=prof) for prof in ("ffpp_vox", "celebdf")
    }

    def cohort_for(report: dict[str, Any]) -> dict[str, float]:
        if not per_profile_cohort:
            return combined_cohort
        prof = profile_from_filename(str(report.get("file") or ""))
        if prof and profile_cohorts.get(prof):
            return profile_cohorts[prof]
        return combined_cohort

    for report in reports:
        if report.get("status") != "ok":
            report["motion_anomaly_score"] = None
            report["pred_label"] = None
            continue
        sb = report.get("score_breakdown") or {}
        aggregate = sb.get("aggregate")
        if not aggregate:
            report["motion_anomaly_score"] = 0.0
            report["pred_label"] = "real"
            continue
        cohort = cohort_for(report)
        score = motion_anomaly_score(aggregate, cohort, signal_weights=signal_weights)
        report["motion_anomaly_score"] = round(score, 6)
        report["pred_label"] = "fake" if score >= threshold else "real"
        sb["threshold"] = threshold
        sb["motion_anomaly_score"] = report["motion_anomaly_score"]
        sb["pred_label"] = report["pred_label"]
        report["score_breakdown"] = sb

    return combined_cohort
