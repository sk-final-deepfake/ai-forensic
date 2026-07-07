"""Feature extraction from GMFlow benchmark JSON (score_breakdown)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

AGGREGATE_KEYS = (
    "flow_mag_mean",
    "flow_mag_max",
    "flow_mag_std",
    "spatial_inconsistency_mean",
    "motion_energy_mean",
    "angle_dispersion_mean",
    "temporal_jitter",
    "frame_pairs",
)

STAT_METRICS = (
    "flow_mag_mean",
    "flow_mag_max",
    "flow_mag_std",
    "spatial_inconsistency",
    "motion_energy",
    "angle_dispersion",
    "flow_u_mean",
    "flow_v_mean",
)

STAT_FIELDS = ("min", "max", "mean", "std", "median", "p25", "p75")


def profile_from_filename(filename: str) -> str | None:
    if filename.startswith("fake_ffpp_") or filename.endswith("_long.mp4"):
        return "ffpp_vox"
    if filename.startswith("celebdf_"):
        return "celebdf"
    return None


def feature_names() -> list[str]:
    names = [f"agg.{k}" for k in AGGREGATE_KEYS]
    for metric in STAT_METRICS:
        for field in STAT_FIELDS:
            names.append(f"stats.{metric}.{field}")
    return names


def _distribution_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {k: 0.0 for k in STAT_FIELDS}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def _temporal_jitter(per_pair: list[dict[str, Any]]) -> float:
    means = [float(row["flow_mag_mean"]) for row in per_pair if row.get("flow_mag_mean") is not None]
    if not means:
        return 0.0
    arr = np.asarray(means, dtype=np.float64)
    return float(arr.std() / (arr.mean() + 1e-6))


def _score_stats_from_pairs(per_pair: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    numeric_keys = list(STAT_METRICS)
    return {key: _distribution_stats([float(row[key]) for row in per_pair if key in row]) for key in numeric_keys}


def _raft_pair_to_per_pair(pair_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_pair: list[dict[str, Any]] = []
    for idx, pair in enumerate(pair_stats):
        mag_mean = float(pair.get("magnitude_mean", 0.0))
        mag_std = float(pair.get("magnitude_std", 0.0))
        per_pair.append(
            {
                "pair_index": idx,
                "flow_mag_mean": mag_mean,
                "flow_mag_max": float(pair.get("magnitude_max", 0.0)),
                "flow_mag_std": mag_std,
                "flow_u_mean": float(pair.get("flow_x_mean", 0.0)),
                "flow_v_mean": float(pair.get("flow_y_mean", 0.0)),
                "spatial_inconsistency": mag_std / (mag_mean + 1e-6),
                "motion_energy": mag_mean**2,
                "angle_dispersion": float(pair.get("angle_std", 0.0)),
            }
        )
    return per_pair


def normalize_report(report: dict[str, Any]) -> dict[str, Any]:
    """Unify RAFT-style pair_stats JSON and partial GMFlow score_breakdown (in-place)."""
    out = report
    sb = out.get("score_breakdown")
    if isinstance(sb, dict) and sb.get("aggregate"):
        aggregate = dict(sb["aggregate"])
        per_pair = list(sb.get("per_frame_pair") or [])
        if not per_pair and isinstance(out.get("pair_stats"), list):
            per_pair = _raft_pair_to_per_pair(out["pair_stats"])
        if per_pair:
            if aggregate.get("flow_mag_mean") is None:
                aggregate["flow_mag_mean"] = float(np.mean([p["flow_mag_mean"] for p in per_pair]))
            if aggregate.get("temporal_jitter") is None:
                aggregate["temporal_jitter"] = _temporal_jitter(per_pair)
            if aggregate.get("spatial_inconsistency_mean") is None:
                aggregate["spatial_inconsistency_mean"] = float(
                    np.mean([p["spatial_inconsistency"] for p in per_pair])
                )
            if aggregate.get("angle_dispersion_mean") is None:
                aggregate["angle_dispersion_mean"] = float(
                    np.mean([p["angle_dispersion"] for p in per_pair])
                )
            if aggregate.get("motion_energy_mean") is None:
                aggregate["motion_energy_mean"] = float(np.mean([p["motion_energy"] for p in per_pair]))
            if aggregate.get("frame_pairs") is None:
                aggregate["frame_pairs"] = len(per_pair)
        stats = sb.get("score_stats") or {}
        if not stats and per_pair:
            stats = _score_stats_from_pairs(per_pair)
        out["score_breakdown"] = {**sb, "aggregate": aggregate, "score_stats": stats, "per_frame_pair": per_pair}
        return out

    pair_stats = out.get("pair_stats")
    agg = out.get("aggregate") or {}
    if isinstance(pair_stats, list) and pair_stats and "magnitude_mean_mean" in agg:
        per_pair = _raft_pair_to_per_pair(pair_stats)
        flow_mag_mean = float(agg.get("magnitude_mean_mean", 0.0))
        internal = {
            "flow_mag_mean": flow_mag_mean,
            "flow_mag_max": float(agg.get("magnitude_max_mean", 0.0)),
            "flow_mag_std": float(agg.get("magnitude_std_mean", 0.0)),
            "spatial_inconsistency_mean": float(agg.get("magnitude_std_mean", 0.0))
            / (flow_mag_mean + 1e-6),
            "motion_energy_mean": flow_mag_mean**2,
            "angle_dispersion_mean": float(agg.get("angle_std_mean", 0.0)),
            "temporal_jitter": _temporal_jitter(per_pair),
            "frame_pairs": int(agg.get("pair_count", len(pair_stats))),
        }
        out["score_breakdown"] = {
            "aggregate": internal,
            "score_stats": _score_stats_from_pairs(per_pair),
            "per_frame_pair": per_pair,
        }
        out.setdefault("flow_mean", flow_mag_mean)
        return out

    if out.get("flow_mean") is not None:
        fm = float(out["flow_mean"])
        out["score_breakdown"] = {
            "aggregate": {
                "flow_mag_mean": fm,
                "flow_mag_max": float(out.get("flow_max") or fm),
                "flow_mag_std": float(out.get("flow_std") or 0.0),
                "spatial_inconsistency_mean": 0.0,
                "motion_energy_mean": fm**2,
                "angle_dispersion_mean": 0.0,
                "temporal_jitter": 0.0,
                "frame_pairs": int(out.get("frame_pairs_used") or out.get("frame_pairs") or 0),
            },
            "score_stats": {},
        }
    return out


def _float_value(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        v = float(value)
        return default if np.isnan(v) or np.isinf(v) else v
    except (TypeError, ValueError):
        return default


def features_from_report(report: dict[str, Any]) -> np.ndarray | None:
    if report.get("status") != "ok":
        return None
    report = normalize_report(report)
    breakdown = report.get("score_breakdown") or {}
    aggregate = breakdown.get("aggregate") or {}
    if not aggregate:
        return None

    stats = breakdown.get("score_stats") or {}
    vec: list[float] = []
    for key in AGGREGATE_KEYS:
        vec.append(_float_value(aggregate.get(key)))
    for metric in STAT_METRICS:
        block = stats.get(metric) or {}
        for field in STAT_FIELDS:
            vec.append(_float_value(block.get(field)))

    arr = np.asarray(vec, dtype=np.float64)
    if not np.isfinite(arr).any():
        return None
    return arr


def load_json_reports(json_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted(json_dir.glob("*.json")):
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports


def build_dataset(
    reports: list[dict[str, Any]],
    *,
    profile: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    files: list[str] = []
    skipped = 0
    for report in reports:
        fn = str(report.get("file") or "")
        if profile and profile_from_filename(fn) != profile:
            continue
        gt = report.get("ground_truth_label")
        if gt not in ("fake", "real"):
            continue
        feat = features_from_report(report)
        if feat is None:
            skipped += 1
            continue
        rows.append(feat)
        labels.append(1 if gt == "fake" else 0)
        files.append(fn)
    if not rows:
        raise ValueError(f"no feature rows (skipped={skipped})")
    return np.vstack(rows), np.asarray(labels, dtype=np.int64), files
