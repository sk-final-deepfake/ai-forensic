#!/usr/bin/env python3
"""GMFlow optical-flow JSON 후처리: motion_anomaly_score / fake_score / pred_label 산출.

infer 기본 출력(aggregate + pair_stats)에는 점수가 없음.
이 스크립트가 RUN 내 real 영상으로 baseline을 잡고 휴리스틱 점수를 붙인다.

v2: aggregate + score_stats + per-pair segment 통계를 모두 scorer에 반영.

Usage (GPU):
  cd ~/forenShield-ai
  python3 scripts/eval/gmflow_motion_score.py \\
    --run-id optical-flow-ffpp-vox-20260622-0544 \\
    --profile ffpp_vox \\
    --s3-run-id gmflow-ffpp-vox-benchmark-20260622-0544
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_THRESHOLD = 0.5
ANOMALY_SCALE = 2.0
ANGLE_SCALE = 48.0
SCORE_METHOD = "optical_flow_motion_heuristic_v2"
SCHEMA_VERSION = "1.3"

# aggregate + segment feature weights (sum = 1.0)
SIGNAL_WEIGHTS = {
    "temporal_jitter": 0.18,
    "spatial_inconsistency_mean": 0.14,
    "angle_dispersion_mean": 0.14,
    "flow_mag_mean": 0.10,
    "motion_energy_mean": 0.08,
    "flow_mag_iqr": 0.10,
    "flow_mag_cv": 0.08,
    "flow_mag_pair_range": 0.08,
    "flow_max_spread": 0.05,
    "motion_energy_jitter": 0.05,
}

SIGNAL_LABELS = {
    "temporal_jitter": "temporal_jitter",
    "spatial_inconsistency_mean": "spatial_inconsistency",
    "angle_dispersion_mean": "angle_dispersion",
    "flow_mag_mean": "flow_magnitude",
    "motion_energy_mean": "motion_energy",
    "flow_mag_iqr": "flow_mag_iqr",
    "flow_mag_cv": "flow_mag_cv",
    "flow_mag_pair_range": "flow_mag_pair_range",
    "flow_max_spread": "flow_max_spread",
    "motion_energy_jitter": "motion_energy_jitter",
}


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _stats_dict(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }
    ordered = sorted(values)
    n = len(ordered)
    p25 = ordered[max(0, (n - 1) // 4)]
    p75 = ordered[min(n - 1, (3 * (n - 1)) // 4)]
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "median": _median(values),
        "mean": statistics.mean(values),
        "std": _pstdev(values),
        "p25": p25,
        "p75": p75,
    }


def pair_stats_to_per_pair(pair_stats: list[dict[str, Any]]) -> list[dict[str, float]]:
    per_pair: list[dict[str, float]] = []
    for i, pair in enumerate(pair_stats):
        mag_mean = float(pair.get("magnitude_mean", 0.0))
        mag_std = float(pair.get("magnitude_std", 0.0))
        mag_max = float(pair.get("magnitude_max", mag_mean))
        angle_std = float(pair.get("angle_std", 0.0))
        fx = float(pair.get("flow_x_mean", 0.0))
        fy = float(pair.get("flow_y_mean", 0.0))
        per_pair.append(
            {
                "pair_index": float(i),
                "flow_mag_mean": mag_mean,
                "flow_mag_max": mag_max,
                "flow_mag_std": mag_std,
                "spatial_inconsistency": mag_std,
                "angle_dispersion": angle_std / ANGLE_SCALE,
                "motion_energy": fx * fx + fy * fy,
            }
        )
    return per_pair


def aggregate_from_per_pair(per_pair: list[dict[str, float]]) -> dict[str, float]:
    mags = [p["flow_mag_mean"] for p in per_pair]
    return {
        "flow_mag_mean": statistics.mean(mags) if mags else 0.0,
        "flow_mag_max": max((p["flow_mag_max"] for p in per_pair), default=0.0),
        "flow_mag_std": statistics.mean([p["flow_mag_std"] for p in per_pair]) if per_pair else 0.0,
        "temporal_jitter": _pstdev(mags),
        "spatial_inconsistency_mean": statistics.mean([p["spatial_inconsistency"] for p in per_pair])
        if per_pair
        else 0.0,
        "angle_dispersion_mean": statistics.mean([p["angle_dispersion"] for p in per_pair])
        if per_pair
        else 0.0,
        "motion_energy_mean": statistics.mean([p["motion_energy"] for p in per_pair])
        if per_pair
        else 0.0,
        "frame_pairs": float(len(per_pair)),
    }


def aggregate_from_infer_blob(aggregate: dict[str, Any]) -> dict[str, float]:
    mag_mean = float(aggregate.get("magnitude_mean_mean", aggregate.get("flow_mag_mean", 0.0)))
    return {
        "flow_mag_mean": mag_mean,
        "flow_mag_max": float(aggregate.get("magnitude_max_mean", aggregate.get("flow_mag_max", 0.0))),
        "flow_mag_std": float(aggregate.get("magnitude_std_mean", aggregate.get("flow_mag_std", 0.0))),
        "temporal_jitter": float(aggregate.get("magnitude_mean_std", aggregate.get("temporal_jitter", 0.0))),
        "spatial_inconsistency_mean": float(
            aggregate.get("magnitude_std_mean", aggregate.get("spatial_inconsistency_mean", 0.0))
        ),
        "angle_dispersion_mean": float(aggregate.get("angle_std_mean", 0.0)) / ANGLE_SCALE,
        "motion_energy_mean": float(aggregate.get("motion_energy_mean", 0.0)),
        "frame_pairs": float(aggregate.get("pair_count", aggregate.get("frame_pairs", 0.0))),
    }


def build_segment_features(
    per_pair: list[dict[str, float]],
    score_stats: dict[str, dict[str, float]],
) -> dict[str, float]:
    sm = score_stats.get("flow_mag_mean", {})
    sx = score_stats.get("flow_mag_max", {})
    sm_mean = float(sm.get("mean", 0.0))
    segment = {
        "flow_mag_iqr": float(sm.get("p75", 0.0)) - float(sm.get("p25", 0.0)),
        "flow_mag_cv": float(sm.get("std", 0.0)) / (sm_mean + 1e-9),
        "flow_mag_pair_range": float(sm.get("max", 0.0)) - float(sm.get("min", 0.0)),
        "flow_max_spread": float(sx.get("max", 0.0)) - float(sx.get("min", 0.0)),
        "motion_energy_jitter": 0.0,
    }
    if per_pair:
        energies = [float(p["motion_energy"]) for p in per_pair]
        segment["motion_energy_jitter"] = _pstdev(energies)
    return segment


def merge_features_for_scoring(
    aggregate: dict[str, float],
    segment: dict[str, float],
) -> dict[str, float]:
    merged = dict(aggregate)
    for key in SIGNAL_WEIGHTS:
        if key in segment:
            merged[key] = float(segment[key])
    return merged


def extract_features(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("status") != "ok":
        return None

    sb = record.get("score_breakdown") or {}
    if sb.get("aggregate"):
        aggregate = {k: float(v) for k, v in sb["aggregate"].items() if isinstance(v, (int, float))}
        per_pair = sb.get("per_frame_pair") or []
        pair_stats = record.get("pair_stats") or []
        if not per_pair and pair_stats:
            per_pair = pair_stats_to_per_pair(pair_stats)
        if not aggregate and per_pair:
            aggregate = aggregate_from_per_pair(per_pair)
    elif record.get("pair_stats"):
        per_pair = pair_stats_to_per_pair(record["pair_stats"])
        aggregate = aggregate_from_per_pair(per_pair)
    elif record.get("aggregate"):
        per_pair = []
        aggregate = aggregate_from_infer_blob(record["aggregate"])
    else:
        return None

    mags = [float(p.get("flow_mag_mean", p.get("magnitude_mean", 0.0))) for p in per_pair]
    maxs = [float(p.get("flow_mag_max", p.get("magnitude_max", 0.0))) for p in per_pair]
    if not mags and aggregate.get("flow_mag_mean") is not None:
        mags = [float(aggregate["flow_mag_mean"])]
        maxs = [float(aggregate.get("flow_mag_max", mags[0]))]

    score_stats = {
        "flow_mag_mean": _stats_dict(mags),
        "flow_mag_max": _stats_dict(maxs or mags),
    }
    segment = build_segment_features(per_pair, score_stats)

    return {
        "aggregate": aggregate,
        "per_pair": per_pair,
        "score_stats": score_stats,
        "segment": segment,
        "flow_mag_pair_range": segment["flow_mag_pair_range"],
        "scoring_vector": merge_features_for_scoring(aggregate, segment),
    }


def build_baseline(ok_records: list[dict[str, Any]]) -> dict[str, float]:
    reals = [r for r in ok_records if r.get("ground_truth_label") == "real"]
    if not reals:
        reals = ok_records

    baselines: dict[str, list[float]] = {k: [] for k in SIGNAL_WEIGHTS}
    for record in reals:
        feats = record.get("_features")
        if not feats:
            continue
        merged = feats.get("scoring_vector") or merge_features_for_scoring(
            feats["aggregate"], feats.get("segment", {})
        )
        for key in SIGNAL_WEIGHTS:
            if key in merged:
                baselines[key].append(float(merged[key]))

    return {k: _median(v) if v else 0.0 for k, v in baselines.items()}


def compute_motion_anomaly_score(
    scoring_vector: dict[str, float],
    baseline: dict[str, float],
) -> float:
    total = 0.0
    for key, weight in SIGNAL_WEIGHTS.items():
        value = float(scoring_vector.get(key, 0.0))
        base = float(baseline.get(key, 0.0))
        rel = abs(value - base) / (abs(base) + 1e-9)
        total += weight * rel
    return min(1.0, total / ANOMALY_SCALE)


def build_signals(
    scoring_vector: dict[str, float],
    baseline: dict[str, float],
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for key in SIGNAL_WEIGHTS:
        value = float(scoring_vector.get(key, 0.0))
        base = float(baseline.get(key, 0.0))
        delta = value - base
        signals.append(
            {
                "name": SIGNAL_LABELS.get(key, key),
                "value": round(value, 6),
                "real_cohort_baseline": round(base, 6),
                "delta": round(delta, 6),
                "direction": "above" if delta > 0 else "below",
            }
        )
    return signals


def enrich_record(
    record: dict[str, Any],
    baseline: dict[str, float],
    threshold: float,
    run_id: str,
) -> dict[str, Any]:
    out = dict(record)
    if out.get("status") != "ok":
        out["motion_anomaly_score"] = None
        out["fake_score"] = None
        out["pred_label"] = None
        return out

    feats = out.pop("_features", None)
    if feats is None:
        feats = extract_features(out)
    if feats is None:
        out["motion_anomaly_score"] = None
        out["fake_score"] = None
        out["pred_label"] = None
        return out

    aggregate = feats["aggregate"]
    segment = feats.get("segment", {})
    scoring_vector = feats.get("scoring_vector") or merge_features_for_scoring(aggregate, segment)
    mas = compute_motion_anomaly_score(scoring_vector, baseline)
    pred = "fake" if mas >= threshold else "real"

    out["schema_version"] = SCHEMA_VERSION
    out["run_id"] = run_id
    out["motion_anomaly_score"] = round(mas, 6)
    out["fake_score"] = round(mas, 6)
    out["pred_label"] = pred
    out["score_breakdown"] = {
        "schema_version": SCHEMA_VERSION,
        "method": SCORE_METHOD,
        "threshold": threshold,
        "frames_sampled": None,
        "frame_pairs_used": int(aggregate.get("frame_pairs", 0)),
        "aggregate": aggregate,
        "segment": segment,
        "score_stats": feats["score_stats"],
        "flow_mag_pair_range": feats["flow_mag_pair_range"],
        "scoring_vector": {k: round(float(scoring_vector[k]), 6) for k in SIGNAL_WEIGHTS if k in scoring_vector},
        "motion_anomaly_score": round(mas, 6),
        "pred_label": pred,
        "real_cohort_baseline": {k: round(v, 6) for k, v in baseline.items()},
    }
    out["interpretation"] = {
        "method": SCORE_METHOD,
        "summary": f"motion_anomaly_score={mas:.4f} (threshold={threshold}); pred_label={pred}",
        "signals": build_signals(scoring_vector, baseline),
    }
    return out


def load_json_dir(json_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(json_dir.glob("*.json")):
        rows.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return rows


def write_outputs(
    root: Path,
    run_id: str,
    profile: str,
    model: str,
    threshold: float,
    enriched: list[dict[str, Any]],
    json_dir: Path,
    json_paths: list[Path],
) -> None:
    for path, record in zip(json_paths, enriched, strict=True):
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    ok_scored = [
        r
        for r in enriched
        if r.get("status") == "ok" and r.get("fake_score") is not None
    ]

    def accuracy(items: list[dict[str, Any]]) -> float | None:
        if not items:
            return None
        correct = sum(1 for x in items if x.get("pred_label") == x.get("ground_truth_label"))
        return round(correct / len(items), 4)

    fake = [x for x in ok_scored if x.get("ground_truth_label") == "fake"]
    real = [x for x in ok_scored if x.get("ground_truth_label") == "real"]

    infer_summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model": model,
        "profile": profile,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "optical_flow",
        "score_method": SCORE_METHOD,
        "threshold": threshold,
        "count": len(enriched),
        "ok": len(ok_scored),
        "error": len(enriched) - len(ok_scored),
        "items": [
            {
                "file": r.get("file") or Path(r.get("source_path", "")).name,
                "ground_truth_label": r.get("ground_truth_label"),
                "status": r.get("status"),
                "model": model,
                "fake_score": r.get("fake_score"),
                "motion_anomaly_score": r.get("motion_anomaly_score"),
                "pred_label": r.get("pred_label"),
                "threshold": threshold,
                "flow_mag_pair_min": (r.get("score_breakdown") or {}).get("score_stats", {})
                .get("flow_mag_mean", {})
                .get("min"),
                "flow_mag_pair_max": (r.get("score_breakdown") or {}).get("score_stats", {})
                .get("flow_mag_mean", {})
                .get("max"),
                "flow_mag_pair_range": (r.get("score_breakdown") or {}).get("flow_mag_pair_range"),
                "flow_mag_iqr": (r.get("score_breakdown") or {}).get("segment", {}).get("flow_mag_iqr"),
                "temporal_jitter": (r.get("score_breakdown") or {}).get("aggregate", {}).get(
                    "temporal_jitter"
                ),
                "flow_mean": (r.get("score_breakdown") or {}).get("aggregate", {}).get(
                    "flow_mag_mean"
                ),
            }
            for r in enriched
        ],
    }

    metrics = {
        "run_id": run_id,
        "model": model,
        "profile": profile,
        "method": "optical_flow",
        "score_method": SCORE_METHOD,
        "threshold": threshold,
        "total": len(enriched),
        "ok": len(ok_scored),
        "error": len(enriched) - len(ok_scored),
        "heuristic_accuracy": accuracy(ok_scored),
        "fake": {
            "total": len(fake),
            "avg_fake_score": round(statistics.mean(x["fake_score"] for x in fake), 6) if fake else None,
            "accuracy": accuracy(fake),
        },
        "real": {
            "total": len(real),
            "avg_fake_score": round(statistics.mean(x["fake_score"] for x in real), 6) if real else None,
            "accuracy": accuracy(real),
        },
        "real_cohort_baseline": None,
        "note": "fake_score=motion_anomaly_score heuristic v2; not CNN probability",
    }
    if ok_scored:
        baseline = (ok_scored[0].get("score_breakdown") or {}).get("real_cohort_baseline")
        metrics["real_cohort_baseline"] = baseline

    infer_dir = root / "results" / "infer" / run_id / "datasets"
    eval_dir = root / "results" / "eval" / run_id
    infer_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    infer_summary_path = infer_dir / "infer_summary.json"
    metrics_path = eval_dir / "metrics.json"
    infer_summary_path.write_text(json.dumps(infer_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"scored json: {json_dir} ({len(enriched)} files)")
    print(f"score_method: {SCORE_METHOD}")
    print(f"infer_summary: {infer_summary_path}")
    print(f"metrics: {metrics_path}")
    print(
        f"heuristic_accuracy={metrics['heuristic_accuracy']} "
        f"ok={metrics['ok']}/{metrics['total']}"
    )


def resolve_json_dir(root: Path, run_id: str) -> Path:
    candidates = [
        root / "results" / "infer" / run_id / "gmflow" / "json",
        root / "results" / "infer" / run_id / "json",
    ]
    for path in candidates:
        if path.is_dir() and any(path.glob("*.json")):
            return path
    raise FileNotFoundError(f"No gmflow json dir found for run_id={run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GMFlow motion heuristic scoring post-process (v2)")
    parser.add_argument("--root", default=".", help="forenShield-ai root")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", default="ffpp_vox")
    parser.add_argument("--model", default="gmflow")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--s3-run-id", default=None, help="optional label for infer_summary run_id")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    json_dir = resolve_json_dir(root, args.run_id)
    rows = load_json_dir(json_dir)

    working: list[dict[str, Any]] = []
    for _, record in rows:
        rec = dict(record)
        if rec.get("ground_truth_label") is None and rec.get("label"):
            rec["ground_truth_label"] = rec["label"]
        feats = extract_features(rec)
        if feats:
            rec["_features"] = feats
        working.append(rec)

    ok_for_baseline = [r for r in working if r.get("status") == "ok" and r.get("_features")]
    baseline = build_baseline(ok_for_baseline)

    summary_run_id = args.s3_run_id or args.run_id
    enriched = [
        enrich_record(rec, baseline, args.threshold, summary_run_id) for rec in working
    ]
    paths = [p for p, _ in rows]
    write_outputs(
        root,
        summary_run_id,
        args.profile,
        args.model,
        args.threshold,
        enriched,
        json_dir,
        paths,
    )


if __name__ == "__main__":
    main()
