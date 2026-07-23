#!/usr/bin/env python3
"""
Compare fake 50 vs real 50 from PWC-Net benchmark_report.json and sweep thresholds.

Usage:
  python scripts/eval/analyze_optical_flow_threshold.py results/infer/RUN_ID/benchmark_report.json
  python scripts/eval/analyze_optical_flow_threshold.py --root . --run-id pwcnet-celebdf-20260622-041153
  python scripts/eval/analyze_optical_flow_threshold.py report1.json report2.json -o threshold_analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

SCORE_KEYS = (
    "fake_score",
    "motion_anomaly_score",
    "temporal_jitter",
    "spatial_inconsistency_mean",
    "angle_dispersion_mean",
    "flow_mean",
    "flow_max",
    "motion_energy_mean",
    "confidence",
    "entropy",
)

AUX_KEYS = (
    "temporal_jitter",
    "spatial_inconsistency_mean",
    "angle_dispersion_mean",
    "flow_mean",
    "frame_vote_ratio",
)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def distribution_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p25": None,
            "p75": None,
        }
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "p25": float(_percentile(values, 25)),
        "p75": float(_percentile(values, 75)),
    }


def _item_value(item: dict[str, Any], key: str) -> float | None:
    if key == "frame_vote_ratio":
        votes = item.get("frame_votes") or item.get("pair_votes") or {}
        fake = votes.get("fake")
        real = votes.get("real")
        if fake is None or real is None:
            return None
        total = fake + real
        return float(fake / total) if total else None
    val = item.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _ok_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [i for i in items if i.get("status", "ok") == "ok" and i.get("ground_truth_label") in ("fake", "real")]


def _split_groups(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ok = _ok_items(items)
    fake = [i for i in ok if i.get("ground_truth_label") == "fake"]
    real = [i for i in ok if i.get("ground_truth_label") == "real"]
    return fake, real


def _collect_values(items: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for item in items:
        val = _item_value(item, key)
        if val is not None and not math.isnan(val):
            out.append(val)
    return out


def _confusion(scores: list[tuple[float, str]], threshold: float) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for score, label in scores:
        pred_fake = score >= threshold
        is_fake = label == "fake"
        if pred_fake and is_fake:
            tp += 1
        elif pred_fake and not is_fake:
            fp += 1
        elif not pred_fake and not is_fake:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _metrics_from_confusion(c: dict[str, int]) -> dict[str, float | None]:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    tpr = recall
    fpr = fp / (fp + tn) if (fp + tn) else None
    youden = (tpr - fpr) if (tpr is not None and fpr is not None) else None
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "youden": youden,
    }


def sweep_threshold(
    fake_items: list[dict[str, Any]],
    real_items: list[dict[str, Any]],
    score_key: str,
    *,
    step: float = 0.01,
) -> tuple[list[dict[str, Any]], list[tuple[float, str]]]:
    scores: list[tuple[float, str]] = []
    for item in fake_items + real_items:
        val = _item_value(item, score_key)
        if val is None:
            continue
        scores.append((val, str(item.get("ground_truth_label"))))

    sweep: list[dict[str, Any]] = []
    threshold = 0.0
    while threshold <= 1.0 + 1e-9:
        c = _confusion(scores, threshold)
        m = _metrics_from_confusion(c)
        sweep.append({"threshold": round(threshold, 4), **c, **m})
        threshold += step
    return sweep, scores


def _best_by(sweep: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    candidates = [row for row in sweep if row.get(metric) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row[metric]))


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def _resolve_score_key(items: list[dict[str, Any]], preferred: str) -> str:
    ok = _ok_items(items)
    if not ok:
        return preferred
    if preferred != "auto":
        return preferred
    for key in ("fake_score", "motion_anomaly_score"):
        if any(_item_value(i, key) is not None for i in ok):
            return key
    return "fake_score"


def analyze_report(
    report_path: Path,
    *,
    score_key: str = "auto",
    step: float = 0.01,
) -> dict[str, Any]:
    doc = json.loads(report_path.read_text(encoding="utf-8"))
    items = doc.get("items") or []
    score_key = _resolve_score_key(items, score_key)
    fake_items, real_items = _split_groups(items)
    current_threshold = float(doc.get("threshold", 0.5))

    group_stats: dict[str, Any] = {}
    for label, group in (("fake", fake_items), ("real", real_items)):
        group_stats[label] = {"count": len(group)}
        for key in SCORE_KEYS:
            vals = _collect_values(group, key)
            if vals:
                group_stats[label][key] = distribution_stats(vals)
        vote_ratio = _collect_values(group, "frame_vote_ratio")
        if vote_ratio:
            group_stats[label]["frame_vote_ratio"] = distribution_stats(vote_ratio)

    fake_scores = _collect_values(fake_items, score_key)
    real_scores = _collect_values(real_items, score_key)
    fake_mean = statistics.mean(fake_scores) if fake_scores else None
    real_mean = statistics.mean(real_scores) if real_scores else None
    midpoint = (fake_mean + real_mean) / 2 if fake_mean is not None and real_mean is not None else None

    sweep, _ = sweep_threshold(fake_items, real_items, score_key, step=step)
    current_row = next((r for r in sweep if abs(r["threshold"] - current_threshold) < 1e-6), None)
    midpoint_row = None
    if midpoint is not None:
        nearest = min(sweep, key=lambda r: abs(r["threshold"] - midpoint))
        midpoint_row = {**nearest, "midpoint_score": midpoint}

    aux_sweeps: dict[str, Any] = {}
    for key in AUX_KEYS:
        if key == score_key:
            continue
        if not _collect_values(fake_items, key) and not _collect_values(real_items, key):
            continue
        aux_sweep, _ = sweep_threshold(fake_items, real_items, key, step=step)
        best_f1 = _best_by(aux_sweep, "f1")
        if best_f1:
            aux_sweeps[key] = {
                "best_f1_threshold": best_f1["threshold"],
                "best_f1": best_f1["f1"],
                "best_f1_accuracy": best_f1["accuracy"],
            }

    overlap_note = None
    if fake_scores and real_scores:
        fake_p25 = _percentile(fake_scores, 25)
        real_p75 = _percentile(real_scores, 75)
        if fake_p25 <= real_p75:
            overlap_note = (
                "fake p25 <= real p75: distributions overlap — single threshold may be weak"
            )
        else:
            overlap_note = "fake p25 > real p75: partial separation exists"

    return {
        "source": str(report_path),
        "run_id": doc.get("run_id"),
        "model": doc.get("model"),
        "profile": doc.get("profile"),
        "method": doc.get("method"),
        "score_key": score_key,
        "counts": {
            "total": len(items),
            "ok": len(_ok_items(items)),
            "fake": len(fake_items),
            "real": len(real_items),
        },
        "group_stats": group_stats,
        "separation": {
            "mean_fake": fake_mean,
            "mean_real": real_mean,
            "mean_gap": (fake_mean - real_mean) if fake_mean is not None and real_mean is not None else None,
            "midpoint_threshold": midpoint,
            "overlap_note": overlap_note,
        },
        "current_threshold": current_threshold,
        "current_metrics": current_row,
        "midpoint_nearest_metrics": midpoint_row,
        "best_thresholds": {
            "by_accuracy": _best_by(sweep, "accuracy"),
            "by_f1": _best_by(sweep, "f1"),
            "by_youden": _best_by(sweep, "youden"),
        },
        "auxiliary_metric_sweeps": aux_sweeps,
        "sweep": sweep,
    }


def _print_summary(result: dict[str, Any]) -> None:
    profile = result.get("profile") or "?"
    run_id = result.get("run_id") or "?"
    key = result["score_key"]
    sep = result["separation"]
    gs = result["group_stats"]

    print("=" * 72)
    print(f"profile: {profile}  run_id: {run_id}")
    print(f"score: {key}  current threshold: {result['current_threshold']}")
    print(f"counts: fake={result['counts']['fake']} real={result['counts']['real']}")
    print("-" * 72)
    print(f"{'':16} {'fake mean':>12} {'real mean':>12} {'gap':>12}")
    fake_mean = sep.get("mean_fake")
    real_mean = sep.get("mean_real")
    gap = sep.get("mean_gap")
    print(f"{key:16} {_fmt(fake_mean):>12} {_fmt(real_mean):>12} {_fmt(gap):>12}")
    for aux in ("temporal_jitter", "spatial_inconsistency_mean", "angle_dispersion_mean", "frame_vote_ratio"):
        fm = (gs.get("fake") or {}).get(aux, {}).get("mean")
        rm = (gs.get("real") or {}).get(aux, {}).get("mean")
        if fm is not None and rm is not None:
            print(f"{aux:16} {_fmt(fm):>12} {_fmt(rm):>12} {_fmt(fm - rm):>12}")
    print("-" * 72)
    if sep.get("overlap_note"):
        print(sep["overlap_note"])
    if sep.get("midpoint_threshold") is not None:
        print(f"suggested midpoint threshold (mean_fake+mean_real)/2: {_fmt(sep['midpoint_threshold'])}")

    cur = result.get("current_metrics") or {}
    print(
        f"at current T={result['current_threshold']}: "
        f"acc={_fmt(cur.get('accuracy'))} "
        f"prec={_fmt(cur.get('precision'))} "
        f"rec={_fmt(cur.get('recall'))} "
        f"f1={_fmt(cur.get('f1'))}"
    )

    for name in ("by_f1", "by_accuracy", "by_youden"):
        best = (result.get("best_thresholds") or {}).get(name)
        if not best:
            continue
        label = name.replace("by_", "")
        print(
            f"best by {label:8}: T={_fmt(best['threshold'], 2)} "
            f"acc={_fmt(best.get('accuracy'))} "
            f"prec={_fmt(best.get('precision'))} "
            f"rec={_fmt(best.get('recall'))} "
            f"f1={_fmt(best.get('f1'))}"
        )

    aux = result.get("auxiliary_metric_sweeps") or {}
    if aux:
        print("-" * 72)
        print("auxiliary single-metric best F1 thresholds:")
        for metric, info in aux.items():
            print(
                f"  {metric}: T={_fmt(info['best_f1_threshold'], 2)} "
                f"f1={_fmt(info['best_f1'])} acc={_fmt(info['best_f1_accuracy'])}"
            )
    print("=" * 72)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        help="benchmark_report.json path(s)",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="forenShield-ai root")
    parser.add_argument("--run-id", action="append", default=[], help="resolve results/infer/RUN_ID/benchmark_report.json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="write combined JSON analysis (default: print only)",
    )
    parser.add_argument(
        "--score-key",
        default="auto",
        help="primary score field for sweep (default: auto = fake_score or motion_anomaly_score)",
    )
    parser.add_argument("--step", type=float, default=0.01, help="threshold sweep step")
    args = parser.parse_args()

    paths: list[Path] = list(args.reports)
    for run_id in args.run_id:
        paths.append(args.root / "results" / "infer" / run_id / "benchmark_report.json")

    if not paths:
        parser.error("provide benchmark_report.json path(s) or --run-id")

    results: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            print(f"ERROR: missing {path}", file=sys.stderr)
            sys.exit(1)
        result = analyze_report(path, score_key=args.score_key, step=args.step)
        results.append(result)
        _print_summary(result)

    if args.output:
        payload = results[0] if len(results) == 1 else {"profiles": results}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
