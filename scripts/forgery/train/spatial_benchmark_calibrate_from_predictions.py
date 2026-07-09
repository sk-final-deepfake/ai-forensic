#!/usr/bin/env python3
"""Write metrics.json from spatial benchmark predictions + optional gate sweep.

Used after spatial_mvtamperbench_benchmark.py when operating threshold != 0.5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load_scores_labels(predictions_path: Path) -> tuple[np.ndarray, np.ndarray]:
    items = json.loads(predictions_path.read_text(encoding="utf-8"))["items"]
    labels: list[int] = []
    scores: list[float] = []
    for x in items:
        g = x.get("ground_truth_label")
        y = 1 if g in ("fake", 1, "1", True) else 0
        labels.append(y)
        scores.append(float(x["tamper_score"]))
    return np.array(labels, dtype=np.int64), np.array(scores, dtype=np.float64)


def confusion_at_threshold(labels: np.ndarray, scores: np.ndarray, thr: float) -> dict[str, int]:
    pred = (scores >= thr).astype(np.int64)
    tp = int(((labels == 1) & (pred == 1)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    tn = int(((labels == 0) & (pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def write_metrics(
    *,
    predictions_path: Path,
    threshold: float,
    weights: str,
    note: str,
) -> dict:
    labels, scores = load_scores_labels(predictions_path)
    conf = confusion_at_threshold(labels, scores, threshold)
    tp, tn, fp, fn = conf["tp"], conf["tn"], conf["fp"], conf["fn"]
    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")
    run_id = predictions_path.parent.name
    metrics = {
        "run_id": run_id,
        "model": "trufor",
        "threshold": threshold,
        "total": len(labels),
        "ok": len(labels),
        "accuracy": (tp + tn) / max(1, len(labels)),
        "confusion": conf,
        "real": {
            "count": int((labels == 0).sum()),
            "avg_tamper_score": float(scores[labels == 0].mean()) if np.any(labels == 0) else float("nan"),
        },
        "fake": {
            "count": int((labels == 1).sum()),
            "avg_tamper_score": float(scores[labels == 1].mean()) if np.any(labels == 1) else float("nan"),
        },
        "roc_auc": auc,
        "weights": weights,
        "note": note,
    }
    out_path = predictions_path.parent / "metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def find_gate_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    min_tp: int,
    max_fp: int,
    step: float,
) -> tuple[float, dict[str, int]] | None:
    best_rank: tuple[float, int, int] | None = None
    best_thr = 0.0
    best_conf: dict[str, int] | None = None
    for thr in np.arange(0.05, 0.95, step):
        thr_f = round(float(thr), 4)
        conf = confusion_at_threshold(labels, scores, thr_f)
        tp, fp = conf["tp"], conf["fp"]
        if tp < min_tp or fp > max_fp:
            continue
        acc = (conf["tp"] + conf["tn"]) / len(labels)
        rank = (acc, tp, -fp)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_thr = thr_f
            best_conf = conf
    if best_conf is None:
        return None
    return best_thr, best_conf


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate spatial benchmark from predictions.json")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None, help="fixed thr; if omitted with --gate, auto-pick")
    parser.add_argument("--weights", default="")
    parser.add_argument("--gate", action="store_true", help="find thr with TP>=min_tp & FP<=max_fp")
    parser.add_argument("--min-tp", type=int, default=63)
    parser.add_argument("--max-fp", type=int, default=51)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument(
        "--note",
        default="generated from predictions.json (calibration thr)",
    )
    args = parser.parse_args()

    labels, scores = load_scores_labels(args.predictions)
    thr = args.threshold
    if thr is None:
        if not args.gate:
            raise SystemExit("Provide --threshold or --gate")
        found = find_gate_threshold(
            labels, scores, min_tp=args.min_tp, max_fp=args.max_fp, step=args.step
        )
        if found is None:
            raise SystemExit(f"No threshold in sweep satisfies TP>={args.min_tp} FP<={args.max_fp}")
        thr, conf = found
        print(f"gate thr={thr:.4f}  TP={conf['tp']} FP={conf['fp']} FN={conf['fn']} TN={conf['tn']}")

    metrics = write_metrics(
        predictions_path=args.predictions,
        threshold=thr,
        weights=args.weights,
        note=args.note,
    )
    c = metrics["confusion"]
    print(
        f"metrics: {args.predictions.parent / 'metrics.json'}  "
        f"thr={thr:.4f} TP={c['tp']} FP={c['fp']} Acc={metrics['accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
