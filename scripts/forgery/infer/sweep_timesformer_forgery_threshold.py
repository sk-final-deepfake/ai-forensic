#!/usr/bin/env python3
"""Re-evaluate TimeSformer forgery benchmark with threshold sweep (no re-infer).

Reads ``results/infer/<run_id>/items.json`` from ``timesformer_forgery_benchmark.py``.
Reports best accuracy / F1 / Youden-J thresholds and optional JSON output.

Example (GPU):
  python3 forgery/scripts/infer/sweep_timesformer_forgery_threshold.py \\
    --items ~/forenShield-ai/forgery/results/infer/timesformer-forgery-v1.1-clip-20260704-0553-csvted200/items.json

  python3 forgery/scripts/infer/sweep_timesformer_forgery_threshold.py \\
    --run-id timesformer-forgery-v1.1-clip-20260704-0553-csvted200 \\
    --forgery-root ~/forenShield-ai/forgery \\
    --write-metrics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_items(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "items" in payload:
        return list(payload["items"])
    raise ValueError(f"unsupported items format: {path}")


def confusion(y_true: np.ndarray, y_score: np.ndarray, th: float) -> dict:
    pred = (y_score >= th).astype(int)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    n = len(y_true)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    tpr = rec
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "threshold": round(float(th), 6),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "youden_j": round(tpr - fpr, 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def build_threshold_grid(y_score: np.ndarray, step: float) -> list[float]:
    uniq = sorted(set(np.round(y_score, 6).tolist()))
    grid = set(np.arange(0.0, 1.0 + step, step).round(6).tolist())
    for u in uniq:
        grid.add(float(np.round(u, 6)))
        grid.add(float(np.round(u - 1e-6, 6)))
        grid.add(float(np.round(u + 1e-6, 6)))
    return sorted(g for g in grid if 0.0 <= g <= 1.0)


def summarize_run(
    items: list[dict],
    *,
    baseline_threshold: float,
    step: float,
) -> dict:
    ok = [x for x in items if x.get("status") == "ok" and x.get("tamper_score") is not None]
    y_true = np.array([1 if x["ground_truth_label"] == "fake" else 0 for x in ok], dtype=np.int64)
    y_score = np.array([float(x["tamper_score"]) for x in ok], dtype=np.float64)

    grid = build_threshold_grid(y_score, step)
    rows = [confusion(y_true, y_score, th) for th in grid]
    best_acc = max(rows, key=lambda r: (r["accuracy"], r["f1"]))
    best_f1 = max(rows, key=lambda r: (r["f1"], r["accuracy"]))
    best_j = max(rows, key=lambda r: (r["youden_j"], r["accuracy"]))
    at_baseline = confusion(y_true, y_score, baseline_threshold)

    try:
        from sklearn.metrics import roc_auc_score

        auc = round(float(roc_auc_score(y_true, y_score)), 4) if len(set(y_true.tolist())) > 1 else None
    except Exception:
        auc = None

    real_scores = y_score[y_true == 0]
    fake_scores = y_score[y_true == 1]

    return {
        "n_ok": len(ok),
        "n_total": len(items),
        "roc_auc": auc,
        "score_range": {
            "real": {
                "min": round(float(real_scores.min()), 6) if len(real_scores) else None,
                "max": round(float(real_scores.max()), 6) if len(real_scores) else None,
                "mean": round(float(real_scores.mean()), 6) if len(real_scores) else None,
            },
            "fake": {
                "min": round(float(fake_scores.min()), 6) if len(fake_scores) else None,
                "max": round(float(fake_scores.max()), 6) if len(fake_scores) else None,
                "mean": round(float(fake_scores.mean()), 6) if len(fake_scores) else None,
            },
        },
        f"metrics_at_{baseline_threshold}": at_baseline,
        "best_accuracy": best_acc,
        "best_f1": best_f1,
        "best_youden_j": best_j,
        "run_id": ok[0].get("run_id") if ok else None,
    }


def resolve_items_path(
    *,
    items: Path | None,
    run_id: str | None,
    forgery_root: Path,
) -> Path:
    if items is not None:
        p = items.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    if not run_id:
        raise ValueError("provide --items or --run-id")
    p = forgery_root / "results/infer" / run_id / "items.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description="Threshold sweep on TimeSformer forgery items.json")
    parser.add_argument("--items", type=Path, default=None, help="path to items.json")
    parser.add_argument("--run-id", default=None, help="infer run_id under results/infer/")
    parser.add_argument(
        "--forgery-root",
        type=Path,
        default=Path("~/forenShield-ai/forgery"),
        help="forgery root when using --run-id",
    )
    parser.add_argument("--baseline-threshold", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument(
        "--write-metrics",
        action="store_true",
        help="write metrics_threshold_sweep.json next to eval/metrics.json",
    )
    args = parser.parse_args()

    forgery_root = Path(args.forgery_root).expanduser().resolve()
    items_path = resolve_items_path(items=args.items, run_id=args.run_id, forgery_root=forgery_root)
    items = load_items(items_path)
    summary = summarize_run(
        items,
        baseline_threshold=args.baseline_threshold,
        step=args.step,
    )
    summary["items_path"] = str(items_path)
    summary["baseline_threshold"] = args.baseline_threshold

    print(f"items: {items_path}")
    print(f"n_ok={summary['n_ok']}/{summary['n_total']}  roc_auc={summary['roc_auc']}")
    sr = summary["score_range"]
    print(
        f"real  min={sr['real']['min']}  max={sr['real']['max']}  mean={sr['real']['mean']}"
    )
    print(
        f"fake  min={sr['fake']['min']}  max={sr['fake']['max']}  mean={sr['fake']['mean']}"
    )
    print()

    def show(title: str, r: dict) -> None:
        c = r["confusion"]
        print(f"=== {title} ===")
        print(
            f"threshold={r['threshold']:.4f}  accuracy={r['accuracy']*100:.1f}%  "
            f"f1={r['f1']:.3f}  youden_j={r['youden_j']:.3f}"
        )
        print(f"  TN={c['tn']}  FP={c['fp']}  FN={c['fn']}  TP={c['tp']}")
        print()

    show(f"baseline @ {args.baseline_threshold}", summary[f"metrics_at_{args.baseline_threshold}"])
    show("best accuracy", summary["best_accuracy"])
    show("best F1", summary["best_f1"])
    show("best Youden J (TPR-FPR)", summary["best_youden_j"])

    if args.write_metrics and summary.get("run_id"):
        eval_dir = forgery_root / "results/eval" / str(summary["run_id"])
        eval_dir.mkdir(parents=True, exist_ok=True)
        out = eval_dir / "metrics_threshold_sweep.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote: {out}")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
