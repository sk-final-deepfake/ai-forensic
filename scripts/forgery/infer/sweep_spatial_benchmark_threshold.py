#!/usr/bin/env python3
"""Sweep classification threshold on spatial benchmark predictions.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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
        "threshold": th,
        "accuracy": acc,
        "f1": f1,
        "youden_j": tpr - fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--step", type=float, default=0.005)
    args = p.parse_args()

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    ok = [x for x in payload.get("items", []) if x.get("tamper_score") is not None]
    y_true = np.array([1 if x["ground_truth_label"] == "fake" else 0 for x in ok])
    y_score = np.array([float(x["tamper_score"]) for x in ok])

    uniq = sorted(set(np.round(y_score, 6).tolist()))
    grid = sorted(
        set(np.arange(0.0, 1.0 + args.step, args.step).round(6).tolist())
        | set((np.array(uniq) - 1e-6).round(6).tolist())
        | set(np.array(uniq).round(6).tolist())
        | set((np.array(uniq) + 1e-6).round(6).tolist())
    )
    grid = [g for g in grid if 0.0 <= g <= 1.0]

    rows = [confusion(y_true, y_score, th) for th in grid]
    best_acc = max(rows, key=lambda r: (r["accuracy"], r["f1"]))
    best_f1 = max(rows, key=lambda r: (r["f1"], r["accuracy"]))
    best_j = max(rows, key=lambda r: (r["youden_j"], r["accuracy"]))

    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = None

    print(f"predictions: {args.predictions}")
    print(f"n={len(ok)}  roc_auc={auc}")
    print(f"real score  min={y_score[y_true==0].min():.4f}  max={y_score[y_true==0].max():.4f}  mean={y_score[y_true==0].mean():.4f}")
    print(f"fake score  min={y_score[y_true==1].min():.4f}  max={y_score[y_true==1].max():.4f}  mean={y_score[y_true==1].mean():.4f}")
    print()

    def show(title: str, r: dict) -> None:
        print(f"=== {title} ===")
        print(f"threshold={r['threshold']:.4f}  accuracy={r['accuracy']*100:.1f}%  f1={r['f1']:.3f}  youden_j={r['youden_j']:.3f}")
        print(f"  TN={r['tn']}  FP={r['fp']}  FN={r['fn']}  TP={r['tp']}")
        print()

    show("best accuracy", best_acc)
    show("best F1", best_f1)
    show("best Youden J (TPR-FPR)", best_j)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
