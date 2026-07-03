#!/usr/bin/env python3
"""Evaluate mvtb hold-out predictions with locked threshold + subset breakdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load_items(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def confusion(labels: np.ndarray, scores: np.ndarray, thr: float) -> dict[str, int]:
    pred = (scores >= thr).astype(np.int64)
    tp = int(((labels == 1) & (pred == 1)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    tn = int(((labels == 0) & (pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def report_block(name: str, items: list[dict], thr: float, gate_tp: int | None, gate_fp: int | None) -> dict:
    labels = np.array([1 if x.get("ground_truth_label") == "fake" else 0 for x in items], dtype=np.int64)
    scores = np.array([float(x["tamper_score"]) for x in items], dtype=np.float64)
    c = confusion(labels, scores, thr)
    acc = (c["tp"] + c["tn"]) / max(1, len(items))
    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")
    gate = ""
    if gate_tp is not None and gate_fp is not None:
        ok = c["tp"] >= gate_tp and c["fp"] <= gate_fp
        gate = f"  gate TP>={gate_tp} FP<={gate_fp}: {'PASS' if ok else 'FAIL'}"
    print(
        f"\n=== {name} (n={len(items)}) @thr={thr:.4f} ===\n"
        f"  TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']}  Acc={acc:.3f}  AUC={auc:.4f}{gate}"
    )
    return {"n": len(items), "threshold": thr, "confusion": c, "accuracy": acc, "roc_auc": auc}


def subset_items(items: list[dict], manifest_path: Path | None, subset: str) -> list[dict]:
    if not manifest_path or not manifest_path.exists():
        return items
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rel_to_subset: dict[str, str] = {}
    for x in manifest.get("items", []):
        sub = x.get("subset", "")
        rel_to_subset[x.get("relative_path", "")] = sub
        if x.get("pool_path"):
            rel_to_subset[x["pool_path"]] = sub
    out = []
    for x in items:
        rel = x.get("relative_path") or x.get("video_rel") or ""
        if rel_to_subset.get(rel) == subset:
            out.append(x)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--locked-thr", type=float, default=0.185)
    parser.add_argument("--gate-tp-200", type=int, default=63)
    parser.add_argument("--gate-fp-200", type=int, default=51)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    items = load_items(args.predictions)
    n = len(items)
    scale = n / 200.0
    gate_tp = int(round(args.gate_tp_200 * scale))
    gate_fp = int(round(args.gate_fp_200 * scale))

    print(f"predictions: {args.predictions}  n={n}")
    print(f"locked threshold (from mvtb200 calibration): {args.locked_thr:.4f}")
    print(f"scaled gate for n={n}: TP>={gate_tp} FP<={gate_fp}")

    results: dict = {"locked_thr": args.locked_thr, "n": n, "scaled_gate": {"min_tp": gate_tp, "max_fp": gate_fp}}

    results["full"] = report_block("FULL", items, args.locked_thr, gate_tp, gate_fp)

    calib = subset_items(items, args.manifest, "calibration_200")
    ood = subset_items(items, args.manifest, "ood_new")
    if calib:
        results["calibration_200"] = report_block(
            "calibration_200 (used for thr tune)",
            calib,
            args.locked_thr,
            args.gate_tp_200,
            args.gate_fp_200,
        )
    if ood:
        ood_n = len(ood)
        ood_gate_tp = int(round(args.gate_tp_200 * ood_n / 200.0))
        ood_gate_fp = int(round(args.gate_fp_200 * ood_n / 200.0))
        results["ood_new"] = report_block(
            "ood_new (PRIMARY overfit check)",
            ood,
            args.locked_thr,
            ood_gate_tp,
            ood_gate_fp,
        )

    if args.out_json:
        args.out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
