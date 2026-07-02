#!/usr/bin/env python3
"""Validate calibrated threshold without re-infer (predictions.json only).

Checks:
  1) mvtb split-half: tune thr on 50% → evaluate on held-out 50%
  2) fixed thr transfer: apply mvtb thr on csvted (or another predictions file)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_items(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def to_arrays(items: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels: list[int] = []
    scores: list[float] = []
    ids: list[str] = []
    for x in items:
        g = x.get("ground_truth_label")
        y = 1 if g in ("fake", 1, "1", True) else 0
        labels.append(y)
        scores.append(float(x["tamper_score"]))
        ids.append(str(x.get("relative_path") or x.get("video_rel") or x.get("path") or ""))
    return np.array(labels, dtype=np.int64), np.array(scores, dtype=np.float64), ids


def confusion(labels: np.ndarray, scores: np.ndarray, thr: float) -> dict[str, int]:
    pred = (scores >= thr).astype(np.int64)
    tp = int(((labels == 1) & (pred == 1)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    tn = int(((labels == 0) & (pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def acc_from_conf(c: dict[str, int]) -> float:
    t = c["tp"] + c["tn"] + c["fp"] + c["fn"]
    return (c["tp"] + c["tn"]) / max(1, t)


def find_best_gate_thr(
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
        c = confusion(labels, scores, thr_f)
        if c["tp"] < min_tp or c["fp"] > max_fp:
            continue
        rank = (acc_from_conf(c), c["tp"], -c["fp"])
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_thr = thr_f
            best_conf = c
    if best_conf is None:
        return None
    return best_thr, best_conf


def stratified_split(items: list[dict], seed: int, frac: float = 0.5) -> tuple[list[dict], list[dict]]:
    rng = np.random.RandomState(seed)
    fake = [x for x in items if x.get("ground_truth_label") == "fake"]
    real = [x for x in items if x.get("ground_truth_label") != "fake"]
    rng.shuffle(fake)
    rng.shuffle(real)
    n_fake = int(round(len(fake) * frac))
    n_real = int(round(len(real) * frac))
    train = fake[:n_fake] + real[:n_real]
    test = fake[n_fake:] + real[n_real:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def scale_gate(n: int, min_tp_full: int, max_fp_full: int, n_full: int = 200) -> tuple[int, int]:
    """Scale TP/FP gate from full 200-video benchmark to subset size n."""
    fake_n = n // 2
    real_n = n - fake_n
    min_tp = max(1, int(round(min_tp_full * fake_n / (n_full / 2))))
    max_fp = max(0, int(round(max_fp_full * real_n / (n_full / 2))))
    return min_tp, max_fp


def print_conf(name: str, thr: float, c: dict[str, int]) -> None:
    print(
        f"{name} @thr={thr:.4f}  TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']}  "
        f"Acc={acc_from_conf(c):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate spatial benchmark calibration")
    parser.add_argument("--mvtb-predictions", type=Path, required=True)
    parser.add_argument("--csvted-predictions", type=Path, default=None)
    parser.add_argument("--fixed-thr", type=float, default=None, help="e.g. 0.185 from full mvtb tune")
    parser.add_argument("--min-tp", type=int, default=63)
    parser.add_argument("--max-fp", type=int, default=51)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-frac", type=float, default=0.5)
    args = parser.parse_args()

    items = load_items(args.mvtb_predictions)
    train_items, test_items = stratified_split(items, args.seed, args.split_frac)
    y_tr, s_tr, _ = to_arrays(train_items)
    y_te, s_te, _ = to_arrays(test_items)

    n_tr, n_te = len(train_items), len(test_items)
    min_tp_tr, max_fp_tr = scale_gate(n_tr, args.min_tp, args.max_fp)
    min_tp_te, max_fp_te = scale_gate(n_te, args.min_tp, args.max_fp)

    print("=== 1) mvtb split-half (stratified, seed=%d, frac=%.2f) ===" % (args.seed, args.split_frac))
    print(f"train n={n_tr}  scaled gate TP>={min_tp_tr} FP<={max_fp_tr}")
    print(f"test  n={n_te}  scaled gate TP>={min_tp_te} FP<={max_fp_te}")

    found = find_best_gate_thr(
        y_tr, s_tr, min_tp=min_tp_tr, max_fp=max_fp_tr, step=args.step
    )
    if found is None:
        print("train: no gate-satisfying threshold")
        thr_tune = args.fixed_thr if args.fixed_thr is not None else 0.5
        print(f"fallback thr={thr_tune}")
    else:
        thr_tune, c_tr = found
        print_conf("train (tune)", thr_tune, c_tr)

    c_te = confusion(y_te, s_te, thr_tune)
    print_conf("test  (hold-out)", thr_tune, c_te)
    gate_ok = c_te["tp"] >= min_tp_te and c_te["fp"] <= max_fp_te
    print(f"test gate pass: {gate_ok}")

    if args.fixed_thr is not None:
        print("")
        print("=== 2) fixed thr on mvtb full (sanity) ===")
        y_all, s_all, _ = to_arrays(items)
        print_conf("mvtb full", args.fixed_thr, confusion(y_all, s_all, args.fixed_thr))

    if args.csvted_predictions is not None:
        print("")
        print("=== 3) fixed thr transfer → csvted (no re-tune) ===")
        c_items = load_items(args.csvted_predictions)
        y_c, s_c, _ = to_arrays(c_items)
        thr_xfer = thr_tune if found is not None else (args.fixed_thr or 0.5)
        print_conf("csvted", thr_xfer, confusion(y_c, s_c, thr_xfer))
        print("(baseline csvted @0.5 ref: TP24 FP11 on full 200)")


if __name__ == "__main__":
    main()
