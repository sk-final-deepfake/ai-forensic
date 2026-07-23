#!/usr/bin/env python3
"""Mine real-FP vs fake-TP window pairs for contrastive forgery training.

Uses a trained head on cached window embeddings:
  - real_fp : real clip windows with highest tamper logits (motion/shake false alarms)
  - fake_tp : fake clip windows at tamper segment (MVTB middle / CSVTED tail) or top logits

Example:
  python3 forgery/scripts/train/mine_timesformer_forgery_contrastive_pairs.py \\
    --feature-cache ~/forenShield-ai/forgery/results/train/timesformer-forgery-v1.6-rank-*/timesformer_window_bags_*_w1.0.npz \\
    --init-head ~/forenShield-ai/forgery/models/train/temporal/timesformer-forgery/timesformer-forgery-v1.4-temporal-ft-20260704-0837/forgery_head.pt \\
    --run-id timesformer-contrastive-mine-20260707
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

MVTB_MIDDLE_RE = re.compile(r"_middle_tampered_", re.I)


def bootstrap_imports(repo_root: Path) -> None:
    forgery_infer = repo_root / "forgery" / "scripts" / "infer"
    root_infer = repo_root / "scripts" / "infer"
    train_infer = Path(__file__).resolve().parents[1] / "infer"
    for cand in (root_infer, forgery_infer, train_infer):
        if cand.is_dir() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))


def split_train_val(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    return idx[n_val:], idx[:n_val]


def window_scores(
    model: torch.nn.Module,
    X: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    n, w, d = X.shape
    scores = np.full((n, w), -1e9, dtype=np.float32)
    with torch.no_grad():
        for i in range(n):
            valid = np.where(mask[i])[0]
            if len(valid) == 0:
                continue
            embs = X[i, valid]
            embs = (embs - mean) / std
            x = torch.from_numpy(embs.astype(np.float32)).to(device)
            for start in range(0, len(x), batch_size):
                chunk = x[start : start + batch_size]
                logits = model(chunk).detach().cpu().numpy()
                scores[i, valid[start : start + len(chunk)]] = logits
    return scores


def pick_fake_tp_window(
    rel_path: str,
    valid_idx: np.ndarray,
    scores: np.ndarray,
    *,
    top_k: int,
) -> list[int]:
    """Prefer tamper-prior window index; fallback to top logits on fake."""
    rel_l = rel_path.replace("\\", "/").lower()
    picks: list[int] = []

    if MVTB_MIDDLE_RE.search(rel_path) or MVTB_MIDDLE_RE.search(rel_l):
        mid_local = len(valid_idx) // 2
        picks.append(int(valid_idx[mid_local]))

    if "eop-" in rel_l or "/eop-frame-" in rel_l:
        picks.append(int(valid_idx[-1]))

    if not picks:
        order = np.argsort(scores[valid_idx])[-top_k:]
        picks.extend(int(valid_idx[j]) for j in order)
    else:
        order = np.argsort(scores[valid_idx])[-top_k:]
        for j in order:
            wi = int(valid_idx[j])
            if wi not in picks:
                picks.append(wi)
            if len(picks) >= top_k:
                break
    return picks[:top_k]


def mine_pairs(
    scores: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    items: list[dict],
    *,
    train_idx: list[int],
    top_k_real_fp: int,
    top_k_fake_tp: int,
    pairs_per_fake: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    real_fp_rows: list[dict] = []
    fake_tp_rows: list[dict] = []

    train_set = set(train_idx)
    for i in train_idx:
        valid = np.where(mask[i])[0]
        if len(valid) == 0:
            continue
        rel = items[i].get("relative_path", f"clip_{i}")
        if int(y[i]) == 0:
            order = np.argsort(scores[i, valid])[-top_k_real_fp:]
            for j in order:
                wi = int(valid[j])
                real_fp_rows.append(
                    {
                        "clip_idx": int(i),
                        "window_idx": wi,
                        "score": round(float(scores[i, wi]), 6),
                        "relative_path": rel,
                        "label": "real_fp",
                    }
                )
        else:
            for wi in pick_fake_tp_window(rel, valid, scores[i], top_k=top_k_fake_tp):
                fake_tp_rows.append(
                    {
                        "clip_idx": int(i),
                        "window_idx": wi,
                        "score": round(float(scores[i, wi]), 6),
                        "relative_path": rel,
                        "label": "fake_tp",
                    }
                )

    if not real_fp_rows or not fake_tp_rows:
        return real_fp_rows, fake_tp_rows, []

    pair_rows: list[dict] = []
    for fake_row in fake_tp_rows:
        for _ in range(pairs_per_fake):
            real_row = rng.choice(real_fp_rows)
            if real_row["clip_idx"] == fake_row["clip_idx"]:
                continue
            margin_gap = float(fake_row["score"] - real_row["score"])
            pair_rows.append(
                {
                    "real_clip_idx": real_row["clip_idx"],
                    "real_window_idx": real_row["window_idx"],
                    "fake_clip_idx": fake_row["clip_idx"],
                    "fake_window_idx": fake_row["window_idx"],
                    "real_score": real_row["score"],
                    "fake_score": fake_row["score"],
                    "score_gap": round(margin_gap, 6),
                    "real_path": real_row["relative_path"],
                    "fake_path": fake_row["relative_path"],
                }
            )
    return real_fp_rows, fake_tp_rows, pair_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine real-FP vs fake-TP contrastive window pairs")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("FORENSHIELD_REPO", "~/forenShield-ai")))
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--init-head", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--top-k-real-fp", type=int, default=2)
    parser.add_argument("--top-k-fake-tp", type=int, default=2)
    parser.add_argument("--pairs-per-fake", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(args.root).expanduser().resolve()
    bootstrap_imports(repo_root)
    from timesformer_forgery_features import WindowMilMLP  # noqa: E402

    cache_path = Path(args.feature_cache).expanduser().resolve()
    if not cache_path.is_file():
        print(f"feature cache not found: {cache_path}", file=sys.stderr)
        return 1

    cached = np.load(cache_path, allow_pickle=True)
    X = cached["X"]
    mask = cached["mask"]
    y = cached["y"]
    meta = json.loads(str(cached["meta"]))
    items = meta.get("items", [])

    forgery_root = repo_root / "forgery" if repo_root.name != "forgery" else repo_root
    run_id = args.run_id or f"timesformer-contrastive-mine-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    out_dir = forgery_root / "results/train" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_idx, va_idx = split_train_val(len(y), args.val_ratio, args.seed)

    init_path = Path(args.init_head).expanduser().resolve()
    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    mean = np.array(ckpt["mean"], dtype=np.float32)
    std = np.array(ckpt["std"], dtype=np.float32)
    embed_dim = int(ckpt.get("embed_dim", X.shape[2]))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = WindowMilMLP(embed_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])

    scores = window_scores(model, X, mask, mean, std, device)
    real_fp, fake_tp, pairs = mine_pairs(
        scores,
        mask,
        y,
        items,
        train_idx=tr_idx,
        top_k_real_fp=args.top_k_real_fp,
        top_k_fake_tp=args.top_k_fake_tp,
        pairs_per_fake=args.pairs_per_fake,
        seed=args.seed,
    )

    gaps = [p["score_gap"] for p in pairs]
    summary = {
        "run_id": run_id,
        "feature_cache": str(cache_path),
        "init_head": str(init_path),
        "n_clips": int(len(y)),
        "n_train": len(tr_idx),
        "n_real_fp_windows": len(real_fp),
        "n_fake_tp_windows": len(fake_tp),
        "n_pairs": len(pairs),
        "score_gap_mean": round(float(np.mean(gaps)), 6) if gaps else None,
        "score_gap_median": round(float(np.median(gaps)), 6) if gaps else None,
        "pct_pairs_fake_higher": round(float(np.mean([g > 0 for g in gaps])), 4) if gaps else None,
        "top_k_real_fp": args.top_k_real_fp,
        "top_k_fake_tp": args.top_k_fake_tp,
        "pairs_per_fake": args.pairs_per_fake,
        "sample_real_fp": sorted(real_fp, key=lambda r: r["score"], reverse=True)[:15],
        "sample_fake_tp": sorted(fake_tp, key=lambda r: r["score"], reverse=True)[:15],
        "sample_pairs_low_gap": sorted(pairs, key=lambda r: r["score_gap"])[:15],
    }

    pair_idx = np.array(
        [[p["real_clip_idx"], p["real_window_idx"], p["fake_clip_idx"], p["fake_window_idx"]] for p in pairs],
        dtype=np.int32,
    )
    np.savez(
        out_dir / "contrastive_pairs.npz",
        pair_idx=pair_idx,
        train_idx=np.array(tr_idx, dtype=np.int32),
        val_idx=np.array(va_idx, dtype=np.int32),
    )
    (out_dir / "contrastive_pairs.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    (out_dir / "mine_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nDONE pairs={len(pairs)} -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
