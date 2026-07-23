#!/usr/bin/env python3
"""TimeSformer window-MIL + real-FP vs fake-TP contrastive margin training.

Loss = bag_BCE + w_real * BCE(real_windows->0) + w_ctr * mean(relu(margin - (logit_fake_tp - logit_real_fp)))

Reuses cached window embeddings. Mines hard pairs each epoch from current model scores.

Example:
  python3 forgery/scripts/train/train_timesformer_forgery_contrastive_mil.py \\
    --feature-cache ~/forenShield-ai/forgery/results/train/timesformer-forgery-v1.6-rank-*/timesformer_window_bags_*_w1.0.npz \\
    --init-head ~/forenShield-ai/forgery/models/train/temporal/timesformer-forgery/timesformer-forgery-v1.4-temporal-ft-20260704-0837/forgery_head.pt \\
    --run-id timesformer-forgery-v1.7-contrastive \\
    --epochs 25 --contrastive-weight 0.5 --contrastive-margin 0.35
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

MVTB_MIDDLE_RE = re.compile(r"_middle_tampered_", re.I)


class PaddedWindowBagDataset(Dataset):
    def __init__(self, X: np.ndarray, mask: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.mask = torch.from_numpy(mask.astype(np.bool_))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.mask[idx], self.y[idx]


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


def metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (probs >= threshold).astype(int)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    out = {"threshold": threshold, "accuracy": round(acc, 4), "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    try:
        from sklearn.metrics import roc_auc_score

        if len(set(y_true.tolist())) > 1:
            out["roc_auc"] = round(float(roc_auc_score(y_true, probs)), 4)
    except Exception:
        pass
    return out


def pick_fake_tp_indices(
    rel_path: str,
    valid_idx: np.ndarray,
    scores: np.ndarray,
    *,
    top_k: int,
) -> list[int]:
    rel_l = rel_path.replace("\\", "/").lower()
    picks: list[int] = []
    if MVTB_MIDDLE_RE.search(rel_path) or MVTB_MIDDLE_RE.search(rel_l):
        picks.append(int(valid_idx[len(valid_idx) // 2]))
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


def build_contrastive_pairs(
    scores: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    items: list[dict],
    train_idx: list[int],
    *,
    top_k_real_fp: int,
    top_k_fake_tp: int,
    pairs_per_fake: int,
    seed: int,
) -> np.ndarray:
    rng = random.Random(seed)
    real_fp: list[tuple[int, int]] = []
    fake_tp: list[tuple[int, int]] = []

    for i in train_idx:
        valid = np.where(mask[i])[0]
        if len(valid) == 0:
            continue
        rel = items[i].get("relative_path", "")
        if int(y[i]) == 0:
            order = np.argsort(scores[i, valid])[-top_k_real_fp:]
            for j in order:
                real_fp.append((int(i), int(valid[j])))
        else:
            for wi in pick_fake_tp_indices(rel, valid, scores[i], top_k=top_k_fake_tp):
                fake_tp.append((int(i), int(wi)))

    pairs: list[list[int]] = []
    for fc, fw in fake_tp:
        for _ in range(pairs_per_fake):
            rc, rw = rng.choice(real_fp)
            if rc == fc:
                continue
            pairs.append([rc, rw, fc, fw])
    if not pairs:
        return np.zeros((0, 4), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Contrastive window-MIL forgery training")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("FORENSHIELD_REPO", "~/forenShield-ai")))
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--init-head", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, default=None, help="Optional pre-mined contrastive_pairs.npz")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--aggregate", choices=("max", "topk"), default="topk")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--real-window-weight", type=float, default=0.2)
    parser.add_argument("--contrastive-weight", type=float, default=0.5)
    parser.add_argument("--contrastive-margin", type=float, default=0.35)
    parser.add_argument("--contrastive-pairs-per-step", type=int, default=64)
    parser.add_argument("--remine-pairs-every", type=int, default=3, help="0=only epoch1")
    parser.add_argument("--top-k-real-fp", type=int, default=2)
    parser.add_argument("--top-k-fake-tp", type=int, default=2)
    parser.add_argument("--pairs-per-fake", type=int, default=3)
    args = parser.parse_args()

    repo_root = Path(args.root).expanduser().resolve()
    bootstrap_imports(repo_root)
    from timesformer_forgery_features import (  # noqa: E402
        WindowMilMLP,
        aggregate_bag_logits,
        aggregate_bag_probs,
    )

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
    embed_dim = int(X.shape[2])
    aggregate = args.aggregate
    top_k = args.top_k

    forgery_root = repo_root / "forgery" if repo_root.name != "forgery" else repo_root
    run_id = args.run_id or f"timesformer-forgery-v1.7-contrastive-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        forgery_root / "models/train/temporal/timesformer-forgery" / run_id
    )
    work_dir = forgery_root / "results/train" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    tr_idx, va_idx = split_train_val(len(y), args.val_ratio, args.seed)
    X_train, m_train, y_train = X[tr_idx], mask[tr_idx], y[tr_idx]
    X_val, m_val, y_val = X[va_idx], mask[va_idx], y[va_idx]

    flat_tr = X_train[m_train]
    mean = flat_tr.mean(axis=0)
    std = flat_tr.std(axis=0)
    std[std < 1e-6] = 1.0
    Xn = (X - mean) / std
    X_train_n = Xn[tr_idx]
    X_val_n = Xn[va_idx]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = WindowMilMLP(embed_dim).to(device)
    init_path = Path(args.init_head).expanduser().resolve()
    init_ckpt = torch.load(init_path, map_location=device, weights_only=False)
    model.load_state_dict(init_ckpt["state_dict"])
    init_backbone_sd = init_ckpt.get("backbone_state_dict")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bag_loss_fn = nn.BCEWithLogitsLoss()
    real_win_loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        PaddedWindowBagDataset(X_train_n, m_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )

    pair_idx = np.zeros((0, 4), dtype=np.int64)
    if args.pair_cache:
        pc = np.load(Path(args.pair_cache).expanduser().resolve(), allow_pickle=True)
        pair_idx = pc["pair_idx"]

    def window_scores_numpy(xb: np.ndarray, mb: np.ndarray) -> np.ndarray:
        model.eval()
        n, w, d = xb.shape
        out = np.full((n, w), -1e9, dtype=np.float32)
        with torch.no_grad():
            for i in range(n):
                valid = np.where(mb[i])[0]
                if len(valid) == 0:
                    continue
                x = torch.from_numpy(xb[i, valid].astype(np.float32)).to(device)
                logits = model(x).detach().cpu().numpy()
                out[i, valid] = logits
        return out

    def eval_split(xb: np.ndarray, mb: np.ndarray, yb: np.ndarray) -> tuple[np.ndarray, dict]:
        ds = PaddedWindowBagDataset(xb, mb, yb)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        probs_all: list[float] = []
        model.eval()
        with torch.no_grad():
            for x_t, m_t, _ in loader:
                x_t = x_t.to(device)
                m_t = m_t.to(device)
                b, w, d = x_t.shape
                logits = model(x_t.view(b * w, d)).view(b, w)
                logits = logits.masked_fill(~m_t, -1e4)
                bag_probs = aggregate_bag_probs(
                    torch.sigmoid(logits), m_t, aggregate=aggregate, top_k=top_k
                )
                probs_all.extend(bag_probs.cpu().numpy().tolist())
        probs_arr = np.asarray(probs_all, dtype=np.float64)
        return probs_arr, metrics_from_probs(yb, probs_arr, threshold=0.5)

    def remine_pairs(epoch: int) -> np.ndarray:
        scores = np.full((len(y), Xn.shape[1]), -1e9, dtype=np.float32)
        scores[tr_idx] = window_scores_numpy(X_train_n, m_train)
        pairs = build_contrastive_pairs(
            scores,
            mask,
            y,
            items,
            tr_idx,
            top_k_real_fp=args.top_k_real_fp,
            top_k_fake_tp=args.top_k_fake_tp,
            pairs_per_fake=args.pairs_per_fake,
            seed=args.seed + epoch,
        )
        print(f"  remined contrastive pairs: {len(pairs)}", flush=True)
        return pairs

    best_val_auc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        if epoch == 1 or (args.remine_pairs_every > 0 and epoch % args.remine_pairs_every == 1):
            if not args.pair_cache or epoch > 1:
                pair_idx = remine_pairs(epoch)

        model.train()
        total_loss = 0.0
        total_bag = 0.0
        total_real = 0.0
        total_ctr = 0.0
        n_seen = 0

        for x_t, m_t, y_t in train_loader:
            x_t = x_t.to(device)
            m_t = m_t.to(device)
            y_t = y_t.to(device)
            b, w, d = x_t.shape
            logits = model(x_t.view(b * w, d)).view(b, w)
            logits = logits.masked_fill(~m_t, -1e4)

            bag_logits = aggregate_bag_logits(logits, m_t, aggregate=aggregate, top_k=top_k)
            loss_bag = bag_loss_fn(bag_logits, y_t)

            real_mask = m_t & (y_t.unsqueeze(1) < 0.5)
            if bool(real_mask.any()) and args.real_window_weight > 0:
                loss_real = real_win_loss_fn(logits[real_mask], torch.zeros_like(logits[real_mask]))
            else:
                loss_real = torch.zeros((), device=device)

            loss_ctr = torch.zeros((), device=device)
            if len(pair_idx) > 0 and args.contrastive_weight > 0:
                n_pairs = min(args.contrastive_pairs_per_step, len(pair_idx))
                sel = np.random.choice(len(pair_idx), size=n_pairs, replace=False)
                batch_pairs = pair_idx[sel]
                real_logits: list[torch.Tensor] = []
                fake_logits: list[torch.Tensor] = []
                for rc, rw, fc, fw in batch_pairs:
                    # map global clip idx -> local batch if present, else direct from Xn
                    real_logits.append(
                        model(torch.from_numpy(Xn[rc, rw].astype(np.float32)).unsqueeze(0).to(device)).squeeze()
                    )
                    fake_logits.append(
                        model(torch.from_numpy(Xn[fc, fw].astype(np.float32)).unsqueeze(0).to(device)).squeeze()
                    )
                real_t = torch.stack(real_logits)
                fake_t = torch.stack(fake_logits)
                loss_ctr = torch.relu(args.contrastive_margin - (fake_t - real_t)).mean()

            loss = (
                loss_bag
                + args.real_window_weight * loss_real
                + args.contrastive_weight * loss_ctr
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * b
            total_bag += float(loss_bag.item()) * b
            total_real += float(loss_real.item()) * b
            total_ctr += float(loss_ctr.item()) * b
            n_seen += b

        val_probs, val_m = eval_split(X_val_n, m_val, y_val)
        print(
            f"epoch {epoch:02d} loss={total_loss / max(1, n_seen):.4f} "
            f"bag={total_bag / max(1, n_seen):.4f} real={total_real / max(1, n_seen):.4f} "
            f"ctr={total_ctr / max(1, n_seen):.4f} val_auc={val_m.get('roc_auc')}",
            flush=True,
        )
        if val_m.get("roc_auc", -1) > best_val_auc:
            best_val_auc = val_m.get("roc_auc", -1)
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    train_probs, train_m = eval_split(X_train_n, m_train, y_train)
    val_probs, val_m = eval_split(X_val_n, m_val, y_val)
    all_probs, fit_m = eval_split(Xn, mask, y)

    ckpt = {
        "model_type": "timesformer_forgery_contrastive_mil_mlp",
        "pretrained_id": meta.get("pretrained_id", "facebook/timesformer-base-finetuned-k400"),
        "embed_dim": embed_dim,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "state_dict": model.state_dict(),
        "init_head": str(init_path),
        "feature_cache": str(cache_path),
        "contrastive_margin": args.contrastive_margin,
        "contrastive_weight": args.contrastive_weight,
        "real_window_weight": args.real_window_weight,
        "aggregate": aggregate,
        "top_k": top_k,
        "window_sec": meta.get("window_sec", 1.0),
        "stride_sec": meta.get("stride_sec", 0.5),
        "clip_frames": meta.get("clip_frames", 8),
        "max_side": meta.get("max_side", 512),
        "train_run_id": run_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    if init_backbone_sd:
        ckpt["backbone_state_dict"] = init_backbone_sd
        ckpt["timesformer_backbone"] = "k400_partial_ft"
    else:
        ckpt["timesformer_backbone"] = "k400_frozen"

    ckpt_path = out_dir / "forgery_head.pt"
    torch.save(ckpt, ckpt_path)

    summary = {
        "run_id": run_id,
        "checkpoint": str(ckpt_path),
        "feature_cache": str(cache_path),
        "init_head": str(init_path),
        "n_pairs_last": int(len(pair_idx)),
        "train_metrics": train_m,
        "val_metrics": val_m,
        "fit_metrics": fit_m,
        "contrastive_margin": args.contrastive_margin,
        "contrastive_weight": args.contrastive_weight,
        "real_window_weight": args.real_window_weight,
    }
    (work_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nDONE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
