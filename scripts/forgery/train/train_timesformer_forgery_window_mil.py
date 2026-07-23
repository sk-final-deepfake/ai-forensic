#!/usr/bin/env python3
"""Train TimeSformer forgery window-MIL head (frozen K400 backbone).

Each video is a bag of 1s sliding-window K400 embeddings. MLP scores every window;
clip score = top-k mean of window probs (same at train and infer).

Example (GPU):
  python3 forgery/scripts/train/train_timesformer_forgery_window_mil.py \\
    --root ~/forenShield-ai \\
    --data-root ~/forenShield-ai/forgery/data/train/video/forgery-gmflow-train-temporal \\
    --run-id timesformer-forgery-v1-temporal \\
    --scan-mode window_1s --window-label-mode segment \\
    --aggregate topk --top-k 3 --epochs 40
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import TimesformerModel


class PaddedWindowBagDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        y: np.ndarray,
        y_window: np.ndarray | None = None,
        window_label_mask: np.ndarray | None = None,
    ) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.mask = torch.from_numpy(mask.astype(np.bool_))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.use_window_labels = y_window is not None and window_label_mask is not None
        if self.use_window_labels:
            self.y_window = torch.from_numpy(y_window.astype(np.float32))
            self.window_label_mask = torch.from_numpy(window_label_mask.astype(np.bool_))
        else:
            self.y_window = None
            self.window_label_mask = None

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        if self.use_window_labels:
            return (
                self.X[idx],
                self.mask[idx],
                self.y[idx],
                self.y_window[idx],
                self.window_label_mask[idx],
            )
        return self.X[idx], self.mask[idx], self.y[idx]


def split_train_val(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    return idx[n_val:], idx[:n_val]


def weighted_window_bce(
    window_probs: torch.Tensor,
    window_targets: torch.Tensor,
    *,
    pos_weight: float,
) -> torch.Tensor:
    if pos_weight <= 1.0:
        return nn.functional.binary_cross_entropy(window_probs, window_targets)
    bce = nn.functional.binary_cross_entropy(window_probs, window_targets, reduction="none")
    w = torch.where(
        window_targets > 0.5,
        torch.full_like(window_targets, pos_weight),
        torch.ones_like(window_targets),
    )
    return (bce * w).mean()


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


def bootstrap_imports(repo_root: Path) -> Path:
    """Prefer forgery/scripts/infer over stale repo scripts/infer copies."""
    repo_root = repo_root.expanduser().resolve()
    train_infer = Path(__file__).resolve().parents[1] / "infer"
    forgery_infer = repo_root / "forgery" / "scripts" / "infer"
    root_infer = repo_root / "scripts" / "infer"
    primary = train_infer if train_infer.is_dir() else forgery_infer
    for cand in (train_infer, forgery_infer, root_infer):
        if cand.is_dir() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
    for mod in ("tamper_segment_labels", "timesformer_forgery_features", "spatial_mvtamperbench_benchmark"):
        if mod in sys.modules:
            del sys.modules[mod]
    print(f"infer_scripts: {primary}", flush=True)
    return primary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train TimeSformer forgery window-MIL head")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("FORENSHIELD_REPO", "~/forenShield-ai")))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--pretrained-id", default="facebook/timesformer-base-finetuned-k400")
    parser.add_argument("--scan-mode", choices=("window_1s",), default="window_1s")
    parser.add_argument(
        "--window-label-mode",
        choices=("clip", "segment", "real_windows"),
        default="segment",
        help="clip=bag BCE only; segment=bag+window BCE on tamper spans; "
        "real_windows=bag for fake + window BCE(0) on real (suppress shake/noise)",
    )
    parser.add_argument("--window-loss-weight", type=float, default=None)
    parser.add_argument("--window-pos-weight", type=float, default=8.0)
    parser.add_argument("--tamper-duration-sec", type=float, default=1.0)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--stride-sec", type=float, default=0.5)
    parser.add_argument("--clip-frames", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--aggregate", choices=("max", "topk"), default="topk")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("~/forenShield-ai/forgery/models/train/temporal/timesformer-forgery/v1-temporal"),
    )
    parser.add_argument("--cache-features", type=Path, default=None)
    parser.add_argument(
        "--decode-cache",
        type=Path,
        default=Path("~/forenShield-ai/forgery/cache/decode-mp4-temporal"),
    )
    parser.add_argument(
        "--init-head",
        type=Path,
        default=None,
        help="Init MIL head (+ optional FT backbone) from prior forgery_head.pt (e.g. v1.4 FT)",
    )
    args = parser.parse_args()

    if args.window_loss_weight is None:
        if args.window_label_mode == "clip":
            args.window_loss_weight = 0.0
        else:
            args.window_loss_weight = 0.25

    repo_root = Path(args.root).expanduser().resolve()
    infer_scripts = bootstrap_imports(repo_root)
    from spatial_mvtamperbench_benchmark import collect_videos  # noqa: E402
    from timesformer_forgery_features import (  # noqa: E402
        WindowMilMLP,
        aggregate_bag_logits,
        aggregate_bag_probs,
        extract_video_window_embeddings,
    )
    from tamper_segment_labels import window_labels_for_video  # noqa: E402

    clip_frames = max(2, int(args.clip_frames))
    data_root = Path(args.data_root).expanduser().resolve()
    run_id = args.run_id or (
        f"timesformer-forgery-train-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    forgery_root = repo_root / "forgery" if repo_root.name != "forgery" else repo_root
    work_dir = forgery_root / "results/train" / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    decode_cache = Path(args.decode_cache).expanduser().resolve()
    decode_cache.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(data_root)
    if not videos:
        print(f"no videos under {data_root}", file=sys.stderr)
        return 1

    embed_dim = 768
    print(
        f"videos: {len(videos)} real={sum(1 for v in videos if v['ground_truth_label']=='real')} "
        f"fake={sum(1 for v in videos if v['ground_truth_label']=='fake')} "
        f"pretrained={args.pretrained_id} clip_frames={clip_frames} "
        f"scan={args.scan_mode} window_labels={args.window_label_mode} "
        f"aggregate={args.aggregate} top_k={args.top_k} decode_cache={decode_cache}",
        flush=True,
    )

    cache_suffix = args.window_label_mode
    if args.init_head:
        cache_suffix = f"{cache_suffix}_init{Path(args.init_head).stem}"
    cache_suffix = f"{cache_suffix}_w{args.window_sec}"
    cache_path = (
        Path(args.cache_features).expanduser().resolve()
        if args.cache_features
        else work_dir / f"timesformer_window_bags_{cache_suffix}.npz"
    )

    if cache_path.is_file():
        print(f"load cached window bags: {cache_path}", flush=True)
        cached = np.load(cache_path, allow_pickle=True)
        X = cached["X"]
        mask = cached["mask"]
        y = cached["y"]
        meta = json.loads(str(cached["meta"]))
        max_windows = int(cached["max_windows"])
        y_window = cached["y_window"] if "y_window" in cached else None
        window_label_mask = cached["window_label_mask"] if "window_label_mask" in cached else None
        embed_dim = int(X.shape[2])
        if args.window_label_mode == "segment" and y_window is None:
            print(f"segment mode requires y_window in cache; delete: {cache_path}", file=sys.stderr)
            return 1
        if args.window_label_mode == "real_windows" and y_window is None:
            print(f"real_windows mode requires y_window in cache; delete: {cache_path}", file=sys.stderr)
            return 1
    else:
        init_ckpt: dict | None = None
        if args.init_head:
            init_path = Path(args.init_head).expanduser().resolve()
            if not init_path.is_file():
                print(f"init-head not found: {init_path}", file=sys.stderr)
                return 1
            init_ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
            print(f"init-head: {init_path} backbone_ft={bool(init_ckpt.get('backbone_state_dict'))}", flush=True)

        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"loading backbone: {args.pretrained_id}", flush=True)
        backbone = TimesformerModel.from_pretrained(args.pretrained_id).to(device)
        if init_ckpt and init_ckpt.get("backbone_state_dict"):
            missing, unexpected = backbone.load_state_dict(init_ckpt["backbone_state_dict"], strict=False)
            print(
                f"loaded FT backbone: missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
        backbone.eval()

        bag_rows: list[np.ndarray] = []
        window_target_rows: list[list[float]] = []
        window_mask_rows: list[list[bool]] = []
        y_rows: list[int] = []
        meta_rows: list[dict] = []
        max_windows = 0

        for i, row in enumerate(videos, start=1):
            per_window, extract_meta = extract_video_window_embeddings(
                Path(row["source_path"]),
                backbone,
                device,
                window_sec=args.window_sec,
                stride_sec=args.stride_sec,
                clip_frames=clip_frames,
                max_side=args.max_side,
                decode_cache=decode_cache,
            )
            label = 1 if row["ground_truth_label"] == "fake" else 0
            print(
                f"[{i:03d}/{len(videos)}] {row['relative_path']} windows={len(per_window)}",
                flush=True,
            )
            if not per_window:
                continue
            feats = np.stack([w["embedding"] for w in per_window])
            embed_dim = int(feats.shape[1])
            bag_rows.append(feats)
            max_windows = max(max_windows, feats.shape[0])
            y_rows.append(label)

            item_meta: dict = {
                "relative_path": row["relative_path"],
                "label": row["ground_truth_label"],
                "n_windows": int(feats.shape[0]),
            }
            if args.window_label_mode in ("segment", "real_windows"):
                total_frames = int(extract_meta.get("total_frames") or 0)
                fps = float(extract_meta.get("fps") or 30.0)
                window_targets, window_masks, wl_meta = window_labels_for_video(
                    per_window,
                    ground_truth_label=row["ground_truth_label"],
                    total_frames=total_frames,
                    fps=fps,
                    video_path=row["source_path"],
                    relative_path=row["relative_path"],
                    duration_sec=args.tamper_duration_sec,
                    label_mode=args.window_label_mode,
                )
                window_target_rows.append(window_targets)
                window_mask_rows.append(window_masks)
                item_meta.update(wl_meta)
                print(
                    f"    segment labeled={wl_meta.get('n_labeled_windows')} "
                    f"pos={wl_meta.get('n_positive_windows')} "
                    f"seg={wl_meta.get('segment')}",
                    flush=True,
                )
            meta_rows.append(item_meta)

        if not bag_rows:
            print("no window embeddings extracted", file=sys.stderr)
            return 1

        X = np.zeros((len(bag_rows), max_windows, embed_dim), dtype=np.float32)
        mask = np.zeros((len(bag_rows), max_windows), dtype=np.bool_)
        y_window = np.zeros((len(bag_rows), max_windows), dtype=np.float32)
        window_label_mask = np.zeros((len(bag_rows), max_windows), dtype=np.bool_)
        for i, feats in enumerate(bag_rows):
            n = feats.shape[0]
            X[i, :n] = feats
            mask[i, :n] = True
            if args.window_label_mode in ("segment", "real_windows") and i < len(window_target_rows):
                y_window[i, :n] = np.asarray(window_target_rows[i][:n], dtype=np.float32)
                window_label_mask[i, :n] = np.asarray(window_mask_rows[i][:n], dtype=np.bool_)

        y = np.array(y_rows, dtype=np.int64)
        meta = {
            "pretrained_id": args.pretrained_id,
            "embed_dim": embed_dim,
            "items": meta_rows,
            "clip_frames": clip_frames,
            "max_side": args.max_side,
            "scan_mode": args.scan_mode,
            "window_label_mode": args.window_label_mode,
            "window_loss_weight": args.window_loss_weight,
            "window_pos_weight": args.window_pos_weight,
            "tamper_duration_sec": args.tamper_duration_sec,
            "window_sec": args.window_sec,
            "stride_sec": args.stride_sec,
            "aggregate": args.aggregate,
            "top_k": args.top_k,
            "decode_cache": str(decode_cache),
        }
        np.savez(
            cache_path,
            X=X,
            mask=mask,
            y=y,
            y_window=y_window,
            window_label_mask=window_label_mask,
            max_windows=max_windows,
            meta=json.dumps(meta),
        )
        print(f"cached -> {cache_path} (max_windows={max_windows})", flush=True)

    tr_idx, va_idx = split_train_val(len(y), args.val_ratio, args.seed)
    X_train, m_train, y_train = X[tr_idx], mask[tr_idx], y[tr_idx]
    X_val, m_val, y_val = X[va_idx], mask[va_idx], y[va_idx]

    flat_tr = X_train[m_train]
    mean = flat_tr.mean(axis=0)
    std = flat_tr.std(axis=0)
    std[std < 1e-6] = 1.0
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_all = (X - mean) / std

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = WindowMilMLP(embed_dim).to(device)
    if args.init_head:
        init_path = Path(args.init_head).expanduser().resolve()
        init_sd = torch.load(init_path, map_location=device, weights_only=False)
        model.load_state_dict(init_sd["state_dict"])
        print(f"initialized MIL head from {init_path}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bag_loss_fn = nn.BCEWithLogitsLoss()
    window_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.window_pos_weight, device=device))

    use_window_labels = args.window_label_mode in ("segment", "real_windows") and y_window is not None
    y_window_train = y_window[tr_idx] if use_window_labels and y_window is not None else None
    window_mask_train = window_label_mask[tr_idx] if use_window_labels and window_label_mask is not None else None

    train_loader = DataLoader(
        PaddedWindowBagDataset(X_train, m_train, y_train, y_window_train, window_mask_train),
        batch_size=args.batch_size,
        shuffle=True,
    )

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
                window_probs = torch.sigmoid(logits)
                bag_probs = aggregate_bag_probs(
                    window_probs, m_t, aggregate=args.aggregate, top_k=args.top_k
                )
                probs_all.extend(bag_probs.cpu().numpy().tolist())
        probs_arr = np.asarray(probs_all, dtype=np.float64)
        return probs_arr, metrics_from_probs(yb, probs_arr, threshold=0.5)

    best_val_auc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            if use_window_labels:
                x_t, m_t, y_t, yw_t, wm_t = batch
                yw_t = yw_t.to(device)
                wm_t = wm_t.to(device)
            else:
                x_t, m_t, y_t = batch
            x_t = x_t.to(device)
            m_t = m_t.to(device)
            y_t = y_t.to(device)
            b, w, d = x_t.shape
            logits = model(x_t.view(b * w, d)).view(b, w)
            logits = logits.masked_fill(~m_t, -1e4)
            bag_logits = aggregate_bag_logits(
                logits, m_t, aggregate=args.aggregate, top_k=args.top_k
            )
            loss_bag = bag_loss_fn(bag_logits, y_t)
            if use_window_labels and args.window_loss_weight > 0:
                valid = m_t & wm_t
                if bool(valid.any()):
                    loss_win = window_loss_fn(logits[valid], yw_t[valid])
                    loss = loss_bag + args.window_loss_weight * loss_win
                else:
                    loss = loss_bag
            else:
                loss = loss_bag
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * b
            n_seen += b

        val_probs, val_m = eval_split(X_val, m_val, y_val)
        print(
            f"epoch {epoch:02d} loss={total_loss / max(1, n_seen):.4f} "
            f"val_acc={val_m['accuracy']:.3f} val_auc={val_m.get('roc_auc')}",
            flush=True,
        )
        if val_m.get("roc_auc", -1) > best_val_auc:
            best_val_auc = val_m.get("roc_auc", -1)
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    init_backbone_sd = None
    if args.init_head:
        init_sd = torch.load(
            Path(args.init_head).expanduser().resolve(), map_location="cpu", weights_only=False
        )
        init_backbone_sd = init_sd.get("backbone_state_dict")

    train_probs, train_m = eval_split(X_train, m_train, y_train)
    val_probs, val_m = eval_split(X_val, m_val, y_val)
    all_probs, fit_m = eval_split(X_all, mask, y)

    ckpt = {
        "model_type": "timesformer_forgery_window_mil_mlp",
        "pretrained_id": args.pretrained_id,
        "embed_dim": embed_dim,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "state_dict": model.state_dict(),
        "init_head": str(Path(args.init_head).expanduser().resolve()) if args.init_head else None,
        "backbone_source": "ft_from_init" if init_backbone_sd else "k400_frozen",
        "clip_frames": clip_frames,
        "max_side": args.max_side,
        "scan_mode": args.scan_mode,
        "window_label_mode": args.window_label_mode,
        "window_loss_weight": args.window_loss_weight,
        "window_pos_weight": args.window_pos_weight,
        "tamper_duration_sec": args.tamper_duration_sec,
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "aggregate": args.aggregate,
        "top_k": args.top_k,
        "decode_cache": str(decode_cache),
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
        "model_type": ckpt["model_type"],
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "checkpoint": str(ckpt_path),
        "feature_cache": str(cache_path),
        "n_samples": int(len(y)),
        "max_windows_cached": max_windows,
        "train_metrics": train_m,
        "val_metrics": val_m,
        "fit_metrics": fit_m,
        "pretrained_id": args.pretrained_id,
        "embed_dim": embed_dim,
        "clip_frames": clip_frames,
        "max_side": args.max_side,
        "scan_mode": args.scan_mode,
        "window_label_mode": args.window_label_mode,
        "window_loss_weight": args.window_loss_weight,
        "window_pos_weight": args.window_pos_weight,
        "tamper_duration_sec": args.tamper_duration_sec,
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "aggregate": args.aggregate,
        "top_k": args.top_k,
    }
    (work_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nDONE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
