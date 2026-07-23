#!/usr/bin/env python3
"""TimeSformer forgery window-MIL benchmark on MVTamperBench / CSVTED / temporal test folders.

Example (GPU):
  python3 forgery/scripts/infer/timesformer_forgery_benchmark.py \\
    --root ~/forenShield-ai \\
    --data-root ~/forenShield-ai/forgery/data/pull/evidence/mvtamperbench-200-s3 \\
    --checkpoint ~/forenShield-ai/forgery/models/train/temporal/timesformer-forgery/v1-temporal/forgery_head.pt \\
    --run-id timesformer-forgery-mvtb200-$(date -u +%Y%m%d-%H%M)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import TimesformerModel


def compute_eval(items: list[dict], threshold: float) -> dict:
    ok = [x for x in items if x.get("status") == "ok" and x.get("tamper_score") is not None]
    y_true = [1 if x["ground_truth_label"] == "fake" else 0 for x in ok]
    y_score = [float(x["tamper_score"]) for x in ok]
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    acc = (tp + tn) / len(ok) if ok else 0.0
    metrics = {
        "threshold": threshold,
        "total": len(items),
        "ok": len(ok),
        "accuracy": round(acc, 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "real": {
            "count": sum(1 for x in ok if x["ground_truth_label"] == "real"),
            "avg_tamper_score": round(
                float(np.mean([x["tamper_score"] for x in ok if x["ground_truth_label"] == "real"])), 6
            )
            if ok
            else None,
        },
        "fake": {
            "count": sum(1 for x in ok if x["ground_truth_label"] == "fake"),
            "avg_tamper_score": round(
                float(np.mean([x["tamper_score"] for x in ok if x["ground_truth_label"] == "fake"])), 6
            )
            if ok
            else None,
        },
    }
    try:
        from sklearn.metrics import roc_auc_score

        if len(set(y_true)) > 1:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
    except Exception:
        pass
    return metrics


def load_head(checkpoint_path: Path, device: torch.device):
    from timesformer_forgery_features import WindowMilMLP  # noqa: WPS433

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    embed_dim = int(ckpt.get("embed_dim", 768))
    mean = np.array(ckpt["mean"], dtype=np.float32)
    std = np.array(ckpt["std"], dtype=np.float32)
    model = WindowMilMLP(embed_dim, hidden=64).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, mean, std, ckpt


def load_backbone_and_head(checkpoint_path: Path, device: torch.device, pretrained_id: str | None):
    from timesformer_forgery_features import load_forgery_bundle  # noqa: WPS433

    ckpt_path = checkpoint_path.expanduser().resolve()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("backbone_state_dict") or ckpt.get("model_type", "").startswith("timesformer_forgery_clip"):
        backbone, head, mean, std, ckpt = load_forgery_bundle(ckpt_path, device, pretrained_id=pretrained_id)
        return backbone, head, mean, std, ckpt
    head, mean, std, ckpt = load_head(ckpt_path, device)
    pid = pretrained_id or ckpt.get("pretrained_id", "facebook/timesformer-base-finetuned-k400")
    backbone = TimesformerModel.from_pretrained(pid).to(device)
    backbone.eval()
    return backbone, head, mean, std, ckpt


def main() -> int:
    parser = argparse.ArgumentParser(description="TimeSformer forgery window-MIL benchmark")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("FORENSHIELD_REPO", "~/forenShield-ai")))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "~/forenShield-ai/forgery/models/train/temporal/timesformer-forgery/"
            "timesformer-forgery-v1.8-csvted-v4-20260708-0022/forgery_head.pt"
        ),
    )
    parser.add_argument("--pretrained-id", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--aggregate", choices=("max", "topk"), default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--window-sec", type=float, default=None)
    parser.add_argument("--stride-sec", type=float, default=None)
    parser.add_argument("--clip-frames", type=int, default=None)
    parser.add_argument("--max-side", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--decode-cache",
        type=Path,
        default=Path("~/forenShield-ai/forgery/cache/decode-mp4-temporal"),
    )
    args = parser.parse_args()

    repo_root = Path(args.root).expanduser().resolve()
    forgery_root = repo_root / "forgery" if repo_root.name != "forgery" else repo_root
    infer_dir = Path(__file__).resolve().parent
    forgery_infer = repo_root / "forgery" / "scripts" / "infer"
    for cand in (infer_dir, forgery_infer, repo_root / "scripts" / "infer"):
        if cand.is_dir() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))

    from spatial_mvtamperbench_benchmark import collect_videos  # noqa: E402
    from timesformer_forgery_features import extract_video_window_embeddings, score_windows_mil  # noqa: E402

    data_root = Path(args.data_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    decode_cache = Path(args.decode_cache).expanduser().resolve()
    decode_cache.mkdir(parents=True, exist_ok=True)
    if not data_root.is_dir():
        print(f"data-root not found: {data_root}", file=sys.stderr)
        return 1
    if not checkpoint.is_file():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    backbone, head, mean, std, ckpt = load_backbone_and_head(checkpoint, device, args.pretrained_id)
    pretrained_id = args.pretrained_id or ckpt.get("pretrained_id", "facebook/timesformer-base-finetuned-k400")
    aggregate = args.aggregate or str(ckpt.get("aggregate", "topk"))
    top_k = args.top_k if args.top_k is not None else int(ckpt.get("top_k", 3))
    window_sec = args.window_sec if args.window_sec is not None else float(ckpt.get("window_sec", 1.0))
    stride_sec = args.stride_sec if args.stride_sec is not None else float(ckpt.get("stride_sec", 0.5))
    clip_frames = args.clip_frames if args.clip_frames is not None else int(ckpt.get("clip_frames", 8))
    max_side = args.max_side if args.max_side is not None else int(ckpt.get("max_side", 512))

    run_id = args.run_id or (
        f"timesformer-forgery-{data_root.name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    )
    out_infer = forgery_root / "results/infer" / run_id
    out_eval = forgery_root / "results/eval" / run_id
    json_dir = out_infer / "json"
    out_infer.mkdir(parents=True, exist_ok=True)
    out_eval.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(data_root)
    if not videos:
        print(f"no videos under {data_root}", file=sys.stderr)
        return 1

    print(
        f"videos: {len(videos)} (real={sum(1 for v in videos if v['ground_truth_label']=='real')}, "
        f"fake={sum(1 for v in videos if v['ground_truth_label']=='fake')}) "
        f"pretrained={pretrained_id} backbone_ft={bool(ckpt.get('backbone_state_dict'))} "
        f"aggregate={aggregate} top_k={top_k} "
        f"window={window_sec}s stride={stride_sec}s clip_frames={clip_frames}",
        flush=True,
    )
    backbone.eval()
    head.eval()
    print(f"head: {checkpoint}", flush=True)

    items: list[dict] = []
    with torch.inference_mode():
        for i, row in enumerate(videos, start=1):
            item = {
                "run_id": run_id,
                "model": "timesformer_forgery_window_mil",
                "pretrained_id": pretrained_id,
                "checkpoint": str(checkpoint),
                "file": row["file"],
                "source_path": row["source_path"],
                "relative_path": row["relative_path"],
                "ground_truth_label": row["ground_truth_label"],
                "status": "error",
                "tamper_score": None,
                "pred_label": None,
            }
            try:
                per_window, extract_meta = extract_video_window_embeddings(
                    Path(row["source_path"]),
                    backbone,
                    device,
                    window_sec=window_sec,
                    stride_sec=stride_sec,
                    clip_frames=clip_frames,
                    max_side=max_side,
                    decode_cache=decode_cache,
                )
                score, detail = score_windows_mil(
                    per_window,
                    head,
                    mean,
                    std,
                    device,
                    aggregate=aggregate,
                    top_k=top_k,
                )
                if score is None:
                    item["error_reason"] = extract_meta.get("error_reason") or "no_windows"
                    item["extract_meta"] = extract_meta
                else:
                    item["status"] = "ok"
                    item["tamper_score"] = round(float(score), 6)
                    item["pred_label"] = "fake" if score >= args.threshold else "real"
                    item["correct"] = item["pred_label"] == item["ground_truth_label"]
                    item["mil_detail"] = detail
                    item["extract_meta"] = {
                        k: extract_meta[k]
                        for k in ("fps", "total_frames", "n_windows", "windows_embedded")
                        if k in extract_meta
                    }
            except Exception as exc:
                item["error_reason"] = f"infer_failed:{exc}"

            items.append(item)
            status = item["status"]
            extra = item.get("tamper_score")
            if extra is None:
                extra = item.get("error_reason", "?")
            print(f"[{i:03d}/{len(videos)}] {status} {row['relative_path']} {extra}", flush=True)
            (json_dir / f"{Path(row['file']).stem}.json").write_text(
                json.dumps(item, indent=2), encoding="utf-8"
            )

    metrics = compute_eval(items, args.threshold)
    metrics.update(
        {
            "run_id": run_id,
            "model": "timesformer_forgery_window_mil",
            "pretrained_id": pretrained_id,
            "checkpoint": str(checkpoint),
            "data_root": str(data_root),
            "aggregate": aggregate,
            "top_k": top_k,
            "window_sec": window_sec,
            "stride_sec": stride_sec,
            "clip_frames": clip_frames,
            "max_side": max_side,
            "decode_cache": str(decode_cache),
        }
    )
    (out_infer / "items.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
    (out_eval / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nMETRICS", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
