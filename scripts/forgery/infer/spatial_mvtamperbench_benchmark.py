#!/usr/bin/env python3
"""TruFor / CAT-Net benchmark on MVTamperBench-style video folders (100 real + 100 fake).

Both models are image-based; videos are sampled to JPEG frames, inferred in one batch,
then aggregated per video (default: max frame score).

Expected layout under --data-root:
  original/{category}/*.mp4          -> ground_truth_label=real
  tampered/{type}/{category}/*.mp4   -> ground_truth_label=fake

Example (GPU, forgery track):
  export FORENSHIELD_AI_ROOT=~/forenShield-ai/forgery
  python3 forgery/scripts/infer/spatial_mvtamperbench_benchmark.py \\
    --model trufor \\
    --data-root ~/forenShield-ai/forgery/data/pull/evidence/mvtamperbench-200-s3 \\
    --run-id trufor-mvtb200-$(date -u +%Y%m%d-%H%M)

  python3 forgery/scripts/infer/spatial_mvtamperbench_benchmark.py \\
    --model catnet \\
    --data-root ~/forenShield-ai/forgery/data/pull/evidence/mvtamperbench-200-s3 \\
    --run-id catnet-mvtb200-$(date -u +%Y%m%d-%H%M)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
PAIR_RE = re.compile(r"(?P<sid>.+?)__(?:org|fake)\.mp4$", re.IGNORECASE)


def sample_id_from_filename(name: str) -> str | None:
    m = PAIR_RE.match(name)
    return m.group("sid") if m else None


def video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return total


def build_pair_align_totals(videos: list[dict]) -> dict[str, int]:
    """Per sample_id: min(org_frame_count, fake_frame_count) for aligned sampling."""
    counts: dict[str, dict[str, int]] = {}
    for row in videos:
        sid = sample_id_from_filename(row["file"])
        if not sid:
            continue
        role = "org" if row["ground_truth_label"] == "real" else "fake"
        n = video_frame_count(Path(row["source_path"]))
        counts.setdefault(sid, {})[role] = n
    align: dict[str, int] = {}
    for sid, roles in counts.items():
        if "org" in roles and "fake" in roles:
            align[sid] = min(roles["org"], roles["fake"])
    return align


def video_frame_indices(
    total: int,
    video_fps: float,
    *,
    num_frames: int = 8,
    sample_fps: float | None = None,
    min_frames: int = 4,
    max_frames: int | None = None,
) -> list[int]:
    """Return frame indices for sampling.

    - sample_fps set: fixed temporal rate (e.g. 2.0 -> one frame every 0.5s).
    - else: linspace num_frames across the clip (legacy default).
    """
    if total < 1:
        return []
    if sample_fps is not None and sample_fps > 0:
        vfps = video_fps if video_fps > 1e-3 else 30.0
        step = max(1, int(round(vfps / sample_fps)))
        indices = list(range(0, total, step))
        if len(indices) < min_frames:
            n = min(min_frames, total)
            indices = np.linspace(0, max(total - 1, 0), num=n, dtype=int).tolist()
        if max_frames is not None and len(indices) > max_frames:
            pick = np.linspace(0, len(indices) - 1, num=max_frames, dtype=int)
            indices = [indices[int(i)] for i in pick]
    else:
        indices = np.linspace(0, max(total - 1, 0), num=num_frames, dtype=int).tolist()
    seen: set[int] = set()
    out: list[int] = []
    for idx in indices:
        i = int(idx)
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def sample_video_frames(
    video_path: Path,
    out_dir: Path,
    stem: str,
    num_frames: int,
    *,
    sample_fps: float | None = None,
    min_frames: int = 4,
    max_frames: int | None = None,
    align_total: int | None = None,
) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if total_video < 1:
        cap.release()
        return []
    total_for_indices = total_video
    if align_total is not None and align_total > 0:
        total_for_indices = min(align_total, total_video)
    indices = video_frame_indices(
        total_for_indices,
        video_fps,
        num_frames=num_frames,
        sample_fps=sample_fps,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for i, idx in enumerate(indices):
        fi = min(int(idx), max(total_video - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        dst = out_dir / f"{stem}_f{i:03d}.jpg"
        cv2.imwrite(str(dst), frame)
        saved.append(dst)
    cap.release()
    return saved


def collect_videos(data_root: Path) -> list[dict]:
    rows: list[dict] = []
    for mp4 in sorted(data_root.rglob("*")):
        if not mp4.is_file() or mp4.suffix.lower() not in VIDEO_EXTS:
            continue
        rel = mp4.relative_to(data_root)
        parts = rel.parts
        if not parts:
            continue
        if parts[0] == "original":
            label = "real"
        elif parts[0] == "tampered":
            label = "fake"
        else:
            continue
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(rel.with_suffix("")))
        if len(stem) > 120:
            stem = f"v{abs(hash(str(rel))) % 10**8:08d}"
        rows.append(
            {
                "file": mp4.name,
                "source_path": str(mp4.resolve()),
                "relative_path": str(rel),
                "ground_truth_label": label,
                "frame_stem": stem,
            }
        )
    return rows


def score_from_trufor_npz(npz_path: Path) -> float | None:
    try:
        data = np.load(npz_path)
        if "score" in data:
            return float(np.asarray(data["score"]).reshape(-1)[0])
        if "map" in data:
            return float(np.max(data["map"]))
    except Exception:
        return None
    return None


def score_from_catnet_heatmap(path: Path) -> float | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return float(np.max(img) / 255.0)


def aggregate_frame_scores(values: list[float], mode: str) -> float:
    if not values:
        return 0.0
    if mode == "top2_mean":
        top = sorted(values, reverse=True)[:2]
        return float(sum(top) / len(top))
    if mode == "top3_mean":
        top = sorted(values, reverse=True)[:3]
        return float(sum(top) / len(top))
    return float(max(values))


def run_trufor(
    frames_dir: Path,
    out_dir: Path,
    weights: Path,
    vendor_test: Path,
    gpu: int,
    *,
    aggregate: str = "max",
) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(vendor_test),
        "-g",
        str(gpu),
        "-in",
        str(frames_dir),
        "-out",
        str(out_dir),
        "-exp",
        "trufor_ph3",
        "TEST.MODEL_FILE",
        str(weights),
    ]
    env = os.environ.copy()
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(vendor_test.parent), check=True, env=env)
    per_video: dict[str, list[float]] = {}
    for npz in sorted(out_dir.rglob("*.npz")):
        stem = npz.stem
        if "_f" not in stem:
            continue
        video_stem = stem.rsplit("_f", 1)[0]
        val = score_from_trufor_npz(npz)
        if val is None:
            continue
        per_video.setdefault(video_stem, []).append(val)
    scores: dict[str, float] = {}
    for video_stem, vals in per_video.items():
        scores[video_stem] = aggregate_frame_scores(vals, aggregate)
    return scores


def setup_catnet_weights(catnet_repo: Path, weights_root: Path) -> None:
    pre = weights_root / "pretrained_models"
    full = weights_root / "CAT_full_v2.pth.tar"
    if not full.is_file():
        raise SystemExit(f"CAT-Net weight missing: {full}")
    if not (pre / "hrnetv2_w48_imagenet_pretrained.pth").is_file():
        raise SystemExit(f"CAT-Net pretrained missing: {pre}/hrnetv2_w48_imagenet_pretrained.pth")
    if not (pre / "DCT_djpeg.pth.tar").is_file():
        raise SystemExit(f"CAT-Net pretrained missing: {pre}/DCT_djpeg.pth.tar")

    dst_pre = catnet_repo / "pretrained_models"
    dst_out = catnet_repo / "output" / "splicing_dataset" / "CAT_full"
    dst_pre.mkdir(parents=True, exist_ok=True)
    dst_out.mkdir(parents=True, exist_ok=True)
    for name in ("hrnetv2_w48_imagenet_pretrained.pth", "DCT_djpeg.pth.tar"):
        link = dst_pre / name
        if not link.exists():
            os.symlink(pre / name, link)
    link_full = dst_out / "CAT_full_v2.pth.tar"
    if not link_full.exists():
        os.symlink(full, link_full)


def run_catnet(frames_dir: Path, catnet_repo: Path, weights_root: Path) -> dict[str, float]:
    setup_catnet_weights(catnet_repo, weights_root)
    input_dir = catnet_repo / "input"
    output_dir = catnet_repo / "output_pred"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(frames_dir, input_dir)
    cmd = [sys.executable, "tools/infer.py"]
    print(f"cd {catnet_repo} && {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(catnet_repo), check=True)
    scores: dict[str, float] = {}
    for heat in sorted(output_dir.glob("*")):
        if not heat.is_file():
            continue
        stem = heat.stem
        if "_f" not in stem:
            continue
        video_stem = stem.rsplit("_f", 1)[0]
        val = score_from_catnet_heatmap(heat)
        if val is None:
            continue
        scores[video_stem] = max(scores.get(video_stem, 0.0), val)
    return scores


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


def main() -> int:
    parser = argparse.ArgumentParser(description="TruFor / CAT-Net video benchmark (frame aggregation)")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("FORENSHIELD_AI_ROOT", "~/forenShield-ai/forgery")))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("~/forenShield-ai/forgery/data/pull/evidence/mvtamperbench-200-s3"),
    )
    parser.add_argument("--model", choices=["trufor", "catnet"], required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--num-frames", type=int, default=8, help="linspace count when --sample-fps is not set")
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="fixed temporal sampling rate (e.g. 2.0 = 2 frames/sec). Overrides --num-frames.",
    )
    parser.add_argument("--min-frames", type=int, default=4, help="floor when --sample-fps yields too few frames")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="cap frames per video (recommended for long clips, e.g. 64)",
    )
    parser.add_argument(
        "--align-pairs",
        action="store_true",
        help="use min(org_frames, fake_frames) and identical frame indices per paired sample_id",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--aggregate",
        choices=["max", "top2_mean", "top3_mean"],
        default="max",
        help="per-video score from frame scores",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--trufor-weights",
        type=Path,
        default=Path("models/test/spatial/trufor/v1.0.0/trufor.pth.tar"),
    )
    parser.add_argument(
        "--catnet-weights",
        type=Path,
        default=Path("models/test/spatial/catnet/v1.0.0"),
    )
    parser.add_argument("--trufor-vendor", type=Path, default=Path("vendor/TruFor/TruFor_train_test"))
    parser.add_argument("--catnet-vendor", type=Path, default=Path("vendor/CAT-Net"))
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        print(f"data-root not found: {data_root}", file=sys.stderr)
        return 1

    run_id = args.run_id or f"{args.model}-mvtb200-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    infer_dir = root / "results/infer" / run_id
    eval_dir = root / "results/eval" / run_id
    work_dir = infer_dir / "work"
    frames_dir = work_dir / "frames"
    json_dir = infer_dir / "json"
    infer_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(data_root)
    if not videos:
        print(f"no videos under {data_root}", file=sys.stderr)
        return 1
    print(f"videos: {len(videos)} (real={sum(1 for v in videos if v['ground_truth_label']=='real')}, "
          f"fake={sum(1 for v in videos if v['ground_truth_label']=='fake')})")

    if args.sample_fps is not None:
        print(
            f"sampling: sample_fps={args.sample_fps} min_frames={args.min_frames} "
            f"max_frames={args.max_frames} align_pairs={args.align_pairs}",
            flush=True,
        )
    else:
        print(
            f"sampling: linspace num_frames={args.num_frames} align_pairs={args.align_pairs}",
            flush=True,
        )

    align_map: dict[str, int] = {}
    if args.align_pairs:
        align_map = build_pair_align_totals(videos)
        role_counts: dict[str, dict[str, int]] = {}
        for row in videos:
            sid = sample_id_from_filename(row["file"])
            if not sid:
                continue
            role = "org" if row["ground_truth_label"] == "real" else "fake"
            role_counts.setdefault(sid, {})[role] = video_frame_count(Path(row["source_path"]))
        mismatches = sum(
            1 for sid, rc in role_counts.items() if "org" in rc and "fake" in rc and rc["org"] != rc["fake"]
        )
        print(
            f"align_pairs: {len(align_map)} pairs, frame_count mismatches={mismatches}",
            flush=True,
        )

    for i, row in enumerate(videos, start=1):
        sid = sample_id_from_filename(row["file"])
        align_total = align_map.get(sid) if args.align_pairs and sid else None
        saved = sample_video_frames(
            Path(row["source_path"]),
            frames_dir,
            row["frame_stem"],
            args.num_frames,
            sample_fps=args.sample_fps,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            align_total=align_total,
        )
        row["frames_extracted"] = len(saved)
        row["sample_id"] = sid
        row["align_total"] = align_total
        suffix = f" align={align_total}" if align_total is not None else ""
        print(
            f"[{i:03d}/{len(videos)}] {row['relative_path']} frames={len(saved)}{suffix}",
            flush=True,
        )

    if args.model == "trufor":
        weights = args.trufor_weights if args.trufor_weights.is_absolute() else root / args.trufor_weights
        vendor_test = args.trufor_vendor if args.trufor_vendor.is_absolute() else root / args.trufor_vendor / "test.py"
        if not vendor_test.is_file():
            vendor_test = (root / args.trufor_vendor).resolve() / "test.py"
        if not vendor_test.is_file():
            print(f"TruFor test.py not found: {vendor_test}", file=sys.stderr)
            print("clone: git clone https://github.com/grip-unina/TruFor.git vendor/TruFor", file=sys.stderr)
            return 1
        scores = run_trufor(
            frames_dir, work_dir / "trufor_out", weights.resolve(), vendor_test, args.gpu, aggregate=args.aggregate
        )
    else:
        catnet_repo = args.catnet_vendor if args.catnet_vendor.is_absolute() else root / args.catnet_vendor
        weights_root = args.catnet_weights if args.catnet_weights.is_absolute() else root / args.catnet_weights
        if not (catnet_repo / "tools" / "infer.py").is_file():
            print(f"CAT-Net repo not found: {catnet_repo}", file=sys.stderr)
            print("clone: git clone https://github.com/mjkwon2021/CAT-Net.git vendor/CAT-Net", file=sys.stderr)
            return 1
        scores = run_catnet(frames_dir, catnet_repo.resolve(), weights_root.resolve())

    items: list[dict] = []
    for row in videos:
        stem = row["frame_stem"]
        score = scores.get(stem)
        item = {
            "run_id": run_id,
            "model": args.model,
            "file": row["file"],
            "source_path": row["source_path"],
            "relative_path": row["relative_path"],
            "ground_truth_label": row["ground_truth_label"],
            "frames_extracted": row.get("frames_extracted", 0),
            "tamper_score": score,
            "pred_label": "fake" if score is not None and score >= args.threshold else "real",
            "status": "ok" if score is not None and row.get("frames_extracted", 0) > 0 else "error",
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }
        if item["status"] == "ok":
            item["correct"] = item["pred_label"] == item["ground_truth_label"]
        items.append(item)
        (json_dir / f"{stem}.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        print(
            f"{row['relative_path']}: score={score} pred={item.get('pred_label')} gt={row['ground_truth_label']}",
            flush=True,
        )

    metrics = compute_eval(items, args.threshold)
    metrics["run_id"] = run_id
    metrics["model"] = args.model
    metrics["num_frames"] = args.num_frames
    metrics["sample_fps"] = args.sample_fps
    metrics["min_frames"] = args.min_frames
    metrics["max_frames"] = args.max_frames
    metrics["align_pairs"] = args.align_pairs
    metrics["aggregate"] = args.aggregate
    metrics["data_root"] = str(data_root)

    payload = {
        "run_id": run_id,
        "model": args.model,
        "task": "spatial_forgery_video_benchmark",
        "data_root": str(data_root),
        "num_frames": args.num_frames,
        "sample_fps": args.sample_fps,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "align_pairs": args.align_pairs,
        "aggregate": args.aggregate,
        "threshold": args.threshold,
        "items": items,
    }
    (infer_dir / "predictions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not args.keep_frames and work_dir.exists():
        shutil.rmtree(work_dir)

    print("\nmetrics:", json.dumps(metrics, indent=2))
    print(f"predictions: {infer_dir / 'predictions.json'}")
    print(f"per-video json: {json_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
