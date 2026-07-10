#!/usr/bin/env python3
"""Optical-flow benchmark (RAFT / GMFlow / PWC-Net) on real + fake video folders."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optical_flow_backends import GmflowBackend, PwcnetBackend, RaftBackend
from optical_flow_common import aggregate_pair_stats, sample_frame_pairs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import s3_deepfake_paths as s3p


BACKENDS = {
    "raft": RaftBackend,
    "gmflow": GmflowBackend,
    "pwcnet": PwcnetBackend,
}


def run_video(
    video_path: Path,
    *,
    backends: list,
    max_pairs: int,
    ground_truth_label: str | None,
    run_id: str,
    device: torch.device,
) -> dict:
    pairs = sample_frame_pairs(video_path, max_pairs=max_pairs)
    if not pairs:
        return {
            "run_id": run_id,
            "file": video_path.name,
            "source_path": str(video_path.resolve()),
            "ground_truth_label": ground_truth_label,
            "status": "no_frames",
            "frame_pairs": 0,
            "models": {},
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
        }

    model_results: dict = {}
    for backend in backends:
        pair_stats: list[dict] = []
        errors: list[str] = []
        for img1, img2, idx1, idx2 in pairs:
            try:
                stats = backend.infer_pair(img1, img2)
                stats["frame_index_a"] = idx1
                stats["frame_index_b"] = idx2
                pair_stats.append(stats)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{idx1}->{idx2}: {exc}")
        model_results[backend.name] = {
            "status": "ok" if pair_stats else "error",
            "pair_stats": pair_stats,
            "aggregate": aggregate_pair_stats(pair_stats),
            "errors": errors,
        }

    return {
        "run_id": run_id,
        "file": video_path.name,
        "source_path": str(video_path.resolve()),
        "ground_truth_label": ground_truth_label,
        "status": "ok",
        "frame_pairs": len(pairs),
        "models": model_results,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
    }


def run_directory(
    input_dir: Path,
    *,
    backends: list,
    max_pairs: int,
    ground_truth_label: str | None,
    run_id: str,
    device: torch.device,
    json_dir: Path,
) -> list[dict]:
    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No mp4 files in {input_dir}")
    items: list[dict] = []
    for video_path in videos:
        item = run_video(
            video_path,
            backends=backends,
            max_pairs=max_pairs,
            ground_truth_label=ground_truth_label,
            run_id=run_id,
            device=device,
        )
        items.append(item)
        json_dir.mkdir(parents=True, exist_ok=True)
        (json_dir / f"{video_path.stem}.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        ok_models = [name for name, row in item.get("models", {}).items() if row.get("status") == "ok"]
        print(f"{video_path.name}: status={item['status']} models_ok={ok_models}", flush=True)
    return items


def summarize_class(items: list[dict], label: str) -> dict:
    rows = [x for x in items if x.get("ground_truth_label") == label and x.get("status") == "ok"]
    summary = {"count": len(rows), "label": label}
    for model_name in BACKENDS:
        values = []
        for row in rows:
            agg = row.get("models", {}).get(model_name, {}).get("aggregate", {})
            key = "magnitude_mean_mean"
            if key in agg:
                values.append(agg[key])
        if values:
            summary[model_name] = {
                "magnitude_mean_mean_avg": round(sum(values) / len(values), 6),
                "videos": len(values),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Optical-flow benchmark infer (RAFT/GMFlow/PWC-Net)")
    parser.add_argument("--root", default=".")
    parser.add_argument("--fake-dir", default="data/test/video/celeb-df-v2/fake")
    parser.add_argument("--real-dir", default="data/test/video/celeb-df-v2/real")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--models",
        default="raft,gmflow,pwcnet",
        help="comma-separated: raft,gmflow,pwcnet",
    )
    parser.add_argument("--max-pairs", type=int, default=8, help="frame pairs per video")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    fake_dir = Path(args.fake_dir)
    real_dir = Path(args.real_dir)
    if not fake_dir.is_absolute():
        fake_dir = (root / fake_dir).resolve()
    if not real_dir.is_absolute():
        real_dir = (root / real_dir).resolve()

    model_names = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    unknown = set(model_names) - set(BACKENDS)
    if unknown:
        raise SystemExit(f"unknown models: {sorted(unknown)}")

    run_id = args.run_id or f"optical-flow-benchmark-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    infer_dir = root / "results/infer" / run_id
    eval_dir = root / "results/eval" / run_id
    json_dir = infer_dir / "json"
    infer_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backends = [BACKENDS[name](root, device) for name in model_names]

    print("run_id:", run_id)
    print("device:", device)
    print("models:", model_names)
    print("fake:", fake_dir)
    print("real:", real_dir)
    print("max_pairs:", args.max_pairs)
    print()

    loaded_backends = []
    for backend in backends:
        print(f"loading {backend.name}...", flush=True)
        try:
            backend.load()
            loaded_backends.append(backend)
            print(f"  ok: {backend.name}", flush=True)
        except Exception as exc:
            print(f"  SKIP {backend.name}: {exc}", flush=True)
    if not loaded_backends:
        raise SystemExit("no optical-flow models loaded; run scripts/download/models/fix_optical_flow_env.sh")
    backends = loaded_backends
    print()

    fake_items = run_directory(
        fake_dir,
        backends=backends,
        max_pairs=args.max_pairs,
        ground_truth_label="fake",
        run_id=run_id,
        device=device,
        json_dir=json_dir,
    )
    print()
    real_items = run_directory(
        real_dir,
        backends=backends,
        max_pairs=args.max_pairs,
        ground_truth_label="real",
        run_id=run_id,
        device=device,
        json_dir=json_dir,
    )

    all_items = fake_items + real_items
    metrics = {
        "run_id": run_id,
        "models": model_names,
        "max_pairs": args.max_pairs,
        "total": len(all_items),
        "fake": summarize_class(all_items, "fake"),
        "real": summarize_class(all_items, "real"),
        "ok_videos": sum(1 for x in all_items if x["status"] == "ok"),
    }

    payload = {
        "run_id": run_id,
        "task": "optical_flow_benchmark",
        "models": model_names,
        "max_pairs": args.max_pairs,
        "fake_dir": str(fake_dir),
        "real_dir": str(real_dir),
        "device": str(device),
        "items": all_items,
    }
    (infer_dir / "predictions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    print("done:", len(list(json_dir.glob("*.json"))), "json files in", json_dir)
    print("metrics:", eval_dir / "metrics.json")
    print()
    print("S3 upload:")
    print(f"  S3_REPORT_PREFIX={s3p.LEGACY_OPTICAL_FLOW} \\")
    print(f"    UPLOAD_VIDEOS=1 bash scripts/upload/s3_upload_optical_flow_results.sh {run_id}")


if __name__ == "__main__":
    main()
