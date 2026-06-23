#!/usr/bin/env python3
"""Run RAFT on one benchmark profile and write S3-ready bundle layout.

Output layout (under --out-dir):
  {profile}/fake/{stem}.json
  {profile}/real/{stem}.json
  {profile}/infer_summary.json
  {profile}/metrics.json

Reuses optical_flow_infer_model helpers (full per-video JSON fields).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optical_flow_backends import BACKENDS
from optical_flow_infer_model import (
    build_infer_summary,
    run_directory,
    summarize_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAFT infer bundle for one benchmark profile")
    parser.add_argument("--root", default=".")
    parser.add_argument("--profile", required=True, help="celebdf | ffpp_vox")
    parser.add_argument("--fake-dir", required=True)
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--out-dir", required=True, help="e.g. results/raft-benchmark-bundle")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument(
        "--s3-dataset-prefix",
        default=None,
        help="optional S3 prefix for metadata (cases/test/video-benchmark-datasets)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    fake_dir = Path(args.fake_dir)
    real_dir = Path(args.real_dir)
    if not fake_dir.is_absolute():
        fake_dir = (root / fake_dir).resolve()
    if not real_dir.is_absolute():
        real_dir = (root / real_dir).resolve()

    profile = args.profile.strip()
    run_id = args.run_id or f"raft-benchmark-{profile}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    out_profile = Path(args.out_dir)
    if not out_profile.is_absolute():
        out_profile = (root / out_profile).resolve()
    out_profile = out_profile / profile
    fake_json_dir = out_profile / "fake"
    real_json_dir = out_profile / "real"
    fake_json_dir.mkdir(parents=True, exist_ok=True)
    real_json_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = BACKENDS["raft"](root, device)

    s3_prefix = args.s3_dataset_prefix or f"cases/test/video-benchmark-datasets/{profile}"

    print("run_id:", run_id)
    print("profile:", profile)
    print("device:", device)
    print("max_pairs:", args.max_pairs)
    print("max_side:", args.max_side)
    print("fake:", fake_dir)
    print("real:", real_dir)
    print("out:", out_profile)
    print()

    backend.load()
    print()

    fake_items = run_directory(
        fake_dir,
        backend,
        max_pairs=args.max_pairs,
        max_side=args.max_side,
        ground_truth_label="fake",
        run_id=run_id,
        model_name="raft",
        device=device,
        json_dir=fake_json_dir,
    )
    print()
    real_items = run_directory(
        real_dir,
        backend,
        max_pairs=args.max_pairs,
        max_side=args.max_side,
        ground_truth_label="real",
        run_id=run_id,
        model_name="raft",
        device=device,
        json_dir=real_json_dir,
    )

    all_items = fake_items + real_items
    weights_path = str(getattr(backend, "weights", ""))

    # Enrich each item with profile / dataset metadata (non-destructive).
    for item in all_items:
        item["profile"] = profile
        item["task"] = "optical_flow_benchmark"
        item["schema_version"] = "1.0"
        item["s3_dataset_prefix"] = s3_prefix
        stem = Path(item["file"]).stem
        label = item.get("ground_truth_label") or "unknown"
        item["s3_source_key"] = f"{s3_prefix}/{label}/{stem}.mp4"
        json_path = fake_json_dir if label == "fake" else real_json_dir
        (json_path / f"{stem}.json").write_text(json.dumps(item, indent=2), encoding="utf-8")

    infer_summary = build_infer_summary(
        run_id=run_id,
        model_name="raft",
        items=all_items,
        fake_dir=fake_dir,
        real_dir=real_dir,
        max_pairs=args.max_pairs,
        max_side=args.max_side,
        device=device,
        weights=weights_path,
    )
    infer_summary["profile"] = profile
    infer_summary["s3_dataset_prefix"] = s3_prefix
    infer_summary["s3_output_prefix"] = f"cases/test/video-benchmark-datasets/raft/{profile}"
    # Full per-video rows (all aggregate + pair_stats counts) for reporting.
    # Full per-video table rows (every top-level + aggregate + pair_stats field).
    infer_summary["videos"] = [
        {
            **item,
            "pair_stats_count": len(item.get("pair_stats") or []),
        }
        for item in all_items
    ]
    # Expand summary items with all aggregate_* keys for quick scanning.
    infer_summary["items"] = [
        {
            "file": item["file"],
            "ground_truth_label": item.get("ground_truth_label"),
            "status": item.get("status"),
            "frame_pairs": item.get("frame_pairs"),
            "max_side": item.get("max_side"),
            "errors_count": len(item.get("errors") or []),
            **(item.get("aggregate") or {}),
        }
        for item in all_items
    ]

    metrics = summarize_metrics(all_items, "raft")
    metrics["run_id"] = run_id
    metrics["profile"] = profile
    metrics["max_pairs"] = args.max_pairs
    metrics["max_side"] = args.max_side
    metrics["fake_dir"] = str(fake_dir)
    metrics["real_dir"] = str(real_dir)
    metrics["weights"] = weights_path
    metrics["device"] = str(device)
    metrics["s3_dataset_prefix"] = s3_prefix
    metrics["generated_at"] = datetime.now(timezone.utc).isoformat()

    (out_profile / "infer_summary.json").write_text(json.dumps(infer_summary, indent=2), encoding="utf-8")
    (out_profile / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    print("done profile:", profile)
    print("  fake json:", len(list(fake_json_dir.glob("*.json"))))
    print("  real json:", len(list(real_json_dir.glob("*.json"))))
    print("  infer_summary:", out_profile / "infer_summary.json")
    print("  metrics:", out_profile / "metrics.json")


if __name__ == "__main__":
    main()
