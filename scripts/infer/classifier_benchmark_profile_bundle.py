#!/usr/bin/env python3
"""Pack classifier infer run into video-benchmark-datasets bundle layout.

Output (under --out-dir / {profile}):
  fake/{stem}.json
  real/{stem}.json
  infer_summary.json
  metrics.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import s3_deepfake_paths as s3p


def main() -> None:
    parser = argparse.ArgumentParser(description="Classifier infer bundle for one benchmark profile")
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", required=True, help="celebdf | ffpp_vox")
    parser.add_argument("--model-slug", required=True, help="e.g. video-swin, convnext")
    parser.add_argument("--out-dir", required=True, help="e.g. results/video-swin-benchmark-bundle")
    parser.add_argument(
        "--s3-dataset-prefix",
        default=None,
        help=f"{s3p.DATASETS_BENCH}/{{profile}}",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    run_id = args.run_id.strip()
    profile = args.profile.strip()
    model_slug = args.model_slug.strip()

    infer_dir = root / "results/infer" / run_id
    eval_dir = root / "results/eval" / run_id
    json_dir = infer_dir / "json"
    predictions_path = infer_dir / "predictions.json"
    infer_summary_path = infer_dir / "datasets/infer_summary.json"
    metrics_path = eval_dir / "metrics.json"

    if not predictions_path.is_file():
        raise SystemExit(f"missing predictions: {predictions_path}")

    if not infer_summary_path.is_file():
        print("==> bundle report", profile)
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts/infer/bundle_xception_benchmark_report.py"),
                run_id,
                "--root",
                str(root),
                "--profile",
                profile,
            ],
            check=True,
        )

    if not infer_summary_path.is_file():
        raise SystemExit(f"missing infer_summary: {infer_summary_path}")
    if not metrics_path.is_file():
        raise SystemExit(f"missing metrics: {metrics_path}")

    out_profile = Path(args.out_dir)
    if not out_profile.is_absolute():
        out_profile = (root / out_profile).resolve()
    out_profile = out_profile / profile
    fake_json_dir = out_profile / "fake"
    real_json_dir = out_profile / "real"
    fake_json_dir.mkdir(parents=True, exist_ok=True)
    real_json_dir.mkdir(parents=True, exist_ok=True)

    s3_prefix = args.s3_dataset_prefix or s3p.bench_profile(profile)
    s3_output = s3p.infer_model(model_slug, profile)

    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    model_id = predictions.get("model", "")

    n_fake = n_real = 0
    for src in sorted(json_dir.glob("*.json")):
        item = json.loads(src.read_text(encoding="utf-8"))
        label = item.get("ground_truth_label") or "unknown"
        stem = Path(item.get("file", src.stem)).stem
        item["profile"] = profile
        item["task"] = "video_classifier_benchmark"
        item["schema_version"] = "1.1"
        item["model_slug"] = model_slug
        item["s3_dataset_prefix"] = s3_prefix
        item["s3_output_prefix"] = s3_output
        item["s3_source_key"] = f"{s3_prefix}/{label}/{stem}.mp4"
        dest_dir = fake_json_dir if label == "fake" else real_json_dir
        (dest_dir / f"{stem}.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        if label == "fake":
            n_fake += 1
        else:
            n_real += 1

    infer_summary = json.loads(infer_summary_path.read_text(encoding="utf-8"))
    infer_summary["model"] = infer_summary.get("model") or model_id
    infer_summary["profile"] = profile
    infer_summary["model_slug"] = model_slug
    infer_summary["run_id"] = run_id
    infer_summary["s3_dataset_prefix"] = s3_prefix
    infer_summary["s3_output_prefix"] = s3_output
    infer_summary["generated_at"] = datetime.now(timezone.utc).isoformat()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["profile"] = profile
    metrics["model_slug"] = model_slug
    metrics["s3_dataset_prefix"] = s3_prefix
    metrics["s3_output_prefix"] = s3_output
    metrics["generated_at"] = datetime.now(timezone.utc).isoformat()

    (out_profile / "infer_summary.json").write_text(json.dumps(infer_summary, indent=2), encoding="utf-8")
    (out_profile / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    print("done profile:", profile)
    print("  model:", model_id)
    print("  fake json:", n_fake)
    print("  real json:", n_real)
    print("  infer_summary:", out_profile / "infer_summary.json")
    print("  metrics:", out_profile / "metrics.json")


if __name__ == "__main__":
    main()
