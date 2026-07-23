#!/usr/bin/env python3
"""Pull Xception fine-tune train clips from S3 (unseen vs golden 200).

S3 layout (upload / maintain on evidence bucket):

  s3://forenshield-evidence-877044078824/cases/train/video/xception/
    stage1/
      manifest.json
      fake/*.mp4
      real/*.mp4
    stage2/
      manifest.json
      fake/*.mp4
      real/*.mp4

Usage (GPU):
  cd ~/forenShield-ai && source .venv/bin/activate && unset AWS_PROFILE
  python3 scripts/download/data/s3_pull_xception_finetune_train.py --stage stage1
  python3 scripts/download/data/s3_pull_xception_finetune_train.py --stage all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "infer"))
from xception_finetune_data import (  # noqa: E402
    DEFAULT_S3_BUCKET,
    DEFAULT_S3_TRAIN_PREFIX,
    STAGE_DEFAULTS,
    docs_manifest_path,
    pull_stage_from_s3,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Xception fine-tune train data from S3")
    parser.add_argument("--root", default=".")
    parser.add_argument("--stage", default="all", choices=["stage1", "stage2", "ff1k", "celeb1k", "all", "all-1k"])
    parser.add_argument("--bucket", default=DEFAULT_S3_BUCKET)
    parser.add_argument("--s3-prefix", default=DEFAULT_S3_TRAIN_PREFIX)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    stages = ["stage1", "stage2"] if args.stage == "all" else ["ff1k", "celeb1k"] if args.stage == "all-1k" else [args.stage]

    summary: dict[str, dict] = {}
    for stage in stages:
        print(f"=== pull {stage} ({STAGE_DEFAULTS[stage]['label']}) ===", flush=True)
        local_dir = pull_stage_from_s3(
            root,
            stage,
            bucket=args.bucket,
            s3_prefix=args.s3_prefix,
        )
        counts = {}
        for label in ("fake", "real"):
            d = local_dir / label
            counts[label] = len(list(d.glob("*.mp4"))) if d.is_dir() else 0
        manifest = local_dir / "manifest.json"
        summary[stage] = {
            "local_dir": str(local_dir),
            "manifest": str(manifest) if manifest.is_file() else None,
            "mp4": counts,
        }
        print(json.dumps(summary[stage], indent=2), flush=True)

        # Copy S3 manifest snapshot into docs if present (filenames reference before train run).
        if manifest.is_file():
            docs_copy = docs_manifest_path(root, f"{stage}_s3_snapshot")
            docs_copy.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"snapshot: {docs_copy}", flush=True)

    out = root / "data/pull/train/video/xception/pull_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
