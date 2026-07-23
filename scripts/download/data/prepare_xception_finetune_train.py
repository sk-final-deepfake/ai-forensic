#!/usr/bin/env python3
"""Build / upload Xception fine-tune train manifests from local RAW (excluding golden 200).

Selects clips not in benchmark manifests, writes ai/docs/xception_finetune_train_{stage}.json,
and optionally syncs mp4 + manifest to S3 train prefix.

Usage:
  python3 scripts/download/data/prepare_xception_finetune_train.py --stage stage1 --write-docs
  python3 scripts/download/data/prepare_xception_finetune_train.py --stage ff1k --write-docs --exclude-prior-train
  python3 scripts/download/data/prepare_xception_finetune_train.py --stage celeb1k --upload-s3
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "infer"))
from face_crop import create_face_cropper
from xception_finetune_data import (
    DEFAULT_BENCHMARK_EXCLUDE_DIRS,
    DEFAULT_S3_BUCKET,
    DEFAULT_S3_TRAIN_PREFIX,
    STAGE_DEFAULTS,
    collect_exclude_paths,
    collect_prior_train_exclude_paths,
    pick_celeb_train_dirs,
    pick_fake_train_dir,
    pick_vox_real_train_dir,
    resolve,
    resolve_train_dir,
    s3_uri,
    vox_real_train_dir_candidates,
    write_train_manifest_doc,
)
from xception_finetune_sampling import list_train_videos, split_train_val  # noqa: E402

ALL_STAGES = tuple(STAGE_DEFAULTS.keys())


def aws_s3_sync(local: Path, dest: str) -> None:
    cmd = ["aws", "s3", "sync", str(local), dest, "--exclude", "*", "--include", "*.mp4"]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def aws_s3_cp(local: Path, dest: str) -> None:
    cmd = ["aws", "s3", "cp", str(local), dest]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def stage_train_dirs(
    root: Path,
    stage: str,
    *,
    fake_dir: Path | None = None,
    real_dir: Path | None = None,
) -> tuple[Path, Path]:
    if stage == "stage1":
        fake = fake_dir or pick_fake_train_dir(
            root, "data/raw/faceforensics/manipulated_sequences/DeepFakeDetection/c40/videos"
        )
        real = real_dir or resolve_train_dir(root, "data/train/video/voxceleb/real", label="real")
        return fake, real
    if stage == "ff1k":
        fake = fake_dir or pick_fake_train_dir(
            root, "data/raw/faceforensics/manipulated_sequences/DeepFakeDetection/c40/videos"
        )
        real = real_dir or pick_vox_real_train_dir(root)
        return fake, real
    if stage in ("stage2", "celeb1k"):
        if fake_dir and real_dir:
            return fake_dir, real_dir
        return pick_celeb_train_dirs(root)
    raise ValueError(stage)


def prior_manifests_for_stage(stage: str) -> list[str]:
    if stage == "celeb1k":
        return [
            "docs/xception_finetune_train_ff1k.json",
            "data/pull/train/video/xception/ff1k/manifest.json",
        ]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Xception fine-tune train manifest")
    parser.add_argument("--root", default=".")
    parser.add_argument("--stage", required=True, choices=list(ALL_STAGES))
    parser.add_argument("--fake-dir", default=None, help="override fake pool")
    parser.add_argument("--real-dir", default=None, help="override real pool")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--val-holdout", type=int, default=None)
    parser.add_argument("--write-docs", action="store_true", help="write ai/docs manifest + markdown")
    parser.add_argument("--upload-s3", action="store_true")
    parser.add_argument("--bucket", default=DEFAULT_S3_BUCKET)
    parser.add_argument("--s3-prefix", default=DEFAULT_S3_TRAIN_PREFIX)
    parser.add_argument(
        "--exclude-dirs",
        nargs="*",
        default=list(DEFAULT_BENCHMARK_EXCLUDE_DIRS),
    )
    parser.add_argument(
        "--exclude-prior-train",
        action="store_true",
        default=False,
        help="also exclude clips from prior fine-tune manifests (recommended for ff1k/celeb1k)",
    )
    parser.add_argument(
        "--no-exclude-prior-train",
        action="store_false",
        dest="exclude_prior_train",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    spec = STAGE_DEFAULTS[args.stage]
    max_per_class = int(args.max_per_class if args.max_per_class is not None else spec["max_per_class"])
    val_holdout = int(args.val_holdout if args.val_holdout is not None else spec.get("val_holdout", 40))
    seed = int(args.seed if args.seed is not None else spec.get("seed", 42))

    exclude_prior = args.exclude_prior_train or args.stage in ("ff1k", "celeb1k")

    fake_override = resolve(root, args.fake_dir) if args.fake_dir else None
    real_override = resolve(root, args.real_dir) if args.real_dir else None
    fake_dir, real_dir = stage_train_dirs(
        root, args.stage, fake_dir=fake_override, real_dir=real_override
    )

    cropper = create_face_cropper(method="mediapipe", padding=0.3, square=True)
    try:
        excluded = collect_exclude_paths(root, args.exclude_dirs)
        if exclude_prior:
            prior_extra = prior_manifests_for_stage(args.stage)
            prior = collect_prior_train_exclude_paths(root, extra_manifests=prior_extra)
            excluded |= prior
            print(f"excluded prior train paths: {len(prior)}", flush=True)

        all_samples = list_train_videos(
            fake_dir,
            real_dir,
            excluded,
            max_per_class,
            seed,
            cropper,
            num_frames=32,
            cache_root=None,
            rebuild_cache=False,
            extra_real_dirs=(
                vox_real_train_dir_candidates(root)[1:]
                if args.stage == "ff1k" and not real_override
                else None
            ),
        )
        if len(all_samples) < max_per_class * 2 * 0.5:
            got_fake = sum(1 for _, y in all_samples if y == 1)
            got_real = sum(1 for _, y in all_samples if y == 0)
            print(
                f"WARN: sampled fake={got_fake} real={got_real} "
                f"(target {max_per_class} per class). Check pools or exclusions.",
                flush=True,
            )
        train_samples, val_samples = split_train_val(all_samples, val_holdout, seed)

        pull_base = root / "data/pull/train/video/xception" / spec["s3_subdir"]
        pull_base.mkdir(parents=True, exist_ok=True)
        for sub in ("fake", "real"):
            (pull_base / sub).mkdir(parents=True, exist_ok=True)

        manifest_rows: list[dict] = []
        for path, label in all_samples:
            sub = "fake" if label == 1 else "real"
            dest = pull_base / sub / path.name
            if not dest.exists():
                shutil.copy2(path, dest)
            manifest_rows.append(
                {
                    "file": dest.name,
                    "label": sub,
                    "source_path": str(path.resolve()),
                    "local_path": str(dest.resolve()),
                }
            )

        manifest_payload = {
            "stage": args.stage,
            "seed": seed,
            "max_per_class": max_per_class,
            "val_holdout": val_holdout,
            "train": [
                {
                    "file": p.name,
                    "label": "fake" if y == 1 else "real",
                    "local_path": str(p.resolve()),
                    "source_path": str(p.resolve()),
                    "split": "train",
                }
                for p, y in train_samples
            ],
            "val": [
                {
                    "file": p.name,
                    "label": "fake" if y == 1 else "real",
                    "local_path": str(p.resolve()),
                    "source_path": str(p.resolve()),
                    "split": "val",
                }
                for p, y in val_samples
            ],
        }
        manifest_path = pull_base / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

        meta = {
            "seed": seed,
            "val_holdout": val_holdout,
            "max_per_class": max_per_class,
            "fake_pool": str(fake_dir),
            "real_pool": str(real_dir),
            "excluded_paths": len(excluded),
            "exclude_dirs": args.exclude_dirs,
            "exclude_prior_train": exclude_prior,
        }

        if args.write_docs:
            json_path, md_path = write_train_manifest_doc(
                root,
                stage=args.stage,
                train_samples=train_samples,
                val_samples=val_samples,
                meta=meta,
            )
            print(f"docs json: {json_path}")
            print(f"docs md:   {md_path}")

        if args.upload_s3:
            s3_base = s3_uri(args.bucket, args.s3_prefix, spec["s3_subdir"])
            aws_s3_sync(pull_base / "fake", f"{s3_base}/fake/")
            aws_s3_sync(pull_base / "real", f"{s3_base}/real/")
            aws_s3_cp(manifest_path, f"{s3_base}/manifest.json")

        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "train": len(train_samples),
                    "val": len(val_samples),
                    "max_per_class": max_per_class,
                },
                indent=2,
            )
        )
    finally:
        cropper.close()


if __name__ == "__main__":
    main()
