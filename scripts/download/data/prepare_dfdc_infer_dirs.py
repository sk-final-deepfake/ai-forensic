#!/usr/bin/env python3
"""Split data/test/video/dfdc/ into fake/ and real/ for benchmark infer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".", help="forenShield-ai root")
    p.add_argument(
        "--dataset-dir",
        default="data/test/video/dfdc",
        help="DFDC folder with manifest.json + mp4 files",
    )
    args = p.parse_args()

    root = Path(args.root).resolve()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = (root / dataset_dir).resolve()

    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest: {manifest_path}")

    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"manifest must be a list: {manifest_path}")

    fake_dir = dataset_dir / "fake"
    real_dir = dataset_dir / "real"
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    fake_n = real_n = missing = 0
    for row in rows:
        filename = row.get("file")
        label = str(row.get("label", "")).lower()
        if not filename:
            continue
        src = dataset_dir / filename
        if not src.is_file():
            missing += 1
            continue
        if label == "fake":
            link_or_copy(src, fake_dir / filename)
            fake_n += 1
        elif label == "real":
            link_or_copy(src, real_dir / filename)
            real_n += 1

    print(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "fake_dir": str(fake_dir),
                "real_dir": str(real_dir),
                "fake": fake_n,
                "real": real_n,
                "missing": missing,
            },
            indent=2,
        )
    )
    if fake_n == 0 or real_n == 0:
        raise SystemExit("need both fake and real videos under dataset-dir")


if __name__ == "__main__":
    main()
