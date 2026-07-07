#!/usr/bin/env python3
"""Download and prepare Celeb-DF v2 on the GPU workstation.

Official access (required first):
  https://forms.gle/2jYBby6y1FBU3u6q9
  Authors email a Google Drive (or similar) link after approval.

After you receive the link:
  bash scripts/download/data/download_celebdf_v2.sh --url '<drive-or-http-url>'

Or if you already downloaded the archive on PC:
  scp celebdf_v2.zip sk4team@GPU:~/forenShield-ai/data/raw/benchmark-downloads/celeb-df-v2/
  bash scripts/download/data/download_celebdf_v2.sh --archive ~/forenShield-ai/data/raw/benchmark-downloads/celeb-df-v2/celebdf_v2.zip

Layout after extract:
  data/raw/celeb-df-v2/Celeb-DF-v2/
    Celeb-real/videos/*.mp4
    YouTube-real/videos/*.mp4
    Celeb-synthesis/videos/*.mp4
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
REAL_DIRS = ("Celeb-real", "YouTube-real", "Youtube-real", "Youtube-real")
FAKE_DIRS = ("Celeb-synthesis",)
REAL_PATH_KEYS = ("celeb-real", "youtube-real", "youtube_real", "celeb_real")
FAKE_PATH_KEYS = ("celeb-synthesis", "celeb_synthesis", "/synthesis/", "\\synthesis\\")


def label_video_path(path: Path) -> str | None:
    joined = str(path).lower().replace("_", "-")
    if any(k in joined for k in FAKE_PATH_KEYS):
        return "fake"
    if any(k in joined for k in REAL_PATH_KEYS):
        return "real"
    for part in path.parts:
        pl = part.lower()
        if pl in {"fake", "fakes", "synthesis"}:
            return "fake"
        if pl in {"real", "reals", "original", "bonafide"}:
            return "real"
    return None


def discover_video_pools(root: Path) -> tuple[list[Path], list[Path]]:
    real_pool: list[Path] = []
    fake_pool: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        label = label_video_path(path)
        if label == "real":
            real_pool.append(path)
        elif label == "fake":
            fake_pool.append(path)
    return sorted(real_pool), sorted(fake_pool)


def summarize_tree(root: Path, limit: int = 25) -> list[str]:
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            lines.append(str(path.relative_to(root)))
            if len(lines) >= limit:
                break
    return lines


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")


def download_url(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    if "drive.google.com" in url or "docs.google.com" in url:
        if shutil.which("gdown") is None:
            raise RuntimeError("pip install gdown")
        run(["gdown", url, "-O", str(dest)])
        return dest

    if shutil.which("wget"):
        run(["wget", "-O", str(dest), url])
        return dest

    if shutil.which("curl"):
        run(["curl", "-L", "-o", str(dest), url])
        return dest

    raise RuntimeError("need wget or curl for direct HTTP download")


def extract_archive(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        return
    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest)
        return
    raise RuntimeError(f"unsupported archive: {archive}")


def download_kaggle_dataset(slug: str, cache_dir: Path, *, force: bool = False) -> Path | None:
    if shutil.which("kaggle") is None:
        raise RuntimeError("pip install kaggle")
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(cache_dir.glob("*.zip")) + sorted(cache_dir.glob("*.tar*"))
    if existing and not force:
        print("reuse existing archive:", max(existing, key=lambda p: p.stat().st_size))
    else:
        run(["kaggle", "datasets", "download", "-d", slug, "-p", str(cache_dir)])
        existing = sorted(cache_dir.glob("*.zip")) + sorted(cache_dir.glob("*.tar*"))
    if not existing:
        try:
            find_dataset_root(cache_dir)
            return None
        except FileNotFoundError as exc:
            raise RuntimeError(f"kaggle download produced no usable files under {cache_dir}") from exc
    return max(existing, key=lambda p: p.stat().st_size)


def find_dataset_root(root: Path) -> Path:
    """Return a directory that contains Celeb-DF v2 videos (official or Kaggle layout)."""
    candidates = [root / "Celeb-DF-v2", root / "Celeb-DF", root / "celeb-df-v2", root]
    for base in candidates:
        if not base.is_dir():
            continue
        real_pool, fake_pool = discover_video_pools(base)
        if real_pool and fake_pool:
            return base

    real_pool, fake_pool = discover_video_pools(root)
    if real_pool and fake_pool:
        return root

    sample = summarize_tree(root)
    hint = "\n  ".join(sample) if sample else "(no files found)"
    raise FileNotFoundError(
        f"Celeb-DF v2 videos not found under {root}.\n"
        "Expected paths containing celeb-real / youtube-real (real) and celeb-synthesis (fake).\n"
        f"Sample files:\n  {hint}"
    )


def iter_videos(category_dir: Path) -> list[Path]:
    videos_dir = category_dir / "videos"
    search_root = videos_dir if videos_dir.is_dir() else category_dir
    return sorted(
        p for p in search_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def copy_sample(src: Path, dst_dir: Path, name: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / name
    if dst.exists():
        return dst
    shutil.copy2(src, dst)
    return dst


def sample_benchmark(
    dataset_root: Path,
    out_dir: Path,
    *,
    target_real: int,
    target_fake: int,
    seed: int,
) -> list[dict]:
    real_pool, fake_pool = discover_video_pools(dataset_root)

    # Fallback: official folder names without path keywords in filenames
    if not real_pool:
        for name in REAL_DIRS:
            cat = dataset_root / name
            if cat.is_dir():
                real_pool.extend(iter_videos(cat))
    if not fake_pool:
        for name in FAKE_DIRS:
            cat = dataset_root / name
            if cat.is_dir():
                fake_pool.extend(iter_videos(cat))
    real_pool = sorted(set(real_pool))
    fake_pool = sorted(set(fake_pool))

    if len(real_pool) < target_real:
        raise RuntimeError(f"need {target_real} real videos, found {len(real_pool)}")
    if len(fake_pool) < target_fake:
        raise RuntimeError(f"need {target_fake} fake videos, found {len(fake_pool)}")

    rng = random.Random(seed)
    picked_real = rng.sample(real_pool, target_real)
    picked_fake = rng.sample(fake_pool, target_fake)

    entries: list[dict] = []
    for i, src in enumerate(picked_real, start=1):
        safe = f"celebdf_real_{i:03d}{src.suffix.lower()}"
        dst = copy_sample(src, out_dir / "real", safe)
        entries.append(
            {
                "file": safe,
                "subdir": "real",
                "dataset": "celeb-df-v2",
                "label": "real",
                "source": str(src.relative_to(dataset_root)),
            }
        )

    for i, src in enumerate(picked_fake, start=1):
        safe = f"celebdf_fake_{i:03d}{src.suffix.lower()}"
        dst = copy_sample(src, out_dir / "fake", safe)
        entries.append(
            {
                "file": safe,
                "subdir": "fake",
                "dataset": "celeb-df-v2",
                "label": "fake",
                "source": str(src.relative_to(dataset_root)),
            }
        )

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"benchmark sample: real={target_real}, fake={target_fake} -> {out_dir}")
    return entries


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".", help="forenShield-ai root")
    p.add_argument(
        "--cache-dir",
        default=None,
        help="raw download/extract dir (default: <root>/data/raw/celeb-df-v2)",
    )
    p.add_argument("--url", default=None, help="download URL from Celeb-DF approval email")
    p.add_argument("--archive", type=Path, default=None, help="existing zip/tar path")
    p.add_argument(
        "--out-dir",
        default=None,
        help="benchmark subset dir (default: <root>/data/test/video/celeb-df-v2)",
    )
    p.add_argument("--sample-real", type=int, default=0, help="copy N real videos to test dir")
    p.add_argument("--sample-fake", type=int, default=0, help="copy N fake videos to test dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--extract-only",
        action="store_true",
        help="only download/extract full dataset; skip benchmark copy",
    )
    p.add_argument(
        "--source",
        choices=["url", "archive", "kaggle"],
        default="url",
        help="download source (default: url)",
    )
    p.add_argument(
        "--kaggle-dataset",
        default="reubensuju/celeb-df-v2",
        help="Kaggle dataset slug (default: reubensuju/celeb-df-v2)",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="re-download Kaggle archive even if zip already exists",
    )
    p.add_argument(
        "--reuse-extract",
        action="store_true",
        help="reuse data/raw/celeb-df-v2/_extract (skip zip extract)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    cache = Path(args.cache_dir) if args.cache_dir else root / "data/raw/celeb-df-v2"
    out_dir = Path(args.out_dir) if args.out_dir else root / "data/test/video/celeb-df-v2"

    cache.mkdir(parents=True, exist_ok=True)

    archive: Path | None
    pre_extracted: Path | None = None

    if args.source == "kaggle":
        archive = download_kaggle_dataset(
            args.kaggle_dataset,
            cache,
            force=args.force_download,
        )
        if archive is None:
            pre_extracted = find_dataset_root(cache)
    elif args.source == "archive":
        archive = args.archive if args.archive.is_absolute() else (root / args.archive).resolve()
        if not archive.is_file():
            raise FileNotFoundError(archive)
    elif args.url:
        archive = cache / "celebdf_v2_download.zip"
        download_url(args.url, archive)
    else:
        existing = sorted(cache.glob("*.zip")) + sorted(cache.glob("*.tar*"))
        if not existing:
            raise SystemExit(
                "provide --source kaggle, --url, or --archive\n"
                "kaggle: --source kaggle --kaggle-dataset reubensuju/celeb-df-v2\n"
                "official form: https://forms.gle/2jYBby6y1FBU3u6q9"
            )
        archive = max(existing, key=lambda p: p.stat().st_size)

    final_root = cache / "Celeb-DF-v2"
    if final_root.exists():
        shutil.rmtree(final_root)

    if pre_extracted is not None:
        if pre_extracted != final_root:
            shutil.move(str(pre_extracted), str(final_root))
    elif args.reuse_extract and (cache / "_extract").is_dir():
        staging = cache / "_extract"
        print("reuse extract:", staging)
        dataset_root = find_dataset_root(staging)
        if dataset_root != staging:
            if final_root.exists():
                shutil.rmtree(final_root)
            shutil.move(str(dataset_root), str(final_root))
        elif final_root.exists():
            shutil.rmtree(final_root)
            shutil.move(str(staging), str(final_root))
        else:
            shutil.move(str(staging), str(final_root))
    else:
        staging = cache / "_extract"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        print("extract:", archive)
        extract_archive(archive, staging)
        dataset_root = find_dataset_root(staging)
        shutil.move(str(dataset_root), str(final_root))

    real_pool, fake_pool = discover_video_pools(final_root)
    real_n = len(real_pool)
    fake_n = len(fake_pool)
    print(
        json.dumps(
            {
                "dataset_root": str(final_root),
                "real_videos": real_n,
                "fake_videos": fake_n,
                "archive": str(archive) if archive else None,
                "source": args.source,
                "kaggle_dataset": args.kaggle_dataset if args.source == "kaggle" else None,
            },
            indent=2,
        )
    )

    if args.extract_only or (args.sample_real == 0 and args.sample_fake == 0):
        print("extract-only: full dataset at", final_root)
        return

    sample_benchmark(
        final_root,
        out_dir,
        target_real=args.sample_real,
        target_fake=args.sample_fake,
        seed=args.seed,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
