#!/usr/bin/env python3
"""Download 50-sample subsets for video deepfake benchmarks.

Outputs under data/test/video/{wilddeepfake,forgerynet,fakeavceleb}/ with manifest.json.

Recommended (no Chinese-site signup):
  pip install gdown huggingface_hub
  --datasets wilddeepfake,fakeavceleb

ForgeryNet: official Google Drive links are often dead. OpenXLab works but requires
a Chinese-platform account. Use wilddeepfake as a drop-in alternative, or pass
--forgerynet-tar if you obtain the tar elsewhere.

FakeAVCeleb: public sample folder has only a few demo clips. For 50 videos,
use --fakeavceleb-root pointing at the full dataset after the request form.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Maintainer-shared links (often broken; see yinanhe/ForgeryNet#11)
FORGERYNET_GDRIVE_CANDIDATES = [
    ("Validation.tar", "1RgVa9n9yJDsMZvO6cAsKJH4YtNP9VeNW"),
    ("public_test_videos.tar", None),  # filled from openxlab if needed
]
FORGERYNET_OPENXLAB_REPO = "OpenDataLab/ForgeryNet"
FORGERYNET_OPENXLAB_FILES = [
    "/public_test_videos.tar",
    "/Validation.tar",
    "public_test_videos.tar",
    "Validation.tar",
]

FAKEAVCELEB_SAMPLE_FOLDER_ID = "1SYMs44Z1W7rlrn0W7t-4LcPPusiBINEB"

WILDDEEPFAKE_REPO = "xingjunm/WildDeepfake"
WILDDEEPFAKE_SHARDS = [
    f"deepfake_in_the_wild/fake_train/{i}.tar.gz" for i in range(6)
]


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")


def write_manifest(out_dir: Path, entries: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return path


def copy_sample(src: Path, dst_dir: Path, name: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / name
    if dst.exists():
        return dst
    shutil.copy2(src, dst)
    return dst


def iter_videos(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def pick_videos(paths: list[Path], target: int, seed: int) -> list[Path]:
    if len(paths) <= target:
        return paths
    rng = random.Random(seed)
    return sorted(rng.sample(paths, target))


def gdown_file(file_id: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        file_id,
        f"https://drive.google.com/uc?id={file_id}",
        f"https://drive.google.com/file/d/{file_id}/view",
    ]
    for url in urls:
        print(f"try gdown: {url}")
        r = subprocess.run(["gdown", url, "-O", str(dest)])
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return True
    return False


def gdown_folder(folder_id: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    last_err: str | None = None
    for attempt in range(1, 4):
        print(f"try gdown folder (attempt {attempt}/3): {url}")
        r = subprocess.run(["gdown", "--folder", url, "-O", str(dest)])
        if r.returncode == 0 and any(dest.iterdir()):
            return
        last_err = f"gdown folder failed (exit {r.returncode})"
    raise RuntimeError(
        f"{last_err}. Google Drive sample folder is unreliable (HTTP 500/quota).\n"
        "FakeAVCeleb 50 videos need form approval: https://bit.ly/38prlVO\n"
        "After approval, pass --fakeavceleb-root /path/to/extracted/dataset"
    )


def download_forgerynet_tar_openxlab(cache_dir: Path, dest: Path) -> bool:
    if shutil.which("openxlab") is None:
        return False
    for source in FORGERYNET_OPENXLAB_FILES:
        print(f"try openxlab: {FORGERYNET_OPENXLAB_REPO} {source}")
        r = subprocess.run(
            [
                "openxlab",
                "dataset",
                "download",
                "--dataset-repo",
                FORGERYNET_OPENXLAB_REPO,
                "--source-path",
                source,
                "--target-path",
                str(cache_dir),
            ]
        )
        if r.returncode != 0:
            continue
        candidates = list(cache_dir.glob("*.tar"))
        if dest.exists() and dest.stat().st_size > 0:
            return True
        if candidates:
            shutil.move(str(max(candidates, key=lambda p: p.stat().st_size)), str(dest))
            return dest.exists()
    return False


def acquire_forgerynet_tar(cache_dir: Path, tar_file: Path, source: str) -> None:
    if tar_file.exists() and tar_file.stat().st_size > 0:
        return

    if source in {"auto", "gdrive"}:
        for name, file_id in FORGERYNET_GDRIVE_CANDIDATES:
            if not file_id:
                continue
            print(f"Downloading ForgeryNet {name} from Google Drive...")
            if gdown_file(file_id, tar_file):
                return

    if source in {"auto", "openxlab"}:
        print("Trying OpenXLab mirror (requires: pip install openxlab && openxlab login)...")
        if download_forgerynet_tar_openxlab(cache_dir, tar_file):
            return

    raise RuntimeError(
        "ForgeryNet download failed. Official Google Drive links are often dead.\n"
        "Without OpenXLab signup, use WildDeepfake instead:\n"
        "  --datasets wilddeepfake\n"
        "Or pass a manually obtained tar: --forgerynet-tar /path/to/Validation.tar"
    )


def iter_image_sequences(root: Path, min_frames: int = 16) -> list[tuple[Path, list[Path]]]:
    """WildDeepfake stores face frame folders, not mp4 files."""
    sequences: list[tuple[Path, list[Path]]] = []
    seen: set[Path] = set()
    for img in root.rglob("*"):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        parent = img.parent
        if parent in seen:
            continue
        frames = sorted(
            p for p in parent.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if len(frames) >= min_frames:
            sequences.append((parent, frames))
            seen.add(parent)
    return sequences


def sequence_to_mp4(frames: list[Path], out_mp4: Path, fps: int = 8) -> bool:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_mp4.parent / f".tmp_{out_mp4.stem}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    ext = frames[0].suffix.lower()
    for i, src in enumerate(frames):
        shutil.copy2(src, tmp / f"frame_{i:05d}{ext}")
    pattern = str(tmp / f"frame_%05d{ext}")
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-frames:v",
            str(len(frames)),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
            "-loglevel",
            "error",
        ]
    )
    shutil.rmtree(tmp, ignore_errors=True)
    return r.returncode == 0 and out_mp4.exists() and out_mp4.stat().st_size > 0


def ensure_wilddeepfake_shards(cache_dir: Path, min_sequences: int) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("pip install huggingface_hub") from exc

    staging = cache_dir / "extract"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    for shard_name in WILDDEEPFAKE_SHARDS:
        shard_path = cache_dir / shard_name
        if not shard_path.exists() or shard_path.stat().st_size == 0:
            print(f"Downloading WildDeepfake shard: {shard_name}")
            try:
                downloaded = hf_hub_download(
                    repo_id=WILDDEEPFAKE_REPO,
                    repo_type="dataset",
                    filename=shard_name,
                    local_dir=str(cache_dir),
                )
                shard_path = Path(downloaded)
            except Exception as exc:
                print(f"skip shard {shard_name}: {exc}")
                continue

        print("extract:", shard_path)
        with tarfile.open(shard_path, "r:*") as tf:
            tf.extractall(staging)

        if len(iter_image_sequences(staging)) >= min_sequences:
            break

    return staging


def download_wilddeepfake(
    *,
    cache_dir: Path,
    out_dir: Path,
    target: int,
    seed: int,
) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    staging = ensure_wilddeepfake_shards(cache_dir, target)

    sequences = iter_image_sequences(staging)
    if len(sequences) < target:
        raise RuntimeError(
            f"WildDeepfake: wanted {target} face sequences, found {len(sequences)}. "
            "Dataset stores image frames, not mp4; need more shards or lower --target."
        )

    rng = random.Random(seed)
    picked = sequences if len(sequences) <= target else rng.sample(sequences, target)

    entries: list[dict] = []
    for i, (seq_dir, frames) in enumerate(picked, start=1):
        out_mp4 = out_dir / f"wilddeepfake_{i:03d}.mp4"
        if not out_mp4.exists():
            print(f"  build mp4 from {len(frames)} frames: {seq_dir.name}")
            if not sequence_to_mp4(frames, out_mp4):
                print(f"  skip (ffmpeg failed): {seq_dir}")
                continue
        entries.append(
            {
                "file": out_mp4.name,
                "dataset": "wilddeepfake",
                "label": "fake",
                "source": str(seq_dir.relative_to(staging)),
                "frame_count": len(frames),
            }
        )

    if len(entries) < target:
        raise RuntimeError(f"WildDeepfake: wanted {target} mp4 clips, built {len(entries)}")

    manifest = write_manifest(out_dir, entries)
    print(f"WildDeepfake: {len(entries)} videos -> {out_dir} ({manifest})")
    return entries


def extract_tar_members(tar_path: Path, members: list[str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        for name in members:
            member = tf.getmember(name)
            tf.extract(member, path=dest)


def download_forgerynet(
    *,
    cache_dir: Path,
    out_dir: Path,
    target: int,
    seed: int,
    tar_path: Path | None,
    source: str,
) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_file = tar_path or (cache_dir / "Validation.tar")
    acquire_forgerynet_tar(cache_dir, tar_file, source)

    with tarfile.open(tar_file, "r:*") as tf:
        video_members = [
            m.name
            for m in tf.getmembers()
            if m.isfile() and Path(m.name).suffix.lower() in VIDEO_EXTS
        ]

    if not video_members:
        raise RuntimeError(f"no video files found in {tar_file}")

    picked = pick_videos([Path(p) for p in video_members], target, seed)
    picked_names = [str(p) for p in picked]

    staging = cache_dir / "forgerynet_extract"
    if staging.exists():
        shutil.rmtree(staging)
    extract_tar_members(tar_file, picked_names, staging)

    entries: list[dict] = []
    extracted = iter_videos(staging)
    for i, src in enumerate(extracted[:target], start=1):
        safe = f"forgerynet_{i:03d}{src.suffix.lower()}"
        dst = copy_sample(src, out_dir, safe)
        entries.append(
            {
                "file": dst.name,
                "dataset": "forgerynet",
                "label": "fake",
                "source": str(src.relative_to(staging)),
            }
        )

    if len(entries) < target:
        raise RuntimeError(f"ForgeryNet: wanted {target}, got {len(entries)}")

    manifest = write_manifest(out_dir, entries)
    print(f"ForgeryNet: {len(entries)} videos -> {out_dir} ({manifest})")
    return entries


def download_fakeavceleb(
    *,
    cache_dir: Path,
    out_dir: Path,
    target: int,
    seed: int,
    full_root: Path | None,
    fake_only: bool,
) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    search_root: Path

    if full_root is not None:
        if not full_root.exists():
            raise FileNotFoundError(f"--fakeavceleb-root not found: {full_root}")
        search_root = full_root
        use_fake_filter = fake_only
    else:
        sample_dir = cache_dir / "fakeavceleb_samples"
        if not sample_dir.exists() or not any(sample_dir.iterdir()):
            print("Downloading FakeAVCeleb public sample folder...")
            gdown_folder(FAKEAVCELEB_SAMPLE_FOLDER_ID, sample_dir)
        search_root = sample_dir
        use_fake_filter = False

    all_videos = iter_videos(search_root)
    if use_fake_filter:
        all_videos = [
            p
            for p in all_videos
            if "fake" in p.as_posix().lower() or "Fake" in p.name
        ]

    if len(all_videos) < target:
        msg = (
            f"FakeAVCeleb: only {len(all_videos)} videos under {search_root}; "
            f"need {target}. Fill the request form "
            "(https://bit.ly/38prlVO) and pass --fakeavceleb-root to the extracted folder."
        )
        if len(all_videos) == 0:
            raise RuntimeError(msg)
        print("WARNING:", msg)

    picked = pick_videos(all_videos, min(target, len(all_videos)), seed)

    entries: list[dict] = []
    for i, src in enumerate(picked, start=1):
        safe = f"fakeavceleb_{i:03d}{src.suffix.lower()}"
        dst = copy_sample(src, out_dir, safe)
        label = "fake"
        if "real" in src.as_posix().lower():
            label = "real"
        entries.append(
            {
                "file": dst.name,
                "dataset": "fakeavceleb",
                "label": label,
                "source": str(src.relative_to(search_root)),
            }
        )

    manifest = write_manifest(out_dir, entries)
    print(f"FakeAVCeleb: {len(entries)} videos -> {out_dir} ({manifest})")
    return entries


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default=".",
        help="forenShield-ai root (default: cwd)",
    )
    p.add_argument("--target", type=int, default=50, help="videos per dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--datasets",
        default="wilddeepfake,fakeavceleb",
        help="comma-separated: wilddeepfake, forgerynet, fakeavceleb",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="raw download cache (default: <root>/data/raw/benchmark-downloads)",
    )
    p.add_argument(
        "--forgerynet-tar",
        type=Path,
        default=None,
        help="existing ForgeryNet tar path (skip download)",
    )
    p.add_argument(
        "--forgerynet-source",
        choices=["auto", "gdrive", "openxlab"],
        default="auto",
        help="ForgeryNet download source (default: auto)",
    )
    p.add_argument(
        "--skip-on-error",
        action="store_true",
        help="continue remaining datasets if one fails",
    )
    p.add_argument(
        "--fakeavceleb-root",
        type=Path,
        default=None,
        help="extracted full FakeAVCeleb dataset root (after form approval)",
    )
    p.add_argument(
        "--fakeavceleb-include-real",
        action="store_true",
        help="FakeAVCeleb: include real videos when sampling from full dataset",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    cache = Path(args.cache_dir) if args.cache_dir else root / "data/raw/benchmark-downloads"
    base_out = root / "data/test/video"
    datasets = {d.strip().lower() for d in args.datasets.split(",") if d.strip()}

    unknown = datasets - {"forgerynet", "fakeavceleb", "wilddeepfake"}
    if unknown:
        raise SystemExit(f"unknown datasets: {', '.join(sorted(unknown))}")

    summary: dict[str, int] = {}
    errors: list[str] = []

    if "wilddeepfake" in datasets:
        try:
            download_wilddeepfake(
                cache_dir=cache / "wilddeepfake",
                out_dir=base_out / "wilddeepfake",
                target=args.target,
                seed=args.seed,
            )
            summary["wilddeepfake"] = args.target
        except Exception as exc:
            errors.append(f"wilddeepfake: {exc}")
            print("ERROR [wilddeepfake]:", exc, file=sys.stderr)
            if not args.skip_on_error:
                raise

    if "forgerynet" in datasets:
        try:
            download_forgerynet(
                cache_dir=cache / "forgerynet",
                out_dir=base_out / "forgerynet",
                target=args.target,
                seed=args.seed,
                tar_path=args.forgerynet_tar,
                source=args.forgerynet_source,
            )
            summary["forgerynet"] = args.target
        except Exception as exc:
            errors.append(f"forgerynet: {exc}")
            print("ERROR [forgerynet]:", exc, file=sys.stderr)
            if not args.skip_on_error:
                raise

    if "fakeavceleb" in datasets:
        try:
            fake_only = not args.fakeavceleb_include_real
            download_fakeavceleb(
                cache_dir=cache / "fakeavceleb",
                out_dir=base_out / "fakeavceleb",
                target=args.target,
                seed=args.seed,
                full_root=args.fakeavceleb_root,
                fake_only=fake_only,
            )
            summary["fakeavceleb"] = args.target
        except Exception as exc:
            errors.append(f"fakeavceleb: {exc}")
            print("ERROR [fakeavceleb]:", exc, file=sys.stderr)
            if not args.skip_on_error:
                raise

    print("done:", json.dumps(summary, indent=2))
    if errors:
        print("failures:", "; ".join(errors), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
