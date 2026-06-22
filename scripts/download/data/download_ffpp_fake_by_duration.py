#!/usr/bin/env python3
"""Download and sample fake FaceForensics++ videos by duration.

Uses the official FaceForensics download script (email approval required).
Prioritizes DeepFakeDetection manipulated videos (longer actor scenes).

Default: 50 fake mp4 clips, 120s–240s (2–4 min).

Default: 50 fake mp4 clips, 120s–240s (2–4 min) from DeepFakeDetection.

Official script (from approval email) — save as e.g. download-FaceForensics.py:

  python download-FaceForensics.py <output_path> \\
    -d DeepFakeDetection -c c40 -t videos --server EU2

Our wrapper (download + sample):

  python3 scripts/download/data/download_ffpp_fake_by_duration.py \\
    --download-script data/raw/faceforensics/download-FaceForensics.py \\
    --ff-root data/raw/faceforensics \\
    --out-dir data/test/video/ffpp/fake
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

# 2–4 min fakes: DeepFakeDetection actor scenes (FF++ youtube clips are ~10s).
DEFAULT_DATASETS = ("DeepFakeDetection",)
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return 0.0
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def run_download(
    script: Path,
    ff_root: Path,
    dataset: str,
    compression: str,
    server: str,
    num_videos: int | None,
) -> None:
    ff_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        str(ff_root),
        "-d",
        dataset,
        "-c",
        compression,
        "-t",
        "videos",
        "--server",
        server,
    ]
    if num_videos is not None and num_videos > 0:
        cmd.extend(["-n", str(num_videos)])
    print("+", " ".join(cmd), flush=True)
    print("(auto-accepting FaceForensics TOS prompt)", flush=True)
    # Official script blocks on input() for terms of use.
    r = subprocess.run(cmd, input="\n", text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FaceForensics download failed ({r.returncode}): {dataset}")


def manipulation_method(path: Path, ff_root: Path) -> str:
    try:
        rel = path.relative_to(ff_root / "manipulated_sequences")
        return rel.parts[0] if rel.parts else "unknown"
    except ValueError:
        return "unknown"

def iter_fake_videos(ff_root: Path, compression: str, datasets: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    manip_root = ff_root / "manipulated_sequences"
    for name in datasets:
        for base in (
            manip_root / name / compression / "videos",
            manip_root / name / compression,
        ):
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*")):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                    found.append(path)
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def load_manifest_counts(out_dir: Path) -> tuple[list[dict], int]:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        return [], 0
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], 0
    ok = 0
    for entry in manifest:
        file_name = entry.get("file")
        if file_name and (out_dir / file_name).is_file():
            ok += 1
    return manifest, ok


def write_manifest(out_dir: Path, manifest: list[dict]) -> Path:
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample fake FF++ videos by duration.")
    parser.add_argument(
        "--download-script",
        required=True,
        help="path to author download-FaceForensics.py (from approval email)",
    )
    parser.add_argument("--ff-root", default="data/raw/faceforensics", help="FF++ download root")
    parser.add_argument("--out-dir", default="data/test/video/ffpp/fake", help="sample output dir")
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--min-sec", type=float, default=120.0, help="minimum duration (default 2 min)")
    parser.add_argument("--max-sec", type=float, default=240.0, help="maximum duration (default 4 min)")
    parser.add_argument("--compression", default="c40", help="c40 (small), c23, or raw")
    parser.add_argument("--server", default="EU2", help="FF server: EU2 (recommended)")
    parser.add_argument(
        "--num-videos",
        type=int,
        default=None,
        help="passed to official script -n (default: full subset)",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="manipulation subsets to download/scan, in order",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="only scan --ff-root (no download)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="download datasets then exit (no sampling)",
    )
    args = parser.parse_args()

    script = Path(args.download_script)
    if not script.is_file():
        print(f"ERROR: download script not found: {script}", file=sys.stderr)
        print("  Place the script from the FaceForensics approval email on the server.", file=sys.stderr)
        sys.exit(1)

    ff_root = Path(args.ff_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = tuple(s.strip() for s in args.datasets.split(",") if s.strip())
    manifest, already = load_manifest_counts(out_dir)
    if already >= args.target and not args.download_only:
        print(f"already complete: {already}/{args.target} in {out_dir}")
        return

    if not args.skip_download:
        for dataset in datasets:
            print(f"downloading: {dataset} ({args.compression}) ...", flush=True)
            try:
                run_download(
                    script,
                    ff_root,
                    dataset,
                    args.compression,
                    args.server,
                    args.num_videos,
                )
            except RuntimeError as exc:
                print(f"WARN: {exc}", file=sys.stderr)
        if args.download_only:
            print("download-only done:", ff_root.resolve())
            return

    candidates = iter_fake_videos(ff_root, args.compression, datasets)
    if not candidates:
        print(
            "ERROR: no manipulated videos found under",
            ff_root / "manipulated_sequences",
            file=sys.stderr,
        )
        print(
            "  Expected e.g.: manipulated_sequences/DeepFakeDetection/c40/videos/*.mp4",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"scanning {len(candidates)} fake videos for {args.min_sec:.0f}-{args.max_sec:.0f}s ...", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    matched: list[tuple[Path, float, str]] = []
    for i, path in enumerate(candidates, start=1):
        if i % 100 == 0:
            print(f"  probed {i}/{len(candidates)}, matched {len(matched)}", flush=True)
        duration = probe_duration(path)
        if args.min_sec <= duration <= args.max_sec:
            matched.append((path, duration, manipulation_method(path, ff_root)))

    if len(matched) < args.target:
        print(
            f"WARN: only {len(matched)} videos in {args.min_sec:.0f}-{args.max_sec:.0f}s "
            f"(need {args.target}).",
            file=sys.stderr,
        )
        print(
            "  FF++ youtube clips are often ~10s. DeepFakeDetection has longer scenes; "
            "ensure that subset downloaded.",
            file=sys.stderr,
        )

    rng.shuffle(matched)
    selected = matched[: args.target]

    saved = already
    for idx, (src, duration, method) in enumerate(selected, start=already + 1):
        if saved >= args.target:
            break
        stem = src.stem.replace(" ", "_")[:64]
        out_name = f"fake_ffpp_{idx:03d}_{method}_{stem}{src.suffix.lower()}"
        dst = out_dir / out_name
        if not dst.exists():
            shutil.copy2(src, dst)
        manifest.append(
            {
                "file": out_name,
                "duration_sec": round(duration, 2),
                "dataset_source": "FaceForensics++",
                "manipulation": method,
                "source_path": str(src.resolve()),
                "label": "fake",
            }
        )
        saved += 1
        print(f"[{saved}/{args.target}] {duration:.1f}s {method} -> {out_name}", flush=True)

    manifest_path = write_manifest(out_dir, manifest)
    print()
    print("done:", saved, "/", args.target)
    print("manifest:", manifest_path.resolve())

    if saved < args.target:
        sys.exit(2)


if __name__ == "__main__":
    main()
