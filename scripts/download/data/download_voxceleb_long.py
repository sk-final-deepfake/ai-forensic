#!/usr/bin/env python3
"""Download longer real clips from VoxCeleb YouTube sources.

VoxCeleb utterance metadata yields ~1s clips. This script downloads the
source YouTube video once per video_id and saves a longer segment.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import subprocess
import sys
from pathlib import Path


def manifest_entries(data) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("items", "videos", "entries"):
            val = data.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def unique_video_ids(meta_root: Path) -> list[str]:
    ids: set[str] = set()
    for txt in glob.glob(str(meta_root / "*" / "*" / "*.txt")):
        video_id = Path(txt).parent.name
        if len(video_id) >= 8:
            ids.add(video_id)
    return sorted(ids)


def video_id_from_filename(name: str) -> str | None:
    stem = Path(name).stem
    if stem.endswith("_long"):
        return stem[: -len("_long")]
    return None


def collect_excluded_video_ids(exclude_dirs: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for directory in exclude_dirs:
        if not directory.is_dir():
            continue
        for mp4 in directory.glob("*.mp4"):
            vid = video_id_from_filename(mp4.name)
            if vid:
                excluded.add(vid)
        manifest = directory / "manifest.json"
        if manifest.is_file():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for entry in manifest_entries(data):
                vid = entry.get("video_id")
                if isinstance(vid, str) and vid:
                    excluded.add(vid)
    return excluded


def load_existing_manifest(out_dir: Path) -> list[dict]:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        return []
    return manifest_entries(json.loads(manifest_path.read_text(encoding="utf-8")))


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


def download_youtube(video_id: str, out_path: Path) -> bool:
    url = f"https://www.youtube.com/watch?v={video_id}"
    r = subprocess.run(
        ["yt-dlp", "-f", "mp4/best", "-o", str(out_path), url],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


def trim_clip(src: Path, dst: Path, start: float, duration: float) -> bool:
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(src),
            "-t",
            str(duration),
            "-c",
            "copy",
            str(dst),
            "-loglevel",
            "error",
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0 and dst.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VoxCeleb long real clips")
    parser.add_argument("--meta-root", default="data/raw/voxceleb/txt")
    parser.add_argument("--out-dir", default="data/test/video/voxceleb/real")
    parser.add_argument("--tmp-dir", default="data/raw/voxceleb/tmp_full")
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="benchmark dirs; their video_id values are skipped (repeatable)",
    )
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--min-sec", type=float, default=15.0, help="minimum saved clip length")
    parser.add_argument("--max-sec", type=float, default=240.0, help="max clip length (4 min)")
    parser.add_argument("--start-sec", type=float, default=5.0, help="skip intro seconds")
    parser.add_argument("--seed", type=int, default=42, help="shuffle order of candidate video_ids")
    args = parser.parse_args()

    meta_root = Path(args.meta_root)
    out_dir = Path(args.out_dir)
    tmp_dir = Path(args.tmp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    exclude_dirs = [Path(p) for p in args.exclude_dir]
    excluded_ids = collect_excluded_video_ids(exclude_dirs)
    manifest = load_existing_manifest(out_dir)
    existing_ids = {entry["video_id"] for entry in manifest if entry.get("video_id")}

    video_ids = unique_video_ids(meta_root)
    if not video_ids:
        print("No metadata under", meta_root)
        sys.exit(1)

    candidates = [
        vid
        for vid in video_ids
        if vid not in excluded_ids and vid not in existing_ids
    ]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    print(f"meta video_ids: {len(video_ids)}")
    print(f"excluded (benchmark): {len(excluded_ids)}")
    print(f"already in out-dir manifest: {len(existing_ids)}")
    print(f"candidates: {len(candidates)}")
    print(f"target new clips: {args.target}")
    print(f"out: {out_dir.resolve()}")
    print()

    ok = sum(
        1
        for entry in manifest
        if (out_dir / entry["file"]).is_file()
        and probe_duration(out_dir / entry["file"]) >= args.min_sec
    )

    for video_id in candidates:
        if ok >= args.target:
            break

        out_mp4 = out_dir / f"{video_id}_long.mp4"
        if out_mp4.exists() and probe_duration(out_mp4) >= args.min_sec:
            if not any(e.get("video_id") == video_id for e in manifest):
                manifest.append(
                    {
                        "file": out_mp4.name,
                        "video_id": video_id,
                        "clip_duration_sec": round(probe_duration(out_mp4), 2),
                        "label": "real",
                        "dataset_source": "VoxCeleb",
                    }
                )
            ok += 1
            continue

        raw_mp4 = tmp_dir / f"{video_id}.mp4"
        if not raw_mp4.exists():
            print("download:", video_id, flush=True)
            if not download_youtube(video_id, raw_mp4):
                print("  skip (download failed):", video_id)
                continue

        full_dur = probe_duration(raw_mp4)
        if full_dur < args.min_sec:
            print(f"  skip (too short {full_dur:.1f}s):", video_id)
            continue

        start = min(args.start_sec, max(0.0, full_dur - args.min_sec))
        clip_len = min(args.max_sec, full_dur - start)
        if clip_len < args.min_sec:
            print(f"  skip (clip too short):", video_id)
            continue

        print(f"  trim {clip_len:.1f}s from {video_id} (full {full_dur:.1f}s)", flush=True)
        if trim_clip(raw_mp4, out_mp4, start, clip_len):
            ok += 1
            manifest.append(
                {
                    "file": out_mp4.name,
                    "video_id": video_id,
                    "full_duration_sec": round(full_dur, 2),
                    "clip_start_sec": round(start, 2),
                    "clip_duration_sec": round(clip_len, 2),
                    "label": "real",
                    "dataset_source": "VoxCeleb",
                }
            )
            print(f"  [{ok}/{args.target}] saved {out_mp4.name}", flush=True)
        else:
            print("  skip (trim failed):", video_id)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print()
    print("done:", ok, "clips in", out_dir)
    print("manifest:", manifest_path.resolve())

    if ok < args.target:
        print(f"WARN: only {ok}/{args.target} clips collected", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
