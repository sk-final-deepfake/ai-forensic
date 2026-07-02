#!/usr/bin/env python3
"""List candidate MVTamperBench pools on GPU (video counts, real/fake)."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from trufor_video_common import VIDEO_SUFFIXES, iter_videos


def classify_rel(rel: str) -> str:
    parts = Path(rel).parts
    if parts and parts[0] == "original":
        return "real"
    if parts and parts[0] == "tampered":
        return "fake"
    low = rel.lower()
    if "original" in low or low.startswith("original_"):
        return "real"
    return "fake"


def scan_root(root: Path) -> tuple[int, int, int]:
    videos: list[str] = []
    for split in ("original", "tampered"):
        for p in iter_videos(root, split):
            videos.append(p.relative_to(root).as_posix())
    if not videos:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
                videos.append(p.relative_to(root).as_posix())
    c = Counter(classify_rel(v) for v in videos)
    return len(videos), c.get("real", 0), c.get("fake", 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[
            "data/pull/evidence",
            "data/train/video",
            "data/raw",
        ],
    )
    args = parser.parse_args()

    print("Scanning MVTamperBench / video pools...\n")
    rows: list[tuple[int, str, int, int, int]] = []
    for base in args.roots:
        base_p = Path(base)
        if not base_p.exists():
            continue
        # self if has original/ or tampered/
        if (base_p / "original").exists() or (base_p / "tampered").exists():
            t, r, f = scan_root(base_p)
            rows.append((t, str(base_p), t, r, f))
        for child in sorted(base_p.iterdir()):
            if not child.is_dir():
                continue
            if (child / "original").exists() or (child / "tampered").exists():
                t, r, f = scan_root(child)
                rows.append((t, str(child), t, r, f))

    rows.sort(reverse=True)
    print(f"{'total':>6}  {'real':>5}  {'fake':>5}  path")
    print("-" * 70)
    for t, path, _, r, f in rows[:30]:
        mark = " <-- pool candidate" if t >= 800 else (" (200 only)" if t <= 220 else "")
        print(f"{t:6d}  {r:5d}  {f:5d}  {path}{mark}")

    if not rows:
        print("No pools found. Pull full MVTamperBench to data/pull/evidence/ first.")
    else:
        best = rows[0]
        if best[0] < 500:
            print(
                f"\nWARN: largest pool has {best[0]} videos — need ~800+ for +300 new (250r+250f) after reserving 200."
            )
        else:
            print(f"\nSuggested --pool-root: {best[1]}")


if __name__ == "__main__":
    main()
