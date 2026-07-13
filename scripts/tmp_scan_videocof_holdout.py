#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import json

VIDEO = {".mp4", ".webm", ".mov", ".avi", ".mkv"}


def scan(root: str):
    root_p = Path(root)
    if not root_p.exists():
        return 0, {}, set()
    vids = [p for p in root_p.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO]
    c = Counter()
    names = set()
    for p in vids:
        names.add(p.name)
        parts = p.relative_to(root_p).parts
        if parts[0] == "original":
            c["original"] += 1
        elif parts[0] == "tampered":
            key = "/".join(parts[:3]) if len(parts) >= 3 else parts[0]
            c[key] += 1
        else:
            c[parts[0]] += 1
    return len(vids), dict(c), names


def main() -> None:
    roots = [
        "data/test/video/spatial-videocof-benchmark",
        "data/test/video/spatial-videocof",
        "data/train/video/spatial-videocof",
        "data/train/video/spatial-videocof-benchmark",
    ]
    for r in roots:
        n, c, _ = scan(r)
        print(f"\n## {r} total={n}")
        for k, v in sorted(c.items()):
            print(f"  {k}: {v}")

    pred_path = Path(
        "results/infer/trufor-videocof-v2-official-test400-f16-align-top3/predictions.json"
    )
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    test_files = {x["file"] for x in pred["items"]}
    _, _, train_names = scan("data/train/video/spatial-videocof")
    _, _, train_bench = scan("data/train/video/spatial-videocof-benchmark")
    print("\n## overlap")
    print("test400 files:", len(test_files))
    print("overlap train/spatial-videocof:", len(test_files & train_names))
    print("overlap train/spatial-videocof-benchmark:", len(test_files & train_bench))
    print(
        "meta:",
        {
            k: pred.get(k)
            for k in [
                "run_id",
                "data_root",
                "num_frames",
                "threshold",
                "aggregate",
                "align_pairs",
                "sample_fps",
            ]
        },
    )
    print("keys:", list(pred.keys()))


if __name__ == "__main__":
    main()
