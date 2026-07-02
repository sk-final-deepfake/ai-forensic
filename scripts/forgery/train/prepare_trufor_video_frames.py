#!/usr/bin/env python3
"""Extract frames + weak masks from forgery-gmflow-train-400 for TruFor fine-tune.

Output layout (under --out-dir):
  frames/<cache_name>.jpg
  masks/<cache_name>.png   # 0=negative, 255=in-window tampered (recipe v2)
  train_list.txt           # rgb_path,mask_path per line (relative to out-dir)
  valid_list.txt
  meta.json

Recipe v2 (default): spatial + middle_tampered only; in-window mask=255,
out-of-window fake frames mask=0 (hard negative). Non-middle spatial videos skipped.

Recipe v4 (--skip-out-of-window-fake): in-window tampered frames only; out-of-window
frames on fake videos are omitted from train/valid lists (no hard negative).

Recipe r3 (Baseline-line, cause #2): v2 layout (hard negatives kept) + duplicate each
in-window positive (label=1) line in train_list.txt (--oversample-positive N; default 3).
Independent from R2 (cause #3 skip-OOW).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from trufor_video_common import (
    DEFAULT_SPATIAL_TAMPER_TYPES,
    build_frame_plans,
    extract_frame_bgr,
    frame_cache_name,
    save_frame_jpg,
    write_mask_png,
)


def group_by_video(plans):
    grouped: dict[str, list] = defaultdict(list)
    for plan in plans:
        key = str(plan.video_path)
        grouped[key].append(plan)
    return grouped


def split_videos(
    video_keys: list[str],
    valid_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    keys = list(video_keys)
    rng.shuffle(keys)
    n_valid = max(1, int(round(len(keys) * valid_ratio)))
    valid = set(keys[:n_valid])
    train = set(keys[n_valid:])
    if not train:
        train = valid
    return train, valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TruFor frame cache from video forgery train set")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/train/video/forgery-gmflow-train-400"),
        help="Input dataset root (original/ + tampered/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/trufor-gmflow-train-400"),
        help="Output cache directory",
    )
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--spatial-types",
        nargs="*",
        default=sorted(DEFAULT_SPATIAL_TAMPER_TYPES),
        help="Tamper folder names to include as fake (default: masking substitution rotate)",
    )
    parser.add_argument(
        "--include-temporal-fakes",
        action="store_true",
        help="Also include frame-deletion/dropping/etc. (not recommended for TruFor)",
    )
    parser.add_argument(
        "--require-middle-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip spatial tampered videos without middle_tampered in filename (recipe v2, default on)",
    )
    parser.add_argument(
        "--skip-out-of-window-fake",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Omit out-of-window frames on middle-tampered fakes (no mask=0 hard neg)",
    )
    parser.add_argument(
        "--recipe-tag",
        type=str,
        default=None,
        help="meta.json recipe field (e.g. r2, r3 for Baseline-line; default auto from flags)",
    )
    parser.add_argument(
        "--oversample-positive",
        type=int,
        default=1,
        help="Repeat each in-window positive (label=1) train line N times total (1=no-op, 3=2 extra copies)",
    )
    parser.add_argument("--max-items", type=int, default=0, help="Debug cap (0=all)")
    args = parser.parse_args()
    if args.oversample_positive < 1:
        parser.error("--oversample-positive must be >= 1")

    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    spatial_types = frozenset(args.spatial_types)

    plans = build_frame_plans(
        data_root,
        frames_per_video=args.frames_per_video,
        spatial_tamper_types=spatial_types,
        include_temporal_fakes=args.include_temporal_fakes,
        require_middle_tamper_window=args.require_middle_window,
        skip_out_of_window_fake=args.skip_out_of_window_fake,
    )
    if args.max_items > 0:
        plans = plans[: args.max_items]

    grouped = group_by_video(plans)
    train_videos, valid_videos = split_videos(list(grouped.keys()), args.valid_ratio, args.seed)

    frames_dir = out_dir / "frames"
    masks_dir = out_dir / "masks"
    train_lines: list[str] = []
    valid_lines: list[str] = []
    train_positive_lines = 0
    train_positive_copies = 0
    label_counter = Counter()
    skipped = 0

    for plan in plans:
        video_key = str(plan.video_path)
        split = "valid" if video_key in valid_videos else "train"
        cache = frame_cache_name(plan.video_path, data_root, plan.frame_index)
        frame_path = frames_dir / f"{cache}.jpg"
        mask_path = masks_dir / f"{cache}.png"

        if not frame_path.exists():
            bgr = extract_frame_bgr(plan.video_path, plan.frame_index)
            if bgr is None:
                skipped += 1
                continue
            h, w = bgr.shape[:2]
            save_frame_jpg(frame_path, bgr)
            write_mask_png(mask_path, plan.label, w, h)
        elif not mask_path.exists():
            import cv2

            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                skipped += 1
                continue
            h, w = bgr.shape[:2]
            write_mask_png(mask_path, plan.label, w, h)

        rel_rgb = frame_path.relative_to(out_dir).as_posix()
        rel_mask = mask_path.relative_to(out_dir).as_posix()
        line = f"{rel_rgb},{rel_mask}"
        if split == "valid":
            valid_lines.append(line)
        else:
            train_lines.append(line)
            if plan.label == 1:
                train_positive_lines += 1
                extra = args.oversample_positive - 1
                if extra > 0:
                    train_lines.extend([line] * extra)
                    train_positive_copies += extra
        label_counter[plan.label] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_list.txt").write_text("\n".join(train_lines) + ("\n" if train_lines else ""), encoding="utf-8")
    (out_dir / "valid_list.txt").write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""), encoding="utf-8")

    if args.recipe_tag:
        recipe = args.recipe_tag
        if args.skip_out_of_window_fake:
            note = (
                f"{recipe}: in-window tampered frames only; "
                "out-of-window fake frames omitted; all original frames mask=0"
            )
        elif args.require_middle_window:
            note = (
                f"{recipe}: v2 layout — mask=255 in-window, OOW fake mask=0 (hard neg); "
                "all original frames mask=0"
            )
        else:
            note = f"{recipe}: custom prepare flags"
        if args.oversample_positive > 1:
            note += f"; train in-window positive oversample x{args.oversample_positive}"
    elif args.skip_out_of_window_fake:
        recipe = "v4"
        note = (
            "v4: in-window tampered frames only (mask=255); "
            "out-of-window fake frames omitted; all original frames mask=0"
        )
    elif args.require_middle_window:
        recipe = "v2"
        note = (
            "v2: mask=255 only for in-window tampered frames; "
            "out-of-window fake frames and all real frames mask=0"
        )
    else:
        recipe = "v1"
        note = "v1: weak full-frame positive on spatial tampered videos"

    meta = {
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "recipe": recipe,
        "frames_per_video": args.frames_per_video,
        "spatial_types": sorted(spatial_types),
        "include_temporal_fakes": args.include_temporal_fakes,
        "require_middle_tamper_window": args.require_middle_window,
        "skip_out_of_window_fake": args.skip_out_of_window_fake,
        "oversample_positive": args.oversample_positive,
        "train_positive_unique": train_positive_lines,
        "train_positive_extra_copies": train_positive_copies,
        "train_frames": len(train_lines),
        "valid_frames": len(valid_lines),
        "train_videos": len(train_videos),
        "valid_videos": len(valid_videos),
        "label_counts": dict(label_counter),
        "skipped": skipped,
        "seed": args.seed,
        "note": note,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
