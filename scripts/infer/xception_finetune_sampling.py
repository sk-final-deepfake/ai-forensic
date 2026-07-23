#!/usr/bin/env python3
"""Lightweight train-clip sampling for Xception fine-tune (no torch / infer model deps)."""
from __future__ import annotations

import random
from pathlib import Path

import cv2

from face_crop import FaceCropper
from xception_finetune_crop_cache import (
    ensure_video_crop_cache,
    has_usable_faces_from_crops,
)


def read_frame_samples(video_path: Path, num_frames: int = 32) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []

    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = [int(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]

    samples: list[dict] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            samples.append({"frame_index": idx, "frame": frame})
    cap.release()
    return samples


def has_usable_faces(
    video_path: Path,
    cropper: FaceCropper,
    *,
    num_frames: int = 32,
) -> bool:
    frame_samples = read_frame_samples(video_path, num_frames=num_frames)
    face_count = sum(1 for sample in frame_samples if cropper.crop(sample["frame"]) is not None)
    return face_count >= max(4, num_frames // 4)


def list_train_videos(
    fake_dir: Path,
    real_dir: Path,
    excluded: set[str],
    max_per_class: int,
    seed: int,
    cropper: FaceCropper,
    *,
    num_frames: int = 32,
    cache_root: Path | None = None,
    rebuild_cache: bool = False,
    extra_real_dirs: list[Path] | None = None,
) -> list[tuple[Path, int]]:
    rng = random.Random(seed)
    samples: list[tuple[Path, int]] = []
    for label, directories in (
        (1, [fake_dir]),
        (0, _merge_real_dirs(real_dir, extra_real_dirs)),
    ):
        label_name = "fake" if label == 1 else "real"
        candidates = _collect_mp4_candidates(directories, excluded)
        rng.shuffle(candidates)
        dir_note = directories[0] if len(directories) == 1 else f"{len(directories)} dirs (primary {directories[0]})"
        print(f"scanning {label_name}: {len(candidates)} mp4 in {dir_note}", flush=True)
        picked = 0
        for path in candidates:
            if picked >= max_per_class:
                break
            if cache_root is not None:
                crops = ensure_video_crop_cache(
                    path,
                    cropper,
                    cache_root,
                    num_frames=num_frames,
                    rebuild=rebuild_cache,
                )
                ok = has_usable_faces_from_crops(crops, num_frames=num_frames)
            else:
                ok = has_usable_faces(path, cropper, num_frames=num_frames)
            if ok:
                samples.append((path, label))
                picked += 1
                if picked % 10 == 0:
                    print(f"  {label_name}: picked {picked}/{max_per_class}", flush=True)
        print(f"  {label_name}: selected {picked} clips", flush=True)
        if picked == 0:
            raise SystemExit(f"No usable {label_name} training videos in {dir_note}")
        if picked < max_per_class:
            print(
                f"  WARN: {label_name} wanted {max_per_class} but only {picked} usable clips "
                f"(lower MAX_PER_CLASS or add data)",
                flush=True,
            )
    rng.shuffle(samples)
    return samples


def _merge_real_dirs(primary: Path, extra: list[Path] | None) -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()
    for d in [primary, *(extra or [])]:
        key = str(d.resolve())
        if key not in seen and d.is_dir():
            seen.add(key)
            dirs.append(d)
    return dirs


def _collect_mp4_candidates(directories: list[Path], excluded: set[str]) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for directory in directories:
        if not directory.is_dir():
            raise SystemExit(f"Missing train dir: {directory}")
        for path in sorted(directory.glob("*.mp4")):
            resolved = str(path.resolve())
            if resolved in excluded or resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(path.resolve())
    return candidates


def split_train_val(
    samples: list[tuple[Path, int]],
    val_holdout: int,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    if val_holdout <= 0:
        return samples, []
    per_class = val_holdout // 2
    if per_class == 0:
        return samples, []
    rng = random.Random(seed + 1)
    fake = [s for s in samples if s[1] == 1]
    real = [s for s in samples if s[1] == 0]
    if len(fake) <= per_class or len(real) <= per_class:
        raise SystemExit(
            f"val_holdout={val_holdout} needs >{per_class} clips per class "
            f"(have fake={len(fake)}, real={len(real)})"
        )
    rng.shuffle(fake)
    rng.shuffle(real)
    val = fake[:per_class] + real[:per_class]
    train = fake[per_class:] + real[per_class:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
