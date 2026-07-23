"""Disk cache for Xception fine-tune face crops (decode + MediaPipe once per video)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from face_crop import FaceCropper
from video_xception_infer import read_frame_samples

DEFAULT_CACHE_SUBDIR = "results/cache/xception_finetune"


def crop_cache_root(
    root: Path,
    *,
    crop_method: str,
    crop_padding: float,
    crop_square: bool,
    num_frames: int,
    cache_dir: str | None = None,
) -> Path:
    if cache_dir:
        base = (root / cache_dir).resolve()
    else:
        pad = str(crop_padding).replace(".", "p")
        spec = f"{crop_method}_pad{pad}_{'sq' if crop_square else 'free'}_nf{num_frames}"
        base = (root / DEFAULT_CACHE_SUBDIR / spec).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _video_fingerprint(video_path: Path) -> dict[str, Any]:
    resolved = video_path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def _video_key(video_path: Path) -> str:
    fp = _video_fingerprint(video_path)
    digest = hashlib.sha256(
        f"{fp['path']}|{fp['mtime_ns']}|{fp['size']}".encode()
    ).hexdigest()
    return digest[:16]


def cache_paths(cache_root: Path, video_path: Path) -> tuple[Path, Path]:
    key = _video_key(video_path)
    return cache_root / f"{key}.npz", cache_root / f"{key}.meta.json"


def cache_meta_matches(meta: dict[str, Any], video_path: Path, *, num_frames: int) -> bool:
    fp = _video_fingerprint(video_path)
    return (
        meta.get("path") == fp["path"]
        and meta.get("mtime_ns") == fp["mtime_ns"]
        and meta.get("size") == fp["size"]
        and meta.get("num_frames") == num_frames
        and int(meta.get("num_crops", 0)) > 0
    )


def load_cached_crops(npz_path: Path, meta_path: Path, video_path: Path, *, num_frames: int) -> np.ndarray | None:
    if not npz_path.is_file() or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not cache_meta_matches(meta, video_path, num_frames=num_frames):
        return None
    try:
        with np.load(npz_path) as data:
            crops = data["crops"]
    except (OSError, KeyError, ValueError):
        return None
    if crops.ndim != 4 or crops.shape[0] == 0:
        return None
    return crops


def extract_face_crops(
    video_path: Path,
    cropper: FaceCropper,
    *,
    num_frames: int,
) -> np.ndarray:
    frame_samples = read_frame_samples(video_path, num_frames=num_frames)
    crops: list[np.ndarray] = []
    for sample in frame_samples:
        crop = cropper.crop(sample["frame"])
        if crop is not None:
            crops.append(crop)
    if not crops:
        return np.zeros((0, 256, 256, 3), dtype=np.uint8)
    return np.stack(crops, axis=0).astype(np.uint8, copy=False)


def save_cached_crops(
    npz_path: Path,
    meta_path: Path,
    video_path: Path,
    crops: np.ndarray,
    *,
    num_frames: int,
) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, crops=crops)
    meta = {
        **_video_fingerprint(video_path),
        "num_frames": num_frames,
        "num_crops": int(crops.shape[0]),
        "crop_shape": list(crops.shape[1:]),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def ensure_video_crop_cache(
    video_path: Path,
    cropper: FaceCropper,
    cache_root: Path,
    *,
    num_frames: int,
    rebuild: bool = False,
) -> np.ndarray:
    npz_path, meta_path = cache_paths(cache_root, video_path)
    if not rebuild:
        cached = load_cached_crops(npz_path, meta_path, video_path, num_frames=num_frames)
        if cached is not None:
            return cached
    crops = extract_face_crops(video_path, cropper, num_frames=num_frames)
    if crops.shape[0] > 0:
        save_cached_crops(npz_path, meta_path, video_path, crops, num_frames=num_frames)
    return crops


def ensure_crop_cache_for_samples(
    samples: list[tuple[Path, int]],
    cropper: FaceCropper,
    cache_root: Path,
    *,
    num_frames: int,
    rebuild: bool = False,
) -> dict[Path, np.ndarray]:
    """Decode + crop each video once; return path -> crops array."""
    store: dict[Path, np.ndarray] = {}
    total = len(samples)
    for i, (video_path, _label) in enumerate(samples, start=1):
        crops = ensure_video_crop_cache(
            video_path,
            cropper,
            cache_root,
            num_frames=num_frames,
            rebuild=rebuild,
        )
        store[video_path.resolve()] = crops
        if i % 10 == 0 or i == total:
            print(f"  crop cache: {i}/{total} videos", flush=True)
    return store


def count_usable_faces(crops: np.ndarray, *, num_frames: int) -> int:
    return int(crops.shape[0]) if crops.ndim == 4 else 0


def has_usable_faces_from_crops(crops: np.ndarray, *, num_frames: int) -> bool:
    return count_usable_faces(crops, num_frames=num_frames) >= max(4, num_frames // 4)
