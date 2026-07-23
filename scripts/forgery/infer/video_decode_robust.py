"""Robust video decode for forgery benchmarks (OpenCV + ffmpeg transcode cache).

No score imputation: if decode still fails after ffmpeg, caller reports error with reason.
"""
from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4", ".AVI", ".MOV", ".MKV", ".WEBM"}

# upsample_factor=8 and attn_splits=2 => feature map H/8 must be divisible by 2 => align 16
GMFLOW_ALIGN = 16

# OpenCV may decode these but frames are often bad for GMFlow (VP9/webm etc.).
PREFER_FFMPEG_SUFFIXES = {".webm", ".mkv", ".WEBM", ".MKV"}


def _frame_is_usable(rgb: np.ndarray, *, min_side: int = 8) -> bool:
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        return False
    h, w = rgb.shape[:2]
    return h >= min_side and w >= min_side


def count_sequential_frames(video_path: Path, *, limit: int = 512) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    n = 0
    while n < limit:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        n += 1
    cap.release()
    return n


def transcode_h264(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-profile:v",
        "main",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-an",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    return subprocess.run(cmd, check=False).returncode == 0


def _cache_path(src: Path, cache_dir: Path) -> Path:
    key = hashlib.sha1(str(src.resolve()).encode("utf-8")).hexdigest()[:20]
    stem = src.stem.replace(" ", "_")[:40]
    return cache_dir / f"{stem}_{key}.mp4"


def _resolve_via_ffmpeg_cache(
    src: Path,
    cache_dir: Path,
    *,
    min_frames: int,
) -> tuple[Path, str]:
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = _cache_path(src, cache_dir)

    if dst.is_file() and count_sequential_frames(dst, limit=max(min_frames + 2, 8)) >= min_frames:
        return dst, "ffmpeg_cache_hit"

    if transcode_h264(src, dst) and count_sequential_frames(dst, limit=max(min_frames + 2, 8)) >= min_frames:
        return dst, "ffmpeg_transcode"

    return src, "decode_failed"


def resolve_decodable_video_path(
    src: Path,
    cache_dir: Path | None,
    *,
    min_frames: int = 2,
) -> tuple[Path, str]:
    """Return (path_to_read, decode_method). Never invents scores — only real decode paths."""
    src = src.expanduser().resolve()
    if not src.is_file():
        return src, "missing_file"

    prefer_ffmpeg = src.suffix in PREFER_FFMPEG_SUFFIXES

    if not prefer_ffmpeg and count_sequential_frames(src, limit=max(min_frames + 2, 8)) >= min_frames:
        return src, "opencv_native"

    if cache_dir is None:
        if count_sequential_frames(src, limit=max(min_frames + 2, 8)) >= min_frames:
            return src, "opencv_native"
        return src, "decode_failed"

    return _resolve_via_ffmpeg_cache(src, cache_dir, min_frames=min_frames)


def force_transcode_path(
    src: Path,
    cache_dir: Path,
    *,
    min_frames: int = 2,
    refresh: bool = True,
) -> tuple[Path, str]:
    """Re-encode with ffmpeg even when OpenCV can read the file (fixes GMFlow-incompatible streams)."""
    src = src.expanduser().resolve()
    if not src.is_file():
        return src, "missing_file"
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = _cache_path(src, cache_dir)
    if refresh and dst.is_file():
        dst.unlink(missing_ok=True)
    resolved, method = _resolve_via_ffmpeg_cache(src, cache_dir, min_frames=min_frames)
    if method in ("ffmpeg_cache_hit", "ffmpeg_transcode"):
        return resolved, "ffmpeg_forced_transcode"
    return resolved, "decode_failed"


def prepare_frame_for_gmflow(
    image_rgb: np.ndarray,
    *,
    max_side: int,
    align: int = GMFLOW_ALIGN,
) -> np.ndarray:
    """Resize (if needed) and pad so H/W are multiples of align (GMFlow upsample_factor=8)."""
    from optical_flow_common import resize_for_flow  # noqa: E402

    out = resize_for_flow(image_rgb, max_side=max_side)
    h, w = out.shape[:2]
    nh = max(align, int(math.ceil(h / align) * align))
    nw = max(align, int(math.ceil(w / align) * align))
    if nh == h and nw == w:
        return out
    canvas = np.zeros((nh, nw, 3), dtype=out.dtype)
    canvas[:h, :w] = out
    return canvas


def read_rgb_frames_sequential(
    video_path: Path,
    *,
    max_frames: int,
    max_side: int,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if not _frame_is_usable(rgb):
            continue
        frames.append(prepare_frame_for_gmflow(rgb, max_side=max_side))
    cap.release()
    return frames


def probe_video_fps_and_frames(video_path: Path, *, probe_limit: int = 10000) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 25.0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    meta_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps < 1.0 or fps > 240.0:
        fps = 25.0
    probed = count_sequential_frames(video_path, limit=probe_limit)
    total = probed if probed >= 2 else meta_count
    return fps, total


def iter_window_starts(
    total_frames: int,
    *,
    fps: float,
    window_sec: float = 1.0,
    stride_sec: float = 0.5,
) -> list[tuple[int, int]]:
    """Return (start_frame, end_frame_exclusive) for each sliding window."""
    if total_frames < 2:
        return []
    window_frames = max(2, int(round(fps * window_sec)))
    stride_frames = max(1, int(round(fps * stride_sec)))
    if window_frames > total_frames:
        window_frames = total_frames
    out: list[tuple[int, int]] = []
    start = 0
    while start < total_frames - 1:
        end = min(start + window_frames, total_frames)
        if end - start < 2:
            break
        out.append((start, end))
        if end >= total_frames:
            break
        start += stride_frames
    return out


def sample_frame_pairs_window_sliding(
    video_path: Path,
    *,
    window_sec: float = 1.0,
    stride_sec: float = 0.5,
    pairs_per_window: int = 8,
    max_side: int = 768,
    max_frames: int = 9000,
) -> tuple[list[tuple[np.ndarray, np.ndarray, int, int]], dict[str, float | int]]:
    """1-second sliding windows; dense adjacent pairs inside each window."""
    fps, _meta_total = probe_video_fps_and_frames(video_path, probe_limit=max_frames + 2)
    frames = read_rgb_frames_sequential(video_path, max_frames=max_frames, max_side=max_side)
    total = len(frames)
    meta: dict[str, float | int] = {
        "fps": fps,
        "total_frames": total,
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "pairs_per_window": pairs_per_window,
    }
    if total < 2:
        return [], meta

    windows = iter_window_starts(total, fps=fps, window_sec=window_sec, stride_sec=stride_sec)
    meta["n_windows"] = len(windows)

    seen: set[int] = set()
    pair_slots: list[int] = []
    for start, end in windows:
        n_slots = end - start - 1
        if n_slots < 1:
            continue
        if n_slots <= pairs_per_window:
            indices = list(range(start, end - 1))
        else:
            local = np.linspace(0, n_slots - 1, num=pairs_per_window, dtype=int)
            indices = [start + int(i) for i in local]
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                pair_slots.append(idx)

    pair_slots.sort()
    meta["pairs_sampled"] = len(pair_slots)
    pairs = [(frames[i], frames[i + 1], int(i), int(i + 1)) for i in pair_slots]
    return pairs, meta


def sample_pairs_from_frames(
    rgb_frames: list[np.ndarray],
    max_pairs: int,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    if len(rgb_frames) < 2:
        return []
    n_slots = len(rgb_frames) - 1
    if n_slots <= max_pairs:
        indices = list(range(n_slots))
    else:
        indices = np.linspace(0, n_slots - 1, num=max_pairs, dtype=int).tolist()
    return [(rgb_frames[i], rgb_frames[i + 1], int(i), int(i + 1)) for i in indices]


def sample_frame_pairs_robust(
    video_path: Path,
    *,
    max_pairs: int,
    max_side: int,
    scan_mode: str = "dense",
    window_sec: float = 1.0,
    stride_sec: float = 0.5,
    pairs_per_window: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    """Dense/sparse/window_1s indexing on buffer loaded via sequential decode (no broken seek)."""
    if scan_mode == "window_1s":
        ppw = pairs_per_window if pairs_per_window is not None else min(8, max(1, max_pairs))
        pairs, _meta = sample_frame_pairs_window_sliding(
            video_path,
            window_sec=window_sec,
            stride_sec=stride_sec,
            pairs_per_window=ppw,
            max_side=max_side,
        )
        return pairs

    need = max_pairs + 1
    if scan_mode == "dense":
        buffer_cap = min(need * 4, 512) if max_pairs <= 128 else min(need * 2, 768)
    else:
        buffer_cap = min(need + 16, 128)

    frames = read_rgb_frames_sequential(video_path, max_frames=buffer_cap, max_side=max_side)
    if len(frames) < 2:
        return []

    n_slots = len(frames) - 1
    if scan_mode == "dense":
        if n_slots <= max_pairs:
            indices = list(range(n_slots))
        else:
            indices = np.linspace(0, n_slots - 1, num=max_pairs, dtype=int).tolist()
    else:
        head = list(range(min(16, n_slots)))
        remaining = max(0, max_pairs - len(head))
        if remaining > 0 and n_slots > len(head):
            step = max(1, (n_slots - 1) // remaining)
            sparse = [min(i * step, n_slots - 1) for i in range(remaining)]
            indices = sorted(set(head + sparse))[:max_pairs]
        else:
            indices = head[:max_pairs]

    return [(frames[i], frames[i + 1], int(i), int(i + 1)) for i in indices]


def diagnose_video_skip(video_path: Path) -> str:
    if not video_path.is_file():
        return "missing_file"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "opencv_open_failed"
    meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    probed = count_sequential_frames(video_path, limit=512)
    if probed < 2:
        return f"decodable_frames={probed}_meta_frames={meta}"
    return "flow_infer_failed_for_all_pairs"
