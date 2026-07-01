#!/usr/bin/env python3
"""Shared helpers for ForenShield TruFor video forgery fine-tune."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2

VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv"}

# TruFor(spatial)에 맞는 조작 유형 — temporal 계열은 기본 제외
DEFAULT_SPATIAL_TAMPER_TYPES = frozenset({"masking", "substitution", "rotate"})

# frame-deletion / dropping / repetition 등 — GMFlow 쪽 학습용
DEFAULT_TEMPORAL_TAMPER_TYPES = frozenset(
    {
        "dropping",
        "repetition",
        "frame-deletion",
        "frame-duplication",
        "frame-insertion",
        "eop-frame-deletion",
        "eop-frame-duplication",
        "eop-frame-insertion",
    }
)

MIDDLE_TAMPER_RE = re.compile(r"middle_tampered_([a-z_]+)_(\d+)sec", re.IGNORECASE)


@dataclass(frozen=True)
class FramePlan:
    video_path: Path
    frame_index: int
    label: int  # 0 real, 1 tampered
    tamper_type: str
    source_split: str  # original | tampered


def read_video_meta(video_path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return total, fps


def uniform_frame_indices(total: int, num_frames: int) -> list[int]:
    if total <= 0:
        return []
    if total <= num_frames:
        return list(range(total))
    return [int(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]


def middle_tamper_window(total: int, fps: float, duration_sec: float = 1.0) -> tuple[int, int]:
    """Return inclusive [start, end] frame indices for middle tamper window."""
    if total <= 0:
        return 0, -1
    if fps <= 0:
        window = max(1, int(round(duration_sec * 25)))
    else:
        window = max(1, int(round(duration_sec * fps)))
    window = min(window, total)
    start = max(0, (total - window) // 2)
    end = min(total - 1, start + window - 1)
    return start, end


def parse_middle_tamper_seconds(name: str, default_sec: float = 1.0) -> float:
    m = MIDDLE_TAMPER_RE.search(name)
    if not m:
        return default_sec
    try:
        return float(m.group(2))
    except ValueError:
        return default_sec


def iter_videos(root: Path, split_name: str) -> list[Path]:
    base = root / split_name
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


def tamper_type_from_path(video_path: Path, data_root: Path) -> str:
    rel = video_path.relative_to(data_root)
    parts = rel.parts
    if len(parts) < 2 or parts[0] != "tampered":
        return "unknown"
    return parts[1]


def build_frame_plans(
    data_root: Path,
    *,
    frames_per_video: int = 8,
    spatial_tamper_types: frozenset[str] = DEFAULT_SPATIAL_TAMPER_TYPES,
    include_temporal_fakes: bool = False,
    require_middle_tamper_window: bool = True,
    skip_out_of_window_fake: bool = False,
) -> list[FramePlan]:
    """Build per-frame plans for TruFor prepare.

    When ``require_middle_tamper_window`` is True (recipe v2), spatial tampered
    videos without ``middle_tampered`` in the filename are skipped instead of
    labelling every frame positive (weak full-frame supervision).

    When ``skip_out_of_window_fake`` is True (recipe v4), middle-tampered spatial
    videos omit out-of-window frames entirely (no hard-negative mask=0).
    """
    plans: list[FramePlan] = []

    for video in iter_videos(data_root, "original"):
        total, _fps = read_video_meta(video)
        for idx in uniform_frame_indices(total, frames_per_video):
            plans.append(
                FramePlan(
                    video_path=video,
                    frame_index=idx,
                    label=0,
                    tamper_type="original",
                    source_split="original",
                )
            )

    for video in iter_videos(data_root, "tampered"):
        tamper_type = tamper_type_from_path(video, data_root)
        if tamper_type not in spatial_tamper_types:
            if not (include_temporal_fakes and tamper_type in DEFAULT_TEMPORAL_TAMPER_TYPES):
                continue

        total, fps = read_video_meta(video)
        indices = uniform_frame_indices(total, frames_per_video)
        has_middle = "middle_tampered" in video.name.lower()

        if tamper_type in spatial_tamper_types:
            if require_middle_tamper_window and not has_middle:
                continue
            if has_middle:
                duration = parse_middle_tamper_seconds(video.name)
                start, end = middle_tamper_window(total, fps, duration)
                for idx in indices:
                    in_window = start <= idx <= end
                    if not in_window and skip_out_of_window_fake:
                        continue
                    label = 1 if in_window else 0
                    plans.append(
                        FramePlan(
                            video_path=video,
                            frame_index=idx,
                            label=label,
                            tamper_type=tamper_type,
                            source_split="tampered",
                        )
                    )
                continue
            # spatial but no middle tag and require_middle_tamper_window=False
            for idx in indices:
                plans.append(
                    FramePlan(
                        video_path=video,
                        frame_index=idx,
                        label=1,
                        tamper_type=tamper_type,
                        source_split="tampered",
                    )
                )
        else:
            # temporal fake (only when include_temporal_fakes=True)
            for idx in indices:
                plans.append(
                    FramePlan(
                        video_path=video,
                        frame_index=idx,
                        label=1,
                        tamper_type=tamper_type,
                        source_split="tampered",
                    )
                )

    return plans


def frame_cache_name(video_path: Path, data_root: Path, frame_index: int) -> str:
    rel = video_path.relative_to(data_root).with_suffix("")
    return f"{rel.as_posix().replace('/', '__')}_f{frame_index:03d}"


def write_mask_png(path: Path, label: int, width: int, height: int) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    value = 255 if label else 0
    arr = np.full((height, width), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def extract_frame_bgr(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def save_frame_jpg(path: Path, frame_bgr) -> bool:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, quality=92)
    return True


def load_manifest(data_root: Path) -> dict:
    manifest_path = data_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))
