"""Weak tamper segment hints for pair-level GMFlow MIL training.

MVTamperBench tampered clips use ``_middle_tampered_<type>_1sec`` — tamper is ~1s at
video center. CSVTED ``eop-*`` types use the last ~1s; other frame edits use center 1s
as a weak prior when no frame index metadata exists.

Real (original) clips: every sampled pair is label 0.
Fake without a parseable hint: pair labels are masked out (clip-level MIL only).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MVTB_MIDDLE_RE = re.compile(
    r"_middle_tampered_(?:dropping|masking|repetition|rotate|substitution|splice)_1sec",
    re.I,
)

CSVTED_EOP_TYPES = frozenset(
    {
        "eop-frame-deletion",
        "eop-frame-duplication",
        "eop-frame-insertion",
    }
)


@dataclass(frozen=True)
class TamperSegment:
    start_frame: int
    end_frame_exclusive: int
    source: str

    def to_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame_exclusive": self.end_frame_exclusive,
            "source": self.source,
        }


def _segment_middle_sec(total_frames: int, fps: float, duration_sec: float) -> TamperSegment:
    if total_frames < 2:
        return TamperSegment(0, total_frames, "middle_fallback")
    mid = total_frames // 2
    half = max(1, int(round(fps * duration_sec / 2.0)))
    start = max(0, mid - half)
    end = min(total_frames, mid + half)
    if end <= start:
        end = min(total_frames, start + 1)
    return TamperSegment(start, end, "middle_sec")


def _segment_tail_sec(total_frames: int, fps: float, duration_sec: float) -> TamperSegment:
    if total_frames < 2:
        return TamperSegment(0, total_frames, "tail_fallback")
    span = max(1, int(round(fps * duration_sec)))
    start = max(0, total_frames - span)
    return TamperSegment(start, total_frames, "tail_sec")


def infer_tamper_type_from_path(video_path: Path | str, *, relative_path: str | None = None) -> str | None:
    rel = relative_path or str(video_path)
    rel_l = rel.replace("\\", "/").lower()
    if "/tampered/" not in rel_l and "tampered" not in Path(rel).parts:
        return None
    parts = Path(rel).parts
    if "tampered" in parts:
        idx = parts.index("tampered")
        if idx + 1 < len(parts):
            return parts[idx + 1].lower()
    return None


def infer_tamper_segment(
    video_path: Path | str,
    *,
    ground_truth_label: str,
    total_frames: int,
    fps: float,
    duration_sec: float = 1.0,
    relative_path: str | None = None,
) -> TamperSegment | None:
    """Return tamper frame span for fake clips, or None if unknown."""
    if ground_truth_label != "fake":
        return None

    path = Path(video_path)
    rel = relative_path or path.name
    name = path.name
    rel_l = rel.replace("\\", "/").lower()
    tamper_type = infer_tamper_type_from_path(path, relative_path=rel)

    if MVTB_MIDDLE_RE.search(name) or MVTB_MIDDLE_RE.search(rel):
        return _segment_middle_sec(total_frames, fps, duration_sec)

    if tamper_type in CSVTED_EOP_TYPES or "/eop-" in rel_l:
        return _segment_tail_sec(total_frames, fps, duration_sec)

    # CSVTED frame ops / spatial: weak center prior (better than clip-only MIL).
    if tamper_type is not None or "csvted" in rel_l or "/tampered/" in rel_l:
        return _segment_middle_sec(total_frames, fps, duration_sec)

    return None


def pair_overlaps_segment(idx_a: int, idx_b: int, seg: TamperSegment) -> bool:
    """True if adjacent pair (idx_a, idx_b) intersects [start, end)."""
    lo = min(int(idx_a), int(idx_b))
    hi = max(int(idx_a), int(idx_b))
    return lo < seg.end_frame_exclusive and hi >= seg.start_frame


def window_overlaps_segment(start: int, end_exclusive: int, seg: TamperSegment) -> bool:
    """True if frame span [start, end_exclusive) intersects tamper segment."""
    return int(start) < seg.end_frame_exclusive and int(end_exclusive) > seg.start_frame


def pair_labels_for_video(
    per_pair: list[dict],
    *,
    ground_truth_label: str,
    total_frames: int,
    fps: float,
    video_path: Path | str,
    relative_path: str | None = None,
    duration_sec: float = 1.0,
) -> tuple[list[float], list[bool], dict]:
    """Per-pair (target, mask). mask=False -> ignore in pair BCE."""
    n = len(per_pair)
    targets = [0.0] * n
    masks = [False] * n
    meta: dict = {"pair_label_mode": "segment", "total_frames": total_frames, "fps": fps}

    if ground_truth_label == "real":
        masks = [True] * n
        meta["segment"] = None
        meta["n_positive_pairs"] = 0
        meta["n_labeled_pairs"] = n
        return targets, masks, meta

    seg = infer_tamper_segment(
        video_path,
        ground_truth_label=ground_truth_label,
        total_frames=total_frames,
        fps=fps,
        duration_sec=duration_sec,
        relative_path=relative_path,
    )
    if seg is None:
        meta["segment"] = None
        meta["n_positive_pairs"] = 0
        meta["n_labeled_pairs"] = 0
        return targets, masks, meta

    meta["segment"] = seg.to_dict()
    pos = 0
    labeled = 0
    for i, row in enumerate(per_pair):
        idx_a = int(row.get("frame_index_a", row.get("frame_index_start", -1)))
        idx_b = int(row.get("frame_index_b", row.get("frame_index_end", -1)))
        if idx_a < 0 or idx_b < 0:
            continue
        masks[i] = True
        labeled += 1
        if pair_overlaps_segment(idx_a, idx_b, seg):
            targets[i] = 1.0
            pos += 1

    meta["n_positive_pairs"] = pos
    meta["n_labeled_pairs"] = labeled
    return targets, masks, meta


def window_labels_for_video(
    per_window: list[dict],
    *,
    ground_truth_label: str,
    total_frames: int,
    fps: float,
    video_path: Path | str,
    relative_path: str | None = None,
    duration_sec: float = 1.0,
    label_mode: str = "segment",
) -> tuple[list[float], list[bool], dict]:
    """Per-window (target, mask). mask=False -> ignore in window BCE.

    label_mode:
      segment      — fake: segment overlap positives; real: all windows negative
      real_windows — real only: all windows negative (suppress shake/noise);
                     fake: no window loss (clip bag only; avoids weak segment priors)
      clip         — no window labels (handled by caller)
    """
    n = len(per_window)
    targets = [0.0] * n
    masks = [False] * n
    meta: dict = {"window_label_mode": label_mode, "total_frames": total_frames, "fps": fps}

    if label_mode == "real_windows" and ground_truth_label == "fake":
        meta["segment"] = None
        meta["n_positive_windows"] = 0
        meta["n_labeled_windows"] = 0
        return targets, masks, meta

    if ground_truth_label == "real":
        masks = [True] * n
        meta["segment"] = None
        meta["n_positive_windows"] = 0
        meta["n_labeled_windows"] = n
        return targets, masks, meta

    seg = infer_tamper_segment(
        video_path,
        ground_truth_label=ground_truth_label,
        total_frames=total_frames,
        fps=fps,
        duration_sec=duration_sec,
        relative_path=relative_path,
    )
    if seg is None:
        meta["segment"] = None
        meta["n_positive_windows"] = 0
        meta["n_labeled_windows"] = 0
        return targets, masks, meta

    meta["segment"] = seg.to_dict()
    pos = 0
    labeled = 0
    for i, row in enumerate(per_window):
        start = int(row.get("frame_index_start", -1))
        end = int(row.get("frame_index_end", -1))
        if start < 0 or end <= start:
            continue
        masks[i] = True
        labeled += 1
        if window_overlaps_segment(start, end, seg):
            targets[i] = 1.0
            pos += 1

    meta["n_positive_windows"] = pos
    meta["n_labeled_windows"] = labeled
    return targets, masks, meta
