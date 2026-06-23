"""
Shared utilities for optical flow benchmark (RAFT / GMFlow / PWC-Net).

Outputs schema_version 1.1 reports with per-frame-pair and segment breakdowns.
Optical flow does not classify fake/real directly; motion_anomaly_score is a
heuristic vs the real-video cohort in the same run.
"""

from __future__ import annotations

import html
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from backends.base import OpticalFlowBackend

SCHEMA_VERSION = "1.1"
DEFAULT_ANOMALY_THRESHOLD = 0.5
DEFAULT_SEGMENT_COUNT = 4
PROGRESS_BAR_WIDTH = 28
BENCHMARK_METHOD = "optical_flow_motion_heuristic_over_frame_pairs"


@dataclass
class FlowStats:
    flow_mean: float
    flow_max: float
    flow_std: float
    frame_pairs: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VideoFlowResult:
    """Legacy summary fields (kept for smoke tests)."""

    video_path: str
    label: str
    model: str
    status: str
    flow_mean: float | None = None
    flow_max: float | None = None
    flow_std: float | None = None
    frame_pairs: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def discover_videos(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    patterns = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(directory.glob(pattern))
    return sorted(files)


def list_benchmark_tasks(fake_dir: Path, real_dir: Path) -> list[tuple[str, Path]]:
    tasks: list[tuple[str, Path]] = []
    for label, directory in (("fake", fake_dir), ("real", real_dir)):
        for video in discover_videos(directory):
            tasks.append((label, video))
    return tasks


def run_dir_for(root: Path, run_id: str) -> Path:
    return root / "results" / "infer" / run_id


def report_json_path(run_dir: Path, video_path: Path) -> Path:
    return run_dir / "json" / f"{video_path.stem}.json"


def load_saved_report(json_path: Path) -> dict[str, Any] | None:
    if not json_path.is_file():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("status") in ("ok", "error"):
        return data
    return None


def save_report_json(json_path: Path, report: dict[str, Any]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class BenchmarkProgress:
    """Terminal progress bar with ETA (no extra dependencies)."""

    def __init__(self, model: str, total: int, run_id: str) -> None:
        self.model = model
        self.total = max(total, 1)
        self.run_id = run_id
        self.completed = 0
        self.skipped = 0
        self.errors = 0
        self.started = time.perf_counter()
        self.process_durations: list[float] = []

    def _eta_seconds(self) -> float | None:
        remaining = self.total - self.completed
        if not self.process_durations or remaining <= 0:
            return None
        avg = sum(self.process_durations) / len(self.process_durations)
        return avg * remaining

    def tick(
        self,
        *,
        file_name: str,
        status: str,
        skipped: bool = False,
        elapsed_sec: float = 0.0,
    ) -> None:
        self.completed += 1
        if skipped:
            self.skipped += 1
        elif status == "error":
            self.errors += 1
        if not skipped and elapsed_sec > 0:
            self.process_durations.append(elapsed_sec)

        filled = int(PROGRESS_BAR_WIDTH * self.completed / self.total)
        bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
        elapsed = _format_duration(time.perf_counter() - self.started)
        eta = self._eta_seconds()
        eta_text = _format_duration(eta) if eta is not None else "--"
        pct = 100.0 * self.completed / self.total
        tag = "skip" if skipped else status
        line = (
            f"\r[{self.model}] [{bar}] {self.completed}/{self.total} ({pct:5.1f}%) "
            f"| elapsed {elapsed} | ETA {eta_text} | {file_name} [{tag}]"
        )
        sys.stdout.write(line.ljust(120))
        sys.stdout.flush()
        if self.completed >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def finish_message(self) -> str:
        elapsed = _format_duration(time.perf_counter() - self.started)
        return (
            f"progress done: {self.completed}/{self.total} "
            f"(new={self.completed - self.skipped}, skipped={self.skipped}, errors={self.errors}, elapsed={elapsed})"
        )


def write_checkpoint(
    run_dir: Path,
    *,
    run_id: str,
    model: str,
    total: int,
    completed: int,
    skipped: int,
    fake_dir: Path,
    real_dir: Path,
    max_frames: int,
    threshold: float,
    finished: bool,
) -> None:
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model": model,
        "total": total,
        "completed": completed,
        "skipped": skipped,
        "remaining": total - completed,
        "finished": finished,
        "fake_dir": str(fake_dir),
        "real_dir": str(real_dir),
        "max_frames": max_frames,
        "threshold": threshold,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sample_frames_with_indices(video_path: Path, max_frames: int = 32) -> tuple[list[np.ndarray], list[int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    indices: list[int] = []
    if total <= 0:
        idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
            indices.append(idx)
            idx += 1
            if len(frames) >= max_frames:
                break
    else:
        sampled = np.linspace(0, max(total - 1, 0), num=min(max_frames, total), dtype=int)
        for idx in sampled:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
                indices.append(int(idx))
    capture.release()
    if len(frames) < 2:
        raise RuntimeError(f"Need at least 2 frames: {video_path}")
    return frames, indices


def sample_frames(video_path: Path, max_frames: int = 32) -> list[np.ndarray]:
    frames, _ = sample_frames_with_indices(video_path, max_frames=max_frames)
    return frames


def resize_for_flow(image_rgb: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image_rgb
    scale = max_side / float(longest)
    nh = max(8, int(round(h * scale)))
    nw = max(8, int(round(w * scale)))
    return cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)


def read_video_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def sample_frame_pairs(
    video_path: Path,
    max_pairs: int = 8,
    *,
    max_side: int = 512,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < 2:
        cap.release()
        return []

    if total <= max_pairs + 1:
        indices = list(range(total - 1))
    else:
        step = max(1, (total - 2) // max_pairs)
        indices = [min(i * step, total - 2) for i in range(max_pairs)]

    pairs: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    for idx in indices:
        f1 = read_video_frame(cap, idx)
        f2 = read_video_frame(cap, idx + 1)
        if f1 is not None and f2 is not None:
            pairs.append(
                (
                    resize_for_flow(f1, max_side=max_side),
                    resize_for_flow(f2, max_side=max_side),
                    idx,
                    idx + 1,
                )
            )
    cap.release()
    return pairs


def flow_to_numpy(flow) -> np.ndarray:
    if hasattr(flow, "detach"):
        arr = flow.detach().float().cpu().numpy()
    else:
        arr = np.asarray(flow, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 2:
        arr = np.transpose(arr, (1, 2, 0))
    return arr.astype(np.float32)


def summarize_flow(flow: np.ndarray) -> dict:
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    angle = np.arctan2(fy, fx)
    return {
        "magnitude_mean": round(float(mag.mean()), 6),
        "magnitude_std": round(float(mag.std()), 6),
        "magnitude_max": round(float(mag.max()), 6),
        "magnitude_p95": round(float(np.percentile(mag, 95)), 6),
        "magnitude_median": round(float(np.median(mag)), 6),
        "angle_std": round(float(angle.std()), 6),
        "flow_x_mean": round(float(fx.mean()), 6),
        "flow_y_mean": round(float(fy.mean()), 6),
    }


def aggregate_pair_stats(pair_stats: list[dict]) -> dict:
    if not pair_stats:
        return {}
    keys = [k for k in pair_stats[0].keys() if k not in {"frame_index_a", "frame_index_b"}]
    out: dict = {"pair_count": len(pair_stats)}
    for key in keys:
        values = [row[key] for row in pair_stats]
        out[f"{key}_mean"] = round(float(np.mean(values)), 6)
        out[f"{key}_std"] = round(float(np.std(values)), 6)
    return out


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)


def distribution_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "median": 0.0, "std": 0.0, "p25": 0.0, "p75": 0.0, "mean": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def compute_pair_metrics(flow: np.ndarray) -> dict[str, float]:
    mag = flow_magnitude(flow)
    u = flow[..., 0]
    v = flow[..., 1]
    angle = np.arctan2(v, u)
    mag_mean = float(mag.mean())
    spatial_inconsistency = float(mag.std() / (mag_mean + 1e-6))
    motion_energy = float(np.mean(mag**2))
    # Circular std approximation for flow direction spread.
    sin_mean = float(np.sin(angle).mean())
    cos_mean = float(np.cos(angle).mean())
    angle_dispersion = float(1.0 - math.sqrt(sin_mean**2 + cos_mean**2))
    return {
        "flow_mag_mean": mag_mean,
        "flow_mag_max": float(mag.max()),
        "flow_mag_std": float(mag.std()),
        "flow_mag_median": float(np.median(mag)),
        "flow_u_mean": float(u.mean()),
        "flow_v_mean": float(v.mean()),
        "spatial_inconsistency": spatial_inconsistency,
        "motion_energy": motion_energy,
        "angle_dispersion": angle_dispersion,
    }


def aggregate_flow_stats(backend: OpticalFlowBackend, frames: list[np.ndarray]) -> FlowStats:
    magnitudes: list[float] = []
    pairs = 0
    for i in range(len(frames) - 1):
        flow = backend.compute_flow(frames[i], frames[i + 1])
        mag = flow_magnitude(flow)
        magnitudes.append(float(mag.mean()))
        magnitudes.append(float(mag.max()))
        pairs += 1

    flat = np.asarray(magnitudes, dtype=np.float64)
    return FlowStats(
        flow_mean=float(flat.mean()) if flat.size else 0.0,
        flow_max=float(flat.max()) if flat.size else 0.0,
        flow_std=float(flat.std()) if flat.size else 0.0,
        frame_pairs=pairs,
    )


def _build_segments(per_pair: list[dict[str, Any]], segment_count: int) -> list[dict[str, Any]]:
    metric_keys = [
        "flow_mag_mean",
        "flow_mag_max",
        "flow_mag_std",
        "flow_mag_median",
        "flow_u_mean",
        "flow_v_mean",
        "spatial_inconsistency",
        "motion_energy",
        "angle_dispersion",
    ]
    if not per_pair:
        return []
    n = len(per_pair)
    segments: list[dict[str, Any]] = []
    for seg_idx in range(segment_count):
        start = (seg_idx * n) // segment_count
        end = ((seg_idx + 1) * n) // segment_count
        if start >= end:
            continue
        chunk = per_pair[start:end]
        seg_metrics = {key: [row[key] for row in chunk] for key in metric_keys}
        segments.append(
            {
                "segment_index": seg_idx,
                "pair_start": start,
                "pair_end": end - 1,
                "pair_count": end - start,
                "aggregate": {k: distribution_stats(v) for k, v in seg_metrics.items()},
            }
        )
    return segments


def _score_stats_from_pairs(per_pair: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    numeric_keys = [
        "flow_mag_mean",
        "flow_mag_max",
        "flow_mag_std",
        "spatial_inconsistency",
        "motion_energy",
        "angle_dispersion",
        "flow_u_mean",
        "flow_v_mean",
    ]
    return {key: distribution_stats([row[key] for row in per_pair]) for key in numeric_keys}


def _temporal_jitter(per_pair: list[dict[str, Any]]) -> float:
    means = [row["flow_mag_mean"] for row in per_pair]
    if not means:
        return 0.0
    arr = np.asarray(means, dtype=np.float64)
    return float(arr.std() / (arr.mean() + 1e-6))


def _build_interpretation(
    report: dict[str, Any],
    cohort: dict[str, float],
    threshold: float,
) -> dict[str, Any]:
    signals: list[dict[str, Any]] = []
    if report.get("status") != "ok":
        return {
            "method": "optical_flow_motion_heuristic",
            "note": "No motion signals ??video processing failed.",
            "signals": signals,
        }

    breakdown = report.get("score_breakdown", {})
    aggregate = breakdown.get("aggregate", {})
    jitter = float(aggregate.get("temporal_jitter", 0.0))
    spatial = float(aggregate.get("spatial_inconsistency_mean", 0.0))
    angle = float(aggregate.get("angle_dispersion_mean", 0.0))
    flow_mean = float(aggregate.get("flow_mag_mean", 0.0))

    checks = [
        (
            "temporal_jitter",
            jitter,
            cohort.get("temporal_jitter_median", jitter),
            "Frame-to-frame motion magnitude varies more than typical real videos.",
        ),
        (
            "spatial_inconsistency",
            spatial,
            cohort.get("spatial_inconsistency_median", spatial),
            "Motion is spatially uneven (patchy flow) vs real baseline.",
        ),
        (
            "angle_dispersion",
            angle,
            cohort.get("angle_dispersion_median", angle),
            "Flow directions are less coherent across the frame.",
        ),
        (
            "flow_magnitude",
            flow_mean,
            cohort.get("flow_mag_mean_median", flow_mean),
            "Average motion magnitude deviates from real-video baseline.",
        ),
    ]
    for name, value, baseline, note in checks:
        delta = value - baseline
        if abs(delta) > 1e-6:
            signals.append(
                {
                    "name": name,
                    "value": round(value, 6),
                    "real_cohort_baseline": round(baseline, 6),
                    "delta": round(delta, 6),
                    "direction": "above" if delta > 0 else "below",
                    "note": note,
                }
            )

    score = float(report.get("motion_anomaly_score", 0.0))
    pred = report.get("pred_label", "real")
    summary = (
        f"motion_anomaly_score={score:.4f} (threshold={threshold}); "
        f"pred_label={pred} from motion heuristic ??not a CNN fake/real classifier."
    )
    return {"method": "optical_flow_motion_heuristic", "summary": summary, "signals": signals}


def analyze_video_detailed(
    backend: OpticalFlowBackend,
    video_path: Path,
    label: str,
    run_id: str,
    max_frames: int = 32,
    segment_count: int = DEFAULT_SEGMENT_COUNT,
) -> dict[str, Any]:
    started = time.perf_counter()
    file_name = video_path.name
    base = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file": file_name,
        "source_path": str(video_path.resolve()),
        "ground_truth_label": label,
        "model": backend.name,
        "status": "pending",
    }
    try:
        frames, frame_indices = sample_frames_with_indices(video_path, max_frames=max_frames)
        per_pair: list[dict[str, Any]] = []
        for pair_idx in range(len(frames) - 1):
            flow = backend.compute_flow(frames[pair_idx], frames[pair_idx + 1])
            metrics = compute_pair_metrics(flow)
            per_pair.append(
                {
                    "pair_index": pair_idx,
                    "frame_index_start": frame_indices[pair_idx],
                    "frame_index_end": frame_indices[pair_idx + 1],
                    **metrics,
                }
            )

        score_stats = _score_stats_from_pairs(per_pair)
        temporal_jitter = _temporal_jitter(per_pair)
        aggregate = {
            "flow_mag_mean": score_stats["flow_mag_mean"]["mean"],
            "flow_mag_max": score_stats["flow_mag_max"]["max"],
            "flow_mag_std": score_stats["flow_mag_std"]["mean"],
            "spatial_inconsistency_mean": score_stats["spatial_inconsistency"]["mean"],
            "motion_energy_mean": score_stats["motion_energy"]["mean"],
            "angle_dispersion_mean": score_stats["angle_dispersion"]["mean"],
            "temporal_jitter": temporal_jitter,
            "frame_pairs": len(per_pair),
        }
        segments = _build_segments(per_pair, segment_count=segment_count)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        base.update(
            {
                "status": "ok",
                "frames_sampled": len(frames),
                "frame_pairs_used": len(per_pair),
                "elapsed_ms": round(elapsed_ms, 3),
                "flow_mean": aggregate["flow_mag_mean"],
                "flow_max": aggregate["flow_mag_max"],
                "flow_std": aggregate["flow_mag_std"],
                "score_breakdown": {
                    "schema_version": SCHEMA_VERSION,
                    "method": "optical_flow_per_frame_pair",
                    "threshold": DEFAULT_ANOMALY_THRESHOLD,
                    "frames_sampled": len(frames),
                    "frame_pairs_used": len(per_pair),
                    "aggregate": aggregate,
                    "score_stats": score_stats,
                    "segments": segments,
                    "per_frame_pair": per_pair,
                },
            }
        )
        return base
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        base.update({"status": "error", "elapsed_ms": round(elapsed_ms, 3), "error": str(exc)})
        return base


def _cohort_baselines(reports: list[dict[str, Any]]) -> dict[str, float]:
    reals = [
        r["score_breakdown"]["aggregate"]
        for r in reports
        if r.get("status") == "ok"
        and r.get("ground_truth_label") == "real"
        and "score_breakdown" in r
    ]
    required = (
        "temporal_jitter",
        "spatial_inconsistency_mean",
        "angle_dispersion_mean",
        "flow_mag_mean",
    )
    complete = [row for row in reals if all(k in row and row[k] is not None for k in required)]
    if not complete:
        return {}

    def med(key: str) -> float:
        return float(np.median([row[key] for row in complete]))

    return {
        "temporal_jitter_median": med("temporal_jitter"),
        "spatial_inconsistency_median": med("spatial_inconsistency_mean"),
        "angle_dispersion_median": med("angle_dispersion_mean"),
        "flow_mag_mean_median": med("flow_mag_mean"),
        "temporal_jitter_std": float(np.std([row["temporal_jitter"] for row in complete])) or 1e-6,
        "spatial_inconsistency_std": float(np.std([row["spatial_inconsistency_mean"] for row in complete])) or 1e-6,
        "angle_dispersion_std": float(np.std([row["angle_dispersion_mean"] for row in complete])) or 1e-6,
        "flow_mag_mean_std": float(np.std([row["flow_mag_mean"] for row in complete])) or 1e-6,
    }


def _is_raft_pair_stats_schema(report: dict[str, Any]) -> bool:
    """RAFT ffpp_vox run json: top-level pair_stats + aggregate (magnitude_* keys)."""
    if report.get("score_breakdown"):
        return False
    aggregate = report.get("aggregate")
    pair_stats = report.get("pair_stats")
    return (
        isinstance(pair_stats, list)
        and isinstance(aggregate, dict)
        and "magnitude_mean_mean" in aggregate
    )


def normalize_report_schema(report: dict[str, Any]) -> dict[str, Any]:
    """
    Convert legacy / alternate optical-flow JSON into schema 1.1 score_breakdown.

    Handles RAFT ffpp_vox per-file JSON that stores pair_stats + magnitude_* aggregate
    (e.g. run_id raft-ffpp-vox-benchmark-20260622-0523).
    """
    if not _is_raft_pair_stats_schema(report):
        return report

    pair_stats = report.get("pair_stats") or []
    agg = report.get("aggregate") or {}

    per_pair: list[dict[str, Any]] = []
    for idx, pair in enumerate(pair_stats):
        mag_mean = float(pair.get("magnitude_mean", 0.0))
        mag_std = float(pair.get("magnitude_std", 0.0))
        per_pair.append(
            {
                "pair_index": idx,
                "frame_index_start": pair.get("frame_index_a"),
                "frame_index_end": pair.get("frame_index_b"),
                "flow_mag_mean": mag_mean,
                "flow_mag_max": float(pair.get("magnitude_max", 0.0)),
                "flow_mag_std": mag_std,
                "flow_mag_median": float(pair.get("magnitude_median", 0.0)),
                "flow_u_mean": float(pair.get("flow_x_mean", 0.0)),
                "flow_v_mean": float(pair.get("flow_y_mean", 0.0)),
                "spatial_inconsistency": mag_std / (mag_mean + 1e-6),
                "motion_energy": mag_mean**2,
                "angle_dispersion": float(pair.get("angle_std", 0.0)),
            }
        )

    flow_mag_mean = float(agg.get("magnitude_mean_mean", 0.0))
    flow_mag_max = float(agg.get("magnitude_max_mean", 0.0))
    flow_mag_std = float(agg.get("magnitude_std_mean", 0.0))
    internal_aggregate = {
        "flow_mag_mean": flow_mag_mean,
        "flow_mag_max": flow_mag_max,
        "flow_mag_std": flow_mag_std,
        "spatial_inconsistency_mean": flow_mag_std / (flow_mag_mean + 1e-6),
        "motion_energy_mean": flow_mag_mean**2,
        "angle_dispersion_mean": float(agg.get("angle_std_mean", 0.0)),
        "temporal_jitter": _temporal_jitter(per_pair),
        "frame_pairs": int(agg.get("pair_count", len(pair_stats))),
    }

    report.setdefault("flow_mean", flow_mag_mean)
    report.setdefault("flow_max", flow_mag_max)
    report.setdefault("flow_std", flow_mag_std)
    report.setdefault("frame_pairs_used", internal_aggregate["frame_pairs"])

    report["score_breakdown"] = {
        "schema_version": SCHEMA_VERSION,
        "method": "optical_flow_per_frame_pair",
        "threshold": DEFAULT_ANOMALY_THRESHOLD,
        "frames_sampled": report.get("frames_sampled"),
        "frame_pairs_used": internal_aggregate["frame_pairs"],
        "aggregate": internal_aggregate,
        "score_stats": _score_stats_from_pairs(per_pair) if per_pair else {},
        "segments": _build_segments(per_pair, segment_count=DEFAULT_SEGMENT_COUNT) if per_pair else [],
        "per_frame_pair": per_pair,
    }
    return report


def normalize_reports_schema(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_report_schema(report) for report in reports]


def _motion_anomaly_score(aggregate: dict[str, float], cohort: dict[str, float]) -> float:
    if not cohort:
        return 0.0
    z_parts: list[float] = []
    mapping = [
        ("temporal_jitter", "temporal_jitter_median", "temporal_jitter_std"),
        ("spatial_inconsistency_mean", "spatial_inconsistency_median", "spatial_inconsistency_std"),
        ("angle_dispersion_mean", "angle_dispersion_median", "angle_dispersion_std"),
        ("flow_mag_mean", "flow_mag_mean_median", "flow_mag_mean_std"),
    ]
    for value_key, med_key, std_key in mapping:
        z = (aggregate[value_key] - cohort[med_key]) / cohort[std_key]
        z_parts.append(max(0.0, z))
    if not z_parts:
        return 0.0
    raw = float(np.mean(z_parts))
    return float(min(1.0, raw / 3.0))


def enrich_reports_with_scores(
    reports: list[dict[str, Any]],
    threshold: float = DEFAULT_ANOMALY_THRESHOLD,
) -> dict[str, float]:
    normalize_reports_schema(reports)

    # predictions.json?먯꽌 ?쒓났?섎뒗 ?꾨뱶紐낆씠 `label` ??寃쎌슦媛 ?덉뼱??
    # downstream(benchmark_report 鍮뚮뜑)??湲곕??섎뒗 `ground_truth_label`濡??뺢퇋?뷀빀?덈떎.
    for report in reports:
        if "ground_truth_label" not in report and "label" in report:
            report["ground_truth_label"] = report.get("label")

    cohort = _cohort_baselines(reports)

    # score_breakdown(?곸꽭 遺꾩꽍 寃곌낵)???녿뒗 "理쒖냼?? predictions.json留??덈뒗 寃쎌슦瑜??꾪빐
    # flow_mean 湲곕컲 fallback cohort瑜?蹂꾨룄濡?以鍮꾪빀?덈떎.
    real_flow_means = [
        r.get("flow_mean")
        for r in reports
        if r.get("status") == "ok" and r.get("ground_truth_label") == "real" and r.get("flow_mean") is not None
    ]
    flow_cohort: dict[str, float] = {}
    if real_flow_means:
        med = float(np.median(real_flow_means))
        std = float(np.std(real_flow_means)) or 1e-6
        flow_cohort = {"flow_mag_mean_median": med, "flow_mag_mean_std": std}

    def _motion_anomaly_score_from_flow_mean(flow_mean: float | None) -> float | None:
        if flow_mean is None or not flow_cohort:
            return None
        z = (float(flow_mean) - flow_cohort["flow_mag_mean_median"]) / flow_cohort["flow_mag_mean_std"]
        # ?먮옒 _motion_anomaly_score??4媛???ぉ??z_parts ?됯퇏??/3 ??clamp?⑸땲??
        # 理쒖냼?뺤뿉?쒕뒗 flow_mean 1媛쒕쭔 ?곕?濡??숈씪???ㅼ??쇰줈 /3 clamp???좎??⑸땲??
        raw = max(0.0, z) / 3.0
        return float(min(1.0, raw))

    for report in reports:
        if report.get("status") != "ok":
            report["motion_anomaly_score"] = None
            report["pred_label"] = None
            continue

        # ?곸꽭(score_breakdown) ?덉쑝硫?湲곗〈 濡쒖쭅 洹몃?濡??ъ슜
        if "score_breakdown" in report:
            aggregate = report["score_breakdown"]["aggregate"]
            score = _motion_anomaly_score(aggregate, cohort)
            report["motion_anomaly_score"] = round(score, 6)
            report["pred_label"] = "fake" if score >= threshold else "real"
            report["score_breakdown"]["threshold"] = threshold
            report["score_breakdown"]["motion_anomaly_score"] = report["motion_anomaly_score"]
            report["score_breakdown"]["pred_label"] = report["pred_label"]
            report["interpretation"] = _build_interpretation(report, cohort, threshold)
            continue

        # 理쒖냼??寃곌낵: score_breakdown???놁쑝誘濡?flow_mean 湲곕컲 heuristic?쇰줈 pred瑜?蹂듭썝
        score = _motion_anomaly_score_from_flow_mean(report.get("flow_mean"))
        if score is None:
            report["motion_anomaly_score"] = None
            report["pred_label"] = None
            continue

        report["motion_anomaly_score"] = round(score, 6)
        report["pred_label"] = "fake" if score >= threshold else "real"

        # UI/HTML?먯꽌 李멸퀬?????덈룄濡?理쒖냼 score_breakdown 援ъ“留?梨꾩썎?덈떎.
        report["score_breakdown"] = {
            "threshold": threshold,
            "motion_anomaly_score": report["motion_anomaly_score"],
            "pred_label": report["pred_label"],
            "aggregate": {
                "flow_mag_mean": float(report.get("flow_mean") or 0.0),
            },
            "per_frame_pair": [],
            "score_stats": {},
            "segments": None,
        }
    return cohort


def analyze_video(
    backend: OpticalFlowBackend,
    video_path: Path,
    label: str,
    max_frames: int = 32,
) -> VideoFlowResult:
    started = time.perf_counter()
    try:
        frames = sample_frames(video_path, max_frames=max_frames)
        stats = aggregate_flow_stats(backend, frames)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return VideoFlowResult(
            video_path=str(video_path),
            label=label,
            model=backend.name,
            status="ok",
            flow_mean=stats.flow_mean,
            flow_max=stats.flow_max,
            flow_std=stats.flow_std,
            frame_pairs=stats.frame_pairs,
            elapsed_ms=elapsed_ms,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return VideoFlowResult(
            video_path=str(video_path),
            label=label,
            model=backend.name,
            status="error",
            elapsed_ms=elapsed_ms,
            error=str(exc),
        )


def run_directory_benchmark(
    backend: OpticalFlowBackend,
    fake_dir: Path,
    real_dir: Path,
    run_id: str,
    root: Path,
    max_frames: int = 32,
    threshold: float = DEFAULT_ANOMALY_THRESHOLD,
    segment_count: int = DEFAULT_SEGMENT_COUNT,
    resume: bool = False,
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    tasks = list_benchmark_tasks(fake_dir, real_dir)
    run_dir = run_dir_for(root, run_id)
    json_dir = run_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    progress = BenchmarkProgress(backend.name, len(tasks), run_id)
    if resume and (run_dir / "checkpoint.json").is_file():
        print(f"[{backend.name}] resuming run_id={run_id} from {run_dir}", flush=True)
    elif show_progress:
        print(f"[{backend.name}] run_id={run_id} total={len(tasks)} output={run_dir}", flush=True)

    for label, video in tasks:
        out_path = report_json_path(run_dir, video)
        existing = load_saved_report(out_path) if resume else None
        if existing is not None:
            if show_progress:
                progress.tick(file_name=video.name, status=existing.get("status", "ok"), skipped=True)
            write_checkpoint(
                run_dir,
                run_id=run_id,
                model=backend.name,
                total=len(tasks),
                completed=progress.completed,
                skipped=progress.skipped,
                fake_dir=fake_dir,
                real_dir=real_dir,
                max_frames=max_frames,
                threshold=threshold,
                finished=False,
            )
            continue

        started = time.perf_counter()
        report = analyze_video_detailed(
            backend,
            video,
            label,
            run_id=run_id,
            max_frames=max_frames,
            segment_count=segment_count,
        )
        elapsed_sec = time.perf_counter() - started
        save_report_json(out_path, report)
        if show_progress:
            progress.tick(
                file_name=video.name,
                status=report.get("status", "error"),
                skipped=False,
                elapsed_sec=elapsed_sec,
            )
        write_checkpoint(
            run_dir,
            run_id=run_id,
            model=backend.name,
            total=len(tasks),
            completed=progress.completed,
            skipped=progress.skipped,
            fake_dir=fake_dir,
            real_dir=real_dir,
            max_frames=max_frames,
            threshold=threshold,
            finished=False,
        )

    reports: list[dict[str, Any]] = []
    for _label, video in tasks:
        loaded = load_saved_report(report_json_path(run_dir, video))
        if loaded is not None:
            reports.append(loaded)

    cohort = enrich_reports_with_scores(reports, threshold=threshold)
    for _label, video in tasks:
        stem_path = report_json_path(run_dir, video)
        for report in reports:
            if Path(report["source_path"]).stem == video.stem:
                save_report_json(stem_path, report)
                break

    write_checkpoint(
        run_dir,
        run_id=run_id,
        model=backend.name,
        total=len(tasks),
        completed=len(tasks),
        skipped=progress.skipped,
        fake_dir=fake_dir,
        real_dir=real_dir,
        max_frames=max_frames,
        threshold=threshold,
        finished=True,
    )
    if show_progress:
        print(progress.finish_message(), flush=True)
    return reports, cohort


def make_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


def _run_metrics(reports: list[dict[str, Any]], cohort: dict[str, float]) -> dict[str, Any]:
    ok = [r for r in reports if r.get("status") == "ok"]
    fake_ok = [r for r in ok if r.get("ground_truth_label") == "fake"]
    real_ok = [r for r in ok if r.get("ground_truth_label") == "real"]
    scored = [r for r in ok if r.get("motion_anomaly_score") is not None]

    def mean_score(items: list[dict[str, Any]]) -> float | None:
        vals = [r["motion_anomaly_score"] for r in items if r.get("motion_anomaly_score") is not None]
        return float(np.mean(vals)) if vals else None

    accuracy = None
    if scored:
        correct = sum(1 for r in scored if r.get("pred_label") == r.get("ground_truth_label"))
        accuracy = correct / len(scored)

    return {
        "schema_version": SCHEMA_VERSION,
        "method": "optical_flow_motion_heuristic",
        "note": "accuracy uses motion_anomaly_score threshold vs ground truth; not CNN accuracy.",
        "count": len(reports),
        "ok": len(ok),
        "error": len(reports) - len(ok),
        "fake_count": sum(1 for r in reports if r.get("ground_truth_label") == "fake"),
        "real_count": sum(1 for r in reports if r.get("ground_truth_label") == "real"),
        "motion_anomaly_score_mean_fake": mean_score(fake_ok),
        "motion_anomaly_score_mean_real": mean_score(real_ok),
        "heuristic_accuracy": accuracy,
        "real_cohort_baseline": cohort,
    }


def _file_stem(item: dict[str, Any]) -> str:
    source = item.get("source_path") or item.get("file") or "unknown"
    return Path(source).stem


def _safe_logit(prob: float, eps: float = 1e-6) -> float:
    p = min(max(float(prob), eps), 1.0 - eps)
    return float(math.log(p / (1.0 - p)))


def _binary_entropy(prob: float, eps: float = 1e-6) -> float:
    p = min(max(float(prob), eps), 1.0 - eps)
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2))


def _pair_aggregate_from_pair(pair: dict[str, Any]) -> dict[str, float]:
    return {
        "temporal_jitter": 0.0,
        "spatial_inconsistency_mean": float(pair["spatial_inconsistency"]),
        "angle_dispersion_mean": float(pair["angle_dispersion"]),
        "flow_mag_mean": float(pair["flow_mag_mean"]),
    }


def _compute_pair_votes(
    per_pair: list[dict[str, Any]],
    cohort: dict[str, float],
    threshold: float,
) -> tuple[dict[str, int], list[float]]:
    if not cohort or not per_pair:
        return {"fake": 0, "real": 0}, []
    fake_votes = 0
    real_votes = 0
    pair_scores: list[float] = []
    for pair in per_pair:
        score = _motion_anomaly_score(_pair_aggregate_from_pair(pair), cohort)
        pair_scores.append(round(score, 6))
        if score >= threshold:
            fake_votes += 1
        else:
            real_votes += 1
    return {"fake": fake_votes, "real": real_votes}, pair_scores


def _build_benchmark_item(
    report: dict[str, Any],
    threshold: float,
    cohort: dict[str, float],
) -> dict[str, Any]:
    gt = report.get("ground_truth_label")
    status = report.get("status")
    item: dict[str, Any] = {
        "file": report.get("file"),
        "ground_truth_label": gt,
        "status": status,
    }
    if status != "ok":
        item.update(
            {
                "pred_label": None,
                "correct": None,
                "error": report.get("error"),
            }
        )
        return item

    score = float(report["motion_anomaly_score"])
    pred = report.get("pred_label")
    prob_fake = score
    prob_real = 1.0 - score
    breakdown = report.get("score_breakdown") or {}
    per_pair = breakdown.get("per_frame_pair") or []
    score_stats = dict(breakdown.get("score_stats") or {})
    pair_votes, pair_scores = _compute_pair_votes(per_pair, cohort, threshold)
    if pair_scores:
        score_stats["pair_anomaly_score"] = distribution_stats(pair_scores)

    aggregate = breakdown.get("aggregate") or {}
    item.update(
        {
            "pred_label": pred,
            "correct": gt == pred,
            "fake_score": round(prob_fake, 6),
            "motion_anomaly_score": round(score, 6),
            "prob_fake": round(prob_fake, 6),
            "prob_real": round(prob_real, 6),
            "logit_fake": round(_safe_logit(prob_fake), 4),
            "logit_real": round(_safe_logit(prob_real), 4),
            "margin": round(prob_fake - prob_real, 4),
            "entropy": round(_binary_entropy(prob_fake), 4),
            "confidence": round(abs(prob_fake - 0.5) * 2.0, 4),
            "frames_used": report.get("frames_sampled"),
            "frame_pairs_used": report.get("frame_pairs_used"),
            "flow_mean": report.get("flow_mean"),
            "flow_max": report.get("flow_max"),
            "flow_std": report.get("flow_std"),
            "temporal_jitter": aggregate.get("temporal_jitter"),
            "spatial_inconsistency_mean": aggregate.get("spatial_inconsistency_mean"),
            "angle_dispersion_mean": aggregate.get("angle_dispersion_mean"),
            "motion_energy_mean": aggregate.get("motion_energy_mean"),
            "pair_votes": pair_votes,
            "frame_votes": pair_votes,
            "score_stats": score_stats,
            "segments": breakdown.get("segments"),
            "per_frame_pair": per_pair,
            "interpretation": report.get("interpretation"),
            "elapsed_ms": report.get("elapsed_ms"),
            "json_path": f"json/{_file_stem(report)}.json",
        }
    )
    return item


def build_benchmark_report_document(
    report_list: list[dict[str, Any]],
    *,
    run_id: str,
    model_name: str,
    profile: str | None,
    threshold: float,
    cohort: dict[str, float],
    fake_dir: Path | None,
    real_dir: Path | None,
) -> dict[str, Any]:
    metrics = _run_metrics(report_list, cohort)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model": model_name,
        "profile": profile,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "method": BENCHMARK_METHOD,
        "count": len(report_list),
        "fake_dir": str(fake_dir) if fake_dir else None,
        "real_dir": str(real_dir) if real_dir else None,
        "metrics": metrics,
        "items": [_build_benchmark_item(item, threshold, cohort) for item in report_list],
        "report_html": "benchmark_report.html",
    }


# Backward-compatible alias
build_toggle_summary_document = build_benchmark_report_document

# Labels for HTML report UI
_METRIC_GLOSSARY: dict[str, tuple[str, str]] = {
    "motion_anomaly_score": ("anomaly score", "0-1 heuristic vs real cohort"),
    "flow_mean": ("flow mean", "mean optical flow magnitude"),
    "flow_max": ("flow max", "peak flow magnitude"),
    "flow_mag_mean": ("pair flow mean", "mean flow per frame pair"),
    "flow_mag_max": ("pair flow max", "max flow per frame pair"),
    "flow_mag_std": ("pair flow std", "flow std per frame pair"),
    "spatial_inconsistency_mean": ("spatial inconsistency", "std/mean across pairs"),
    "motion_energy_mean": ("motion energy", "mean squared flow"),
    "angle_dispersion_mean": ("angle dispersion", "flow direction spread"),
    "temporal_jitter": ("temporal jitter", "frame-to-frame motion variation"),
    "frame_pairs": ("frame pairs", "number of (t,t+1) pairs analyzed"),
    "spatial_inconsistency": ("spatial inconsistency", "per-pair std/mean"),
    "angle_dispersion": ("angle dispersion", "per-pair direction spread"),
}

_OVERALL_METRICS_KO: dict[str, tuple[str, str]] = {
    "heuristic_accuracy": ("heuristic accuracy", "motion score threshold match rate (not CNN accuracy)"),
    "motion_anomaly_score_mean_fake": ("fake mean score", "mean motion_anomaly_score for fake"),
    "motion_anomaly_score_mean_real": ("real mean score", "mean motion_anomaly_score for real"),
    "count": ("total", "number of items"),
    "ok": ("ok", "successful flow analysis"),
    "error": ("error", "decode/flow errors"),
    "fake_count": ("fake", "fake items"),
    "real_count": ("real", "real items"),
}

_SIGNAL_NAMES_KO: dict[str, str] = {
    "temporal_jitter": "temporal jitter",
    "spatial_inconsistency": "spatial inconsistency",
    "angle_dispersion": "angle dispersion",
    "flow_magnitude": "flow magnitude",
}


def _glossary(key: str) -> tuple[str, str]:
    if key in _METRIC_GLOSSARY:
        return _METRIC_GLOSSARY[key]
    if key in _OVERALL_METRICS_KO:
        label, hint = _OVERALL_METRICS_KO[key]
        return label, hint or ""
    return key, ""


def _fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _label_badge(label: str | None) -> str:
    if label == "fake":
        return '<span class="badge fake">fake</span>'
    if label == "real":
        return '<span class="badge real">real</span>'
    return f'<span class="badge">{html.escape(label or "-")}</span>'


def _render_labeled_table(rows: list[tuple[str, Any, str]]) -> str:
    body = []
    for label, value, hint in rows:
        hint_html = f'<div class="hint">{html.escape(hint)}</div>' if hint else ""
        if isinstance(value, (int, float)):
            val_html = _fmt_num(value)
        else:
            val_html = html.escape(str(value))
        body.append(f"<tr><td>{html.escape(label)}{hint_html}</td><td>{val_html}</td></tr>")
    return f'<table class="labeled"><tbody>{"".join(body)}</tbody></table>'


def _render_simple_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = []
        for cell in row:
            if isinstance(cell, (int, float)):
                cells.append(f"<td>{_fmt_num(cell)}</td>")
            else:
                cells.append(f"<td>{html.escape(str(cell))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def _render_overall_metrics(metrics: dict[str, Any]) -> str:
    priority = [
        "heuristic_accuracy",
        "motion_anomaly_score_mean_fake",
        "motion_anomaly_score_mean_real",
        "count",
        "ok",
        "error",
        "fake_count",
        "real_count",
    ]
    rows: list[tuple[str, Any, str]] = []
    for key in priority:
        if key in metrics and metrics[key] is not None:
            label, hint = _OVERALL_METRICS_KO.get(key, (key, ""))
            rows.append((label, metrics[key], hint or ""))
    cohort = metrics.get("real_cohort_baseline")
    html_parts = [_render_labeled_table(rows)] if rows else ""
    if isinstance(cohort, dict) and cohort:
        cohort_rows = []
        for key, val in sorted(cohort.items()):
            label, hint = _glossary(key.replace("_median", "").replace("_std", ""))
            if "_median" in key:
                label = f"{label} (real median)"
            elif "_std" in key:
                label = f"{label} (real std)"
            cohort_rows.append((label, val, hint))
        html_parts.append(
            '<details class="sub"><summary>real baseline (cohort)</summary>'
            f'<div class="sub-body">{_render_labeled_table(cohort_rows)}</div></details>'
        )
    note = metrics.get("note")
    if note:
        html_parts.append(f'<p class="note">{html.escape(str(note))}</p>')
    return "".join(html_parts)


def _render_file_body(item: dict[str, Any], threshold: float) -> str:
    status = item.get("status")
    if status != "ok":
        return f'<p class="error">failed: {html.escape(str(item.get("error") or "error"))}</p>'

    gt = item.get("ground_truth_label")
    pred = item.get("pred_label")
    correct = item.get("correct")
    match_txt = "match" if correct else "mismatch"
    match_cls = "ok-text" if correct else "warn-text"
    parts: list[str] = [
        f'<div class="verdict">'
        f'<div class="verdict-row"><span>truth</span>{_label_badge(gt)}</div>'
        f'<div class="verdict-row"><span>pred</span>{_label_badge(pred)} <em class="{match_cls}">{match_txt}</em></div>'
        f'<div class="verdict-row"><span>fake_score</span><strong>{_fmt_num(item.get("fake_score"))}</strong>'
        f' <span class="note">(threshold {_fmt_num(threshold)})</span></div>'
        f"</div>"
    ]

    key_rows = [
        (_glossary("motion_anomaly_score")[0], item.get("motion_anomaly_score"), _glossary("motion_anomaly_score")[1]),
        (_glossary("temporal_jitter")[0], item.get("temporal_jitter"), _glossary("temporal_jitter")[1]),
        (_glossary("spatial_inconsistency_mean")[0], item.get("spatial_inconsistency_mean"), _glossary("spatial_inconsistency_mean")[1]),
        (_glossary("angle_dispersion_mean")[0], item.get("angle_dispersion_mean"), _glossary("angle_dispersion_mean")[1]),
        (_glossary("flow_mean")[0], item.get("flow_mean"), _glossary("flow_mean")[1]),
        (_glossary("flow_max")[0], item.get("flow_max"), _glossary("flow_max")[1]),
    ]
    parts.append("<h4>key metrics</h4>")
    parts.append(_render_labeled_table(key_rows))

    votes = item.get("frame_votes") or item.get("pair_votes") or {}
    parts.append(
        f'<p class="note">frame_votes: fake {votes.get("fake", 0)}, real {votes.get("real", 0)} '
        f'(frames_used={item.get("frames_used")}, pairs={item.get("frame_pairs_used")})</p>'
    )

    interpretation = item.get("interpretation") or {}
    signals = interpretation.get("signals") or []
    if signals:
        parts.append("<h4>vs real cohort</h4>")
        sig_rows = [
            [
                _SIGNAL_NAMES_KO.get(sig.get("name", ""), sig.get("name")),
                _fmt_num(sig.get("value")),
                _fmt_num(sig.get("real_cohort_baseline")),
                sig.get("note"),
            ]
            for sig in signals
        ]
        parts.append(_render_simple_table(["metric", "value", "real baseline", "note"], sig_rows))

    pairs = item.get("per_frame_pair") or []
    if pairs:
        pair_rows = [
            [
                p.get("pair_index"),
                f"{p.get('frame_index_start')}-{p.get('frame_index_end')}",
                p.get("flow_mag_mean"),
                p.get("spatial_inconsistency"),
                p.get("angle_dispersion"),
            ]
            for p in pairs[:12]
        ]
        parts.append(
            f'<details class="sub"><summary>frame pairs ({len(pairs)} total, first 12)</summary><div class="sub-body">'
            + _render_simple_table(["#", "frames", "flow", "spatial", "angle"], pair_rows)
            + "</div></details>"
        )

    json_path = item.get("json_path")
    if json_path:
        parts.append(f'<p class="note">json: <code>{html.escape(json_path)}</code></p>')
    return "".join(parts)


def generate_benchmark_html(document: dict[str, Any]) -> str:
    metrics = document.get("metrics") or {}
    items = document.get("items") or []
    profile = document.get("profile") or "-"
    run_id = document.get("run_id") or "-"
    model = document.get("model") or "-"
    threshold = document.get("threshold")
    count = document.get("count", len(items))
    ok_count = metrics.get("ok", sum(1 for i in items if i.get("status") == "ok"))

    file_sections: list[str] = []
    for item in items:
        file_name = item.get("file") or "unknown"
        gt = item.get("ground_truth_label")
        pred = item.get("pred_label")
        score = item.get("fake_score")
        status = item.get("status")
        correct = item.get("correct")
        match = correct is True and status == "ok"
        match_cls = "match" if match else "mismatch"
        if status != "ok":
            title = f"{html.escape(file_name)} | <span class='warn-text'>error</span>"
        else:
            match_icon = "OK" if match else "X"
            title = (
                f"{html.escape(file_name)} | "
                f"truth {_label_badge(gt)} pred {_label_badge(pred)} {match_icon} | "
                f"fake_score {_fmt_num(score)}"
            )
        body = _render_file_body(item, float(threshold or DEFAULT_ANOMALY_THRESHOLD))
        file_sections.append(
            f'<details class="file {match_cls}"><summary>{title}</summary>'
            f'<div class="file-body">{body}</div></details>'
        )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(model)} benchmark - {html.escape(profile)}</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a2332;
      --text: #e7ecf3;
      --muted: #9aa7b8;
      --fake: #f87171;
      --real: #4ade80;
      --match: #14532d;
      --mismatch: #7f1d1d;
      --border: #2d3a4d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Malgun Gothic", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 1.5rem;
    }}
    h1, h2, h3, h4 {{ margin: 0 0 0.75rem; }}
    h4 {{ font-size: 0.95rem; color: #cbd5e1; margin-top: 1rem; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem 1.25rem;
      margin-bottom: 1rem;
    }}
    .meta {{ color: var(--muted); font-size: 0.95rem; }}
    .badge {{
      display: inline-block;
      padding: 0.1rem 0.45rem;
      border-radius: 999px;
      font-size: 0.8rem;
      background: #334155;
    }}
    .badge.fake {{ background: rgba(248,113,113,0.2); color: var(--fake); }}
    .badge.real {{ background: rgba(74,222,128,0.2); color: var(--real); }}
    details.file {{
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 0.5rem;
      background: #121a26;
    }}
    details.file.match {{ border-left: 4px solid var(--real); }}
    details.file.mismatch {{ border-left: 4px solid var(--fake); }}
    details.file > summary {{
      cursor: pointer;
      padding: 0.75rem 1rem;
      font-weight: 600;
      list-style: none;
    }}
    details.file > summary::-webkit-details-marker {{ display: none; }}
    details.file[open] > summary {{ border-bottom: 1px solid var(--border); }}
    details.sub {{
      border: 1px dashed var(--border);
      border-radius: 6px;
      margin: 0.75rem 0;
      background: #0f1724;
    }}
    details.sub > summary {{
      cursor: pointer;
      padding: 0.5rem 0.75rem;
      color: #cbd5e1;
      font-size: 0.9rem;
    }}
    .sub-body {{ padding: 0 0.75rem 0.75rem; }}
    .file-body {{ padding: 1rem; }}
    .verdict {{
      background: #0f1724;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem 1rem;
      margin-bottom: 0.5rem;
    }}
    .verdict-row {{ margin-bottom: 0.35rem; }}
    .verdict-row span {{ color: var(--muted); margin-right: 0.5rem; }}
    .verdict-note {{ margin: 0.5rem 0 0; color: var(--muted); font-size: 0.9rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
      margin-bottom: 0.5rem;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 0.35rem 0.5rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #243044; }}
    table.labeled td:first-child {{ width: 42%; }}
    .hint {{ color: var(--muted); font-size: 0.78rem; margin-top: 0.15rem; }}
    .note {{ color: var(--muted); font-size: 0.85rem; }}
    .error {{ color: var(--fake); }}
    .ok-text {{ color: var(--real); font-style: normal; }}
    .warn-text {{ color: var(--fake); font-style: normal; }}
    code {{ background: #243044; padding: 0.1rem 0.35rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>{html.escape(model)} / {html.escape(profile)}</h1>
    <p class="meta">
      run_id: {html.escape(run_id)} |
      threshold: {_fmt_num(threshold)} |
      items {count} (ok {ok_count})
    </p>
    <p class="note">
      Optical-flow <strong>motion heuristic</strong> benchmark.
      Not comparable to CNN classifier accuracy.
    </p>
  </div>
  <div class="panel">
    <h2>Overall</h2>
    {_render_overall_metrics(metrics)}
  </div>
  <div class="panel">
    <h2>Per video ({len(items)})</h2>
    <p class="note">Expand each row for metrics and frame-pair details.</p>
    {"".join(file_sections)}
  </div>
</body>
</html>
"""


def write_run_outputs(
    root: Path,
    run_id: str,
    model_name: str,
    reports: Iterable[dict[str, Any]],
    cohort: dict[str, float] | None = None,
    threshold: float = DEFAULT_ANOMALY_THRESHOLD,
    profile: str | None = None,
    fake_dir: Path | None = None,
    real_dir: Path | None = None,
) -> Path:
    run_dir = root / "results" / "infer" / run_id
    json_dir = run_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    report_list = list(reports)
    for item in report_list:
        stem = _file_stem(item)
        out_path = json_dir / f"{stem}.json"
        if not out_path.is_file():
            save_report_json(out_path, item)

    toggle_doc = build_benchmark_report_document(
        report_list,
        run_id=run_id,
        model_name=model_name,
        profile=profile,
        threshold=threshold,
        cohort=cohort or {},
        fake_dir=fake_dir,
        real_dir=real_dir,
    )

    (run_dir / "summary.json").write_text(
        json.dumps(toggle_doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "benchmark_report.json").write_text(
        json.dumps(toggle_doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "benchmark_report.html").write_text(
        generate_benchmark_html(toggle_doc),
        encoding="utf-8",
    )

    predictions = {
        "schema_version": SCHEMA_VERSION,
        "runId": run_id,
        "model": model_name,
        "count": len(report_list),
        "items": report_list,
    }
    (run_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (run_dir / f"infer_summary_{model_name}.json").write_text(
        json.dumps(toggle_doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / f"metrics_{model_name}.json").write_text(
        json.dumps(toggle_doc["metrics"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir
