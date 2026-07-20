"""Single-video TruFor (spatial) + TimeSformer (temporal forgery) inference for GPU worker."""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("gpu_worker.forgery")

VIDEO_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ForgeryInferConfig:
    enabled: bool = True
    repo_root: Path = Path.home() / "forenShield-ai"
    forgery_root: Path | None = None
    trufor_ckpt: Path | None = None
    timesformer_ckpt: Path | None = None
    trufor_num_frames: int = 16
    trufor_max_side: int = 720
    trufor_aggregate: str = "top3_mean"
    trufor_threshold: float = 0.515
    ts_aggregate: str = "max"
    ts_threshold: float = 0.173386
    ts_top_k: int = 3
    ts_window_sec: float = 1.0
    ts_stride_sec: float = 0.5
    ts_clip_frames: int = 8
    ts_max_side: int = 512
    device: str = "cuda"
    gpu: int = 0
    decode_cache: Path | None = None

    @staticmethod
    def from_worker_config(cfg: Any) -> ForgeryInferConfig:
        repo = Path(getattr(cfg, "project_root", Path.home() / "forenShield-ai")).expanduser().resolve()
        forgery_root = _resolve_forgery_root(repo)
        device = str(getattr(cfg, "device", "cuda") or "cuda")
        return ForgeryInferConfig(
            enabled=_env_bool("FORGERY_ENABLED", True),
            repo_root=repo,
            forgery_root=forgery_root,
            trufor_ckpt=_resolve_path(
                os.getenv("TRUFOR_CKPT", ""),
                forgery_root
                / "models/train/spatial/trufor/videocof-v2/trufor-videocof-v2-20260710-0800/trufor_videocof_v2_ft/best.pth.tar",
            ),
            timesformer_ckpt=_resolve_path(
                os.getenv("FORGERY_TS_CKPT", ""),
                forgery_root
                / "models/train/temporal/timesformer-forgery/timesformer-forgery-v1.9-hardneg-20260714-0342/forgery_head.pt",
            ),
            trufor_num_frames=int(os.getenv("TRUFOR_NUM_FRAMES", "16")),
            trufor_max_side=int(os.getenv("TRUFOR_MAX_SIDE", "720")),
            trufor_aggregate=os.getenv("TRUFOR_AGGREGATE", "top3_mean"),
            trufor_threshold=float(os.getenv("TRUFOR_THRESHOLD", "0.515")),
            ts_aggregate=os.getenv("FORGERY_TS_AGGREGATE", "max"),
            ts_threshold=float(os.getenv("FORGERY_TS_THRESHOLD", "0.173386")),
            ts_top_k=int(os.getenv("FORGERY_TS_TOP_K", "3")),
            ts_window_sec=float(os.getenv("FORGERY_TS_WINDOW_SEC", "1.0")),
            ts_stride_sec=float(os.getenv("FORGERY_TS_STRIDE_SEC", "0.5")),
            ts_clip_frames=int(os.getenv("FORGERY_TS_CLIP_FRAMES", "8")),
            ts_max_side=int(os.getenv("FORGERY_TS_MAX_SIDE", "512")),
            device=device,
            gpu=_gpu_index(device),
            decode_cache=forgery_root / "cache/decode-mp4-temporal",
        )


@dataclass
class ForgeryLaneResult:
    lane_ran: bool = False
    spatial_score: float = 0.0
    temporal_score: float = 0.0
    spatial_detected: bool = False
    temporal_detected: bool = False
    frame_risks: list[dict[str, Any]] = field(default_factory=list)
    clip_risks: list[dict[str, Any]] = field(default_factory=list)
    spatial_segments: list[dict[str, Any]] = field(default_factory=list)
    temporal_segments: list[dict[str, Any]] = field(default_factory=list)
    model_spatial_version: str = "videocof-v2"
    model_temporal_version: str = "v1.8-csvted-v4"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _gpu_index(device: str) -> int:
    if device.startswith("cuda") and ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def _resolve_forgery_root(repo: Path) -> Path:
    explicit = os.getenv("FORGERY_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidate = repo / "forgery"
    return candidate.resolve() if candidate.is_dir() else repo.resolve()


def _resolve_path(env_value: str, default: Path) -> Path:
    if env_value.strip():
        p = Path(env_value.strip()).expanduser()
        if p.is_file():
            return p.resolve()
        repo = Path(os.getenv("FORENSHIELD_AI_ROOT", Path.home() / "forenShield-ai")).expanduser()
        alt = repo / env_value.strip()
        if alt.is_file():
            return alt.resolve()
    return default.expanduser().resolve()


def _setup_forgery_imports(repo_root: Path, forgery_root: Path) -> None:
    for cand in (
        forgery_root / "scripts" / "infer",
        repo_root / "forgery" / "scripts" / "infer",
        repo_root / "scripts" / "infer",
    ):
        if cand.is_dir() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))


def _safe_video_stem(video_path: Path) -> str:
    stem = VIDEO_STEM_RE.sub("_", video_path.stem)
    return stem[:120] if len(stem) <= 120 else f"v{abs(hash(stem)) % 10**8:08d}"


def _video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 30.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    return fps if fps > 1e-3 else 30.0


def _probe_bin(name: str) -> str | None:
    override = os.getenv(f"{name.upper()}_PATH", "").strip()
    if override and Path(override).is_file():
        return override
    found = shutil.which(name)
    if found:
        return found
    for candidate in (f"/usr/bin/{name}", f"/usr/local/bin/{name}"):
        if Path(candidate).is_file():
            return candidate
    return None


def _run_probe(cmd: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def _duration_from_ffprobe_format(video_path: Path) -> float:
    ffprobe = _probe_bin("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        probe = _run_probe(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ]
        )
        if probe.returncode != 0 or not (probe.stdout or "").strip():
            logger.warning(
                "ffprobe format duration failed for %s (rc=%s): %s",
                video_path.name,
                probe.returncode,
                (probe.stderr or "").strip()[:200],
            )
            return 0.0
        return round(float(probe.stdout.strip()), 4)
    except Exception as exc:
        logger.warning("ffprobe format duration error for %s: %s", video_path.name, exc)
        return 0.0


def _duration_from_ffprobe(video_path: Path) -> float:
    ffprobe = _probe_bin("ffprobe")
    if not ffprobe:
        logger.warning("ffprobe not found on PATH for %s", video_path.name)
        return 0.0
    try:
        import json

        probe = _run_probe(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=duration",
                "-of",
                "json",
                str(video_path),
            ]
        )
        if probe.returncode != 0:
            logger.warning(
                "ffprobe json duration failed for %s (rc=%s): %s",
                video_path.name,
                probe.returncode,
                (probe.stderr or "").strip()[:200],
            )
            return 0.0
        payload = json.loads(probe.stdout or "{}")
        candidates: list[float] = []
        fmt = (payload.get("format") or {}).get("duration")
        if fmt is not None:
            candidates.append(float(fmt))
        for stream in payload.get("streams") or []:
            dur = stream.get("duration")
            if dur is not None:
                candidates.append(float(dur))
        if candidates:
            return round(max(candidates), 4)
    except Exception as exc:
        logger.warning("ffprobe json duration error for %s: %s", video_path.name, exc)
    return 0.0


def _duration_from_ffmpeg_stderr(video_path: Path) -> float:
    ffmpeg = _probe_bin("ffmpeg")
    if not ffmpeg:
        return 0.0
    try:
        probe = _run_probe([ffmpeg, "-hide_banner", "-i", str(video_path)], timeout=20)
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", probe.stderr or "")
        if not match:
            return 0.0
        hours, minutes, seconds = match.groups()
        return round(int(hours) * 3600 + int(minutes) * 60 + float(seconds), 4)
    except Exception as exc:
        logger.warning("ffmpeg duration parse error for %s: %s", video_path.name, exc)
        return 0.0


def _duration_from_opencv_ratio(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    try:
        # Broken mobile mp4 often reports wrong FRAME_COUNT but correct POS_MSEC at end.
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
        msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if msec > 0:
            return round(msec / 1000.0, 4)
    finally:
        cap.release()
    return 0.0


def _duration_from_opencv_decode(video_path: Path, fps: float) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    meta_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if meta_frames > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, meta_frames - 1)
        ok, _ = cap.read()
        if ok:
            msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if msec > 0:
                cap.release()
                return round(msec / 1000.0, 4)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        decoded += 1
        if decoded > 72000:
            break
    cap.release()
    if decoded > 0 and fps > 1e-3:
        return round(max(decoded - 1, 0) / fps, 4)
    if meta_frames > 0 and fps > 1e-3:
        return round(max(meta_frames - 1, 0) / fps, 4)
    return 0.0


def _video_duration_sec(video_path: Path, fps: float) -> float:
    """Collect duration from all probes; mobile mp4 often reports ~16 decodable frames (~0.6s)."""
    probes: list[tuple[str, float]] = []
    for name, probe in (
        ("ffprobe_format", _duration_from_ffprobe_format),
        ("ffprobe_json", _duration_from_ffprobe),
        ("ffmpeg_stderr", _duration_from_ffmpeg_stderr),
        ("opencv_ratio", _duration_from_opencv_ratio),
        ("opencv_decode", lambda p: _duration_from_opencv_decode(p, fps)),
    ):
        duration = probe(video_path)
        if duration > 0:
            probes.append((name, duration))

    if not probes:
        return 0.0

    chosen = max(probes, key=lambda item: item[1])[1]
    if len(probes) > 1:
        short = min(probes, key=lambda item: item[1])
        if chosen >= 2.0 and short[1] < 2.0 and chosen > short[1] * 2:
            logger.warning(
                "Duration probe conflict for %s: using max=%.3fs over %s=%.3fs (all=%s)",
                video_path.name,
                chosen,
                short[0],
                short[1],
                probes,
            )
        else:
            logger.info("Duration probes for %s: %s -> %.3fs", video_path.name, probes, chosen)
    return round(chosen, 4)


def _sample_timestamps_sec(n: int, duration_sec: float, fps: float, frame_indices: list[int]) -> list[float]:
    """Map each sample to a timeline position (prefer full-clip spread over idx/fps)."""
    if n <= 0:
        return []
    if duration_sec > 0 and n > 1:
        return [round(i * duration_sec / (n - 1), 4) for i in range(n)]
    if n == 1:
        idx = frame_indices[0] if frame_indices else 0
        return [round(idx / fps, 4)]
    return [round(idx / fps, 4) for idx in frame_indices]


def _aggregate(values: list[float], mode: str) -> float:
    if not values:
        return 0.0
    if mode == "top2_mean":
        top = sorted(values, reverse=True)[:2]
        return float(sum(top) / len(top))
    if mode == "top3_mean":
        top = sorted(values, reverse=True)[:3]
        return float(sum(top) / len(top))
    return float(max(values))


def _score_from_trufor_npz(npz_path: Path) -> float | None:
    try:
        data = np.load(npz_path)
        if "score" in data:
            return float(np.asarray(data["score"]).reshape(-1)[0])
        if "map" in data:
            return float(np.max(data["map"]))
    except Exception:
        return None
    return None


def _segments_from_risks(
    risks: list[dict[str, Any]],
    *,
    threshold: float,
    reason: str,
    time_key: str = "timestampSec",
) -> list[dict[str, Any]]:
    hot = [r for r in risks if float(r.get("riskScore", 0.0)) >= threshold]
    if not hot:
        return []
    hot.sort(key=lambda r: float(r.get("riskScore", 0.0)), reverse=True)
    top = hot[0]
    t = float(top.get(time_key, 0.0))
    span = 0.5
    return [
        {
            "startTime": round(max(0.0, t - span / 2), 3),
            "endTime": round(t + span / 2, 3),
            "maxRiskScore": round(float(top.get("riskScore", 0.0)), 6),
            "reason": reason,
        }
    ]


def _resize_saved_frames(frame_paths: list[Path], max_side: int) -> None:
    if max_side < 1:
        return
    for path in frame_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            continue
        scale = max_side / float(longest)
        resized = cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(str(path), resized)


def _frame_index_from_trufor_npz(name: str) -> int | None:
    """Parse frame index from TruFor outputs like stem_f000.jpg or stem_f000123."""
    match = re.search(r"_f(\d+)(?:\.jpg)?$", name, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _reset_dir(path: Path) -> Path:
    """Remove path if present, then recreate empty directory (avoids cross-video npz mix)."""
    import shutil

    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _npz_belongs_to_stem(npz_stem: str, video_stem: str) -> bool:
    """Keep only TruFor outputs for this video stem (ignore leftover files)."""
    # startswith: vendor may use stem_f###### or slight suffix variants
    return npz_stem.lower().startswith(f"{video_stem.lower()}_f")


def _collect_trufor_frame_scores(
    saved_jpgs: list[Path],
    trufor_out: Path,
    video_stem: str,
    *,
    duration_sec: float,
    fps: float,
    video_size: tuple[int, int] | None = None,
) -> list[tuple[float, float, int, list[dict[str, Any]]]]:
    """Pair TruFor npz scores (+ tamper bboxes) with sampled JPEG order."""
    jpgs = sorted(
        [p for p in saved_jpgs if p.is_file()],
        key=lambda p: p.name.lower(),
    )
    npzs = sorted(
        [
            p
            for p in trufor_out.rglob("*.npz")
            if _npz_belongs_to_stem(p.stem, video_stem)
        ],
        key=lambda p: p.name.lower(),
    )

    if not jpgs or not npzs:
        return []

    pair_count = min(len(jpgs), len(npzs))
    if len(jpgs) != len(npzs):
        logger.warning(
            "TruFor: jpg/npz count mismatch (%d vs %d); pairing first %d by sort order",
            len(jpgs),
            len(npzs),
            pair_count,
        )

    video_w, video_h = video_size or (0, 0)

    frame_indices: list[int] = []
    scores: list[float] = []
    bbox_rows: list[list[dict[str, Any]]] = []
    for i in range(pair_count):
        jpg = jpgs[i]
        npz = npzs[i]
        val = _score_from_trufor_npz(npz)
        if val is None:
            continue
        frame_idx = _frame_index_from_trufor_npz(jpg.stem)
        if frame_idx is None:
            frame_idx = i
        frame_indices.append(frame_idx)
        scores.append(float(val))
        bbox_rows.append(_bboxes_from_trufor_pair(jpg, npz, video_w=video_w, video_h=video_h))

    if not scores:
        return []

    timestamps = _sample_timestamps_sec(len(scores), duration_sec, fps, frame_indices)
    return [
        (timestamps[i], scores[i], frame_indices[i], bbox_rows[i])
        for i in range(len(scores))
    ]


def _bboxes_from_trufor_pair(
    jpg_path: Path,
    npz_path: Path,
    *,
    video_w: int,
    video_h: int,
) -> list[dict[str, Any]]:
    """Extract tamper bboxes in original video pixel space."""
    # Ensure ai-forensic repo root is importable (gpu_worker may not have it yet).
    try:
        repo = Path(__file__).resolve().parents[2]  # .../ai-forensic
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from app.services.trufor_overlay import bboxes_from_npz
    except Exception:
        logger.warning("TruFor bbox import failed; overlays will lack boxes", exc_info=True)
        return []

    img = cv2.imread(str(jpg_path))
    if img is None:
        return []
    jh, jw = img.shape[:2]
    target_w = video_w if video_w > 0 else jw
    target_h = video_h if video_h > 0 else jh
    try:
        boxes, _ = bboxes_from_npz(npz_path, jw, jh)
    except Exception:
        logger.warning("TruFor bbox extract failed for %s", npz_path, exc_info=True)
        return []

    sx = target_w / float(jw) if jw else 1.0
    sy = target_h / float(jh) if jh else 1.0
    out: list[dict[str, Any]] = []
    for box in boxes:
        out.append(
            {
                "x": int(round(box.x * sx)),
                "y": int(round(box.y * sy)),
                "w": max(1, int(round(box.w * sx))),
                "h": max(1, int(round(box.h * sy))),
                "score": round(float(box.score), 4),
            }
        )
    if not out:
        logger.info("TruFor bbox empty for %s (map may be flat)", npz_path.name)
    return out


def _video_size(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return w, h


def infer_trufor_spatial(video_path: Path, work_dir: Path, cfg: ForgeryInferConfig) -> tuple[float, list[dict], list[dict]]:
    from spatial_mvtamperbench_benchmark import run_trufor, sample_video_frames  # noqa: WPS433

    if not cfg.trufor_ckpt or not cfg.trufor_ckpt.is_file():
        raise FileNotFoundError(f"TruFor checkpoint not found: {cfg.trufor_ckpt}")

    vendor_test = cfg.forgery_root / "vendor/TruFor/TruFor_train_test/test.py"
    if not vendor_test.is_file():
        raise FileNotFoundError(f"TruFor vendor test.py not found: {vendor_test}")

    fps = _video_fps(video_path)
    duration_sec = _video_duration_sec(video_path, fps)
    if duration_sec > 0 and duration_sec < 2.0 and cfg.trufor_num_frames >= 10:
        logger.warning(
            "TruFor: short duration probe %.3fs for %s (fps=%.2f); timeline may be truncated",
            duration_sec,
            video_path.name,
            fps,
        )
    stem = _safe_video_stem(video_path)
    # Fresh dirs every call — reusing frames/out mixes org+fake scores (false ~0.99).
    frames_dir = _reset_dir(work_dir / "trufor_frames")
    trufor_out = _reset_dir(work_dir / "trufor_out")
    saved = sample_video_frames(video_path, frames_dir, stem, cfg.trufor_num_frames)
    if not saved:
        logger.warning("TruFor: no frames sampled from %s", video_path)
        return 0.0, [], []

    _resize_saved_frames(saved, cfg.trufor_max_side)

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        run_trufor(
            frames_dir,
            trufor_out,
            cfg.trufor_ckpt,
            vendor_test,
            cfg.gpu,
            aggregate=cfg.trufor_aggregate,
        )
    except TypeError:
        # Older spatial_mvtamperbench_benchmark.run_trufor without aggregate kwarg
        run_trufor(
            frames_dir,
            trufor_out,
            cfg.trufor_ckpt,
            vendor_test,
            cfg.gpu,
        )

    # Prefer sampled JPEG order; spread timestamps across full clip duration.
    video_w, video_h = _video_size(video_path)
    frame_scores = _collect_trufor_frame_scores(
        saved,
        trufor_out,
        stem,
        duration_sec=duration_sec,
        fps=fps,
        video_size=(video_w, video_h),
    )

    if not frame_scores:
        logger.warning("TruFor: no npz scores under %s", trufor_out)
        return 0.0, [], []

    values = [s for _, s, _, _ in frame_scores]
    video_score = _aggregate(values, cfg.trufor_aggregate)
    # Keep timeline scores for every sample, but attach localization boxes only when
    # the frame clears TruFor detection threshold (real-tamper display contract).
    frame_risks = [
        {
            "frameIndex": frame_idx,
            "timestampSec": round(ts, 4),
            "riskScore": round(score, 6),
            "bboxes": bboxes if float(score) >= float(cfg.trufor_threshold) else [],
        }
        for ts, score, frame_idx, bboxes in frame_scores
    ]
    bbox_frames = sum(1 for row in frame_risks if row.get("bboxes"))
    logger.info(
        "TruFor frameRisks=%d with_bboxes=%d (threshold=%.3f) duration=%.3fs",
        len(frame_risks),
        bbox_frames,
        float(cfg.trufor_threshold),
        duration_sec,
    )
    if frame_risks:
        max_ts = max(r["timestampSec"] for r in frame_risks)
        logger.info(
            "TruFor timeline: duration=%.3fs fps=%.2f samples=%d max_ts=%.3fs",
            duration_sec,
            fps,
            len(frame_risks),
            max_ts,
        )
    segments = _segments_from_risks(
        frame_risks,
        threshold=cfg.trufor_threshold,
        reason="TruFor spatial tamper signal",
    )
    return round(video_score, 6), frame_risks, segments


def infer_timesformer_temporal(
    video_path: Path, work_dir: Path, cfg: ForgeryInferConfig
) -> tuple[float, list[dict], list[dict], dict[str, Any]]:
    import torch
    from timesformer_forgery_features import (  # noqa: WPS433
        extract_video_window_embeddings,
        load_forgery_bundle,
        score_windows_mil,
    )

    if not cfg.timesformer_ckpt or not cfg.timesformer_ckpt.is_file():
        raise FileNotFoundError(f"TimeSformer forgery checkpoint not found: {cfg.timesformer_ckpt}")

    device = torch.device(f"cuda:{cfg.gpu}" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    backbone, head, mean, std, ckpt = load_forgery_bundle(cfg.timesformer_ckpt, device)
    aggregate = cfg.ts_aggregate or str(ckpt.get("aggregate", "max"))
    top_k = cfg.ts_top_k if cfg.ts_top_k else int(ckpt.get("top_k", 3))
    window_sec = cfg.ts_window_sec if cfg.ts_window_sec else float(ckpt.get("window_sec", 1.0))
    stride_sec = cfg.ts_stride_sec if cfg.ts_stride_sec else float(ckpt.get("stride_sec", 0.5))
    clip_frames = cfg.ts_clip_frames if cfg.ts_clip_frames else int(ckpt.get("clip_frames", 8))
    max_side = cfg.ts_max_side if cfg.ts_max_side else int(ckpt.get("max_side", 512))

    decode_cache = cfg.decode_cache or (work_dir / "decode_cache")
    decode_cache.mkdir(parents=True, exist_ok=True)

    per_window, meta = extract_video_window_embeddings(
        video_path,
        backbone,
        device,
        window_sec=window_sec,
        stride_sec=stride_sec,
        clip_frames=clip_frames,
        max_side=max_side,
        decode_cache=decode_cache,
    )
    score, detail = score_windows_mil(
        per_window,
        head,
        mean,
        std,
        device,
        aggregate=aggregate,
        top_k=top_k,
    )
    if score is None:
        logger.warning("TimeSformer forgery: no windows for %s (%s)", video_path, meta.get("error_reason"))
        return 0.0, [], [], meta

    fps = float(meta.get("fps") or _video_fps(video_path))

    # Per-window scores for clipRisks chart
    import torch

    head.eval()
    embs = np.stack([np.asarray(w["embedding"], dtype=np.float32) for w in per_window])
    embs = (embs - mean) / std
    x = torch.from_numpy(embs).to(device)
    with torch.inference_mode():
        window_probs = torch.sigmoid(head(x)).detach().cpu().numpy().tolist()

    clip_risks: list[dict[str, Any]] = []
    for i, (win, window_score) in enumerate(zip(per_window, window_probs)):
        start_idx = int(win.get("frame_index_start", 0))
        end_idx = int(win.get("frame_index_end", start_idx))
        start_sec = start_idx / fps
        end_sec = max(start_sec, end_idx / fps)
        clip_risks.append(
            {
                "clipIndex": i,
                "startFrameIndex": start_idx,
                "endFrameIndex": end_idx,
                "startTimeSec": round(start_sec, 4),
                "endTimeSec": round(end_sec, 4),
                "riskScore": round(float(window_score), 6),
            }
        )

    temporal_segments = [
        {
            "startTime": round(float(tw.get("start", 0)) / fps, 3),
            "endTime": round(float(tw.get("end", 0)) / fps, 3),
            "maxRiskScore": round(float(tw.get("score", 0.0)), 6),
            "reason": "TimeSformer temporal window signal",
        }
        for tw in detail.get("top_windows", [])[:5]
        if float(tw.get("score", 0.0)) >= cfg.ts_threshold
    ]

    return round(float(score), 6), clip_risks, temporal_segments, meta


def run_forgery_modules(video_path: Path, worker_cfg: Any, *, work_dir: Path | None = None) -> ForgeryLaneResult:
    cfg = ForgeryInferConfig.from_worker_config(worker_cfg)
    if not cfg.enabled:
        logger.info("Forgery lane disabled (FORGERY_ENABLED=0)")
        return ForgeryLaneResult()

    _setup_forgery_imports(cfg.repo_root, cfg.forgery_root)
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Always isolate per video under a unique tmp (never reuse fixed lane/ dir across jobs).
    parent = work_dir if work_dir is not None else Path(getattr(worker_cfg, "work_dir", tempfile.gettempdir()))
    parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="forgery_lane_", dir=str(parent)))

    try:
        spatial_work = tmp / "spatial"
        temporal_work = tmp / "temporal"
        spatial_score, frame_risks, spatial_segments = infer_trufor_spatial(video_path, spatial_work, cfg)
        temporal_score, clip_risks, temporal_segments, tmeta = infer_timesformer_temporal(
            video_path, temporal_work, cfg
        )
        return ForgeryLaneResult(
            lane_ran=True,
            spatial_score=spatial_score,
            temporal_score=temporal_score,
            spatial_detected=spatial_score >= cfg.trufor_threshold,
            temporal_detected=temporal_score >= cfg.ts_threshold,
            frame_risks=frame_risks,
            clip_risks=clip_risks,
            spatial_segments=spatial_segments,
            temporal_segments=temporal_segments,
            model_temporal_version=cfg.timesformer_ckpt.parent.name if cfg.timesformer_ckpt else "v1.8-csvted-v4",
        )
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
