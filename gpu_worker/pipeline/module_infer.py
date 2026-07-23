from __future__ import annotations

import os

import logging
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch

from gpu_worker.config import WorkerConfig
from gpu_worker.pipeline.paths import resolve_under_root, setup_script_paths

logger = logging.getLogger("gpu_worker.pipeline.module_infer")

DEFAULT_TRUFOR_THRESHOLD = 0.515
TRUFOR_MODEL_NAME = "TruFor"
TRUFOR_MODEL_VERSION = "v1.0.0"
TRUFOR_MODULE = "forgery_spatial"

NO_FACE_STATUSES = frozenset({"no_face", "no_human_face", "skipped_no_human_face"})
FACE_QUALITY_STATUSES = frozenset({"face_too_small", "insufficient_face_samples"})
CNN_GATE_STATUSES = NO_FACE_STATUSES | FACE_QUALITY_STATUSES
TEMPORAL_UNAVAILABLE_STATUSES = frozenset(
    {
        "insufficient_face_samples",
        "insufficient_temporal_clips",
        "face_too_small",
        "error",
        "skipped",
        *NO_FACE_STATUSES,
    }
)


@dataclass(frozen=True)
class ModuleRunResult:
    module: str
    model_name: str
    model_version: str
    video_score: float
    threshold: float
    detected: bool
    confidence: float
    frame_risks: list[dict[str, Any]]
    clip_risks: list[dict[str, Any]]
    pair_risks: list[dict[str, Any]]
    suspicious_segments: list[dict[str, Any]]
    temporal_suspicious_segments: list[dict[str, Any]]
    optical_suspicious_segments: list[dict[str, Any]]
    raw: dict[str, Any]


def _video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    return fps if fps > 0 else 25.0


def _frame_time(frame_index: int, fps: float) -> float:
    return round(frame_index / fps, 3)


def _resolve_xception_weights(cfg: WorkerConfig) -> Path:
    explicit = (cfg.model_checkpoint or "").strip()
    if explicit:
        return resolve_under_root(cfg, explicit)
    return resolve_under_root(
        cfg,
        "models/test/video/xception/v1.0.0/xception_finetuned_celeb1k.pth",
    )


def _resolve_timesformer_weights(cfg: WorkerConfig) -> Path:
    explicit = (cfg.timesformer_weights or "").strip()
    if explicit:
        return resolve_under_root(cfg, explicit)
    return resolve_under_root(
        cfg,
        "models/test/video/timesformer/v1.0.0/timesformer_finetuned_celeb1k.pth",
    )


def _cnn_segments(frame_risks: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    from gpu_worker.pipeline.segments import build_suspicious_segments

    points = [
        (row["timestampSec"], row["timestampSec"], row["riskScore"])
        for row in frame_risks
    ]
    return [
        item.model_dump()
        for item in build_suspicious_segments(
            points,
            threshold=threshold,
            reason="프레임 fake 확률이 임계값을 초과했습니다.",
        )
    ]


def _clip_segments(clip_risks: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [
        {
            "startTime": row["startTimeSec"],
            "endTime": row["endTimeSec"],
            "maxRiskScore": row["riskScore"],
            "reason": "클립 시계열 점수가 임계값을 초과했습니다.",
        }
        for row in clip_risks
        if row["riskScore"] >= threshold
    ]


def _pair_segments(pair_risks: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [
        {
            "startTime": row["timestampSec"],
            "endTime": row["timestampSec"] + 0.1,
            "maxRiskScore": row["riskScore"],
            "reason": "프레임쌍 움직임 이상이 관찰되었습니다.",
        }
        for row in pair_risks
        if row["riskScore"] >= threshold
    ]


def run_xception_module(video_path: Path, cfg: WorkerConfig, *, threshold: float, fps: float) -> ModuleRunResult:
    setup_script_paths(cfg)
    from face_crop import create_face_cropper  # type: ignore
    from video_xception_infer import infer_video, load_model  # type: ignore

    weights = _resolve_xception_weights(cfg)
    if not weights.is_file():
        raise FileNotFoundError(f"Xception weights not found: {weights}")

    device = torch.device("cuda" if cfg.device.lower().startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)
    face_cropper = create_face_cropper(method="yunet", padding=0.3, square=True, human_only=True)
    try:
        result = infer_video(
            model,
            video_path,
            face_cropper,
            device,
            threshold=threshold,
            aggregate="max",
        )
    finally:
        face_cropper.close()
    if result.get("status") in CNN_GATE_STATUSES or result.get("fake_score") is None:
        return ModuleRunResult(
            module="cnn",
            model_name="Xception",
            model_version=cfg.model_version or "v1.0.0-celeb1k",
            video_score=0.0,
            threshold=threshold,
            detected=False,
            confidence=0.0,
            frame_risks=[],
            clip_risks=[],
            pair_risks=[],
            suspicious_segments=[],
            temporal_suspicious_segments=[],
            optical_suspicious_segments=[],
            raw=result,
        )

    breakdown = result.get("score_breakdown") or {}
    aggregate = breakdown.get("aggregate") or {}
    per_frame = breakdown.get("per_frame") or []
    frame_risks = [
        {
            "frameIndex": int(row.get("frame_index", idx)),
            "timestampSec": _frame_time(int(row.get("frame_index", idx)), fps),
            "riskScore": round(float(row.get("prob_fake", 0.0)), 4),
        }
        for idx, row in enumerate(per_frame)
    ]
    video_score = round(float(result["fake_score"]), 4)
    confidence = round(float(aggregate.get("confidence", max(video_score, 1.0 - video_score))), 4)
    return ModuleRunResult(
        module="cnn",
        model_name="Xception",
        model_version=cfg.model_version or "v1.0.0-celeb1k",
        video_score=video_score,
        threshold=threshold,
        detected=video_score >= threshold,
        confidence=confidence,
        frame_risks=frame_risks,
        clip_risks=[],
        pair_risks=[],
        suspicious_segments=_cnn_segments(frame_risks, threshold),
        temporal_suspicious_segments=[],
        optical_suspicious_segments=[],
        raw=result,
    )


def run_timesformer_module(video_path: Path, cfg: WorkerConfig, *, threshold: float, fps: float) -> ModuleRunResult:
    setup_script_paths(cfg)
    from face_crop import create_face_cropper  # type: ignore
    from video_clip_transformer_common import infer_video_clip_model  # type: ignore
    from video_timesformer_infer import (  # type: ignore
        CLIP_FRAMES,
        CLIP_SIZE,
        MAX_CLIPS,
        clip_to_tensor,
        load_model,
    )

    weights = _resolve_timesformer_weights(cfg)
    if not weights.is_file():
        raise FileNotFoundError(f"TimeSformer weights not found: {weights}")

    device = torch.device("cuda" if cfg.device.lower().startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)
    face_cropper = create_face_cropper(
        method="yunet",
        padding=0.3,
        square=True,
        human_only=True,
        size=CLIP_SIZE,
    )
    try:
        import inspect

        clip_kwargs = dict(
            clip_to_tensor=clip_to_tensor,
            method="timesformer_clip_classification_outputs",
            clip_frames=CLIP_FRAMES,
            clip_size=CLIP_SIZE,
            max_clips=MAX_CLIPS,
            threshold=threshold,
            face_cropper=face_cropper,
            aggregate="max",
        )
        supported = inspect.signature(infer_video_clip_model).parameters
        clip_kwargs = {k: v for k, v in clip_kwargs.items() if k in supported}
        result = infer_video_clip_model(
            model,
            video_path,
            None,
            device,
            **clip_kwargs,
        )
    except RuntimeError as exc:
        logger.exception("TimeSformer inference failed for %s", video_path.name)
        result = {
            "file": video_path.name,
            "status": "error",
            "fake_score": None,
            "pred_label": None,
            "score_breakdown": {"message": str(exc)},
        }
    finally:
        face_cropper.close()
    if result.get("status") in TEMPORAL_UNAVAILABLE_STATUSES or result.get("fake_score") is None:
        return ModuleRunResult(
            module="temporal",
            model_name="TimeSformer",
            model_version="v1.0.0-celeb1k",
            video_score=0.0,
            threshold=threshold,
            detected=False,
            confidence=0.0,
            frame_risks=[],
            clip_risks=[],
            pair_risks=[],
            suspicious_segments=[],
            temporal_suspicious_segments=[],
            optical_suspicious_segments=[],
            raw=result,
        )

    breakdown = result.get("score_breakdown") or {}
    aggregate = breakdown.get("aggregate") or {}
    per_clip = breakdown.get("per_clip") or []
    clip_risks = []
    for row in per_clip:
        start_idx = int(row.get("clip_start_frame", row.get("frame_indices", [0])[0]))
        end_idx = int(row.get("clip_end_frame", row.get("frame_indices", [start_idx])[-1]))
        clip_risks.append(
            {
                "clipIndex": int(row.get("clip_index", len(clip_risks))),
                "startFrameIndex": start_idx,
                "endFrameIndex": end_idx,
                "startTimeSec": _frame_time(start_idx, fps),
                "endTimeSec": _frame_time(end_idx, fps),
                "riskScore": round(float(row.get("prob_fake", 0.0)), 4),
            }
        )
    video_score = round(float(result["fake_score"]), 4)
    confidence = round(float(aggregate.get("confidence", max(video_score, 1.0 - video_score))), 4)
    temporal_segments = _clip_segments(clip_risks, threshold)
    return ModuleRunResult(
        module="temporal",
        model_name="TimeSformer",
        model_version="v1.0.0-celeb1k",
        video_score=video_score,
        threshold=threshold,
        detected=video_score >= threshold,
        confidence=confidence,
        frame_risks=[],
        clip_risks=clip_risks,
        pair_risks=[],
        suspicious_segments=[],
        temporal_suspicious_segments=temporal_segments,
        optical_suspicious_segments=[],
        raw=result,
    )


def _gmflow_pair_risks(pair_stats: list[dict[str, Any]], fps: float, video_score: float) -> list[dict[str, Any]]:
    if not pair_stats:
        return []
    max_mag = max(float(p.get("magnitude_mean", 0.0)) for p in pair_stats) or 1.0
    pair_risks: list[dict[str, Any]] = []
    for idx, pair in enumerate(pair_stats):
        mag = float(pair.get("magnitude_mean", 0.0))
        idx_a = int(pair.get("frame_index_a", idx))
        idx_b = int(pair.get("frame_index_b", idx + 1))
        risk = round(min(1.0, (mag / max_mag) * video_score), 4) if max_mag > 0 else round(video_score, 4)
        pair_risks.append(
            {
                "pairIndex": idx,
                "frameIndexA": idx_a,
                "frameIndexB": idx_b,
                "timestampSec": _frame_time(idx_a, fps),
                "riskScore": risk,
                "motionMagnitude": round(mag, 4),
            }
        )
    return pair_risks


def run_gmflow_module(video_path: Path, cfg: WorkerConfig, *, threshold: float, fps: float) -> ModuleRunResult:
    setup_script_paths(cfg)
    from gmflow_learned_head_infer import fake_score_from_report, load_scoring_config  # type: ignore
    from optical_flow_backends import GmflowBackend  # type: ignore
    from optical_flow_infer_model import infer_video  # type: ignore

    device = torch.device("cuda" if cfg.device.lower().startswith("cuda") and torch.cuda.is_available() else "cpu")
    backend_root = cfg.deepfake_root
    for candidate in (cfg.deepfake_root, cfg.project_root, cfg.project_root.parent):
        if (candidate / "vendor/optical-flow/gmflow").is_dir():
            backend_root = candidate
            break
    backend = GmflowBackend(backend_root, device)
    preferred = resolve_under_root(cfg, cfg.gmflow_pretrained)
    if preferred.is_file():
        backend.weights = preferred
    backend.load()

    env_pairs = int(os.getenv("GMFLOW_MAX_PAIRS", "4"))
    env_side = int(os.getenv("GMFLOW_MAX_SIDE", "384"))
    attempts = []
    for mp, ms in ((env_pairs, env_side), (4, 384), (2, 320)):
        if (mp, ms) not in attempts:
            attempts.append((mp, ms))

    infer_result: dict = {"status": "error", "errors": [], "pair_stats": []}
    for max_pairs, max_side in attempts:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        infer_result = infer_video(
            video_path,
            backend,
            max_pairs=max_pairs,
            max_side=max_side,
            run_id="gateway",
            model_name="gmflow",
            ground_truth_label=None,
            device=device,
        )
        if infer_result.get("status") == "ok":
            break
        logger.warning(
            "GMFlow attempt failed max_pairs=%s max_side=%s status=%s errors=%s",
            max_pairs,
            max_side,
            infer_result.get("status"),
            infer_result.get("errors"),
        )

    if infer_result.get("status") != "ok":
        errors = infer_result.get("errors") or []
        logger.error("GMFlow inference failed after retries: errors=%s", errors)
        return ModuleRunResult(
            module="optical",
            model_name="GMFlow",
            model_version="v1.0.0",
            video_score=0.0,
            threshold=threshold,
            detected=False,
            confidence=0.0,
            frame_risks=[],
            clip_risks=[],
            pair_risks=[],
            suspicious_segments=[],
            temporal_suspicious_segments=[],
            optical_suspicious_segments=[],
            raw={"status": "error", "message": f"GMFlow failed: {errors[:3]}", "errors": errors, "fake_score": None},
        )

    scorer_root = cfg.project_root
    meta_path = scorer_root / "models/test/video/optical-flow/gmflow/v1.0.0/gmflow_best.meta.json"
    if not meta_path.is_file():
        scorer_root = cfg.deepfake_root
    scorer, meta = load_scoring_config(scorer_root)
    video_score = fake_score_from_report(infer_result, scorer, meta)
    if video_score is None:
        raise RuntimeError("GMFlow learned head could not score video")

    video_score = round(float(video_score), 4)
    pair_stats = infer_result.get("pair_stats") or []
    pair_risks = _gmflow_pair_risks(pair_stats, fps, video_score)
    optical_segments = _pair_segments(pair_risks, threshold)
    confidence = round(max(video_score, 1.0 - video_score), 4)

    return ModuleRunResult(
        module="optical",
        model_name="GMFlow",
        model_version="v1.0.0",
        video_score=video_score,
        threshold=threshold,
        detected=video_score >= threshold,
        confidence=confidence,
        frame_risks=[],
        clip_risks=[],
        pair_risks=pair_risks,
        suspicious_segments=[],
        temporal_suspicious_segments=[],
        optical_suspicious_segments=optical_segments,
        raw=infer_result,
    )


def _forgery_search_roots(cfg: WorkerConfig) -> list[Path]:
    """Candidate roots for TruFor vendor/ + models/ (deepfake or forgery track layout)."""
    ai_repo = Path(__file__).resolve().parents[2]
    candidates = [
        cfg.project_root,
        cfg.project_root / "forgery",
        cfg.project_root.parent / "forgery",
        cfg.deepfake_root,
        cfg.deepfake_root.parent / "forgery",
        ai_repo,
        ai_repo / "forgery",
    ]
    roots: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def _resolve_trufor_assets(cfg: WorkerConfig) -> tuple[Path, Path, Path] | None:
    """Return (forgery_root, test_py, weights) or None if vendor/weights missing."""
    explicit_weights = (cfg.trufor_weights or "").strip()
    weight_rel = explicit_weights or "models/test/spatial/trufor/v1.0.0/trufor.pth.tar"
    weight_candidate = Path(weight_rel)
    if weight_candidate.is_file():
        weights = weight_candidate.resolve()
        # Prefer root that also has vendor when weights are absolute.
        for root in _forgery_search_roots(cfg):
            test_py = root / "vendor" / "TruFor" / "TruFor_train_test" / "test.py"
            if test_py.is_file():
                return root, test_py, weights
        return None

    for root in _forgery_search_roots(cfg):
        test_py = root / "vendor" / "TruFor" / "TruFor_train_test" / "test.py"
        weights = (root / weight_rel).resolve()
        if test_py.is_file() and weights.is_file():
            return root, test_py, weights
    return None


def _uniform_frame_indices(total: int, num_samples: int) -> list[int]:
    if total <= 0:
        return []
    if total <= num_samples:
        return list(range(total))
    return [int(i * (total - 1) / (num_samples - 1)) for i in range(num_samples)]


def _extract_trufor_frames(
    video_path: Path,
    out_dir: Path,
    frames_per_video: int,
) -> list[tuple[int, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = _uniform_frame_indices(total_frames, frames_per_video)
    extracted: list[tuple[int, Path]] = []
    try:
        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue
            frame_path = out_dir / f"frame_{i:03d}.jpg"
            if cv2.imwrite(str(frame_path), frame_bgr):
                extracted.append((frame_idx, frame_path))
    finally:
        cap.release()
    return extracted


def _trufor_skipped_result(
    *,
    threshold: float,
    status: str,
    message: str,
) -> ModuleRunResult:
    return ModuleRunResult(
        module=TRUFOR_MODULE,
        model_name=TRUFOR_MODEL_NAME,
        model_version=TRUFOR_MODEL_VERSION,
        video_score=0.0,
        threshold=threshold,
        detected=False,
        confidence=0.0,
        frame_risks=[],
        clip_risks=[],
        pair_risks=[],
        suspicious_segments=[],
        temporal_suspicious_segments=[],
        optical_suspicious_segments=[],
        raw={"status": status, "message": message, "fake_score": None},
    )


def _run_vendor_trufor_test(
    *,
    gpu: int,
    frames_root: Path,
    npz_out_dir: Path,
    trufor_test_py: Path,
    experiment: str,
    model_file: Path,
) -> None:
    npz_out_dir.mkdir(parents=True, exist_ok=True)
    trufor_root = trufor_test_py.parent
    config_yaml = trufor_root / "lib" / "config" / f"{experiment}.yaml"
    if not config_yaml.is_file():
        raise FileNotFoundError(f"Missing TruFor config: {config_yaml}")

    cmd = [
        sys.executable,
        str(trufor_test_py),
        "-g",
        str(gpu),
        "-in",
        str(frames_root.resolve()),
        "-out",
        str(npz_out_dir.resolve()),
        "-exp",
        experiment,
        "TEST.MODEL_FILE",
        str(model_file.resolve()),
    ]
    subprocess.run(cmd, check=True, cwd=trufor_root, timeout=600)


def _frame_scores_from_npz(
    npz_dir: Path,
    extracted: list[tuple[int, Path]],
    fps: float,
) -> tuple[float, list[dict[str, Any]]]:
    import numpy as np

    scores: list[float] = []
    frame_risks: list[dict[str, Any]] = []
    for frame_idx, frame_path in extracted:
        npz_path = npz_dir / f"{frame_path.stem}.npz"
        if not npz_path.is_file():
            # TruFor may nest under a mirrored relative path; search by stem.
            matches = list(npz_dir.rglob(f"{frame_path.stem}.npz"))
            npz_path = matches[0] if matches else npz_path
        if not npz_path.is_file():
            continue
        data = np.load(npz_path)
        if "score" not in data:
            continue
        score = float(data["score"])
        if not math.isfinite(score):
            continue
        scores.append(score)
        frame_risks.append(
            {
                "frameIndex": int(frame_idx),
                "timestampSec": _frame_time(int(frame_idx), fps),
                "riskScore": round(score, 4),
            }
        )
    if not scores:
        return float("nan"), []
    return float(sum(scores) / len(scores)), frame_risks


def run_trufor_module(
    video_path: Path,
    cfg: WorkerConfig,
    *,
    threshold: float | None = None,
    fps: float | None = None,
    work_dir: Path | None = None,
) -> ModuleRunResult:
    """Best-effort TruFor spatial forgery. Never raises — returns skipped/error raw status."""
    thr = float(threshold if threshold is not None else cfg.trufor_threshold or DEFAULT_TRUFOR_THRESHOLD)
    video_fps = float(fps if fps is not None else _video_fps(video_path))

    try:
        assets = _resolve_trufor_assets(cfg)
        if assets is None:
            return _trufor_skipped_result(
                threshold=thr,
                status="skipped_unavailable",
                message="TruFor vendor/test.py or weights not found",
            )

        _forgery_root, test_py, weights = assets
        run_work = work_dir or (cfg.work_dir / "trufor" / video_path.stem)
        if run_work.exists():
            shutil.rmtree(run_work, ignore_errors=True)
        frames_root = run_work / "frames"
        frame_dir = frames_root / "video"
        npz_out_dir = run_work / "npz_frames"

        extracted = _extract_trufor_frames(
            video_path,
            frame_dir,
            max(1, int(cfg.trufor_frames_per_video or 8)),
        )
        if not extracted:
            return _trufor_skipped_result(
                threshold=thr,
                status="skipped",
                message="No frames extracted for TruFor",
            )

        gpu = 0
        device = (cfg.device or "").lower()
        if device.startswith("cuda:") and device.split(":", 1)[1].isdigit():
            gpu = int(device.split(":", 1)[1])

        _run_vendor_trufor_test(
            gpu=gpu,
            frames_root=frames_root,
            npz_out_dir=npz_out_dir,
            trufor_test_py=test_py,
            experiment=(cfg.trufor_experiment or "trufor_ph3").strip() or "trufor_ph3",
            model_file=weights,
        )

        video_score, frame_risks = _frame_scores_from_npz(
            npz_out_dir / "video",
            extracted,
            video_fps,
        )
        if not math.isfinite(video_score):
            # Fallback: any npz under output tree
            video_score, frame_risks = _frame_scores_from_npz(npz_out_dir, extracted, video_fps)
        if not math.isfinite(video_score):
            return _trufor_skipped_result(
                threshold=thr,
                status="error",
                message="TruFor produced no finite frame scores",
            )

        video_score = round(float(video_score), 4)
        confidence = round(max(video_score, 1.0 - video_score), 4)
        segments = [
            {
                "startTime": row["timestampSec"],
                "endTime": row["timestampSec"] + max(0.1, 1.0 / video_fps),
                "maxRiskScore": row["riskScore"],
                "reason": "TruFor spatial score exceeded threshold",
            }
            for row in frame_risks
            if row["riskScore"] >= thr
        ]
        return ModuleRunResult(
            module=TRUFOR_MODULE,
            model_name=TRUFOR_MODEL_NAME,
            model_version=TRUFOR_MODEL_VERSION,
            video_score=video_score,
            threshold=thr,
            detected=video_score >= thr,
            confidence=confidence,
            frame_risks=frame_risks,
            clip_risks=[],
            pair_risks=[],
            suspicious_segments=segments,
            temporal_suspicious_segments=[],
            optical_suspicious_segments=[],
            raw={
                "status": "ok",
                "fake_score": video_score,
                "frames_scored": len(frame_risks),
                "weights": str(weights),
            },
        )
    except Exception as exc:  # noqa: BLE001 — soft-gate must never fail the job
        logger.exception("TruFor forgery soft-continuation failed for %s", video_path)
        return _trufor_skipped_result(
            threshold=thr,
            status="error",
            message=str(exc),
        )


def forgery_ran_successfully(forgery: ModuleRunResult | None) -> bool:
    if forgery is None:
        return False
    return str((forgery.raw or {}).get("status", "")) == "ok"
