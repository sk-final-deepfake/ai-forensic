from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch

from gpu_worker.config import WorkerConfig
from gpu_worker.pipeline.paths import resolve_under_root, setup_script_paths

logger = logging.getLogger("gpu_worker.pipeline.module_infer")

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
        result = infer_video_clip_model(
            model,
            video_path,
            None,
            device,
            clip_to_tensor=clip_to_tensor,
            method="timesformer_clip_classification_outputs",
            clip_frames=CLIP_FRAMES,
            clip_size=CLIP_SIZE,
            max_clips=MAX_CLIPS,
            threshold=threshold,
            face_cropper=face_cropper,
            aggregate="max",
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
    if not (backend_root / "vendor/optical-flow/gmflow").is_dir():
        backend_root = cfg.project_root
    backend = GmflowBackend(backend_root, str(device))
    preferred = resolve_under_root(cfg, cfg.gmflow_pretrained)
    if preferred.is_file():
        backend.weights = preferred
    backend.load()
    infer_result = infer_video(
        video_path,
        backend,
        max_pairs=8,
        max_side=512,
        run_id="gateway",
        model_name="gmflow",
        ground_truth_label=None,
        device=device,
    )
    if infer_result.get("status") != "ok":
        raise RuntimeError(f"GMFlow inference failed: status={infer_result.get('status')}")

    scorer, meta = load_scoring_config(cfg.deepfake_root)
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
