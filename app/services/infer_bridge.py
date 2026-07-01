from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import torch

from app.core.model_settings import ModelSettings, load_json_config
from app.core.paths import ensure_infer_scripts_on_path
from app.services.late_fusion import FusionConfig, load_fusion_config, optical_score_from_aggregate


@dataclass
class ModuleInferResult:
    module: str
    model_name: str
    model_version: str
    status: str
    fake_score: float | None
    pred_label: str | None
    details: dict[str, Any]


class InferRuntime:
    def __init__(self, settings: ModelSettings):
        self.settings = settings
        self.fusion_config: FusionConfig = load_fusion_config(settings.fusion_config_path)
        self.optical_cohort: dict[str, float] = load_json_config(settings.optical_cohort_path)
        ensure_infer_scripts_on_path()
        self.device = torch.device(settings.infer_device)
        self._xception_model = None
        self._xception_cropper = None
        self._timesformer_model = None
        self._timesformer_cascade = None
        self._gmflow_backend = None

    def _import_infer_modules(self):
        from face_crop import create_face_cropper
        from video_timesformer_infer import TimeSformerDetectorLite, clip_to_tensor, load_model as load_timesformer
        from video_xception_infer import infer_video as infer_xception
        from video_xception_infer import load_model as load_xception
        from video_clip_transformer_common import infer_video_clip_model
        from optical_flow_backends import BACKENDS
        from optical_flow_infer_model import infer_video as infer_optical_video

        return {
            "create_face_cropper": create_face_cropper,
            "infer_xception": infer_xception,
            "load_xception": load_xception,
            "load_timesformer": load_timesformer,
            "clip_to_tensor": clip_to_tensor,
            "infer_video_clip_model": infer_video_clip_model,
            "TimeSformerDetectorLite": TimeSformerDetectorLite,
            "BACKENDS": BACKENDS,
            "infer_optical_video": infer_optical_video,
        }

    def _ensure_xception(self, modules: dict[str, Any]) -> None:
        if self._xception_model is not None:
            return
        if not self.settings.xception_weights.is_file():
            raise FileNotFoundError(f"Xception weights not found: {self.settings.xception_weights}")
        self._xception_model = modules["load_xception"](self.settings.xception_weights, self.device)
        self._xception_cropper = modules["create_face_cropper"](
            method="mediapipe",
            padding=0.3,
            square=True,
        )

    def _ensure_timesformer(self, modules: dict[str, Any]) -> None:
        if self._timesformer_model is not None:
            return
        if not self.settings.timesformer_weights.is_file():
            raise FileNotFoundError(f"TimeSformer weights not found: {self.settings.timesformer_weights}")
        self._timesformer_model = modules["load_timesformer"](self.settings.timesformer_weights, self.device)
        self._timesformer_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _ensure_gmflow(self, modules: dict[str, Any]) -> None:
        if self._gmflow_backend is not None:
            return
        backend_cls = modules["BACKENDS"]["gmflow"]
        backend = backend_cls(self.settings.gmflow_root, self.device)
        backend.load()
        self._gmflow_backend = backend

    def run_cnn(self, video_path: Path, modules: dict[str, Any]) -> ModuleInferResult:
        self._ensure_xception(modules)
        result = modules["infer_xception"](
            self._xception_model,
            video_path,
            self._xception_cropper,
            self.device,
            threshold=self.fusion_config.module_thresholds["cnn"],
            num_frames=32,
            aggregate="topk",
            top_k=5,
        )
        breakdown = result.get("score_breakdown") or {}
        per_frame = breakdown.get("per_frame_scores") or []
        return ModuleInferResult(
            module="cnn",
            model_name="xception",
            model_version=self.fusion_config.model_versions.get("cnn", "xception/v1.0.0"),
            status=str(result.get("status", "error")),
            fake_score=result.get("fake_score"),
            pred_label=result.get("pred_label"),
            details={
                "score_breakdown": breakdown,
                "per_frame_scores": per_frame,
                "frames_used": result.get("frames_used"),
            },
        )

    def run_temporal(self, video_path: Path, modules: dict[str, Any]) -> ModuleInferResult:
        self._ensure_timesformer(modules)
        result = modules["infer_video_clip_model"](
            self._timesformer_model,
            video_path,
            self._timesformer_cascade,
            self.device,
            clip_to_tensor=modules["clip_to_tensor"],
            method="timesformer_clip_classification_outputs",
            threshold=self.fusion_config.module_thresholds["temporal"],
        )
        breakdown = result.get("score_breakdown") or {}
        return ModuleInferResult(
            module="temporal",
            model_name="timesformer",
            model_version=self.fusion_config.model_versions.get("temporal", "timesformer/v1.0.0"),
            status=str(result.get("status", "error")),
            fake_score=result.get("fake_score"),
            pred_label=result.get("pred_label"),
            details={
                "score_breakdown": breakdown,
                "per_clip_scores": breakdown.get("per_clip_scores") or [],
                "frames_used": result.get("frames_used"),
            },
        )

    def run_optical(self, video_path: Path, modules: dict[str, Any]) -> ModuleInferResult:
        self._ensure_gmflow(modules)
        run_id = f"runtime-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        raw = modules["infer_optical_video"](
            video_path,
            self._gmflow_backend,
            max_pairs=8,
            max_side=512,
            run_id=run_id,
            model_name="gmflow",
            ground_truth_label=None,
            device=self.device,
        )
        aggregate = raw.get("aggregate") or {}
        fake_score = optical_score_from_aggregate(aggregate, self.optical_cohort)
        threshold = self.fusion_config.module_thresholds["optical"]
        pred_label = None
        if fake_score is not None:
            pred_label = "fake" if fake_score >= threshold else "real"
        return ModuleInferResult(
            module="optical",
            model_name="gmflow",
            model_version=self.fusion_config.model_versions.get("optical", "gmflow/v1.0.0"),
            status=str(raw.get("status", "error")),
            fake_score=fake_score,
            pred_label=pred_label,
            details={
                "aggregate": aggregate,
                "pair_stats": raw.get("pair_stats") or [],
                "frame_pairs": raw.get("frame_pairs"),
            },
        )

    def analyze_modules(self, video_path: Path) -> list[ModuleInferResult]:
        modules = self._import_infer_modules()
        return [
            self.run_cnn(video_path, modules),
            self.run_temporal(video_path, modules),
            self.run_optical(video_path, modules),
        ]
