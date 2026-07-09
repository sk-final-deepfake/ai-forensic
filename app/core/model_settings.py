from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from app.core.paths import ai_root, config_dir, resolve_path


@dataclass(frozen=True)
class ModelSettings:
    ai_root: Path
    fusion_config_path: Path
    optical_cohort_path: Path
    xception_weights: Path
    timesformer_weights: Path
    gmflow_root: Path
    infer_device: str
    use_mock_infer: bool


def _load_default_fusion_path() -> Path:
    return config_dir() / "fusion_v0.json"


def _load_default_cohort_path() -> Path:
    return config_dir() / "optical_flow_cohort_v0.json"


def load_model_settings() -> ModelSettings:
    root = ai_root()
    return ModelSettings(
        ai_root=root,
        fusion_config_path=resolve_path(
            os.getenv("FUSION_CONFIG_PATH", str(_load_default_fusion_path())),
            root=root,
        ),
        optical_cohort_path=resolve_path(
            os.getenv("OPTICAL_COHORT_PATH", str(_load_default_cohort_path())),
            root=root,
        ),
        xception_weights=resolve_path(
            os.getenv(
                "XCEPTION_WEIGHTS",
                "models/test/video/xception/v1.0.0/xception_best.pth",
            ),
            root=root,
        ),
        timesformer_weights=resolve_path(
            os.getenv(
                "TIMESFORMER_WEIGHTS",
                "models/test/video/timesformer/v1.0.0/timesformer_finetuned.pth",
            ),
            root=root,
        ),
        gmflow_root=resolve_path(os.getenv("AI_ROOT", str(root)), root=root),
        infer_device=os.getenv("INFER_DEVICE", "cuda" if _cuda_available() else "cpu"),
        use_mock_infer=os.getenv("USE_MOCK_INFER", "0").lower() in {"1", "true", "yes"},
    )


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def load_json_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
