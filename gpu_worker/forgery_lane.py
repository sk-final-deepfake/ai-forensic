"""Public entry: run forgery after deepfake pipeline and merge into worker response."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from gpu_worker.pipeline.forgery_infer import ForgeryInferConfig, run_forgery_modules
from gpu_worker.pipeline.forgery_merge import merge_forgery_into_response

logger = logging.getLogger("gpu_worker.forgery")


def enrich_with_forgery(response: Any, video_path: Path, worker_cfg: Any) -> Any:
    """Run TruFor + TimeSformer forgery lane and merge scores into an existing response."""
    fcfg = ForgeryInferConfig.from_worker_config(worker_cfg)
    if not fcfg.enabled:
        return response
    if str(getattr(response, "status", "")).upper() not in ("COMPLETED", "SUCCESS", ""):
        return response

    try:
        # Parent dir only; each call uses a unique forgery_lane_* child (cleaned after).
        work_root = Path(worker_cfg.work_dir) / "forgery_lane"
        work_root.mkdir(parents=True, exist_ok=True)
        forgery = run_forgery_modules(video_path, worker_cfg, work_dir=work_root)
        return merge_forgery_into_response(response, forgery, worker_cfg=worker_cfg)
    except Exception:
        logger.exception("Forgery lane failed for %s; returning deepfake-only response", video_path)
        return response
