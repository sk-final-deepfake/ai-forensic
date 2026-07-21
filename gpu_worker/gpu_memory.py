"""Release GPU memory held by the deepfake pipeline before forgery lane."""
from __future__ import annotations

import gc
import logging

logger = logging.getLogger("gpu_worker.gpu_memory")


def release_deepfake_gpu_memory() -> None:
    """Drop cached deepfake weights and free CUDA allocations."""
    try:
        from gpu_worker.models import xception_video

        xception_video.clear_model_cache()
    except Exception:
        logger.debug("xception cache clear skipped", exc_info=True)

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        logger.debug("cuda empty_cache skipped", exc_info=True)

    gc.collect()
    logger.info("Released deepfake GPU memory before forgery lane")
