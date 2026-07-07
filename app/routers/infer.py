"""GPU Gateway POST /infer — EKS ai-fastapi consumer가 VPN으로 호출."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.gateway import GatewayInferRequest

logger = logging.getLogger("ai_fastapi.infer")

router = APIRouter(tags=["gateway"])


@router.post("/infer")
def infer(request: GatewayInferRequest) -> dict:
    try:
        from app.services.gateway_infer import run_gateway_infer
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="GPU inference dependencies not installed on this host (gpu_worker/torch).",
        ) from exc

    try:
        result = run_gateway_infer(request)
        return result.model_dump(mode="json", exclude_none=True)
    except Exception as exc:
        logger.exception(
            "Inference failed analysisRequestId=%s evidenceId=%s",
            request.analysis_request_id,
            request.evidence_id,
        )
        raise HTTPException(
            status_code=500,
            detail=str(exc)[:500],
        ) from exc
