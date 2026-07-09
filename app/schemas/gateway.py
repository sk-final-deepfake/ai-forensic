"""GPU Gateway POST /infer request body (Infra 7.ai-deploy.md)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GatewayInferRequest(BaseModel):
    case_id: str
    evidence_id: int
    analysis_request_id: int
    evidence_path: str = Field(description="s3://bucket/key or presigned HTTPS URL")
    local_path: str | None = None
