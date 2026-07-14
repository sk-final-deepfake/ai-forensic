"""Publish mid-run IN_PROGRESS analysis updates to RabbitMQ (Method B GPU gateway path)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pika

from app.core.config import Settings
from app.schemas.messaging import AnalysisResponseMessage
from gpu_worker.config import WorkerConfig

logger = logging.getLogger("ai_fastapi.analysis_progress")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_message(
    analysis_request_id: int,
    evidence_id: int,
    percent: int,
    message: str | None,
) -> AnalysisResponseMessage:
    return AnalysisResponseMessage(
        analysisRequestId=analysis_request_id,
        evidenceId=evidence_id,
        status="IN_PROGRESS",
        progressPercent=max(0, min(99, int(percent))),
        message=message,
        analyzedAt=_utc_now(),
    )


def publish_analysis_progress_with_channel(
    channel: pika.channel.Channel,
    settings: Settings,
    analysis_request_id: int,
    evidence_id: int,
    percent: int,
    message: str | None = None,
) -> None:
    if not settings.rabbit_host or not settings.rabbit_user:
        return
    payload = _build_message(analysis_request_id, evidence_id, percent, message)
    body = json.dumps(payload.model_dump(mode="json", exclude_none=True), ensure_ascii=False).encode("utf-8")
    channel.basic_publish(
        exchange=settings.result_exchange,
        routing_key=settings.result_routing_key,
        body=body,
        properties=pika.BasicProperties(content_type="application/json", delivery_mode=2),
    )


def publish_analysis_progress_with_config(
    cfg: WorkerConfig,
    analysis_request_id: int,
    evidence_id: int,
    percent: int,
    message: str | None = None,
) -> None:
    if not cfg.rabbit_host or not cfg.rabbit_user:
        return
    payload = _build_message(analysis_request_id, evidence_id, percent, message)
    body = json.dumps(payload.model_dump(mode="json", exclude_none=True), ensure_ascii=False).encode("utf-8")
    try:
        credentials = pika.PlainCredentials(cfg.rabbit_user, cfg.rabbit_password)
        params = pika.ConnectionParameters(
            host=cfg.rabbit_host,
            port=cfg.rabbit_port,
            virtual_host=cfg.rabbit_vhost,
            credentials=credentials,
        )
        with pika.BlockingConnection(params) as connection:
            channel = connection.channel()
            channel.basic_publish(
                exchange=cfg.result_exchange,
                routing_key=cfg.result_routing_key,
                body=body,
                properties=pika.BasicProperties(content_type="application/json", delivery_mode=2),
            )
    except Exception:
        logger.warning(
            "Failed to publish analysis progress %s%% analysisRequestId=%s",
            percent,
            analysis_request_id,
            exc_info=True,
        )
