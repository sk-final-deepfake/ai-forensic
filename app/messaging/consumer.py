from __future__ import annotations

import json
import logging
from typing import Callable

import pika
from pika.adapters.blocking_connection import BlockingChannel

from app.messaging.publisher import publish_analysis_result
from app.messaging.settings import RabbitMqSettings
from app.messaging.topology import ensure_topology
from app.schemas.ai_response import AnalysisResponseMessage
from app.schemas.queue_job import AnalysisJobMessage
from app.services.job_adapter import job_to_analysis_request
from app.services.video_deepfake_analyzer import analyze_video_request

logger = logging.getLogger(__name__)


def process_job_message(raw_body: bytes) -> AnalysisResponseMessage:
    payload = json.loads(raw_body.decode("utf-8"))
    job = AnalysisJobMessage.model_validate(payload)
    if job.fileType != "video":
        return AnalysisResponseMessage(
            analysisRequestId=job.analysisRequestId,
            evidenceId=job.evidenceId,
            status="FAILED",
            analyzedAt=_utc_now(),
            errorCode="UNSUPPORTED_FILE_TYPE",
            message=f"Only video is supported, got: {job.fileType}",
        )
    request = job_to_analysis_request(job)
    return analyze_video_request(request)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _on_message(
    channel: BlockingChannel,
    method: pika.spec.Basic.Deliver,
    properties: pika.spec.BasicProperties,
    body: bytes,
    *,
    settings: RabbitMqSettings,
) -> None:
    delivery_tag = method.delivery_tag
    try:
        response = process_job_message(body)
        publish_analysis_result(channel, settings, response)
        channel.basic_ack(delivery_tag)
        logger.info(
            "Published AI result analysisRequestId=%s evidenceId=%s status=%s",
            response.analysisRequestId,
            response.evidenceId,
            response.status,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to process analysis job: %s", exc)
        try:
            payload = json.loads(body.decode("utf-8"))
            failed = AnalysisResponseMessage(
                analysisRequestId=int(payload.get("analysisRequestId", 0)),
                evidenceId=int(payload.get("evidenceId", 0)),
                status="FAILED",
                analyzedAt=_utc_now(),
                errorCode="MODEL_INFERENCE_FAILED",
                message=str(exc),
            )
            publish_analysis_result(channel, settings, failed)
            channel.basic_ack(delivery_tag)
        except Exception:  # noqa: BLE001
            logger.exception("Could not publish failure response; nacking message")
            channel.basic_nack(delivery_tag, requeue=True)


def run_consumer(
    settings: RabbitMqSettings | None = None,
    *,
    on_message: Callable[[bytes], AnalysisResponseMessage] | None = None,
) -> None:
    settings = settings or RabbitMqSettings.from_env()
    parameters = pika.URLParameters(settings.url)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    ensure_topology(channel, settings)
    channel.basic_qos(prefetch_count=settings.prefetch_count)

    def callback(ch, method, properties, body):
        if on_message is not None:
            response = on_message(body)
            publish_analysis_result(ch, settings, response)
            ch.basic_ack(method.delivery_tag)
            return
        _on_message(ch, method, properties, body, settings=settings)

    channel.basic_consume(queue=settings.analysis_queue, on_message_callback=callback, auto_ack=False)
    logger.info(
        "AI analysis worker listening queue=%s host=%s mock=%s",
        settings.analysis_queue,
        settings.display_host(),
        __import__("os").getenv("USE_MOCK_INFER", "0"),
    )
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("Stopping AI analysis worker")
    finally:
        if channel.is_open:
            channel.close()
        if connection.is_open:
            connection.close()
