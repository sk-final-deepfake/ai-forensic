"""RabbitMQ consumer: forenshield.analysis.queue → GPU Gateway → result publish."""

from __future__ import annotations

import json
import logging
import threading

import pika
from pika.adapters.blocking_connection import BlockingConnection

from app.core.config import Settings
from app.gpu_client import _utc_now, call_gpu_gateway
from app.schemas.messaging import AnalysisJobMessage, AnalysisResponseMessage

logger = logging.getLogger("ai_fastapi.consumer")


class AnalysisConsumer:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._thread: threading.Thread | None = None
        self._connection: BlockingConnection | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="analysis-consumer", daemon=True)
        self._thread.start()
        logger.info(
            "Analysis consumer thread started queue=%s gateway=%s",
            self._settings.analysis_queue,
            self._settings.ai_gateway_url,
        )

    def stop(self) -> None:
        conn = self._connection
        if conn and conn.is_open:
            try:
                conn.close()
            except Exception:
                logger.exception("Failed to close RabbitMQ connection")

    def _connect(self) -> BlockingConnection:
        s = self._settings
        credentials = pika.PlainCredentials(s.rabbit_user, s.rabbit_password)
        params = pika.ConnectionParameters(
            host=s.rabbit_host,
            port=s.rabbit_port,
            virtual_host=s.rabbit_vhost,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300,
        )
        return BlockingConnection(params)

    def _publish_result(self, channel: pika.channel.Channel, message: AnalysisResponseMessage) -> None:
        body = json.dumps(
            message.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
        ).encode("utf-8")
        channel.basic_publish(
            exchange=self._settings.result_exchange,
            routing_key=self._settings.result_routing_key,
            body=body,
            properties=pika.BasicProperties(content_type="application/json", delivery_mode=2),
        )
        logger.info(
            "Published result analysisRequestId=%s status=%s -> %s/%s",
            message.analysisRequestId,
            message.status,
            self._settings.result_exchange,
            self._settings.result_routing_key,
        )

    def _process_job(self, channel: pika.channel.Channel, body: bytes) -> None:
        job = AnalysisJobMessage.model_validate_json(body)
        logger.info(
            "Processing job analysisRequestId=%s evidenceId=%s filePath=%s",
            job.analysisRequestId,
            job.evidenceId,
            job.filePath,
        )
        try:
            result = call_gpu_gateway(job, self._settings)
            self._publish_result(channel, result)
        except Exception as exc:
            logger.exception("Gateway inference failed analysisRequestId=%s", job.analysisRequestId)
            failed = AnalysisResponseMessage(
                analysisRequestId=job.analysisRequestId,
                evidenceId=job.evidenceId,
                status="FAILED",
                analyzedAt=_utc_now(),
                errorCode="MODEL_INFERENCE_FAILED",
                message=str(exc)[:500],
            )
            self._publish_result(channel, failed)

    def _on_message(self, channel, method, _properties, body) -> None:
        try:
            self._process_job(channel, body)
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception:
            logger.exception("Unhandled job error")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def _run(self) -> None:
        s = self._settings
        self._connection = self._connect()
        channel = self._connection.channel()
        channel.basic_qos(prefetch_count=1)
        # Queue is created by backend (Spring) with DLX; passive attach only.
        channel.queue_declare(queue=s.analysis_queue, passive=True)
        channel.basic_consume(
            queue=s.analysis_queue,
            on_message_callback=self._on_message,
        )
        logger.info("Waiting for messages on %s", s.analysis_queue)
        try:
            channel.start_consuming()
        except Exception:
            if self._connection and self._connection.is_open:
                logger.exception("Consumer stopped with error")
            else:
                logger.info("Consumer connection closed")
