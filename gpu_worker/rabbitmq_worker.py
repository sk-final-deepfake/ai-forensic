"""RabbitMQ consumer: download from S3, run inference, publish result."""

from __future__ import annotations

import json
import logging
import sys
from functools import partial

import pika
from pika.adapters.blocking_connection import BlockingConnection

from gpu_worker.config import WorkerConfig, load_config
from gpu_worker.inference_runner import _utc_now, run_inference
from gpu_worker.s3_download import download_job_file
from gpu_worker.schemas import AnalysisJobMessage, AnalysisResponseMessage

logger = logging.getLogger("gpu_worker")


def _connect(cfg: WorkerConfig) -> BlockingConnection:
    credentials = pika.PlainCredentials(cfg.rabbit_user, cfg.rabbit_password)
    params = pika.ConnectionParameters(
        host=cfg.rabbit_host,
        port=cfg.rabbit_port,
        virtual_host=cfg.rabbit_vhost,
        credentials=credentials,
        heartbeat=600,
        blocked_connection_timeout=300,
    )
    return BlockingConnection(params)


def publish_result(channel: pika.channel.Channel, cfg: WorkerConfig, message: AnalysisResponseMessage) -> None:
    body = json.dumps(
        message.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
    ).encode("utf-8")
    channel.basic_publish(
        exchange=cfg.result_exchange,
        routing_key=cfg.result_routing_key,
        body=body,
        properties=pika.BasicProperties(
            content_type="application/json",
            delivery_mode=2,
        ),
    )
    logger.info(
        "Published result analysisRequestId=%s status=%s progress=%s frameRisks=%s -> %s/%s",
        message.analysisRequestId,
        message.status,
        message.progressPercent,
        len(message.results[0].frameRisks) if message.results and message.results[0].frameRisks else 0,
        cfg.result_exchange,
        cfg.result_routing_key,
    )


def process_job(channel: pika.channel.Channel, cfg: WorkerConfig, body: bytes) -> None:
    job = AnalysisJobMessage.model_validate_json(body)
    logger.info(
        "Processing job analysisRequestId=%s evidenceId=%s filePath=%s",
        job.analysisRequestId,
        job.evidenceId,
        job.filePath,
    )

    def report_progress(percent: int, message: str | None = None) -> None:
        publish_result(
            channel,
            cfg,
            AnalysisResponseMessage(
                analysisRequestId=job.analysisRequestId,
                evidenceId=job.evidenceId,
                status="IN_PROGRESS",
                progressPercent=percent,
                message=message,
                analyzedAt=_utc_now(),
            ),
        )

    report_progress(5, "영상 다운로드 중")
    local_path = download_job_file(job, cfg)
    logger.info("Downloaded to %s", local_path)
    report_progress(12, "모델 추론 준비 중")
    try:
        result = run_inference(job, local_path, cfg, on_progress=report_progress)
        if result.status == "COMPLETED" and result.progressPercent is None:
            result.progressPercent = 100
        publish_result(channel, cfg, result)
    except Exception as exc:
        logger.exception("Inference failed analysisRequestId=%s", job.analysisRequestId)
        failed = AnalysisResponseMessage(
            analysisRequestId=job.analysisRequestId,
            evidenceId=job.evidenceId,
            status="FAILED",
            analyzedAt=_utc_now(),
            errorCode="MODEL_INFERENCE_FAILED",
            message=str(exc)[:500],
        )
        publish_result(channel, cfg, failed)


def _on_message(channel, method, _properties, body, cfg: WorkerConfig) -> None:
    try:
        process_job(channel, cfg, body)
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception:
        logger.exception("Unhandled job error")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    cfg = load_config()
    logger.info(
        "Starting GPU worker mode=%s device=%s rabbit=%s:%s",
        cfg.inference_mode,
        cfg.device,
        cfg.rabbit_host,
        cfg.rabbit_port,
    )

    connection = _connect(cfg)
    channel = connection.channel()
    channel.basic_qos(prefetch_count=cfg.prefetch_count)
    channel.queue_declare(queue=cfg.analysis_queue, durable=True)
    channel.basic_consume(
        queue=cfg.analysis_queue,
        on_message_callback=partial(_on_message, cfg=cfg),
    )
    logger.info("Waiting for messages on %s", cfg.analysis_queue)
    channel.start_consuming()


if __name__ == "__main__":
    main()
