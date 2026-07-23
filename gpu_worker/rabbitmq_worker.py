"""RabbitMQ consumer: analysis jobs + on-demand overlay jobs."""

from __future__ import annotations

import json
import logging
import os
import sys
from functools import partial

import pika
from pika.adapters.blocking_connection import BlockingConnection

from gpu_worker.config import WorkerConfig, load_config
from gpu_worker.infer_lock import gpu_infer_lock
from gpu_worker.inference_runner import _utc_now, run_inference
from gpu_worker.overlay_runner import run_overlay_job
from gpu_worker.s3_download import download_job_file
from gpu_worker.schemas import (
    AnalysisJobMessage,
    AnalysisResponseMessage,
    OverlayJobMessage,
    OverlayResultMessage,
)

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


def _publish_json(
    channel: pika.channel.Channel,
    cfg: WorkerConfig,
    *,
    routing_key: str,
    payload: dict,
) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    channel.basic_publish(
        exchange=cfg.result_exchange,
        routing_key=routing_key,
        body=body,
        properties=pika.BasicProperties(
            content_type="application/json",
            delivery_mode=2,
        ),
    )


def publish_result(channel: pika.channel.Channel, cfg: WorkerConfig, message: AnalysisResponseMessage) -> None:
    _publish_json(
        channel,
        cfg,
        routing_key=cfg.result_routing_key,
        payload=message.model_dump(mode="json", exclude_none=True),
    )
    logger.info(
        "Published result analysisRequestId=%s status=%s progress=%s -> %s/%s",
        message.analysisRequestId,
        message.status,
        message.progressPercent,
        cfg.result_exchange,
        cfg.result_routing_key,
    )


def publish_overlay_result(
    channel: pika.channel.Channel,
    cfg: WorkerConfig,
    message: OverlayResultMessage,
) -> None:
    _publish_json(
        channel,
        cfg,
        routing_key=cfg.overlay_result_routing_key,
        payload=message.model_dump(mode="json", exclude_none=True),
    )
    logger.info(
        "Published overlay result overlayJobId=%s module=%s status=%s progress=%s -> %s/%s",
        message.overlayJobId,
        message.module,
        message.status,
        message.progressPercent,
        cfg.result_exchange,
        cfg.overlay_result_routing_key,
    )


def process_analysis_job(channel: pika.channel.Channel, cfg: WorkerConfig, body: bytes) -> None:
    job = AnalysisJobMessage.model_validate_json(body)
    logger.info(
        "Processing analysis job analysisRequestId=%s evidenceId=%s filePath=%s",
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


def process_overlay_job(channel: pika.channel.Channel, cfg: WorkerConfig, body: bytes) -> None:
    job = OverlayJobMessage.model_validate_json(body)
    logger.info(
        "Processing overlay job overlayJobId=%s evidenceId=%s module=%s",
        job.overlayJobId,
        job.evidenceId,
        job.module,
    )

    def report_progress(percent: int, message: str | None = None) -> None:
        publish_overlay_result(
            channel,
            cfg,
            OverlayResultMessage(
                overlayJobId=job.overlayJobId,
                analysisRequestId=job.analysisRequestId,
                evidenceId=job.evidenceId,
                module=job.module,
                status="IN_PROGRESS",
                progressPercent=percent,
                message=message,
                analyzedAt=_utc_now(),
            ),
        )

    report_progress(5, "영상 다운로드 중")
    # Reuse analysis downloader shape via a shim AnalysisJobMessage.
    analysis_shim = AnalysisJobMessage(
        analysisRequestId=job.analysisRequestId,
        evidenceId=job.evidenceId,
        filePath=job.filePath,
        s3ObjectKey=job.s3ObjectKey,
        s3Bucket=job.s3Bucket,
        s3Region=job.s3Region,
        presignedDownloadUrl=job.presignedDownloadUrl,
    )
    try:
        local_path = download_job_file(analysis_shim, cfg)
        with gpu_infer_lock(work_dir=cfg.work_dir, label=f"overlay_{job.overlayJobId}"):
            result = run_overlay_job(job, local_path, cfg, on_progress=report_progress)
        publish_overlay_result(channel, cfg, result)
    except Exception as exc:
        logger.exception("Overlay failed overlayJobId=%s", job.overlayJobId)
        publish_overlay_result(
            channel,
            cfg,
            OverlayResultMessage(
                overlayJobId=job.overlayJobId,
                analysisRequestId=job.analysisRequestId,
                evidenceId=job.evidenceId,
                module=job.module,
                status="FAILED",
                progressPercent=100,
                analyzedAt=_utc_now(),
                errorCode="OVERLAY_FAILED",
                message=str(exc)[:500],
            ),
        )


def _on_analysis_message(channel, method, _properties, body, cfg: WorkerConfig) -> None:
    try:
        process_analysis_job(channel, cfg, body)
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception:
        logger.exception("Unhandled analysis job error")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def _on_overlay_message(channel, method, _properties, body, cfg: WorkerConfig) -> None:
    try:
        process_overlay_job(channel, cfg, body)
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception:
        logger.exception("Unhandled overlay job error")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    cfg = load_config()
    consume_analysis = os.getenv("CONSUME_ANALYSIS_QUEUE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    logger.info(
        "Starting GPU worker mode=%s device=%s rabbit=%s:%s analysis_queue=%s overlay_queue=%s consume_analysis=%s",
        cfg.inference_mode,
        cfg.device,
        cfg.rabbit_host,
        cfg.rabbit_port,
        cfg.analysis_queue,
        cfg.overlay_queue,
        consume_analysis,
    )

    connection = _connect(cfg)
    channel = connection.channel()
    channel.basic_qos(prefetch_count=cfg.prefetch_count)
    if consume_analysis:
        channel.queue_declare(queue=cfg.analysis_queue, durable=True)
    channel.queue_declare(queue=cfg.overlay_queue, durable=True, passive=True)
    if consume_analysis:
        channel.basic_consume(
            queue=cfg.analysis_queue,
            on_message_callback=partial(_on_analysis_message, cfg=cfg),
        )
    channel.basic_consume(
        queue=cfg.overlay_queue,
        on_message_callback=partial(_on_overlay_message, cfg=cfg),
    )
    if consume_analysis:
        logger.info("Waiting for messages on %s and %s", cfg.analysis_queue, cfg.overlay_queue)
    else:
        logger.info("Waiting for messages on %s only (analysis via Method B gateway)", cfg.overlay_queue)
    channel.start_consuming()


if __name__ == "__main__":
    main()
