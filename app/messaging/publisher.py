from __future__ import annotations

import json

import pika

from app.messaging.settings import RabbitMqSettings
from app.schemas.ai_response import AnalysisResponseMessage


def publish_analysis_result(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    settings: RabbitMqSettings,
    response: AnalysisResponseMessage,
) -> None:
    body = json.dumps(response.model_dump(mode="json"), ensure_ascii=False)
    channel.basic_publish(
        exchange=settings.result_exchange,
        routing_key=settings.video_result_routing_key,
        body=body.encode("utf-8"),
        properties=pika.BasicProperties(
            content_type="application/json",
            delivery_mode=2,
        ),
    )
