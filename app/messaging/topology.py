from __future__ import annotations

import pika

from app.messaging.settings import RabbitMqSettings


def ensure_topology(channel: pika.adapters.blocking_connection.BlockingChannel, settings: RabbitMqSettings) -> None:
    """Declare exchanges/queues/bindings aligned with backend RabbitMqConfig."""
    channel.exchange_declare(
        exchange=settings.analysis_exchange,
        exchange_type="topic",
        durable=True,
    )
    channel.exchange_declare(
        exchange=settings.result_exchange,
        exchange_type="topic",
        durable=True,
    )
    channel.exchange_declare(
        exchange=settings.dead_letter_exchange,
        exchange_type="direct",
        durable=True,
    )

    channel.queue_declare(queue=settings.analysis_dlq, durable=True)
    channel.queue_bind(
        queue=settings.analysis_dlq,
        exchange=settings.dead_letter_exchange,
        routing_key=settings.analysis_dlq,
    )

    channel.queue_declare(
        queue=settings.analysis_queue,
        durable=True,
        arguments={
            "x-dead-letter-exchange": settings.dead_letter_exchange,
            "x-dead-letter-routing-key": settings.analysis_dlq,
        },
    )
    channel.queue_bind(
        queue=settings.analysis_queue,
        exchange=settings.analysis_exchange,
        routing_key=settings.video_analysis_routing_key,
    )

    channel.queue_declare(queue=settings.result_queue, durable=True)
    channel.queue_bind(
        queue=settings.result_queue,
        exchange=settings.result_exchange,
        routing_key=settings.video_result_routing_key,
    )
