#!/usr/bin/env python3
"""Listen on backend.ai.result.queue and print AI responses (dev smoke test)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pika

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.messaging.settings import RabbitMqSettings
from app.messaging.topology import ensure_topology


def main() -> None:
    settings = RabbitMqSettings.from_env()
    connection = pika.BlockingConnection(pika.URLParameters(settings.url))
    channel = connection.channel()
    ensure_topology(channel, settings)

    def callback(ch, method, properties, body):
        print(json.dumps(json.loads(body.decode("utf-8")), indent=2, ensure_ascii=False))
        ch.basic_ack(method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=settings.result_queue, on_message_callback=callback, auto_ack=False)
    print(f"listening on {settings.result_queue} ...")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        pass
    finally:
        connection.close()


if __name__ == "__main__":
    main()
