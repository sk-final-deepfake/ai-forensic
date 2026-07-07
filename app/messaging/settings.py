from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class RabbitMqSettings:
    url: str
    analysis_exchange: str
    result_exchange: str
    dead_letter_exchange: str
    analysis_queue: str
    result_queue: str
    analysis_dlq: str
    video_analysis_routing_key: str
    video_result_routing_key: str
    prefetch_count: int

    @classmethod
    def from_env(cls) -> RabbitMqSettings:
        url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
        return cls(
            url=url,
            analysis_exchange=os.getenv("ANALYSIS_EXCHANGE", "ai.analysis.exchange"),
            result_exchange=os.getenv("AI_RESULT_EXCHANGE", "ai.result.exchange"),
            dead_letter_exchange=os.getenv("ANALYSIS_DLX", "ai.dead.exchange"),
            analysis_queue=os.getenv("ANALYSIS_QUEUE", "forenshield.analysis.queue"),
            result_queue=os.getenv("AI_RESULT_QUEUE", "backend.ai.result.queue"),
            analysis_dlq=os.getenv("ANALYSIS_DLQ", "forenshield.analysis.dlq"),
            video_analysis_routing_key=os.getenv("ANALYSIS_ROUTING_KEY", "analyze.video"),
            video_result_routing_key=os.getenv("AI_RESULT_ROUTING_KEY", "result.video"),
            prefetch_count=int(os.getenv("RABBITMQ_PREFETCH", "1")),
        )

    def display_host(self) -> str:
        parsed = urlparse(self.url)
        return parsed.hostname or "localhost"
