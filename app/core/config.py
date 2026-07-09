import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    ai_server_name: str = _env("AI_SERVER_NAME", "forenshield-ai-fastapi")
    ai_server_port: int = int(_env("AI_SERVER_PORT", "8000"))

    rabbit_host: str = _env("RABBITMQ_HOST")
    rabbit_port: int = int(_env("RABBITMQ_PORT", "5672") or "5672")
    rabbit_user: str = _env("RABBITMQ_USER")
    rabbit_password: str = _env("RABBITMQ_PASSWORD")
    rabbit_vhost: str = _env("RABBITMQ_VHOST", "/") or "/"

    analysis_queue: str = _env("ANALYSIS_QUEUE", "forenshield.analysis.queue")
    result_exchange: str = _env("AI_RESULT_EXCHANGE", "ai.result.exchange")
    result_routing_key: str = _env("AI_RESULT_ROUTING_KEY", "result.video")

    ai_gateway_url: str = _env("AI_GATEWAY_URL")
    ai_gateway_timeout_sec: float = float(_env("AI_GATEWAY_TIMEOUT_SEC", "1800") or "1800")

    consumer_enabled: bool = _env("AI_CONSUMER_ENABLED", "true").lower() in ("1", "true", "yes")


settings = Settings()
