import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    ai_server_name: str = os.getenv("AI_SERVER_NAME", "forenshield-ai")
    ai_server_port: int = int(os.getenv("AI_SERVER_PORT", "8000"))
    backend_result_api_url: str = os.getenv("BACKEND_RESULT_API_URL", "")
    rabbitmq_url: str = os.getenv("RABBITMQ_URL", "")


settings = Settings()
