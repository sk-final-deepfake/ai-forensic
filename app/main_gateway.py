"""On-Prem GPU Gateway only — RabbitMQ consumer 없음 (Method B)."""

from fastapi import FastAPI

from app.core.config import settings
from app.routers import analyze, health, infer

app = FastAPI(title=settings.ai_server_name or "forenshield-ai-gateway")

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(infer.router)
