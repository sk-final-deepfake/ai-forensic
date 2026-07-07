from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.consumer import AnalysisConsumer
from app.core.config import settings
from app.routers import analyze, health, infer

logger = logging.getLogger("ai_fastapi")

_consumer: AnalysisConsumer | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _consumer
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if settings.consumer_enabled and settings.rabbit_host and settings.ai_gateway_url:
        if not settings.rabbit_user or not settings.rabbit_password:
            logger.warning("RabbitMQ credentials missing — analysis consumer not started")
        else:
            _consumer = AnalysisConsumer(settings)
            _consumer.start()
    else:
        logger.warning(
            "Analysis consumer disabled (enabled=%s host=%s gateway=%s)",
            settings.consumer_enabled,
            bool(settings.rabbit_host),
            bool(settings.ai_gateway_url),
        )
    yield
    if _consumer:
        _consumer.stop()


app = FastAPI(title=settings.ai_server_name, lifespan=lifespan)

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(infer.router)
