from fastapi import APIRouter

from app.core.config import settings


router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    consumer_on = settings.consumer_enabled and bool(settings.rabbit_host)
    return {
        "status": "ok",
        "service": settings.ai_server_name,
        "consumer": "enabled" if consumer_on else "disabled",
    }
