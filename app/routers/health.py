from fastapi import APIRouter

from app.core.config import settings


router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.ai_server_name,
        "consumer": settings.consumer_enabled and bool(settings.rabbit_host),
    }
