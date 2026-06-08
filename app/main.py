from fastapi import FastAPI

from app.core.config import settings
from app.routers import analyze, health


app = FastAPI(title=settings.ai_server_name)

app.include_router(health.router)
app.include_router(analyze.router)
