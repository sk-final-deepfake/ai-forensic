from fastapi import APIRouter

from app.schemas.analysis import AnalysisRequest, AnalysisResponse
from app.services.mock_analyzer import analyze_mock


router = APIRouter(prefix="/ai", tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest) -> AnalysisResponse:
    return analyze_mock(request)
