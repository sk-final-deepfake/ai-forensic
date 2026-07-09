from dataclasses import replace

from fastapi import APIRouter

from app.core.model_settings import load_model_settings
from app.schemas.ai_response import AnalysisResponseMessage
from app.schemas.analysis import AnalysisRequest, AnalysisResponse
from app.services.mock_analyzer import analyze_mock
from app.services.video_deepfake_analyzer import analyze_video_request


router = APIRouter(prefix="/ai", tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResponseMessage)
def analyze(request: AnalysisRequest) -> AnalysisResponseMessage:
    return analyze_video_request(request)


@router.post("/analyze/legacy-mock", response_model=AnalysisResponse)
def analyze_legacy_mock(request: AnalysisRequest) -> AnalysisResponse:
    return analyze_mock(request)


@router.post("/analyze/mock-fusion", response_model=AnalysisResponseMessage)
def analyze_mock_fusion(request: AnalysisRequest) -> AnalysisResponseMessage:
    settings = replace(load_model_settings(), use_mock_infer=True)
    return analyze_video_request(request, settings=settings)
