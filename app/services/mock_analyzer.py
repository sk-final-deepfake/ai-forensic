from app.schemas.analysis import AnalysisRequest, AnalysisResponse


def analyze_mock(request: AnalysisRequest) -> AnalysisResponse:
    return AnalysisResponse(
        analysisRequestId=request.analysisRequestId,
        fileId=request.fileId,
        fileType=request.fileType,
        status="MOCK_ANALYSIS_COMPLETED",
        rawScore=0.75,
        confidence=70,
        evidence=[
            "Mock 분석 결과입니다. 실제 모델은 Sprint 2 연동 예정."
        ],
        modelName="mock-deepfake-detector",
        modelVersion="v0.1",
    )
