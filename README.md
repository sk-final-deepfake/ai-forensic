# ForenShield AI Server

ForenShield AI의 Python FastAPI 기반 AI 서버이다.

Sprint 1에서는 실제 딥페이크 탐지 모델 없이 `GET /health`, `POST /ai/analyze` Mock API를 제공한다.

## 가상환경 생성

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 패키지 설치

```bash
pip install -r requirements.txt
```

## 환경변수 설정

```bash
cp .env.example .env
```

## 서버 실행

```bash
uvicorn app.main:app --reload --port 8000
```

## Health API 테스트

```bash
curl http://localhost:8000/health
```

예상 응답:

```json
{
  "status": "ok",
  "service": "forenshield-ai"
}
```

## Mock Analyze API 테스트

```bash
curl -X POST http://localhost:8000/ai/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "analysisRequestId": 101,
    "fileId": 15,
    "caseId": 3,
    "fileType": "video",
    "s3ObjectKey": "original-files/3/15/original.mp4",
    "presignedDownloadUrl": "https://example.com/presigned-url",
    "originalSha256": "abc123...",
    "requestedAt": "2026-06-10T10:30:00"
  }'
```

예상 응답:

```json
{
  "analysisRequestId": 101,
  "fileId": 15,
  "fileType": "video",
  "status": "MOCK_ANALYSIS_COMPLETED",
  "rawScore": 0.75,
  "confidence": 70,
  "evidence": [
    "Mock 분석 결과입니다. 실제 모델은 Sprint 2 연동 예정."
  ],
  "modelName": "mock-deepfake-detector",
  "modelVersion": "v0.1"
}
```

## Sprint 1 범위

- FastAPI 기본 서버 구조
- `GET /health`
- `POST /ai/analyze` Mock API
- Request/Response 스키마
- SHA-256 유틸 함수

## Sprint 1 제외 범위

- 실제 탐지 모델
- PyTorch 모델 로딩
- OpenCV/Librosa 분석
- STT
- 화자 분리 및 화자 검증
- RabbitMQ Consumer
- S3 다운로드 실구현
- Result API 호출 실구현
- GPU 추론
