# ForenShield AI Server

ForenShield AI의 Python FastAPI 기반 AI 서버이다.

현재 서버는 Spring Boot 백엔드 연동 테스트를 위한 Mock 분석 API를 제공한다.

김민희 AI 서버 작업 정리는 아래 문서에 정리되어 있다.

```text
docs/KIMMINHEE_AI_SERVER_WORK_SUMMARY.md
```

## 실행 방법

프로젝트 위치로 이동한다.

```bash
cd /Users/kimmini/sk-final-deepfake/ai-forensic
```

Python 가상환경을 생성한다.

```bash
python3 -m venv .venv
```

가상환경을 활성화한다.

```bash
source .venv/bin/activate
```

패키지를 설치한다.

```bash
pip install -r requirements.txt
```

환경변수 예시 파일을 복사한다.

```bash
cp .env.example .env
```

FastAPI 서버를 실행한다.

```bash
uvicorn app.main:app --port 8000
```

브라우저에서 Swagger UI를 확인한다.

```text
http://localhost:8000/docs
```

## API 테스트

`/health` 상태 확인 API를 테스트한다.

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

`/ai/analyze` Mock 분석 API를 테스트한다.

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

## Docker 실행

이미지를 빌드한다.

```bash
docker build -t ai-forensic .
```

컨테이너를 실행한다.

```bash
docker run -p 8000:8000 ai-forensic
```

실행 후 동일하게 Swagger UI와 API를 확인한다.

```text
http://localhost:8000/docs
GET /health
POST /ai/analyze
```

## 테스트했던 실행 방식

로컬 검증에서는 다음 순서로 확인했다.

```text
1. python3 --version
2. python3 -m venv .venv
3. source .venv/bin/activate
4. pip install -r requirements.txt
5. uvicorn app.main:app --port 8000
6. http://localhost:8000/docs 확인
7. GET /health 호출
8. POST /ai/analyze 호출
```

`uvicorn app.main:app --reload --port 8000`은 로컬 권한 문제로 실패해서, 실제 API 검증은 `--reload` 없이 진행했다.
