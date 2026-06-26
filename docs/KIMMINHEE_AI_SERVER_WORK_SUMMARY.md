# 김민희 AI 서버 작업 정리

## 1. 작업 개요

ForenShield AI 프로젝트에서 김민희는 AI 서버 담당으로 Sprint 1 범위의 FastAPI 기반 AI 서버 준비 작업을 진행했다.

이번 작업의 목적은 실제 딥페이크 탐지 모델을 붙이기 전에, 백엔드와 연동 가능한 AI 서버 기본 구조와 Mock 분석 API를 준비하는 것이다.

## 2. 프로젝트 위치

AI 서버 프로젝트는 다음 위치에 생성했다.

```text
/Users/kimmini/sk-final-deepfake/ai-forensic
```

프로젝트 이름은 백엔드 프로젝트명 `backend-forensic`과 구분하기 위해 `ai-forensic`으로 정했다.

## 3. 문서 정리 작업

AI 서버 작업을 시작하기 전에 프로젝트 맥락과 구조를 문서로 정리했다.

작성한 주요 문서는 다음과 같다.

```text
docs/AI_SERVER_CONTEXT.md
docs/AI_SERVER_STRUCTURE.md
docs/AI_SERVER_BUILD_MEMORY.md
docs/KIMMINHEE_NEXT_CHECKLIST.md
docs/AI_SERVER_VENV_SETUP.md
```

각 문서의 목적은 다음과 같다.

- `AI_SERVER_CONTEXT.md`: ForenShield AI 전체 목적, AI 서버 역할, 백엔드와의 연결 방식, Sprint 1 범위 정리
- `AI_SERVER_STRUCTURE.md`: Sprint 1 최소 파일 구조와 Sprint 2 이후 확장 구조 정리
- `AI_SERVER_BUILD_MEMORY.md`: AI 서버 프로젝트를 어떤 기준으로 만들었는지에 대한 작업 기록
- `KIMMINHEE_NEXT_CHECKLIST.md`: 김민희가 다음에 해야 할 작업 체크리스트
- `AI_SERVER_VENV_SETUP.md`: 가상환경 생성, 서버 실행, API 테스트 절차 정리

## 4. Sprint 1 AI 서버 구조 생성

Sprint 1 기준으로 FastAPI 서버 최소 구조를 생성했다.

생성한 주요 구조는 다음과 같다.

```text
app/
├── main.py
├── core/
│   └── config.py
├── routers/
│   ├── health.py
│   └── analyze.py
├── schemas/
│   └── analysis.py
├── services/
│   └── mock_analyzer.py
└── utils/
    └── hash_utils.py
```

추가로 실행에 필요한 다음 파일을 준비했다.

```text
requirements.txt
.env.example
.gitignore
README.md
```

## 5. 구현한 API

Sprint 1에서 구현한 API는 다음 두 개이다.

```http
GET /health
POST /ai/analyze
```

`GET /health`는 AI 서버 상태 확인용 API이다.

예상 응답:

```json
{
  "status": "ok",
  "service": "forenshield-ai"
}
```

`POST /ai/analyze`는 실제 모델 없이 백엔드 연동 테스트를 위한 Mock 분석 API이다.

Mock 응답에는 다음 정보가 포함된다.

```text
analysisRequestId
fileId
fileType
status
rawScore
confidence
evidence
modelName
modelVersion
```

## 6. 가상환경 및 실행 검증

로컬에서 Python 가상환경을 만들고 FastAPI 서버 실행까지 확인했다.

확인한 Python 버전:

```text
Python 3.12.2
```

진행한 실행 검증은 다음과 같다.

```text
1. python3 -m venv .venv
2. .venv/bin/pip install -r requirements.txt
3. .venv/bin/uvicorn app.main:app --port 8000
4. http://localhost:8000/docs 확인
5. GET /health 호출 테스트
6. POST /ai/analyze 호출 테스트
```

`--reload` 옵션은 로컬 권한 문제로 실패했기 때문에, API 동작 검증은 reload 없이 진행했다.

정상 확인된 결과는 다음과 같다.

```text
HEAD /docs -> 200 OK
GET /health -> 200 OK
POST /ai/analyze -> 200 OK
```

## 7. Sprint 1에서 하지 않은 것

Sprint 1 범위를 넘지 않기 위해 다음 작업은 진행하지 않았다.

- 실제 딥페이크 탐지 모델 구현
- PyTorch 모델 로딩
- OpenCV/Librosa 기반 실제 분석
- STT 구현
- 화자 분리 및 화자 검증
- RabbitMQ Consumer 구현
- S3 다운로드 실제 구현
- Spring Boot Result API 호출 실제 구현
- GPU 추론
- Sprint 2 확장 폴더 생성

## 8. 현재 상태

현재 AI 서버는 Sprint 1 Mock API 기준으로 로컬 실행과 API 응답 확인이 완료된 상태이다.

다음 단계에서는 백엔드 담당자에게 AI 서버 Base URL과 API 정보를 공유하고, 백엔드의 요청/응답 JSON 명세와 현재 Mock API 스키마가 일치하는지 확인해야 한다.

공유할 기본 정보는 다음과 같다.

```text
AI Server Base URL:
http://localhost:8000

Health Check:
GET /health

Mock Analyze:
POST /ai/analyze
```

## 9. 다음 작업

다음 작업 후보는 다음과 같다.

```text
1. README 실행 방법 보완
2. 백엔드 담당자에게 연동 정보 공유
3. /ai/analyze 요청/응답 JSON 백엔드 명세와 비교
4. Sprint 2 모델 후보 정리
5. 실제 모델 포팅 준비
```

