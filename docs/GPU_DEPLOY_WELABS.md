# welabs GPU Worker 배포 가이드

> **서버:** `sk4team@58.151.205.220` (welabs)  
> **코드 repo:** `/home/sk4team/ai-forensic`  
> **모델/작업 루트:** `/home/sk4team/forenShield-ai/deepfake`  
> **환경설정:** `/home/sk4team/forenShield-ai/config/env.local`  
> **브랜치:** `feature/ai-multi-face-infer` (또는 `develop` — 동일 커밋 `5053904` 이후)

---

## 1. 이번 배포로 바뀌는 것

| 항목 | 내용 |
|------|------|
| **멀티 페이스 CNN** | 프레임당 YuNet으로 검출된 **모든 얼굴**에 Xception 점수 |
| **멀티 페이스 TimeSformer** | 클립 내 얼굴 슬롯별 temporal 점수 |
| **프레임별 Late Fusion** | CNN + TimeSformer per-face → fusion 점수로 오버레이 색상 |
| **오버레이 MP4** | 얼굴별 박스 + 위험도 컬러 → S3 업로드 → `overlayVideoUrl` |
| **프론트** | 증거 상세 **오버레이** 탭에서 MP4 재생 (재분석 필요) |

히트맵 JPG는 생성하지 않습니다. 오버레이 MP4만 사용합니다.

---

## 2. 서버 경로 맵

```
/home/sk4team/
├── ai-forensic/                    # git repo (gpu_worker, scripts/infer)
│   ├── gpu_worker/
│   ├── scripts/infer/              # DEEPFAKE_ROOT → infer 스크립트
│   └── scripts/deploy/welabs-gpu-worker.sh
├── forenShield-ai/
│   ├── .venv/                      # Python venv
│   ├── config/env.local            # ★ 환경변수 (아래 필수 항목 확인)
│   └── deepfake/                   # FORENSHIELD_AI_ROOT
│       ├── work/                   # 다운로드·시각화 임시 파일
│       ├── models/test/video/...   # Xception, TimeSformer, GMFlow weights
│       └── config/fusion_v4_ts_gated.json
└── .cache/forenshield/opencv/
    └── face_detection_yunet_2023mar.onnx
```

**S3 오버레이 경로:**
```
s3://forenshield-evidence-877044078824/deepfake/artifacts/analysis/{evidence_id}/{analysis_request_id}/overlay.mp4
```

---

## 3. env.local — Git 금지, SCP로만 적용

**`env.local`은 절대 Git에 올리지 않습니다.** (RabbitMQ 비밀번호, AWS 키 등 포함)

서버에 이미 있는 파일을 유지·수정합니다:

```
/home/sk4team/forenShield-ai/config/env.local   ← 운영 env (SCP 또는 서버에서 직접 편집)
```

### 로컬에서 서버로 SCP (비밀 env 갱신 시)

```bash
# 로컬에 env.local을 준비한 뒤 (Git에 커밋하지 말 것)
scp -i <SSH_KEY> ./env.local sk4team@58.151.205.220:/home/sk4team/forenShield-ai/config/env.local
```

### env.local에 반드시 있어야 하는 항목

기존 RabbitMQ/AWS 값은 **그대로 두고**, 아래만 추가·확인:

```bash
INFERENCE_MODE=local_model    # ★ 없으면 test 모드(가짜 점수)
USE_MOCK_INFER=0
INFER_DEVICE=cuda

FORENSHIELD_AI_ROOT=/home/sk4team/forenShield-ai/deepfake
DEEPFAKE_ROOT=/home/sk4team/ai-forensic

AI_VISUALIZATION_ENABLED=1
AI_VISUALIZATION_UPLOAD=1
AI_VISUALIZATION_OVERLAY=1
```

모델 weight 절대경로·`AI_VISUALIZATION_*`·RabbitMQ는 welabs 서버 기존 `env.local` 값 유지.

`gpu_worker/.env`는 **쓰지 않아도 됩니다.** `config.py`가 `forenShield-ai/config/env.local`을 자동 로드합니다.  
또는 worker 기동 전 `source env.local` (둘 다 가능).

---

## 4. 코드 배포 — git pull (env는 SCP 별도)

**코드**는 repo에서 pull, **설정**은 SCP/서버 편집으로 분리합니다.

```bash
ssh sk4team@58.151.205.220

cd /home/sk4team/ai-forensic
git fetch origin
git checkout feature/ai-multi-face-infer
git pull origin feature/ai-multi-face-infer
```

로컬에서 특정 파일만 급히 넘길 때 (비상용):

```bash
scp -i <SSH_KEY> -r gpu_worker/ scripts/infer/ app/services/visualization_artifacts.py \
  sk4team@58.151.205.220:/home/sk4team/ai-forensic/
```

일반적으로는 **git pull이 정석**이고, env만 SCP입니다.

---

## 5. worker 재시작

```bash
ssh sk4team@58.151.205.220

cd /home/sk4team/ai-forensic
git fetch origin
git checkout feature/ai-multi-face-infer
git pull origin feature/ai-multi-face-infer

# env에 INFERENCE_MODE=local_model 있는지 확인
grep INFERENCE_MODE /home/sk4team/forenShield-ai/config/env.local

pkill -f gpu_worker.rabbitmq_worker || true

source /home/sk4team/forenShield-ai/.venv/bin/activate
source /home/sk4team/forenShield-ai/config/env.local
unset AWS_PROFILE

cd /home/sk4team/ai-forensic
mkdir -p /home/sk4team/forenShield-ai/logs
nohup python -m gpu_worker.rabbitmq_worker \
  >> /home/sk4team/forenShield-ai/logs/gpu_worker.log 2>&1 &

pgrep -af gpu_worker
tail -f /home/sk4team/forenShield-ai/logs/gpu_worker.log
```

### 원클릭 스크립트

```bash
cd /home/sk4team/ai-forensic
chmod +x scripts/deploy/welabs-gpu-worker.sh
./scripts/deploy/welabs-gpu-worker.sh
```

---

## 6. 배포 후 검증

### 6.1 Worker 로그

```bash
tail -f /home/sk4team/forenShield-ai/logs/gpu_worker.log
```

분석 요청 시 기대 로그:
```
Processing job analysisRequestId=... evidenceId=...
Using fused per-face scores for visualization: ... points=N
Visualization artifacts attached: ... overlay=True
Published result ... -> ai.result.exchange/result.video
```

### 6.2 S3 오버레이 확인

```bash
aws s3 ls s3://forenshield-evidence-877044078824/deepfake/artifacts/analysis/{evidence_id}/{analysis_request_id}/
# overlay.mp4, frame_00.jpg 등
```

### 6.3 프론트 (증거 상세)

1. **새로 분석** 실행 (기존 증거는 이전 overlay 없으면 재분석 필요)
2. 사이트 F5 + step-up 재인증
3. 증거 상세 → **오버레이** 탭
4. BE가 `overlayVideoUrl` presigned refresh (`VisualizationArtifactUrlRefresher`)

---

## 7. EKS ai-fastapi와 동시 실행 주의

현재 welabs는 **Method A** (`rabbitmq_worker`가 queue consume)입니다.

| 실행 중 | 결과 |
|---------|------|
| GPU `rabbitmq_worker` + EKS `AnalysisConsumer` **동시** | job이 나뉘어 처리됨 — **금지** |
| GPU worker만 | ✅ welabs 운영 방식 |
| EKS consumer만 + GPU Gateway | Method B (다른 배포 방식) |

EKS `ai-fastapi` consumer가 켜져 있으면 GPU worker와 **하나만** 사용하세요.

---

## 8. 트러블슈팅

| 증상 | 원인 | 조치 |
|------|------|------|
| 오버레이 탭 빈 화면 / mock UI | `overlayVideoUrl` 없음 | 재분석, 로그에서 `overlay=True` 확인 |
| 가짜 점수만 나옴 | `INFERENCE_MODE=test` (기본값) | env.local에 `local_model` 설정 |
| overlay.mp4 S3 없음 | `AI_VISUALIZATION_UPLOAD=0` 또는 얼굴 미검출 | env 확인, YuNet 캐시 확인 |
| ModuleNotFoundError torch | venv 미활성화 | `source .venv/bin/activate` |
| Xception weights not found | 경로 오타 | `MODEL_CHECKPOINT_PATH` 절대경로 확인 |
| 프론트 URL 만료 | 예전 presigned 저장 | BE 배포 후 detail API 재조회 (F5) |

---

## 9. 프론트·백엔드 (이미 배포된 경우)

| 레이어 | 필요 작업 |
|--------|----------|
| **GPU (welabs)** | 이 문서대로 pull + worker 재시작 |
| **Backend** | `overlayVideoUrl` presigned refresh — develop 배포됨 |
| **Frontend** | 오버레이 탭 — develop 배포됨 |
| **재분석** | GPU 배포 **이후** 돌린 분석만 멀티페이스 오버레이 반영 |

---

*커밋 기준: `5053904` Add multi-face TimeSformer inference and fused overlay scores.*
