# forenShield-ai GPU 워크스테이션 디렉터리 구조

> GPU 서버(예: `welabs`, `~/forenShield-ai`)에서 모델 다운로드·추론·평가·S3 배포를 관리하는 **작업 공간** 구조입니다.  
> 운영 API 서비스 코드는 이 저장소(`ai-forensic`)에, 모델 실험 작업 공간은 GPU의 `forenShield-ai`에 둡니다.

## 역할 구분

| 위치 | 역할 |
|------|------|
| `~/forenShield-ai` (GPU) | 모델 test/dev/deploy, 데이터, infer/eval, S3 sync |
| `ai-forensic` (Git) | FastAPI `/ai/analyze`, Celery Worker, 배포용 서비스 코드 |
| S3 `forenshield-models-*` | `models/deploy/` 와 1:1 대응하는 배포 가중치 |
| S3 `forenshield-evidence-*` | 분석용 증거 copy (pull 테스트 입력) |

## 디렉터리 트리

```text
~/forenShield-ai/
├── .venv/                         # Python 3.12 가상환경
├── README.md                      # 워크스테이션 로컬 안내
│
├── config/
│   ├── env.local                  # AWS_PROFILE, 리전 등 (git 제외)
│   ├── env.local.example
│   ├── models.yaml                # 모델별 test/dev/deploy 경로·S3 prefix
│   └── buckets.yaml               # evidence·models 버킷·prefix
│
├── models/                        # AI 가중치 (.pth, .bin 등)
│   ├── test/                      # 테스트용 — 실험·초기 다운로드
│   ├── dev/                       # 개발용 — 튜닝·통합·1차 검증 통과본
│   └── deploy/                    # 배포용 — S3 업로드 원본 (운영 확정)
│
├── data/                          # 입력 데이터 (모델 아님)
│   ├── test/                      # 실험용 테스트 데이터셋 (가변·임시)
│   │   └── video/                 # mp4 등 영상 샘플
│   ├── golden/                    # 회귀 테스트 고정셋 (로컬 master)
│   ├── pull/                      # S3에서 내려받은 데이터
│   │   ├── evidence/              # evidence 버킷 copy
│   │   ├── models/                # models 버킷 deploy pull (parity)
│   │   └── golden/                # (선택) S3 golden 백업 sync
│   ├── raw/                       # 로컬 임시·수동 입력
│   └── cache/                     # HF 캐시 등
│
├── results/                       # 출력물
│   ├── infer/                     # 추론 결과 (파일별 점수·판정)
│   ├── eval/                      # 평가 결과 (정확도·F1·latency)
│   └── reports/                   # 사람이 읽는 요약 (md/csv)
│
├── checkpoints/                   # dev 튜닝 중간 .pth
│
├── scripts/
│   ├── download/                  # 다운로드 전용
│   │   ├── models/                # HF/GitHub → models/test
│   │   ├── data/                  # golden 구성, S3 pull
│   │   └── deps/                  # torch 등 환경
│   ├── infer/                     # 추론 실행 → results/infer
│   ├── eval/                      # golden 대비 채점 → results/eval
│   ├── promote/                   # test→dev→deploy 승격
│   └── upload/                    # deploy·golden → S3
│
└── logs/                          # 스크립트 실행 로그
```

---

## 폴더별 설명

### 최상위

| 경로 | 기능 |
|------|------|
| `.venv/` | `torch`, `transformers` 등 AI 패키지가 설치되는 Python 가상환경 |
| `README.md` | GPU 서버에서의 빠른 시작·경로 요약 |
| `logs/` | 스크립트 stderr/stdout 보관, 장애 시 추적 |

### `config/`

| 파일 | 기능 |
|------|------|
| `env.local` | `AWS_PROFILE`, `AWS_REGION`, `S3_MODELS_BUCKET` 등 비밀·환경값 |
| `env.local.example` | `env.local` 작성 템플릿 |
| `models.yaml` | 모델 ID, test/dev/deploy 로컬 경로, S3 deploy prefix, 외부 다운로드 URL |
| `buckets.yaml` | evidence·models 버킷 이름, pull/upload 시 사용할 S3 prefix |

### `models/` — 3단계

| 단계 | 경로 | 용도 | S3 |
|------|------|------|-----|
| 테스트 | `models/test/` | 첫 다운로드, 빠른 실험, 깨진 가중치 허용 | ❌ |
| 개발 | `models/dev/` | 튜닝·`ai-forensic` 연동 전 통합, golden 1차 통과 | ❌ |
| 배포 | `models/deploy/` | 운영 확정본, `manifest.json` + SHA-256 고정 | ✅ sync 대상 |

각 모델 버전 폴더 예:

```text
models/deploy/video/xception/v1.0.0/
├── xception_best.pth
└── manifest.json
```

`manifest.json`: `modelId`, `version`, 파일 SHA-256, 입력 규격(예: 299×299 face crop).

### `data/`

| 경로 | 기능 |
|------|------|
| `test/video/` | **실험용 영상 샘플**. 유튜브·공개 데이터셋 등 빠른 infer 테스트 입력 (가변, S3 무관) |
| `golden/v1/video/` | **고정 회귀셋**. `manifest.json`에 파일·정답 라벨(real/fake). 모델 변경 시 동일 입력으로 비교 |
| `pull/evidence/` | `forenshield-evidence` 버킷에서 copy 객체를 pull — 운영 증거로 로컬 infer 테스트 |
| `pull/models/` | `forenshield-models` 버킷 deploy 경로 pull — 운영과 동일 가중치로 parity 검증 |
| `pull/golden/` | S3에 백업한 golden을 다른 머신과 sync (선택) |
| `raw/` | 다운로드 직후·전처리 전 임시 보관 (unzip, 변환 중간물) |
| `cache/` | HuggingFace·임시 unzip 캐시 |

#### `test/` vs `golden/` vs `pull/`

| 구분 | 경로 | 용도 |
|------|------|------|
| **test** | `data/test/` | 자유롭게 넣는 실험용 샘플. 정답 라벨 없어도 됨 |
| **golden** | `data/golden/` | 버전 고정 회귀셋. eval 시 정답과 비교 |
| **pull** | `data/pull/` | S3·운영 환경에서 내려받은 데이터 |

### `results/`

| 경로 | 기능 |
|------|------|
| `infer/` | **추론(inference)** 1회 = 폴더 1개. 파일별 fake 점수, 구간 점수, 처리 시간 |
| `eval/` | **평가(evaluation)**. infer 결과를 golden 정답과 비교 — accuracy, F1, latency |
| `reports/` | 팀 공유용 요약 문서 |

#### infer vs eval

| 용어 | 의미 | 출력 예 |
|------|------|---------|
| **infer** | 모델에 입력을 넣고 **점수·판정을 산출** | `video_a.mp4 → fake 0.87` |
| **eval** | infer 결과를 **정답과 비교해 모델 품질 지표** 산출 | `golden 10건 중 9건 일치, F1=0.92` |

### `checkpoints/`

dev 단계 **파인튜닝** 중 epoch별 중간 `.pth`. 최종본만 골라 `models/dev/` 또는 `deploy/`로 복사.

### `scripts/`

| 하위 | 기능 |
|------|------|
| `download/models/` | 외부 URL/HF → `models/test/` |
| `download/data/` | golden 구성, `s3_pull_evidence`, `s3_pull_deploy_model` 등 |
| `download/deps/` | torch·CUDA 패키지 설치 |
| `infer/` | 단일·배치 추론 → `results/infer/` |
| `eval/` | golden 대비 metrics → `results/eval/` |
| `promote/` | `test_to_dev`, `dev_to_deploy` 복사·manifest 갱신 |
| `upload/` | `models/deploy/` → S3 models 버킷 |

---

## S3 경로 매핑

### Models 버킷 (`forenshield-models-877044078824`)

```text
로컬  models/deploy/video/xception/v1.0.0/
  ↔  s3://forenshield-models-877044078824/video/xception/v1.0.0/

로컬  data/golden/v1/  (선택 백업)
  ↔  s3://forenshield-models-877044078824/golden-set/v1/
```

### Evidence 버킷 (`forenshield-evidence-877044078824`)

```text
s3://forenshield-evidence-.../cases/{case_id}/{file_id}/copy/...
  →  data/pull/evidence/cases/...
```

---

## 워크플로

```text
1. download/models     → models/test/
2. infer (data/test)   → results/infer/   # 빠른 스모크 테스트
2b. infer (golden)     → results/infer/   # 회귀·품질 검증
3. eval                → results/eval/
4. promote test→dev    → models/dev/   (튜닝 시 checkpoints/)
5. infer + eval (재검)
6. promote dev→deploy  → models/deploy/
7. upload              → S3 models 버킷
8. ai-forensic / Worker → S3 deploy pull → /ai/analyze
```

운영 증거로 테스트할 때:

```text
download/data/s3_pull_evidence  → data/pull/evidence/
infer (pull/evidence)           → results/infer/
```

---

## 초기 구조 생성

GPU SSH에서 `ai-forensic` 저장소를 clone한 뒤:

```bash
cd ~/forenShield-ai   # 또는 원하는 루트
bash /path/to/ai-forensic/scripts/init_forenShield_ai_layout.sh
```

기본 루트는 `$HOME/forenShield-ai`입니다. 변경:

```bash
FORENSHIELD_AI_ROOT=/data/forenShield-ai bash scripts/init_forenShield_ai_layout.sh
```

---

## 관련 문서

- [KIMMINHEE_AI_SERVER_WORK_SUMMARY.md](./KIMMINHEE_AI_SERVER_WORK_SUMMARY.md) — AI 서버 API·Mock 연동
- [../README.md](../README.md) — `ai-forensic` FastAPI 실행
