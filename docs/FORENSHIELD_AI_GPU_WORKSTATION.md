# forenShield-ai GPU 워크스테이션 디렉터리 구조

> GPU 서버(예: `welabs`, `~/forenShield-ai`)에서 모델 다운로드·추론·평가·S3 배포를 관리하는 **작업 공간** 구조입니다.  
> 운영 API 서비스 코드는 이 저장소(`ai-forensic`)에, 모델 실험 작업 공간은 GPU의 `forenShield-ai`에 둡니다.

## 역할 구분

| 위치 | 역할 |
|------|------|
| `~/forenShield-ai` (GPU) | 공통 venv·config, **deepfake** / **forgery** 2트랙 |
| `~/forenShield-ai/deepfake` | **1차 딥페이크** 모델·데이터·infer/eval |
| `~/forenShield-ai/forgery` | **2차 위변조** TruFor·GMFlow(컷)·DC·region 프로파일 |
| `ai-forensic` (Git) | FastAPI `/ai/analyze`, Celery Worker, 배포용 서비스 코드 |
| S3 `forenshield-models-*` | `*/deploy/` 와 1:1 대응하는 배포 가중치 |
| S3 `forenshield-evidence-*` | 분석용 증거 copy (pull 테스트 입력) |

## 트랙 선택 (환경변수)

스크립트는 `FORENSHIELD_AI_ROOT`를 트랙 루트로 사용합니다.

```bash
cd ~/forenShield-ai
source .venv/bin/activate

# 1차 딥페이크
export FORENSHIELD_TRACK=deepfake
export FORENSHIELD_AI_ROOT="$HOME/forenShield-ai/deepfake"

# 2차 위변조
export FORENSHIELD_TRACK=forgery
export FORENSHIELD_AI_ROOT="$HOME/forenShield-ai/forgery"
```

`env.local`에 넣어 두고 `source config/env.local` 해도 됩니다.

---

## 디렉터리 트리 (전체)

```text
~/forenShield-ai/
├── .venv/                         # Python 3.12 — deepfake·forgery 공통
├── README.md
│
├── config/                        # 트랙 공통 설정
│   ├── env.local
│   ├── env.local.example
│   ├── buckets.yaml
│   ├── models.deepfake.yaml       # 1차 모델 레지스트리
│   └── models.forgery.yaml        # 2차 모델·프로파일 레지스트리
│
├── deepfake/                      # ===== 1차 딥페이크 =====
│   ├── models/
│   │   ├── test/
│   │   │   ├── video/xception|convnext|videomae|timesformer|video-swin/v1.0.0/
│   │   │   └── optical/gmflow|raft|pwcnet/
│   │   ├── dev/
│   │   └── deploy/
│   ├── data/
│   │   ├── test/video/
│   │   ├── train/video/           # fine-tune 풀
│   │   ├── golden/v1/
│   │   ├── pull/evidence|models|golden/
│   │   ├── raw/faceforensics|voxceleb|celeb-df-v2/
│   │   └── cache/
│   ├── results/infer|eval|reports/
│   ├── checkpoints/
│   └── scripts/download|infer|eval|promote|upload/
│
├── forgery/                       # ===== 2차 위변조 =====
│   ├── config/
│   │   ├── thresholds.yaml        # blur/block/fft H/L (보정 후 고정)
│   │   └── weights_2nd.yaml       # region fusion 가중치
│   ├── models/
│   │   ├── test/
│   │   │   ├── spatial/trufor|catnet|sparsevit/v1.0.0/
│   │   │   ├── temporal/gmflow/   # discontinuity (flow 가중치)
│   │   │   └── compression/dc/v1.0.0/
│   │   ├── dev/
│   │   └── deploy/
│   │       ├── spatial/trufor/v1.0.0/
│   │       └── profile/thresholds/v1.0.0/
│   ├── data/
│   │   ├── test/video|image/casia|imd2020/
│   │   ├── cal/threshold/         # 임계값 보정용 (GT + 랜덤 가공)
│   │   ├── cal/fusion/            # weights_2nd 튜닝용
│   │   ├── synth/regions/R0..R7/  # region 합성 eval
│   │   ├── golden/v1/
│   │   ├── pull/
│   │   ├── raw/
│   │   └── cache/
│   ├── results/infer|eval|reports/
│   ├── checkpoints/
│   └── scripts/
│       ├── profile/               # compression_profile, calibrate_thresholds
│       ├── data/                  # synthesize_region_variants
│       ├── download|infer|eval|promote|upload/
│
└── logs/
    ├── deepfake/
    └── forgery/
```

---

## deepfake vs forgery

| | `deepfake/` | `forgery/` |
|---|-------------|------------|
| **질문** | AI로 사람·신체가 합성됐나? | 편집·컷·재압축 흔적이 있나? |
| **모델** | Xception, VideoMAE, TimeSformer, GMFlow(motion) | TruFor, GMFlow(discontinuity), DC |
| **데이터** | FF++, Vox, Celeb-DF, DFDC | CASIA, IMD, CSVTED, cal/synth |
| **config** | `config/models.deepfake.yaml` | `forgery/config/thresholds.yaml`, `weights_2nd.yaml` |
| **문서** | [VIDEO_DEEPFAKE_MODEL_BENCHMARK_3x3.md](./VIDEO_DEEPFAKE_MODEL_BENCHMARK_3x3.md) | [TAMPERING_DETECTION_PIPELINE.md](./TAMPERING_DETECTION_PIPELINE.md) · [REGION_THRESHOLD_CALIBRATION.md](./REGION_THRESHOLD_CALIBRATION.md) |

### GMFlow 가중치 공유

1차·2차 모두 flow backbone은 동일할 수 있습니다.

```bash
# 선택: forgery에서 deepfake optical 가중치 symlink
ln -s ../../../deepfake/models/test/optical/gmflow \
  ~/forenShield-ai/forgery/models/test/temporal/gmflow
```

2차는 **점수 스크립트만 분리** (`motion_anomaly` vs `discontinuity`).

---

## 폴더별 설명 (공통 패턴)

각 트랙(`deepfake/`, `forgery/`) 안은 **동일한 3단계**를 따릅니다.

### `models/` — test / dev / deploy

| 단계 | 용도 | S3 |
|------|------|-----|
| `test/` | 첫 다운로드·실험 | ❌ |
| `dev/` | 튜닝·통합 검증 | ❌ |
| `deploy/` | 운영 확정·S3 sync | ✅ |

**deepfake 예:**

```text
deepfake/models/deploy/video/xception/v1.0.0/
├── xception_best.pth
└── manifest.json
```

**forgery 예:**

```text
forgery/models/deploy/spatial/trufor/v1.0.0/
├── trufor_weights.pth
└── manifest.json

forgery/models/deploy/profile/thresholds/v1.0.0/
├── thresholds.yaml
└── weights_2nd.yaml
```

### `data/`

| 경로 | deepfake | forgery |
|------|----------|---------|
| `test/` | Celeb-DF 50+50 등 벤치 | CASIA·IMD 샘플 |
| `train/` | FF++ fake + Vox real | (fine-tune 시) tamper+mask |
| `golden/` | 1차 회귀셋 | 2차 spatial+cut GT |
| `cal/threshold/` | — | GT + 랜덤 blur/JPEG → `thresholds.yaml` |
| `cal/fusion/` | — | region별 fusion 튜닝 |
| `synth/regions/` | — | R0~R7 합성 variant |
| `pull/evidence/` | S3 증거 copy | 동일 |

### `results/`

| 경로 | 기능 |
|------|------|
| `infer/` | 추론 1회 = RUN_ID 폴더 (json, heatmap) |
| `eval/` | golden·cal 대비 metrics |
| `reports/` | 팀 공유 md/csv |

### `scripts/`

| 하위 | 기능 |
|------|------|
| `download/` | HF/GitHub → `models/test/` |
| `infer/` | 배치 추론 → `results/infer/` |
| `eval/` | 채점 → `results/eval/` |
| `promote/` | test→dev→deploy |
| `upload/` | deploy → S3 |

**forgery 전용**

| 하위 | 기능 |
|------|------|
| `profile/` | `compression_profile.py`, `calibrate_thresholds.py` |
| `data/` | `synthesize_region_variants.py` |

`ai-forensic/scripts/` 의 스크립트는 GPU에 clone 후, 트랙 루트에 맞게 `FORENSHIELD_AI_ROOT`만 바꿔 실행합니다.

---

## S3 경로 매핑

### Models 버킷

```text
# 1차 (기존)
deepfake/models/deploy/video/xception/v1.0.0/
  ↔  s3://forenshield-models-.../video/xception/v1.0.0/

# 2차 (신규 prefix 예정)
forgery/models/deploy/spatial/trufor/v1.0.0/
  ↔  s3://forenshield-models-.../forgery/spatial/trufor/v1.0.0/

forgery/models/deploy/profile/thresholds/v1.0.0/
  ↔  s3://forenshield-models-.../forgery/profile/thresholds/v1.0.0/
```

### Evidence 버킷 (벤치 결과)

```text
# 1차 벤치
s3://forenshield-evidence-.../cases/test/video-benchmark-datasets/...

# 2차 벤치 (예정)
s3://forenshield-evidence-.../cases/test/forgery-benchmark-datasets/...
```

---

## 워크플로

### 1차 deepfake

```text
FORENSHIELD_AI_ROOT=~/forenShield-ai/deepfake

1. download/models     → deepfake/models/test/
2. infer (data/test)   → deepfake/results/infer/
3. eval                → deepfake/results/eval/
4. promote → deploy
5. upload              → S3 video/*
```

### 2차 forgery

```text
FORENSHIELD_AI_ROOT=~/forenShield-ai/forgery

1. download/models     → forgery/models/test/spatial/trufor/
2. data cal set + 랜덤 가공 → forgery/data/cal/threshold/
3. profile + calibrate → forgery/config/thresholds.yaml
4. infer TruFor+GMFlow+DC → forgery/results/infer/
5. eval_by_region      → forgery/results/eval/
6. tune weights_2nd.yaml
7. promote → deploy (trufor + profile bundle)
8. upload              → S3 forgery/*
```

---

## 초기 구조 생성 (원클릭)

GPU SSH 접속 후 **한 줄**:

```bash
bash ~/ai-forensic/scripts/setup_gpu_workstation.sh
```

`ai-forensic`이 없으면 먼저:

```bash
cd ~ && git clone https://github.com/sk-final-deepfake/ai-forensic.git
bash ~/ai-forensic/scripts/setup_gpu_workstation.sh
```

스크립트가 자동으로:

1. `deepfake/` · `forgery/` 폴더 skeleton 생성  
2. 루트에 옛 `models/`, `data/` 등이 있으면 → `deepfake/`로 이전  
3. `config/env.local` 템플릿 생성  
4. `.venv` 없으면 생성  

옵션:

```bash
SKIP_MIGRATE=1 bash ~/ai-forensic/scripts/setup_gpu_workstation.sh   # 이전 없이 빈 forgery만
DRY_RUN=1 bash ~/ai-forensic/scripts/setup_gpu_workstation.sh          # migrate 미리보기
FORENSHIELD_AI_ROOT=/data/forenShield-ai bash ~/ai-forensic/scripts/setup_gpu_workstation.sh
```

### 수동 (세부 단계)

<details>
<summary>init / migrate 를 따로 실행할 때</summary>

```bash
bash ~/ai-forensic/scripts/init_forenShield_ai_layout.sh
DRY_RUN=1 bash ~/ai-forensic/scripts/migrate_flat_to_track_layout.sh
bash ~/ai-forensic/scripts/migrate_flat_to_track_layout.sh
```

</details>

---

## 기존 스크립트 호환

`ai-forensic/scripts/infer/*.sh` 는 기본값이 `FORENSHIELD_AI_ROOT=$HOME/forenShield-ai` 입니다.  
**트랙 분리 후** 실행 예:

```bash
export FORENSHIELD_AI_ROOT=~/forenShield-ai/deepfake
bash ai-forensic/scripts/infer/run_videomae_celebdf_benchmark.sh
```

향후 스크립트는 `FORENSHIELD_TRACK` 기본값 `deepfake` 로 통일 예정.

---

## 관련 문서

- [TAMPERING_DETECTION_PIPELINE.md](./TAMPERING_DETECTION_PIPELINE.md) — 2차 파이프라인 설계
- [REGION_THRESHOLD_CALIBRATION.md](./REGION_THRESHOLD_CALIBRATION.md) — 임계값·region 합성
- [VIDEO_DEEPFAKE_MODEL_BENCHMARK_3x3.md](./VIDEO_DEEPFAKE_MODEL_BENCHMARK_3x3.md) — 1차 벤치
- [KIMMINHEE_AI_SERVER_WORK_SUMMARY.md](./KIMMINHEE_AI_SERVER_WORK_SUMMARY.md) — AI 서버 API
- [../README.md](../README.md) — `ai-forensic` FastAPI 실행
