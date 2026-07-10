# S3 딥페이크 폴더 구조 (정리안)

> **원칙:** 삭제 없이 **복사(sync)** 로 새 prefix에 정리. 기존 경로는 당분간 유지.  
> **범위:** 딥페이크(1차)만. `forgery-*`·운영 `cases/{caseKey}/` 는 별도.  
> **버킷:** `forenshield-evidence-877044078824`, `forenshield-models-877044078824`

GPU에서 S3 작업 전:

```bash
source ~/forenShield-ai/config/env.local
unset AWS_PROFILE
aws sts get-caller-identity
```

---

## 1. 폴더 이름 규칙 (prefix = 용도)

| prefix | 의미 | 예시 |
|--------|------|------|
| `datasets/train/` | 학습·fine-tune용 clip/manifest | Xception 1k finetune |
| `datasets/golden/` | 회귀·골든셋 (고정 라벨) | golden v1 200건 |
| `datasets/bench/` | 공식 벤치마크 입력 mp4 (50+50 등) | celebdf, ffpp_vox |
| `datasets/field/` | 현장·유튜브·adhoc 테스트 영상 | youtube-shorts |
| `results/perf/` | 성능 요약 (metrics, benchmark_report, CM/ROC) | 모델×데이터셋 점수 |
| `results/infer/` | 영상별 infer JSON bundle | xception/celebdf/*.json |
| `artifacts/analysis/` | 운영 분석 시각화 (heatmap, overlay) | evidence별 artifact |
| `archive/legacy/` | 초기 실험·레거시 benchmark (더 이상 파이프라인 미사용) | video-xception-benchmark |
| `deploy/` | **모델 버킷** — 운영 배포 가중치 | xception, timesformer, gmflow |
| `bench/` | **모델 버킷** — 3×3 비교·실험용 가중치 | convnext, video-swin |
| `archive/` | **모델 버킷** — 구버전 스냅샷 | v1.0, v1.1 루트 |

---

## 2. Evidence 버킷 — 목표 트리

```text
s3://forenshield-evidence-877044078824/

cases/                                    # [서비스] BE 업로드 — 이동 금지
└── {caseKey}/{evidenceId}/
    ├── original/                         # WORM 원본
    ├── copy/                             # AI 분석 입력
    ├── manifest/
    └── reports/

deepfake/                                 # [AI] 딥페이크 전용 — 이번 정리 대상
├── README.md                             # 이 문서 링크·버전
│
├── datasets/
│   ├── train/
│   │   └── video/xception/               # ← cases/train/video/xception/
│   ├── golden/
│   │   └── v1/video/                     # 골든 200 등 (로컬 golden → 업로드)
│   ├── bench/
│   │   ├── celebdf/{fake,real}/          # ← video-benchmark-datasets/celebdf/
│   │   └── ffpp_vox/{fake,real}/         # ← video-benchmark-datasets/ffpp_vox/
│   └── field/
│       └── youtube-shorts/               # ← cases/test/youtube-shorts-adhoc/
│
├── results/
│   ├── perf/
│   │   └── {model}/{profile}/            # metrics.json, benchmark_report.json
│   └── infer/
│       └── {model}/{profile}/            # per-video JSON, infer_summary.json
│           ├── fake/
│           └── real/
│
├── artifacts/
│   └── analysis/
│       └── {evidence_id}/{analysis_request_id}/   # heatmap, overlay mp4
│
└── archive/
    └── legacy-benchmarks/                # 레거시 per-model reports (복사본)
        ├── video-xception-benchmark/
        ├── video-videomae-celebdf-benchmark/
        ├── video-timesformer-celebdf-benchmark/
        ├── video-swin-celebdf-benchmark/
        ├── video-convnext-celebdf-benchmark/
        ├── video-videomae-benchmark/
        ├── video-optical-flow-benchmark/
        └── video-raft-ffpp-vox-benchmark/
```

### 2-1. `video-benchmark-datasets/` 분리 규칙

현재 통합 prefix 안에 **데이터셋**과 **모델 결과**가 섞여 있음 → 복사 시 분리:

| 현재 (`cases/test/video-benchmark-datasets/`) | 새 위치 | 분류 |
|-----------------------------------------------|---------|------|
| `celebdf/{fake,real}/` | `deepfake/datasets/bench/celebdf/` | bench 입력 |
| `ffpp_vox/{fake,real}/` | `deepfake/datasets/bench/ffpp_vox/` | bench 입력 |
| `xception/{profile}/` | `deepfake/results/infer/xception/{profile}/` | infer |
| `timesformer/{profile}/` | `deepfake/results/infer/timesformer/{profile}/` | infer |
| `videomae/{profile}/` | `deepfake/results/infer/videomae/{profile}/` | infer |
| `video-swin/{profile}/` | `deepfake/results/infer/video-swin/{profile}/` | infer |
| `convnext/{profile}/` | `deepfake/results/infer/convnext/{profile}/` | infer |
| `raft/{profile}/` | `deepfake/results/infer/raft/{profile}/` | infer |
| `gmflow/{profile}/` | `deepfake/results/infer/gmflow/{profile}/` | infer |
| `PWC-Net/{profile}/` | `deepfake/archive/legacy-benchmarks/pwcnet/{profile}/` | 실험 종료 |

각 `results/infer/{model}/{profile}/` 에 `infer_summary.json`, `metrics.json` 이 있으면  
**복사본**을 `results/perf/{model}/{profile}/` 에도 두면 성능만 따로 찾기 쉬움.

---

## 3. Models 버킷 — 목표 트리

```text
s3://forenshield-models-877044078824/

deepfake/
├── README.md
├── deploy/                               # 운영 RabbitMQ/GPU 워커가 pull
│   └── video/
│       ├── xception/v1.0.0/
│       ├── timesformer/v1.0.0/
│       └── optical/gmflow/               # learned head 포함 시 하위 manifest
├── bench/                                # 3×3 비교·ablation (운영 미사용)
│   └── video/
│       ├── convnext/v1.0.0/
│       ├── videomae/v1.0.0/
│       └── video-swin/v1.0.0/
└── archive/                              # 구조 실험·미사용 스냅샷
    ├── root-v1.0/                        # ← 기존 루트 v1.0/
    ├── root-v1.1/                        # ← 기존 루트 v1.1/
    ├── root-test/                        # ← 기존 루트 test/
    └── root-test-sets/                   # ← 기존 루트 test-sets/

video/                                    # [레거시] deploy 복사 완료 전까지 유지
```

### 운영 모델 vs 벤치 모델

| 용도 | 모델 | S3 위치 |
|------|------|---------|
| **운영 late fusion** | Xception, TimeSformer, GMFlow learned head | `deepfake/deploy/video/...` |
| **벤치 3×3만** | ConvNeXt-S, VideoMAE, Video Swin, RAFT, PWC-Net | `deepfake/bench/` 또는 `archive/` |

---

## 4. 현재 → 신규 매핑 (삭제 없음, 복사만)

### Evidence

| # | 현재 prefix | 신규 prefix | 태그 |
|---|-------------|-------------|------|
| 1 | `cases/{caseKey}/...` | *(변경 없음)* | **서비스** |
| 2 | `cases/train/video/xception/` | `deepfake/datasets/train/video/xception/` | 학습 |
| 3 | `cases/test/video-benchmark-datasets/celebdf/` | `deepfake/datasets/bench/celebdf/` | 벤치 |
| 4 | `cases/test/video-benchmark-datasets/ffpp_vox/` | `deepfake/datasets/bench/ffpp_vox/` | 벤치 |
| 5 | `cases/test/video-benchmark-datasets/{model}/` | `deepfake/results/infer/{model}/` | infer |
| 6 | `cases/test/video-*-benchmark/` | `deepfake/archive/legacy-benchmarks/{name}/` | 레거시 |
| 7 | `cases/test/youtube-shorts-adhoc/` | `deepfake/datasets/field/youtube-shorts/` | 현장 |
| 8 | `cases/test/test-sine/` | `deepfake/archive/legacy-experiments/test-sine/` | 미사용 |
| 9 | `cases/analysis-artifacts/` (신규) | `deepfake/artifacts/analysis/` | 서비스 artifact |

### Models

| # | 현재 prefix | 신규 prefix | 태그 |
|---|-------------|-------------|------|
| 1 | `video/xception/`, `video/timesformer/`, `video/gmflow/` | `deepfake/deploy/video/...` | 운영 |
| 2 | `video/convnext/`, `video/video-swin/` 등 | `deepfake/bench/video/...` | 벤치 |
| 3 | `v1.0/`, `v1.1/` | `deepfake/archive/root-v1.0/` 등 | 아카이브 |
| 4 | `test/`, `test-sets/` | `deepfake/archive/root-test/` 등 | 아카이브 |

---

## 5. 마이그레이션 실행

스크립트: `scripts/upload/s3_reorganize_deepfake_layout.sh`

```bash
cd ~/forenShield-ai   # 또는 ai-forensic clone 경로
source config/env.local
unset AWS_PROFILE

# 1) 미리보기 (기본)
bash scripts/upload/s3_reorganize_deepfake_layout.sh

# 2) Evidence 복사만
APPLY=1 PHASE=evidence bash scripts/upload/s3_reorganize_deepfake_layout.sh

# 3) Models 복사만
APPLY=1 PHASE=models bash scripts/upload/s3_reorganize_deepfake_layout.sh
```

- `APPLY=0` (기본): `--dryrun` 만 출력  
- **삭제 없음** — `aws s3 sync` 복사만  
- 완료 후 신규 경로 목록으로 검증  
- **스크립트 기본 prefix**는 `scripts/common/s3_deepfake_paths.{py,sh}` 로 `deepfake/` 사용 (env로 override 가능)  
- 레거시 prefix 삭제는 팀 합의 + 30일 유예 후 별도 작업

---

## 5-1. Phase 2 — 옮긴 파일만 원본 삭제

**위변조(`forgery-*`)·운영 `cases/{caseKey}/` 는 절대 삭제하지 않음.**

스크립트: `scripts/upload/s3_prune_migrated_deepfake_sources.py`

삭제 조건 (객체 단위):

1. Phase 1 `sync` 에 포함된 **src → dst 쌍**에만 해당  
2. `dst` 에 **동일 key·동일 ContentLength** 가 있을 때만 `src` 삭제  
3. dst 가 없거나 크기가 다르면 **SKIP** (삭제 안 함)

```bash
source ~/forenShield-ai/config/env.local
unset AWS_PROFILE

# dry-run (삭제 예정 목록만)
python3 ~/ai-forensic/scripts/upload/s3_prune_migrated_deepfake_sources.py

# Evidence 원본만 삭제
APPLY=1 PHASE=evidence python3 ~/ai-forensic/scripts/upload/s3_prune_migrated_deepfake_sources.py

# Models 원본만 삭제
APPLY=1 PHASE=models python3 ~/ai-forensic/scripts/upload/s3_prune_migrated_deepfake_sources.py
```

`SKIP (no matching dst)` 가 많으면 Phase 1 복사를 해당 prefix 에 대해 다시 실행한 뒤 prune 하세요.

---

## 6. 환경 변수 (신규 prefix)

`config/env.local` 또는 워커 `.env` 예시:

```bash
S3_DEEPFAKE_DATASETS_BENCH=deepfake/datasets/bench
S3_DEEPFAKE_RESULTS_INFER=deepfake/results/infer
S3_DEEPFAKE_RESULTS_PERF=deepfake/results/perf
S3_DEEPFAKE_MODELS_DEPLOY=deepfake/deploy/video
AI_VISUALIZATION_PREFIX=deepfake/artifacts/analysis/{evidence_id}/{analysis_request_id}
```

---

## 7. 관련 문서

- [FORENSHIELD_AI_GPU_WORKSTATION.md](./FORENSHIELD_AI_GPU_WORKSTATION.md) — GPU 로컬 `~/forenShield-ai/deepfake/` 구조
- [VIDEO_DEEPFAKE_MODEL_BENCHMARK_3x3.md](./VIDEO_DEEPFAKE_MODEL_BENCHMARK_3x3.md) — 벤치 RUN_ID (레거시 경로 참조)
- [AI_VISUALIZATION_ARTIFACTS.md](./AI_VISUALIZATION_ARTIFACTS.md) — 분석 artifact
