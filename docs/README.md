# ForenShield AI docs

문서 목차. 레인(정식/폐기) 기준은 [overview/MODEL_LANES.md](./overview/MODEL_LANES.md).

## overview

| Doc | 내용 |
|-----|------|
| [MODEL_LANES.md](./overview/MODEL_LANES.md) | Deepfake / Forgery PROD·TRAIN·ARCHIVE 구분 |

## ops (배포 · GPU · S3)

| Doc | 내용 |
|-----|------|
| [GPU_WORKER_GUIDE.md](./ops/GPU_WORKER_GUIDE.md) | gpu_worker / Gateway |
| [GPU_DEPLOY_WELABS.md](./ops/GPU_DEPLOY_WELABS.md) | welabs 배포 |
| [FORENSHIELD_AI_GPU_WORKSTATION.md](./ops/FORENSHIELD_AI_GPU_WORKSTATION.md) | GPU 작업 공간 레이아웃 |
| [S3_DEEPFAKE_FOLDER_LAYOUT.md](./ops/S3_DEEPFAKE_FOLDER_LAYOUT.md) | S3 deepfake prefix |

## deepfake (1차 레인)

| Doc | 내용 |
|-----|------|
| [VIDEO_DEEPFAKE_MODEL_BENCHMARK_3x3.md](./deepfake/VIDEO_DEEPFAKE_MODEL_BENCHMARK_3x3.md) | 3×3 모델 벤치 |
| [VIDEO_BENCHMARK_DATASETS.md](./deepfake/VIDEO_BENCHMARK_DATASETS.md) | 벤치 데이터셋 |
| [VIDEO_DATASET_INVENTORY.md](./deepfake/VIDEO_DATASET_INVENTORY.md) | 데이터 인벤토리 |
| [GMFLOW_DEEPFAKE_SCORE.md](./deepfake/GMFLOW_DEEPFAKE_SCORE.md) | GMFlow 점수 |

## forgery (2차 레인)

| Doc | 내용 |
|-----|------|
| [TAMPERING_DETECTION_PIPELINE.md](./forgery/TAMPERING_DETECTION_PIPELINE.md) | 위변조 파이프라인 |
| [REGION_THRESHOLD_CALIBRATION.md](./forgery/REGION_THRESHOLD_CALIBRATION.md) | region / threshold (P2 설계 포함) |

## contracts (BE · JSON · 시각화)

| Doc | 내용 |
|-----|------|
| [AI_JSON_TIMELINE.md](./contracts/AI_JSON_TIMELINE.md) | 타임라인 JSON |
| [AI_VISUALIZATION_ARTIFACTS.md](./contracts/AI_VISUALIZATION_ARTIFACTS.md) | overlay / artifact |
| [integrations/sample_video_fusion_response.json](./contracts/integrations/sample_video_fusion_response.json) | 샘플 fusion 응답 |

## notebooks

| Path | 내용 |
|------|------|
| [notebooks/output/](./notebooks/output/) | KPI confusion-matrix notebooks |
| [notebooks/test/](./notebooks/test/) | `video_readiness` 출처 실험 노트북 |

## archive

| Path | 내용 |
|------|------|
| [_archive/](./_archive/) | 폐기 문서·노트북 (PWC-Net 등) |
