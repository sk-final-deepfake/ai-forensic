# 영상 딥페이크 모델 벤치마크 (3×3)`y`

> **작성 기준일:** 2026-06-22  
> **평가 환경:** welabs GPU (`sk4team@58.127.241.84`, `~/forenShield-ai`)  
> **S3 evidence 버킷:** `forenshield-evidence-877044078824`  
> **스크립트 위치:** `ai-forensic/scripts/infer/`, `scripts/upload/`

영상 딥페이크 탐지 모델을 **3가지 접근법 × 각 3개 모델**로 나누어 GPU에서 벤치마크합니다.


| 분류                | 영문                  | 보는 것                       | 출력                           |
| ----------------- | ------------------- | -------------------------- | ---------------------------- |
| **CNN (공간)**      | Spatial / Frame CNN | 프레임·얼굴 crop의 **공간적 위조 흔적** | `fake_score` (0~1)           |
| **Temporal (시간)** | Video Transformer   | **클립/프레임 시퀀스**의 시간적 불일치    | `fake_score` (0~1)           |
| **Optical (광학)**  | Optical Flow        | 연속 프레임 간 **움직임(flow) 통계**  | flow mean/max/std (분류 점수 없음) |


---

## 1. 모델 3×3 요약

### 1-A. CNN (Spatial) — 프레임 분류기

공통 전처리: OpenCV Haar 얼굴 crop 256×256 → 32프레임 샘플 → 프레임별 logits → `fake_score` = mean(`prob_fake`), threshold **0.5**


| #   | 모델           | model_id          | 가중치                               | Fine-tune             | 벤치마크 스크립트                          |
| --- | ------------ | ----------------- | --------------------------------- | --------------------- | ---------------------------------- |
| 1   | **Xception** | `xception/v1.0.0` | DeepfakeBench `xception_best.pth` | 없음 (FF++ 학습 ckpt 그대로) | `video_xception_benchmark_infer.p` |



|     |                     |                         |                                     |                              |                                           |
| --- | ------------------- | ----------------------- | ----------------------------------- | ---------------------------- | ----------------------------------------- |
| 2   | **EfficientNet-B4** | `efficientnetb4/v1.0.0` | DeepfakeBench `effnb4_best.pth`     | 없음                           | `video_efficientnetb4_benchmark_infer.py` |
| 3   | **ConvNeXt-Small**  | `convnext/v1.0.0`       | ImageNet → FF++100+Vox100 fine-tune | `video_convnext_finetune.py` | `video_convnext_benchmark_infer.py`       |


원클릭 (ConvNeXt): `run_convnext_celebdf_benchmark.sh`

### 1-B. Temporal — 비디오 트랜스포머 / MAE

공통 전처리: 얼굴 crop → clip(16프레임) → `fake_score`, threshold **0.5**  
Fine-tune: FF++fake 100 + Vox/FF++ real 100 (Celeb-DF 테스트셋 제외)


| #   | 모델              | model_id             | 백본                                         | Fine-tune                                           | 벤치마크 스크립트                                                  |
| --- | --------------- | -------------------- | ------------------------------------------ | --------------------------------------------------- | ---------------------------------------------------------- |
| 1   | **VideoMAE**    | `videomae/v1.0.0`    | `MCG-NJU/videomae-base`                    | `video_videomae_finetune.py`                        | `video_videomae_benchmark_infer.py`                        |
| 2   | **TimeSformer** | `timesformer/v1.0.0` | `facebook/timesformer-base-finetuned-k400` | `video_transformer_finetune.py --model timesformer` | `video_transformer_benchmark_infer.py --model timesformer` |
| 3   | **Video Swin**  | `video-swin/v1.0.0`  | torchvision `swin3d_t` (Kinetics400)       | `video_transformer_finetune.py --model video-swin`  | `video_transformer_benchmark_infer.py --model video-swin`  |


원클릭: `run_videomae_celebdf_benchmark.sh`, `run_video_transformer_celebdf_benchmark.sh` (`MODEL=timesformer|video-swin`)

### 1-C. Optical Flow — 움직임 분석

공통: 연속 프레임 쌍 optical flow → 통계 집계 (`optical_flow_common.py`). **fake/real 이진 점수 없음** — flow 특성만 JSON으로 저장.


| #   | 모델          | backend         | 가중치                                      | 벤치마크 스크립트                                      |
| --- | ----------- | --------------- | ---------------------------------------- | ---------------------------------------------- |
| 1   | **RAFT**    | `RaftBackend`   | `models/test/video/optical-flow/raft/`   | `optical_flow_infer_model.py --model raft`     |
| 2   | **GMFlow**  | `GmflowBackend` | `models/test/video/optical-flow/gmflow/` | `optical_flow_infer_model.py --model gmflow`   |
| 3   | **PWC-Net** | (레거시)           | `models/test/video/optical-flow/pwcnet/` | `optical_flow_benchmark_infer.py` (현재 백엔드 미연결) |


원클릭: `run_optical_flow_celebdf_benchmark.sh` (raft → gmflow → S3)

---

## 2. 테스트 데이터셋

벤치마크마다 **real/fake 출처가 다릅니다.** 숫자를 직접 비교할 때 주의하세요.

### 프로필 A — 교차 데이터셋 (`ffpp_vox`)


|          | 출처                                            | GPU 경로                               | 개수  |
| -------- | --------------------------------------------- | ------------------------------------ | --- |
| **Fake** | FaceForensics++ `DeepFakeDetection` c40, ≥60s | `data/test/video/ffpp/fake_over60s/` | 50  |
| **Real** | VoxCeleb2 YouTube long clip                   | `data/test/video/voxceleb/real/`     | 50  |


사용 모델: **Xception** (CNN #1)

### 프로필 B — Celeb-DF v2 in-domain (`celebdf`)


|          | 출처                                        | GPU 경로                              | 개수  |
| -------- | ----------------------------------------- | ----------------------------------- | --- |
| **Fake** | Celeb-DF v2 `Celeb-synthesis`             | `data/test/video/celeb-df-v2/fake/` | 50  |
| **Real** | Celeb-DF v2 `Celeb-real` + `YouTube-real` | `data/test/video/celeb-df-v2/real/` | 50  |


사용 모델: **VideoMAE, TimeSformer, Video Swin, RAFT, GMFlow, ConvNeXt** (및 EfficientNet-B4 기본값)

리포트 프로필: `bundle_xception_benchmark_report.py --profile celebdf`

---

## 3. GPU 실행 현황 (2026-06-22 기준)


| 분류       | 모델              | RUN_ID (대표)                                  | 테스트셋           | S3 prefix                                                 | 상태                          |
| -------- | --------------- | -------------------------------------------- | -------------- | --------------------------------------------------------- | --------------------------- |
| CNN      | Xception        | `xception-benchmark-20260618-0411`           | FF++50 + Vox50 | `cases/test/video-xception-benchmark/reports/`            | ✅ 완료                        |
| CNN      | EfficientNet-B4 | —                                            | —              | `cases/test/video-efficientnetb4-benchmark/reports/`      | ⏸ 스크립트만 (미실행)               |
| CNN      | ConvNeXt-Small  | `convnext-celebdf-benchmark-`*               | Celeb-DF 50+50 | `cases/test/video-convnext-celebdf-benchmark/reports/`    | 🔄 fine-tune 진행/재시작         |
| Temporal | VideoMAE        | `videomae-celebdf200-eval-20260619-0218`     | Celeb-DF 50+50 | `cases/test/video-videomae-celebdf-benchmark/reports/`    | ✅ 완료                        |
| Temporal | TimeSformer     | `optical-flow-celebdf-20260619-0504` ⚠️      | Celeb-DF 50+50 | `cases/test/video-timesformer-celebdf-benchmark/reports/` | ✅ 완료 (RUN_ID 오염 — 재실행 권장)   |
| Temporal | Video Swin      | `video-swin-celebdf-benchmark-20260619-0606` | Celeb-DF 50+50 | `cases/test/video-swin-celebdf-benchmark/reports/`        | ✅ 완료                        |
| Optical  | RAFT            | `optical-flow-celebdf-20260619-0504`         | Celeb-DF 50+50 | `cases/test/video-optical-flow-benchmark/reports/`        | ✅ 100/100                   |
| Optical  | GMFlow          | (동일 RUN_ID)                                  | Celeb-DF 50+50 | (동일)                                                      | ✅ 83/100                    |
| Optical  | PWC-Net         | —                                            | —              | —                                                         | ❌ correlation 빌드 실패, 백엔드 제거 |


⚠️ TimeSformer RUN_ID가 shell에 남아 있던 `RUN_ID=optical-flow-...` 값으로 섞임. 깨끗한 S3 경로가 필요하면 `unset RUN_ID` 후 재실행.

---

## 4. S3 결과 구조 (공통)

각 RUN_ID 아래:

```text
s3://forenshield-evidence-877044078824/cases/test/<prefix>/<RUN_ID>/
├── json/                    # 영상별 per-file JSON
├── predictions.json         # 전체 infer 결과
├── metrics.json             # accuracy 등 (CNN/Temporal)
├── benchmark_report.json    # bundle 통합 리포트
└── datasets/
    ├── fake/                # mp4 + manifest.json
    └── real/
```

Optical flow 추가:

```text
├── infer_summary_raft.json
├── infer_summary_gmflow.json
└── metrics_raft.json / metrics_gmflow.json
```

---

## 5. 모델별 재실행 명령 (GPU SSH)

공통 준비:

```bash
cd ~/forenShield-ai
source .venv/bin/activate
unset AWS_PROFILE
export OPENCV_FFMPEG_CAPTURE_OPTIONS="loglevel;quiet"
```

### CNN

```bash
# Xception — FF++50 + Vox50
python3 scripts/infer/video_xception_benchmark_infer.py --root . \
  --fake-dir data/test/video/ffpp/fake_over60s \
  --real-dir data/test/video/voxceleb/real
python3 scripts/infer/bundle_xception_benchmark_report.py <RUN_ID> --root . --profile ffpp_vox
S3_REPORT_PREFIX=cases/test/video-xception-benchmark/reports \
  bash scripts/upload/s3_upload_video_infer_results.sh <RUN_ID>

# EfficientNet-B4 — Celeb-DF (기본값)
python3 scripts/infer/video_efficientnetb4_benchmark_infer.py --root .

# ConvNeXt — fine-tune + Celeb-DF (가중치 있으면 SKIP_FINETUNE=1)
bash scripts/infer/run_convnext_celebdf_benchmark.sh
# 또는 테스트만:
SKIP_FINETUNE=1 bash scripts/infer/run_convnext_celebdf_benchmark.sh
```

### Temporal

```bash
# VideoMAE — Celeb-DF
bash scripts/infer/run_videomae_celebdf_benchmark.sh

# TimeSformer / Video Swin — Celeb-DF
unset RUN_ID
MODEL=timesformer bash scripts/infer/run_video_transformer_celebdf_benchmark.sh
MODEL=video-swin  bash scripts/infer/run_video_transformer_celebdf_benchmark.sh
```

### Optical

```bash
unset RUN_ID
bash scripts/infer/run_optical_flow_celebdf_benchmark.sh
```

---

## 6. 스크립트·가중치 경로 맵

```text
models/test/video/
├── xception/v1.0.0/xception_best.pth
├── efficientnetb4/v1.0.0/effnb4_best.pth
├── convnext/v1.0.0/convnext_finetuned.pth
├── videomae/v1.0.0/videomae_finetuned*.pth
├── timesformer/v1.0.0/timesformer_finetuned.pth
├── video-swin/v1.0.0/video_swin_finetuned.pth
└── optical-flow/{raft,gmflow,pwcnet}/

results/
├── infer/<RUN_ID>/json/
└── eval/<RUN_ID>/metrics.json
```

---

## 7. 분류별 선택 가이드


| 분류           | 강점                                        | 한계                                 |
| ------------ | ----------------------------------------- | ---------------------------------- |
| **CNN**      | 빠름, DeepfakeBench 베이스라인과 비교 용이            | 프레임 단위 — 시간적 불일치 약함                |
| **Temporal** | 클립 단위 시간 패턴, VideoMAE/Transformer SOTA 계열 | fine-tune 필요, VRAM·시간 큼            |
| **Optical**  | 조작과 무관한 **motion anomaly** 탐지 가능          | fake_score 없음 — 별도 threshold/규칙 필요 |


운영 연동 시: CNN/Temporal → `deepfakeDetected`, `deepfakeScore` / Optical → `frameAnomaly`, flow 기반 보조 신호.

---

## 8. 관련 문서


| 문서                                                           | 내용                                                  |
| ------------------------------------------------------------ | --------------------------------------------------- |
| [VIDEO_BENCHMARK_DATASETS.md](./VIDEO_BENCHMARK_DATASETS.md) | FF++ / VoxCeleb / Celeb-DF / DFDC 데이터셋·프로필·경로       |
| `FORENSHIELD_AI_GPU_WORKSTATION.md`                          | GPU 디렉터리 구조                                         |
| `AUDIO_DEEPFAKE_MODEL_EVALUATION_REPORT.md`                  | 음성 3모델 (Spectra / RawNet2 / Wav2Vec2) — 동일 평가 방식 참고 |
| `backend/docs/04-ai-json-spec.md`                            | API JSON 스키마                                        |


---

