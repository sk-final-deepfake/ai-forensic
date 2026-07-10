# 영상 벤치마크 데이터셋 정리

> **작성 기준일:** 2026-06-22  
> **GPU 작업 경로:** `~/forenShield-ai` (`sk4team@58.127.241.84`)  
> **관련 문서:** [VIDEO_DEEPFAKE_MODEL_BENCHMARK_3x3.md](./VIDEO_DEEPFAKE_MODEL_BENCHMARK_3x3.md), [FORENSHIELD_AI_GPU_WORKSTATION.md](./FORENSHIELD_AI_GPU_WORKSTATION.md), [VIDEO_DATASET_INVENTORY.md](./VIDEO_DATASET_INVENTORY.md) (GPU 실측 개수)

ForenShield AI 영상 딥페이크 벤치마크에서 사용하는 **데이터셋·프로필·경로·다운로드 방법**을 정리합니다.

---

## 1. 한눈에 보기

### 1-1. 벤치마크 프로필 (테스트 50+50)

| 프로필 ID | Fake 50 | Real 50 | 의미 | 사용 모델 (대표) |
|-----------|---------|---------|------|------------------|
| **`ffpp_vox`** | FF++ DeepFakeDetection | VoxCeleb long | **교차 데이터셋** (fake/real 출처 다름) | Xception |
| **`celebdf`** | Celeb-DF v2 synthesis | Celeb-DF v2 real+YouTube | **동일 데이터셋 in-domain** | VideoMAE, TimeSformer, Video Swin, RAFT, GMFlow, ConvNeXt |
| **`dfdc`** | DFDC / HF subset fake | 동일 소스 real | in-domain (준비 중) | VideoMAE (`run_videomae_dfdc_benchmark.sh`) |

리포트 bundle: `bundle_xception_benchmark_report.py --profile {ffpp_vox|celebdf|dfdc}`

### 1-2. Train / Test 분리

| 용도 | Fake | Real | 개수 | 비고 |
|------|------|------|------|------|
| **Fine-tune** | FF++ `DeepFakeDetection/c40` | VoxCeleb 또는 FF++ original | 100 + 100 | Celeb-DF·벤치마크 경로 `--exclude-dirs`로 제외 |
| **Test (Celeb-DF)** | `celeb-df-v2/fake` | `celeb-df-v2/real` | 50 + 50 | 학습에 사용 금지 |
| **Test (FF+++Vox)** | `ffpp/fake_over60s` | `voxceleb/real` | 50 + 50 | Xception 1차 벤치마크 |

**주의:** Xception(`ffpp_vox`)과 VideoMAE 등(`celebdf`)의 **accuracy 숫자는 직접 비교하지 마세요.**

---

## 2. GPU 디렉터리 구조

```text
~/forenShield-ai/data/
├── raw/                              # 원본·풀 (대량)
│   ├── faceforensics/                # FF++ 공식 다운로드
│   │   ├── original_sequences/       # real 원본 (youtube, actors)
│   │   └── manipulated_sequences/    # DeepFakeDetection, Deepfakes, …
│   ├── voxceleb/
│   │   ├── txt/                      # utterance 메타 (video_id)
│   │   └── tmp_full/                 # YouTube 전체 다운로드 임시
│   ├── celeb-df-v2/                  # Celeb-DF v2 전체 추출본
│   └── benchmark-downloads/          # zip/cache (celeb-df, dfdc 등)
│
├── train/                            # Fine-tune 전용 (테스트와 분리)
│   └── video/
│       └── voxceleb/real/            # real 100 clips
│
└── test/                             # 벤치마크 고정셋
    └── video/
        ├── ffpp/fake_over60s/        # fake 50 (≥60s)
        ├── voxceleb/real/            # real 50 (long)
        ├── celeb-df-v2/
        │   ├── fake/                 # celebdf_fake_001.mp4 …
        │   ├── real/                 # celebdf_real_001.mp4 …
        │   └── manifest.json
        └── dfdc/                     # (준비) fake/real + manifest
```

각 테스트 폴더에는 `manifest.json`(파일명, source, label, duration 등)이 있을 수 있습니다.

---

## 3. 데이터셋 상세

### 3-1. FaceForensics++ (FF++)

| 항목 | 내용 |
|------|------|
| **역할** | Fake 풀 + (선택) real original |
| **조작 종류** | Deepfakes, Face2Face, FaceSwap, NeuralTextures, DeepFakeDetection 등 |
| **압축** | c23 / c40 (벤치마크·학습: 주로 **c40**) |
| **접근** | [공식 신청](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform) → `download-FaceForensics.py` |

**벤치마크 fake 50 (`ffpp_vox`):**

| 항목 | 값 |
|------|-----|
| 서브셋 | `DeepFakeDetection` manipulated |
| 길이 필터 | 기본 **120–240초** (스크립트 `--min-sec` / `--max-sec`, alias `fake_over60s`는 ≥60s) |
| GPU 경로 | `data/test/video/ffpp/fake_over60s/` |
| 스크립트 | `scripts/download/data/download_ffpp_fake_by_duration.py` |

**Fine-tune fake 100:**

| 항목 | 값 |
|------|-----|
| 풀 | `data/raw/faceforensics/manipulated_sequences/DeepFakeDetection/c40/videos/` |
| 제외 | 벤치마크 50개의 `source_path` (manifest 기준) |

**Fine-tune / fallback real (FF++ original):**

```text
data/raw/faceforensics/original_sequences/youtube/c23/videos/
data/raw/faceforensics/original_sequences/youtube/c40/videos/
data/raw/faceforensics/original_sequences/actors/c40/videos/
```

---

### 3-2. VoxCeleb2 (Real)

| 항목 | 내용 |
|------|------|
| **역할** | **Real** 영상 (화자 인식용 대규모 코퍼스에서 샘플) |
| **특징** | utterance 클립은 ~1초 → **YouTube 원본에서 long segment** 추출 |
| **접근** | VoxCeleb2 메타 + `yt-dlp` |

**벤치마크 real 50 (`ffpp_vox`):**

| 항목 | 값 |
|------|-----|
| GPU 경로 | `data/test/video/voxceleb/real/` |
| 파일명 패턴 | `{video_id}_long.mp4` |
| 스크립트 | `scripts/download/data/download_voxceleb_long.py` |

**Fine-tune real 100:**

| 항목 | 값 |
|------|-----|
| GPU 경로 | `data/train/video/voxceleb/real/` |
| 제외 | 벤치마크 50개의 `video_id` (`--exclude-dir data/test/video/voxceleb/real`) |
| 준비 | `scripts/download/data/prepare_videomae_train_data.sh` |

---

### 3-3. Celeb-DF v2

| 항목 | 내용 |
|------|------|
| **역할** | **Fake + Real 모두** Celeb-DF v2에서 샘플 (in-domain 벤치마크) |
| **Fake** | `Celeb-synthesis` — 유명인 얼굴 deepfake |
| **Real** | `Celeb-real` + `YouTube-real` 풀에서 랜덤 50 |
| **접근** | [공식 신청](https://forms.gle/2jYBby6y1FBU3u6q9) 또는 Kaggle `reubensuju/celeb-df-v2` |

**벤치마크 50+50 (`celebdf`):**

| 항목 | 값 |
|------|-----|
| Fake 경로 | `data/test/video/celeb-df-v2/fake/` |
| Real 경로 | `data/test/video/celeb-df-v2/real/` |
| 파일명 | `celebdf_fake_001.mp4`, `celebdf_real_001.mp4` … |
| 원본 추출 | `data/raw/celeb-df-v2/Celeb-DF-v2/` |
| 스크립트 | `scripts/download/data/download_celebdf_v2.py`, `.sh` |

**Fine-tune 시 제외 (필수):**

```bash
--exclude-dirs data/test/video/celeb-df-v2/fake data/test/video/celeb-df-v2/real
```

---

### 3-4. DFDC / HuggingFace 통합 (준비·미완)

| 항목 | 내용 |
|------|-----|
| **역할** | VideoMAE DFDC 프로필 벤치마크 (`--profile dfdc`) |
| **HF 대안** | [belkhir-nacim/deepfake-videos](https://huggingface.co/datasets/belkhir-nacim/deepfake-videos) (약관 동의 후) |
| **GPU 경로** | `data/test/video/dfdc/{fake,real}/` |
| **스크립트** | `download_dfdc_subset.py`, `prepare_dfdc_infer_dirs.py`, `run_videomae_dfdc_benchmark.sh` |
| **상태** | GPU 다운로드·벤치마크 **미완** (Kaggle rules / zip 404 이슈) |

HF 소스 후보 (승인 없이 상대적으로 쉬움):

| HF `dataset_source` | Real | Fake | 비고 |
|---------------------|------|------|------|
| DFD | 363 | 3,068 | Google DFD, FF++ 계열 |
| HIDF | 4,361 | 4,361 | 균형 |
| SDFVD2.0 | 456 | 471 | 소형 균형 |
| UADFV | 49 | 49 | 파일럿 규모 |

---

## 4. 프로필별 비교

### 4-1. `ffpp_vox` — 교차 일반화

```
Fake: FaceForensics++ DeepFakeDetection (actor deepfake, c40, long)
Real: VoxCeleb2 YouTube interviews (다른 도메인)
```

- **장점:** real/fake가 다른 출처 → **OOD 일반화** 시험
- **단점:** 점수가 낮아도 “모델 불량” vs “도메인 불일치” 구분 어려움
- **사용:** Xception (`xception-benchmark-20260618-0411`)

### 4-2. `celebdf` — in-domain

```
Fake: Celeb-DF v2 Celeb-synthesis
Real: Celeb-DF v2 Celeb-real + YouTube-real (혼합 50)
```

- **장점:** 동일 벤치마크 내 공정 비교, Temporal/Optical/CNN(ConvNeXt) **동일 테스트셋**
- **단점:** FF++ 학습 모델(Xception)과 테스트 도메인 불일치
- **사용:** VideoMAE, TimeSformer, Video Swin, RAFT, GMFlow, ConvNeXt

### 4-3. `dfdc` — (예정)

```
Fake / Real: DFDC 또는 HF DFD subset
```

---

## 5. 모델 × 데이터셋 매트릭스

| 모델 | 분류 | Test 프로필 | Train (fine-tune) |
|------|------|-------------|-------------------|
| Xception | CNN | **ffpp_vox** | 없음 (DeepfakeBench ckpt) |
| EfficientNet-B4 | CNN | celebdf (기본값) | 없음 |
| ConvNeXt-S | CNN | **celebdf** | FF++100 + Vox100 |
| VideoMAE | Temporal | **celebdf** | FF++100 + Vox100 |
| TimeSformer | Temporal | **celebdf** | FF++100 + Vox100 |
| Video Swin | Temporal | **celebdf** | FF++100 + Vox100 |
| RAFT | Optical | **celebdf** | 없음 |
| GMFlow | Optical | **celebdf** | 없음 |
| PWC-Net | Optical | celebdf (스크립트) | 없음 |

---

## 6. S3 업로드 경로 (데이터셋·리포트)

**Evidence 버킷:** `s3://forenshield-evidence-877044078824/`

| 종류 | S3 prefix 예시 |
|------|----------------|
| Xception 리포트 | `deepfake/archive/legacy-benchmarks/video-xception-benchmark/reports/<RUN_ID>/` |
| Celeb-DF 리포트 (모델별) | `deepfake/archive/legacy-benchmarks/video-{model}-celebdf-benchmark/reports/<RUN_ID>/` |
| Optical flow | `deepfake/archive/legacy-benchmarks/video-optical-flow-benchmark/reports/<RUN_ID>/` |
| 벤치 입력 mp4 | `deepfake/datasets/bench/{celebdf\|ffpp_vox}/{fake,real}/` |
| 모델별 infer bundle | `deepfake/results/infer/{model}/{profile}/` |
| 업로드 mp4 | `.../datasets/fake/`, `.../datasets/real/` |
| manifest | `.../datasets/fake/manifest.json`, `.../real/manifest.json` |

각 RUN_ID 아래 `benchmark_report.json`에 `profile`, `fake_source`, `real_source` 메타가 포함됩니다.

---

## 7. 다운로드·준비 명령 (GPU)

```bash
cd ~/forenShield-ai
source .venv/bin/activate

# Celeb-DF v2 → 50+50 벤치마크
bash scripts/download/data/download_celebdf_v2.sh --archive /path/to/celebdf_v2.zip

# FF++ fake 50 (승인 후)
python3 scripts/download/data/download_ffpp_fake_by_duration.py \
  --download-script data/raw/faceforensics/download-FaceForensics.py \
  --out-dir data/test/video/ffpp/fake_over60s

# VoxCeleb real 50 (벤치마크)
python3 scripts/download/data/download_voxceleb_long.py \
  --out-dir data/test/video/voxceleb/real --target 50

# Fine-tune용 real 100 + VideoMAE 학습
bash scripts/download/data/prepare_videomae_train_data.sh

# DFDC subset (HF, 약관 동의 필요)
bash scripts/download/data/download_dfdc_subset.sh --source hf --balanced --target 50
```

---

## 8. manifest.json 스키마 (요약)

벤치마크 폴더의 `manifest.json` 항목 예:

```json
{
  "file": "celebdf_fake_001.mp4",
  "subdir": "fake",
  "dataset": "celeb-df-v2",
  "label": "fake",
  "source": "Celeb-synthesis/videos/xxx.mp4"
}
```

FF++ fake 예:

```json
{
  "file": "fake_ffpp_001_DeepFakeDetection_....mp4",
  "duration_sec": 185.2,
  "dataset_source": "FaceForensics++",
  "manipulation": "DeepFakeDetection"
}
```

VoxCeleb real 예:

```json
{
  "file": "dQw4w9WgXcQ_long.mp4",
  "video_id": "dQw4w9WgXcQ",
  "label": "real"
}
```

---

## 9. 초기 파일럿 (별도, 3×3와 다름)

승인 대기 전 **HuggingFace DFD** 로 빠르게 돌린 초기 3모델 파일럿:

| Fake / Real | HF `belkhir-nacim/deepfake-videos` → DFD subset 50+50 |
| 모델 | Xception, GenConViT, LipForensics |
| 스크립트 | (구) `16_download_dfd_video_sample.py`, `17_infer_video_all_models.py` |

현재 공식 벤치마크 파이프라인은 **§1–§7** 기준입니다.

---

## 10. TODO

- [ ] DFDC / HF DFD 프로필 50+50 확정 및 `dfdc` 벤치마크 실행
- [ ] CNN 3종 **동일 Celeb-DF 50+50**에서 재벤치마크 (Xception은 Celeb-DF infer 별도 run)
- [ ] Celeb-DF real 50의 Celeb-real vs YouTube-real 비율 manifest 집계
- [ ] FF++ `original` real 50 + FF++ fake 50 **in-domain** 프로필 추가 (논문 표준 비교용)
