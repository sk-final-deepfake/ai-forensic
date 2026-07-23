# GPU 데이터셋 종류별 개수 (실측)

> **집계 일시:** 2026-06-22 (UTC)  
> **호스트:** `sk4team@58.127.241.84`  
> **작업 경로:** `~/forenShield-ai`  
> **관련 문서:** [VIDEO_BENCHMARK_DATASETS.md](./VIDEO_BENCHMARK_DATASETS.md)

GPU에서 `find`로 직접 집계한 **영상 mp4 개수**입니다. 벤치마크 프로필·학습 풀·RAW 코퍼스를 데이터셋 종류별로 구분합니다.

---

## 1. 합계

| 구분 | mp4 개수 | 비고 |
|------|----------|------|
| **TEST** (`data/test/video/`) | **200** | 벤치마크 고정셋 |
| **TRAIN** (`data/train/video/`) | **500** | fine-tune용 (세부 폴더 breakdown은 §4 참고) |
| **RAW** (`data/raw/`) | **9,561** | 원본·풀·다운로드 캐시 포함 |

---

## 2. 데이터셋 종류별

### 2-1. FaceForensics++ (FF++)

| 용도 | GPU 경로 | mp4 |
|------|----------|-----|
| TEST fake | `data/test/video/ffpp/fake_over60s/` | **50** |
| TEST fake (빈 폴더) | `data/test/video/ffpp/fake/` | 0 |
| TRAIN 풀 (DeepFakeDetection c40) | `data/raw/faceforensics/manipulated_sequences/DeepFakeDetection/c40/videos/` | **2,866** |
| RAW 전체 | `data/raw/faceforensics/` | **2,866** |

- 벤치마크 프로필: **`ffpp_vox`** (fake 50)
- manifest: `data/test/video/ffpp/fake_over60s/manifest.json`

### 2-2. VoxCeleb2

| 용도 | GPU 경로 | mp4 |
|------|----------|-----|
| TEST real | `data/test/video/voxceleb/real/` | **50** |
| TRAIN real | `data/train/video/voxceleb/real/` | **100** |
| RAW | `data/raw/voxceleb/` | **166** |

- 벤치마크 프로필: **`ffpp_vox`** (real 50)
- fine-tune 설계: real **100** (벤치마크 50개 video_id 제외)

### 2-3. Celeb-DF v2

| 용도 | GPU 경로 | mp4 |
|------|----------|-----|
| TEST fake | `data/test/video/celeb-df-v2/fake/` | **50** |
| TEST real | `data/test/video/celeb-df-v2/real/` | **50** |
| RAW | `data/raw/celeb-df-v2/` | **6,529** |

- 벤치마크 프로필: **`celebdf`** (fake 50 + real 50 = 100)
- manifest: `data/test/video/celeb-df-v2/manifest.json`

### 2-4. HF deepfake (초기 파일럿용, 미사용)

| 용도 | GPU 경로 | mp4 |
|------|----------|-----|
| TEST fake | `data/test/video/hf-deepfake/fake/` | 0 |
| TEST real | `data/test/video/hf-deepfake/real/` | 0 |

- 폴더는 존재하나 mp4 없음 (공식 벤치마크 파이프라인과 별도)

### 2-5. DFDC (준비 중)

| 용도 | GPU 경로 | mp4 |
|------|----------|-----|
| TEST fake | `data/test/video/dfdc/fake/` | — (폴더 없음) |
| TEST real | `data/test/video/dfdc/real/` | — (폴더 없음) |

- 벤치마크 프로필 **`dfdc`**: 아직 미구성

---

## 3. 벤치마크 프로필 × 실측 개수

| 프로필 | Fake 출처 | Real 출처 | Fake | Real | 합계 | 상태 |
|--------|-----------|-----------|------|------|------|------|
| **`ffpp_vox`** | FF++ `fake_over60s` | VoxCeleb `real` | 50 | 50 | **100** | ✅ |
| **`celebdf`** | Celeb-DF v2 `fake` | Celeb-DF v2 `real` | 50 | 50 | **100** | ✅ |
| **`dfdc`** | DFDC / HF subset | 동일 소스 real | 0 | 0 | 0 | ⏳ 미준비 |

**TEST 전체 mp4:** 200 (= ffpp_vox 100 + celebdf 100)

---

## 4. TRAIN 500 vs 설계 200

문서상 fine-tune 설계는 **FF++ fake 100 + Vox real 100** (합 200)이며, fake는 `data/raw/faceforensics/.../c40/videos/` 풀에서 샘플링합니다.

실측:

| 경로 | mp4 (maxdepth 1) |
|------|------------------|
| `data/train/video/voxceleb/real/` | 100 |
| `data/train/video/` **전체** (재귀) | **500** |

TRAIN 500과 voxceleb 100의 차이(약 400)는 다른 `data/train/video/` 하위 폴더에 있을 수 있습니다. 아래 명령으로 breakdown 확인:

```bash
cd ~/forenShield-ai
find data/train/video -mindepth 1 -maxdepth 2 -type d | sort
for d in $(find data/train/video -mindepth 1 -maxdepth 2 -type d); do
  echo "$d: $(find "$d" -name '*.mp4' | wc -l) mp4"
done
```

---

## 5. RAW 상위 디렉터리

```text
data/raw/
├── benchmark-downloads/   # zip/cache (celeb-df, dfdc 등)
├── celeb-df-v2/           # 6,529 mp4
├── faceforensics/         # 2,866 mp4
└── voxceleb/              # 166 mp4
```

---

## 6. 재집계 명령 (복사용)

```bash
cd ~/forenShield-ai

count_mp4() { find "$1" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l; }

echo "=== FaceForensics++ ==="
echo "  TEST fake_over60s: $(count_mp4 data/test/video/ffpp/fake_over60s)"
echo "  RAW DeepFakeDetection c40: $(count_mp4 data/raw/faceforensics/manipulated_sequences/DeepFakeDetection/c40/videos)"
echo "  RAW faceforensics total: $(find data/raw/faceforensics -name '*.mp4' 2>/dev/null | wc -l)"

echo "=== VoxCeleb2 ==="
echo "  TEST real: $(count_mp4 data/test/video/voxceleb/real)"
echo "  TRAIN real: $(count_mp4 data/train/video/voxceleb/real)"
echo "  RAW total: $(find data/raw/voxceleb -name '*.mp4' 2>/dev/null | wc -l)"

echo "=== Celeb-DF v2 ==="
echo "  TEST fake: $(count_mp4 data/test/video/celeb-df-v2/fake)"
echo "  TEST real: $(count_mp4 data/test/video/celeb-df-v2/real)"
echo "  RAW total: $(find data/raw/celeb-df-v2 -name '*.mp4' 2>/dev/null | wc -l)"

echo "=== HF deepfake / DFDC ==="
echo "  hf-deepfake fake: $(count_mp4 data/test/video/hf-deepfake/fake)"
echo "  hf-deepfake real: $(count_mp4 data/test/video/hf-deepfake/real)"
echo "  dfdc fake: $(count_mp4 data/test/video/dfdc/fake 2>/dev/null || echo 0)"
echo "  dfdc real: $(count_mp4 data/test/video/dfdc/real 2>/dev/null || echo 0)"

echo "=== 합계 ==="
echo "  TEST:  $(find data/test/video -name '*.mp4' | wc -l)"
echo "  TRAIN: $(find data/train/video -name '*.mp4' 2>/dev/null | wc -l)"
echo "  RAW:   $(find data/raw -name '*.mp4' 2>/dev/null | wc -l)"
```

---

## 7. 음성 데이터셋 (미집계)

영상 집계와 별도로 `samples/datasets/` 아래 wav 세트가 있습니다. 종류별 개수:

```bash
cd ~/forenShield-ai
for d in samples/datasets/*/; do
  [ -d "$d" ] && echo "$(basename "$d"): $(find "$d" -maxdepth 1 -name '*.wav' | wc -l) wav"
done
```

실행 후 이 문서 §8에 결과를 추가할 수 있습니다.
