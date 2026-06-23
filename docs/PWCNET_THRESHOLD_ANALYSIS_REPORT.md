# PWC-Net Optical Flow Threshold 분석 보고서

> **작성일:** 2026-06-22  
> **분석 환경:** welabs GPU (`sk4team@58.127.241.84`, `~/forenShield-ai`)  
> **분석 스크립트:** `scripts/eval/analyze_optical_flow_threshold.py`  
> **결과 JSON:** `results/infer/pwcnet_threshold_analysis.json` (GPU 로컬)  

---

## 1. 요약 (Executive Summary)

PWC-Net optical flow 벤치마크 결과(각 **fake 50 + real 50**, 총 100영상)에 대해 `fake_score` 기준 threshold sweep(0.00~1.00, step 0.01)을 수행했다.

| 프로필 | RUN_ID | fake/real 분리 방향 | T=0.5 성능 | 권장 threshold | 결론 |
|--------|--------|---------------------|------------|----------------|------|
| **celebdf** | `pwcnet-celebdf-20260622-041153` | fake > real (gap +0.053) | acc **49%** (사실상 무의미) | **0.19** (균형) 또는 **0.03** (recall 우선) | 약한 분리. threshold 조정 시 **최대 acc ~65%** 기대. 단독 배포는 비권장. |
| **ffpp_vox** | `pwcnet-ffpp_vox-20260622-041348` | fake < real (gap **-0.119**, **역전**) | acc 50% | 없음 (T=0.00 = 전부 fake) | **실패.** threshold 튜닝으로 해결 불가. |

**핵심 결론**

1. 기본 threshold **0.5**는 PWC-Net motion 휴리스틱에 맞지 않다. 대부분의 `fake_score`가 0.5 미만이라 fake를 거의 예측하지 않거나, ffpp_vox에서는 신호 자체가 반대다.
2. Celeb-DF에서는 **0.15~0.20** 구간이 accuracy·Youden 기준 최적에 가깝고, recall을 극대화하려면 **~0.03**까지 낮춰야 한다.
3. FF++ Vox 프로필에서는 PWC-Net motion anomaly 점수가 **딥페이크 탐지 지표로 쓰이기 어렵다.** CNN/Temporal 모델과 직접 accuracy 비교도 부적절하다.

---

## 2. 배경 및 분석 목적

### 2.1 PWC-Net 벤치마크 개요

PWC-Net은 연속 프레임 쌍에 대해 dense optical flow를 추정하고, flow 통계를 집계해 **motion anomaly** 점수를 산출한다. CNN·Temporal 모델과 달리 원래는 fake/real 이진 분류기가 아니며, ForenShield AI 파이프라인에서는 `optical_flow_common.py`의 **휴리스틱**으로 0~1 점수를 만든다.

| 항목 | 내용 |
|------|------|
| 모델 | PWC-Net (`PwcnetBackend`) |
| 가중치 | `models/test/video/optical-flow/pwcnet/` |
| 벤치마크 데이터 | S3 `cases/test/video-benchmark-datasets/{celebdf\|ffpp_vox}/` |
| 결과 S3 | `cases/test/video-benchmark-datasets/PWC-Net/{celebdf\|ffpp_vox}/` |
| 리포트 형식 | EfficientNet-B4 스타일 `benchmark_report.json` (`items[]`, `fake_score`, `frame_votes` 등) |

### 2.2 분석 목적

- fake 50 vs real 50 **코호트**에서 점수 분포 비교
- threshold 0.00~1.00 sweep으로 **accuracy, precision, recall, F1, Youden** 최적점 탐색
- 현재 기본값 **T=0.5**의 적절성 검증
- 프로필(celebdf / ffpp_vox)별 **권장 threshold** 및 모델 채택 여부 판단

### 2.3 점수 산출 방식 (`motion_anomaly_score` / `fake_score`)

`benchmark_report.json`의 `items[].fake_score`는 `motion_anomaly_score`와 동일하다.

1. **real 50개만**으로 코호트 baseline 계산 (median, std)
   - `temporal_jitter`, `spatial_inconsistency_mean`, `angle_dispersion_mean`, `flow_mag_mean`
2. 각 영상 aggregate에 대해 real baseline 대비 **양의 z-score**만 취함 (`max(0, z)`)
3. 4개 지표 z-score의 평균을 3으로 나누고 **0~1로 cap** → `motion_anomaly_score`
4. `pred_label`: score ≥ threshold → `fake`, 미만 → `real` (기본 threshold = **0.5**)

```text
motion_anomaly_score = min(1.0, mean(max(0, z_i)) / 3)
```

이 방식은 **같은 run 내 real 50개를 정상 기준**으로 삼는 상대 점수이므로, 프로필·데이터셋마다 분포가 달라지고 **절대 threshold는 프로필별 튜닝**이 필요하다.

---

## 3. 분석 방법

### 3.1 입력 데이터

| 프로필 | RUN_ID | 로컬 경로 (GPU) | S3 |
|--------|--------|-----------------|-----|
| celebdf | `pwcnet-celebdf-20260622-041153` | `results/infer/pwcnet-celebdf-20260622-041153/benchmark_report.json` | `.../PWC-Net/celebdf/benchmark_report.json` |
| ffpp_vox | `pwcnet-ffpp_vox-20260622-041348` | `results/infer/pwcnet-ffpp_vox-20260622-041348/benchmark_report.json` | `.../PWC-Net/ffpp_vox/benchmark_report.json` |

각 리포트: `count=100`, `ok` 항목 fake 50 + real 50.

### 3.2 실행 명령

Windows PowerShell에서 GPU로 스크립트 전송 후 일괄 실행:

```powershell
Get-Content c:\sw_study\finalpjt\ai\scripts\eval\gpu_install_and_run_threshold.sh -Raw | ssh sk4team@58.127.241.84 "bash -s"
```

GPU에서 동등 명령:

```bash
cd ~/forenShield-ai
python3 scripts/eval/analyze_optical_flow_threshold.py \
  --run-id pwcnet-celebdf-20260622-041153 \
  --run-id pwcnet-ffpp_vox-20260622-041348 \
  -o results/infer/pwcnet_threshold_analysis.json
```

### 3.3 평가 지표 정의

| 지표 | 설명 |
|------|------|
| **accuracy** | (TP + TN) / 100 |
| **precision** | TP / (TP + FP) — fake로 예측한 것 중 실제 fake 비율 |
| **recall** | TP / (TP + FN) — 실제 fake 중 잡아낸 비율 |
| **F1** | precision과 recall의 조화평균 |
| **Youden's J** | TPR − FPR (민감도·특이도 균형) |
| **mean_gap** | mean(fake_score \| fake) − mean(fake_score \| real) |
| **midpoint** | (mean_fake + mean_real) / 2 — 단순 제안 threshold |

**overlap 판정:** fake 분포의 p25 ≤ real 분포의 p75이면 두 집단이 크게 겹친다 → 단일 threshold로 완전 분리 어려움.

---

## 4. 프로필별 상세 결과

### 4.1 Celeb-DF (`celebdf`)

**RUN_ID:** `pwcnet-celebdf-20260622-041153`  
**데이터:** Celeb-DF v2 fake 50 + real 50 (in-domain)

#### 4.1.1 그룹별 점수 요약

| 지표 | fake mean | real mean | gap (fake − real) |
|------|-----------|-----------|-------------------|
| **fake_score** | **0.1800** | **0.1272** | **+0.0528** |
| temporal_jitter | 0.0000 | 0.0000 | 0.0000 |
| spatial_inconsistency_mean | 0.0667 | 0.0667 | 0.0000 |
| angle_dispersion_mean | 0.0027 | 0.0027 | 0.0000 |
| frame_vote_ratio | 0.0000 | 0.0000 | 0.0000 |

- **fake_score만** fake가 real보다 높다 (기대 방향).
- 보조 flow 지표 4종은 fake/real 평균이 **동일**하여 단독 판별력 없음.
- `frame_vote_ratio`가 0에 가깝다 → 프레임 쌍 단위 mini-score가 threshold 0.5를 넘는 경우가 거의 없음.

#### 4.1.2 분포 겹침

```
fake p25 <= real p75: distributions overlap — single threshold may be weak
```

fake와 real 점수 분포가 상당 부분 겹친다. 완벽한 이진 분리는 기대하기 어렵다.

#### 4.1.3 Threshold별 성능

| 구분 | Threshold | Accuracy | Precision | Recall | F1 |
|------|-----------|----------|-----------|--------|-----|
| **현재 (기본)** | **0.50** | **0.49** | 0.00 | 0.00 | — |
| midpoint 제안 | 0.154 | (sweep 최근접) | — | — | — |
| **best F1** | **0.03** | 0.61 | 0.5679 | **0.92** | **0.7023** |
| **best accuracy** | **0.19** | **0.65** | **0.6923** | 0.54 | 0.6067 |
| **best Youden** | **0.19** | **0.65** | **0.6923** | 0.54 | 0.6067 |

**T=0.5 해석:** 점수 대부분이 0.5 미만 → 거의 모든 영상을 `real`로 예측 → recall 0%, accuracy ~49% (real 쪽으로 치우친 무작위 수준).

**T=0.03 해석:** fake의 92%를 잡지만 precision ~57% → false positive가 많다. “놓치지 않기” 우선 시나리오.

**T=0.19 해석:** accuracy·Youden 최적. precision ~69%, recall 54% → **균형 잡힌 운영점**으로 적합.

#### 4.1.4 Celeb-DF 결론

| 항목 | 판단 |
|------|------|
| 신호 방향 | ✅ fake > real (유효) |
| T=0.5 | ❌ 부적절 |
| 권장 T (균형) | **0.19** |
| 권장 T (recall) | **0.03** |
| 기대 상한 | acc ~65%, F1 ~0.61~0.70 |
| 배포 권고 | 단독 primary detector **비권장**. 보조·앙상블·Celeb-DF 유사 도메인 참고용 |

---

### 4.2 FF++ Vox (`ffpp_vox`)

**RUN_ID:** `pwcnet-ffpp_vox-20260622-041348`  
**데이터:** FF++ DeepFakeDetection fake 50 + VoxCeleb real 50 (교차 데이터셋)

#### 4.2.1 그룹별 점수 요약

| 지표 | fake mean | real mean | gap (fake − real) |
|------|-----------|-----------|-------------------|
| **fake_score** | **0.0187** | **0.1375** | **-0.1188** |
| temporal_jitter | 0.0000 | 0.0000 | 0.0000 |
| spatial_inconsistency_mean | 0.0267 | 0.0267 | 0.0000 |
| angle_dispersion_mean | 0.0002 | 0.0002 | 0.0000 |
| frame_vote_ratio | 0.0000 | 0.0087 | -0.0087 |

- **fake_score가 fake에서 오히려 낮다** → 휴리스틱 가정(딥페이크 = motion 이상 증가)과 **반대**.
- real 영상이 상대적으로 더 “이상”하게 측정됨 (교차 도메인·촬영 조건·얼굴 움직임 특성 영향 가능).

#### 4.2.2 분포 겹침

```
fake p25 <= real p75: distributions overlap — single threshold may be weak
```

겹침 메시지는 동일하나, 근본 원인은 **역전된 평균 gap**이다.

#### 4.2.3 Threshold별 성능

| 구분 | Threshold | Accuracy | Precision | Recall | F1 |
|------|-----------|----------|-----------|--------|-----|
| **현재 (기본)** | **0.50** | 0.50 | — | 0.00 | — |
| midpoint 제안 | 0.078 | — | — | — | — |
| best F1 / acc / Youden | **0.00** | 0.50 | 0.50 | 1.00 | 0.6667 |

**T=0.00의 의미:** score ≥ 0인 모든 샘플을 fake로 분류 → **100개 전부 fake 예측**과 동일.  
accuracy 50% = fake 50개 맞음 + real 50개 틀림. F1 0.667은 “항상 fake” baseline에서 나오는 **degenerate optimum**이다.

유의미한 operating point가 **존재하지 않는다.**

#### 4.2.4 FF++ Vox 결론

| 항목 | 판단 |
|------|------|
| 신호 방향 | ❌ fake < real (역전) |
| T=0.5 | ❌ 부적절 (recall 0) |
| threshold 튜닝 | ❌ 해결 불가 |
| 배포 권고 | PWC-Net motion 휴리스틱 **ffpp_vox 프로필에서 사용 금지** |

---

## 5. 보조 지표 단독 sweep 결과

두 프로필 공통으로 아래 보조 지표에 대해 단독 best-F1 threshold가 **T=0.00**, F1=0.6667, acc=0.5000으로 동일하게 나왔다.

| 보조 지표 | celebdf best T | ffpp_vox best T | 비고 |
|-----------|----------------|-----------------|------|
| temporal_jitter | 0.00 | 0.00 | fake/real gap 0 |
| spatial_inconsistency_mean | 0.00 | 0.00 | fake/real gap 0 |
| angle_dispersion_mean | 0.00 | 0.00 | fake/real gap 0 |
| flow_mean | 0.00 | 0.00 | 단독 분리 없음 |
| frame_vote_ratio | 0.00 | 0.00 | 거의 0, 분리 없음 |

이는 해당 지표 값이 대부분 0 근처에 몰려 있거나 fake/real 평균이 같아, **“전부 fake” 규칙**이 sweep 상 최적으로 잡힌 것이다. 실질적 판별 지표로 쓸 수 없다.

**유일하게 의미 있는 primary score:** `fake_score` (= `motion_anomaly_score`).

---

## 6. CNN/Temporal 모델과의 비교 시 주의사항

| 비교 항목 | CNN / Temporal | PWC-Net (본 분석) |
|-----------|----------------|-------------------|
| 점수 의미 | 학습된 분류 확률 | real 코호트 대비 휴리스틱 z-score |
| 기본 threshold | 0.5 (학습 분포 기준) | 0.5는 **임의값**, 본 분석에서 부적절 |
| 프로필 의존성 | fine-tune 데이터에 따름 | **run 내 real 50에 강하게 의존** |
| ffpp_vox | Xception 등 별도 벤치마크 | motion 점수 **역전**, 비교 무의미 |
| accuracy 해석 | 표준 분류 accuracy | `heuristic_accuracy` (휴리스틱 threshold 정확도) |

동일 accuracy 숫자라도 **모델 역할과 신뢰도가 다르다.** 3×3 벤치마크 문서(`VIDEO_DEEPFAKE_MODEL_BENCHMARK_3x3.md`)의 Optical Flow 행은 “분류 점수 없음”으로 기술되어 있었으나, 현재 파이프라인은 휴리스틱 `fake_score`를 산출한다. **수치를 CNN과 직접 순위 비교하지 말 것.**

---

## 7. 권장 사항

### 7.1 Threshold 운영

| 프로필 | 용도 | 권장 threshold | 기대 성능 (벤치마크 기준) |
|--------|------|----------------|---------------------------|
| celebdf | 균형 (precision/recall) | **0.19** | acc 65%, prec 69%, rec 54%, F1 0.61 |
| celebdf | recall 우선 (fake 놓침 최소화) | **0.03** | acc 61%, prec 57%, rec 92%, F1 0.70 |
| ffpp_vox | — | **적용하지 않음** | — |

프로덕션 기본값 **0.5는 변경 필요.** 최소한 celebdf 전용으로 **0.19** 검토.

### 7.2 리포트 재생성 (threshold 반영 시)

Celeb-DF run에 threshold 0.19를 적용해 리포트를 다시 만들려면 GPU에서:

```bash
cd ~/forenShield-ai
source .venv/bin/activate

# regenerate_optical_flow_reports.py가 threshold 인자를 지원하는지 확인 후 실행.
# 지원 시 예시:
python scripts/infer/regenerate_optical_flow_reports.py pwcnet-celebdf-20260622-041153 --threshold 0.19

export S3_EVIDENCE_BUCKET=s3://forenshield-evidence-877044078824
unset AWS_PROFILE
PROFILE=celebdf bash scripts/upload/s3_upload_pwcnet_video_benchmark.sh pwcnet-celebdf-20260622-041153
```

### 7.3 모델 채택 의견

1. **PWC-Net 단독 딥페이크 판정기:** 채택 **비권장** (특히 ffpp_vox).
2. **Celeb-DF 유사 도메인 보조 신호:** threshold 튜닝 후 **제한적 참고** 가능.
3. **개선 방향 (향후 연구):**
   - flow 특성 + supervised head (real/fake 라벨 학습)
   - 프로필별 코호트 baseline 대신 고정 calibration set
   - RAFT/GMFlow와 동일 run에서 상관 분석
   - `temporal_jitter` 등 0으로 수렴하는 지표의 전처리·정규화 점검

---

## 8. 실행 로그 (원문 발췌)

분석 실행 시 터미널 출력 (2026-06-22):

```text
profile: celebdf  run_id: pwcnet-celebdf-20260622-041153
score: fake_score  current threshold: 0.5
counts: fake=50 real=50
fake_score             0.1800       0.1272       0.0528
suggested midpoint threshold: 0.1536
at current T=0.5: acc=0.4900 prec=0.0000 rec=0.0000 f1=-
best by f1      : T=0.03 acc=0.6100 prec=0.5679 rec=0.9200 f1=0.7023
best by accuracy: T=0.19 acc=0.6500 prec=0.6923 rec=0.5400 f1=0.6067

profile: ffpp_vox  run_id: pwcnet-ffpp_vox-20260622-041348
score: fake_score  current threshold: 0.5
counts: fake=50 real=50
fake_score             0.0187       0.1375      -0.1188
suggested midpoint threshold: 0.0781
at current T=0.5: acc=0.5000 prec=- rec=0.0000 f1=-
best by f1      : T=0.00 acc=0.5000 prec=0.5000 rec=1.0000 f1=0.6667
```

실행 말미 `bash: line 31: $'\r': command not found` 는 Windows CRLF로 인한 경고이며, **JSON 저장은 정상 완료**되었다 (`Wrote results/infer/pwcnet_threshold_analysis.json`).

---

## 9. 관련 파일

| 경로 | 설명 |
|------|------|
| `ai/docs/PWCNET_THRESHOLD_ANALYSIS_REPORT.md` | 본 보고서 |
| `ai/scripts/eval/analyze_optical_flow_threshold.py` | threshold sweep 분석기 |
| `ai/scripts/eval/run_threshold_analysis_gpu.sh` | GPU 일괄 실행 wrapper |
| `ai/scripts/eval/gpu_install_and_run_threshold.sh` | Windows SSH pipe용 설치+실행 |
| `ai/scripts/infer/optical_flow_common.py` | 점수 산출·리포트 생성 |
| GPU `results/infer/pwcnet_threshold_analysis.json` | 전체 sweep JSON (profiles 배열) |
| S3 `.../PWC-Net/celebdf/benchmark_report.json` | Celeb-DF 100항목 리포트 |
| S3 `.../PWC-Net/ffpp_vox/benchmark_report.json` | FF++ Vox 100항목 리포트 |

---

## 10. 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-06-22 | 초안 작성 — GPU threshold sweep 결과 반영 (celebdf, ffpp_vox) |
