# AI 분석 응답 — 모듈별 타임라인 확장 (Late Fusion)

> **브랜치:** `feature/ai-deepfake-late-fusion-rabbitmq`  
> **목적:** 상세페이지용 프레임/클립/쌍 단위 점수를 AI → BE 계약에 포함

---

## 1. 기존 필드 (하위 호환)

| 필드 | 출처 | 설명 |
|------|------|------|
| `frameRisks[]` | Xception | 프레임별 fake 확률 (`frameIndex`, `timestampSec`, `riskScore`) |
| `suspiciousSegments[]` | Xception | 고위험 프레임 구간 (기존 BE/FE 그대로 사용) |

---

## 2. 신규 필드 (`results[].*`)

| 필드 | 출처 | 설명 |
|------|------|------|
| `clipRisks[]` | TimeSformer | 클립 구간별 fake 확률 |
| `pairRisks[]` | GMFlow | 연속 프레임 쌍별 motion anomaly |
| `temporalSuspiciousSegments[]` | TimeSformer | 클립 점수 기반 의심 구간 |
| `opticalSuspiciousSegments[]` | GMFlow | optical motion 기반 의심 구간 |
| `moduleTimelines[]` | 3모듈 통합 | 상세 UI용 모듈별 타임라인 묶음 |

### 2.1 `clipRisks[]`

```json
{
  "clipIndex": 0,
  "startFrameIndex": 0,
  "endFrameIndex": 81,
  "startTimeSec": 0.0,
  "endTimeSec": 2.7,
  "riskScore": 0.0001
}
```

### 2.2 `pairRisks[]`

```json
{
  "pairIndex": 0,
  "frameIndexA": 0,
  "frameIndexB": 1,
  "timestampSec": 0.033,
  "riskScore": 0.62,
  "motionMagnitude": 0.434
}
```

- `riskScore`: 영상 내 pair motion 상대값 (0~1, UI 히트맵용)
- `motionMagnitude`: GMFlow raw flow magnitude mean
- 영상 전체 GMFlow fake 판정은 `modelScores` / `moduleTimelines[].videoScore` 사용 (learned head)

### 2.3 `moduleTimelines[]`

```json
{
  "module": "temporal",
  "modelName": "timesformer",
  "modelVersion": "timesformer/v1.1.0-celeb1k",
  "videoScore": 0.0,
  "threshold": 0.5,
  "detected": false,
  "frameRisks": [],
  "clipRisks": [ ... ],
  "pairRisks": [],
  "suspiciousSegments": [ ... ]
}
```

`module` 값: `cnn` | `temporal` | `optical`

---

## 3. BE 연동 메모

- Jackson `@JsonIgnoreProperties(ignoreUnknown = true)` — 신규 필드는 **무시되지 않고** 역직렬화 가능
- BE 저장/상세 API 노출은 별도 PR 필요 (`VideoAnalysisDetailsBuilder` 확장)
- AI 측에서는 RabbitMQ / `POST /ai/analyze` 응답에 위 필드를 포함

---

## 4. 로컬 검증

```powershell
cd ai
..\.venv\Scripts\python.exe -m unittest tests.test_module_timelines -v
..\.venv\Scripts\python.exe scripts\eval\export_youtube_fresh_timeline_response.py
```

출력: `results/eval/youtube_fresh_timeline_response_sample.json`
