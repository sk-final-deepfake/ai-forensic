# GMFlow 딥페이크 점수 (motion heuristic) 가이드

> **작성 기준일:** 2026-06-22  
> **대상:** `video-benchmark-datasets/gmflow/{celebdf|ffpp_vox}/<RUN_ID>/`  
> **GPU 파이프라인:** `scripts/infer/optical_flow_infer_model.py` → `optical_flow_common.py`

GMFlow는 CNN처럼 학습된 fake/real 분류기가 **아닙니다**.  
infer 1단계는 **flow 통계만** 저장하고, **딥페이크 점수는 후처리**로 붙입니다.

| 단계 | 스크립트 | 출력 |
|------|----------|------|
| 1. infer | `optical_flow_infer_model.py` | `gmflow/json/*.json` (`aggregate`, `pair_stats`) |
| 2. score | **`scripts/eval/gmflow_motion_score.py`** | `motion_anomaly_score`, `fake_score`, `pred_label` |

---

## 0. ffpp_vox 빠른 실행 (GPU)

infer가 끝난 RUN (`optical-flow-ffpp-vox-20260622-0544` 등)에 대해:

```bash
cd ~/forenShield-ai
source .venv/bin/activate

# 스크립트 복사 (repo에서 GPU로 아직 없으면)
# scp 또는 git pull: scripts/eval/gmflow_motion_score.py

export RUN_ID=optical-flow-ffpp-vox-20260622-0544
export S3_RUN_ID=gmflow-ffpp-vox-benchmark-20260622-0544

python3 scripts/eval/gmflow_motion_score.py \
  --root . \
  --run-id $RUN_ID \
  --profile ffpp_vox \
  --s3-run-id $S3_RUN_ID

# 확인
python3 -c "
import json, os
from pathlib import Path
p = next(Path(f'results/infer/{os.environ[\"RUN_ID\"]}/gmflow/json').glob('*.json'))
d = json.loads(p.read_text())
print('fake_score', d.get('fake_score'), 'pred', d.get('pred_label'))
print('pair range', d.get('score_breakdown',{}).get('flow_mag_pair_range'))
"

# S3 업로드
export S3_PREFIX=s3://forenshield-evidence-877044078824/deepfake/results/infer/gmflow/ffpp_vox/$S3_RUN_ID
aws s3 sync results/infer/$RUN_ID/gmflow/json/ $S3_PREFIX/json/
aws s3 cp results/infer/$RUN_ID/datasets/infer_summary.json $S3_PREFIX/infer_summary.json
aws s3 cp results/eval/$RUN_ID/metrics.json $S3_PREFIX/metrics.json
```

---

## 1. 한눈에 보기

| 구분 | CNN (EfficientNet) | GMFlow (Optical) |
|------|-------------------|------------------|
| 점수 | `fake_score` (0~1, 학습된 확률) | `motion_anomaly_score` (휴리스틱, 0~1 근처) |
| 판정 | `pred_label` | `pred_label` (`motion_anomaly_score >= threshold`) |
| 근거 | 프레임/얼굴 texture | **프레임 간 flow** + 시간·공간 불일치 |
| threshold | 보통 `0.5` | `score_breakdown.threshold` 또는 `DEFAULT_ANOMALY_THRESHOLD` (`0.5`) |

**운영 매핑 (권장):**

- API `deepfakeScore` ← CNN `fake_score` **또는** optical `motion_anomaly_score` (모델별 분리 권장)
- API `frameAnomaly` / 보조 신호 ← optical flow 전용

---

## 2. JSON 어디에 점수가 있나

### 2-1. 영상별 상세 — `json/<video>.json`

**infer 직후:** `aggregate`, `pair_stats`만 (~3–4KB)  
**`gmflow_motion_score.py` 실행 후:** 아래 점수 필드 추가

```json
{
  "file": "celebdf_fake_001.mp4",
  "ground_truth_label": "fake",
  "model": "gmflow",
  "status": "ok",
  "flow_mean": 0.032,
  "flow_max": 0.083,
  "flow_std": 0.005,
  "motion_anomaly_score": 0.062791,
  "pred_label": "real",
  "score_breakdown": {
    "threshold": 0.5,
    "frames_sampled": 32,
    "frame_pairs_used": 31,
    "aggregate": {
      "flow_mag_mean": 0.032,
      "flow_mag_max": 0.083,
      "flow_mag_std": 0.005,
      "spatial_inconsistency_mean": 0.150,
      "motion_energy_mean": 0.00105,
      "angle_dispersion_mean": 0.042,
      "temporal_jitter": 0.00463
    },
    "score_stats": {
      "flow_mag_mean": { "min": 0.0319, "max": 0.0328, "mean": 0.032, "std": 0.00015 },
      "flow_mag_max":  { "min": 0.066, "max": 0.083, "mean": 0.070, "std": 0.003 },
      "temporal_jitter": { "min": ..., "max": ..., "mean": ..., "std": ... }
    }
  },
  "interpretation": {
    "method": "optical_flow_motion_heuristic",
    "summary": "motion_anomaly_score=0.0628 (threshold=0.5); pred_label=real ...",
    "signals": [
      { "name": "temporal_jitter", "value": 0.0046, "real_cohort_baseline": 0.0064, "delta": -0.0018 },
      { "name": "spatial_inconsistency", "value": 0.150, "real_cohort_baseline": 0.163, "delta": -0.014 },
      { "name": "angle_dispersion", "value": 0.042, "real_cohort_baseline": 0.056, "delta": -0.014 },
      { "name": "flow_magnitude", "value": 0.032, "real_cohort_baseline": 0.031, "delta": 0.0015 }
    ]
  }
}
```

### 2-2. RUN 요약 — `infer_summary.json` (수동 생성 시 점수 누락 가능)

`predictions.json`만으로 만든 `infer_summary.json`에는 `flow_mean` 등만 있고  
**`motion_anomaly_score` / `fake_score` / `pred_label`이 없을 수 있습니다.**

→ **`gmflow_motion_score.py`가 `infer_summary.json` + `metrics.json`을 함께 생성** (아래 §5)

---

## 3. 프레임 최대·최소 차이 — 무슨 뜻인가

영상에서 **32프레임**을 샘플링하고, 인접 프레임 **31쌍**마다 flow를 계산합니다.

### 3-1. `score_stats.*.min` / `max` (프레임 쌍별 통계의 범위)

| 필드 | 의미 |
|------|------|
| `score_stats.flow_mag_mean.min` | 31쌍 중 **가장 작은** 평균 flow 크기 |
| `score_stats.flow_mag_mean.max` | 31쌍 중 **가장 큰** 평균 flow 크기 |
| `max - min` | 프레임 구간마다 움직임 크기가 **얼마나 들쭉날쭉한지** (시간적 변동) |

예시 (celebdf_fake_001):

```text
flow_mag_mean: min=0.0319, max=0.0328  → 범위 ≈ 0.0009 (매우 안정)
flow_mag_max:  min=0.066,  max=0.083   → 범위 ≈ 0.017  (순간 피크 변동)
```

**꼭 `flow_mag_max`만 쓸 필요는 없습니다.**  
파이프라인은 아래 신호를 **함께** 씁니다.

| 신호 | aggregate 필드 | 직관 |
|------|----------------|------|
| 시간 흔들림 | `temporal_jitter` | 프레임 쌍별 flow 크기 변동 (jitter) |
| 공간 불일치 | `spatial_inconsistency_mean` | 한 프레임 안에서 flow가 고르지 않음 |
| 방향 분산 | `angle_dispersion_mean` | flow 방향이 산만함 |
| 평균 크기 | `flow_mag_mean` | 전체 움직임 강도 |

`score_stats`의 min/max는 위 신호를 **프레임 쌍 단위로 훑은 뒤** 요약한 것입니다.

### 3-2. RAFT vs GMFlow 스케일

| 모델 | `flow_mean` 스케일 예 |
|------|----------------------|
| GMFlow (celebdf) | 대략 `0.03 ~ 0.10` |
| RAFT (celebdf, 패딩 후) | `10 ~ 1500+` (모델·입력 크기에 따라 큼) |

**프로필·모델마다 threshold/baseline을 따로 잡아야 합니다.**  
GMFlow celebdf baseline을 RAFT에 그대로 쓰면 안 됩니다.

---

## 4. `motion_anomaly_score` 계산 (`gmflow_motion_score.py`)

`scripts/eval/gmflow_motion_score.py`가 다음 순서로 동작합니다.

```text
1) pair_stats → 프레임 쌍별 flow_mag_mean, spatial_inconsistency, angle_dispersion
2) score_stats → flow_mag_mean min/max/mean/std (프레임 쌍 간 최소·최대 차이)
3) flow_mag_pair_range = max - min
4) 동일 RUN의 real 영상 median → real_cohort_baseline
5) 4신호 가중 편차 합산 → motion_anomaly_score (0~1 clamp)
6) motion_anomaly_score >= 0.5 → pred_label = "fake", else "real"
```

신호 가중치: `temporal_jitter` 0.30, `spatial_inconsistency` 0.25, `angle_dispersion` 0.25, `flow_magnitude` 0.20

**중요:**

- `ground_truth_label`은 데이터셋 정답이지, 모델 출력이 아님
- `pred_label`이 틀릴 수 있음 (예: fake인데 score 0.06 → pred real)
- CNN `fake_score`와 **숫자·정확도를 직접 비교하지 말 것**

### 4-1. EfficientNet 형식으로 맞추기

| Optical 필드 | CNN 호환 필드 (후처리) |
|--------------|------------------------|
| `motion_anomaly_score` | `fake_score` (동일 값 복사) |
| `pred_label` | `pred_label` |
| `score_breakdown.threshold` | `threshold` |
| — | `prob_fake` = `fake_score`, `prob_real` = `1 - fake_score` (선택) |

```json
"fake_score": 0.062791,
"pred_label": "real",
"threshold": 0.5,
"score_source": "gmflow_motion_heuristic"
```

---

## 5. 출력 파일

`gmflow_motion_score.py` 실행 후:

| 파일 | 내용 |
|------|------|
| `gmflow/json/*.json` | 점수 필드 in-place 갱신 |
| `datasets/infer_summary.json` | items 100개, `fake_score`, `pred_label` |
| `results/eval/<RUN_ID>/metrics.json` | `heuristic_accuracy` |

### 5-1. (레거시) 수동 Python merge — 스크립트 대신 쓸 때만

```bash
cd ~/forenShield-ai
source .venv/bin/activate

export LOCAL_RUN_ID=optical-flow-celebdf-20260622-0142   # 예: gmflow celebdf 로컬 RUN
export S3_RUN_ID=gmflow-celebdf-benchmark-20260622-0142
export PROFILE=celebdf
export MODEL=gmflow

python3 - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path

local_run_id = os.environ["LOCAL_RUN_ID"]
s3_run_id = os.environ["S3_RUN_ID"]
profile = os.environ["PROFILE"]
model = os.environ["MODEL"]
base = Path(f"results/infer/{local_run_id}")
json_dir = base / "json"

items = []
ok = err = 0
for p in sorted(json_dir.glob("*.json")):
    d = json.loads(p.read_text())
    if d.get("status") != "ok":
        err += 1
        continue
    ok += 1
    mas = d.get("motion_anomaly_score")
    sb = d.get("score_breakdown") or {}
    agg = sb.get("aggregate") or {}
    stats = sb.get("score_stats") or {}
    fmm = stats.get("flow_mag_mean") or {}

    items.append({
        "file": d.get("file") or p.stem + ".mp4",
        "ground_truth_label": d.get("ground_truth_label"),
        "status": d.get("status"),
        "model": d.get("model", model),
        "flow_mean": d.get("flow_mean") or agg.get("flow_mag_mean"),
        "flow_max": d.get("flow_max") or agg.get("flow_mag_max"),
        "flow_std": d.get("flow_std") or agg.get("flow_mag_std"),
        "frame_pairs": d.get("frame_pairs_used") or sb.get("frame_pairs_used"),
        "elapsed_ms": d.get("elapsed_ms"),
        "motion_anomaly_score": mas,
        "fake_score": mas,
        "pred_label": d.get("pred_label"),
        "threshold": sb.get("threshold", 0.5),
        "temporal_jitter": agg.get("temporal_jitter"),
        "spatial_inconsistency_mean": agg.get("spatial_inconsistency_mean"),
        "angle_dispersion_mean": agg.get("angle_dispersion_mean"),
        "flow_mag_pair_min": fmm.get("min"),
        "flow_mag_pair_max": fmm.get("max"),
        "flow_mag_pair_range": (
            (fmm.get("max") - fmm.get("min"))
            if fmm.get("max") is not None and fmm.get("min") is not None
            else None
        ),
        "error": d.get("error"),
    })

out = {
    "schema_version": "1.2",
    "run_id": s3_run_id,
    "model": model,
    "profile": profile,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "method": "optical_flow",
    "score_method": "motion_anomaly_heuristic",
    "threshold": 0.5,
    "count": len(items),
    "ok": ok,
    "error": err,
    "items": items,
}
(base / "infer_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"wrote infer_summary.json items={len(items)} ok={ok}")
PY
```

### 5-2. S3 재업로드

```bash
export S3_PREFIX=s3://forenshield-evidence-877044078824/deepfake/results/infer/gmflow/celebdf/$S3_RUN_ID

aws s3 cp results/infer/$LOCAL_RUN_ID/infer_summary.json $S3_PREFIX/infer_summary.json
```

---

## 6. `metrics.json`에 정확도 넣기

Optical heuristic accuracy는 **CNN accuracy와 별개**입니다.

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
from statistics import mean

base = Path(f"results/infer/{os.environ['LOCAL_RUN_ID']}")
items = json.loads((base / "infer_summary.json").read_text())["items"]
thr = 0.5

def acc(subset):
    if not subset:
        return None
    correct = sum(
        1 for x in subset
        if x.get("pred_label") == x.get("ground_truth_label")
    )
    return round(correct / len(subset), 4)

ok_items = [x for x in items if x.get("status") == "ok" and x.get("fake_score") is not None]
fake = [x for x in ok_items if x.get("ground_truth_label") == "fake"]
real = [x for x in ok_items if x.get("ground_truth_label") == "real"]

metrics = {
    "run_id": os.environ["S3_RUN_ID"],
    "model": "gmflow",
    "profile": os.environ["PROFILE"],
    "method": "optical_flow",
    "threshold": thr,
    "total": len(items),
    "ok": len(ok_items),
    "error": len(items) - len(ok_items),
    "heuristic_accuracy": acc(ok_items),
    "fake": {
        "total": len(fake),
        "avg_fake_score": round(mean(x["fake_score"] for x in fake), 6) if fake else None,
        "accuracy": acc(fake),
        "avg_flow_mag_pair_range": round(
            mean(x["flow_mag_pair_range"] for x in fake if x.get("flow_mag_pair_range") is not None), 6
        ) if fake else None,
    },
    "real": {
        "total": len(real),
        "avg_fake_score": round(mean(x["fake_score"] for x in real), 6) if real else None,
        "accuracy": acc(real),
        "avg_flow_mag_pair_range": round(
            mean(x["flow_mag_pair_range"] for x in real if x.get("flow_mag_pair_range") is not None), 6
        ) if real else None,
    },
    "note": "fake_score = motion_anomaly_score. Not comparable to CNN fake_score magnitude.",
}
(base / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print(json.dumps(metrics, indent=2))
PY

aws s3 cp results/infer/$LOCAL_RUN_ID/metrics.json $S3_PREFIX/metrics.json
```

---

## 7. threshold 튜닝 (선택)

기본 `0.5`는 고정값입니다. 프로필별로 맞추려면:

1. **같은 RUN**의 real 50개 `motion_anomaly_score` 분포 확인
2. fake 50개 분포와 겹치는지 확인
3. ROC/Youden 또는 운영 FPR 목표로 threshold 조정
4. `metrics.json`에 `threshold`와 `heuristic_accuracy` 함께 기록

```bash
python3 - <<'PY'
import json
from pathlib import Path
items = json.loads(Path("results/infer/YOUR_RUN/infer_summary.json").read_text())["items"]
for label in ("fake", "real"):
    scores = [x["fake_score"] for x in items if x.get("ground_truth_label")==label and x.get("fake_score") is not None]
  print(label, "n=", len(scores), "min=", min(scores), "max=", max(scores), "mean=", sum(scores)/len(scores))
PY
```

celebdf GMFlow는 fake/real `avg_flow_mean`이 비슷해 **분리가 어려울 수 있음** → threshold만으로 높은 accuracy 기대는 낮추세요.

---

## 8. 체크리스트

| 단계 | 확인 |
|------|------|
| infer | `ok=100/100` |
| `json/*.json` | `motion_anomaly_score`, `pred_label`, `score_breakdown` 존재 |
| `infer_summary.json` | `fake_score`, `pred_label`, `flow_mag_pair_range` 포함 (§5) |
| `metrics.json` | `heuristic_accuracy`, fake/real `avg_fake_score` |
| S3 | `gmflow/<profile>/<RUN_ID>/` 3파일 구조 |
| 보고서 | CNN accuracy와 optical heuristic **분리 표기** |

---

## 9. 흔한 실수

| 실수 | 올바른 해석 |
|------|-------------|
| `flow_mean`만 보고 fake/real 판별 | flow 크기만으로는 부족 → `motion_anomaly_score` 사용 |
| 수동 `infer_summary`만 업로드 | `json/` 기반으로 점수 필드 재생성 필요 |
| RAFT `flow_mean=271`과 GMFlow `0.03` 비교 | 스케일 다름 — 모델별 별도 baseline |
| infer `error` 있는 RUN으로 accuracy | error 샘플 제외 시 편향 (예: real만 남음) |
| `pred_label` = ground truth | 휴리스틱 오류 가능 — `metrics.heuristic_accuracy`로 평가 |

---

## 10. 관련 문서

| 문서 | 내용 |
|------|------|
| [VIDEO_DEEPFAKE_MODEL_BENCHMARK_3x3.md](./VIDEO_DEEPFAKE_MODEL_BENCHMARK_3x3.md) | 3×3 벤치마크, Optical 한계 |
| [VIDEO_BENCHMARK_DATASETS.md](./VIDEO_BENCHMARK_DATASETS.md) | celebdf / ffpp_vox 프로필 |
| `backend/docs/04-ai-json-spec.md` | API JSON (`deepfakeScore`, `frameAnomaly`) |

---

## 11. 요약

1. **GMFlow infer** → `aggregate` / `pair_stats` JSON
2. **`gmflow_motion_score.py`** → `fake_score`, `pred_label`, min/max range
3. EfficientNet처럼 쓰려면 `fake_score = motion_anomaly_score` (스크립트가 자동 설정)
4. **CNN fake_score와 숫자·accuracy 직접 비교 금지** — 보조 신호로 취급
