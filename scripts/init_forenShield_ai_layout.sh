#!/usr/bin/env bash
# forenShield-ai GPU 워크스테이션 — deepfake / forgery 2트랙 디렉터리 초기화
#
# 사용 (GPU 서버):
#   bash init_forenShield_ai_layout.sh
#   FORENSHIELD_AI_ROOT=/data/forenShield-ai bash init_forenShield_ai_layout.sh
#
# 문서: docs/FORENSHIELD_AI_GPU_WORKSTATION.md
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
MODEL_VERSION="${MODEL_VERSION:-v1.0.0}"
PROFILE_VERSION="${PROFILE_VERSION:-v1.0.0}"

echo "==> forenShield-ai layout root: ${ROOT}"
mkdir -p "${ROOT}"

SHARED_DIRS=(
  "config"
  "logs/deepfake"
  "logs/forgery"
)

DEEPFAKE_DIRS=(
  "deepfake/models/test/video/xception/${MODEL_VERSION}"
  "deepfake/models/test/video/convnext/${MODEL_VERSION}"
  "deepfake/models/test/video/videomae/${MODEL_VERSION}"
  "deepfake/models/test/video/timesformer/${MODEL_VERSION}"
  "deepfake/models/test/video/video-swin/${MODEL_VERSION}"
  "deepfake/models/test/optical/gmflow"
  "deepfake/models/test/optical/raft"
  "deepfake/models/test/optical/pwcnet"
  "deepfake/models/dev/video/xception/${MODEL_VERSION}"
  "deepfake/models/deploy/video/xception/${MODEL_VERSION}"
  "deepfake/data/test/video"
  "deepfake/data/train/video"
  "deepfake/data/golden/v1/video"
  "deepfake/data/pull/evidence"
  "deepfake/data/pull/models"
  "deepfake/data/pull/golden/v1"
  "deepfake/data/raw/faceforensics"
  "deepfake/data/raw/voxceleb"
  "deepfake/data/raw/celeb-df-v2"
  "deepfake/data/cache"
  "deepfake/results/infer"
  "deepfake/results/eval"
  "deepfake/results/reports"
  "deepfake/checkpoints"
  "deepfake/scripts/download/models"
  "deepfake/scripts/download/data"
  "deepfake/scripts/download/deps"
  "deepfake/scripts/infer"
  "deepfake/scripts/eval"
  "deepfake/scripts/promote"
  "deepfake/scripts/upload"
)

FORGERY_DIRS=(
  "forgery/config"
  "forgery/models/test/spatial/trufor/${MODEL_VERSION}"
  "forgery/models/test/spatial/catnet/${MODEL_VERSION}"
  "forgery/models/test/spatial/sparsevit/${MODEL_VERSION}"
  "forgery/models/test/temporal/gmflow"
  "forgery/models/test/compression/dc/${MODEL_VERSION}"
  "forgery/models/dev/spatial/trufor/${MODEL_VERSION}"
  "forgery/models/deploy/spatial/trufor/${MODEL_VERSION}"
  "forgery/models/deploy/profile/thresholds/${PROFILE_VERSION}"
  "forgery/data/test/video"
  "forgery/data/test/image/casia"
  "forgery/data/test/image/imd2020"
  "forgery/data/cal/threshold"
  "forgery/data/cal/fusion"
  "forgery/data/synth/regions"
  "forgery/data/golden/v1/video"
  "forgery/data/golden/v1/image"
  "forgery/data/pull/evidence"
  "forgery/data/pull/models"
  "forgery/data/raw"
  "forgery/data/cache"
  "forgery/results/infer"
  "forgery/results/eval"
  "forgery/results/reports"
  "forgery/checkpoints"
  "forgery/scripts/profile"
  "forgery/scripts/download/models"
  "forgery/scripts/download/data"
  "forgery/scripts/data"
  "forgery/scripts/infer"
  "forgery/scripts/eval"
  "forgery/scripts/promote"
  "forgery/scripts/upload"
)

for d in "${SHARED_DIRS[@]}" "${DEEPFAKE_DIRS[@]}" "${FORGERY_DIRS[@]}"; do
  mkdir -p "${ROOT}/${d}"
done

write_if_missing() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    echo "    skip (exists): ${path}"
    return 0
  fi
  cat > "${path}"
  echo "    created: ${path}"
}

echo "==> shared config templates"

write_if_missing "${ROOT}/config/env.local.example" <<'EOF'
# cp config/env.local.example config/env.local && 편집
export AWS_PROFILE=forenshield
export AWS_REGION=ap-northeast-2
export S3_MODELS_BUCKET=forenshield-models-877044078824
export S3_EVIDENCE_BUCKET=forenshield-evidence-877044078824

# 스크립트 기본 트랙 (deepfake | forgery)
export FORENSHIELD_TRACK=deepfake
export FORENSHIELD_AI_ROOT="${HOME}/forenShield-ai/${FORENSHIELD_TRACK}"
EOF

write_if_missing "${ROOT}/config/buckets.yaml" <<'EOF'
# S3 버킷·prefix — 상세: ai-forensic/docs/FORENSHIELD_AI_GPU_WORKSTATION.md
buckets:
  evidence: forenshield-evidence-877044078824
  models: forenshield-models-877044078824

pull:
  evidence_prefix: cases/
  models_prefix: deepfake/deploy/video/
  golden_prefix: golden-set/

upload:
  models_from: models/deploy/
  golden_from: data/golden/
  golden_to_prefix: golden-set/

tracks:
  deepfake:
    s3_benchmark_prefix: deepfake/datasets/bench/
    s3_infer_prefix: deepfake/results/infer/
    s3_legacy_reports_prefix: deepfake/archive/legacy-benchmarks/
  forgery:
    s3_benchmark_prefix: cases/test/forgery-benchmark-datasets/
EOF

write_if_missing "${ROOT}/config/models.deepfake.yaml" <<EOF
# 1차 딥페이크 모델 레지스트리 — 로컬 루트: deepfake/
track: deepfake
models:
  - id: xception
    modality: video
    version: ${MODEL_VERSION}
    paths:
      test: deepfake/models/test/video/xception/${MODEL_VERSION}
      dev: deepfake/models/dev/video/xception/${MODEL_VERSION}
      deploy: deepfake/models/deploy/video/xception/${MODEL_VERSION}
    s3_deploy_prefix: deepfake/deploy/video/xception/${MODEL_VERSION}
  - id: convnext
    modality: video
    version: ${MODEL_VERSION}
    paths:
      test: deepfake/models/test/video/convnext/${MODEL_VERSION}
  - id: videomae
    modality: video
    version: ${MODEL_VERSION}
    paths:
      test: deepfake/models/test/video/videomae/${MODEL_VERSION}
  - id: timesformer
    modality: video
    version: ${MODEL_VERSION}
    paths:
      test: deepfake/models/test/video/timesformer/${MODEL_VERSION}
  - id: gmflow
    modality: optical
    paths:
      test: deepfake/models/test/optical/gmflow
EOF

write_if_missing "${ROOT}/config/models.forgery.yaml" <<EOF
# 2차 위변조 모델·프로파일 레지스트리 — 로컬 루트: forgery/
track: forgery
models:
  - id: trufor
    role: spatial
    version: ${MODEL_VERSION}
    paths:
      test: forgery/models/test/spatial/trufor/${MODEL_VERSION}
      dev: forgery/models/dev/spatial/trufor/${MODEL_VERSION}
      deploy: forgery/models/deploy/spatial/trufor/${MODEL_VERSION}
    s3_deploy_prefix: forgery/spatial/trufor/${MODEL_VERSION}
  - id: catnet
    role: spatial
    version: ${MODEL_VERSION}
    paths:
      test: forgery/models/test/spatial/catnet/${MODEL_VERSION}
  - id: gmflow-discontinuity
    role: temporal
    paths:
      test: forgery/models/test/temporal/gmflow
    note: flow backbone은 deepfake/optical/gmflow 와 동일 가중치 공유 가능
  - id: compression-profile
    role: profile
    version: ${PROFILE_VERSION}
    paths:
      deploy: forgery/models/deploy/profile/thresholds/${PROFILE_VERSION}
    files:
      - thresholds.yaml
      - weights_2nd.yaml
EOF

write_if_missing "${ROOT}/forgery/config/thresholds.yaml" <<EOF
version: "${PROFILE_VERSION}"
status: draft
calibration:
  dataset: null
  method: module_performance_changepoint
metrics:
  blur:
    high_if_gte: null
  blockiness:
    high_if_gte: null
  fft_peak:
    high_if_gte: null
synthesis_margin_eps: 0.05
EOF

write_if_missing "${ROOT}/forgery/config/weights_2nd.yaml" <<'EOF'
version: "0.1"
status: draft
buckets:
  CLEAN:   { regions: [R0, R1], w_tr: 0.55, w_fl: 0.30, w_dc: 0.15 }
  COMPRESS: { regions: [R2, R3, R5], w_tr: 0.35, w_fl: 0.25, w_dc: 0.40 }
  BLUR:    { regions: [R4, R6], w_tr: 0.25, w_fl: 0.15, w_dc: 0.30 }
  HARD:    { regions: [R7], w_tr: 0.25, w_fl: 0.15, w_dc: 0.35, confidence_cap: 0.85 }
EOF

write_if_missing "${ROOT}/deepfake/data/golden/v1/manifest.json" <<'EOF'
{
  "version": "v1",
  "track": "deepfake",
  "description": "1차 딥페이크 회귀 고정셋",
  "items": []
}
EOF

write_if_missing "${ROOT}/forgery/data/golden/v1/manifest.json" <<'EOF'
{
  "version": "v1",
  "track": "forgery",
  "description": "2차 위변조 회귀 고정셋 — spatial/temporal 라벨 분리 권장",
  "items": []
}
EOF

write_if_missing "${ROOT}/README.md" <<'EOF'
# forenShield-ai (GPU 워크스테이션)

1차 **deepfake** · 2차 **forgery** 트랙을 분리한 모델 실험 작업 공간.

## 트랙 선택

```bash
cd ~/forenShield-ai
source .venv/bin/activate

# 1차 딥페이크 (기본)
export FORENSHIELD_TRACK=deepfake
export FORENSHIELD_AI_ROOT="$HOME/forenShield-ai/deepfake"

# 2차 위변조
export FORENSHIELD_TRACK=forgery
export FORENSHIELD_AI_ROOT="$HOME/forenShield-ai/forgery"
```

## 구조 상세

`ai-forensic/docs/FORENSHIELD_AI_GPU_WORKSTATION.md`

## 디렉터리 요약

| 경로 | 용도 |
|------|------|
| `deepfake/` | 1차 Xception·Transformer·GMFlow 벤치 |
| `forgery/` | 2차 TruFor·GMFlow(컷)·DC·region 프로파일 |
| `config/` | 공통 env·S3·모델 레지스트리 (deepfake/forgery yaml) |
| `.venv/` | 공통 Python 환경 |
EOF

write_if_missing "${ROOT}/deepfake/scripts/download/models/README.md" <<'EOF'
# deepfake — download/models

ai-forensic 저장소 `scripts/download/models/` 스크립트를 복사하거나 symlink.
실행 전: `export FORENSHIELD_AI_ROOT=~/forenShield-ai/deepfake`
EOF

write_if_missing "${ROOT}/forgery/scripts/profile/README.md" <<'EOF'
# forgery — profile

| 스크립트 (P2 예정) | 기능 |
|-------------------|------|
| compression_profile.py | blur, blockiness, fft_peak, region_id |
| calibrate_thresholds.py | cal set → thresholds.yaml |
EOF

write_if_missing "${ROOT}/forgery/scripts/data/README.md" <<'EOF'
# forgery — data

| 스크립트 (P2 예정) | 기능 |
|-------------------|------|
| synthesize_region_variants.py | region box 안 랜덤 가공 |
EOF

echo ""
echo "==> Done. Layout at: ${ROOT}"
echo "    deepfake track: ${ROOT}/deepfake"
echo "    forgery track:  ${ROOT}/forgery"
echo ""
echo "    기존 flat 구조 마이그레이션:"
echo "    bash ai-forensic/scripts/migrate_flat_to_track_layout.sh"
echo ""
echo "    Full doc: ai-forensic/docs/FORENSHIELD_AI_GPU_WORKSTATION.md"
