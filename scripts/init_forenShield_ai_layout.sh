#!/usr/bin/env bash
# forenShield-ai GPU 워크스테이션 디렉터리·설정 템플릿 초기화
#
# 사용 (GPU 서버):
#   bash init_forenShield_ai_layout.sh
#   FORENSHIELD_AI_ROOT=/data/forenShield-ai bash init_forenShield_ai_layout.sh
#
# 문서: docs/FORENSHIELD_AI_GPU_WORKSTATION.md
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
MODEL_VERSION="${MODEL_VERSION:-v1.0.0}"

echo "==> forenShield-ai layout root: ${ROOT}"
mkdir -p "${ROOT}"

DIRS=(
  "config"
  "models/test/video/xception/${MODEL_VERSION}"
  "models/dev/video/xception/${MODEL_VERSION}"
  "models/deploy/video/xception/${MODEL_VERSION}"
  "data/test/video"
  "data/golden/v1/video"
  "data/pull/evidence"
  "data/pull/models"
  "data/pull/golden/v1"
  "data/raw"
  "data/cache"
  "results/infer"
  "results/eval"
  "results/reports"
  "checkpoints"
  "scripts/download/models"
  "scripts/download/data"
  "scripts/download/deps"
  "scripts/infer"
  "scripts/eval"
  "scripts/promote"
  "scripts/upload"
  "logs"
)

for d in "${DIRS[@]}"; do
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

echo "==> config templates"

write_if_missing "${ROOT}/config/env.local.example" <<'EOF'
# cp config/env.local.example config/env.local && 편집
export AWS_PROFILE=forenshield
export AWS_REGION=ap-northeast-2
export S3_MODELS_BUCKET=forenshield-models-877044078824
export S3_EVIDENCE_BUCKET=forenshield-evidence-877044078824
EOF

write_if_missing "${ROOT}/config/buckets.yaml" <<'EOF'
# S3 버킷·prefix — 상세: ai-forensic/docs/FORENSHIELD_AI_GPU_WORKSTATION.md
buckets:
  evidence: forenshield-evidence-877044078824
  models: forenshield-models-877044078824

pull:
  evidence_prefix: cases/
  models_prefix: video/
  golden_prefix: golden-set/

upload:
  models_from: models/deploy/
  golden_from: data/golden/
  golden_to_prefix: golden-set/
EOF

write_if_missing "${ROOT}/config/models.yaml" <<EOF
# 모델 레지스트리 — test / dev / deploy 로컬 경로 + S3 deploy prefix
models:
  - id: xception
    modality: video
    version: ${MODEL_VERSION}
    paths:
      test: models/test/video/xception/${MODEL_VERSION}
      dev: models/dev/video/xception/${MODEL_VERSION}
      deploy: models/deploy/video/xception/${MODEL_VERSION}
    s3_deploy_prefix: video/xception/${MODEL_VERSION}
    external_url: https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth
EOF

write_if_missing "${ROOT}/data/golden/v1/manifest.json" <<'EOF'
{
  "version": "v1",
  "description": "회귀 테스트 고정셋 — video 파일과 기대 라벨을 items에 추가하세요",
  "items": []
}
EOF

write_if_missing "${ROOT}/README.md" <<'EOF'
# forenShield-ai (GPU 워크스테이션)

모델 test / dev / deploy, 데이터, infer/eval, S3 sync 작업 공간.

## 빠른 시작

```bash
cd ~/forenShield-ai
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
# cp config/env.local.example config/env.local
```

## 구조 상세

`ai-forensic` 저장소: `docs/FORENSHIELD_AI_GPU_WORKSTATION.md`

## 디렉터리 요약

| 경로 | 용도 |
|------|------|
| models/test | 실험·초기 다운로드 |
| models/dev | 개발·튜닝 통과본 |
| models/deploy | S3 업로드·운영 확정본 |
| data/test | 실험용 테스트 데이터셋 (video) |
| data/golden | 회귀 테스트 고정셋 |
| data/pull/evidence | S3 evidence copy pull |
| data/pull/models | S3 deploy 모델 pull |
| results/infer | 추론 결과 |
| results/eval | 평가 지표 |
| scripts/download | 모델·데이터 다운로드 |
| scripts/infer | 추론 실행 |
| scripts/eval | golden 채점 |
| scripts/promote | test→dev→deploy |
| scripts/upload | S3 업로드 |
EOF

write_if_missing "${ROOT}/scripts/download/data/README.md" <<'EOF'
# download/data

| 스크립트 (추가 예정) | 기능 |
|---------------------|------|
| s3_pull_evidence.sh | evidence 버킷 copy → data/pull/evidence/ |
| s3_pull_deploy_model.sh | models 버킷 deploy → data/pull/models/ |
| s3_pull_golden.sh | golden-set S3 ↔ data/golden/ |
| golden_fetch_samples.sh | golden 로컬 샘플 구성 |
EOF

write_if_missing "${ROOT}/scripts/download/models/README.md" <<'EOF'
# download/models

| 스크립트 (추가 예정) | 기능 |
|---------------------|------|
| video_download_xception.sh | DeepfakeBench → models/test/video/xception/ |
EOF

echo ""
echo "==> Done. Layout at: ${ROOT}"
echo "    find ${ROOT} -maxdepth 3 -type d | sort"
echo "    Full doc: ai-forensic/docs/FORENSHIELD_AI_GPU_WORKSTATION.md"
