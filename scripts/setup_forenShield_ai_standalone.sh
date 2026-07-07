#!/usr/bin/env bash
# forenShield-ai: deepfake + forgery 트랙 원클릭 (단일 파일, git 불필요)
# 사용: bash setup_forenShield_ai_standalone.sh
# 옵션: SKIP_MIGRATE=1 | DRY_RUN=1 | FORENSHIELD_AI_ROOT=~/forenShield-ai
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
MODEL_VERSION="${MODEL_VERSION:-v1.0.0}"
PROFILE_VERSION="${PROFILE_VERSION:-v1.0.0}"
SKIP_MIGRATE="${SKIP_MIGRATE:-0}"
DRY_RUN="${DRY_RUN:-0}"

run() { if [[ "$DRY_RUN" == "1" ]]; then echo "[dry-run] $*"; else "$@"; fi; }

echo "==> forenShield-ai setup ROOT=${ROOT}"

mkdir -p "${ROOT}"

DIRS=(
  config logs/deepfake logs/forgery
  deepfake/models/test/video/xception/${MODEL_VERSION}
  deepfake/models/test/video/convnext/${MODEL_VERSION}
  deepfake/models/test/video/videomae/${MODEL_VERSION}
  deepfake/models/test/video/timesformer/${MODEL_VERSION}
  deepfake/models/test/video/video-swin/${MODEL_VERSION}
  deepfake/models/test/optical/gmflow deepfake/models/test/optical/raft deepfake/models/test/optical/pwcnet
  deepfake/models/dev/video/xception/${MODEL_VERSION}
  deepfake/models/deploy/video/xception/${MODEL_VERSION}
  deepfake/data/test/video deepfake/data/train/video deepfake/data/golden/v1/video
  deepfake/data/pull/evidence deepfake/data/pull/models deepfake/data/pull/golden/v1
  deepfake/data/raw/faceforensics deepfake/data/raw/voxceleb deepfake/data/raw/celeb-df-v2
  deepfake/data/cache deepfake/results/infer deepfake/results/eval deepfake/results/reports
  deepfake/checkpoints
  deepfake/scripts/download/models deepfake/scripts/download/data deepfake/scripts/download/deps
  deepfake/scripts/infer deepfake/scripts/eval deepfake/scripts/promote deepfake/scripts/upload
  forgery/config
  forgery/models/test/spatial/trufor/${MODEL_VERSION}
  forgery/models/test/spatial/catnet/${MODEL_VERSION}
  forgery/models/test/spatial/sparsevit/${MODEL_VERSION}
  forgery/models/test/temporal/gmflow
  forgery/models/test/compression/dc/${MODEL_VERSION}
  forgery/models/dev/spatial/trufor/${MODEL_VERSION}
  forgery/models/deploy/spatial/trufor/${MODEL_VERSION}
  forgery/models/deploy/profile/thresholds/${PROFILE_VERSION}
  forgery/data/test/video forgery/data/test/image/casia forgery/data/test/image/imd2020
  forgery/data/cal/threshold forgery/data/cal/fusion forgery/data/synth/regions
  forgery/data/golden/v1/video forgery/data/golden/v1/image
  forgery/data/pull/evidence forgery/data/pull/models forgery/data/raw forgery/data/cache
  forgery/results/infer forgery/results/eval forgery/results/reports forgery/checkpoints
  forgery/scripts/profile forgery/scripts/download/models forgery/scripts/download/data
  forgery/scripts/data forgery/scripts/infer forgery/scripts/eval forgery/scripts/promote forgery/scripts/upload
)
for d in "${DIRS[@]}"; do run mkdir -p "${ROOT}/${d}"; done

wim() { local p="$1"; [[ -f "$p" ]] && return 0; cat >"$p"; echo "  created $p"; }

if [[ ! -f "${ROOT}/config/env.local.example" ]]; then
wim "${ROOT}/config/env.local.example" <<'EOF'
export AWS_PROFILE=forenshield
export AWS_REGION=ap-northeast-2
export S3_MODELS_BUCKET=forenshield-models-877044078824
export S3_EVIDENCE_BUCKET=forenshield-evidence-877044078824
export FORENSHIELD_TRACK=deepfake
export FORENSHIELD_AI_ROOT="${HOME}/forenShield-ai/${FORENSHIELD_TRACK}"
EOF
fi

[[ -f "${ROOT}/config/env.local" ]] || run cp "${ROOT}/config/env.local.example" "${ROOT}/config/env.local"

if [[ ! -f "${ROOT}/forgery/config/thresholds.yaml" ]]; then
wim "${ROOT}/forgery/config/thresholds.yaml" <<EOF
version: "${PROFILE_VERSION}"
status: draft
metrics:
  blur: { high_if_gte: null }
  blockiness: { high_if_gte: null }
  fft_peak: { high_if_gte: null }
synthesis_margin_eps: 0.05
EOF
fi

if [[ ! -f "${ROOT}/forgery/config/weights_2nd.yaml" ]]; then
wim "${ROOT}/forgery/config/weights_2nd.yaml" <<'EOF'
version: "0.1"
status: draft
buckets:
  CLEAN:    { regions: [R0, R1], w_tr: 0.55, w_fl: 0.30, w_dc: 0.15 }
  COMPRESS: { regions: [R2, R3, R5], w_tr: 0.35, w_fl: 0.25, w_dc: 0.40 }
  BLUR:     { regions: [R4, R6], w_tr: 0.25, w_fl: 0.15, w_dc: 0.30 }
  HARD:     { regions: [R7], w_tr: 0.25, w_fl: 0.15, w_dc: 0.35, confidence_cap: 0.85 }
EOF
fi

# migrate flat → deepfake/ (rsync: skeleton과 겹쳐도 내용 병합)
if [[ "$SKIP_MIGRATE" != "1" ]]; then
  merge_into_deepfake() {
    local name="$1"
    if [[ ! -d "${ROOT}/${name}" ]]; then
      return 0
    fi
    echo "==> migrate ${name}/ -> deepfake/${name}/ (rsync merge)"
    run mkdir -p "${ROOT}/deepfake/${name}"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[dry-run] rsync -a ${ROOT}/${name}/ deepfake/${name}/"
      echo "[dry-run] rm -rf ${ROOT}/${name}"
    else
      rsync -a "${ROOT}/${name}/" "${ROOT}/deepfake/${name}/"
      rm -rf "${ROOT}/${name}"
    fi
  }
  for name in models data results checkpoints scripts; do
    merge_into_deepfake "${name}"
  done
fi

if [[ ! -d "${ROOT}/.venv" ]]; then
  if command -v python3.12 &>/dev/null; then run python3.12 -m venv "${ROOT}/.venv"
  elif command -v python3 &>/dev/null; then run python3 -m venv "${ROOT}/.venv"
  fi
fi

echo ""
echo "Done. deepfake=${ROOT}/deepfake  forgery=${ROOT}/forgery"
echo "  source ${ROOT}/.venv/bin/activate"
echo "  export FORENSHIELD_AI_ROOT=${ROOT}/deepfake   # 1차"
echo "  export FORENSHIELD_AI_ROOT=${ROOT}/forgery    # 2차"
echo "  find ${ROOT} -maxdepth 2 -type d | sort"
