#!/usr/bin/env bash
# forenShield-ai GPU 워크스테이션 원클릭 설정
#   deepfake/ + forgery/ 트랙 생성, (선택) 기존 flat 구조 자동 이전
#
# 사용 (GPU SSH 후 한 줄):
#   curl -fsSL ... | bash
#   또는 ai-forensic clone 후:
#   bash ~/ai-forensic/scripts/setup_gpu_workstation.sh
#
# 옵션:
#   FORENSHIELD_AI_ROOT=~/forenShield-ai   루트 (기본)
#   SKIP_MIGRATE=1                         기존 데이터 이전 안 함
#   SKIP_VENV=1                            venv 생성/활성 안내 생략
#   DRY_RUN=1                              migrate만 미리보기
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
SKIP_MIGRATE="${SKIP_MIGRATE:-0}"
SKIP_VENV="${SKIP_VENV:-0}"

echo "=============================================="
echo " forenShield-ai GPU setup"
echo " ROOT=${ROOT}"
echo "=============================================="

mkdir -p "${ROOT}"

# --- 1) 디렉터리 skeleton (deepfake + forgery) ---
echo ""
echo "==> [1/4] create deepfake/ + forgery/ layout"
FORENSHIELD_AI_ROOT="${ROOT}" bash "${SCRIPT_DIR}/init_forenShield_ai_layout.sh"

# --- 2) 기존 flat 구조 자동 감지 → deepfake/ 로 이전 ---
needs_migrate=0
if [[ "${SKIP_MIGRATE}" != "1" ]]; then
  for name in models data results checkpoints scripts; do
    if [[ -e "${ROOT}/${name}" && ! -L "${ROOT}/${name}" ]]; then
      # deepfake/ 안만 있고 루트에 flat 폴더가 있으면 migrate
      needs_migrate=1
      break
    fi
  done
fi

echo ""
if [[ "${needs_migrate}" == "1" ]]; then
  echo "==> [2/4] migrate existing flat folders → deepfake/"
  DRY_RUN="${DRY_RUN:-0}" FORENSHIELD_AI_ROOT="${ROOT}" \
    bash "${SCRIPT_DIR}/migrate_flat_to_track_layout.sh"
else
  echo "==> [2/4] skip migrate (no flat models/data at root, or SKIP_MIGRATE=1)"
fi

# --- 3) env.local ---
echo ""
echo "==> [3/4] config/env.local"
if [[ ! -f "${ROOT}/config/env.local" ]]; then
  cp "${ROOT}/config/env.local.example" "${ROOT}/config/env.local"
  echo "    created ${ROOT}/config/env.local (edit AWS_PROFILE if needed)"
else
  echo "    skip (exists): config/env.local"
fi

# --- 4) venv ---
echo ""
echo "==> [4/4] Python venv"
if [[ "${SKIP_VENV}" == "1" ]]; then
  echo "    skipped (SKIP_VENV=1)"
elif [[ ! -d "${ROOT}/.venv" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "${ROOT}/.venv"
    echo "    created ${ROOT}/.venv"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m venv "${ROOT}/.venv"
    echo "    created ${ROOT}/.venv (python3)"
  else
    echo "    WARN: python3 not found — create venv manually"
  fi
else
  echo "    skip (exists): .venv"
fi

# --- 완료 안내 ---
cat <<EOF

==============================================
 Done.
==============================================

트리 확인:
  find ${ROOT} -maxdepth 3 -type d | sort | head -40

매번 작업 전:
  cd ${ROOT}
  source .venv/bin/activate

1차 딥페이크:
  export FORENSHIELD_TRACK=deepfake
  export FORENSHIELD_AI_ROOT=${ROOT}/deepfake

2차 위변조:
  export FORENSHIELD_TRACK=forgery
  export FORENSHIELD_AI_ROOT=${ROOT}/forgery

문서: ai-forensic/docs/FORENSHIELD_AI_GPU_WORKSTATION.md
==============================================
EOF
