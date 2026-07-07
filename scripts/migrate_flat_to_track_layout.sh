#!/usr/bin/env bash
# 기존 ~/forenShield-ai flat 구조 → deepfake/ 서브트랙으로 이동
#
# 사용 (GPU, 백업 후):
#   DRY_RUN=1 bash migrate_flat_to_track_layout.sh   # 미리보기
#   bash migrate_flat_to_track_layout.sh
#
# forgery/ 는 init_forenShield_ai_layout.sh 로 생성. 이 스크립트는 deepfake 마이그레이션만.
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
DRY_RUN="${DRY_RUN:-0}"

run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

echo "==> migrate flat layout under: ${ROOT}"
echo "    DRY_RUN=${DRY_RUN}"

if [[ ! -d "${ROOT}" ]]; then
  echo "ERROR: root not found: ${ROOT}" >&2
  exit 1
fi

# forgery + deepfake skeleton
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run bash "${SCRIPT_DIR}/init_forenShield_ai_layout.sh"

move_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -e "${ROOT}/${src}" && ! -e "${ROOT}/${dst}" ]]; then
    run mkdir -p "$(dirname "${ROOT}/${dst}")"
    run mv "${ROOT}/${src}" "${ROOT}/${dst}"
    echo "    moved: ${src} -> ${dst}"
  fi
}

# flat top-level → deepfake/
for name in models data results checkpoints; do
  if [[ -d "${ROOT}/${name}" && "${ROOT}/${name}" != "${ROOT}/deepfake/${name}" ]]; then
    # merge: if deepfake/name exists from init, move children
    if [[ -d "${ROOT}/deepfake/${name}" ]]; then
      echo "==> merge ${name}/ into deepfake/${name}/"
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] rsync -a ${ROOT}/${name}/ ${ROOT}/deepfake/${name}/"
        echo "[dry-run] rm -rf ${ROOT}/${name}"
      else
        shopt -s dotglob
        for item in "${ROOT}/${name}"/*; do
          base="$(basename "${item}")"
          if [[ ! -e "${ROOT}/deepfake/${name}/${base}" ]]; then
            mv "${item}" "${ROOT}/deepfake/${name}/"
            echo "    moved: ${name}/${base} -> deepfake/${name}/${base}"
          else
            echo "    skip (dest exists): deepfake/${name}/${base}"
          fi
        done
        rmdir "${ROOT}/${name}" 2>/dev/null || true
      fi
    else
      move_if_exists "${name}" "deepfake/${name}"
    fi
  fi
done

# flat scripts/ → deepfake/scripts/ (merge)
if [[ -d "${ROOT}/scripts" && ! -L "${ROOT}/scripts" ]]; then
  echo "==> merge scripts/ into deepfake/scripts/"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] merge ${ROOT}/scripts -> ${ROOT}/deepfake/scripts"
  else
    mkdir -p "${ROOT}/deepfake/scripts"
    shopt -s dotglob
    for item in "${ROOT}/scripts"/*; do
      base="$(basename "${item}")"
      if [[ ! -e "${ROOT}/deepfake/scripts/${base}" ]]; then
        mv "${item}" "${ROOT}/deepfake/scripts/${base}"
        echo "    moved: scripts/${base} -> deepfake/scripts/${base}"
      fi
    done
    rmdir "${ROOT}/scripts" 2>/dev/null || true
  fi
fi

echo ""
echo "==> Done."
echo "    export FORENSHIELD_TRACK=deepfake"
echo "    export FORENSHIELD_AI_ROOT=${ROOT}/deepfake"
echo "    forgery: export FORENSHIELD_TRACK=forgery && FORENSHIELD_AI_ROOT=${ROOT}/forgery"
