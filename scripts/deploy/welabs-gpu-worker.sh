#!/usr/bin/env bash
# welabs GPU worker deploy + restart
# Server: sk4team@58.151.205.220
# Repo:   /home/sk4team/ai-forensic
set -euo pipefail

AI_REPO="${AI_REPO:-/home/sk4team/ai-forensic}"
VENV="${VENV:-/home/sk4team/forenShield-ai/.venv}"
ENV_FILE="${ENV_FILE:-/home/sk4team/forenShield-ai/config/env.local}"
BRANCH="${BRANCH:-feature/ai-multi-face-infer}"
LOG_DIR="${LOG_DIR:-/home/sk4team/forenShield-ai/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/gpu_worker.log}"

echo "==> Fetch ${BRANCH} in ${AI_REPO}"
cd "${AI_REPO}"
git fetch origin
git checkout "${BRANCH}"
git pull origin "${BRANCH}"

echo "==> Stop legacy workers (Method B gateway optional — keep rabbitmq_worker only)"
pkill -f 'gpu_worker.rabbitmq_worker' || true
sleep 1
if pgrep -af 'gpu_worker.rabbitmq_worker' >/dev/null; then
  echo "ERROR: gpu_worker still running" >&2
  pgrep -af 'gpu_worker.rabbitmq_worker' >&2
  exit 1
fi

echo "==> Activate venv + env"
# shellcheck disable=SC1090
source "${VENV}/bin/activate"
# shellcheck disable=SC1090
source "${ENV_FILE}"
unset AWS_PROFILE

: "${INFERENCE_MODE:=local_model}"
: "${USE_MOCK_INFER:=0}"
export INFERENCE_MODE USE_MOCK_INFER

echo "INFERENCE_MODE=${INFERENCE_MODE} FORENSHIELD_AI_ROOT=${FORENSHIELD_AI_ROOT:-?} DEEPFAKE_ROOT=${DEEPFAKE_ROOT:-?}"

mkdir -p "${LOG_DIR}"
mkdir -p "${FORENSHIELD_AI_ROOT}/work"

echo "==> Start gpu_worker.rabbitmq_worker"
cd "${AI_REPO}"
nohup python -m gpu_worker.rabbitmq_worker >>"${LOG_FILE}" 2>&1 &
sleep 2

if ! pgrep -af 'gpu_worker.rabbitmq_worker' >/dev/null; then
  echo "ERROR: worker failed to start — tail log:" >&2
  tail -n 40 "${LOG_FILE}" >&2 || true
  exit 1
fi

echo "==> Worker running"
pgrep -af 'gpu_worker.rabbitmq_worker'
echo "Log: tail -f ${LOG_FILE}"
