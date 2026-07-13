#!/usr/bin/env bash
# welabs Method B — GPU Gateway only (no rabbitmq_worker)
# Server: sk4team@58.151.205.220
set -euo pipefail

AI_REPO="${AI_REPO:-/home/sk4team/ai-forensic}"
ENV_LOCAL="${ENV_LOCAL:-/home/sk4team/forenShield-ai/config/env.local}"
VENV="${VENV:-/home/sk4team/forenShield-ai/.venv}"
LOG_DIR="${LOG_DIR:-/home/sk4team/forenShield-ai/logs}"
GW_LOG="${GW_LOG:-${LOG_DIR}/gpu_gateway.log}"
GPU_ENV="${AI_REPO}/gpu_worker/.env"
BRANCH="${BRANCH:-feature/ai-module-overlays}"

echo "==> Fetch ${BRANCH} in ${AI_REPO}"
cd "${AI_REPO}"
git fetch origin
if git show-ref --verify --quiet "refs/remotes/origin/${BRANCH}"; then
  git checkout "${BRANCH}"
  git pull origin "${BRANCH}"
else
  git checkout main
  git pull origin main
fi
echo "HEAD: $(git log -1 --oneline)"

echo "==> Stop rabbitmq_worker + old gateway"
pkill -f 'gpu_worker.rabbitmq_worker' || true
pkill -f 'uvicorn app.main_gateway' || true
sleep 2
if pgrep -af 'gpu_worker.rabbitmq_worker' >/dev/null; then
  echo "ERROR: rabbitmq_worker still running" >&2
  pgrep -af 'gpu_worker.rabbitmq_worker' >&2
  exit 1
fi

echo "==> Write ${GPU_ENV}"
mkdir -p "${AI_REPO}/gpu_worker"
if [[ -f "${ENV_LOCAL}" ]]; then
  sed -i 's/\r$//' "${ENV_LOCAL}" || true
fi

cat > "${GPU_ENV}" <<'ENV_EOF'
FORENSHIELD_AI_ROOT=/home/sk4team/forenShield-ai/deepfake
DEEPFAKE_ROOT=/home/sk4team/ai-forensic
PYTHONPATH=/home/sk4team/ai-forensic

INFERENCE_MODE=local_model
USE_MOCK_INFER=0
INFER_DEVICE=cuda

MODEL_CHECKPOINT_PATH=/home/sk4team/forenShield-ai/deepfake/models/test/video/xception/v1.0.0/xception_finetuned_celeb1k.pth
TIMESFORMER_WEIGHTS=/home/sk4team/forenShield-ai/deepfake/models/test/video/timesformer/v1.0.0/timesformer_finetuned_celeb1k.pth
FUSION_CONFIG_PATH=/home/sk4team/forenShield-ai/deepfake/config/fusion_v4_ts_gated.json
GMFLOW_PRETRAINED=/home/sk4team/forenShield-ai/deepfake/models/test/video/optical-flow/gmflow/pretrained/gmflow_things-e9887eda.pth
GMFLOW_LEARNED_HEAD=/home/sk4team/forenShield-ai/deepfake/models/test/video/optical-flow/gmflow/v1.0.0/gmflow_learned_head.joblib
GMFLOW_META=/home/sk4team/forenShield-ai/deepfake/models/test/video/optical-flow/gmflow/v1.0.0/gmflow_best.meta.json

INFERENCE_MAX_FRAMES=32
INFERENCE_SAMPLE_FPS=1
DEEPFAKE_THRESHOLD=0.5

AWS_REGION=ap-northeast-2
S3_EVIDENCE_BUCKET=forenshield-evidence-877044078824

AI_VISUALIZATION_ENABLED=1
AI_VISUALIZATION_UPLOAD=1
AI_VISUALIZATION_OVERLAY=1
AI_VISUALIZATION_MAX_FRAMES=3
AI_VISUALIZATION_OVERLAY_MAX_SEC=60
AI_VISUALIZATION_PRESIGN_SEC=604800
AI_VISUALIZATION_PREFIX=deepfake/artifacts/analysis/{evidence_id}/{analysis_request_id}
ENV_EOF

if [[ -f "${ENV_LOCAL}" ]]; then
  grep -E '^(RABBITMQ_|AI_RESULT_)' "${ENV_LOCAL}" >> "${GPU_ENV}" || true
fi

grep -E '^(FORENSHIELD_AI_ROOT|DEEPFAKE_ROOT|INFERENCE_MODE|AI_VISUALIZATION_OVERLAY)=' "${GPU_ENV}"

echo "==> Verify model files"
for f in \
  /home/sk4team/forenShield-ai/deepfake/models/test/video/xception/v1.0.0/xception_finetuned_celeb1k.pth \
  /home/sk4team/forenShield-ai/deepfake/models/test/video/timesformer/v1.0.0/timesformer_finetuned_celeb1k.pth \
  /home/sk4team/forenShield-ai/deepfake/config/fusion_v4_ts_gated.json; do
  if [[ -f "$f" ]]; then echo "OK $f"; else echo "MISSING $f" >&2; fi
done

echo "==> Start gateway"
mkdir -p "${LOG_DIR}" /home/sk4team/forenShield-ai/deepfake/work
# shellcheck disable=SC1090
source "${VENV}/bin/activate"
unset AWS_PROFILE
cd "${AI_REPO}"
nohup python -m uvicorn app.main_gateway:app --host 0.0.0.0 --port 8000 >>"${GW_LOG}" 2>&1 &
sleep 4

if ! curl -sf http://127.0.0.1:8000/health; then
  echo "ERROR: health check failed" >&2
  tail -n 40 "${GW_LOG}" >&2 || true
  exit 1
fi
echo

pgrep -af 'uvicorn app.main_gateway'
echo "Log: tail -f ${GW_LOG}"
echo "DEPLOY_OK"
