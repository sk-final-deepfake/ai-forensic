#!/usr/bin/env bash
# GPU 서버(~/forenShield-ai)에서 Xception local_model 추론 설정 + 스모크 테스트
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "$ROOT"

echo "== forenShield-ai root: $ROOT"

if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found. Create venv first."
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$ROOT"

echo "== Install inference deps"
pip install -q -U pip
pip install -q torch torchvision timm opencv-python-headless numpy pika httpx boto3 pydantic python-dotenv

echo "== Find Xception checkpoint under models/test/video/xception"
CHECKPOINT="$(find models/test/video/xception -name '*.pth' 2>/dev/null | head -1 || true)"
if [[ -z "$CHECKPOINT" ]]; then
  echo "ERROR: No .pth under models/test/video/xception"
  find models/test/video/xception -type f 2>/dev/null || true
  exit 1
fi
echo "Using checkpoint: $CHECKPOINT"

ENV_FILE="gpu_worker/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  cp gpu_worker/.env.example "$ENV_FILE"
fi

upsert_env() {
  local key="$1" val="$2"
  if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
  else
    echo "${key}=${val}" >> "$ENV_FILE"
  fi
}

upsert_env FORENSHIELD_AI_ROOT "$ROOT"
upsert_env PYTHONPATH "$ROOT"
upsert_env INFERENCE_MODE local_model
upsert_env INFERENCE_DEVICE cuda
upsert_env INFERENCE_MODEL_ID xception
upsert_env INFERENCE_MODEL_VERSION test
upsert_env MODEL_CHECKPOINT_PATH "$CHECKPOINT"
upsert_env INFERENCE_SAMPLE_FPS 1
upsert_env INFERENCE_MAX_FRAMES 32
upsert_env DEEPFAKE_THRESHOLD 0.5

echo "== gpu_worker/.env (inference section)"
grep -E '^(FORENSHIELD|PYTHONPATH|INFERENCE_|MODEL_|DEEPFAKE_)' "$ENV_FILE" || true

VIDEO="${1:-}"
if [[ -z "$VIDEO" ]]; then
  VIDEO="$(ls -1 work/*.mp4 2>/dev/null | head -1 || true)"
fi
if [[ -z "$VIDEO" ]]; then
  VIDEO="$(ls -1 data/test/video/*.mp4 2>/dev/null | head -1 || true)"
fi
if [[ -z "$VIDEO" || ! -f "$VIDEO" ]]; then
  echo "WARN: No sample video. Pass path: bash scripts/infer/gpu_setup_xception.sh /path/to/video.mp4"
  exit 0
fi

echo "== Smoke test: Xception inference on $VIDEO"
python scripts/infer/run_xception_infer.py "$VIDEO"

echo ""
echo "== Full pipeline JSON (backend format)"
python -m gpu_worker.run_offline sample "$VIDEO" --evidence-id 9001 --request-id 90001

echo ""
echo "Done. JSON files:"
ls -la results/infer/*xception*.json results/analysis_90001_9001.json 2>/dev/null || true

echo ""
echo "Next steps:"
echo "  1) Restart worker:  pkill -f gpu_worker.rabbitmq_worker; nohup python -m gpu_worker.rabbitmq_worker >> logs/worker.log 2>&1 &"
echo "  2) 프론트에서 증거 재분석(또는 새 업로드) → 상세 페이지에서 Xception 결과 확인"
