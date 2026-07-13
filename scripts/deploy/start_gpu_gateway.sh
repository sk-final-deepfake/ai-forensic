#!/usr/bin/env bash
set -euo pipefail
pkill -f gpu_worker.rabbitmq_worker || true
pkill -f "uvicorn app.main_gateway" || true
sleep 2
source /home/sk4team/forenShield-ai/.venv/bin/activate
set -a
source /home/sk4team/forenShield-ai/config/env.local
set +a
unset AWS_PROFILE
export INFERENCE_MODE=local_model
export USE_MOCK_INFER=0
export AI_VISUALIZATION_ENABLED=1
export AI_VISUALIZATION_UPLOAD=1
export AI_VISUALIZATION_OVERLAY=1
export PYTHONPATH="/home/sk4team/ai-forensic:/home/sk4team/forenShield-ai/deepfake/scripts/infer:/home/sk4team/forenShield-ai/deepfake/scripts/eval"
cd /home/sk4team/ai-forensic
nohup python -m uvicorn app.main_gateway:app --host 0.0.0.0 --port 8000 >> /home/sk4team/forenShield-ai/logs/gpu_gateway.log 2>&1 &
sleep 4
curl -sf http://127.0.0.1:8000/health
echo
pgrep -af "uvicorn app.main_gateway" | head -1
