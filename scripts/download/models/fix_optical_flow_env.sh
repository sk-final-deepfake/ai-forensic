#!/usr/bin/env bash
# Fix RAFT weights + optional GMFlow check before benchmark.
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"

python3 - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "scripts/infer")
from optical_flow_backends import ensure_raft_weights

path = ensure_raft_weights(Path("models/test/video/optical-flow/raft/raft-things.pth"))
print("RAFT ok:", path, path.stat().st_size)
PY

GM=$(ls models/test/video/optical-flow/gmflow/pretrained/gmflow_things*.pth 2>/dev/null | head -1 || true)
if [[ -n "$GM" ]]; then
  echo "GMFlow ok: $GM"
else
  echo "WARN: GMFlow weights not found under models/test/video/optical-flow/gmflow/pretrained/"
fi

echo "next: bash scripts/infer/run_optical_flow_celebdf_benchmark.sh"
