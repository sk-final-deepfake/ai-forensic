#!/usr/bin/env bash
# Download Kaggle DFDC train_sample_videos (~4GB) to GPU storage.
#
# Usage (GPU):
#   cd ~/forenShield-ai
#   source .venv/bin/activate
#   bash scripts/download/data/download_dfdc_subset.sh
#
# Options passed through to Python, e.g.:
#   bash scripts/download/data/download_dfdc_subset.sh --balanced --target 50
#   bash scripts/download/data/download_dfdc_subset.sh --full-only
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "$ROOT"

if [ -d "$ROOT/.venv" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

unset AWS_PROFILE

pip install -U datasets huggingface_hub

if ! command -v kaggle >/dev/null 2>&1; then
  pip install -U kaggle
fi

python3 "$ROOT/scripts/download/data/download_dfdc_subset.py" --root "$ROOT" "$@"

echo ""
echo "DFDC locations:"
echo "  full sample: $ROOT/data/raw/benchmark-downloads/dfdc/train_sample_videos/"
echo "  benchmark:   $ROOT/data/test/video/dfdc/"
