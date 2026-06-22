#!/usr/bin/env bash
# Celeb-DF v2 download/extract helper for forenShield-ai GPU.
#
# Step 1 (browser): https://forms.gle/2jYBby6y1FBU3u6q9
# Step 2: wait for approval email with download link
# Kaggle (no official form wait):
#   bash scripts/download/data/download_celebdf_v2.sh --source kaggle --sample-real 50 --sample-fake 50
#
# Official email link:
#   bash scripts/download/data/download_celebdf_v2.sh --source url --url 'https://drive.google.com/...'
#
# Optional benchmark sample (50 real + 50 fake):
#   bash scripts/download/data/download_celebdf_v2.sh --url '...' --sample-real 50 --sample-fake 50
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
unset AWS_PROFILE

pip install -q gdown

python3 "$ROOT/scripts/download/data/download_celebdf_v2.py" --root "$ROOT" "$@"

echo ""
echo "Celeb-DF v2 locations:"
echo "  full:      $ROOT/data/raw/celeb-df-v2/Celeb-DF-v2/"
echo "  benchmark: $ROOT/data/test/video/celeb-df-v2/"
