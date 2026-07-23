#!/usr/bin/env bash
# Extract xception fine-tune bundle into ~/forenShield-ai (run ON GPU).
# Get bundle onto GPU first (from Windows):
#   scp c:\sw_study\finalpjt\ai\xception_finetune_bundle.tgz sk4team@58.127.241.84:~/
# Then on GPU:
#   bash ~/forenShield-ai/scripts/deploy/install_xception_finetune_bundle.sh
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
BUNDLE="${1:-$HOME/xception_finetune_bundle.tgz}"

if [[ ! -f "$BUNDLE" ]]; then
  echo "ERROR: bundle not found: $BUNDLE"
  echo "From Windows (PowerShell):"
  echo "  scp c:\\sw_study\\finalpjt\\ai\\xception_finetune_bundle.tgz sk4team@58.127.241.84:~/"
  exit 1
fi

mkdir -p "$ROOT"
cd "$ROOT"
tar xzf "$BUNDLE"
chmod +x scripts/infer/run_xception_finetune_staged.sh 2>/dev/null || true
chmod +x scripts/deploy/install_xception_finetune_bundle.sh 2>/dev/null || true
# Windows tarball may ship CRLF; bash rejects e.g. set -euo pipefail\r
find scripts -name '*.sh' -exec sed -i 's/\r$//' {} +

echo "Installed into $ROOT:"
ls -la scripts/download/data/prepare_xception_finetune_train.py
ls -la scripts/download/data/s3_pull_xception_finetune_train.py
ls -la scripts/infer/run_xception_finetune_staged.sh
ls -la scripts/infer/xception_finetune_data.py
ls -la scripts/infer/video_xception_finetune.py
ls -la docs/XCEPTION_FINETUNE_TRAIN_MANIFESTS.md
echo ""
echo "Next:"
echo "  cd $ROOT && source .venv/bin/activate && unset AWS_PROFILE"
echo "  python3 scripts/download/data/prepare_xception_finetune_train.py --stage stage1 --write-docs --upload-s3"
echo "  python3 scripts/download/data/prepare_xception_finetune_train.py --stage stage2 --write-docs --upload-s3"
echo "  bash scripts/infer/run_xception_finetune_staged.sh"
