#!/usr/bin/env bash
# Download backbone weights for TimeSformer (HF cache) and Video Swin (torchvision + optional .pth).
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"

pip install -U transformers torchvision timm

mkdir -p models/test/video/timesformer/v1.0.0
mkdir -p models/test/video/video-swin/v1.0.0

echo "==> TimeSformer (HuggingFace)"
python3 - <<'PY'
from transformers import TimesformerModel
pid = "facebook/timesformer-base-finetuned-k400"
TimesformerModel.from_pretrained(pid)
print("ok:", pid)
PY

echo "==> Video Swin3D-T (torchvision)"
python3 - <<'PY'
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
m = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
print("ok: swin3d_t", sum(p.numel() for p in m.parameters()))
PY

SWIN_PTH="models/test/video/video-swin/v1.0.0/swin_tiny_k400.pth"
if [[ ! -f "$SWIN_PTH" ]] || [[ "$(stat -c%s "$SWIN_PTH" 2>/dev/null || echo 0)" -lt 50000000 ]]; then
  echo "==> Video Swin .pth backup"
  wget -O "$SWIN_PTH" \
    https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth
fi
ls -lh "$SWIN_PTH"

echo ""
echo "next:"
echo "  MODEL=timesformer bash scripts/infer/run_video_transformer_celebdf_benchmark.sh"
echo "  MODEL=video-swin  bash scripts/infer/run_video_transformer_celebdf_benchmark.sh"
