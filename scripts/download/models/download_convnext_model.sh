#!/usr/bin/env bash
# Download ConvNeXt ImageNet backbone via torchvision (DeepfakeBench has no convnext_best.pth).
#
# Usage:
#   cd ~/forenShield-ai && source .venv/bin/activate
#   sed -i 's/\r$//' scripts/download/models/download_convnext_model.sh
#   bash scripts/download/models/download_convnext_model.sh
#
# Optional env:
#   VARIANT=small|base   (default: small)
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

VARIANT="${VARIANT:-small}"
MODEL_DIR="models/test/video/convnext/v1.0.0"
mkdir -p "$MODEL_DIR"

pip install -U torch torchvision

echo "==> ConvNeXt-$VARIANT (torchvision IMAGENET1K_V1)"
python3 - <<PY
from pathlib import Path
import torch
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Small_Weights,
    convnext_base,
    convnext_small,
)

variant = "${VARIANT}"
out = Path("${MODEL_DIR}") / f"convnext_{variant}_imagenet_backbone.pth"
if variant == "small":
    model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
elif variant == "base":
    model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
else:
    raise SystemExit(f"unknown VARIANT={variant}")

torch.save(model.state_dict(), out)
print("saved:", out, "params:", sum(p.numel() for p in model.parameters()))
PY

echo ""
echo "next (fine-tune on FF++100 + Vox100, then Celeb-DF benchmark):"
echo "  bash scripts/infer/run_convnext_celebdf_benchmark.sh"
echo ""
echo "or step-by-step:"
echo "  python3 scripts/infer/video_convnext_finetune.py --root . --variant ${VARIANT}"
echo "  python3 scripts/infer/video_convnext_benchmark_infer.py --root . --variant ${VARIANT}"
