#!/usr/bin/env bash
# Clone RAFT / GMFlow / PWC-Net repos and download pretrained weights.
#
# Usage (GPU):
#   cd ~/forenShield-ai && source .venv/bin/activate && unset AWS_PROFILE
#   sed -i 's/\r$//' scripts/download/models/download_optical_flow_models.sh
#   bash scripts/download/models/download_optical_flow_models.sh
set -euo pipefail

ROOT="${FORENSHIELD_AI_ROOT:-$HOME/forenShield-ai}"
cd "$ROOT"

VENDOR="$ROOT/vendor/optical-flow"
MODEL_DIR="$ROOT/models/test/video/optical-flow"
mkdir -p "$VENDOR" "$MODEL_DIR/raft" "$MODEL_DIR/gmflow" "$MODEL_DIR/pwcnet"

clone_if_missing() {
  local url="$1"
  local dest="$2"
  if [[ -d "$dest/.git" ]]; then
    echo "skip clone (exists): $dest"
  else
    git clone --depth 1 "$url" "$dest"
  fi
}

echo "==> clone repos"
clone_if_missing https://github.com/princeton-vl/RAFT.git "$VENDOR/RAFT"
clone_if_missing https://github.com/haofeixu/gmflow.git "$VENDOR/gmflow"
clone_if_missing https://github.com/NVlabs/PWC-Net.git "$VENDOR/PWC-Net"

echo "==> RAFT weights (official models.zip; direct .pth link is 404)"
RAFT_DIR="$MODEL_DIR/raft"
RAFT_W="$RAFT_DIR/raft-things.pth"
mkdir -p "$RAFT_DIR"
if [[ ! -f "$RAFT_W" ]] || [[ "$(stat -c%s "$RAFT_W" 2>/dev/null || echo 0)" -lt 5000000 ]]; then
  rm -f "$RAFT_W" "$RAFT_DIR/models.zip"
  wget -O "$RAFT_DIR/models.zip" "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip" \
    || curl -L -o "$RAFT_DIR/models.zip" "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
  unzip -o "$RAFT_DIR/models.zip" -d "$RAFT_DIR"
  if [[ -f "$RAFT_DIR/models/raft-things.pth" ]]; then
    mv "$RAFT_DIR/models/raft-things.pth" "$RAFT_W"
    rmdir "$RAFT_DIR/models" 2>/dev/null || true
  fi
  rm -f "$RAFT_DIR/models.zip"
fi

echo "==> GMFlow weights"
GM_DIR="$MODEL_DIR/gmflow"
if ! ls "$GM_DIR"/gmflow*.pth >/dev/null 2>&1; then
  pip install -q gdown
  ZIP="$GM_DIR/gmflow_pretrained.zip"
  gdown "https://drive.google.com/uc?id=1d5C5cgHIxWGsFR1vYs5XrQbbUiZl9TX2" -O "$ZIP"
  unzip -o "$ZIP" -d "$GM_DIR"
  rm -f "$ZIP"
fi

echo "==> PWC-Net weights (GitHub official; NVIDIA mirror often 404)"
PWC_DIR="$MODEL_DIR/pwcnet"
PWC_W="$PWC_DIR/pwc_net.pth.tar"
PWC_CHAIRS="$PWC_DIR/pwc_net_chairs.pth.tar"
PWC_GH_SINTEL="https://raw.githubusercontent.com/NVlabs/PWC-Net/master/PyTorch/pwc_net.pth.tar"
PWC_GH_CHAIRS="https://raw.githubusercontent.com/NVlabs/PWC-Net/master/PyTorch/pwc_net_chairs.pth.tar"

if [[ -f "$VENDOR/PWC-Net/PyTorch/pwc_net.pth.tar" && ! -f "$PWC_W" ]]; then
  cp "$VENDOR/PWC-Net/PyTorch/pwc_net.pth.tar" "$PWC_W"
  echo "copied from cloned repo: $PWC_W"
fi

if [[ ! -f "$PWC_W" ]]; then
  echo "download: $PWC_GH_SINTEL"
  wget -O "$PWC_W" "$PWC_GH_SINTEL" || curl -L -o "$PWC_W" "$PWC_GH_SINTEL"
fi

if [[ ! -f "$PWC_CHAIRS" ]]; then
  echo "download chairs fallback: $PWC_GH_CHAIRS"
  wget -O "$PWC_CHAIRS" "$PWC_GH_CHAIRS" || curl -L -o "$PWC_CHAIRS" "$PWC_GH_CHAIRS" || true
fi

if [[ ! -f "$PWC_W" && -f "$PWC_CHAIRS" ]]; then
  echo "WARN: sintel weights missing; using chairs weights as fallback"
  cp "$PWC_CHAIRS" "$PWC_W"
fi

if [[ ! -f "$PWC_W" ]]; then
  echo "ERROR: PWC-Net weights download failed"
  echo "  manual: wget -O $PWC_W $PWC_GH_SINTEL"
  exit 1
fi

echo "==> build PWC correlation extension (required once)"
CORR="$VENDOR/PWC-Net/PyTorch/external_packages/correlation-pytorch-master/correlation-pytorch"
if [[ -d "$CORR" ]]; then
  pip install -q cffi
  (cd "$CORR" && python3 setup.py build_ext --inplace 2>/dev/null || python3 setup.py install)
fi

echo ""
echo "DONE optical-flow models"
echo "  RAFT:    $RAFT_W"
echo "  GMFlow:  $(ls "$GM_DIR"/gmflow*.pth 2>/dev/null | head -1 || echo missing)"
echo "  PWC-Net: $PWC_W"
echo ""
echo "next:"
echo "  bash scripts/infer/run_optical_flow_celebdf_benchmark.sh"
