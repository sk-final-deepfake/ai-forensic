#!/bin/bash
# Remove Kaggle API setup and DFDC download cache from forenShield-ai GPU workstation.
set -e

ROOT="${1:-$HOME/forenShield-ai}"

echo "Removing Kaggle credentials..."
rm -f "$HOME/.kaggle/access_token" "$HOME/.kaggle/kaggle.json"
rmdir "$HOME/.kaggle" 2>/dev/null || true

echo "Removing DFDC download cache (if any)..."
rm -rf "$ROOT/data/raw/benchmark-downloads/dfdc"
rm -rf "$ROOT/data/test/video/dfdc"

if [ -d "$ROOT/.venv" ]; then
  echo "Uninstalling kaggle package from venv..."
  "$ROOT/.venv/bin/pip" uninstall -y kaggle kagglesdk jupytext 2>/dev/null || true
fi

echo "Done. gdown is kept for ForgeryNet / FakeAVCeleb downloads."
