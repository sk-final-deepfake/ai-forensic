#!/usr/bin/env python3
"""Fix mistaken PyTorch 2.6 patch: weights_only belongs on torch.load, not torch.device.

Broken:  torch.load(..., map_location=torch.device('cpu', weights_only=False))
Fixed:   torch.load(..., map_location=torch.device('cpu'), weights_only=False)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("vendor/TruFor/TruFor_train_test")
PAT = re.compile(r"torch\.device\(([^,)]+),\s*weights_only=False\)")


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not PAT.search(text):
        return False
    new = PAT.sub(r"torch.device(\1), weights_only=False", text)
    if new == text:
        return False
    bak = path.with_suffix(path.suffix + ".bak_torch_load")
    if not bak.exists():
        bak.write_text(text, encoding="utf-8")
    path.write_text(new, encoding="utf-8")
    print("patched:", path)
    return True


def main() -> None:
    root = ROOT if len(sys.argv) < 2 else Path(sys.argv[1])
    if not root.is_dir():
        raise SystemExit(f"missing {root}")
    n = sum(patch_file(p) for p in root.rglob("*.py"))
    print(f"done: {n} file(s)")


if __name__ == "__main__":
    main()
