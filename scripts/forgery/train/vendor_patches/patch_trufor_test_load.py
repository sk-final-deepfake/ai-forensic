#!/usr/bin/env python3
"""Patch vendor TruFor test.py torch.load for PyTorch 2.6+ / older versions."""
from __future__ import annotations

from pathlib import Path

TARGET = Path("vendor/TruFor/TruFor_train_test/test.py")

NEW_BLOCK = """print('=> loading model from {}'.format(model_state_file))
_map_loc = device if isinstance(device, str) else str(device)
try:
    checkpoint = torch.load(model_state_file, map_location=_map_loc, weights_only=False)
except TypeError:
    checkpoint = torch.load(model_state_file, map_location=_map_loc)
print("Epoch: {}".format(checkpoint['epoch']))"""

PATTERNS = [
    "checkpoint = torch.load(model_state_file, map_location=torch.device(device))",
    "checkpoint = torch.load(model_state_file, map_location=torch.device(device), weights_only=False)",
    "checkpoint = torch.load(model_state_file, map_location=torch.device(device, weights_only=False))",
]


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"missing {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    if "_map_loc = device" in text and "weights_only=False" in text:
        print("already patched:", TARGET)
        return

    bak = TARGET.with_suffix(".py.bak")
    if not bak.exists():
        bak.write_text(text, encoding="utf-8")
        print("backup:", bak)

    for old in PATTERNS:
        if old in text:
            # replace load line + following epoch print as one block if possible
            needle_single = "print('=> loading model from {}'.format(model_state_file))"
            needle_double = 'print("=> loading model from {}".format(model_state_file))'
            start = text.find(needle_single)
            if start == -1:
                start = text.find(needle_double)
            end_marker = 'print("Epoch: {}".format(checkpoint[\'epoch\']))'
            end = text.find(end_marker)
            if start != -1 and end != -1:
                end = end + len(end_marker)
                text = text[:start] + NEW_BLOCK + text[end:]
            else:
                text = text.replace(old, NEW_BLOCK.split("\n")[2], 1)  # fallback
            TARGET.write_text(text, encoding="utf-8")
            print("patched:", TARGET)
            return

    raise SystemExit("no matching torch.load pattern — edit test.py manually")


if __name__ == "__main__":
    main()
