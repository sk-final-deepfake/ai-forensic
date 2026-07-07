#!/usr/bin/env python3
"""Fix TruFor builder_np_conf.py: weights_only must be on torch.load, not torch.device."""
from __future__ import annotations

from pathlib import Path

TARGET = Path("vendor/TruFor/TruFor_train_test/lib/models/cmx/builder_np_conf.py")

BROKEN = "map_location=torch.device('cpu', weights_only=False))"
FIXED = "map_location=torch.device('cpu'), weights_only=False)"

ALT_BROKEN = 'map_location=torch.device("cpu", weights_only=False))'
ALT_FIXED = 'map_location=torch.device("cpu"), weights_only=False)'


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"missing {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    if BROKEN not in text and ALT_BROKEN not in text:
        if "weights_only=False" in text and "torch.device('cpu'), weights_only=False)" in text:
            print("already patched:", TARGET)
            return
        raise SystemExit("no broken pattern — inspect builder_np_conf.py manually")

    bak = TARGET.with_suffix(".py.bak_np_load")
    if not bak.exists():
        bak.write_text(text, encoding="utf-8")
        print("backup:", bak)

    text = text.replace(BROKEN, FIXED).replace(ALT_BROKEN, ALT_FIXED)
    TARGET.write_text(text, encoding="utf-8")
    print("patched:", TARGET)


if __name__ == "__main__":
    main()
