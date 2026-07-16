#!/usr/bin/env python3
"""Ensure vendor TruFor test.py always writes a real localization ``map``.

Official TruFor already saves map/conf/score via np.savez. This patch:
1. Softens skip-if-exists so stale/empty NPZs are overwritten.
2. Guarantees ``map`` is present and 2D before save (fail loud otherwise).

Run from forgery root (where vendor/TruFor lives):
  python scripts/forgery/train/vendor_patches/patch_trufor_npz_save_map.py
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path("vendor/TruFor/TruFor_train_test/test.py")

MARKER = "# forenshield: ensure localization map saved"

ENSURE_BLOCK = '''
                    # forenshield: ensure localization map saved
                    if 'map' not in out_dict or out_dict['map'] is None:
                        raise RuntimeError('TruFor output missing localization map')
                    _m = np.asarray(out_dict['map'])
                    if _m.ndim != 2 or _m.size < 4:
                        raise RuntimeError(f'TruFor map has invalid shape: {_m.shape}')
'''


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"missing {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        print("already patched:", TARGET)
        return

    bak = TARGET.with_suffix(".py.bak_npz_map")
    if not bak.exists():
        bak.write_text(text, encoding="utf-8")
        print("backup:", bak)

    # Prefer re-running inference even if npz exists (avoids stale empty maps).
    old_skip = "if not (os.path.isfile(filename_out)):"
    new_skip = "if True:  # forenshield: always rewrite npz (was: skip if exists)"
    if old_skip in text:
        text = text.replace(old_skip, new_skip, 1)
        print("patched skip-if-exists → always rewrite")

    needle = "np.savez(filename_out, **out_dict)"
    if needle not in text:
        raise SystemExit("np.savez(filename_out, **out_dict) not found — edit manually")

    if MARKER not in text:
        text = text.replace(needle, ENSURE_BLOCK + "\n                    " + needle, 1)

    TARGET.write_text(text, encoding="utf-8")
    print("patched:", TARGET)


if __name__ == "__main__":
    main()
