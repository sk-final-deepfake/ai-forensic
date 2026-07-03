#!/usr/bin/env python3
"""Merge a partial TruFor fine-tune checkpoint into a full infer-ready weights file.

Training config (trufor_forgery_video.yaml) uses MODULES: NP++, backbone, loc_head only.
Saved best.pth.tar therefore lacks decode_head_conf and detection keys required by
vendor test.py (-exp trufor_ph3).

Infer/eval: start from baseline trufor.pth.tar and overlay tuned keys.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _load_checkpoint(path: Path) -> dict:
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(obj)}")
    return obj


def _state_dict(checkpoint: dict) -> dict:
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge partial TruFor tune ckpt for infer")
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Full baseline weights (e.g. models/test/spatial/trufor/v1.0.0/trufor.pth.tar)",
    )
    parser.add_argument(
        "--tuned",
        type=Path,
        required=True,
        help="Fine-tuned partial checkpoint (e.g. weights/forgery-.../best.pth.tar)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for merged infer weights",
    )
    args = parser.parse_args()

    base_ckpt = _load_checkpoint(args.base.resolve())
    tuned_sd = _state_dict(_load_checkpoint(args.tuned.resolve()))
    base_sd = _state_dict(base_ckpt)

    merged_sd = dict(base_sd)
    updated, added, skipped = 0, 0, 0
    for key, val in tuned_sd.items():
        if key not in merged_sd:
            merged_sd[key] = val
            added += 1
            continue
        if merged_sd[key].shape != val.shape:
            skipped += 1
            print(f"skip shape mismatch: {key} base={tuple(merged_sd[key].shape)} tuned={tuple(val.shape)}")
            continue
        merged_sd[key] = val
        updated += 1

    out_ckpt = dict(base_ckpt)
    out_ckpt["state_dict"] = merged_sd

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, str(args.out.resolve()))

    only_in_tuned = sorted(set(tuned_sd) - set(base_sd))
    still_missing = sorted(
        k
        for k in base_sd
        if k.startswith(("decode_head_conf.", "detection.")) and k not in merged_sd
    )
    print(f"base keys: {len(base_sd)}")
    print(f"tuned keys: {len(tuned_sd)}")
    print(f"updated: {updated}, added: {added}, skipped (shape): {skipped}")
    print(f"merged keys: {len(merged_sd)}")
    print(f"checkpoint keys preserved: {sorted(k for k in out_ckpt if k != 'state_dict')}")
    print(f"written: {args.out.resolve()}")
    if only_in_tuned:
        print(f"keys only in tuned ({len(only_in_tuned)}): {only_in_tuned[:8]}{'...' if len(only_in_tuned) > 8 else ''}")
    if still_missing:
        print(f"WARNING: still missing infer keys: {still_missing[:5]}")


if __name__ == "__main__":
    main()
