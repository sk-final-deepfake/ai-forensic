#!/usr/bin/env python3
"""Delete migrated deepfake S3 sources only when the destination copy exists.

Safety rules:
- Same bucket + explicit src/dst pairs from s3_reorganize_deepfake_layout.sh only.
- Per object: delete src key ONLY if dst key exists with identical ContentLength.
- Never touches forgery paths or operational cases/{caseKey}/ uploads.
- Default dry-run; set APPLY=1 to delete.

Usage:
  source ~/forenShield-ai/config/env.local && unset AWS_PROFILE
  python3 scripts/upload/s3_prune_migrated_deepfake_sources.py
  APPLY=1 PHASE=evidence python3 scripts/upload/s3_prune_migrated_deepfake_sources.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterator


EVIDENCE_BUCKET = os.getenv("S3_EVIDENCE_BUCKET", "forenshield-evidence-877044078824")
MODELS_BUCKET = os.getenv("S3_MODELS_BUCKET", "forenshield-models-877044078824")
APPLY = os.getenv("APPLY", "0") == "1"
PHASE = os.getenv("PHASE", "all")


@dataclass(frozen=True)
class PrefixPair:
    bucket: str
    src_prefix: str
    dst_prefix: str
    tag: str


def _norm(prefix: str) -> str:
    return prefix.strip("/")


def _aws(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["aws", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _iter_keys(bucket: str, prefix: str) -> Iterator[str]:
    prefix = _norm(prefix)
    token = ""
    while True:
        cmd = [
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            f"{prefix}/",
            "--output",
            "json",
        ]
        if token:
            cmd.extend(["--starting-token", token])
        proc = _aws(*cmd)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            if "NoSuchBucket" in err or "AccessDenied" in err:
                raise RuntimeError(err)
            return
        payload = json.loads(proc.stdout or "{}")
        for row in payload.get("Contents") or []:
            key = row.get("Key")
            if key:
                yield key
        if not payload.get("IsTruncated"):
            break
        token = payload.get("NextContinuationToken") or ""
        if not token:
            break


def _head_size(bucket: str, key: str) -> int | None:
    proc = _aws("s3api", "head-object", "--bucket", bucket, "--key", key, "--output", "json")
    if proc.returncode != 0:
        return None
    try:
        return int(json.loads(proc.stdout)["ContentLength"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _delete_key(bucket: str, key: str) -> bool:
    proc = _aws("s3api", "delete-object", "--bucket", bucket, "--key", key)
    return proc.returncode == 0


def evidence_pairs() -> list[PrefixPair]:
    pairs = [
        PrefixPair(EVIDENCE_BUCKET, "cases/test/video-benchmark-datasets/celebdf", "deepfake/datasets/bench/celebdf", "bench-dataset"),
        PrefixPair(EVIDENCE_BUCKET, "cases/test/video-benchmark-datasets/ffpp_vox", "deepfake/datasets/bench/ffpp_vox", "bench-dataset"),
        PrefixPair(EVIDENCE_BUCKET, "cases/train/video/xception", "deepfake/datasets/train/video/xception", "train"),
        PrefixPair(EVIDENCE_BUCKET, "cases/test/youtube-shorts-adhoc", "deepfake/datasets/field/youtube-shorts", "field"),
    ]
    for model in ("xception", "timesformer", "videomae", "video-swin", "convnext", "raft", "gmflow", "efficientnetb4"):
        pairs.append(
            PrefixPair(
                EVIDENCE_BUCKET,
                f"cases/test/video-benchmark-datasets/{model}",
                f"deepfake/results/infer/{model}",
                "infer",
            )
        )
    pairs.append(
        PrefixPair(
            EVIDENCE_BUCKET,
            "cases/test/video-benchmark-datasets/PWC-Net",
            "deepfake/archive/legacy-benchmarks/pwcnet",
            "archive",
        )
    )
    for legacy in (
        "video-xception-benchmark",
        "video-videomae-benchmark",
        "video-videomae-celebdf-benchmark",
        "video-timesformer-celebdf-benchmark",
        "video-swin-celebdf-benchmark",
        "video-convnext-celebdf-benchmark",
        "video-optical-flow-benchmark",
        "video-raft-ffpp-vox-benchmark",
    ):
        pairs.append(
            PrefixPair(
                EVIDENCE_BUCKET,
                f"cases/test/{legacy}",
                f"deepfake/archive/legacy-benchmarks/{legacy}",
                "legacy",
            )
        )
    pairs.append(
        PrefixPair(
            EVIDENCE_BUCKET,
            "cases/test/test-sine",
            "deepfake/archive/legacy-experiments/test-sine",
            "archive",
        )
    )
    return pairs


def models_pairs() -> list[PrefixPair]:
    pairs: list[PrefixPair] = []
    for model in ("xception", "timesformer"):
        pairs.append(
            PrefixPair(MODELS_BUCKET, f"video/{model}", f"deepfake/deploy/video/{model}", "deploy")
        )
    pairs.append(
        PrefixPair(MODELS_BUCKET, "video/gmflow", "deepfake/deploy/video/optical/gmflow", "deploy")
    )
    for model in ("convnext", "videomae", "video-swin"):
        pairs.append(
            PrefixPair(MODELS_BUCKET, f"video/{model}", f"deepfake/bench/video/{model}", "bench")
        )
    for root, archive in (
        ("v1.0", "deepfake/archive/root-v1.0"),
        ("v1.1", "deepfake/archive/root-v1.1"),
        ("test", "deepfake/archive/root-test"),
        ("test-sets", "deepfake/archive/root-test-sets"),
    ):
        pairs.append(PrefixPair(MODELS_BUCKET, root, archive, "archive"))
    return pairs


def prune_pair(pair: PrefixPair) -> tuple[int, int, int]:
    src_root = _norm(pair.src_prefix)
    dst_root = _norm(pair.dst_prefix)
    print(f"\n--- [{pair.tag}] s3://{pair.bucket}/{src_root}/")
    print(f"    verify dst: s3://{pair.bucket}/{dst_root}/")

    checked = 0
    deleted = 0
    skipped = 0

    try:
        keys = list(_iter_keys(pair.bucket, src_root))
    except RuntimeError as exc:
        print(f"    (skip pair: {exc})")
        return 0, 0, 0

    if not keys:
        print("    (skip: source empty or missing)")
        return 0, 0, 0

    for src_key in keys:
        rel = src_key[len(src_root) :].lstrip("/")
        if not rel:
            continue
        dst_key = f"{dst_root}/{rel}"
        checked += 1
        src_size = _head_size(pair.bucket, src_key)
        dst_size = _head_size(pair.bucket, dst_key)
        if src_size is None or dst_size is None or src_size != dst_size:
            skipped += 1
            print(f"    SKIP (no matching dst): {src_key}")
            continue
        if APPLY:
            if _delete_key(pair.bucket, src_key):
                deleted += 1
                print(f"    DELETE: {src_key}")
            else:
                skipped += 1
                print(f"    SKIP (delete failed): {src_key}")
        else:
            deleted += 1
            print(f"    WOULD DELETE: {src_key}")

    print(f"    summary: checked={checked} delete={deleted} skipped={skipped}")
    return checked, deleted, skipped


def main() -> int:
    if APPLY:
        print("==> APPLY=1: deleting sources ONLY when dst size matches")
    else:
        print("==> dry-run (set APPLY=1 to delete)")

    pairs: list[PrefixPair] = []
    if PHASE in ("evidence", "all"):
        pairs.extend(evidence_pairs())
    if PHASE in ("models", "all"):
        pairs.extend(models_pairs())
    if not pairs:
        print(f"Unknown PHASE={PHASE}", file=sys.stderr)
        return 1

    total_checked = 0
    total_deleted = 0
    total_skipped = 0
    for pair in pairs:
        c, d, s = prune_pair(pair)
        total_checked += c
        total_deleted += d
        total_skipped += s

    print("\n========== TOTAL ==========")
    print(f"checked={total_checked} delete={total_deleted} skipped={total_skipped}")
    if not APPLY:
        print("No objects were deleted (dry-run).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
