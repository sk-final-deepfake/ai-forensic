#!/usr/bin/env python3
"""Sample fake deepfake videos from HuggingFace by duration buckets.

Dataset: belkhir-nacim/deepfake-videos
  - HF account login + click-through agreement (no email approval wait)
  - Streams metadata and downloads only matching mp4 files

Default buckets (50 total):
  20s–60s:   10
  60s–120s:  20
  120s–180s: 10
  180s–240s: 10

Example:
  pip install huggingface_hub pyarrow
  hf auth login

  python3 scripts/download/data/download_hf_fake_by_duration.py \\
    --out-dir data/test/video/hf-deepfake/fake
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

REPO_ID = "belkhir-nacim/deepfake-videos"

# Subsets with actual files in this HF repo (not external LINK-only metadata).
HOSTED_SOURCES = frozenset(
    {
        "DFD",
        "DeepfakeTIMIT",
        "GenVideo-100K",
        "HIDF",
        "SDFVD2.0",
        "UADFV",
    }
)

# Longer clips are more common here; GenVideo-100K is mostly a few seconds.
DEFAULT_SOURCES = ("DFD", "HIDF", "SDFVD2.0", "UADFV", "DeepfakeTIMIT", "GenVideo-100K")


@dataclass(frozen=True)
class DurationBucket:
    key: str
    label: str
    min_sec: float
    max_sec: float
    target: int
    min_inclusive: bool = True
    max_inclusive: bool = True

    def contains(self, duration_sec: float) -> bool:
        if self.min_inclusive:
            if duration_sec < self.min_sec:
                return False
        elif duration_sec <= self.min_sec:
            return False
        if self.max_inclusive:
            return duration_sec <= self.max_sec
        return duration_sec < self.max_sec


DEFAULT_BUCKETS = (
    DurationBucket("b1_20_60", "20s-60s", 20.0, 60.0, 10),
    DurationBucket("b2_60_120", "60s-120s", 60.0, 120.0, 20, min_inclusive=False),
    DurationBucket("b3_120_180", "120s-180s", 120.0, 180.0, 10, min_inclusive=False),
    DurationBucket("b4_180_240", "180s-240s", 180.0, 240.0, 10, min_inclusive=False),
)


def probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return 0.0
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def bucket_for_duration(duration_sec: float, buckets: tuple[DurationBucket, ...]) -> DurationBucket | None:
    for bucket in buckets:
        if bucket.contains(duration_sec):
            return bucket
    return None


def buckets_remaining(counts: dict[str, int], buckets: tuple[DurationBucket, ...]) -> bool:
    return any(counts[b.key] < b.target for b in buckets)


def total_target(buckets: tuple[DurationBucket, ...]) -> int:
    return sum(b.target for b in buckets)


def load_existing_manifest(out_dir: Path, buckets: tuple[DurationBucket, ...]) -> tuple[list[dict], dict[str, int]]:
    manifest_path = out_dir / "manifest.json"
    manifest: list[dict] = []
    counts = {b.key: 0 for b in buckets}

    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = []

    for entry in manifest:
        bucket_key = entry.get("bucket_key")
        file_name = entry.get("file")
        if bucket_key not in counts or not file_name:
            continue
        file_path = out_dir / file_name
        if file_path.is_file() and probe_duration(file_path) > 0:
            counts[bucket_key] += 1

    return manifest, counts


def write_manifest(out_dir: Path, manifest: list[dict]) -> Path:
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def resolve_hf_token(explicit: str | None) -> str | bool:
    if explicit:
        return explicit.strip()
    return True


def list_parquet_shards(token: str | bool) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset", token=token)
    shards = [name for name in files if name.endswith(".parquet")]
    if not shards:
        raise RuntimeError(f"no parquet metadata found in dataset repo: {REPO_ID}")
    return sorted(shards)


def iter_metadata_rows(
    token: str | bool,
    seed: int,
    sources: tuple[str, ...],
) -> Iterator[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    shards = list_parquet_shards(token)
    rng = random.Random(seed)
    rng.shuffle(shards)

    for source_name in sources:
        print(f"metadata pass: {source_name}", flush=True)
        for shard in shards:
            print(f"  loading parquet: {shard}", flush=True)
            local = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=shard,
                token=token,
            )
            parquet = pq.ParquetFile(local)
            batch_rows: list[dict] = []
            for batch in parquet.iter_batches(batch_size=2048):
                batch_rows.extend(batch.to_pylist())
            rng.shuffle(batch_rows)
            for row in batch_rows:
                if row.get("label") != "fake":
                    continue
                if row.get("dataset_source") != source_name:
                    continue
                yield row


def verify_dataset_access(token: str | bool) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError

    api = HfApi()
    try:
        who = api.whoami(token=token)
        name = who.get("name") or who.get("fullname") or "unknown"
        print("hf user:", name)
    except Exception as exc:
        print("warn: could not read HF identity:", exc, file=sys.stderr)

    try:
        api.dataset_info(REPO_ID, token=token)
    except GatedRepoError as exc:
        print(
            "ERROR: gated dataset access denied for this token/account.\n"
            f"  dataset: https://huggingface.co/datasets/{REPO_ID}\n"
            "  1) Run: hf auth whoami  (note the username)\n"
            "  2) Open the dataset URL in a browser logged in as THAT user\n"
            "  3) Click agree / accept, then: hf auth login --force\n"
            "  4) Or pass an explicit token: --hf-token hf_xxx",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def download_hf_video(hf_path: str, token: str | bool) -> Path | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

    try:
        local = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=hf_path,
            token=token,
        )
    except (EntryNotFoundError, HfHubHTTPError, OSError, ValueError):
        return None
    path = Path(local)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download fake deepfake videos from HF by duration buckets.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/test/video/hf-deepfake/fake",
        help="output directory for sampled mp4 files",
    )
    parser.add_argument(
        "--all-sources",
        action="store_true",
        help="scan all fake rows in metadata (default: HF-hosted subsets only)",
    )
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="comma-separated dataset_source scan order (default: DFD,HIDF,...)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="print scan progress every N probed candidates (0=off)",
    )
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=10_000,
        help="unused (kept for CLI compatibility)",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=0,
        help="stop after scanning N fake candidates (0 = no limit)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HF token (default: HF_TOKEN env or cached huggingface-cli login)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = DEFAULT_BUCKETS
    manifest, counts = load_existing_manifest(out_dir, buckets)
    target_total = total_target(buckets)

    if sum(counts.values()) >= target_total:
        print("already complete:", sum(counts.values()), "/", target_total)
        for bucket in buckets:
            print(f"  {bucket.label}: {counts[bucket.key]}/{bucket.target}")
        return

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("missing dependency: pip install huggingface_hub pyarrow")
        sys.exit(1)

    hosted_only = not args.all_sources
    hf_token = resolve_hf_token(args.hf_token)
    verify_dataset_access(hf_token)

    source_list = [s.strip() for s in args.sources.split(",") if s.strip()]
    if hosted_only:
        source_list = [s for s in source_list if s in HOSTED_SOURCES]
        if not source_list:
            print("ERROR: no valid --sources after filtering to HF-hosted subsets", file=sys.stderr)
            sys.exit(1)

    stream = iter_metadata_rows(hf_token, args.shuffle_seed, tuple(source_list))

    scanned = 0
    downloaded = sum(counts.values())

    print("repo:", REPO_ID)
    print("out:", out_dir.resolve())
    print("buckets:")
    for bucket in buckets:
        print(f"  {bucket.label}: {counts[bucket.key]}/{bucket.target}")
    if hosted_only:
        print("sources:", ", ".join(source_list))
    print(flush=True)

    skipped_short = 0
    for row in stream:
        if not buckets_remaining(counts, buckets):
            break

        source = row.get("dataset_source") or ""
        hf_path = row.get("hf_path") or ""
        if not hf_path.lower().endswith(".mp4"):
            continue

        scanned += 1
        if args.max_scan and scanned > args.max_scan:
            print("max-scan reached:", args.max_scan, flush=True)
            break

        if args.progress_every and scanned % args.progress_every == 0:
            print(
                f"scan: {scanned} probed, saved {downloaded}/{target_total}, "
                f"skipped_short {skipped_short}, current {source}",
                flush=True,
            )

        local = download_hf_video(hf_path, hf_token)
        if local is None:
            continue

        duration = probe_duration(local)
        bucket = bucket_for_duration(duration, buckets)
        if bucket is None:
            skipped_short += 1
            continue
        if counts[bucket.key] >= bucket.target:
            continue

        idx = counts[bucket.key] + 1
        safe_id = str(row.get("id") or Path(hf_path).stem).replace("/", "_")[:48]
        out_name = f"fake_{bucket.key}_{idx:03d}_{safe_id}.mp4"
        out_path = out_dir / out_name
        if not out_path.exists():
            shutil.copy2(local, out_path)

        counts[bucket.key] += 1
        downloaded += 1
        manifest.append(
            {
                "file": out_name,
                "bucket_key": bucket.key,
                "bucket_label": bucket.label,
                "duration_sec": round(duration, 2),
                "dataset_source": source,
                "hf_path": hf_path,
                "id": row.get("id"),
                "label": row.get("label"),
            }
        )
        write_manifest(out_dir, manifest)

        print(
            f"[{downloaded}/{target_total}] {bucket.label} "
            f"{duration:.1f}s {source} -> {out_name}"
        )

    manifest_path = write_manifest(out_dir, manifest)
    print()
    print("done:", downloaded, "/", target_total)
    for bucket in buckets:
        got = counts[bucket.key]
        print(f"  {bucket.label}: {got}/{bucket.target}", end="")
        if got < bucket.target:
            print("  (need more — re-run or use --all-sources / higher --max-scan)")
        else:
            print()
    print("manifest:", manifest_path.resolve())

    if downloaded < target_total:
        sys.exit(2)


if __name__ == "__main__":
    main()
