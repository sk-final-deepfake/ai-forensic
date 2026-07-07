#!/usr/bin/env python3
"""Download DFDC-style video subset to the GPU workstation.

Recommended (no Kaggle rules UI):
  --source hf
  pip install huggingface_hub datasets
  huggingface-cli login
  Accept dataset terms once on:
    https://huggingface.co/datasets/belkhir-nacim/deepfake-videos

Legacy Kaggle path (--source kaggle):
  Kaggle API cannot auto-accept competition rules (official limitation).
  You must click accept in the browser once. Also, train_sample_videos.zip
  is often missing from the competition file list (404).

Stores extracted/cache videos under:
  <root>/data/raw/benchmark-downloads/dfdc/

Benchmark copy:
  <root>/data/test/video/dfdc/
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DFDC_COMPETITION = "deepfake-detection-challenge"
DFDC_SAMPLE_ZIP = "train_sample_videos.zip"
HF_DATASET = "belkhir-nacim/deepfake-videos"
# belkhir unified repo has no "DFDC" rows; use DFD / SDFVD2.0 / HIDF instead.
HF_DEFAULT_SOURCES = ["DFD", "SDFVD2.0", "HIDF", "UADFV"]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, text=True)


def write_manifest(out_dir: Path, entries: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return path


def copy_sample(src: Path, dst_dir: Path, name: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / name
    if dst.exists():
        return dst
    shutil.copy2(src, dst)
    return dst


def kaggle_download_dfdc(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / DFDC_SAMPLE_ZIP
    bundle_zip = cache_dir / f"{DFDC_COMPETITION}.zip"

    if zip_path.exists() and zip_path.stat().st_size > 0:
        return zip_path

    print("listing DFDC competition files...")
    listed = subprocess.run(
        ["kaggle", "competitions", "files", "-c", DFDC_COMPETITION],
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0:
        raise RuntimeError(
            "Cannot list DFDC files. Join the competition and accept rules:\n"
            "  https://www.kaggle.com/c/deepfake-detection-challenge/rules\n"
            f"kaggle: {listed.stderr.strip() or listed.stdout.strip()}"
        )
    print(listed.stdout)

    single = run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            DFDC_COMPETITION,
            "-f",
            DFDC_SAMPLE_ZIP,
            "-p",
            str(cache_dir),
        ]
    )
    if single.returncode == 0 and zip_path.exists():
        return zip_path

    print("single-file download failed; trying full competition bundle...")
    full = run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            DFDC_COMPETITION,
            "-p",
            str(cache_dir),
        ]
    )
    if full.returncode != 0:
        raise RuntimeError(
            "DFDC download failed. Usually competition rules are not accepted.\n"
            "1) https://www.kaggle.com/c/deepfake-detection-challenge/rules\n"
            "2) Click 'I Understand and Accept'\n"
            "3) Retry"
        )

    if zip_path.exists():
        return zip_path

    if bundle_zip.exists():
        print("extract sample zip from bundle:", bundle_zip)
        with zipfile.ZipFile(bundle_zip) as zf:
            candidates = [
                n
                for n in zf.namelist()
                if n.endswith(DFDC_SAMPLE_ZIP) or n == DFDC_SAMPLE_ZIP
            ]
            if not candidates:
                raise RuntimeError(f"{DFDC_SAMPLE_ZIP} not found inside {bundle_zip}")
            zf.extract(candidates[0], path=cache_dir)
            inner = cache_dir / candidates[0]
            if inner.exists() and inner != zip_path:
                if inner.is_dir():
                    nested = inner / DFDC_SAMPLE_ZIP
                    if nested.exists():
                        shutil.move(str(nested), str(zip_path))
                else:
                    shutil.move(str(inner), str(zip_path))

    if not zip_path.exists():
        raise RuntimeError(f"DFDC download finished but {zip_path} is missing")

    return zip_path


def extract_dfdc(zip_path: Path, cache_dir: Path) -> Path:
    extract_dir = cache_dir / "train_sample_videos"
    meta_path = extract_dir / "metadata.json"
    if meta_path.exists():
        return extract_dir

    print("extract:", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found under {extract_dir}")

    return extract_dir


def load_metadata_rows(extract_dir: Path) -> list[tuple[str, str]]:
    meta_path = extract_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if isinstance(meta, list):
        rows: list[tuple[str, str]] = []
        for item in meta:
            filename = str(item.get("filename", ""))
            label = str(item.get("label", "")).upper()
            if filename and (extract_dir / filename).exists():
                rows.append((filename, label))
        return rows

    rows = []
    for filename, info in meta.items():
        label = str(info.get("label", "")).upper()
        if (extract_dir / filename).exists():
            rows.append((filename, label))
    return rows


def pick_rows(
    rows: list[tuple[str, str]],
    *,
    target: int,
    seed: int,
    balanced: bool,
    fake_only: bool,
    real_only: bool,
) -> list[tuple[str, str]]:
    if fake_only:
        rows = [r for r in rows if r[1] == "FAKE"]
    if real_only:
        rows = [r for r in rows if r[1] == "REAL"]

    if not rows:
        raise RuntimeError("no DFDC videos matched filters")

    rng = random.Random(seed)

    if balanced:
        fake_rows = [r for r in rows if r[1] == "FAKE"]
        real_rows = [r for r in rows if r[1] == "REAL"]
        half = target // 2
        need_fake = half + (target % 2)
        need_real = half
        if len(fake_rows) < need_fake or len(real_rows) < need_real:
            raise RuntimeError(
                f"balanced sample needs {need_fake} fake + {need_real} real; "
                f"have {len(fake_rows)} fake, {len(real_rows)} real"
            )
        picked = rng.sample(fake_rows, need_fake) + rng.sample(real_rows, need_real)
        rng.shuffle(picked)
        return picked

    if len(rows) <= target:
        return rows
    return rng.sample(rows, target)


def _label_from_hf_row(row: dict) -> str:
    for key in ("label", "is_fake", "fake", "target"):
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, bool):
            return "fake" if value else "real"
        text = str(value).strip().upper()
        if text in {"FAKE", "1", "TRUE", "SPOOF"}:
            return "fake"
        if text in {"REAL", "0", "FALSE", "BONAFIDE"}:
            return "real"
    source = str(row.get("dataset_source", "")).lower()
    if "fake" in source and "real" not in source:
        return "fake"
    return "unknown"


def download_via_hf(
    cache_dir: Path,
    out_dir: Path,
    *,
    target: int,
    seed: int,
    balanced: bool,
    fake_only: bool,
    real_only: bool,
    full_only: bool,
    hf_sources: list[str],
) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("pip install datasets huggingface_hub") from exc

    staging = cache_dir / "hf_staging"
    staging.mkdir(parents=True, exist_ok=True)

    allowed = {s.upper() for s in hf_sources}
    print(f"streaming HF dataset: {HF_DATASET}")
    print(f"hf sources filter: {', '.join(hf_sources)}")
    ds = load_dataset(HF_DATASET, split="train", streaming=True)

    rng = random.Random(seed)
    need_fake = target
    need_real = 0
    if balanced:
        half = target // 2
        need_fake = half + (target % 2)
        need_real = half
    if real_only:
        need_fake, need_real = 0, target
    if fake_only:
        need_fake, need_real = target, 0

    collected: list[dict] = []
    scanned = 0
    for row in ds:
        scanned += 1
        source = str(row.get("dataset_source", ""))
        if source.upper() not in allowed:
            continue

        label = _label_from_hf_row(row)
        if label == "unknown":
            continue
        if fake_only and label != "fake":
            continue
        if real_only and label != "real":
            continue
        if balanced:
            have_fake = sum(1 for e in collected if e["label"] == "fake")
            have_real = sum(1 for e in collected if e["label"] == "real")
            if label == "fake" and have_fake >= need_fake:
                continue
            if label == "real" and have_real >= need_real:
                continue

        video = row.get("video")
        if video is None:
            continue

        idx = len(collected) + 1
        out_mp4 = staging / f"dfdc_hf_{idx:03d}.mp4"
        if not out_mp4.exists():
            if hasattr(video, "save"):
                video.save(str(out_mp4))
            else:
                path = getattr(video, "path", None)
                if not path:
                    continue
                shutil.copy2(path, out_mp4)

        collected.append(
            {
                "file": out_mp4.name,
                "dataset": "dfdc",
                "label": label,
                "source": source,
                "hf_path": str(row.get("path", "")),
            }
        )
        if len(collected) >= target:
            break
        if scanned % 500 == 0:
            print(f"  scanned={scanned}, collected={len(collected)}")

    if not collected:
        raise RuntimeError(
            "No videos matched HF source filter.\n"
            f"1) Open https://huggingface.co/datasets/{HF_DATASET}\n"
            "2) Log in (browser) and click the yellow access box\n"
            "3) huggingface-cli login / hf auth login on GPU (same account)\n"
            f"4) Retry with e.g. --hf-source DFD or --hf-source SDFVD2.0"
        )

    if full_only:
        print(f"HF DFDC cache: {len(collected)} videos -> {staging}")
        return collected

    entries: list[dict] = []
    for i, item in enumerate(collected, start=1):
        src = staging / item["file"]
        safe = f"dfdc_{i:03d}_{src.name}"
        dst = copy_sample(src, out_dir, safe)
        entries.append({**item, "file": dst.name})

    manifest = write_manifest(out_dir, entries)
    print(f"DFDC HF benchmark: {len(entries)} videos -> {out_dir} ({manifest})")
    return entries


def sample_to_benchmark(
    extract_dir: Path,
    out_dir: Path,
    *,
    target: int,
    seed: int,
    balanced: bool,
    fake_only: bool,
    real_only: bool,
) -> list[dict]:
    rows = load_metadata_rows(extract_dir)
    picked = pick_rows(
        rows,
        target=target,
        seed=seed,
        balanced=balanced,
        fake_only=fake_only,
        real_only=real_only,
    )

    entries: list[dict] = []
    for i, (filename, label) in enumerate(picked, start=1):
        src = extract_dir / filename
        safe = f"dfdc_{i:03d}_{Path(filename).name}"
        dst = copy_sample(src, out_dir, safe)
        entries.append(
            {
                "file": dst.name,
                "dataset": "dfdc",
                "label": label.lower(),
                "source": filename,
            }
        )

    manifest = write_manifest(out_dir, entries)
    print(f"DFDC benchmark: {len(entries)} videos -> {out_dir} ({manifest})")
    return entries


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".", help="forenShield-ai root")
    p.add_argument(
        "--cache-dir",
        default=None,
        help="download cache (default: <root>/data/raw/benchmark-downloads/dfdc)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="benchmark copy dir (default: <root>/data/test/video/dfdc)",
    )
    p.add_argument("--target", type=int, default=50, help="videos to copy for benchmark")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--balanced",
        action="store_true",
        help="sample half REAL + half FAKE (e.g. 25+25 for target=50)",
    )
    p.add_argument("--fake-only", action="store_true", help="sample only FAKE videos")
    p.add_argument("--real-only", action="store_true", help="sample only REAL videos")
    p.add_argument(
        "--full-only",
        action="store_true",
        help="download + extract only; skip benchmark copy",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="use existing zip/extract under cache-dir (kaggle only)",
    )
    p.add_argument(
        "--source",
        choices=["hf", "kaggle", "auto"],
        default="hf",
        help="download source (default: hf; avoids Kaggle rules UI)",
    )
    p.add_argument(
        "--hf-source",
        default=",".join(HF_DEFAULT_SOURCES),
        help="HF dataset_source filter (default: DFD,SDFVD2.0,HIDF,UADFV)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.fake_only and args.real_only:
        raise SystemExit("choose only one of --fake-only / --real-only")
    if args.balanced and (args.fake_only or args.real_only):
        raise SystemExit("--balanced cannot be used with --fake-only/--real-only")

    root = Path(args.root).resolve()
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else root / "data/raw/benchmark-downloads/dfdc"
    )
    out_dir = Path(args.out_dir) if args.out_dir else root / "data/test/video/dfdc"
    hf_sources = [s.strip() for s in args.hf_source.split(",") if s.strip()]

    if args.source in {"hf", "auto"}:
        try:
            download_via_hf(
                cache_dir,
                out_dir,
                target=args.target,
                seed=args.seed,
                balanced=args.balanced,
                fake_only=args.fake_only,
                real_only=args.real_only,
                full_only=args.full_only,
                hf_sources=hf_sources,
            )
            return
        except Exception as exc:
            if args.source == "hf":
                raise
            print(f"HF source failed, falling back to kaggle: {exc}", file=sys.stderr)

    if args.skip_download:
        zip_path = cache_dir / DFDC_SAMPLE_ZIP
        if not zip_path.exists():
            raise FileNotFoundError(f"missing cache zip: {zip_path}")
    else:
        zip_path = kaggle_download_dfdc(cache_dir)

    extract_dir = extract_dfdc(zip_path, cache_dir)
    rows = load_metadata_rows(extract_dir)
    fake_n = sum(1 for _, label in rows if label == "FAKE")
    real_n = sum(1 for _, label in rows if label == "REAL")
    print(
        json.dumps(
            {
                "cache_dir": str(cache_dir),
                "extract_dir": str(extract_dir),
                "zip_path": str(zip_path),
                "videos_total": len(rows),
                "fake": fake_n,
                "real": real_n,
            },
            indent=2,
        )
    )

    if args.full_only:
        print("full-only: kept extracted DFDC under", extract_dir)
        return

    sample_to_benchmark(
        extract_dir,
        out_dir,
        target=args.target,
        seed=args.seed,
        balanced=args.balanced,
        fake_only=args.fake_only,
        real_only=args.real_only,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
