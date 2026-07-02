#!/usr/bin/env python3
"""Build expanded MVTamperBench hold-out (e.g. 500 = prior 200 + 300 new).

Excludes train video ids (forgery-gmflow-train-400) and optionally reuses the
fixed calibration-200 list from predictions.json.

Output layout matches spatial_mvtamperbench_benchmark.py:
  <out>/original/...
  <out>/tampered/<type>/...
  manifest.json
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from trufor_video_common import VIDEO_SUFFIXES, iter_videos


def video_id_from_rel(rel: str) -> str:
    return Path(rel).stem


def load_paths_from_predictions(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items") or data
    out: list[str] = []
    for x in items:
        rel = x.get("relative_path") or x.get("video_rel") or x.get("path")
        if rel:
            out.append(str(rel).replace("\\", "/"))
    return out


def collect_train_ids(train_root: Path) -> set[str]:
    ids: set[str] = set()
    if not train_root.exists():
        return ids
    for split in ("original", "tampered"):
        for p in iter_videos(train_root, split):
            ids.add(p.stem)
    return ids


def classify_rel(rel: str) -> tuple[str, str]:
    parts = Path(rel).parts
    if not parts:
        return "unknown", "unknown"
    if parts[0] == "original":
        return "real", "original"
    if parts[0] == "tampered" and len(parts) >= 2:
        return "fake", parts[1]
  # filename heuristics (flat pools)
    name = Path(rel).name.lower()
    if name.startswith("original_") or "/original/" in rel.replace("\\", "/"):
        return "real", "original"
    for tok in ("masking", "rotate", "substitution", "dropping", "repetition"):
        if tok in name or tok in rel.lower():
            return "fake", tok
    return "fake", "unknown"


def index_pool(pool_root: Path) -> dict[str, Path]:
    """Map relative path (posix) -> absolute file under pool_root."""
    idx: dict[str, Path] = {}
    for split in ("original", "tampered"):
        for p in iter_videos(pool_root, split):
            rel = p.relative_to(pool_root).as_posix()
            idx[rel] = p
    if idx:
        return idx
    # flat scan fallback
    for p in sorted(pool_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
            rel = p.relative_to(pool_root).as_posix()
            idx[rel] = p
    return idx


def symlink_or_copy(src: Path, dst: Path, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare mvtb hold-out benchmark folder")
    parser.add_argument(
        "--pool-root",
        type=Path,
        required=True,
        help="Full MVTamperBench video pool (original/ + tampered/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/pull/evidence/mvtamperbench-500-holdout"),
    )
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument(
        "--existing-data-root",
        type=Path,
        default=Path("data/pull/evidence/mvtamperbench-200-s3"),
        help="Data root for --existing-predictions videos",
    )
    parser.add_argument(
        "--existing-predictions",
        type=Path,
        default=None,
        help="Prior hold-out predictions.json (e.g. mvtb200) — kept as calibration subset",
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path("data/train/video/forgery-gmflow-train-400"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symlink", action="store_true", help="symlink videos (default: copy)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.total % 2 != 0:
        raise SystemExit("--total must be even (balanced real/fake)")

    pool = index_pool(args.pool_root)
    if not pool:
        raise SystemExit(f"No videos under pool-root: {args.pool_root}")

    train_ids = collect_train_ids(args.train_root)
    existing_rels: list[str] = []
    if args.existing_predictions and args.existing_predictions.exists():
        existing_rels = load_paths_from_predictions(args.existing_predictions)

    n_existing = len(existing_rels)
    n_need = args.total - n_existing
    if n_need < 0:
        raise SystemExit(f"--total {args.total} < existing {n_existing}")
    if n_need % 2 != 0:
        raise SystemExit(f"After reserving {n_existing} existing, remainder must be even")

    n_real_new = n_need // 2
    n_fake_new = n_need // 2
    n_real_exist = sum(1 for r in existing_rels if classify_rel(r)[0] == "real")
    n_fake_exist = len(existing_rels) - n_real_exist

    existing_ids = {video_id_from_rel(r) for r in existing_rels}

    candidates_real: list[str] = []
    candidates_fake: dict[str, list[str]] = defaultdict(list)
    for rel in pool:
        vid = video_id_from_rel(rel)
        if vid in train_ids or vid in existing_ids:
            continue
        label, bucket = classify_rel(rel)
        if label == "real":
            candidates_real.append(rel)
        else:
            candidates_fake[bucket].append(rel)

    rng = random.Random(args.seed)
    rng.shuffle(candidates_real)

    picked_new_fake: list[str] = []
    buckets = sorted(candidates_fake.keys())
    per_bucket = max(1, n_fake_new // max(1, len(buckets)))
    for b in buckets:
        rng.shuffle(candidates_fake[b])
        picked_new_fake.extend(candidates_fake[b][:per_bucket])
    rng.shuffle(picked_new_fake)
    picked_new_fake = picked_new_fake[:n_fake_new]
    if len(picked_new_fake) < n_fake_new:
        rest = [r for b in buckets for r in candidates_fake[b] if r not in picked_new_fake]
        rng.shuffle(rest)
        picked_new_fake.extend(rest[: n_fake_new - len(picked_new_fake)])

    picked_new_real = candidates_real[:n_real_new]
    if len(picked_new_real) < n_real_new:
        raise SystemExit(
            f"Not enough new real videos: need {n_real_new}, have {len(picked_new_real)} "
            f"(pool={args.pool_root}, train excluded={len(train_ids)})"
        )
    if len(picked_new_fake) < n_fake_new:
        raise SystemExit(
            f"Not enough new fake videos: need {n_fake_new}, have {len(picked_new_fake)}"
        )

    selected: list[dict] = []
    for rel in existing_rels:
        selected.append(
            {
                "relative_path": rel,
                "ground_truth_label": classify_rel(rel)[0],
                "subset": "calibration_200",
                "tamper_type": classify_rel(rel)[1],
            }
        )
    for rel in picked_new_real:
        selected.append(
            {
                "relative_path": rel,
                "ground_truth_label": "real",
                "subset": "ood_new",
                "tamper_type": "original",
            }
        )
    for rel in picked_new_fake:
        selected.append(
            {
                "relative_path": rel,
                "ground_truth_label": "fake",
                "subset": "ood_new",
                "tamper_type": classify_rel(rel)[1],
            }
        )

    # resolve paths: existing may point into mvtamperbench-200-s3 layout
    resolve_roots = [args.pool_root]
    if args.existing_predictions:
        resolve_roots.insert(0, args.existing_data_root)

    def resolve(rel: str) -> Path | None:
        if rel in pool:
            return pool[rel]
        for root in resolve_roots:
            cand = root / rel
            if cand.is_file():
                return cand
        return None

    missing = [s["relative_path"] for s in selected if resolve(s["relative_path"]) is None]
    if missing:
        raise SystemExit(f"Missing {len(missing)} videos in pool. First: {missing[:5]}")

    manifest = {
        "name": args.out_dir.name,
        "total": len(selected),
        "real": sum(1 for s in selected if s["ground_truth_label"] == "real"),
        "fake": sum(1 for s in selected if s["ground_truth_label"] == "fake"),
        "calibration_subset": n_existing,
        "ood_new_subset": n_need,
        "seed": args.seed,
        "pool_root": str(args.pool_root),
        "train_root_excluded": str(args.train_root),
        "items": selected,
        "tamper_type_counts": dict(Counter(s["tamper_type"] for s in selected if s["ground_truth_label"] == "fake")),
    }

    print(f"pool videos: {len(pool)}  train_ids excluded: {len(train_ids)}")
    print(f"selected: total={len(selected)}  calib={n_existing}  ood_new={n_need}")
    print(f"  real={manifest['real']} fake={manifest['fake']}")
    print(f"  fake types: {manifest['tamper_type_counts']}")

    if args.dry_run:
        print("dry-run — manifest only")
        print(json.dumps(manifest, ensure_ascii=False, indent=2)[:2000])
        return

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for s in selected:
        rel = s["relative_path"]
        src = resolve(rel)
        assert src is not None
        dst = args.out_dir / rel
        symlink_or_copy(src, dst, args.symlink)

    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("wrote", args.out_dir / "manifest.json")


if __name__ == "__main__":
    main()
