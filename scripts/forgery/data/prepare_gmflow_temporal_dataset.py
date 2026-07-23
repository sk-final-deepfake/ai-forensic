#!/usr/bin/env python3
"""Build GMFlow **temporal-only** train / val / test datasets (MVTB + CSVTED).

GMFlow targets frame-drop, duplication, insertion, cut-like edits — not spatial
masking or spatial-tampering. Layout matches existing forgery benchmarks:

  original/{category}/*.mp4
  tampered/{type}/{category}/*.mp4

Hold-out TEST 200 manifests are excluded from TRAIN/VAL sources.

Example (GPU):
  # 1) Train + val (15%% video-level split, no pair leakage)
  python3 forgery/scripts/data/prepare_gmflow_temporal_dataset.py train \\
    --train-stage ~/forenShield-ai/forgery/data/train/video/forgery-gmflow-train-temporal \\
    --val-stage ~/forenShield-ai/forgery/data/train/video/forgery-gmflow-val-temporal \\
    --mvtb-train-dir ~/forenShield-ai/forgery/data/train/video/forgery-gmflow-train-mvtb-1k \\
    --csvted-root ~/forenShield-ai/forgery/data/test/video/csvted \\
    --mvtb-test-manifest ~/forenShield-ai/forgery/data/pull/evidence/mvtamperbench-200-s3/manifest.json \\
    --csvted-test-manifest ~/forenShield-ai/forgery/data/pull/evidence/csvted-200-balanced/manifest.json \\
    --val-ratio 0.15 --seed 123 --fresh

  # 2) Test temporal-only (dev eval; mixed 200 stays separate for sign-off)
  python3 forgery/scripts/data/prepare_gmflow_temporal_dataset.py test-temporal \\
    --stage ~/forenShield-ai/forgery/data/pull/evidence/gmflow-test-temporal-200 \\
    --mvtb-test ~/forenShield-ai/forgery/data/pull/evidence/mvtamperbench-200-s3 \\
    --csvted-test ~/forenShield-ai/forgery/data/pull/evidence/csvted-200-balanced \\
    --fresh
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_csvted_balanced_200 as csvted_mod  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# GMFlow temporal whitelist / spatial blacklist
MVTB_TEMPORAL_DEFAULT = ("dropping", "repetition", "substitution")
# masking + rotate are not temporal-lane targets (spatial / geometric edit).
MVTB_SPATIAL_DEFAULT = ("masking", "rotate")
CSVTED_TEMPORAL_DEFAULT = (
    "frame-deletion",
    "frame-duplication",
    "frame-insertion",
    "eop-frame-deletion",
    "eop-frame-duplication",
    "eop-frame-insertion",
)
CSVTED_SPATIAL_DEFAULT = ("spatial-tampering",)

CLIP_GROUP_RE = re.compile(r"^(mvtb|csvted)_(\d+)_", re.I)


def list_videos(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out.append(p)
    return out


def infer_label(rel: Path) -> str | None:
    parts = rel.parts
    if not parts:
        return None
    if parts[0] == "original":
        return "real"
    if parts[0] == "tampered":
        return "fake"
    return None


def infer_tamper_type(rel: Path) -> str | None:
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "tampered":
        return parts[1].lower()
    return None


def clip_group_key(rel: Path) -> str:
    m = CLIP_GROUP_RE.match(rel.name)
    if m:
        return f"{m.group(1).lower()}_{m.group(2)}"
    return rel.stem.lower()


@dataclass
class VideoRow:
    rel: Path
    abs_path: Path
    label: str
    tamper_type: str | None
    group: str


def scan_stage(stage: Path) -> list[VideoRow]:
    rows: list[VideoRow] = []
    for mp4 in list_videos(stage):
        rel = mp4.relative_to(stage)
        label = infer_label(rel)
        if label is None:
            continue
        rows.append(
            VideoRow(
                rel=rel,
                abs_path=mp4,
                label=label,
                tamper_type=infer_tamper_type(rel),
                group=clip_group_key(rel),
            )
        )
    return rows


def dataset_stats(stage: Path) -> dict:
    rows = scan_stage(stage)
    tamper_types = Counter(r.tamper_type for r in rows if r.label == "fake")
    return {
        "stage": str(stage),
        "total_videos": len(rows),
        "real": sum(1 for r in rows if r.label == "real"),
        "fake": sum(1 for r in rows if r.label == "fake"),
        "tamper_types": dict(sorted(tamper_types.items())),
        "clip_groups": len({r.group for r in rows}),
    }


def write_manifest(stage: Path, meta: dict) -> None:
    meta = dict(meta)
    meta["stats"] = dataset_stats(stage)
    (stage / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_video(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def filter_copy_tree(
    src: Path,
    dst: Path,
    *,
    include_tamper: set[str],
    exclude_tamper: set[str],
) -> int:
    """Copy original/ and whitelisted tampered/{type}/. Returns file count."""
    n = 0
    orig = src / "original"
    if orig.is_dir():
        for mp4 in list_videos(orig):
            rel = mp4.relative_to(src)
            copy_video(mp4, dst / rel)
            n += 1

    tam_root = src / "tampered"
    if not tam_root.is_dir():
        return n

    for tdir in sorted(tam_root.iterdir()):
        if not tdir.is_dir():
            continue
        t = tdir.name.lower()
        if t in exclude_tamper:
            continue
        if include_tamper and t not in include_tamper:
            continue
        for mp4 in list_videos(tdir):
            rel = mp4.relative_to(src)
            copy_video(mp4, dst / rel)
            n += 1
    return n


def max_clip_index(stage: Path) -> int:
    mx = 0
    for p in list_videos(stage):
        m = CLIP_GROUP_RE.match(p.name)
        if m:
            mx = max(mx, int(m.group(2)))
    return mx


def load_csvted_exclude(manifest_path: Path) -> tuple[set[str], set[str]]:
    """Paths from TEST manifest to withhold from TRAIN.

    - ``pairs`` / ``extra_real``: block originals (appear as real in TEST).
    - ``pairs`` / ``extra_fake``: block specific tampered files only.
    - Originals behind ``extra_fake`` may remain in TRAIN with other tamper types
      (those originals are not shown as real in TEST).
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    originals: set[str] = set()
    tampered: set[str] = set()
    for row in data.get("pairs", []):
        if row.get("original_path"):
            originals.add(str(Path(row["original_path"]).resolve()))
        if row.get("tampered_path"):
            tampered.add(str(Path(row["tampered_path"]).resolve()))
    for row in data.get("extra_real", []):
        if row.get("original_path"):
            originals.add(str(Path(row["original_path"]).resolve()))
    for row in data.get("extra_fake", []):
        if row.get("tampered_path"):
            tampered.add(str(Path(row["tampered_path"]).resolve()))
    return originals, tampered


def filter_csvted_groups(
    groups: dict[str, csvted_mod.ClipGroup],
    exclude_originals: set[str],
    exclude_tampered: set[str],
) -> dict[str, csvted_mod.ClipGroup]:
    out: dict[str, csvted_mod.ClipGroup] = {}
    for k, g in groups.items():
        if g.original and str(g.original.resolve()) in exclude_originals:
            continue
        tampered = {
            t: p for t, p in g.tampered.items() if str(p.resolve()) not in exclude_tampered
        }
        if g.original and tampered:
            out[k] = csvted_mod.ClipGroup(source_id=g.source_id, original=g.original, tampered=tampered)
    return out


def stage_csvted_temporal(
    dst: Path,
    csvted_root: Path,
    exclude_orig: set[str],
    exclude_tam: set[str],
    include_tamper: set[str],
    exclude_tamper: set[str],
    max_pairs: int,
    seed: int,
    index_offset: int,
) -> tuple[int, int]:
    all_groups = csvted_mod.index_csvted(csvted_root)
    groups = filter_csvted_groups(all_groups, exclude_orig, exclude_tam)
    print(
        f"  CSVTED groups: indexed={len(all_groups)} after_exclude={len(groups)} "
        f"withhold_orig={len(exclude_orig)} withhold_tam={len(exclude_tam)}",
        flush=True,
    )

    picks: list[tuple] = []
    rng = random.Random(seed)
    candidates: list[tuple] = []
    for g in groups.values():
        for t in include_tamper:
            if t in g.tampered and t not in exclude_tamper:
                candidates.append((g, t))
    rng.shuffle(candidates)
    for g, t in candidates[:max_pairs]:
        picks.append((g, t))

    n = 0
    for i, (g, tamper_type) in enumerate(picks, start=1):
        idx = index_offset + i
        assert g.original is not None
        orig_dst = (
            dst
            / "original"
            / csvted_mod.CATEGORY
            / f"csvted_{idx:03d}_{csvted_mod.safe_name(g.original.name)}"
        )
        tam_dst = (
            dst
            / "tampered"
            / tamper_type
            / csvted_mod.CATEGORY
            / f"csvted_{idx:03d}_{csvted_mod.safe_name(g.tampered[tamper_type].name)}"
        )
        copy_video(g.original, orig_dst)
        copy_video(g.tampered[tamper_type], tam_dst)
        n += 2
    return len(picks), n


def split_train_val(
    rows: list[VideoRow],
    val_ratio: float,
    seed: int,
) -> tuple[list[VideoRow], list[VideoRow]]:
    groups: dict[str, list[VideoRow]] = {}
    for r in rows:
        groups.setdefault(r.group, []).append(r)

    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_ratio)) if keys else 0
    val_keys = set(keys[:n_val])

    train_rows: list[VideoRow] = []
    val_rows: list[VideoRow] = []
    for k, items in groups.items():
        (val_rows if k in val_keys else train_rows).extend(items)
    return train_rows, val_rows


def materialize_rows(rows: list[VideoRow], src_stage: Path, dst_stage: Path) -> None:
    for r in rows:
        copy_video(r.abs_path, dst_stage / r.rel)


def cmd_train(args: argparse.Namespace) -> int:
    train_stage = args.train_stage.expanduser().resolve()
    val_stage = args.val_stage.expanduser().resolve() if args.val_stage else None
    mvtb_dir = args.mvtb_train_dir.expanduser().resolve()
    csvted_root = args.csvted_root.expanduser().resolve() if args.csvted_root else None

    include_mvtb = set(t.lower() for t in args.mvtb_include)
    exclude_mvtb = set(t.lower() for t in args.mvtb_exclude)
    include_csvted = set(t.lower() for t in args.csvted_include)
    exclude_csvted = set(t.lower() for t in args.csvted_exclude)

    if args.fresh:
        if train_stage.exists():
            shutil.rmtree(train_stage)
        if val_stage and val_stage.exists():
            shutil.rmtree(val_stage)

    tmp = train_stage.parent / f".{train_stage.name}_build"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    n_mvtb = 0
    if args.skip_mvtb:
        print("=== MVTB skipped (--skip-mvtb, CSVTED-only train)")
    else:
        print(f"=== MVTB temporal from {mvtb_dir}")
        if not mvtb_dir.is_dir():
            print(f"ERROR: mvtb train dir missing: {mvtb_dir}", file=sys.stderr)
            return 1
        n_mvtb = filter_copy_tree(
            mvtb_dir,
            tmp,
            include_tamper=include_mvtb,
            exclude_tamper=exclude_mvtb,
        )
        print(f"  copied {n_mvtb} files (temporal tamper types: {sorted(include_mvtb)})")

    n_csvted_pairs = 0
    if csvted_root and args.csvted_max_pairs > 0:
        mvtb_test = args.mvtb_test_manifest.expanduser().resolve()
        csvted_test = args.csvted_test_manifest.expanduser().resolve()
        ex_orig, ex_tam = load_csvted_exclude(csvted_test)
        print(f"\n=== CSVTED temporal from {csvted_root} (exclude test manifest)")
        n_csvted_pairs, n_csv = stage_csvted_temporal(
            tmp,
            csvted_root,
            ex_orig,
            ex_tam,
            include_csvted,
            exclude_csvted,
            args.csvted_max_pairs,
            args.seed + 10,
            index_offset=max_clip_index(tmp),
        )
        print(f"  pairs={n_csvted_pairs} files={n_csv}")

    rows = scan_stage(tmp)
    if not rows:
        print("ERROR: no videos after filter", file=sys.stderr)
        return 1

    profile = "csvted-temporal-only" if args.skip_mvtb else "temporal-only"
    meta = {
        "dataset": "gmflow-temporal" if not args.skip_mvtb else "csvted-temporal",
        "split": "train+val",
        "profile": profile,
        "skip_mvtb": bool(args.skip_mvtb),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "mvtb_train_dir": None if args.skip_mvtb else str(mvtb_dir),
        "mvtb_include": sorted(include_mvtb),
        "mvtb_exclude": sorted(exclude_mvtb),
        "csvted_root": str(csvted_root) if csvted_root else None,
        "csvted_include": sorted(include_csvted),
        "csvted_exclude": sorted(exclude_csvted),
        "csvted_pairs_added": n_csvted_pairs,
        "mvtb_test_manifest": str(args.mvtb_test_manifest.expanduser().resolve()),
        "csvted_test_manifest": str(args.csvted_test_manifest.expanduser().resolve()),
    }

    if val_stage and args.val_ratio > 0:
        train_rows, val_rows = split_train_val(rows, args.val_ratio, args.seed + 99)
        train_stage.mkdir(parents=True, exist_ok=True)
        val_stage.mkdir(parents=True, exist_ok=True)
        materialize_rows(train_rows, tmp, train_stage)
        materialize_rows(val_rows, tmp, val_stage)
        meta["train_clip_groups"] = len({r.group for r in train_rows})
        meta["val_clip_groups"] = len({r.group for r in val_rows})
        write_manifest(train_stage, {**meta, "split": "train"})
        write_manifest(val_stage, {**meta, "split": "val"})
        print(f"\nTrain -> {train_stage}")
        print(json.dumps(dataset_stats(train_stage), indent=2))
        print(f"\nVal -> {val_stage}")
        print(json.dumps(dataset_stats(val_stage), indent=2))
    else:
        shutil.move(str(tmp), str(train_stage))
        write_manifest(train_stage, {**meta, "split": "train"})
        print(f"\nTrain (no val split) -> {train_stage}")
        print(json.dumps(dataset_stats(train_stage), indent=2))

    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


def cmd_test_temporal(args: argparse.Namespace) -> int:
    stage = args.stage.expanduser().resolve()
    include_mvtb = set(t.lower() for t in args.mvtb_include)
    exclude_mvtb = set(t.lower() for t in args.mvtb_exclude)
    include_csvted = set(t.lower() for t in args.csvted_include)
    exclude_csvted = set(t.lower() for t in args.csvted_exclude)

    if args.fresh and stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    n = 0
    mvtb_test = args.mvtb_test.expanduser().resolve()
    csvted_test = args.csvted_test.expanduser().resolve()
    if args.skip_mvtb:
        print("=== MVTB test skipped (--skip-mvtb)")
    elif mvtb_test.is_dir():
        print(f"=== MVTB test temporal from {mvtb_test}")
        n += filter_copy_tree(
            mvtb_test,
            stage,
            include_tamper=include_mvtb,
            exclude_tamper=exclude_mvtb,
        )
    if csvted_test.is_dir():
        print(f"=== CSVTED test temporal from {csvted_test}")
        n += filter_copy_tree(
            csvted_test,
            stage,
            include_tamper=include_csvted,
            exclude_tamper=exclude_csvted,
        )

    if n == 0:
        print("ERROR: no test videos copied", file=sys.stderr)
        return 1

    meta = {
        "dataset": "gmflow-temporal",
        "split": "test-temporal",
        "profile": "temporal-only",
        "mvtb_test": str(mvtb_test),
        "csvted_test": str(csvted_test),
        "mvtb_include": sorted(include_mvtb),
        "csvted_include": sorted(include_csvted),
    }
    write_manifest(stage, meta)
    print(f"\nTest temporal -> {stage}")
    print(json.dumps(dataset_stats(stage), indent=2))
    print(
        "\nNOTE: mixed sign-off remains at "
        "data/pull/evidence/mvtamperbench-200-s3 and csvted-200-balanced (all tamper types)."
    )
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    for p in args.stage:
        st = p.expanduser().resolve()
        if not st.is_dir():
            print(f"missing: {st}")
            continue
        print(json.dumps(dataset_stats(st), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare GMFlow temporal-only datasets")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="Build temporal TRAIN (+ optional VAL) from MVTB 1k + CSVTED")
    pt.add_argument(
        "--train-stage",
        type=Path,
        default=Path("~/forenShield-ai/forgery/data/train/video/forgery-gmflow-train-temporal"),
    )
    pt.add_argument(
        "--val-stage",
        type=Path,
        default=Path("~/forenShield-ai/forgery/data/train/video/forgery-gmflow-val-temporal"),
    )
    pt.add_argument(
        "--mvtb-train-dir",
        type=Path,
        default=Path("~/forenShield-ai/forgery/data/train/video/forgery-gmflow-train-mvtb-1k"),
    )
    pt.add_argument(
        "--csvted-root",
        type=Path,
        default=Path("~/forenShield-ai/forgery/data/test/video/csvted"),
    )
    pt.add_argument("--csvted-max-pairs", type=int, default=200)
    pt.add_argument(
        "--mvtb-test-manifest",
        type=Path,
        default=Path("~/forenShield-ai/forgery/data/pull/evidence/mvtamperbench-200-s3/manifest.json"),
    )
    pt.add_argument(
        "--csvted-test-manifest",
        type=Path,
        default=Path("~/forenShield-ai/forgery/data/pull/evidence/csvted-200-balanced/manifest.json"),
    )
    pt.add_argument("--val-ratio", type=float, default=0.15)
    pt.add_argument("--seed", type=int, default=123)
    pt.add_argument("--fresh", action="store_true")
    pt.add_argument(
        "--skip-mvtb",
        action="store_true",
        help="CSVTED-only train (no MVTB clips)",
    )
    pt.add_argument(
        "--mvtb-include",
        nargs="+",
        default=list(MVTB_TEMPORAL_DEFAULT),
    )
    pt.add_argument(
        "--mvtb-exclude",
        nargs="+",
        default=list(MVTB_SPATIAL_DEFAULT),
    )
    pt.add_argument(
        "--csvted-include",
        nargs="+",
        default=list(CSVTED_TEMPORAL_DEFAULT),
    )
    pt.add_argument(
        "--csvted-exclude",
        nargs="+",
        default=list(CSVTED_SPATIAL_DEFAULT),
    )
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("test-temporal", help="Build temporal-only TEST from MVTB200 + CSVTED200")
    pe.add_argument(
        "--stage",
        type=Path,
        default=Path("~/forenShield-ai/forgery/data/pull/evidence/gmflow-test-temporal-200"),
    )
    pe.add_argument(
        "--mvtb-test",
        type=Path,
        default=Path("~/forenShield-ai/forgery/data/pull/evidence/mvtamperbench-200-s3"),
    )
    pe.add_argument(
        "--csvted-test",
        type=Path,
        default=Path("~/forenShield-ai/forgery/data/pull/evidence/csvted-200-balanced"),
    )
    pe.add_argument("--fresh", action="store_true")
    pe.add_argument(
        "--skip-mvtb",
        action="store_true",
        help="CSVTED-only test stage (no MVTB clips)",
    )
    pe.add_argument("--mvtb-include", nargs="+", default=list(MVTB_TEMPORAL_DEFAULT))
    pe.add_argument("--mvtb-exclude", nargs="+", default=list(MVTB_SPATIAL_DEFAULT))
    pe.add_argument("--csvted-include", nargs="+", default=list(CSVTED_TEMPORAL_DEFAULT))
    pe.add_argument("--csvted-exclude", nargs="+", default=list(CSVTED_SPATIAL_DEFAULT))
    pe.set_defaults(func=cmd_test_temporal)

    ps = sub.add_parser("stats", help="Print manifest stats for staged dirs")
    ps.add_argument("stage", type=Path, nargs="+")
    ps.set_defaults(func=cmd_stats)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
