#!/usr/bin/env python3
"""Xception fine-tune dataset helpers: S3 pull, unseen-sample selection, manifest I/O."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_S3_BUCKET = "forenshield-evidence-877044078824"
DEFAULT_S3_TRAIN_PREFIX = "cases/train/video/xception"

# Golden / benchmark — must never be used for training.
DEFAULT_BENCHMARK_EXCLUDE_DIRS = (
    "data/test/video/celeb-df-v2/fake",
    "data/test/video/celeb-df-v2/real",
    "data/test/video/ffpp/fake_over60s",
    "data/test/video/voxceleb/real",
    "data/benchmark/video-benchmark-datasets/celebdf/fake",
    "data/benchmark/video-benchmark-datasets/celebdf/real",
    "data/benchmark/video-benchmark-datasets/ffpp_vox/fake",
    "data/benchmark/video-benchmark-datasets/ffpp_vox/real",
)

STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "stage1": {
        "label": "1차",
        "description": "Core FF++ fake + Vox real (backbone freeze)",
        "s3_subdir": "stage1",
        "max_per_class": 100,
        "val_holdout": 40,
        "unfreeze_backbone": False,
        "lr": 1e-4,
        "epochs": 10,
        "early_stop_patience": 2,
        "seed": 42,
    },
    "stage2": {
        "label": "2차",
        "description": "Stage1 best ckpt + partial backbone unfreeze (lr=1e-5)",
        "s3_subdir": "stage2",
        "max_per_class": 30,
        "val_holdout": 40,
        "unfreeze_backbone": True,
        "lr": 1e-5,
        "epochs": 5,
        "early_stop_patience": 2,
        "seed": 42,
    },
    "ff1k": {
        "label": "FF 1k",
        "description": "FF++ DeepFakeDetection fake + Vox real pools (prior 100-clip train excluded)",
        "s3_subdir": "ff1k",
        "max_per_class": 1000,
        "val_holdout": 200,
        "unfreeze_backbone": False,
        "lr": 1e-4,
        "epochs": 15,
        "early_stop_patience": 4,
        "seed": 1001,
        "real_source": "vox",
    },
    "celeb1k": {
        "label": "Celeb 1k",
        "description": "Celeb-DF v2 fake/real 1000 each, init from ff1k ckpt",
        "s3_subdir": "celeb1k",
        "max_per_class": 1000,
        "val_holdout": 200,
        "unfreeze_backbone": True,
        "lr": 1e-5,
        "epochs": 10,
        "early_stop_patience": 3,
        "seed": 2002,
    },
}

# Previous 100-clip staged runs — never reuse for ff1k / celeb1k sampling.
DEFAULT_PRIOR_TRAIN_MANIFESTS: tuple[str, ...] = (
    "docs/xception_finetune_train_stage1.json",
    "docs/xception_finetune_train_stage2.json",
    "data/pull/train/video/xception/stage1/manifest.json",
    "data/pull/train/video/xception/stage2/manifest.json",
)


def resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def dir_has_mp4(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob("*.mp4"))


def discover_mp4_dirs(base: Path, *, min_count: int = 1) -> list[Path]:
    if not base.is_dir():
        return []
    counts: dict[Path, int] = {}
    for mp4 in base.rglob("*.mp4"):
        counts[mp4.parent] = counts.get(mp4.parent, 0) + 1
    return [d for d, n in sorted(counts.items(), key=lambda x: (-x[1], str(x[0]))) if n >= min_count]


def _add_manifest_entry_paths(
    excluded: set[str],
    entry: dict,
    root: Path,
    base_dir: Path | None,
) -> None:
    for key in ("source_path", "origin_path", "local_path", "path"):
        val = entry.get(key)
        if not val:
            continue
        p = Path(val)
        if p.is_absolute():
            excluded.add(str(p.resolve()))
        elif base_dir is not None:
            excluded.add(str((base_dir / p).resolve()))
        else:
            excluded.add(str(resolve(root, str(val))))
    file_name = entry.get("file")
    if file_name and base_dir is not None:
        for sub in ("fake", "real"):
            candidate = base_dir / sub / file_name
            if candidate.is_file():
                excluded.add(str(candidate.resolve()))


def collect_prior_train_exclude_paths(
    root: Path,
    *,
    extra_manifests: list[str] | None = None,
) -> set[str]:
    """Paths used in earlier fine-tune runs (stage1/stage2 100-clip, optional ff1k for celeb1k)."""
    excluded: set[str] = set()
    for rel in (*DEFAULT_PRIOR_TRAIN_MANIFESTS, *(extra_manifests or [])):
        manifest = resolve(root, rel)
        if not manifest.is_file():
            continue
        data = json.loads(manifest.read_text(encoding="utf-8"))
        base_dir = manifest.parent
        for entry in manifest_entries(data):
            _add_manifest_entry_paths(excluded, entry, root, base_dir)
        if isinstance(data, dict):
            for split in ("train", "val"):
                for entry in manifest_entries(data.get(split, [])):
                    _add_manifest_entry_paths(excluded, entry, root, base_dir)
    return excluded


def collect_exclude_paths(root: Path, exclude_dirs: list[str]) -> set[str]:
    excluded: set[str] = set()
    tmp_full = resolve(root, "data/raw/voxceleb/tmp_full")
    for rel in exclude_dirs:
        d = resolve(root, rel)
        if not d.is_dir():
            continue
        for mp4 in d.glob("*.mp4"):
            excluded.add(str(mp4.resolve()))
        manifest = d / "manifest.json"
        if manifest.is_file():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for entry in manifest_entries(data):
                for key in ("source_path", "origin_path", "local_path", "path", "file", "source"):
                    val = entry.get(key)
                    if not val:
                        continue
                    p = Path(val)
                    if p.is_absolute():
                        excluded.add(str(p.resolve()))
                    elif key == "source":
                        excluded.add(
                            str((resolve(root, "data/raw/celeb-df-v2/Celeb-DF-v2") / p).resolve())
                        )
                    else:
                        excluded.add(str((d / p).resolve()))
                video_id = entry.get("video_id")
                if isinstance(video_id, str) and video_id:
                    excluded.add(str((d / f"{video_id}_long.mp4").resolve()))
                    excluded.add(str((tmp_full / f"{video_id}.mp4").resolve()))
    return excluded


def resolve_train_dir(root: Path, preferred: str, *, label: str) -> Path:
    candidate = resolve(root, preferred)
    if dir_has_mp4(candidate):
        print(f"{label} train dir: {candidate}", flush=True)
        return candidate
    raise SystemExit(
        f"Missing {label} train dir (no mp4): {candidate}\n"
        + (
            "  Run: bash scripts/download/data/prepare_videomae_train_data.sh\n"
            if label == "real"
            else f"  Check FF++ fake pool under {preferred}"
        )
    )


def pick_fake_train_dir(root: Path, preferred: str) -> Path:
    tried: list[Path] = []
    explicit = [
        preferred,
        "data/raw/faceforensics/manipulated_sequences/DeepFakeDetection/c40/videos",
        "data/raw/faceforensics/manipulated_sequences/DeepFakeDetection/c23/videos",
        "data/raw/faceforensics/manipulated_sequences/Deepfakes/c40/videos",
        "data/raw/faceforensics/manipulated_sequences/Face2Face/c40/videos",
    ]
    for rel in explicit:
        candidate = resolve(root, rel)
        tried.append(candidate)
        if dir_has_mp4(candidate):
            print(f"fake train dir: {candidate}", flush=True)
            return candidate

    search_roots = [
        resolve(root, "data/raw/faceforensics/manipulated_sequences"),
        resolve(root, "data/raw/faceforensics"),
    ]
    seen: set[str] = set()
    for base in search_roots:
        if not base.is_dir():
            continue
        for min_count in (10, 1):
            for candidate in discover_mp4_dirs(base, min_count=min_count):
                key = str(candidate)
                if key in seen:
                    continue
                if "original" in candidate.parts:
                    continue
                seen.add(key)
                print(f"fake train dir (discovered): {candidate}", flush=True)
                return candidate

    ff_root = resolve(root, "data/raw/faceforensics")
    tried_lines = "\n  ".join(str(p) for p in tried)
    raise SystemExit(
        "No FF++ fake training pool found. Tried:\n  "
        + tried_lines
        + f"\n\nOn GPU check:\n"
        f"  ls -la {ff_root}\n"
        f"  find {ff_root} -name '*.mp4' 2>/dev/null | wc -l\n"
        "Expected (docs/deepfake/VIDEO_DATASET_INVENTORY.md):\n"
        "  data/raw/faceforensics/manipulated_sequences/DeepFakeDetection/c40/videos/\n"
        "Override:\n"
        "  TRAIN_FAKE_POOL=<path> bash scripts/infer/run_xception_finetune_staged.sh"
    )


def vox_real_train_dir_candidates(root: Path) -> list[Path]:
    """Vox / long-form real pools (benchmark dirs excluded later at sample time)."""
    tried: list[Path] = []
    explicit = [
        "data/raw/voxceleb/tmp_full",
        "data/raw/voxceleb",
        "data/train/video/voxceleb/real",
    ]
    found: list[tuple[int, Path]] = []
    for rel in explicit:
        candidate = resolve(root, rel)
        tried.append(candidate)
        if dir_has_mp4(candidate):
            count = len(list(candidate.glob("*.mp4")))
            found.append((count, candidate))
    vox_root = resolve(root, "data/raw/voxceleb")
    if vox_root.is_dir():
        for candidate in discover_mp4_dirs(vox_root, min_count=1):
            if candidate in tried:
                continue
            count = len(list(candidate.glob("*.mp4")))
            found.append((count, candidate))
            tried.append(candidate)
    found.sort(key=lambda x: (-x[0], str(x[1])))
    return [path for _, path in found]


def pick_vox_real_train_dir(root: Path, preferred: str | None = None) -> Path:
    """Best single Vox real dir (most mp4). Sampling may merge all Vox pools."""
    if preferred:
        candidate = resolve(root, preferred)
        if dir_has_mp4(candidate):
            print(f"real train dir (Vox override): {candidate}", flush=True)
            return candidate
    pools = vox_real_train_dir_candidates(root)
    if pools:
        best = pools[0]
        count = len(list(best.glob("*.mp4")))
        print(f"real train dir (Vox primary): {best} ({count} mp4)", flush=True)
        if len(pools) > 1:
            extra = sum(len(list(d.glob("*.mp4"))) for d in pools[1:])
            print(
                f"  + extra Vox pools: {len(pools) - 1} dirs (~{extra} more mp4 before dedupe)",
                flush=True,
            )
        return best
    tried = "\n  ".join(
        str(resolve(root, rel))
        for rel in (
            "data/raw/voxceleb/tmp_full",
            "data/raw/voxceleb",
            "data/train/video/voxceleb/real",
        )
    )
    raise SystemExit(
        "No Vox real training pool found. Tried:\n  "
        + tried
        + "\n\nOverride:\n  TRAIN_REAL_DIR=<path> bash scripts/infer/run_xception_finetune_1k.sh"
    )


def pick_ffpp_real_train_dir(root: Path, preferred: str | None = None) -> Path:
    """FF++ original (youtube) real pool for large-scale FF fine-tune."""
    tried: list[Path] = []
    explicit = [
        preferred,
        "data/raw/faceforensics/original_sequences/youtube/c40/videos",
        "data/raw/faceforensics/original_sequences/youtube/c23/videos",
        "data/raw/faceforensics/original_sequences/actors/c40/videos",
    ]
    for rel in explicit:
        if not rel:
            continue
        candidate = resolve(root, rel)
        tried.append(candidate)
        if dir_has_mp4(candidate):
            print(f"real train dir (FF++ original): {candidate}", flush=True)
            return candidate

    ff_root = resolve(root, "data/raw/faceforensics/original_sequences")
    for base in (ff_root, resolve(root, "data/raw/faceforensics")):
        if not base.is_dir():
            continue
        for candidate in discover_mp4_dirs(base, min_count=100):
            text = str(candidate).lower()
            if "original" in text or "youtube" in text or "actors" in text:
                if "manipulated" not in text:
                    print(f"real train dir (discovered): {candidate}", flush=True)
                    return candidate

    tried_lines = "\n  ".join(str(p) for p in tried)
    raise SystemExit(
        "No FF++ original real training pool found. Tried:\n  "
        + tried_lines
        + "\n\nExpected:\n"
        "  data/raw/faceforensics/original_sequences/youtube/c40/videos/\n"
        "Override:\n"
        "  TRAIN_REAL_DIR=<path> bash scripts/infer/run_xception_finetune_1k.sh"
    )


def pick_celeb_train_dirs(root: Path) -> tuple[Path, Path]:
    tried: list[Path] = []
    celeb_root = resolve(root, "data/raw/celeb-df-v2")
    bases = [
        resolve(root, "data/raw/celeb-df-v2/Celeb-DF-v2"),
        celeb_root,
    ]

    def pick_pool(kind: str) -> Path | None:
        fake_names = ("Celeb-synthesis", "Fake", "fake")
        real_names = ("Celeb-real", "YouTube-real", "Youtube-real", "Real", "real")
        names = fake_names if kind == "fake" else real_names
        for base in bases:
            if not base.is_dir():
                continue
            for name in names:
                for candidate in (base / name / "videos", base / name):
                    tried.append(candidate)
                    if dir_has_mp4(candidate):
                        return candidate
            for candidate in discover_mp4_dirs(base, min_count=10):
                text = f"{candidate.parent.name}/{candidate.name}".lower()
                if kind == "fake" and ("synthesis" in text or text.endswith("/fake")):
                    return candidate
                if kind == "real" and (
                    "youtube" in text or text.endswith("/real") or "/celeb-real" in text
                ):
                    if "synthesis" not in text and "/fake" not in text:
                        return candidate
        return None

    fake = pick_pool("fake")
    real = pick_pool("real")
    if fake is not None and real is not None:
        print(f"fake train dir: {fake}", flush=True)
        print(f"real train dir: {real}", flush=True)
        return fake, real

    tried_lines = "\n  ".join(str(p) for p in tried)
    raise SystemExit(
        "No Celeb-DF v2 train pool found. Tried:\n  "
        + tried_lines
        + f"\n\nOn GPU check:\n"
        f"  find {celeb_root} -type d -name 'Celeb-real' -o -name 'YouTube-real'\n"
        f"  find {celeb_root} -name '*.mp4' -printf '%h\\n' | sort | uniq -c | sort -rn | head -10\n"
        "Override:\n"
        "  TRAIN_FAKE_POOL=<fake> TRAIN_REAL_DIR=<real> bash scripts/infer/run_xception_finetune_staged.sh"
    )


def infer_scripts_dir_from_repo_file(repo_file: Path) -> Path:
    """Return absolute scripts/infer from a file under scripts/download/data/."""
    return repo_file.resolve().parents[3] / "scripts" / "infer"


def manifest_entries(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("train", "clips", "items", "videos", "entries"):
            val = data.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def s3_uri(bucket: str, prefix: str, *parts: str) -> str:
    key = "/".join(p.strip("/") for p in (prefix, *parts) if p)
    return f"s3://{bucket}/{key}"


def aws_s3_sync(src: str, dest: Path, *, include_mp4: bool = True) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "sync", src, str(dest)]
    if include_mp4:
        cmd.extend(["--exclude", "*", "--include", "*.mp4"])
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def aws_s3_cp(src: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "cp", src, str(dest)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def pull_stage_from_s3(
    root: Path,
    stage: str,
    *,
    bucket: str = DEFAULT_S3_BUCKET,
    s3_prefix: str = DEFAULT_S3_TRAIN_PREFIX,
    local_root: str | None = None,
) -> Path:
    """Pull stage train mp4 + manifest.json from S3 into data/pull/train/video/xception/{stage}/."""
    spec = STAGE_DEFAULTS.get(stage)
    if spec is None:
        raise ValueError(f"unknown stage: {stage} (use stage1|stage2)")
    sub = spec["s3_subdir"]
    base = resolve(root, local_root or f"data/pull/train/video/xception/{sub}")
    s3_base = s3_uri(bucket, s3_prefix, sub)
    for label in ("fake", "real"):
        aws_s3_sync(f"{s3_base}/{label}/", base / label)
    manifest_s3 = f"{s3_base}/manifest.json"
    manifest_local = base / "manifest.json"
    try:
        aws_s3_cp(manifest_s3, manifest_local)
    except subprocess.CalledProcessError:
        print(f"WARN: no S3 manifest at {manifest_s3}", flush=True)
    return base


def manifest_path_for_stage(root: Path, stage: str) -> Path:
    sub = STAGE_DEFAULTS[stage]["s3_subdir"]
    return resolve(root, f"data/pull/train/video/xception/{sub}/manifest.json")


def docs_manifest_path(root: Path, stage: str) -> Path:
    return resolve(root, f"docs/xception_finetune_train_{stage}.json")


def docs_markdown_path(root: Path) -> Path:
    return resolve(root, "docs/XCEPTION_FINETUNE_TRAIN_MANIFESTS.md")


def entry_to_path(root: Path, entry: dict, base_dir: Path | None = None) -> Path | None:
    for key in ("local_path", "path", "file"):
        val = entry.get(key)
        if not val:
            continue
        p = Path(val)
        if p.is_absolute():
            return p.resolve()
        if base_dir is not None:
            return (base_dir / p).resolve()
        return resolve(root, str(val))
    return None


def _entries_to_samples(
    entries: list[dict],
    root: Path,
    base_dir: Path,
) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for entry in entries:
        label_str = str(entry.get("label", "")).lower()
        if label_str in {"fake", "1"}:
            label = 1
        elif label_str in {"real", "0"}:
            label = 0
        else:
            continue
        path = entry_to_path(root, entry, base_dir)
        if path is None or not path.is_file():
            name = entry.get("file") or entry.get("path") or "?"
            raise FileNotFoundError(f"missing clip from manifest: {name} ({path})")
        samples.append((path, label))
    return samples


def load_train_manifest(manifest_path: Path, root: Path) -> list[tuple[Path, int]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    if isinstance(data, dict) and isinstance(data.get("train"), list):
        samples = _entries_to_samples(manifest_entries(data["train"]), root, base_dir)
    else:
        samples = _entries_to_samples(manifest_entries(data), root, base_dir)
    if not samples:
        raise ValueError(f"no train clips in manifest: {manifest_path}")
    return samples


def load_train_val_from_manifest(
    manifest_path: Path,
    root: Path,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]] | None:
    if not manifest_path.is_file():
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "train" not in data:
        return None
    base_dir = manifest_path.parent
    train = _entries_to_samples(manifest_entries(data["train"]), root, base_dir)
    val = _entries_to_samples(manifest_entries(data.get("val", [])), root, base_dir)
    return train, val


def samples_to_manifest_entries(
    samples: list[tuple[Path, int]],
    *,
    stage: str,
    split: str,
    root: Path,
    source: str = "local",
) -> list[dict]:
    rows: list[dict] = []
    for path, label in samples:
        try:
            rel = str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            rel = str(path.resolve())
        rows.append(
            {
                "file": path.name,
                "path": rel,
                "local_path": str(path.resolve()),
                "label": "fake" if label == 1 else "real",
                "stage": stage,
                "split": split,
                "source": source,
            }
        )
    return rows


def write_train_manifest_doc(
    root: Path,
    *,
    stage: str,
    train_samples: list[tuple[Path, int]],
    val_samples: list[tuple[Path, int]],
    meta: dict[str, Any],
) -> tuple[Path, Path]:
    """Write JSON manifest + append/update markdown index under ai/docs/."""
    spec = STAGE_DEFAULTS.get(stage, {"label": stage, "description": ""})
    now = datetime.now(timezone.utc).isoformat()
    json_path = docs_manifest_path(root, stage)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "stage": stage,
        "stage_label": spec.get("label", stage),
        "description": spec.get("description", ""),
        "created_at": now,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "meta": meta,
        "train": samples_to_manifest_entries(train_samples, stage=stage, split="train", root=root),
        "val": samples_to_manifest_entries(val_samples, stage=stage, split="val", root=root),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = docs_markdown_path(root)
    train_names = [p.name for p, _ in train_samples]
    val_names = [p.name for p, _ in val_samples]
    section = (
        f"\n## {spec.get('label', stage)} (`{stage}`)\n\n"
        f"- **설명:** {spec.get('description', '')}\n"
        f"- **기록 시각 (UTC):** {now}\n"
        f"- **train:** {len(train_samples)} clips · **val:** {len(val_samples)} clips\n"
        f"- **JSON:** `{json_path.relative_to(root)}`\n\n"
        f"### Train 파일 ({len(train_names)})\n\n"
        + "\n".join(f"- `{n}`" for n in sorted(train_names))
        + f"\n\n### Val 파일 ({len(val_names)})\n\n"
        + "\n".join(f"- `{n}`" for n in sorted(val_names))
        + "\n"
    )

    header = (
        "# Xception Fine-tune 학습 데이터 manifest\n\n"
        "골든/벤치마크 200편은 **학습 제외**. "
        "아래 목록은 `video_xception_finetune.py` 실행 시 실제 사용된 clip 파일명입니다.\n\n"
        "| Stage | 설명 | JSON |\n"
        "|-------|------|------|\n"
        f"| 1차 (stage1) | Core FF++ fake + Vox real | `docs/xception_finetune_train_stage1.json` |\n"
        f"| 2차 (stage2) | Celeb proxy (S3) + stage1 ckpt 이어학습 | `docs/xception_finetune_train_stage2.json` |\n"
        f"| FF 1k (ff1k) | FF++ fake + Vox real | `docs/xception_finetune_train_ff1k.json` |\n"
        f"| Celeb 1k (celeb1k) | Celeb-DF 1000 each on ff1k ckpt | `docs/xception_finetune_train_celeb1k.json` |\n"
    )
    marker = f"<!-- stage:{stage} -->"
    if md_path.is_file():
        text = md_path.read_text(encoding="utf-8")
        if marker in text:
            start = text.index(marker)
            end = text.find("\n<!-- stage:", start + 1)
            if end == -1:
                end = len(text)
            text = text[:start] + marker + section + text[end:]
        else:
            text = text.rstrip() + "\n\n" + marker + section
        if "# Xception Fine-tune" not in text:
            text = header + text
    else:
        text = header + "\n" + marker + section
    md_path.write_text(text, encoding="utf-8")
    return json_path, md_path
