#!/usr/bin/env python3
"""Short VideoMAE fine-tune (head + fc_norm; optional backbone unfreeze)."""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_videomae_infer import CLIP_FRAMES, CLIP_SIZE, VideoMAEDetectorLite, clip_to_tensor
from video_xception_infer import crop_face, read_frame_samples


def resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def manifest_entries(data) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("items", "videos", "entries"):
            val = data.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


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
                        excluded.add(str((resolve(root, "data/raw/celeb-df-v2/Celeb-DF-v2") / p).resolve()))
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


def dir_has_mp4(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob("*.mp4"))


def discover_mp4_dirs(base: Path, *, min_count: int = 1) -> list[Path]:
    if not base.is_dir():
        return []
    counts: dict[Path, int] = {}
    for mp4 in base.rglob("*.mp4"):
        counts[mp4.parent] = counts.get(mp4.parent, 0) + 1
    return [d for d, n in sorted(counts.items(), key=lambda x: (-x[1], str(x[0]))) if n >= min_count]


def pick_real_train_dir(root: Path, preferred: str) -> Path:
    explicit = [
        preferred,
        "data/raw/faceforensics/original_sequences/youtube/c23/videos",
        "data/raw/faceforensics/original_sequences/youtube/c40/videos",
        "data/raw/faceforensics/original_sequences/youtube/raw/videos",
        "data/raw/faceforensics/original_sequences/actors/c23/videos",
        "data/raw/faceforensics/original_sequences/actors/c40/videos",
        "data/raw/voxceleb/tmp_full",
        "data/raw/voxceleb/real",
    ]
    tried: list[Path] = []
    seen: set[str] = set()
    for rel in explicit:
        candidate = resolve(root, rel)
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        tried.append(candidate)
        if dir_has_mp4(candidate):
            print(f"real train dir: {candidate}", flush=True)
            return candidate

    ff_orig = root / "data/raw/faceforensics/original_sequences"
    for candidate in discover_mp4_dirs(ff_orig, min_count=10):
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        tried.append(candidate)
        print(f"real train dir (discovered): {candidate}", flush=True)
        return candidate

    vox = root / "data/raw/voxceleb"
    for candidate in discover_mp4_dirs(vox, min_count=3):
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        tried.append(candidate)
        print(f"real train dir (voxceleb): {candidate}", flush=True)
        return candidate

    tried_lines = "\n  ".join(str(p) for p in tried)
    raise SystemExit(
        "No real training videos found. Tried:\n  "
        + tried_lines
        + "\n\n"
        "GPU에 FF++ real 원본이 없습니다. 예:\n"
        "  cd ~/forenShield-ai/data/raw/faceforensics\n"
        "  python download-FaceForensics.py . -d original -c c23 -t videos --server EU2\n"
        "또는 voxceleb tmp가 있으면:\n"
        "  python3 scripts/infer/video_videomae_finetune.py --train-real-dir data/raw/voxceleb/tmp_full"
    )


def pick_fake_train_dir(root: Path, preferred: str) -> Path:
    explicit = [
        preferred,
        "data/raw/faceforensics/manipulated_sequences/DeepFakeDetection/c40/videos",
        "data/raw/faceforensics/manipulated_sequences/DeepFakeDetection/c23/videos",
    ]
    for rel in explicit:
        candidate = resolve(root, rel)
        if dir_has_mp4(candidate):
            return candidate
    ff_manip = root / "data/raw/faceforensics/manipulated_sequences"
    for candidate in discover_mp4_dirs(ff_manip, min_count=10):
        return candidate
    raise SystemExit(f"Missing fake train dir under {ff_manip}")


def has_usable_faces(
    video_path: Path,
    face_cascade: cv2.CascadeClassifier,
    num_frames: int = CLIP_FRAMES,
    clip_size: int = CLIP_SIZE,
) -> bool:
    frame_samples = read_frame_samples(video_path, num_frames=num_frames)
    face_count = sum(
        1
        for sample in frame_samples
        if crop_face(sample["frame"], face_cascade, size=clip_size) is not None
    )
    return face_count >= max(4, num_frames // 4)


def list_train_videos(
    root: Path,
    fake_dir: Path,
    real_dir: Path,
    excluded: set[str],
    max_per_class: int,
    seed: int,
    face_cascade: cv2.CascadeClassifier,
) -> list[tuple[Path, int]]:
    rng = random.Random(seed)
    samples: list[tuple[Path, int]] = []
    for label, directory in ((1, fake_dir), (0, real_dir)):
        if not directory.is_dir():
            raise SystemExit(f"Missing train dir: {directory}")
        label_name = "fake" if label == 1 else "real"
        candidates = [
            p.resolve()
            for p in sorted(directory.glob("*.mp4"))
            if str(p.resolve()) not in excluded
        ]
        rng.shuffle(candidates)
        print(f"scanning {label_name}: {len(candidates)} mp4 in {directory}", flush=True)
        picked = 0
        for path in candidates:
            if picked >= max_per_class:
                break
            if has_usable_faces(path, face_cascade):
                samples.append((path, label))
                picked += 1
                if picked % 10 == 0:
                    print(f"  {label_name}: picked {picked}/{max_per_class}", flush=True)
        print(f"  {label_name}: selected {picked} clips", flush=True)
        if picked == 0:
            raise SystemExit(
                f"No usable {label_name} training videos in {directory} "
                f"(excluded={len(excluded)} paths)."
            )
    rng.shuffle(samples)
    return samples


class VideoClipDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        face_cascade: cv2.CascadeClassifier,
        num_frames: int = CLIP_FRAMES,
        clip_size: int = CLIP_SIZE,
    ):
        self.samples = samples
        self.face_cascade = face_cascade
        self.num_frames = num_frames
        self.clip_size = clip_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        frame_samples = read_frame_samples(path, num_frames=self.num_frames)
        crops: list[np.ndarray] = []
        indices: list[int] = []
        for sample in frame_samples:
            crop = crop_face(sample["frame"], self.face_cascade, size=self.clip_size)
            if crop is not None:
                crops.append(crop)
                indices.append(sample["frame_index"])

        if len(crops) < max(4, self.num_frames // 4):
            raise RuntimeError(f"no_face:{path.name}")

        if len(crops) > self.num_frames:
            step = len(crops) / self.num_frames
            picked = [crops[int(i * step)] for i in range(self.num_frames)]
        else:
            picked = list(crops)
            while len(picked) < self.num_frames:
                picked.append(picked[-1])

        arr = np.stack(picked[: self.num_frames], axis=0).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)
        return tensor, torch.tensor(label, dtype=torch.long)


def collate_skip_no_face(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors, dim=0), torch.stack(labels, dim=0)


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_logits(clips)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * clips.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += clips.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Short VideoMAE fine-tune")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--train-fake-dir",
        default="data/raw/faceforensics/manipulated_sequences/DeepFakeDetection/c40/videos",
    )
    parser.add_argument(
        "--train-real-dir",
        default="data/train/video/voxceleb/real",
    )
    parser.add_argument(
        "--exclude-dirs",
        nargs="*",
        default=["data/test/video/ffpp/fake_over60s", "data/test/video/voxceleb/real"],
    )
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained-id", default="MCG-NJU/videomae-base")
    parser.add_argument(
        "--output",
        default="models/test/video/videomae/v1.0.0/videomae_finetuned.pth",
    )
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument(
        "--init-weights",
        default=None,
        help="optional checkpoint to continue from (default: VideoMAE pretrained only)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    fake_dir = pick_fake_train_dir(root, args.train_fake_dir)
    real_dir = resolve_train_dir(root, args.train_real_dir, label="real")
    output = resolve(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    excluded = collect_exclude_paths(root, args.exclude_dirs)
    samples = list_train_videos(
        root, fake_dir, real_dir, excluded, args.max_per_class, args.seed, face_cascade
    )
    if len(samples) < 16:
        raise SystemExit(f"Too few training videos ({len(samples)}). Check FF++ paths.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VideoClipDataset(samples, face_cascade)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_skip_no_face,
    )

    model = VideoMAEDetectorLite(pretrained_id=args.pretrained_id).to(device)
    init_weights = resolve(root, args.init_weights) if args.init_weights else None
    if init_weights and init_weights.is_file():
        ckpt = torch.load(init_weights, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print(f"init weights: {init_weights}", flush=True)
    for param in model.backbone.parameters():
        param.requires_grad = False
    if args.unfreeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"train fake: {fake_dir}")
    print(f"train real: {real_dir}")
    print(f"excluded benchmark paths: {len(excluded)}")
    print(f"samples: {len(samples)} (fake={sum(1 for _, y in samples if y==1)}, real={sum(1 for _, y in samples if y==0)})")
    print(f"device: {device}")
    print(f"epochs: {args.epochs}, batch: {args.batch_size}, lr: {args.lr}")
    print()

    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, device)
        row = {"epoch": epoch, "loss": round(loss, 4), "train_acc": round(acc, 4)}
        history.append(row)
        print(f"epoch {epoch}/{args.epochs} loss={row['loss']} train_acc={row['train_acc']}", flush=True)

    torch.save(model.state_dict(), output)
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pretrained_id": args.pretrained_id,
        "output": str(output),
        "train_fake_dir": str(fake_dir),
        "train_real_dir": str(real_dir),
        "exclude_dirs": args.exclude_dirs,
        "max_per_class": args.max_per_class,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "unfreeze_backbone": args.unfreeze_backbone,
        "init_weights": str(init_weights) if init_weights and init_weights.is_file() else None,
        "num_samples": len(samples),
        "history": history,
    }
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print()
    print(f"saved weights: {output}")
    print(f"saved meta:    {meta_path}")


if __name__ == "__main__":
    main()
