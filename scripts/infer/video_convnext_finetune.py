#!/usr/bin/env python3
"""Short ConvNeXt fine-tune on FF++ fake + Vox/FF++ real face crops."""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_convnext_infer import DEFAULT_VARIANT, ConvNeXtDetectorLite
from video_videomae_finetune import collect_exclude_paths, pick_fake_train_dir, resolve, resolve_train_dir
from video_xception_infer import crop_face, read_frame_samples


def has_usable_faces(
    video_path: Path,
    face_cascade: cv2.CascadeClassifier,
    *,
    num_frames: int = 32,
    clip_size: int = 256,
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
            raise SystemExit(f"No usable {label_name} training videos in {directory}")
    rng.shuffle(samples)
    return samples


class FaceFrameDataset(Dataset):
    """Random face crop per sample; repeats each video frames_per_video times per epoch."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        face_cascade: cv2.CascadeClassifier,
        *,
        num_frames: int = 32,
        clip_size: int = 256,
        frames_per_video: int = 8,
        seed: int = 42,
    ):
        self.samples = samples
        self.face_cascade = face_cascade
        self.num_frames = num_frames
        self.clip_size = clip_size
        self.frames_per_video = frames_per_video
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples) * self.frames_per_video

    def __getitem__(self, idx: int):
        path, label = self.samples[idx // self.frames_per_video]
        frame_samples = read_frame_samples(path, num_frames=self.num_frames)
        crops: list[np.ndarray] = []
        for sample in frame_samples:
            crop = crop_face(sample["frame"], self.face_cascade, size=self.clip_size)
            if crop is not None:
                crops.append(crop)
        if not crops:
            raise RuntimeError(f"no_face:{path.name}")
        crop = crops[self.rng.randrange(len(crops))]
        arr = crop.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, torch.tensor(label, dtype=torch.long)


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_logits(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += images.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def set_trainable(model: ConvNeXtDetectorLite, *, unfreeze_backbone: bool) -> None:
    for param in model.backbone.stem_and_stages.parameters():
        param.requires_grad = unfreeze_backbone
    for param in model.backbone.norm.parameters():
        param.requires_grad = True
    for param in model.backbone.head.parameters():
        param.requires_grad = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Short ConvNeXt fine-tune")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--train-fake-dir",
        default="data/raw/faceforensics/manipulated_sequences/DeepFakeDetection/c40/videos",
    )
    parser.add_argument("--train-real-dir", default="data/train/video/voxceleb/real")
    parser.add_argument(
        "--exclude-dirs",
        nargs="*",
        default=["data/test/video/celeb-df-v2/fake", "data/test/video/celeb-df-v2/real"],
    )
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=["small", "base"])
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="models/test/video/convnext/v1.0.0/convnext_finetuned.pth",
    )
    parser.add_argument("--unfreeze-backbone", action="store_true")
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
    dataset = FaceFrameDataset(
        samples,
        face_cascade,
        frames_per_video=args.frames_per_video,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ConvNeXtDetectorLite(variant=args.variant, pretrained=True).to(device)
    set_trainable(model, unfreeze_backbone=args.unfreeze_backbone)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"train fake: {fake_dir}")
    print(f"train real: {real_dir}")
    print(f"variant: {args.variant}")
    print(f"excluded benchmark paths: {len(excluded)}")
    print(
        f"samples: {len(samples)} "
        f"(fake={sum(1 for _, y in samples if y == 1)}, real={sum(1 for _, y in samples if y == 0)})"
    )
    print(f"device: {device}")
    print(f"epochs: {args.epochs}, batch: {args.batch_size}, lr: {args.lr}")
    print(f"unfreeze_backbone: {args.unfreeze_backbone}")
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
        "variant": args.variant,
        "pretrained": "torchvision IMAGENET1K_V1",
        "output": str(output),
        "train_fake_dir": str(fake_dir),
        "train_real_dir": str(real_dir),
        "exclude_dirs": args.exclude_dirs,
        "max_per_class": args.max_per_class,
        "frames_per_video": args.frames_per_video,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "unfreeze_backbone": args.unfreeze_backbone,
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
