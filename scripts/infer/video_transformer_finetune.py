#!/usr/bin/env python3
"""Short fine-tune for TimeSformer or Video Swin (head + fc_norm; frozen backbone)."""
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
from video_clip_transformer_common import normalize_face_crops
from video_swin3d_infer import CLIP_FRAMES as SWIN_CLIP_FRAMES
from video_swin3d_infer import CLIP_SIZE as SWIN_CLIP_SIZE
from video_swin3d_infer import VideoSwinDetectorLite, clip_to_tensor as swin_clip_to_tensor
from video_timesformer_infer import CLIP_FRAMES as TS_CLIP_FRAMES
from video_timesformer_infer import CLIP_SIZE as TS_CLIP_SIZE
from video_timesformer_infer import DEFAULT_PRETRAINED, TimeSformerDetectorLite, clip_to_tensor as ts_clip_to_tensor
from video_videomae_finetune import collect_exclude_paths, pick_fake_train_dir, resolve, resolve_train_dir
from video_xception_infer import crop_face, read_frame_samples


def has_usable_faces(
    video_path: Path,
    face_cascade: cv2.CascadeClassifier,
    *,
    num_frames: int,
    clip_size: int,
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
    *,
    num_frames: int,
    clip_size: int,
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
            if has_usable_faces(path, face_cascade, num_frames=num_frames, clip_size=clip_size):
                samples.append((path, label))
                picked += 1
                if picked % 10 == 0:
                    print(f"  {label_name}: picked {picked}/{max_per_class}", flush=True)
        print(f"  {label_name}: selected {picked} clips", flush=True)
        if picked == 0:
            raise SystemExit(f"No usable {label_name} training videos in {directory}")
    rng.shuffle(samples)
    return samples


def train_one_epoch(model, loader, criterion, optimizer, device, *, clip_to_tensor_fn) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for clips, labels in loader:
        labels = labels.to(device)
        logits_list = []
        for i in range(clips.size(0)):
            sample = clips[i]
            crop_list = []
            for t in range(sample.size(0)):
                frame = sample[t].permute(1, 2, 0).numpy()
                frame = np.clip(frame * std + mean, 0.0, 1.0)
                crop_list.append((frame * 255.0).astype(np.uint8))
            logits_list.append(model.forward_logits(clip_to_tensor_fn(crop_list, device)))
        logits = torch.cat(logits_list, dim=0)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)
    return total_loss / max(1, total), correct / max(1, total)

MODEL_SPECS = {
    "timesformer": {
        "clip_frames": TS_CLIP_FRAMES,
        "clip_size": TS_CLIP_SIZE,
        "default_output": "models/test/video/timesformer/v1.0.0/timesformer_finetuned.pth",
    },
    "video-swin": {
        "clip_frames": SWIN_CLIP_FRAMES,
        "clip_size": SWIN_CLIP_SIZE,
        "default_output": "models/test/video/video-swin/v1.0.0/video_swin_finetuned.pth",
    },
}


class VideoClipDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        face_cascade: cv2.CascadeClassifier,
        *,
        num_frames: int,
        clip_size: int,
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
        for sample in frame_samples:
            crop = crop_face(sample["frame"], self.face_cascade, size=self.clip_size)
            if crop is not None:
                crops.append(crop)

        if len(crops) < max(4, self.num_frames // 4):
            raise RuntimeError(f"no_face:{path.name}")

        if len(crops) > self.num_frames:
            step = len(crops) / self.num_frames
            picked = [crops[int(i * step)] for i in range(self.num_frames)]
        else:
            picked = list(crops)
            while len(picked) < self.num_frames:
                picked.append(picked[-1])

        arr = normalize_face_crops(picked[: self.num_frames])
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)
        return tensor, torch.tensor(label, dtype=torch.long)


def collate_clips(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors, dim=0), torch.stack(labels, dim=0)


def build_model(model_name: str, device: torch.device, pretrained_id: str):
    if model_name == "timesformer":
        return TimeSformerDetectorLite(pretrained_id=pretrained_id).to(device)
    if model_name == "video-swin":
        return VideoSwinDetectorLite().to(device)
    raise SystemExit(f"unknown model: {model_name}")


def clip_to_tensor_fn_for(model_name: str):
    if model_name == "timesformer":
        return ts_clip_to_tensor
    return swin_clip_to_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune TimeSformer or Video Swin for deepfake detection")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS.keys()))
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
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1, help="keep 1 for video transformers on 24GB GPU")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained-id", default=DEFAULT_PRETRAINED)
    parser.add_argument("--output", default=None)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--init-weights", default=None)
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model]
    root = Path(args.root).resolve()
    fake_dir = pick_fake_train_dir(root, args.train_fake_dir)
    real_dir = resolve_train_dir(root, args.train_real_dir, label="real")
    output = resolve(root, args.output or spec["default_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    excluded = collect_exclude_paths(root, args.exclude_dirs)
    samples = list_train_videos(
        root,
        fake_dir,
        real_dir,
        excluded,
        args.max_per_class,
        args.seed,
        face_cascade,
        num_frames=spec["clip_frames"],
        clip_size=spec["clip_size"],
    )
    if len(samples) < 16:
        raise SystemExit(f"Too few training videos ({len(samples)}).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VideoClipDataset(
        samples,
        face_cascade,
        num_frames=spec["clip_frames"],
        clip_size=spec["clip_size"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_clips,
    )

    model = build_model(args.model, device, args.pretrained_id)
    init_weights = resolve(root, args.init_weights) if args.init_weights else None
    if init_weights and init_weights.is_file():
        ckpt = torch.load(init_weights, map_location=device, weights_only=False)
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
    to_tensor = clip_to_tensor_fn_for(args.model)

    print(f"model: {args.model}")
    print(f"train fake: {fake_dir}")
    print(f"train real: {real_dir}")
    print(f"excluded benchmark paths: {len(excluded)}")
    print(f"samples: {len(samples)} (fake={sum(1 for _, y in samples if y==1)}, real={sum(1 for _, y in samples if y==0)})")
    print(f"clip_frames: {spec['clip_frames']}, device: {device}")
    print(f"epochs: {args.epochs}, batch: {args.batch_size}, lr: {args.lr}")
    print()

    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, device, clip_to_tensor_fn=to_tensor)
        row = {"epoch": epoch, "loss": round(loss, 4), "train_acc": round(acc, 4)}
        history.append(row)
        print(f"epoch {epoch}/{args.epochs} loss={row['loss']} train_acc={row['train_acc']}", flush=True)

    torch.save(model.state_dict(), output)
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "pretrained_id": args.pretrained_id if args.model == "timesformer" else "swin3d_t/KINETICS400_V1",
        "output": str(output),
        "train_fake_dir": str(fake_dir),
        "train_real_dir": str(real_dir),
        "exclude_dirs": args.exclude_dirs,
        "clip_frames": spec["clip_frames"],
        "max_per_class": args.max_per_class,
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
