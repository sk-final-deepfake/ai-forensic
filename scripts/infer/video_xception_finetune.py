#!/usr/bin/env python3
"""Short Xception fine-tune on FF++ fake + Vox real (DeepfakeBench ckpt init).

Step 3 package: focal loss, cosine LR, early stopping, per-epoch savepoint resume,
JPEG/blur/flip aug. Training crop matches step-2 infer defaults (MediaPipe p0.3 square).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_crop import FaceCropper, create_face_cropper
from video_videomae_finetune import collect_exclude_paths, pick_fake_train_dir, resolve, resolve_train_dir
from video_xception_infer import XceptionDetectorLite, read_frame_samples
from xception_finetune_crop_cache import (
    cache_paths,
    crop_cache_root,
    ensure_crop_cache_for_samples,
    ensure_video_crop_cache,
    has_usable_faces_from_crops,
    load_cached_crops,
)

DEFAULT_WEIGHTS = "models/test/video/xception/v1.0.0/xception_best.pth"
DEFAULT_OUTPUT = "models/test/video/xception/v1.0.0/xception_finetuned.pth"


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def has_usable_faces(
    video_path: Path,
    cropper: FaceCropper,
    *,
    num_frames: int = 32,
) -> bool:
    frame_samples = read_frame_samples(video_path, num_frames=num_frames)
    face_count = sum(1 for sample in frame_samples if cropper.crop(sample["frame"]) is not None)
    return face_count >= max(4, num_frames // 4)


def list_train_videos(
    fake_dir: Path,
    real_dir: Path,
    excluded: set[str],
    max_per_class: int,
    seed: int,
    cropper: FaceCropper,
    *,
    num_frames: int = 32,
    cache_root: Path | None = None,
    rebuild_cache: bool = False,
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
            if cache_root is not None:
                crops = ensure_video_crop_cache(
                    path,
                    cropper,
                    cache_root,
                    num_frames=num_frames,
                    rebuild=rebuild_cache,
                )
                ok = has_usable_faces_from_crops(crops, num_frames=num_frames)
            else:
                ok = has_usable_faces(path, cropper, num_frames=num_frames)
            if ok:
                samples.append((path, label))
                picked += 1
                if picked % 10 == 0:
                    print(f"  {label_name}: picked {picked}/{max_per_class}", flush=True)
        print(f"  {label_name}: selected {picked} clips", flush=True)
        if picked == 0:
            raise SystemExit(f"No usable {label_name} training videos in {directory}")
    rng.shuffle(samples)
    return samples


def split_train_val(
    samples: list[tuple[Path, int]],
    val_holdout: int,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    if val_holdout <= 0:
        return samples, []
    per_class = val_holdout // 2
    if per_class == 0:
        return samples, []
    rng = random.Random(seed + 1)
    fake = [s for s in samples if s[1] == 1]
    real = [s for s in samples if s[1] == 0]
    if len(fake) <= per_class or len(real) <= per_class:
        raise SystemExit(
            f"val_holdout={val_holdout} needs >{per_class} clips per class "
            f"(have fake={len(fake)}, real={len(real)})"
        )
    rng.shuffle(fake)
    rng.shuffle(real)
    val = fake[:per_class] + real[:per_class]
    train = fake[per_class:] + real[per_class:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def apply_augmentation(crop: np.ndarray, rng: random.Random, aug: set[str]) -> np.ndarray:
    out = crop
    if "flip" in aug and rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if "blur" in aug and rng.random() < 0.5:
        sigma = rng.uniform(0.0, 1.5)
        k = int(max(3, sigma * 4)) | 1
        out = cv2.GaussianBlur(out, (k, k), sigma)
    if "jpeg" in aug and rng.random() < 0.5:
        quality = rng.randint(60, 95)
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


class FaceFrameDataset(Dataset):
    """Random face crop per sample; repeats each video frames_per_video times per epoch."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        cropper: FaceCropper,
        *,
        num_frames: int = 32,
        frames_per_video: int = 8,
        seed: int = 42,
        augment: set[str] | None = None,
        train: bool = True,
    ):
        self.samples = samples
        self.cropper = cropper
        self.num_frames = num_frames
        self.frames_per_video = frames_per_video
        self.rng = random.Random(seed)
        self.augment = augment or set()
        self.train = train

    def __len__(self) -> int:
        return len(self.samples) * self.frames_per_video

    def __getitem__(self, idx: int):
        path, label = self.samples[idx // self.frames_per_video]
        frame_samples = read_frame_samples(path, num_frames=self.num_frames)
        crops: list[np.ndarray] = []
        for sample in frame_samples:
            crop = self.cropper.crop(sample["frame"])
            if crop is not None:
                crops.append(crop)
        if not crops:
            raise RuntimeError(f"no_face:{path.name}")
        crop = crops[self.rng.randrange(len(crops))]
        if self.train and self.augment:
            crop = apply_augmentation(crop, self.rng, self.augment)
        arr = crop.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, torch.tensor(label, dtype=torch.long)


class CachedFaceFrameDataset(Dataset):
    """Face crops loaded from disk cache (no per-epoch video decode)."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        cache_root: Path,
        *,
        num_frames: int = 32,
        frames_per_video: int = 8,
        seed: int = 42,
        augment: set[str] | None = None,
        train: bool = True,
    ):
        self.samples = samples
        self.cache_root = cache_root
        self.num_frames = num_frames
        self.frames_per_video = frames_per_video
        self.rng = random.Random(seed)
        self.augment = augment or set()
        self.train = train

    def __len__(self) -> int:
        return len(self.samples) * self.frames_per_video

    def _load_crops(self, path: Path) -> np.ndarray:
        npz_path, meta_path = cache_paths(self.cache_root, path)
        crops = load_cached_crops(npz_path, meta_path, path, num_frames=self.num_frames)
        if crops is None or crops.shape[0] == 0:
            raise RuntimeError(f"missing_crop_cache:{path.name}")
        return crops

    def __getitem__(self, idx: int):
        path, label = self.samples[idx // self.frames_per_video]
        crops = self._load_crops(path)
        crop = crops[self.rng.randrange(crops.shape[0])]
        if self.train and self.augment:
            crop = apply_augmentation(crop, self.rng, self.augment)
        arr = crop.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, torch.tensor(label, dtype=torch.long)


def set_trainable(model: XceptionDetectorLite, *, unfreeze_backbone: bool) -> None:
    for param in model.parameters():
        param.requires_grad = False
    backbone = model.backbone
    if unfreeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = True
        return
    for name in ("block11", "block12", "conv3", "bn3", "conv4", "bn4", "last_linear"):
        module = getattr(backbone, name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True


def build_criterion(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.loss == "focal":
        if args.label_smoothing > 0:
            print("warning: label_smoothing ignored when loss=focal", flush=True)
        weight = None
        if args.class_weight_fake > 1.0:
            weight = torch.tensor([1.0, args.class_weight_fake], dtype=torch.float32, device=device)
        return FocalLoss(gamma=args.focal_gamma, weight=weight)
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)


def train_one_epoch(
    model: XceptionDetectorLite,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
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


def clip_metrics_at_threshold(
    labels: list[int],
    probs: list[float],
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    """Binary clip metrics at a fixed fake threshold (default 0.5)."""
    tp = fp = fn = tn = 0
    for y, p in zip(labels, probs):
        pred_fake = p >= threshold
        if y == 1 and pred_fake:
            tp += 1
        elif y == 0 and pred_fake:
            fp += 1
        elif y == 1 and not pred_fake:
            fn += 1
        else:
            tn += 1
    fake_total = tp + fn
    real_total = tn + fp
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    fake_recall = tp / fake_total if fake_total > 0 else None
    real_recall = tn / real_total if real_total > 0 else None
    f1 = None
    if precision is not None and fake_recall is not None and (precision + fake_recall) > 0:
        f1 = 2 * precision * fake_recall / (precision + fake_recall)
    acc = (tp + tn) / max(1, len(labels))
    return {
        "val_acc": acc,
        "val_tp": tp,
        "val_fp": fp,
        "val_fn": fn,
        "val_tn": tn,
        "val_precision": precision,
        "val_fake_recall": fake_recall,
        "val_real_recall": real_recall,
        "val_f1": f1,
    }


@torch.no_grad()
def evaluate_val(
    model: XceptionDetectorLite,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    model.eval()
    all_labels: list[int] = []
    all_probs: list[float] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model.forward_logits(images)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    auc = None
    if len(set(all_labels)) > 1:
        auc = float(roc_auc_score(all_labels, all_probs))
    stats = clip_metrics_at_threshold(all_labels, all_probs, threshold=threshold)
    stats["val_auc"] = auc
    return stats


def parse_aug(value: str) -> set[str]:
    if not value or value.lower() in {"none", ""}:
        return set()
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def savepoint_path_for(output: Path) -> Path:
    return output.with_name(f"{output.stem}.savepoint.pt")


def train_config_snapshot(args: argparse.Namespace) -> dict:
    return {
        "weights": args.weights,
        "train_fake_dir": args.train_fake_dir,
        "train_real_dir": args.train_real_dir,
        "exclude_dirs": list(args.exclude_dirs),
        "max_per_class": args.max_per_class,
        "frames_per_video": args.frames_per_video,
        "num_frames": args.num_frames,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "crop_method": args.crop_method,
        "crop_padding": args.crop_padding,
        "crop_square": not args.no_crop_square,
        "loss": args.loss,
        "scheduler": args.scheduler,
        "val_metric": args.val_metric,
        "val_holdout": args.val_holdout,
        "aug": args.aug,
        "unfreeze_backbone": args.unfreeze_backbone,
    }


def write_running_meta(
    meta_path: Path,
    *,
    output: Path,
    weights_path: Path,
    args: argparse.Namespace,
    fake_dir: Path,
    real_dir: Path,
    excluded_count: int,
    train_count: int,
    val_count: int,
    use_cache: bool,
    cache_root: Path | None,
    num_workers: int,
    best_epoch: int,
    best_metric: float,
    history: list[dict],
    resumed_from_epoch: int,
    savepoint_path: Path,
    status: str,
) -> None:
    meta = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "init_weights": str(weights_path),
        "output": str(output),
        "savepoint": str(savepoint_path),
        "best_epoch": best_epoch,
        "best_val_metric": None if best_metric == float("-inf") else round(best_metric, 4),
        "val_metric": args.val_metric,
        "resumed_from_epoch": resumed_from_epoch,
        "train_fake_dir": str(fake_dir),
        "train_real_dir": str(real_dir),
        "exclude_dirs": args.exclude_dirs,
        "crop_method": args.crop_method,
        "crop_padding": args.crop_padding,
        "crop_square": not args.no_crop_square,
        "max_per_class": args.max_per_class,
        "val_holdout": args.val_holdout,
        "frames_per_video": args.frames_per_video,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma,
        "label_smoothing": args.label_smoothing,
        "scheduler": args.scheduler,
        "early_stop_patience": args.early_stop_patience,
        "augment": sorted(parse_aug(args.aug)),
        "unfreeze_backbone": args.unfreeze_backbone,
        "crop_cache": use_cache,
        "crop_cache_dir": None if cache_root is None else str(cache_root),
        "num_workers": num_workers,
        "num_train_clips": train_count,
        "num_val_clips": val_count,
        "excluded_benchmark_paths": excluded_count,
        "history": history,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_savepoint(
    path: Path,
    *,
    last_completed_epoch: int,
    best_epoch: int,
    best_metric: float,
    val_metric_name: str,
    stale_epochs: int,
    history: list[dict],
    model: nn.Module,
    best_state: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    plateau: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    train_config: dict,
) -> None:
    payload = {
        "version": 1,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "last_completed_epoch": last_completed_epoch,
        "best_epoch": best_epoch,
        "best_val_metric": None if best_metric == float("-inf") else float(best_metric),
        "val_metric": val_metric_name,
        "stale_epochs": stale_epochs,
        "history": history,
        "model_state": model.state_dict(),
        "best_state": best_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "plateau_state": plateau.state_dict() if plateau is not None else None,
        "train_config": train_config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_savepoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def configs_compatible(saved: dict, current: dict) -> list[str]:
    keys = (
        "crop_method",
        "crop_padding",
        "crop_square",
        "max_per_class",
        "val_holdout",
        "seed",
        "val_metric",
        "loss",
        "scheduler",
    )
    mismatches: list[str] = []
    for key in keys:
        if saved.get(key) != current.get(key):
            mismatches.append(f"{key}: savepoint={saved.get(key)!r} current={current.get(key)!r}")
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Short Xception fine-tune (step 3)")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="Initial checkpoint (DeepfakeBench xception_best.pth)",
    )
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
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--crop-method", default="mediapipe", choices=["haar", "mediapipe"])
    parser.add_argument("--crop-padding", type=float, default=0.3)
    parser.add_argument("--no-crop-square", action="store_true")
    parser.add_argument("--loss", default="focal", choices=["ce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--class-weight-fake", type=float, default=1.0)
    parser.add_argument("--scheduler", default="cosine", choices=["none", "cosine", "plateau"])
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--val-holdout", type=int, default=40)
    parser.add_argument("--val-metric", default="auc", choices=["auc", "fake_recall"])
    parser.add_argument("--aug", default="jpeg,flip", help="comma list: jpeg,blur,flip or none")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="face crop cache root (default: results/cache/xception_finetune/<crop-spec>)",
    )
    parser.add_argument("--no-cache", action="store_true", help="decode+crop on every sample (slow)")
    parser.add_argument("--rebuild-cache", action="store_true", help="ignore existing crop cache files")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1: 2 with cache, 0 without)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="resume from .savepoint.pt if present (default: auto when savepoint exists)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore/delete savepoint and train from --weights init",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    fake_dir = pick_fake_train_dir(root, args.train_fake_dir)
    real_dir = resolve_train_dir(root, args.train_real_dir, label="real")
    weights_path = resolve(root, args.weights)
    output = resolve(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    augment = parse_aug(args.aug)
    savepoint_path = savepoint_path_for(output)
    meta_path = output.with_suffix(".meta.json")
    train_config = train_config_snapshot(args)

    if args.fresh and savepoint_path.is_file():
        savepoint_path.unlink()
        print(f"removed savepoint: {savepoint_path}")

    resume = args.resume
    if resume is None:
        resume = savepoint_path.is_file() and not args.fresh

    cropper = create_face_cropper(
        method=args.crop_method,
        padding=args.crop_padding,
        square=not args.no_crop_square,
    )
    use_cache = not args.no_cache
    cache_root = None
    if use_cache:
        cache_root = crop_cache_root(
            root,
            crop_method=args.crop_method,
            crop_padding=args.crop_padding,
            crop_square=not args.no_crop_square,
            num_frames=args.num_frames,
            cache_dir=args.cache_dir,
        )
    num_workers = args.num_workers
    if num_workers < 0:
        num_workers = 2 if use_cache else 0
    try:
        excluded = collect_exclude_paths(root, args.exclude_dirs)
        all_samples = list_train_videos(
            fake_dir,
            real_dir,
            excluded,
            args.max_per_class,
            args.seed,
            cropper,
            num_frames=args.num_frames,
            cache_root=cache_root,
            rebuild_cache=args.rebuild_cache,
        )
        if len(all_samples) < 16:
            raise SystemExit(f"Too few training videos ({len(all_samples)}). Check FF++ paths.")

        train_samples, val_samples = split_train_val(all_samples, args.val_holdout, args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pin_memory = device.type == "cuda"

        if use_cache and cache_root is not None:
            print(f"crop cache: {cache_root}", flush=True)
            ensure_crop_cache_for_samples(
                train_samples + val_samples,
                cropper,
                cache_root,
                num_frames=args.num_frames,
                rebuild=args.rebuild_cache,
            )
            train_ds: Dataset = CachedFaceFrameDataset(
                train_samples,
                cache_root,
                num_frames=args.num_frames,
                frames_per_video=args.frames_per_video,
                seed=args.seed,
                augment=augment,
                train=True,
            )
            val_ds: Dataset | None = None
            if val_samples:
                val_ds = CachedFaceFrameDataset(
                    val_samples,
                    cache_root,
                    num_frames=args.num_frames,
                    frames_per_video=max(2, args.frames_per_video // 2),
                    seed=args.seed + 2,
                    augment=set(),
                    train=False,
                )
        else:
            train_ds = FaceFrameDataset(
                train_samples,
                cropper,
                num_frames=args.num_frames,
                frames_per_video=args.frames_per_video,
                seed=args.seed,
                augment=augment,
                train=True,
            )
            val_ds = None
            if val_samples:
                val_ds = FaceFrameDataset(
                    val_samples,
                    cropper,
                    num_frames=args.num_frames,
                    frames_per_video=max(2, args.frames_per_video // 2),
                    seed=args.seed + 2,
                    augment=set(),
                    train=False,
                )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        model = XceptionDetectorLite().to(device)
        set_trainable(model, unfreeze_backbone=args.unfreeze_backbone)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
        criterion = build_criterion(args, device)

        scheduler = None
        plateau = None
        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-6
            )
        elif args.scheduler == "plateau":
            plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=1, factor=0.5
            )

        print(f"init weights: {weights_path}")
        print(f"train fake: {fake_dir}")
        print(f"train real: {real_dir}")
        print(f"crop: {args.crop_method} padding={args.crop_padding} square={not args.no_crop_square}")
        print(f"excluded benchmark paths: {len(excluded)}")
        print(
            f"clips: train={len(train_samples)} val={len(val_samples)} "
            f"(fake train={sum(1 for _, y in train_samples if y == 1)}, "
            f"real train={sum(1 for _, y in train_samples if y == 0)})"
        )
        print(f"device: {device}")
        print(f"crop cache: {use_cache} (workers={num_workers})")
        print(
            f"epochs≤{args.epochs} batch={args.batch_size} lr={args.lr} "
            f"loss={args.loss} scheduler={args.scheduler} aug={sorted(augment)}"
        )
        print(f"unfreeze_backbone: {args.unfreeze_backbone}")
        print(f"early_stop patience={args.early_stop_patience} val_metric={args.val_metric}")
        print("val clip metrics: threshold=0.5 (val_auc + val_precision/val_fake_recall/val_f1/val_fp)")
        print(f"savepoint: {savepoint_path} (resume={'on' if resume else 'off'})")
        print()

        history: list[dict] = []
        best_metric = float("-inf")
        best_state: dict | None = None
        best_epoch = 0
        stale_epochs = 0
        start_epoch = 1
        resumed_from_epoch = 0

        if resume and savepoint_path.is_file():
            ckpt = load_savepoint(savepoint_path, device)
            mismatches = configs_compatible(ckpt.get("train_config", {}), train_config)
            if mismatches:
                print("savepoint config mismatch (use --fresh to restart):", flush=True)
                for line in mismatches:
                    print(f"  - {line}", flush=True)
                raise SystemExit(1)
            model.load_state_dict(ckpt["model_state"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if scheduler is not None and ckpt.get("scheduler_state"):
                scheduler.load_state_dict(ckpt["scheduler_state"])
            if plateau is not None and ckpt.get("plateau_state"):
                plateau.load_state_dict(ckpt["plateau_state"])
            history = list(ckpt.get("history", []))
            best_epoch = int(ckpt.get("best_epoch", 0))
            best_metric = float(ckpt.get("best_val_metric") or float("-inf"))
            stale_epochs = int(ckpt.get("stale_epochs", 0))
            best_state = ckpt.get("best_state")
            last_done = int(ckpt.get("last_completed_epoch", 0))
            resumed_from_epoch = last_done
            start_epoch = last_done + 1
            print(
                f"resumed savepoint: last_completed_epoch={last_done} "
                f"best_epoch={best_epoch} best_{args.val_metric}={best_metric} "
                f"stale_epochs={stale_epochs}",
                flush=True,
            )
            if best_state is not None and output.is_file():
                print(f"best weights on disk: {output}", flush=True)
            elif best_state is not None:
                torch.save(best_state, output)
                print(f"restored best weights: {output} (epoch {best_epoch})", flush=True)
        else:
            ckpt = torch.load(weights_path, map_location=device)
            model.load_state_dict(ckpt, strict=True)

        if start_epoch > args.epochs:
            print(
                f"savepoint already completed {resumed_from_epoch}/{args.epochs} epochs; "
                f"delete savepoint or use --fresh to retrain",
                flush=True,
            )
            if best_state is not None:
                torch.save(best_state, output)
            raise SystemExit(0)

        for epoch in range(start_epoch, args.epochs + 1):
            loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            row: dict = {
                "epoch": epoch,
                "loss": round(loss, 4),
                "train_acc": round(acc, 4),
                "lr": round(optimizer.param_groups[0]["lr"], 8),
            }
            if val_loader is not None:
                val_stats = evaluate_val(model, val_loader, device)
                row.update(
                    {
                        "val_acc": round(float(val_stats["val_acc"]), 4),
                        "val_auc": (
                            None
                            if val_stats["val_auc"] is None
                            else round(float(val_stats["val_auc"]), 4)
                        ),
                        "val_precision": (
                            None
                            if val_stats["val_precision"] is None
                            else round(float(val_stats["val_precision"]), 4)
                        ),
                        "val_fake_recall": (
                            None
                            if val_stats["val_fake_recall"] is None
                            else round(float(val_stats["val_fake_recall"]), 4)
                        ),
                        "val_real_recall": (
                            None
                            if val_stats["val_real_recall"] is None
                            else round(float(val_stats["val_real_recall"]), 4)
                        ),
                        "val_f1": (
                            None if val_stats["val_f1"] is None else round(float(val_stats["val_f1"]), 4)
                        ),
                        "val_fp": int(val_stats["val_fp"]),
                        "val_fn": int(val_stats["val_fn"]),
                    }
                )
                metric_key = "val_auc" if args.val_metric == "auc" else "val_fake_recall"
                metric_val = val_stats[metric_key]
                if metric_val is not None and metric_val > best_metric:
                    best_metric = metric_val
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    stale_epochs = 0
                    row["best"] = True
                    torch.save(best_state, output)
                    print(
                        f"  >> savepoint best epoch {best_epoch} "
                        f"{args.val_metric}={round(best_metric, 4)} -> {output.name}",
                        flush=True,
                    )
                else:
                    stale_epochs += 1
                    row["best"] = False
                if plateau is not None and metric_val is not None:
                    plateau.step(metric_val)
            else:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch

            history.append(row)
            parts = [f"epoch {epoch}/{args.epochs}", f"loss={row['loss']}", f"train_acc={row['train_acc']}"]
            if "val_auc" in row:
                parts.append(f"val_auc={row['val_auc']}")
            if "val_precision" in row and row["val_precision"] is not None:
                parts.append(f"val_prec={row['val_precision']}")
            if "val_fake_recall" in row and row["val_fake_recall"] is not None:
                parts.append(f"val_fake_recall={row['val_fake_recall']}")
            if "val_f1" in row and row["val_f1"] is not None:
                parts.append(f"val_f1={row['val_f1']}")
            if "val_real_recall" in row and row["val_real_recall"] is not None:
                parts.append(f"val_real_recall={row['val_real_recall']}")
            if "val_fp" in row:
                parts.append(f"val_fp={row['val_fp']}")
            print(" ".join(parts), flush=True)

            if best_state is None:
                best_state = copy.deepcopy(model.state_dict())
            write_savepoint(
                savepoint_path,
                last_completed_epoch=epoch,
                best_epoch=best_epoch,
                best_metric=best_metric,
                val_metric_name=args.val_metric,
                stale_epochs=stale_epochs,
                history=history,
                model=model,
                best_state=best_state,
                optimizer=optimizer,
                scheduler=scheduler,
                plateau=plateau,
                train_config=train_config,
            )
            write_running_meta(
                meta_path,
                output=output,
                weights_path=weights_path,
                args=args,
                fake_dir=fake_dir,
                real_dir=real_dir,
                excluded_count=len(excluded),
                train_count=len(train_samples),
                val_count=len(val_samples),
                use_cache=use_cache,
                cache_root=cache_root,
                num_workers=num_workers,
                best_epoch=best_epoch,
                best_metric=best_metric,
                history=history,
                resumed_from_epoch=resumed_from_epoch,
                savepoint_path=savepoint_path,
                status="running",
            )

            if scheduler is not None:
                scheduler.step()
            if val_loader is not None and args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
                print(f"early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
                break

        if best_state is None:
            best_state = model.state_dict()
        torch.save(best_state, output)

        write_running_meta(
            meta_path,
            output=output,
            weights_path=weights_path,
            args=args,
            fake_dir=fake_dir,
            real_dir=real_dir,
            excluded_count=len(excluded),
            train_count=len(train_samples),
            val_count=len(val_samples),
            use_cache=use_cache,
            cache_root=cache_root,
            num_workers=num_workers,
            best_epoch=best_epoch,
            best_metric=best_metric,
            history=history,
            resumed_from_epoch=resumed_from_epoch,
            savepoint_path=savepoint_path,
            status="completed",
        )
        print()
        print(f"saved weights: {output} (best epoch {best_epoch})")
        print(f"saved meta:    {meta_path}")
        print(f"savepoint:   {savepoint_path} (resume next run or --fresh to restart)")
    finally:
        cropper.close()


if __name__ == "__main__":
    main()
