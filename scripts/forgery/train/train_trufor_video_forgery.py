#!/usr/bin/env python3
"""ForenShield TruFor video forgery fine-tune entrypoint.

Vendor TruFor train.py(CASIA/IMD) 대신 사용합니다.
영상 학습셋 → prepare_trufor_video_frames.py 로 프레임+weak mask 생성 후
TruFor segmentation head 를 fine-tune 합니다.

필수 사전 작업 (서버, 1회):
  1) vendor_patches/dataset_ForenShieldVideo.py 를
     vendor/TruFor/TruFor_train_test/dataset/ 에 복사
  2) data_core.py train/valid 블록에 FSVIDEO 등록 (vendor_patches/data_core_FSVIDEO.patch 참고)
  3) vendor_patches/trufor_forgery_video.yaml 을
     vendor/TruFor/TruFor_train_test/lib/config/ 에 복사
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# TruFor vendor code uses removed NumPy 1.x aliases (np.int, etc.)
for _name, _target in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _target)
import torch
import torch.backends.cudnn as cudnn

FORGERY_ROOT = Path(__file__).resolve().parents[2]  # .../forgery
TRUFOR_ROOT = FORGERY_ROOT / "vendor" / "TruFor" / "TruFor_train_test"


def _torch_load_checkpoint(path: str):
    """PyTorch 2.6+ defaults weights_only=True; TruFor checkpoints need False."""
    map_location = lambda storage, loc: storage
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _ensure_trufor_path() -> None:
    trufor = str(TRUFOR_ROOT)
    if trufor not in sys.path:
        sys.path.insert(0, trufor)
    os.chdir(trufor)


def _register_fsvideo_dataset(cache_root: Path, train_list: Path, valid_list: Path) -> None:
    """Register FSVIDEO without editing data_core.py when patch not applied."""
    import dataset.data_core as data_core
    from dataset.dataset_ForenShieldVideo import ForenShieldVideo

    cache_root = cache_root.resolve()
    train_list = train_list.resolve()
    valid_list = valid_list.resolve()

    original_init = data_core.myDataset.__init__

    def patched_init(self, config, crop_size, grid_crop, mode="train", max_dim=None, aug=None):
        self.dataset_list = []
        training_set = config.DATASET.TRAIN
        valid_set = config.DATASET.VALID

        if mode == "train" and "FSVIDEO" in training_set:
            self.dataset_list.append(
                ForenShieldVideo(
                    cache_root=cache_root,
                    list_file=train_list,
                    crop_size=crop_size,
                    grid_crop=grid_crop,
                    aug=aug,
                )
            )
            for tag in training_set:
                if tag != "FSVIDEO":
                    logging.warning("Ignoring unsupported train dataset tag: %s", tag)
        elif mode == "valid" and "FSVIDEO" in valid_set:
            self.dataset_list.append(
                ForenShieldVideo(
                    cache_root=cache_root,
                    list_file=valid_list,
                    crop_size=crop_size,
                    grid_crop=grid_crop,
                    max_dim=max_dim,
                    aug=aug,
                )
            )
            for tag in valid_set:
                if tag != "FSVIDEO":
                    logging.warning("Ignoring unsupported valid dataset tag: %s", tag)
        else:
            return original_init(self, config, crop_size, grid_crop, mode=mode, max_dim=max_dim, aug=aug)

        self.crop_size = crop_size
        self.grid_crop = grid_crop
        self.mode = mode
        lengths = [len(ds) for ds in self.dataset_list]
        self.smallest = min(lengths)
        if config.TRAIN.NUM_SAMPLES > 0 and config.TRAIN.NUM_SAMPLES < self.smallest:
            self.smallest = config.TRAIN.NUM_SAMPLES

    data_core.myDataset.__init__ = patched_init


def main() -> None:
    parser = argparse.ArgumentParser(description="ForenShield TruFor video forgery fine-tune")
    parser.add_argument(
        "-exp",
        "--experiment",
        type=str,
        default="trufor_forgery_video",
        help="TruFor config yaml stem → vendor/.../lib/config/<exp>.yaml (NOT a free-form run id)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Log/checkpoint folder name under log/train/ (default: forgery-YYYYMMDD-HHMM)",
    )
    parser.add_argument("-g", "--gpu", type=int, default=[0], nargs="+")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=FORGERY_ROOT / "data/processed/trufor-gmflow-train-400",
    )
    parser.add_argument("--train-list", type=Path, default=None)
    parser.add_argument("--valid-list", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=TRUFOR_ROOT / "lib/config/trufor_forgery_video.yaml",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help="Optional existing TruFor weights (e.g. models/test/spatial/trufor/v1.0.0/trufor.pth.tar)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from log/train/<run-name>/checkpoint.pth.tar (same --run-name required)",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="TruFor config overrides, e.g. TRAIN.END_EPOCH 3")
    args = parser.parse_args()

    def _resolve_forgery_path(p: Path) -> Path:
        if p.is_absolute():
            return p.resolve()
        return (FORGERY_ROOT / p).resolve()

    cache_root = _resolve_forgery_path(args.cache_root)
    train_list = (args.train_list or cache_root / "train_list.txt").resolve()
    valid_list = (args.valid_list or cache_root / "valid_list.txt").resolve()
    pretrained_ckpt = _resolve_forgery_path(args.pretrained_checkpoint) if args.pretrained_checkpoint else None
    if not train_list.exists() or not valid_list.exists():
        raise FileNotFoundError(
            f"Missing list files. Run prepare_trufor_video_frames.py first.\n"
            f"  train: {train_list}\n  valid: {valid_list}"
        )

    _ensure_trufor_path()
    _register_fsvideo_dataset(cache_root, train_list, valid_list)

    config_yaml = TRUFOR_ROOT / "lib" / "config" / f"{args.experiment}.yaml"
    if not config_yaml.is_file():
        raise FileNotFoundError(
            f"TruFor config not found: {config_yaml}\n"
            f"Copy vendor_patches/trufor_forgery_video.yaml to that path.\n"
            f"Use -exp trufor_forgery_video (yaml file stem, not a dated run name)."
        )

    run_name = args.run_name or datetime.now().strftime("forgery-%Y%m%d-%H%M")

    from lib.config import config, update_config
    from lib.core.function import train, validate
    from lib.utils import FullModel, adjust_learning_rate, create_logger, get_model, get_optimizer
    from dataset.data_core import myDataset

    from types import SimpleNamespace

    class _DummySummaryWriter:
        def add_scalar(self, *args, **kwargs):
            pass

        def add_image(self, *args, **kwargs):
            pass

        def close(self):
            pass

    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        SummaryWriter = _DummySummaryWriter
        logging.warning("tensorboardX not installed; TensorBoard logs disabled")

    def _load_aug(path):
        if not path:
            return None
        try:
            import albumentations
        except AttributeError as exc:
            if "sctypes" in str(exc):
                raise RuntimeError(
                    "albumentations/imgaug is incompatible with NumPy 2.x. "
                    "Use TRAIN.AUG null VALID.AUG null (trufor_forgery_video.yaml) "
                    "or pip install 'numpy<2.0'."
                ) from exc
            raise
        return albumentations.load(path, data_format="yaml")

    cfg_opts = []
    if pretrained_ckpt:
        cfg_opts.extend(["TRAIN.PRETRAINING", str(pretrained_ckpt)])
    if args.resume:
        cfg_opts.append("TRAIN.RESUME")
        cfg_opts.append("true")
    cfg_opts.extend(args.opts)

    update_config(
        config,
        SimpleNamespace(experiment=args.experiment, gpu=args.gpu, opts=cfg_opts),
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.gpu)
    gpus = list(config.GPUS)

    logger, final_output_dir, tb_log_dir = create_logger(config, run_name, "train")
    logger.info(config)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    writer_dict = {
        "writer": SummaryWriter(tb_log_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    aug_train = _load_aug(config.TRAIN.AUG)
    aug_valid = _load_aug(config.VALID.AUG)

    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = myDataset(config, crop_size=crop_size, grid_crop=False, mode="train", aug=aug_train)
    valid_dataset = myDataset(config, crop_size=None, grid_crop=False, mode="valid", aug=aug_valid, max_dim=config.VALID.MAX_SIZE)

    logger.info(train_dataset.get_info())

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
    )
    validloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
    )

    model = get_model(config)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model = FullModel(model, config)
    optimizer = get_optimizer(model, config)

    best_key = config.VALID.BEST_KEY
    best_value = np.inf if "loss" in best_key else 0

    if config.TRAIN.PRETRAINING:
        ckpt_path = config.TRAIN.PRETRAINING
        assert os.path.isfile(ckpt_path), ckpt_path
        checkpoint = _torch_load_checkpoint(ckpt_path)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        try:
            model.model.module.load_state_dict(state_dict, strict=False)
        except Exception:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("detection")}
            model.model.module.load_state_dict(state_dict, strict=False)
        logger.info("Loaded pretraining: %s", ckpt_path)

    last_epoch = config.TRAIN.BEGIN_EPOCH
    if getattr(config.TRAIN, "RESUME", False):
        resume_path = os.path.join(final_output_dir, "checkpoint.pth.tar")
        if os.path.isfile(resume_path):
            checkpoint = _torch_load_checkpoint(resume_path)
            best_value = checkpoint["best_value"]
            assert checkpoint["best_key"] == best_key
            last_epoch = int(checkpoint["epoch"])
            model.model.module.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            writer_dict["train_global_steps"] = last_epoch
            logger.info("Resumed from %s (epoch %s)", resume_path, last_epoch)
        else:
            logger.warning("TRAIN.RESUME=true but no checkpoint: %s", resume_path)

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / max(len(gpus), 1))
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * max(epoch_iters, 1)

    for epoch in range(last_epoch, end_epoch):
        train_dataset.shuffle()
        logging.info("TRAINING epoch %s", epoch)
        train(
            epoch,
            config.TRAIN.END_EPOCH,
            epoch_iters,
            config.TRAIN.LR,
            num_iters,
            trainloader,
            optimizer,
            model,
            writer_dict,
            adjust_learning_rate=adjust_learning_rate,
        )

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1.0)

        logging.info("VALIDATION epoch %s", epoch)
        writer_dict["valid_global_steps"] = epoch
        value_valid, iou_array, confusion_matrix = validate(config, validloader, model, writer_dict, "valid")

        if "loss" in best_key:
            improved = value_valid[best_key] < best_value
        else:
            improved = value_valid[best_key] > best_value
        if improved:
            best_value = value_valid[best_key]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "best_value": best_value,
                    "best_key": best_key,
                    "state_dict": model.model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(final_output_dir, "best.pth.tar"),
            )
            logger.info("best.pth.tar updated (%s=%s)", best_key, best_value)

        logger.info("Valid loss=%.4f best_%s=%.4f", value_valid["loss"], best_key, best_value)
        logger.info("%s", iou_array)
        logger.info("confusion_matrix: %s", confusion_matrix)

        torch.save(
            {
                "epoch": epoch + 1,
                "best_value": best_value,
                "best_key": best_key,
                "state_dict": model.model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(final_output_dir, "checkpoint.pth.tar"),
        )

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1.0)

    logger.info("Done. artifacts under %s", final_output_dir)


if __name__ == "__main__":
    main()
