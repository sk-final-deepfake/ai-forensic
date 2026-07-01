"""
ForenShield video-forgery frame list loader for TruFor_train_test.

Copy to:
  vendor/TruFor/TruFor_train_test/dataset/dataset_ForenShieldVideo.py

List format (relative paths under cache root):
  frames/foo.jpg,masks/foo.png
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from dataset.AbstractDataset import AbstractDataset


class ForenShieldVideo(AbstractDataset):
    def __init__(
        self,
        cache_root: str | Path,
        list_file: str | Path,
        crop_size,
        grid_crop: bool,
        max_dim=None,
        aug=None,
    ):
        super().__init__(crop_size, grid_crop, max_dim, aug=aug)
        self._root_path = Path(cache_root)
        list_path = Path(list_file)
        if not list_path.is_absolute():
            # allow path relative to TruFor_train_test cwd
            list_path = Path.cwd() / list_path
        with open(list_path, "r", encoding="utf-8") as f:
            self.img_list = [t.strip().split(",") for t in f.readlines() if t.strip()]

    def get_img(self, index):
        assert 0 <= index < len(self.img_list), f"Index {index} is not available!"
        rgb_rel, mask_rel = self.img_list[index]
        rgb_path = self._root_path / rgb_rel
        mask_path = self._root_path / mask_rel
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 0).astype(np.uint8)
        assert os.path.isfile(rgb_path)
        return self._create_tensor(mask=mask, rgb_path=str(rgb_path))
