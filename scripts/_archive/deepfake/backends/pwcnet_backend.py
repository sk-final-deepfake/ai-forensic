from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .base import OpticalFlowBackend
from .pwcnet_model import load_pwcnet


class PwcnetBackend(OpticalFlowBackend):
    """PWC-Net backend using pure PyTorch (no CUDA correlation build)."""

    def __init__(self, weights_dir: Path, device: str | None = None) -> None:
        self.weights_dir = Path(weights_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights_path = self._resolve_weights_path()
        self.model = load_pwcnet(str(weights_path) if weights_path else None, self.device)

    @property
    def name(self) -> str:
        return "pwcnet"

    def _resolve_weights_path(self) -> Path | None:
        candidates = [
            self.weights_dir / "pwcnet.pth",
            self.weights_dir / "network-default.pth",
            self.weights_dir / "pwc_net_chairs.pth",
        ]
        for path in candidates:
            if path.is_file():
                return path
        return None

    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        h, w = frame1.shape[:2]
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        if pad_h or pad_w:
            frame1 = cv2.copyMakeBorder(frame1, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)
            frame2 = cv2.copyMakeBorder(frame2, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)

        im1 = self._to_tensor(frame1)
        im2 = self._to_tensor(frame2)
        with torch.no_grad():
            flow = self.model(im1, im2)[0].permute(1, 2, 0).cpu().numpy()
        return flow[:h, :w, :]

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor
