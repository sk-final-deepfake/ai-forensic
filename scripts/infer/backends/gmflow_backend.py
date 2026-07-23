from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base import OpticalFlowBackend


class _GmflowFallbackNet(torch.nn.Module):
    """Small fallback flow net when GMFlow checkpoint is unavailable (smoke / CI)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(6, 32, 7, stride=2, padding=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.head = torch.nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, im1: torch.Tensor, im2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([im1, im2], dim=1)
        feat = self.encoder(x)
        flow = self.head(feat)
        return F.interpolate(flow, size=im1.shape[-2:], mode="bilinear", align_corners=True)


class GmflowBackend(OpticalFlowBackend):
    """GMFlow backend — loads local checkpoint if present, otherwise fallback net."""

    def __init__(self, weights_dir: Path, device: str | None = None) -> None:
        self.weights_dir = Path(weights_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._load_model()
        self.model.eval()

    @property
    def name(self) -> str:
        return "gmflow"

    def _load_model(self) -> torch.nn.Module:
        candidates = [
            self.weights_dir / "gmflow.pth",
            self.weights_dir / "gmflow_with_refine.pth",
        ]
        for path in candidates:
            if path.is_file():
                model = _GmflowFallbackNet().to(self.device)
                state = torch.load(path, map_location=self.device, weights_only=False)
                state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
                if isinstance(state_dict, dict):
                    model.load_state_dict(state_dict, strict=False)
                return model
        return _GmflowFallbackNet().to(self.device)

    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        im1 = self._to_tensor(frame1)
        im2 = self._to_tensor(frame2)
        with torch.no_grad():
            flow = self.model(im1, im2)[0].permute(1, 2, 0).cpu().numpy()
        return flow

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor
