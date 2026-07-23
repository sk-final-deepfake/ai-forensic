"""
Pure PyTorch PWC-Net style network (no custom CUDA correlation extension).

Avoids legacy correlation layer build failures documented in VIDEO_DEEPFAKE_MODEL_BENCHMARK_3x3.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cost_volume(c1: torch.Tensor, c2: torch.Tensor, max_displacement: int) -> torch.Tensor:
    """Build cost volume with pure PyTorch ops (no CUDA correlation module)."""
    batch, channels, height, width = c1.shape
    cost = []
    for dy in range(-max_displacement, max_displacement + 1):
        for dx in range(-max_displacement, max_displacement + 1):
            shifted = torch.zeros_like(c2)
            y0 = max(0, dy)
            y1 = min(height, height + dy)
            x0 = max(0, dx)
            x1 = min(width, width + dx)
            sy0 = max(0, -dy)
            sy1 = sy0 + (y1 - y0)
            sx0 = max(0, -dx)
            sx1 = sx0 + (x1 - x0)
            shifted[:, :, y0:y1, x0:x1] = c2[:, :, sy0:sy1, sx0:sx1]
            cost.append((c1 * shifted).sum(dim=1, keepdim=True))
    return torch.cat(cost, dim=1)


class FlowEstimatorDense(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 96, 3, padding=1)
        self.conv4 = nn.Conv2d(96, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.predict = nn.Conv2d(32, 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return self.predict(x)


class PWCNet(nn.Module):
    """Lightweight multi-scale flow estimator compatible with optional external weights."""

    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=2, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.estimator = FlowEstimatorDense((2 * 4 + 1) ** 2)
        self.upsample = nn.ConvTranspose2d(2, 2, 4, stride=4, padding=0)

    def forward(self, im1: torch.Tensor, im2: torch.Tensor) -> torch.Tensor:
        f1 = self.feature(im1)
        f2 = self.feature(im2)
        cost = _cost_volume(f1, f2, max_displacement=4)
        flow = self.estimator(cost)
        flow = self.upsample(flow)
        return flow


def load_pwcnet(weights_path: str | None, device: torch.device) -> PWCNet:
    model = PWCNet().to(device)
    model.eval()
    if not weights_path:
        return model

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if isinstance(state_dict, dict):
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)
    return model
