from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base import OpticalFlowBackend


def _pad_to_multiple_of_8(tensor: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Pad (B,C,H,W) so H and W are divisible by 8."""
    _, _, h, w = tensor.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return tensor, h, w


class RaftBackend(OpticalFlowBackend):
    """RAFT via torchvision pretrained weights."""

    def __init__(self, weights_dir: Path, device: str | None = None):
        self.weights_dir = Path(weights_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        local_ckpt = self.weights_dir / "raft-things.pth"
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device)

        if local_ckpt.is_file():
            state = torch.load(local_ckpt, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # raft-things.pth is original RAFT (module.fnet.*) — skip if incompatible
            if isinstance(state, dict) and not any(str(k).startswith("module.") for k in state):
                try:
                    self.model = raft_large(weights=None, progress=False).to(self.device)
                    self.model.load_state_dict(state, strict=True)
                except RuntimeError:
                    self.model = raft_large(
                        weights=Raft_Large_Weights.DEFAULT, progress=False
                    ).to(self.device)

        self.model.eval()

    @property
    def name(self) -> str:
        return "raft"

    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        im1 = self._to_tensor(frame1)
        im2 = self._to_tensor(frame2)
        im1, orig_h, orig_w = _pad_to_multiple_of_8(im1)
        im2, _, _ = _pad_to_multiple_of_8(im2)
        with torch.no_grad():
            flow = self.model(im1, im2)[-1][0].permute(1, 2, 0)
        flow = flow[:orig_h, :orig_w].cpu().numpy()
        return flow

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor
