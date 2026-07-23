from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class OpticalFlowBackend(ABC):
    """Optical flow model adapter (RAFT / GMFlow / PWC-Net)."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow between two RGB uint8 frames (H, W, 3).

        Returns:
            flow array (H, W, 2) in pixel units.
        """
        raise NotImplementedError
