"""RAFT / GMFlow optical-flow backends (single-model benchmark)."""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from optical_flow_common import flow_to_numpy, summarize_flow


# Official RAFT repo script: download_models.sh (direct .pth link is dead/404)
RAFT_MODELS_ZIP_URL = "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"


def _to_tensor(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image_rgb).permute(2, 0, 1).float()[None].to(device)


def _torch_load(path: Path, device: torch.device):
    return torch.load(path, map_location=device, weights_only=False)


def _weights_look_valid(path: Path, *, min_bytes: int = 5_000_000) -> bool:
    return path.is_file() and path.stat().st_size >= min_bytes


def ensure_raft_weights(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _weights_look_valid(path):
        return path
    if path.is_file() and path.stat().st_size < 5_000_000:
        path.unlink()

    print(f"downloading RAFT models.zip -> {path.parent}", flush=True)
    zip_path = path.parent / "models.zip"
    urllib.request.urlretrieve(RAFT_MODELS_ZIP_URL, zip_path)
    if not zip_path.is_file() or zip_path.stat().st_size < 1_000_000:
        raise RuntimeError(f"RAFT models.zip download failed: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path.parent)

    candidates = [
        path,
        path.parent / "models" / "raft-things.pth",
        path.parent / "raft-things.pth",
    ]
    for candidate in candidates:
        if _weights_look_valid(candidate):
            if candidate != path:
                path.write_bytes(candidate.read_bytes())
                if candidate != path:
                    candidate.unlink(missing_ok=True)
            zip_path.unlink(missing_ok=True)
            return path

    raise RuntimeError(f"RAFT weights not found after unzip under {path.parent}")


class RaftBackend:
    name = "raft"

    def __init__(self, root: Path, device: torch.device):
        self.root = root
        self.device = device
        self.vendor = root / "vendor/optical-flow/RAFT"
        self.weights = root / "models/test/video/optical-flow/raft/raft-things.pth"
        self.model = None
        self.padder_cls = None

    def load(self) -> None:
        if not self.vendor.is_dir():
            raise FileNotFoundError(f"missing RAFT repo: {self.vendor}")
        ensure_raft_weights(self.weights)
        sys.path.insert(0, str(self.vendor / "core"))
        from raft import RAFT  # type: ignore
        from utils.utils import InputPadder  # type: ignore

        args = Namespace(
            small=False,
            mixed_precision=False,
            alternate_corr=False,
            dropout=0,
            corr_levels=4,
            corr_radius=4,
        )
        model = RAFT(args)
        state = _torch_load(self.weights, self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        self.model = model.to(self.device).eval()
        self.padder_cls = InputPadder
        print(f"RAFT loaded: {self.weights} ({self.weights.stat().st_size // 1024 // 1024}MB)", flush=True)

    def infer_pair(self, img1: np.ndarray, img2: np.ndarray) -> dict:
        if self.model is None:
            self.load()
        t1 = _to_tensor(img1, self.device)
        t2 = _to_tensor(img2, self.device)
        with torch.no_grad():
            padder = self.padder_cls(t1.shape)
            t1, t2 = padder.pad(t1, t2)
            _, flow_up = self.model(t1, t2, iters=20, test_mode=True)
            flow = padder.unpad(flow_up)
        return summarize_flow(flow_to_numpy(flow))


class GmflowBackend:
    name = "gmflow"

    def __init__(self, root: Path, device: torch.device):
        self.root = root
        self.device = device
        self.vendor = root / "vendor/optical-flow/gmflow"
        preferred = root / "models/test/video/optical-flow/gmflow/pretrained/gmflow_things-e9887eda.pth"
        self.weights = preferred if preferred.is_file() else root / "models/test/video/optical-flow/gmflow/gmflow_things-0c07dcb3.pth"
        self.model = None

    def load(self) -> None:
        if not self.vendor.is_dir():
            raise FileNotFoundError(f"missing GMFlow repo: {self.vendor}")
        if not self.weights.is_file():
            matches = sorted((self.root / "models/test/video/optical-flow/gmflow").rglob("gmflow*.pth"))
            if not matches:
                raise FileNotFoundError("missing GMFlow weights under models/test/video/optical-flow/gmflow")
            self.weights = matches[0]
        sys.path.insert(0, str(self.vendor))
        from gmflow.gmflow import GMFlow  # type: ignore

        model = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type="swin",
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        ).to(self.device)
        checkpoint = _torch_load(self.weights, self.device)
        weights = checkpoint.get("model", checkpoint)
        model.load_state_dict(weights, strict=False)
        self.model = model.eval()
        print(f"GMFlow loaded: {self.weights}", flush=True)

    def infer_pair(self, img1: np.ndarray, img2: np.ndarray) -> dict:
        if self.model is None:
            self.load()
        t1 = _to_tensor(img1, self.device)
        t2 = _to_tensor(img2, self.device)
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast(enabled=True):
                    results = self.model(
                        t1,
                        t2,
                        attn_splits_list=[2],
                        corr_radius_list=[-1],
                        prop_radius_list=[-1],
                        pred_bidir_flow=False,
                    )
            else:
                results = self.model(
                    t1,
                    t2,
                    attn_splits_list=[2],
                    corr_radius_list=[-1],
                    prop_radius_list=[-1],
                    pred_bidir_flow=False,
                )
            flow = results["flow_preds"][-1]
        return summarize_flow(flow_to_numpy(flow))


BACKENDS = {
    "raft": RaftBackend,
    "gmflow": GmflowBackend,
}
