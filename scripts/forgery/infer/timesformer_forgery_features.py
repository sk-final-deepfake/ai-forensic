"""TimeSformer K400 window features for forgery temporal MIL (train + infer)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import TimesformerModel

CLIP_FRAMES = 8
CLIP_SIZE = 224
DEFAULT_PRETRAINED = "facebook/timesformer-base-finetuned-k400"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class WindowMilMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


ClipMilMLP = WindowMilMLP


def setup_infer_imports(repo_root: Path) -> Path:
    forgery_root = repo_root / "forgery" if repo_root.name != "forgery" else repo_root
    for cand in (
        Path(__file__).resolve().parent,
        forgery_root / "scripts" / "infer",
        repo_root / "scripts" / "infer",
    ):
        if cand.is_dir() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
    return forgery_root


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def frames_to_pixel_values(
    frames_bgr: list[np.ndarray],
    device: torch.device,
    *,
    clip_frames: int = CLIP_FRAMES,
) -> torch.Tensor | None:
    if len(frames_bgr) < clip_frames:
        return None
    picked = frames_bgr[:clip_frames]
    arr = np.zeros((clip_frames, CLIP_SIZE, CLIP_SIZE, 3), dtype=np.float32)
    for i, bgr in enumerate(picked[:clip_frames]):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (CLIP_SIZE, CLIP_SIZE), interpolation=cv2.INTER_AREA)
        arr[i] = rgb.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to(device)


def pool_clip_embedding(model: TimesformerModel, pixel_values: torch.Tensor) -> torch.Tensor:
    """Mean-pool TimeSformer token embeddings -> [B, D] or [D] when B=1 squeezed later."""
    outputs = model(pixel_values=pixel_values)
    return outputs.last_hidden_state.mean(dim=1)


@torch.no_grad()
def embed_pixel_values(model: TimesformerModel, pixel_values: torch.Tensor) -> np.ndarray:
    model.eval()
    return pool_clip_embedding(model, pixel_values).squeeze(0).detach().cpu().numpy().astype(np.float32)


def set_timesformer_trainable_last_layers(model: TimesformerModel, unfreeze_layers: int = 1) -> int:
    """Freeze all params, then unfreeze the last N encoder blocks (+ final layernorm)."""
    for param in model.parameters():
        param.requires_grad = False
    layers = model.encoder.layer
    n = max(1, min(int(unfreeze_layers), len(layers)))
    for layer in layers[-n:]:
        for param in layer.parameters():
            param.requires_grad = True
    if getattr(model, "layernorm", None) is not None:
        for param in model.layernorm.parameters():
            param.requires_grad = True
    return n


def load_forgery_head_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ClipMilMLP, np.ndarray, np.ndarray, dict]:
    ckpt = torch.load(checkpoint_path.expanduser().resolve(), map_location=device, weights_only=False)
    embed_dim = int(ckpt.get("embed_dim", 768))
    mean = np.array(ckpt["mean"], dtype=np.float32)
    std = np.array(ckpt["std"], dtype=np.float32)
    head = ClipMilMLP(embed_dim, hidden=64).to(device)
    head.load_state_dict(ckpt["state_dict"])
    return head, mean, std, ckpt


def load_forgery_bundle(
    checkpoint_path: Path,
    device: torch.device,
    *,
    pretrained_id: str | None = None,
) -> tuple[TimesformerModel, ClipMilMLP, np.ndarray, np.ndarray, dict]:
    """Load finetuned (or frozen) backbone + clip head from ``forgery_head.pt``."""
    ckpt_path = checkpoint_path.expanduser().resolve()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pid = pretrained_id or ckpt.get("pretrained_id", DEFAULT_PRETRAINED)
    backbone = TimesformerModel.from_pretrained(pid).to(device)
    if ckpt.get("backbone_state_dict"):
        missing, unexpected = backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
        if missing or unexpected:
            print(
                f"backbone load_state_dict: missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
    head, mean, std, _ = load_forgery_head_checkpoint(ckpt_path, device)
    return backbone, head, mean, std, ckpt


def read_clip_frames_bgr(
    video_path: Path,
    *,
    clip_frames: int = CLIP_FRAMES,
    decode_cache: Path | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Probe-aligned: linspace 8 frames across full clip (after optional ffmpeg cache)."""
    from video_decode_robust import resolve_decodable_video_path  # noqa: WPS433

    src = video_path.expanduser().resolve()
    resolved, decode_method = resolve_decodable_video_path(
        src,
        decode_cache.expanduser().resolve() if decode_cache is not None else None,
        min_frames=clip_frames,
    )
    meta: dict[str, Any] = {
        "video_path": str(src),
        "video_path_used": str(resolved),
        "decode_method": decode_method,
        "clip_frames": clip_frames,
        "sampling": "linspace_full_clip_probe_aligned",
    }
    cap = cv2.VideoCapture(str(resolved))
    if not cap.isOpened():
        meta["error_reason"] = "opencv_open_failed"
        return [], meta
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    meta["total_frames"] = total
    if total <= 0:
        cap.release()
        meta["error_reason"] = "zero_frames"
        return [], meta
    if total <= clip_frames:
        indices = list(range(total))
    else:
        indices = [int(i * (total - 1) / (clip_frames - 1)) for i in range(clip_frames)]
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()
    if len(frames) < clip_frames:
        meta["error_reason"] = f"need>={clip_frames}_frames got={len(frames)}"
        return [], meta
    meta["frames_read"] = len(frames)
    return frames[:clip_frames], meta


@torch.no_grad()
def extract_video_clip_embedding(
    video_path: Path,
    model: TimesformerModel,
    device: torch.device,
    *,
    clip_frames: int = CLIP_FRAMES,
    decode_cache: Path | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    frames_bgr, meta = read_clip_frames_bgr(
        video_path,
        clip_frames=clip_frames,
        decode_cache=decode_cache,
    )
    if not frames_bgr:
        return None, meta
    pixel_values = frames_to_pixel_values(frames_bgr, device, clip_frames=clip_frames)
    if pixel_values is None:
        meta["error_reason"] = "tensor_build_failed"
        return None, meta
    emb = embed_pixel_values(model, pixel_values)
    meta["embedding_dim"] = int(emb.shape[0])
    return emb, meta


def aggregate_bag_probs(
    probs: torch.Tensor,
    mask: torch.Tensor,
    *,
    aggregate: str,
    top_k: int,
) -> torch.Tensor:
    probs = probs.masked_fill(~mask, 0.0)
    if aggregate == "max":
        return probs.max(dim=1).values
    k = max(1, min(top_k, probs.shape[1]))
    topk, _ = torch.topk(probs, k, dim=1)
    return topk.mean(dim=1)


def aggregate_bag_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    aggregate: str,
    top_k: int,
) -> torch.Tensor:
    """Top-k / max over window logits (train with BCEWithLogitsLoss on bag)."""
    logits = logits.masked_fill(~mask, -1e4)
    if aggregate == "max":
        return logits.max(dim=1).values
    k = max(1, min(top_k, logits.shape[1]))
    topk, _ = torch.topk(logits, k, dim=1)
    return topk.mean(dim=1)


def sample_temporal_windows(
    video_path: Path,
    *,
    window_sec: float = 1.0,
    stride_sec: float = 0.5,
    clip_frames: int = CLIP_FRAMES,
    max_side: int = 512,
    max_frames: int = 9000,
    decode_cache: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from video_decode_robust import (  # noqa: WPS433
        iter_window_starts,
        probe_video_fps_and_frames,
        read_rgb_frames_sequential,
        resolve_decodable_video_path,
    )

    src = video_path.expanduser().resolve()
    resolved, decode_method = resolve_decodable_video_path(
        src,
        decode_cache.expanduser().resolve() if decode_cache is not None else None,
        min_frames=clip_frames,
    )
    fps, _ = probe_video_fps_and_frames(resolved, probe_limit=max_frames + 2)
    rgb_frames = read_rgb_frames_sequential(resolved, max_frames=max_frames, max_side=max_side)
    total = len(rgb_frames)
    meta: dict[str, Any] = {
        "fps": fps,
        "total_frames": total,
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "clip_frames": clip_frames,
        "max_side": max_side,
        "video_path": str(src),
        "video_path_used": str(resolved),
        "decode_method": decode_method,
    }
    if total < clip_frames:
        meta["n_windows"] = 0
        meta["error_reason"] = f"need>={clip_frames}_frames got={total} decode={decode_method}"
        return [], meta

    windows = iter_window_starts(total, fps=fps, window_sec=window_sec, stride_sec=stride_sec)
    meta["n_windows"] = len(windows)
    out: list[dict[str, Any]] = []
    for start, end in windows:
        span = end - start
        if span < 2:
            continue
        if span <= clip_frames:
            local_idx = list(range(start, end))
            while len(local_idx) < clip_frames:
                local_idx.append(end - 1)
            local_idx = local_idx[:clip_frames]
        else:
            local_idx = [start + int(i) for i in np.linspace(0, span - 1, num=clip_frames, dtype=int)]
        frames_bgr = [rgb_to_bgr(rgb_frames[i]) for i in local_idx]
        out.append(
            {
                "frame_index_start": int(start),
                "frame_index_end": int(end),
                "frames_bgr": frames_bgr,
            }
        )
    meta["windows_sampled"] = len(out)
    return out, meta


@torch.no_grad()
def extract_video_window_embeddings(
    video_path: Path,
    model: TimesformerModel,
    device: torch.device,
    *,
    window_sec: float = 1.0,
    stride_sec: float = 0.5,
    clip_frames: int = CLIP_FRAMES,
    max_side: int = 512,
    decode_cache: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    windows, meta = sample_temporal_windows(
        video_path,
        window_sec=window_sec,
        stride_sec=stride_sec,
        clip_frames=clip_frames,
        max_side=max_side,
        decode_cache=decode_cache,
    )
    rows: list[dict[str, Any]] = []
    for win in windows:
        pixel_values = frames_to_pixel_values(win["frames_bgr"], device, clip_frames=clip_frames)
        if pixel_values is None:
            continue
        emb = embed_pixel_values(model, pixel_values)
        rows.append(
            {
                "frame_index_start": win["frame_index_start"],
                "frame_index_end": win["frame_index_end"],
                "embedding": emb,
            }
        )
    meta["windows_embedded"] = len(rows)
    return rows, meta


@torch.no_grad()
def score_windows_mil(
    per_window: list[dict[str, Any]],
    head: WindowMilMLP,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    *,
    aggregate: str,
    top_k: int,
) -> tuple[float | None, dict[str, Any]]:
    if not per_window:
        return None, {"n_windows": 0}
    head.eval()
    embs = np.stack([np.asarray(w["embedding"], dtype=np.float32) for w in per_window])
    embs = (embs - mean) / std
    x = torch.from_numpy(embs).to(device)
    logits = head(x)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    prob_t = torch.from_numpy(probs).unsqueeze(0)
    mask_t = torch.ones((1, probs.shape[0]), dtype=torch.bool)
    clip_prob = float(aggregate_bag_probs(prob_t, mask_t, aggregate=aggregate, top_k=top_k).item())
    top_windows = sorted(
        [
            {
                "start": int(w.get("frame_index_start", -1)),
                "end": int(w.get("frame_index_end", -1)),
                "score": round(float(p), 6),
            }
            for w, p in zip(per_window, probs.tolist())
        ],
        key=lambda row: row["score"],
        reverse=True,
    )[:5]
    return clip_prob, {"n_windows": len(per_window), "top_windows": top_windows}


def window_labels_for_video(
    per_window: list[dict],
    *,
    ground_truth_label: str,
    total_frames: int,
    fps: float,
    video_path: Path | str,
    relative_path: str | None = None,
    duration_sec: float = 1.0,
    label_mode: str = "segment",
) -> tuple[list[float], list[bool], dict]:
    """Delegate to tamper_segment_labels (single source of truth)."""
    from tamper_segment_labels import window_labels_for_video as _labels  # noqa: WPS433

    return _labels(
        per_window,
        ground_truth_label=ground_truth_label,
        total_frames=total_frames,
        fps=fps,
        video_path=video_path,
        relative_path=relative_path,
        duration_sec=duration_sec,
        label_mode=label_mode,
    )
