from __future__ import annotations

import sys
from pathlib import Path

from gpu_worker.config import WorkerConfig


def setup_script_paths(cfg: WorkerConfig) -> Path:
    """deepfake infer + project eval scripts를 import path에 추가."""
    candidates = [
        cfg.deepfake_root / "scripts" / "infer",
        cfg.project_root / "scripts" / "eval",
        cfg.project_root / "deepfake" / "scripts" / "infer",
    ]
    for path in candidates:
        if path.is_dir():
            resolved = str(path.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
    return cfg.deepfake_root


def resolve_under_root(cfg: WorkerConfig, relative: str) -> Path:
    rel = relative.strip()
    if not rel:
        raise ValueError("empty relative path")
    candidate = Path(rel)
    if candidate.is_file():
        return candidate.resolve()
    for base in (cfg.project_root, cfg.deepfake_root):
        path = (base / rel).resolve()
        if path.is_file():
            return path
    return (cfg.project_root / rel).resolve()
