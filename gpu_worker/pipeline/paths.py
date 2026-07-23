from __future__ import annotations

import sys
from pathlib import Path

from gpu_worker.config import WorkerConfig


def setup_script_paths(cfg: WorkerConfig) -> Path:
    """Prefer ai-forensic (DEEPFAKE_ROOT) infer scripts over legacy FORENSHIELD tree."""
    legacy = [
        cfg.project_root / "scripts" / "infer",
        cfg.project_root / "scripts" / "eval",
        cfg.project_root.parent / "scripts" / "infer",
        cfg.project_root.parent / "scripts" / "eval",
    ]
    primary_infer = cfg.deepfake_root / "scripts" / "infer"
    primary_eval = cfg.deepfake_root / "scripts" / "eval"

    for path in legacy:
        if path.is_dir():
            resolved = str(path.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)

    for path in (primary_eval, primary_infer):
        if not path.is_dir():
            continue
        resolved = str(path.resolve())
        if resolved in sys.path:
            sys.path.remove(resolved)
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
