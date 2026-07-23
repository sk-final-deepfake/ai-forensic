from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def ai_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def infer_scripts_dir() -> Path:
    return ai_root() / "scripts" / "infer"


@lru_cache(maxsize=1)
def config_dir() -> Path:
    return ai_root() / "config"


def _infer_script_candidates() -> list[Path]:
    foren = os.getenv("FORENSHIELD_AI_ROOT", "").strip()
    deepfake = os.getenv("DEEPFAKE_ROOT", "").strip()
    candidates = [
        Path(foren) / "scripts" / "infer" if foren else None,
        Path(foren) / "scripts" / "eval" if foren else None,
        Path(foren).parent / "scripts" / "infer" if foren else None,
        Path(deepfake) / "scripts" / "infer" if deepfake else None,
        Path(deepfake) / "scripts" / "eval" if deepfake else None,
        infer_scripts_dir(),
        ai_root() / "scripts" / "eval",
    ]
    return [p for p in candidates if p is not None]


def ensure_infer_scripts_on_path() -> Path:
    """Prefer ai-forensic scripts/infer over legacy FORENSHIELD tree."""
    primary = infer_scripts_dir()
    for path in _infer_script_candidates():
        if path == primary:
            continue
        resolved = str(path.resolve())
        if path.is_dir() and resolved not in sys.path:
            sys.path.insert(0, resolved)
    if primary.is_dir():
        resolved = str(primary.resolve())
        if resolved in sys.path:
            sys.path.remove(resolved)
        sys.path.insert(0, resolved)
    return primary


def resolve_path(value: str | Path, *, root: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    base = root or ai_root()
    return (base / path).resolve()
