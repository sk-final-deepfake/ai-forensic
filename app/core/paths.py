from __future__ import annotations

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


def ensure_infer_scripts_on_path() -> Path:
    scripts = infer_scripts_dir()
    scripts_str = str(scripts)
    if scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)
    return scripts


def resolve_path(value: str | Path, *, root: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    base = root or ai_root()
    return (base / path).resolve()
