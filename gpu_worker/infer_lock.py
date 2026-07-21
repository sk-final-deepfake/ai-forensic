"""Exclusive GPU lock — gateway /infer and overlay worker must not overlap."""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("gpu_worker.infer_lock")


class GpuInferLockBusy(RuntimeError):
    """Raised when non-blocking GPU lock acquisition fails."""


def _lock_path(work_dir: Path | None = None) -> Path:
    if work_dir is not None:
        return Path(work_dir) / ".gpu_infer.lock"
    root = os.getenv("FORENSHIELD_AI_ROOT", str(Path.home() / "forenShield-ai")).strip()
    base = Path(root)
    if base.name == "deepfake":
        base = base.parent
    return base / "work" / ".gpu_infer.lock"


@contextmanager
def gpu_infer_lock(
    lock_path: Path | None = None,
    *,
    work_dir: Path | None = None,
    blocking: bool = True,
    label: str = "gpu_job",
) -> Iterator[None]:
    path = lock_path or _lock_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o666)
    import fcntl

    flags = fcntl.LOCK_EX
    if not blocking:
        flags |= fcntl.LOCK_NB
    try:
        fcntl.flock(fd, flags)
    except BlockingIOError as exc:
        os.close(fd)
        raise GpuInferLockBusy(f"GPU busy ({path.name})") from exc

    logger.info("GPU lock acquired label=%s path=%s", label, path)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        logger.info("GPU lock released label=%s path=%s", label, path)
