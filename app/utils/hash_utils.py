import hashlib
from pathlib import Path


def calculate_sha256(file_path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    sha256 = hashlib.sha256()
    path = Path(file_path)

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            sha256.update(chunk)

    return sha256.hexdigest()
