"""Download evidence files for GPU inference (S3 or presigned URL)."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import httpx

from gpu_worker.config import WorkerConfig
from gpu_worker.schemas import AnalysisJobMessage

_S3_URI = re.compile(r"^s3://([^/]+)/(.+)$")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    match = _S3_URI.match(uri.strip())
    if not match:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return match.group(1), match.group(2)


def download_job_file(job: AnalysisJobMessage, cfg: WorkerConfig) -> Path:
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(job.filePath).suffix or ".mp4"
    local_path = cfg.work_dir / f"{job.evidenceId}_{job.analysisRequestId}{suffix}"

    if job.presignedDownloadUrl:
        with httpx.Client(timeout=600.0) as client:
            response = client.get(job.presignedDownloadUrl)
            response.raise_for_status()
            local_path.write_bytes(response.content)
        return local_path

    bucket = (job.s3Bucket or cfg.evidence_bucket or "").strip()
    key = (job.s3ObjectKey or job.filePath or "").strip().lstrip("/")
    if not bucket and job.filePath.startswith("s3://"):
        bucket, key = parse_s3_uri(job.filePath)
    if not bucket or not key:
        raise ValueError(
            f"Missing S3 location for evidenceId={job.evidenceId} "
            f"(bucket={bucket!r}, key={key!r})"
        )

    import boto3

    region = job.s3Region or cfg.aws_region
    client = boto3.client("s3", region_name=region)
    client.download_file(bucket, key, str(local_path))
    return local_path


def download_from_evidence_path(evidence_path: str, cfg: WorkerConfig, dest: Path) -> Path:
    evidence_path = evidence_path.strip()
    if evidence_path.startswith("s3://"):
        bucket, key = parse_s3_uri(evidence_path)
        import boto3

        client = boto3.client("s3", region_name=cfg.aws_region)
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(dest))
        return dest

    parsed = urlparse(evidence_path)
    if parsed.scheme in ("http", "https"):
        with httpx.Client(timeout=600.0) as client:
            response = client.get(evidence_path)
            response.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(response.content)
        return dest

    path = Path(evidence_path)
    if path.is_file():
        return path
    raise ValueError(f"Unsupported evidence_path: {evidence_path}")
