"""Download evidence files for GPU inference (S3 or presigned URL)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx

from gpu_worker.config import WorkerConfig
from gpu_worker.schemas import AnalysisJobMessage

logger = logging.getLogger("gpu_worker.s3_download")

_S3_URI = re.compile(r"^s3://([^/]+)/(.+)$")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    match = _S3_URI.match(uri.strip())
    if not match:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return match.group(1), match.group(2)


def _resolve_bucket_key(job: AnalysisJobMessage, cfg: WorkerConfig) -> tuple[str, str]:
    bucket = (job.s3Bucket or cfg.evidence_bucket or "").strip()
    key = (job.s3ObjectKey or job.filePath or "").strip().lstrip("/")
    if not bucket and (job.filePath or "").startswith("s3://"):
        bucket, key = parse_s3_uri(job.filePath)
    return bucket, key


def _download_via_boto3(bucket: str, key: str, dest: Path, region: str | None) -> Path:
    import boto3

    dest.parent.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3", region_name=region)
    client.download_file(bucket, key, str(dest))
    return dest


def download_job_file(job: AnalysisJobMessage, cfg: WorkerConfig) -> Path:
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(job.filePath or "").suffix or ".mp4"
    local_path = cfg.work_dir / f"{job.evidenceId}_{job.analysisRequestId}{suffix}"

    # Reuse cache from prior analysis/overlay of the same evidence+request (avoids stale STS URL).
    if local_path.is_file() and local_path.stat().st_size > 0:
        logger.info(
            "Reusing cached video path=%s size=%s evidenceId=%s analysisRequestId=%s",
            local_path,
            local_path.stat().st_size,
            job.evidenceId,
            job.analysisRequestId,
        )
        return local_path

    bucket, key = _resolve_bucket_key(job, cfg)
    region = job.s3Region or cfg.aws_region

    if job.presignedDownloadUrl:
        try:
            with httpx.Client(timeout=600.0) as client:
                response = client.get(job.presignedDownloadUrl)
                response.raise_for_status()
                local_path.write_bytes(response.content)
            return local_path
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (400, 403) and bucket and key:
                logger.warning(
                    "Presigned download HTTP %s — falling back to boto3 s3://%s/%s "
                    "(evidenceId=%s analysisRequestId=%s)",
                    status,
                    bucket,
                    key,
                    job.evidenceId,
                    job.analysisRequestId,
                )
                return _download_via_boto3(bucket, key, local_path, region)
            raise
        except httpx.HTTPError:
            if bucket and key:
                logger.warning(
                    "Presigned download failed — falling back to boto3 s3://%s/%s "
                    "(evidenceId=%s analysisRequestId=%s)",
                    bucket,
                    key,
                    job.evidenceId,
                    job.analysisRequestId,
                    exc_info=True,
                )
                return _download_via_boto3(bucket, key, local_path, region)
            raise

    if not bucket or not key:
        raise ValueError(
            f"Missing S3 location for evidenceId={job.evidenceId} "
            f"(bucket={bucket!r}, key={key!r})"
        )

    return _download_via_boto3(bucket, key, local_path, region)


def download_from_evidence_path(evidence_path: str, cfg: WorkerConfig, dest: Path) -> Path:
    evidence_path = evidence_path.strip()
    if evidence_path.startswith("s3://"):
        bucket, key = parse_s3_uri(evidence_path)
        return _download_via_boto3(bucket, key, dest, cfg.aws_region)

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
