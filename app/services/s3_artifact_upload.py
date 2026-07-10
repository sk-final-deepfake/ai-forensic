from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("ai_fastapi.s3_artifact_upload")


def s3_upload_enabled() -> bool:
    if os.getenv("AI_VISUALIZATION_UPLOAD", "1").lower() in {"0", "false", "no"}:
        return False
    return bool(os.getenv("S3_EVIDENCE_BUCKET") or os.getenv("S3_ARTIFACT_BUCKET"))


def artifact_bucket() -> str:
    return os.getenv("S3_ARTIFACT_BUCKET") or os.getenv("S3_EVIDENCE_BUCKET") or ""


def artifact_prefix(evidence_id: int, analysis_request_id: int) -> str:
    template = os.getenv(
        "AI_VISUALIZATION_PREFIX",
        "deepfake/artifacts/analysis/{evidence_id}/{analysis_request_id}",
    )
    return template.format(
        evidence_id=evidence_id,
        analysis_request_id=analysis_request_id,
    ).strip("/")


def upload_file(local_path: Path, *, bucket: str, key: str) -> str | None:
    """Upload a local file to S3 and return a presigned GET URL (or None on failure)."""
    try:
        import boto3
    except ImportError:
        logger.exception("boto3 is not installed; skipping S3 artifact upload")
        return None

    if not local_path.is_file() or not bucket or not key:
        logger.warning(
            "Skipping S3 artifact upload because inputs are invalid: path=%s bucket=%s key=%s",
            local_path,
            bucket,
            key,
        )
        return None

    region = os.getenv("AWS_REGION", "ap-northeast-2")
    client = boto3.client("s3", region_name=region)
    content_type = _content_type(local_path)
    extra = {"ContentType": content_type} if content_type else {}

    try:
        client.upload_file(str(local_path), bucket, key, ExtraArgs=extra)
    except Exception:
        logger.exception(
            "Failed to upload visualization artifact to S3: path=%s bucket=%s key=%s",
            local_path,
            bucket,
            key,
        )
        return None

    expires = int(os.getenv("AI_VISUALIZATION_PRESIGN_SEC", "604800"))
    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )
    except Exception:
        logger.exception(
            "Failed to create presigned URL for visualization artifact: bucket=%s key=%s",
            bucket,
            key,
        )
        return f"s3://{bucket}/{key}"


def _content_type(path: Path) -> str | None:
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".mp4": "video/mp4",
    }.get(suffix)
