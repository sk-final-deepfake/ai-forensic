"""On-GPU inference for POST /infer (Xception via gpu_worker)."""

from __future__ import annotations

import logging
from pathlib import Path

from app.messaging.analysis_progress import publish_analysis_progress_with_config
from app.schemas.gateway import GatewayInferRequest
from gpu_worker.config import WorkerConfig, load_config
from gpu_worker.inference_runner import run_inference
from gpu_worker.s3_download import download_from_evidence_path, parse_s3_uri
from gpu_worker.schemas import AnalysisJobMessage, AnalysisResponseMessage

logger = logging.getLogger("ai_fastapi.gateway_infer")


def _job_from_request(req: GatewayInferRequest) -> AnalysisJobMessage:
    file_path = req.evidence_path
    s3_bucket: str | None = None
    if req.evidence_path.startswith("s3://"):
        s3_bucket, file_path = parse_s3_uri(req.evidence_path)

    return AnalysisJobMessage(
        analysisRequestId=req.analysis_request_id,
        evidenceId=req.evidence_id,
        fileType="video",
        filePath=file_path,
        s3Bucket=s3_bucket,
        caseName=req.case_id,
    )


def _report_progress(cfg: WorkerConfig, job: AnalysisJobMessage, percent: int, message: str | None) -> None:
    publish_analysis_progress_with_config(
        cfg,
        job.analysisRequestId,
        job.evidenceId,
        percent,
        message,
    )


def run_gateway_infer(req: GatewayInferRequest) -> AnalysisResponseMessage:
    cfg = load_config()
    job = _job_from_request(req)

    suffix = Path(job.filePath).suffix or ".mp4"
    local_path = cfg.work_dir / f"{job.evidenceId}_{job.analysisRequestId}{suffix}"

    _report_progress(cfg, job, 5, "영상 다운로드 중")
    if req.local_path:
        local_path = Path(req.local_path)
    else:
        logger.info("Downloading evidence_path=%s", req.evidence_path)
        local_path = download_from_evidence_path(req.evidence_path, cfg, local_path)

    _report_progress(cfg, job, 12, "모델 추론 준비 중")
    logger.info(
        "Running inference mode=%s analysisRequestId=%s local_path=%s",
        cfg.inference_mode,
        job.analysisRequestId,
        local_path,
    )

    def on_progress(percent: int, message: str | None = None) -> None:
        _report_progress(cfg, job, percent, message)

    return run_inference(job, local_path, cfg, on_progress=on_progress)
