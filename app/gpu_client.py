"""On-Prem GPU Gateway HTTP client (VPN 경유 AI_GATEWAY_URL/infer)."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from app.core.config import Settings
from app.schemas.messaging import (
    AnalysisJobMessage,
    AnalysisResponseMessage,
    AnalysisVideoResultItem,
    FrameRiskItem,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _risk_from_scores(scores: dict[str, float]) -> tuple[float, str]:
    weights = {
        "deepfake": 0.45,
        "frameEdit": 0.25,
        "lipSync": 0.15,
        "reEncoding": 0.10,
        "splicing": 0.05,
    }
    risk = sum(scores.get(k, 0.0) * weights[k] for k in weights) * 100.0
    risk = round(min(100.0, max(0.0, risk)), 1)
    if risk >= 70.0:
        level = "HIGH"
    elif risk >= 40.0:
        level = "MEDIUM"
    else:
        level = "LOW"
    return risk, level


def _build_video_item(scores: dict[str, float]) -> AnalysisVideoResultItem:
    threshold = 0.5
    return AnalysisVideoResultItem(
        lipSyncDetected=scores.get("lipSync", 0) >= threshold,
        lipSyncScore=scores.get("lipSync"),
        frameEditDetected=scores.get("frameEdit", 0) >= threshold,
        frameEditScore=scores.get("frameEdit"),
        deepfakeDetected=scores.get("deepfake", 0) >= threshold,
        deepfakeScore=scores.get("deepfake"),
        splicingDetected=scores.get("splicing", 0) >= threshold,
        splicingScore=scores.get("splicing"),
        reEncodingDetected=scores.get("reEncoding", 0) >= threshold,
        reEncodingScore=scores.get("reEncoding"),
    )


def _evidence_path(job: AnalysisJobMessage) -> str:
    if job.s3Bucket and job.filePath:
        return f"s3://{job.s3Bucket}/{job.filePath}"
    if job.presignedDownloadUrl:
        return job.presignedDownloadUrl
    return job.filePath


def call_gpu_gateway(job: AnalysisJobMessage, settings: Settings) -> AnalysisResponseMessage:
    url = settings.ai_gateway_url.rstrip("/") + "/infer"
    body = {
        "case_id": job.caseName or str(job.evidenceId),
        "evidence_id": job.evidenceId,
        "analysis_request_id": job.analysisRequestId,
        "evidence_path": _evidence_path(job),
    }
    timeout = httpx.Timeout(settings.ai_gateway_timeout_sec)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=body)
        response.raise_for_status()
        data = response.json()

    if data.get("status") in ("COMPLETED", "FAILED") and "analysisRequestId" in data:
        return AnalysisResponseMessage.model_validate(data)

    if data.get("status") == "FAILED":
        return AnalysisResponseMessage(
            analysisRequestId=job.analysisRequestId,
            evidenceId=job.evidenceId,
            status="FAILED",
            analyzedAt=_utc_now(),
            errorCode=data.get("errorCode", "MODEL_INFERENCE_FAILED"),
            message=data.get("message", "GPU gateway failed"),
        )

    scores: dict[str, float] = {}
    for key, json_key in (
        ("lipSync", "lipSyncScore"),
        ("frameEdit", "frameEditScore"),
        ("deepfake", "deepfakeScore"),
        ("splicing", "splicingScore"),
        ("reEncoding", "reEncodingScore"),
    ):
        if json_key in data and data[json_key] is not None:
            scores[key] = float(data[json_key])

    if not scores:
        raise RuntimeError("GPU gateway returned no module scores")

    risk_score, risk_level = _risk_from_scores(scores)
    video = _build_video_item(scores)

    frame_risks_raw = data.get("frameRisks") or data.get("frame_risks")
    if frame_risks_raw:
        video.frameRisks = [
            FrameRiskItem(
                frameIndex=int(item.get("frameIndex", item.get("frame_index", i))),
                timestampSec=float(item.get("timestampSec", item.get("timestamp_sec", 0))),
                riskScore=float(item.get("riskScore", item.get("risk_score", 0))),
            )
            for i, item in enumerate(frame_risks_raw)
        ]

    if data.get("modelName"):
        video.modelName = data["modelName"]
    if data.get("modelVersion"):
        video.modelVersion = data["modelVersion"]

    return AnalysisResponseMessage(
        analysisRequestId=job.analysisRequestId,
        evidenceId=job.evidenceId,
        status="COMPLETED",
        riskScore=risk_score,
        confidenceScore=float(data.get("confidenceScore", 0.9)),
        riskLevel=risk_level,
        modelName=data.get("modelName"),
        modelVersion=data.get("modelVersion"),
        analysisReasons=list(data.get("analysisReasons", [])),
        results=[video],
        analyzedAt=data.get("analyzedAt", _utc_now()),
    )
