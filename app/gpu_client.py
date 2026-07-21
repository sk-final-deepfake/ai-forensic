"""On-Prem GPU Gateway HTTP client (VPN 경유 AI_GATEWAY_URL/infer)."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.config import Settings
from app.schemas.messaging import (
    AnalysisJobMessage,
    AnalysisResponseMessage,
    AnalysisVideoResultItem,
    FrameRiskItem,
    TamperBBoxItem,
)


def _coerce_score(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(parsed) or math.isinf(parsed):
        return 0.0
    return parsed


def _parse_tamper_bboxes(raw: Any) -> list[TamperBBoxItem] | None:
    if not isinstance(raw, list) or not raw:
        return None
    out: list[TamperBBoxItem] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if not all(k in item for k in ("x", "y", "w", "h")):
            continue
        try:
            out.append(
                TamperBBoxItem(
                    x=int(item["x"]),
                    y=int(item["y"]),
                    w=int(item["w"]),
                    h=int(item["h"]),
                    score=None if item.get("score") is None else float(item["score"]),
                )
            )
        except (TypeError, ValueError):
            continue
    return out or None


def _sanitize_model_score_item(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    out["score"] = _coerce_score(out.get("score"))
    return out


def _sanitize_module_timeline_item(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    out["videoScore"] = _coerce_score(out.get("videoScore"))
    if out.get("threshold") is None:
        out["threshold"] = 0.5
    else:
        out["threshold"] = _coerce_score(out.get("threshold"))
    return out


def _sanitize_per_frame_face_score_item(item: dict[str, Any]) -> dict[str, Any] | None:
    out = dict(item)
    if out.get("frameIndex") is None and out.get("frame_index") is None:
        return None
    if "frameIndex" not in out and out.get("frame_index") is not None:
        out["frameIndex"] = out["frame_index"]
    if "faceIndex" not in out and out.get("face_index") is not None:
        out["faceIndex"] = out["face_index"]
    out["faceIndex"] = int(out.get("faceIndex") or 0)
    out["frameIndex"] = int(out["frameIndex"])
    out["riskScore"] = _coerce_score(out.get("riskScore", out.get("fake_score", out.get("prob_fake"))))
    return out


def _sanitize_gateway_response(data: dict[str, Any]) -> dict[str, Any]:
    """GPU may return null module scores when TimeSformer/GMFlow soft-fail."""
    out = dict(data)
    if isinstance(out.get("modelScores"), list):
        out["modelScores"] = [_sanitize_model_score_item(row) for row in out["modelScores"] if isinstance(row, dict)]
    results = out.get("results")
    if isinstance(results, list):
        sanitized_results: list[Any] = []
        for item in results:
            if not isinstance(item, dict):
                sanitized_results.append(item)
                continue
            row = dict(item)
            if isinstance(row.get("modelScores"), list):
                row["modelScores"] = [
                    _sanitize_model_score_item(score) for score in row["modelScores"] if isinstance(score, dict)
                ]
            if isinstance(row.get("moduleTimelines"), list):
                row["moduleTimelines"] = [
                    _sanitize_module_timeline_item(timeline)
                    for timeline in row["moduleTimelines"]
                    if isinstance(timeline, dict)
                ]
            if isinstance(row.get("perFrameFaceScores"), list):
                faces: list[dict[str, Any]] = []
                for face in row["perFrameFaceScores"]:
                    if not isinstance(face, dict):
                        continue
                    sanitized_face = _sanitize_per_frame_face_score_item(face)
                    if sanitized_face is not None:
                        faces.append(sanitized_face)
                row["perFrameFaceScores"] = faces
            sanitized_results.append(row)
        out["results"] = sanitized_results
    return out


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
        return AnalysisResponseMessage.model_validate(_sanitize_gateway_response(data))

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
                bboxes=_parse_tamper_bboxes(item.get("bboxes")),
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
