from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from gpu_worker.config import WorkerConfig
from gpu_worker.schemas import (
    AnalysisJobMessage,
    AnalysisResponseMessage,
    AnalysisVideoResultItem,
    FrameRiskItem,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _scores_from_seed(seed: str) -> dict[str, float]:
    """테스트셋: 파일/해시 기반 재현 가능한 점수."""
    digest = hashlib.sha256(seed.encode()).digest()
    nums = [b / 255.0 for b in digest[:8]]
    return {
        "lipSync": round(0.15 + nums[0] * 0.75, 4),
        "frameEdit": round(0.10 + nums[1] * 0.80, 4),
        "deepfake": round(0.20 + nums[2] * 0.75, 4),
        "splicing": round(0.05 + nums[3] * 0.55, 4),
        "reEncoding": round(0.10 + nums[4] * 0.70, 4),
    }


def _risk_from_scores(scores: dict[str, float]) -> tuple[float, str]:
    weights = {
        "deepfake": 0.45,
        "frameEdit": 0.25,
        "lipSync": 0.15,
        "reEncoding": 0.10,
        "splicing": 0.05,
    }
    risk = sum(scores[k] * weights[k] for k in weights) * 100.0
    risk = round(min(100.0, max(0.0, risk)), 1)
    if risk >= 70.0:
        level = "HIGH"
    elif risk >= 40.0:
        level = "MEDIUM"
    else:
        level = "LOW"
    return risk, level


def _risk_from_deepfake_only(deepfake_score: float) -> tuple[float, str]:
    risk = round(min(100.0, max(0.0, deepfake_score * 100.0)), 1)
    if risk >= 70.0:
        level = "HIGH"
    elif risk >= 40.0:
        level = "MEDIUM"
    else:
        level = "LOW"
    return risk, level


def _build_video_item(scores: dict[str, float]) -> AnalysisVideoResultItem:
    """test / gateway — 5모듈 전체 (test 파이프라인용)."""
    threshold = 0.5
    return AnalysisVideoResultItem(
        lipSyncDetected=scores["lipSync"] >= threshold,
        lipSyncScore=scores["lipSync"],
        frameEditDetected=scores["frameEdit"] >= threshold,
        frameEditScore=scores["frameEdit"],
        deepfakeDetected=scores["deepfake"] >= threshold,
        deepfakeScore=scores["deepfake"],
        splicingDetected=scores["splicing"] >= threshold,
        splicingScore=scores["splicing"],
        reEncodingDetected=scores["reEncoding"] >= threshold,
        reEncodingScore=scores["reEncoding"],
    )


def _build_xception_video_item(
    *,
    model_name: str,
    model_version: str,
    deepfake_score: float,
    threshold: float,
    frame_risks: list[FrameRiskItem],
) -> AnalysisVideoResultItem:
    """Xception-only — deepfake + frameRisks 만 전송 (미실행 모듈 필드 없음)."""
    return AnalysisVideoResultItem(
        modelName=model_name,
        modelVersion=model_version,
        deepfakeDetected=deepfake_score >= threshold,
        deepfakeScore=round(deepfake_score, 4),
        frameRisks=frame_risks,
    )


def _resolve_checkpoint_path(cfg: WorkerConfig) -> Path:
    from gpu_worker.models.xception_video import resolve_checkpoint

    explicit = (cfg.model_checkpoint or "").strip()
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_file():
            candidate = cfg.project_root / explicit
        if candidate.is_file():
            return candidate

    for base in (
        cfg.project_root / "deepfake" / "models" / "test",
        cfg.models_test_dir,
    ):
        try:
            return resolve_checkpoint(base, "")
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        "Xception checkpoint not found. Set MODEL_CHECKPOINT_PATH or XCEPTION_WEIGHTS in gpu_worker/.env"
    )


def run_test_inference(job: AnalysisJobMessage, local_path: Path, cfg: WorkerConfig) -> AnalysisResponseMessage:
    seed = job.originalSha256 or job.originalHash or str(local_path)
    scores = _scores_from_seed(seed)
    risk_score, risk_level = _risk_from_scores(scores)
    video = _build_video_item(scores)

    reasons: list[str] = []
    if video.deepfakeDetected:
        reasons.append(f"Deepfake score {video.deepfakeScore:.2f} (test pipeline)")
    if video.frameEditDetected:
        reasons.append(f"Frame edit score {video.frameEditScore:.2f} (test pipeline)")
    if not reasons:
        reasons.append("Test pipeline: no strong manipulation signal")

    payload = AnalysisResponseMessage(
        analysisRequestId=job.analysisRequestId,
        evidenceId=job.evidenceId,
        status="COMPLETED",
        riskScore=risk_score,
        confidenceScore=round(0.75 + (risk_score / 100.0) * 0.2, 4),
        riskLevel=risk_level,
        analysisReasons=reasons,
        results=[video],
        analyzedAt=_utc_now(),
    )

    out_path = cfg.results_dir / f"analysis_{job.analysisRequestId}_{job.evidenceId}.json"
    out_path.write_text(
        json.dumps(payload.model_dump(mode="json", exclude_none=True), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def run_gateway_inference(job: AnalysisJobMessage, local_path: Path, cfg: WorkerConfig) -> AnalysisResponseMessage:
    """GPU Gateway FastAPI POST /infer (Infra 7.ai-deploy.md)."""
    url = cfg.gpu_gateway_url.rstrip("/") + "/infer"
    body = {
        "case_id": job.caseName or str(job.evidenceId),
        "evidence_id": job.evidenceId,
        "analysis_request_id": job.analysisRequestId,
        "evidence_path": f"s3://{job.s3Bucket}/{job.filePath}" if job.s3Bucket else str(local_path),
        "local_path": str(local_path),
    }
    with httpx.Client(timeout=1800.0) as client:
        response = client.post(url, json=body)
        response.raise_for_status()
        data = response.json()

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


def run_local_model_inference(job: AnalysisJobMessage, local_path: Path, cfg: WorkerConfig) -> AnalysisResponseMessage:
    """3모듈(Xception + TimeSformer + GMFlow) + Late Fusion — BE 계약 JSON 그대로."""
    if cfg.use_mock_infer:
        return run_test_inference(job, local_path, cfg)

    from gpu_worker.pipeline.response_builder import build_analysis_response

    payload = build_analysis_response(
        analysis_request_id=job.analysisRequestId,
        evidence_id=job.evidenceId,
        video_path=local_path,
        cfg=cfg,
    )

    out_path = cfg.results_dir / f"analysis_{job.analysisRequestId}_{job.evidenceId}.json"
    out_path.write_text(
        json.dumps(payload.model_dump(mode="json", exclude_none=True), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def run_inference(job: AnalysisJobMessage, local_path: Path, cfg: WorkerConfig) -> AnalysisResponseMessage:
    mode = cfg.inference_mode.lower()
    if mode == "gateway":
        return run_gateway_inference(job, local_path, cfg)
    if mode == "local_model":
        return run_local_model_inference(job, local_path, cfg)
    return run_test_inference(job, local_path, cfg)
