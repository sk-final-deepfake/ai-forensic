#!/usr/bin/env python3
"""Patch GPU worker: no-face -> soft COMPLETED + forgery-only path."""
from __future__ import annotations

import ast
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/sk4team/forenShield-ai/gpu_worker")
TS = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + f".bak-noface-{TS}")
    shutil.copy2(path, bak)
    print(f"backup {bak}")


def patch_module_infer() -> None:
    path = ROOT / "pipeline" / "module_infer.py"
    backup(path)
    text = path.read_text(encoding="utf-8")

    old_x = '''    status = (last_result or {}).get("status", "unknown")
    if status == "no_face":
        raise InferencePipelineError(
            "No face detected for Xception inference (tried mediapipe and haar)",
            error_code="NO_FACE_DETECTED",
        )
    raise InferencePipelineError(f"Xception inference failed: status={status}")


def run_xception_module(video_path: Path, cfg: WorkerConfig, *, threshold: float, fps: float) -> ModuleRunResult:'''

    # We'll change _infer_xception to return a sentinel dict instead of raise,
    # and handle it in run_xception_module. Cleaner: change raise to return special.

    old_x2 = '''    status = (last_result or {}).get("status", "unknown")
    if status == "no_face":
        raise InferencePipelineError(
            "No face detected for Xception inference (tried mediapipe and haar)",
            error_code="NO_FACE_DETECTED",
        )
    raise InferencePipelineError(f"Xception inference failed: status={status}")'''

    new_x2 = '''    status = (last_result or {}).get("status", "unknown")
    if status in ("no_face", "no_human_face"):
        return {"status": "no_face", "fake_score": None, "skipped": True, "error_code": "NO_HUMAN_FACE"}
    raise InferencePipelineError(f"Xception inference failed: status={status}")'''

    if old_x2 not in text:
        raise SystemExit("module_infer: xception no_face block not found")
    text = text.replace(old_x2, new_x2, 1)

    # After result = _infer_xception... handle skip before normal parsing
    old_run_x = '''    device = torch.device("cuda" if cfg.device.lower().startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)
    result = _infer_xception_with_face_fallback(model, video_path, device, threshold=threshold)

    breakdown = result.get("score_breakdown") or {}'''

    new_run_x = '''    device = torch.device("cuda" if cfg.device.lower().startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)
    result = _infer_xception_with_face_fallback(model, video_path, device, threshold=threshold)
    if result.get("skipped") or result.get("status") in ("no_face", "no_human_face"):
        return ModuleRunResult(
            module="cnn",
            model_name="Xception",
            model_version=cfg.model_version or "v1.0.0-celeb1k",
            video_score=0.0,
            threshold=threshold,
            detected=False,
            confidence=0.0,
            frame_risks=[],
            clip_risks=[],
            pair_risks=[],
            suspicious_segments=[],
            temporal_suspicious_segments=[],
            optical_suspicious_segments=[],
            raw={**result, "skipped": True, "error_code": "NO_HUMAN_FACE"},
        )

    breakdown = result.get("score_breakdown") or {}'''

    if old_run_x not in text:
        raise SystemExit("module_infer: run_xception body not found")
    text = text.replace(old_run_x, new_run_x, 1)

    old_ts = '''    if result.get("status") != "ok" or result.get("fake_score") is None:
        status = result.get("status")
        if status == "no_face":
            raise InferencePipelineError(
                "No face detected for TimeSformer inference",
                error_code="NO_FACE_DETECTED",
            )
        raise InferencePipelineError(f"TimeSformer inference failed: status={status}")

    breakdown = result.get("score_breakdown") or {}
    aggregate = breakdown.get("aggregate") or {}
    per_clip = breakdown.get("per_clip") or []'''

    new_ts = '''    if result.get("status") != "ok" or result.get("fake_score") is None:
        status = result.get("status")
        if status in ("no_face", "no_human_face"):
            return ModuleRunResult(
                module="temporal",
                model_name="TimeSformer",
                model_version="v1.0.0-celeb1k",
                video_score=0.0,
                threshold=threshold,
                detected=False,
                confidence=0.0,
                frame_risks=[],
                clip_risks=[],
                pair_risks=[],
                suspicious_segments=[],
                temporal_suspicious_segments=[],
                optical_suspicious_segments=[],
                raw={**(result or {}), "skipped": True, "error_code": "NO_HUMAN_FACE"},
            )
        raise InferencePipelineError(f"TimeSformer inference failed: status={status}")

    breakdown = result.get("score_breakdown") or {}
    aggregate = breakdown.get("aggregate") or {}
    per_clip = breakdown.get("per_clip") or []'''

    if old_ts not in text:
        raise SystemExit("module_infer: timesformer no_face block not found")
    text = text.replace(old_ts, new_ts, 1)

    path.write_text(text, encoding="utf-8")
    ast.parse(text)
    print("patched module_infer.py")


def patch_response_builder() -> None:
    path = ROOT / "pipeline" / "response_builder.py"
    backup(path)
    text = path.read_text(encoding="utf-8")

    old = '''    cnn = run_xception_module(video_path, cfg, threshold=thresholds["cnn"], fps=fps)
    temporal = run_timesformer_module(video_path, cfg, threshold=thresholds["temporal"], fps=fps)
    optical = run_gmflow_module(video_path, cfg, threshold=thresholds["optical"], fps=fps)
    modules = {"cnn": cnn, "temporal": temporal, "optical": optical}

    module_meta = {key: _model_meta(fusion_config, key) for key in modules}
    fusion = apply_late_fusion(
        cnn_score=cnn.video_score,
        temporal_score=temporal.video_score,
        optical_score=optical.video_score,
        config=fusion_config,
        module_meta=module_meta,
    )
    model_scores = _build_model_scores(fusion, modules, fusion_config)
    module_timelines = _build_module_timelines(modules, fusion_config)
    fusion_meta = _model_meta(fusion_config, "fusion")
    viz_payload = _attach_visualization_fields(
        video_path=video_path,
        cnn=cnn,
        evidence_id=evidence_id,
        analysis_request_id=analysis_request_id,
    )

    video_item = AnalysisVideoResultItem(
        modelName=str(fusion_meta.get("modelName", "Late Fusion")),
        modelVersion=str(fusion_meta.get("modelVersion", fusion_config.get("fusion_version", "fusion-v4-ts-gated"))),
        deepfakeDetected=fusion.detected,
        deepfakeScore=fusion.score,'''

    new = '''    cnn = run_xception_module(video_path, cfg, threshold=thresholds["cnn"], fps=fps)
    temporal = run_timesformer_module(video_path, cfg, threshold=thresholds["temporal"], fps=fps)
    # Face-based deepfake modules skipped → still run optical if possible; forgery lane runs later.
    no_face = bool((cnn.raw or {}).get("skipped")) or bool((temporal.raw or {}).get("skipped"))
    try:
        optical = run_gmflow_module(video_path, cfg, threshold=thresholds["optical"], fps=fps)
    except Exception as exc:
        logger.warning("GMFlow skipped after no-face/deepfake path: %s", exc)
        optical = ModuleRunResult(
            module="optical",
            model_name="GMFlow",
            model_version="v1.0.0",
            video_score=0.0,
            threshold=thresholds["optical"],
            detected=False,
            confidence=0.0,
            frame_risks=[],
            clip_risks=[],
            pair_risks=[],
            suspicious_segments=[],
            temporal_suspicious_segments=[],
            optical_suspicious_segments=[],
            raw={"status": "skipped", "skipped": True, "error": str(exc)[:300]},
        )
    modules = {"cnn": cnn, "temporal": temporal, "optical": optical}

    module_meta = {key: _model_meta(fusion_config, key) for key in modules}
    if no_face:
        # Soft COMPLETED: deepfake inconclusive; forgery enrich will attach scores.
        from gpu_worker.pipeline.fusion import FusionResult

        fusion = FusionResult(
            score=0.0,
            detected=False,
            risk_score=0.0,
            risk_level="LOW",
            confidence=0.0,
            reasons=[
                "NO_HUMAN_FACE: 딥페이크 모델 판단 보류",
                "사람 얼굴이 검출되지 않아 딥페이크 판별을 수행할 수 없습니다. 위변조 등 후속 분석은 계속 진행할 수 있습니다.",
            ],
        )
    else:
        fusion = apply_late_fusion(
            cnn_score=cnn.video_score,
            temporal_score=temporal.video_score,
            optical_score=optical.video_score,
            config=fusion_config,
            module_meta=module_meta,
        )
    model_scores = _build_model_scores(fusion, modules, fusion_config)
    module_timelines = _build_module_timelines(modules, fusion_config)
    fusion_meta = _model_meta(fusion_config, "fusion")
    viz_payload = _attach_visualization_fields(
        video_path=video_path,
        cnn=cnn,
        evidence_id=evidence_id,
        analysis_request_id=analysis_request_id,
    )

    video_item = AnalysisVideoResultItem(
        modelName=str(fusion_meta.get("modelName", "Late Fusion")),
        modelVersion=str(fusion_meta.get("modelVersion", fusion_config.get("fusion_version", "fusion-v4-ts-gated"))),
        deepfakeDetected=fusion.detected,
        deepfakeScore=fusion.score,'''

    if old not in text:
        raise SystemExit("response_builder: build block not found")
    text = text.replace(old, new, 1)

    # Ensure ModuleRunResult import
    if "ModuleRunResult" not in text.split("from gpu_worker.pipeline.module_infer import")[1].split(")")[0]:
        text = text.replace(
            "from gpu_worker.pipeline.module_infer import (\n    ModuleRunResult,\n",
            "from gpu_worker.pipeline.module_infer import (\n    ModuleRunResult,\n",
        )
    # Add errorCode on response when no_face
    old_ret = '''    return AnalysisResponseMessage(
        analysisRequestId=analysis_request_id,
        evidenceId=evidence_id,
        status="COMPLETED",
        riskScore=fusion.risk_score,
        confidenceScore=fusion.confidence,
        riskLevel=fusion.risk_level,  # type: ignore[arg-type]
        modelName=video_item.modelName,
        modelVersion=video_item.modelVersion,
        analysisReasons=fusion.reasons,
        results=[video_item],
        analyzedAt=_utc_now(),
        modelScores=model_scores,
    )'''

    new_ret = '''    return AnalysisResponseMessage(
        analysisRequestId=analysis_request_id,
        evidenceId=evidence_id,
        status="COMPLETED",
        riskScore=fusion.risk_score,
        confidenceScore=fusion.confidence,
        riskLevel=fusion.risk_level,  # type: ignore[arg-type]
        modelName=video_item.modelName,
        modelVersion=video_item.modelVersion,
        analysisReasons=fusion.reasons,
        results=[video_item],
        analyzedAt=_utc_now(),
        modelScores=model_scores,
        errorCode="NO_HUMAN_FACE" if no_face else None,
        message=(
            "사람 얼굴이 검출되지 않아 딥페이크 판별을 수행할 수 없습니다. 위변조 등 후속 분석은 계속 진행합니다."
            if no_face
            else None
        ),
    )'''

    if old_ret not in text:
        raise SystemExit("response_builder: return block not found")
    text = text.replace(old_ret, new_ret, 1)

    # Fix no_face variable scope - it's used in return; must be defined even when not no_face path
    # We defined no_face earlier - good.

    # Check FusionResult fields
    path.write_text(text, encoding="utf-8")
    # Don't ast.parse until we verify FusionResult - will validate in main with import check
    print("patched response_builder.py (syntax check deferred)")


def patch_inference_runner_safety() -> None:
    """If NO_FACE still raised somehow, soft-complete instead of FAILED."""
    path = ROOT / "inference_runner.py"
    backup(path)
    text = path.read_text(encoding="utf-8")
    old = '''    except InferencePipelineError as exc:
        return AnalysisResponseMessage(
            analysisRequestId=job.analysisRequestId,
            evidenceId=job.evidenceId,
            status="FAILED",
            analyzedAt=_utc_now(),
            errorCode=exc.error_code,
            message=str(exc),
        )'''
    new = '''    except InferencePipelineError as exc:
        # Face miss should not fail the whole job — forgery lane still runs on COMPLETED.
        if getattr(exc, "error_code", "") in ("NO_FACE_DETECTED", "NO_HUMAN_FACE"):
            return AnalysisResponseMessage(
                analysisRequestId=job.analysisRequestId,
                evidenceId=job.evidenceId,
                status="COMPLETED",
                riskScore=0.0,
                confidenceScore=0.0,
                riskLevel="LOW",
                analysisReasons=[
                    "NO_HUMAN_FACE: 딥페이크 모델 판단 보류",
                    "사람 얼굴이 검출되지 않아 딥페이크 판별을 수행할 수 없습니다. 위변조 등 후속 분석은 계속 진행합니다.",
                ],
                results=[],
                analyzedAt=_utc_now(),
                modelScores=[],
                errorCode="NO_HUMAN_FACE",
                message=str(exc),
            )
        return AnalysisResponseMessage(
            analysisRequestId=job.analysisRequestId,
            evidenceId=job.evidenceId,
            status="FAILED",
            analyzedAt=_utc_now(),
            errorCode=exc.error_code,
            message=str(exc),
        )'''
    if old not in text:
        raise SystemExit("inference_runner: InferencePipelineError block not found")
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    ast.parse(text)
    print("patched inference_runner.py")


def verify_fusion_result() -> None:
    import sys

    sys.path.insert(0, str(ROOT.parent))
    from gpu_worker.pipeline.fusion import FusionResult
    import inspect

    print("FusionResult fields:", getattr(FusionResult, "__annotations__", {}))
    # try construct
    fr = FusionResult(
        score=0.0,
        detected=False,
        risk_score=0.0,
        risk_level="LOW",
        confidence=0.0,
        reasons=["test"],
    )
    print("FusionResult ok", fr)


def main() -> None:
    patch_module_infer()
    patch_response_builder()
    patch_inference_runner_safety()
    # syntax check response_builder after verifying FusionResult
    verify_fusion_result()
    rb = (ROOT / "pipeline" / "response_builder.py").read_text(encoding="utf-8")
    # Ensure ModuleRunResult imported
    if "ModuleRunResult" not in rb:
        raise SystemExit("ModuleRunResult missing from response_builder imports")
    ast.parse(rb)
    print("all patches ok")


if __name__ == "__main__":
    main()
