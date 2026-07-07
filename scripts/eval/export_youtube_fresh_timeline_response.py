#!/usr/bin/env python3
"""Build AnalysisResponseMessage JSON from youtube-fresh infer cache (no GPU infer).

Usage:
  cd ai
  ..\\.venv\\Scripts\\python.exe scripts\\eval\\export_youtube_fresh_timeline_response.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from app.schemas.analysis import AnalysisRequest
from app.services.infer_bridge import ModuleInferResult
from app.services.late_fusion import load_fusion_config
from app.services.video_deepfake_analyzer import build_response_from_modules

CACHE = AI_ROOT / "results" / "infer" / "youtube-fresh-late-fusion-tuned"
VIDEO_ROOT = AI_ROOT / "data" / "test" / "video" / "youtube-fresh"
FUSION_CONFIG = AI_ROOT / "config" / "fusion_v1_tuned.json"
OUT = AI_ROOT / "results" / "eval" / "youtube_fresh_timeline_response_sample.json"


def load_module(module: str, stem: str) -> dict:
    path = CACHE / module / "json" / f"{stem}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def module_rows(stem: str, gt: str) -> list[ModuleInferResult]:
    x = load_module("xception", stem)
    t = load_module("timesformer", stem)
    g = load_module("gmflow", stem)
    x_breakdown = x.get("score_breakdown") or {}
    per_frame = x_breakdown.get("per_frame_scores") or [
        {"frame_index": row["frame_index"], "fake_score": row["prob_fake"]}
        for row in x_breakdown.get("per_frame") or []
        if row.get("frame_index") is not None and row.get("prob_fake") is not None
    ]
    t_breakdown = t.get("score_breakdown") or {}
    return [
        ModuleInferResult(
            module="cnn",
            model_name="xception",
            model_version="xception/v1.1.0-celeb1k",
            status=str(x.get("status", "ok")),
            fake_score=x.get("fake_score"),
            pred_label=x.get("pred_label"),
            details={"per_frame_scores": per_frame, "score_breakdown": x_breakdown},
        ),
        ModuleInferResult(
            module="temporal",
            model_name="timesformer",
            model_version="timesformer/v1.1.0-celeb1k",
            status=str(t.get("status", "ok")),
            fake_score=t.get("fake_score"),
            pred_label=t.get("pred_label"),
            details={
                "score_breakdown": t_breakdown,
                "per_clip_scores": t_breakdown.get("per_clip_scores") or [],
                "per_clip": t_breakdown.get("per_clip") or [],
            },
        ),
        ModuleInferResult(
            module="optical",
            model_name="gmflow",
            model_version="gmflow/v1.0.0-rf_pooled",
            status=str(g.get("status", "ok")),
            fake_score=g.get("gmflow_learned_score", g.get("fake_score")),
            pred_label=g.get("pred_label"),
            details={
                "pair_stats": g.get("pair_stats") or [],
                "per_frame_pair": (g.get("score_breakdown") or {}).get("per_frame_pair") or [],
                "aggregate": g.get("aggregate") or {},
            },
        ),
    ]


def main() -> int:
    config = load_fusion_config(FUSION_CONFIG)
    stem = "ai_0wJezYHWA1c"
    video_path = VIDEO_ROOT / "fake" / f"{stem}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    request = AnalysisRequest(
        analysisRequestId=9001,
        evidenceId=9001,
        fileType="video",
        localVideoPath=str(video_path),
        requestedAt="2026-07-07T00:00:00Z",
    )
    response = build_response_from_modules(
        request,
        video_path,
        module_rows(stem, "fake"),
        config=config,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(response.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    video = response.results[0]
    print(f"saved: {OUT}")
    print(
        f"frameRisks={len(video.frameRisks)} "
        f"clipRisks={len(video.clipRisks)} "
        f"pairRisks={len(video.pairRisks)} "
        f"moduleTimelines={len(video.moduleTimelines)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
