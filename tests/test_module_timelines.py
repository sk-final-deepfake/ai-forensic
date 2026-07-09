from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.schemas.ai_response import AnalysisVideoResultItem
from app.schemas.analysis import AnalysisRequest
from app.services.infer_bridge import ModuleInferResult
from app.services.late_fusion import (
    build_clip_risks,
    build_module_timelines,
    build_pair_risks,
    load_fusion_config,
)
from app.services.video_deepfake_analyzer import build_response_from_modules


AI_ROOT = Path(__file__).resolve().parents[1]
FUSION_CONFIG = AI_ROOT / "config" / "fusion_v1_tuned.json"
SAMPLE_VIDEO = AI_ROOT / "data" / "test" / "video" / "youtube-fresh" / "fake" / "ai_0wJezYHWA1c.mp4"
XCEPTION_JSON = (
    AI_ROOT
    / "results"
    / "infer"
    / "youtube-fresh-late-fusion-tuned"
    / "xception"
    / "json"
    / "ai_0wJezYHWA1c.json"
)


class ModuleTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not FUSION_CONFIG.is_file():
            raise unittest.SkipTest(f"missing fusion config: {FUSION_CONFIG}")
        cls.config = load_fusion_config(FUSION_CONFIG)

    def test_build_clip_and_pair_risks_from_cached_json(self) -> None:
        if not SAMPLE_VIDEO.is_file() or not XCEPTION_JSON.is_file():
            self.skipTest("youtube-fresh sample assets missing")

        x_payload = json.loads(XCEPTION_JSON.read_text(encoding="utf-8"))
        breakdown = x_payload["score_breakdown"]
        per_frame = breakdown.get("per_frame_scores") or [
            {"frame_index": row["frame_index"], "fake_score": row["prob_fake"]}
            for row in breakdown.get("per_frame") or []
        ]

        ts_path = XCEPTION_JSON.parent.parent.parent / "timesformer" / "json" / "ai_0wJezYHWA1c.json"
        gm_path = XCEPTION_JSON.parent.parent.parent / "gmflow" / "json" / "ai_0wJezYHWA1c.json"
        if not ts_path.is_file() or not gm_path.is_file():
            self.skipTest("cached timesformer/gmflow json missing")

        ts_payload = json.loads(ts_path.read_text(encoding="utf-8"))
        gm_payload = json.loads(gm_path.read_text(encoding="utf-8"))

        modules = [
            ModuleInferResult(
                module="cnn",
                model_name="xception",
                model_version="xception/v1.1.0-celeb1k",
                status="ok",
                fake_score=x_payload["fake_score"],
                pred_label=x_payload["pred_label"],
                details={"per_frame_scores": per_frame},
            ),
            ModuleInferResult(
                module="temporal",
                model_name="timesformer",
                model_version="timesformer/v1.1.0-celeb1k",
                status="ok",
                fake_score=ts_payload["fake_score"],
                pred_label=ts_payload["pred_label"],
                details={
                    "score_breakdown": ts_payload["score_breakdown"],
                    "per_clip_scores": ts_payload["score_breakdown"].get("per_clip_scores") or [],
                    "per_clip": ts_payload["score_breakdown"].get("per_clip") or [],
                },
            ),
            ModuleInferResult(
                module="optical",
                model_name="gmflow",
                model_version="gmflow/v1.0.0-rf_pooled",
                status="ok",
                fake_score=gm_payload.get("gmflow_learned_score"),
                pred_label=gm_payload.get("pred_label"),
                details={
                    "pair_stats": gm_payload.get("pair_stats") or [],
                    "per_frame_pair": [],
                },
            ),
        ]

        clip_risks = build_clip_risks(
            SAMPLE_VIDEO,
            per_clip=ts_payload["score_breakdown"].get("per_clip") or [],
        )
        pair_risks = build_pair_risks(SAMPLE_VIDEO, gm_payload.get("pair_stats") or [])
        timelines = build_module_timelines(SAMPLE_VIDEO, modules, config=self.config)

        self.assertGreater(len(per_frame), 0)
        self.assertGreater(len(clip_risks), 0)
        self.assertGreater(len(pair_risks), 0)
        self.assertEqual(len(timelines), 3)
        self.assertEqual({row["module"] for row in timelines}, {"cnn", "temporal", "optical"})

    def test_build_response_includes_module_timelines(self) -> None:
        request = AnalysisRequest(
            analysisRequestId=1,
            evidenceId=1,
            fileType="video",
            requestedAt="2026-07-07T00:00:00Z",
        )
        modules = [
            ModuleInferResult(
                module="cnn",
                model_name="xception",
                model_version="xception/v1.0.0",
                status="ok",
                fake_score=0.81,
                pred_label="fake",
                details={"per_frame_scores": [{"frame_index": 10, "fake_score": 0.81}]},
            ),
            ModuleInferResult(
                module="temporal",
                model_name="timesformer",
                model_version="timesformer/v1.0.0",
                status="ok",
                fake_score=0.68,
                pred_label="fake",
                details={
                    "per_clip_scores": [
                        {
                            "clip_index": 0,
                            "fake_score": 0.75,
                            "clip_start_frame": 0,
                            "clip_end_frame": 80,
                        }
                    ]
                },
            ),
            ModuleInferResult(
                module="optical",
                model_name="gmflow",
                model_version="gmflow/v1.0.0",
                status="ok",
                fake_score=0.42,
                pred_label="fake",
                details={
                    "pair_stats": [
                        {"frame_index_a": 0, "frame_index_b": 1, "magnitude_mean": 0.5},
                    ]
                },
            ),
        ]
        response = build_response_from_modules(
            request,
            Path("mock.mp4"),
            modules,
            config=self.config,
        )
        video = response.results[0]
        self.assertIsInstance(video, AnalysisVideoResultItem)
        self.assertEqual(len(video.frameRisks), 1)
        self.assertEqual(len(video.clipRisks), 1)
        self.assertEqual(len(video.pairRisks), 1)
        self.assertEqual(len(video.moduleTimelines), 3)
        payload = video.model_dump()
        self.assertIn("clipRisks", payload)
        self.assertIn("pairRisks", payload)
        self.assertIn("moduleTimelines", payload)


if __name__ == "__main__":
    unittest.main()
