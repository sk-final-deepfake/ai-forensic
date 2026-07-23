from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from app.schemas.analysis import AnalysisRequest
from app.services.infer_bridge import ModuleInferResult
from app.services.video_deepfake_analyzer import build_response_from_modules

AI_ROOT = Path(__file__).resolve().parents[1]


class FaceQualityAndTemporalGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from app.core.paths import ensure_infer_scripts_on_path

        ensure_infer_scripts_on_path()

    def test_min_face_side_rejects_small_bbox(self) -> None:
        from face_crop import FaceCropConfig, FaceCropper

        cropper = FaceCropper(FaceCropConfig(method="yunet", human_only=True, min_face_side_px=48))
        # Bypass YuNet: feed filter directly
        kept = cropper._filter_min_face_size([(10, 10, 30, 42), (100, 100, 80, 90)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][2:], (80, 90))
        self.assertEqual(cropper.last_detect_stats["rejected_small"], 1)
        self.assertEqual(
            cropper.classify_empty_face_status(
                unique_usable_frames=0,
                min_faces=4,
                raw_detections=10,
                rejected_small=10,
            ),
            "face_too_small",
        )

    def test_analyzer_face_too_small_soft_complete(self) -> None:
        from app.services.late_fusion import load_fusion_config

        config_path = AI_ROOT / "config" / "fusion_v4_ts_gated.json"
        if not config_path.is_file():
            config_path = AI_ROOT / "config" / "test" / "fusion_v1_tuned.json"
        if not config_path.is_file():
            self.skipTest("fusion config missing")
        config = load_fusion_config(config_path)
        request = AnalysisRequest(
            analysisRequestId=1,
            evidenceId=99,
            fileType="video",
            localVideoPath="mock.mp4",
            requestedAt="2026-07-07T00:00:00Z",
        )
        modules = [
            ModuleInferResult(
                module="cnn",
                model_name="xception",
                model_version="xception/v1.0.0",
                status="face_too_small",
                fake_score=None,
                pred_label=None,
                details={"score_breakdown": {"frames_sampled": 32, "rejected_small_faces": 32}},
            ),
            ModuleInferResult(
                module="temporal",
                model_name="timesformer",
                model_version="timesformer/v1.0.0",
                status="face_too_small",
                fake_score=None,
                pred_label=None,
                details={},
            ),
            ModuleInferResult(
                module="optical",
                model_name="gmflow",
                model_version="gmflow/v1.0.0",
                status="ok",
                fake_score=0.2,
                pred_label="real",
                details={},
            ),
        ]
        response = build_response_from_modules(request, Path("mock.mp4"), modules, config=config)
        self.assertEqual(response.status, "COMPLETED")
        self.assertEqual(response.errorCode, "FACE_TOO_SMALL")
        self.assertIn("작", response.message or "")
        self.assertIn("위변조", response.message or "")
        self.assertNotEqual(response.status, "FAILED")

    def test_temporal_unavailable_does_not_become_no_human(self) -> None:
        from app.services.late_fusion import load_fusion_config

        config_path = AI_ROOT / "config" / "fusion_v4_ts_gated.json"
        if not config_path.is_file():
            config_path = AI_ROOT / "config" / "test" / "fusion_v1_tuned.json"
        if not config_path.is_file():
            self.skipTest("fusion config missing")
        config = load_fusion_config(config_path)
        request = AnalysisRequest(
            analysisRequestId=2,
            evidenceId=100,
            fileType="video",
            localVideoPath="mock.mp4",
            requestedAt="2026-07-07T00:00:00Z",
        )
        modules = [
            ModuleInferResult(
                module="cnn",
                model_name="xception",
                model_version="xception/v1.0.0",
                status="ok",
                fake_score=0.82,
                pred_label="fake",
                details={
                    "per_frame_scores": [{"frame_index": 0, "fake_score": 0.82, "bbox": [10, 10, 80, 90]}]
                },
            ),
            ModuleInferResult(
                module="temporal",
                model_name="timesformer",
                model_version="timesformer/v1.0.0",
                status="insufficient_temporal_clips",
                fake_score=None,
                pred_label=None,
                details={},
            ),
            ModuleInferResult(
                module="optical",
                model_name="gmflow",
                model_version="gmflow/v1.0.0",
                status="ok",
                fake_score=0.4,
                pred_label="real",
                details={},
            ),
        ]
        # collapse_frame_risks needs a readable video; use tiny synthetic via numpy write if needed.
        # build_response opens video for timestamps — provide a real short file if available.
        video = AI_ROOT / "data" / "test" / "video" / "youtube-pilot" / "fake" / "KakaoTalk_20260713_100918820.mp4"
        if not video.is_file():
            self.skipTest(f"missing fixture video: {video}")
        response = build_response_from_modules(request, video, modules, config=config)
        self.assertEqual(response.status, "COMPLETED")
        self.assertEqual(response.errorCode, "TEMPORAL_MODULE_UNAVAILABLE")
        self.assertNotEqual(response.errorCode, "NO_HUMAN_FACE")
        self.assertTrue(response.results)
        self.assertIsNotNone(response.results[0].deepfakeScore)


if __name__ == "__main__":
    unittest.main()
