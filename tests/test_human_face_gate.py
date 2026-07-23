from __future__ import annotations

import unittest
from pathlib import Path

import cv2

from app.schemas.analysis import AnalysisRequest
from app.services.infer_bridge import ModuleInferResult
from app.services.video_deepfake_analyzer import build_response_from_modules

AI_ROOT = Path(__file__).resolve().parents[1]
FACE_CHECK = AI_ROOT / "docs" / "notebooks" / "output" / "output" / "face-check"


class HumanFaceGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from app.core.paths import ensure_infer_scripts_on_path

        ensure_infer_scripts_on_path()

    def test_yunet_rejects_dog_frame(self) -> None:
        from face_crop import create_face_cropper

        dog = FACE_CHECK / "ai_MOGpd-E-elw_frame175.jpg"
        if not dog.is_file():
            self.skipTest(f"missing fixture: {dog}")
        cropper = create_face_cropper(method="yunet", human_only=True)
        self.assertIsNone(cropper.crop(cv2.imread(str(dog))))
        self.assertEqual(cropper.no_face_status(), "no_human_face")

    def test_yunet_accepts_human_frame(self) -> None:
        from face_crop import create_face_cropper

        human = FACE_CHECK / "ai_KTYafGwBb9A_frame0.jpg"
        if not human.is_file():
            self.skipTest(f"missing fixture: {human}")
        cropper = create_face_cropper(method="yunet", human_only=True)
        crop = cropper.crop(cv2.imread(str(human)))
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape[:2], (256, 256))

    def test_analyzer_returns_no_human_face(self) -> None:
        from app.services.late_fusion import load_fusion_config

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
                status="no_human_face",
                fake_score=None,
                pred_label=None,
                details={"score_breakdown": {"frames_sampled": 32, "frames_with_face": 0}},
            ),
            ModuleInferResult(
                module="temporal",
                model_name="timesformer",
                model_version="timesformer/v1.0.0",
                status="no_human_face",
                fake_score=None,
                pred_label=None,
                details={},
            ),
            ModuleInferResult(
                module="optical",
                model_name="gmflow",
                model_version="gmflow/v1.0.0",
                status="skipped_no_human_face",
                fake_score=None,
                pred_label=None,
                details={},
            ),
        ]
        response = build_response_from_modules(
            request,
            Path("mock.mp4"),
            modules,
            config=config,
        )
        self.assertEqual(response.status, "COMPLETED")
        self.assertEqual(response.errorCode, "NO_HUMAN_FACE")
        self.assertIn("사람 얼굴", response.message or "")
        self.assertIn("위변조", response.message or "")
        self.assertEqual(len(response.results), 1)
        self.assertFalse(response.results[0].deepfakeDetected)
        # Soft path must never FAIL; forgery is optional when vendor/weights missing.
        self.assertNotEqual(response.status, "FAILED")

    def test_analyzer_soft_gate_includes_forgery_when_provided(self) -> None:
        from app.services.late_fusion import load_fusion_config

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
                status="no_human_face",
                fake_score=None,
                pred_label=None,
                details={"score_breakdown": {"frames_sampled": 16}},
            ),
            ModuleInferResult(
                module="forgery_spatial",
                model_name="TruFor",
                model_version="v1.0.0",
                status="ok",
                fake_score=0.62,
                pred_label="fake",
                details={
                    "threshold": 0.515,
                    "frame_risks": [
                        {"frameIndex": 0, "timestampSec": 0.0, "riskScore": 0.62},
                    ],
                    "suspicious_segments": [
                        {
                            "startTime": 0.0,
                            "endTime": 0.1,
                            "maxRiskScore": 0.62,
                            "reason": "TruFor spatial score exceeded threshold",
                        }
                    ],
                },
            ),
        ]
        response = build_response_from_modules(
            request,
            Path("mock.mp4"),
            modules,
            config=config,
        )
        self.assertEqual(response.status, "COMPLETED")
        self.assertEqual(response.errorCode, "NO_HUMAN_FACE")
        self.assertIn("이어서 수행", response.message or "")
        scores = {item.moduleName: item for item in (response.modelScores or [])}
        self.assertIn("forgery_spatial", scores)
        self.assertAlmostEqual(scores["forgery_spatial"].score, 0.62, places=4)
        self.assertTrue(response.results[0].moduleTimelines)
        self.assertEqual(response.results[0].moduleTimelines[0].module, "forgery_spatial")
        self.assertTrue(response.results[0].frameEditDetected)
        self.assertAlmostEqual(response.results[0].frameEditScore, 0.62, places=4)


if __name__ == "__main__":
    unittest.main()
