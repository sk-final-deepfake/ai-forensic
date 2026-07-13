from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app.services.module_overlays import (
    _clip_risks_to_frame_scores,
    _pair_risks_to_frame_scores,
    build_module_overlay_set,
)


class ModuleOverlayScoreMapTests(unittest.TestCase):
    def test_clip_risks_expand_to_frame_range(self) -> None:
        scores = _clip_risks_to_frame_scores(
            [
                {
                    "clipIndex": 0,
                    "startFrameIndex": 2,
                    "endFrameIndex": 4,
                    "riskScore": 0.8,
                },
                {
                    "clipIndex": 1,
                    "startFrameIndex": 4,
                    "endFrameIndex": 5,
                    "riskScore": 0.9,
                },
            ]
        )
        self.assertEqual(scores[2], 0.8)
        self.assertEqual(scores[3], 0.8)
        self.assertEqual(scores[4], 0.9)
        self.assertEqual(scores[5], 0.9)

    def test_pair_risks_mark_both_frames(self) -> None:
        scores = _pair_risks_to_frame_scores(
            [
                {
                    "pairIndex": 0,
                    "frameIndexA": 10,
                    "frameIndexB": 11,
                    "riskScore": 0.7,
                }
            ]
        )
        self.assertEqual(scores[10], 0.7)
        self.assertEqual(scores[11], 0.7)


class ModuleOverlayBuildTests(unittest.TestCase):
    def test_build_module_overlay_set_creates_all_three(self) -> None:
        os.environ["AI_VISUALIZATION_ENABLED"] = "1"
        os.environ["AI_VISUALIZATION_OVERLAY"] = "1"
        os.environ["AI_VISUALIZATION_OVERLAY_MAX_SEC"] = "2"

        def fake_upload(local_path, *, evidence_id, analysis_request_id, name):
            self.assertTrue(Path(local_path).is_file())
            return f"https://cdn.test/{name}"

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            video = td / "sample.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
            for _ in range(12):
                frame = np.full((120, 160, 3), 40, dtype=np.uint8)
                cv2.rectangle(frame, (40, 30), (100, 90), (200, 180, 160), -1)
                writer.write(frame)
            writer.release()

            with (
                patch("app.services.module_overlays._maybe_upload", side_effect=fake_upload),
                patch("app.services.visualization_artifacts._maybe_upload", side_effect=fake_upload),
            ):
                result = build_module_overlay_set(
                    video_path=video,
                    evidence_id=1,
                    analysis_request_id=2,
                    work_dir=td / "viz",
                    cnn_per_frame_scores=[
                        {
                            "frame_index": i,
                            "face_index": 0,
                            "fake_score": 0.8,
                            "bbox": {"x": 40, "y": 30, "w": 60, "h": 60},
                        }
                        for i in range(0, 12, 2)
                    ],
                    clip_risks=[
                        {
                            "clipIndex": 0,
                            "startFrameIndex": 2,
                            "endFrameIndex": 8,
                            "riskScore": 0.9,
                        }
                    ],
                    pair_risks=[
                        {
                            "pairIndex": 0,
                            "frameIndexA": 3,
                            "frameIndexB": 4,
                            "riskScore": 0.6,
                        }
                    ],
                )

        self.assertTrue(result.overlay_by_module["cnn"])
        self.assertTrue(result.overlay_by_module["temporal"])
        self.assertTrue(result.overlay_by_module["optical"])
        self.assertTrue(result.legacy_cnn_overlay_url.endswith("overlay_cnn.mp4"))
        ready = {row["key"]: row["status"] for row in result.model_overlay_artifacts}
        self.assertEqual(ready["deepfake:cnn"], "ready")
        self.assertEqual(ready["deepfake:temporal"], "ready")
        self.assertEqual(ready["deepfake:optical"], "ready")


if __name__ == "__main__":
    unittest.main()
