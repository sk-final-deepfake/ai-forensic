from __future__ import annotations

import unittest

import numpy as np

from app.core.paths import ensure_infer_scripts_on_path


class MultiFaceInferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_infer_scripts_on_path()

    def test_build_score_breakdown_keeps_every_face_crop(self) -> None:
        from video_xception_infer import build_score_breakdown

        face_samples = [
            {"frame_index": 10, "face_index": 0, "bbox": (10, 20, 30, 40)},
            {"frame_index": 10, "face_index": 1, "bbox": (100, 20, 25, 25)},
            {"frame_index": 20, "face_index": 0, "bbox": (12, 22, 28, 38)},
        ]
        logits = np.array(
            [
                [1.0, -1.0],
                [-1.0, 2.5],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )

        breakdown = build_score_breakdown(
            face_samples,
            logits,
            threshold=0.5,
            frames_sampled=32,
            frames_without_face=30,
            aggregate="max",
        )

        self.assertEqual(len(breakdown["per_frame_scores"]), 3)
        self.assertEqual(breakdown["unique_frames_with_face"], 2)
        self.assertTrue(breakdown["multi_face"])
        self.assertGreaterEqual(float(breakdown["aggregate_fake_score"]), 0.9)
        frame_10_scores = [
            row["fake_score"]
            for row in breakdown["per_frame_scores"]
            if row["frame_index"] == 10
        ]
        self.assertEqual(len(frame_10_scores), 2)

    def test_score_map_by_frame_groups_faces(self) -> None:
        from app.services import visualization_artifacts as viz

        grouped = viz._score_map_by_frame(
            [
                {"frame_index": 1, "face_index": 0, "fake_score": 0.2},
                {"frame_index": 1, "face_index": 1, "fake_score": 0.9},
            ]
        )
        self.assertEqual(len(grouped[1]), 2)
        self.assertEqual(grouped[1][1]["score"], 0.9)


if __name__ == "__main__":
    unittest.main()
